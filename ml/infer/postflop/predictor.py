from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, cast

import torch
import torch.nn.functional as F

from ml.core.contracts import SignalBundle, ActionProb, normalize_action_probs, Meta, PolicyRequest
from ml.infer.postflop.encoder import (
    PostflopInferArtifacts,
    load_postflop_infer_artifacts,
    encode_postflop_state_dense,
)
from ml.infer.postflop.router import route_postflop_state
from ml.models.vocab_actions import ROOT_ACTION_VOCAB, FACING_ACTION_VOCAB


Target = Literal["root", "facing"]


@dataclass
class LoadedNet:
    net: torch.nn.Module
    artifacts: PostflopInferArtifacts
    action_vocab: list[str]


def _load_torch_module(
    ckpt_path: str | Path,
    *,
    device: str,
) -> torch.nn.Module:
    """
    Minimal loader:
      - expects a pure torch state_dict or a dict with 'state_dict'
      - requires the caller to have saved a torch.nn.Module-compatible state_dict
    """
    blob = torch.load(str(ckpt_path), map_location=device)

    # If someone saved a scripted module
    if isinstance(blob, torch.nn.Module):
        return blob.eval()

    # If it’s a Lightning checkpoint-like dict
    state = blob.get("state_dict") if isinstance(blob, dict) else None
    if state is None and isinstance(blob, dict):
        # maybe they saved raw model state
        state = {k: v for k, v in blob.items() if hasattr(v, "shape")}

    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")

    # We need a model class to instantiate. For safety, we require scripted modules OR
    # that your training saved 'model_ctor' metadata. If not present, we fail loudly.
    meta = blob.get("meta") if isinstance(blob, dict) else None
    if not (isinstance(meta, dict) and "model_ctor" in meta and "model_kwargs" in meta):
        raise RuntimeError(
            "Checkpoint does not include ctor metadata. "
            "For inference, either save a scripted module, or include meta.model_ctor/model_kwargs."
        )

    # Dynamic import of ctor
    mod_path = str(meta["model_ctor"])
    kw = dict(meta.get("model_kwargs") or {})
    module_name, attr = mod_path.rsplit(":", 1)
    m = __import__(module_name, fromlist=[attr])
    ctor = getattr(m, attr)
    net: torch.nn.Module = ctor(**kw)
    net.load_state_dict(state, strict=True)
    return net.eval().to(device)

StreetId = int  # or Literal[0,1,2,3] if you want strictness

def _state_street_id(state: Dict[str, Any]) -> int:
    st = state.get("street")

    # ints / numeric strings
    if st in (0, "0"): return 0
    if st in (1, "1"): return 1
    if st in (2, "2"): return 2
    if st in (3, "3"): return 3

    # string labels
    if isinstance(st, str):
        s = st.strip().lower()
        if s == "preflop": return 0
        if s == "flop": return 1
        if s == "turn": return 2
        if s == "river": return 3

    # safest fallback for postflop predictor
    return 1


class PostflopPredictor:
    """
    Loads root + facing nets and routes each state to the correct one.

    Note: this assumes your checkpoints are saved in an inference-friendly way
    (scripted module OR includes ctor metadata). That’s intentional: inference should be strict.
    """

    def __init__(
        self,
        *,
        root_ckpt: str | Path,
        facing_ckpt: str | Path,
        root_sidecar: str | Path,
        facing_sidecar: str | Path,
        cat_vocabs_json: str | Path,
        device: str = "cpu",
    ) -> None:
        self.device = device

        root_art = load_postflop_infer_artifacts(sidecar_path=root_sidecar, cat_vocabs_path=cat_vocabs_json)
        face_art = load_postflop_infer_artifacts(sidecar_path=facing_sidecar, cat_vocabs_path=cat_vocabs_json)

        # enforce vocab consistency with your canonical definitions
        if list(root_art.action_vocab) != list(ROOT_ACTION_VOCAB):
            raise ValueError("root sidecar action_vocab does not match ROOT_ACTION_VOCAB")
        if list(face_art.action_vocab) != list(FACING_ACTION_VOCAB):
            raise ValueError("facing sidecar action_vocab does not match FACING_ACTION_VOCAB")

        self.root = LoadedNet(
            net=_load_torch_module(root_ckpt, device=device),
            artifacts=root_art,
            action_vocab=list(ROOT_ACTION_VOCAB),
        )
        self.facing = LoadedNet(
            net=_load_torch_module(facing_ckpt, device=device),
            artifacts=face_art,
            action_vocab=list(FACING_ACTION_VOCAB),
        )

    def predict_bundle(self, pr: PolicyRequest) -> SignalBundle:
        """
        Adapter: PolicyRequest -> state dict -> existing predict(state)
        Keeps PolicyRuntime clean (it passes PolicyRequest everywhere).
        """
        state: Dict[str, Any] = {
            "street": pr.street,
            "stakes_id": pr.stakes_id,
            "ip_pos": pr.ip_pos,
            "oop_pos": pr.oop_pos,
            "pot_bb": pr.pot_bb,
            "effective_stack_bb": pr.effective_stack_bb,
            "board": pr.board,
            "board_cluster_id": pr.board_cluster_id,
            "ctx": pr.ctx,
            "topology": pr.topology,
            "bet_sizing_id": pr.bet_sizing_id,
            # router needs these depending on target
            "size_pct": pr.size_pct,
            "faced_size_pct": pr.faced_size_pct,
            # optional debug passthrough
            **(pr.debug or {}),
        }
        return self.predict(state)

    @torch.no_grad()
    def predict(self, state: Dict[str, Any]) -> SignalBundle:
        routed = route_postflop_state(state)

        if routed.target == "root":
            net = self.root

            if routed.size_pct is None:
                raise ValueError("root inference requires state['size_pct'] (percent-of-pot)")

            x = encode_postflop_state_dense(state=routed.state, artifacts=net.artifacts).to(self.device)
            logits = net.net(x.unsqueeze(0)).squeeze(0)
            probs = F.softmax(logits, dim=-1).cpu().tolist()

            aps = [ActionProb(action=a, p=float(p)) for a, p in zip(net.action_vocab, probs)]
            aps = normalize_action_probs(aps)

            return SignalBundle(
                kind="policy_postflop",
                street=_state_street_id(routed.state),
                action_probs=aps,
                confidence=1.0,
                meta=Meta(
                    model_name="PostflopPolicyRoot",
                    model_version=(getattr(net.artifacts, "model_version", None) or None),
                    ckpt_path=str(getattr(net.artifacts, "ckpt_path", None) or Path("")),
                    sidecar_path=str(getattr(net.artifacts, "sidecar_path", None) or Path("")),
                    stakes_id=int(routed.state.get("stakes_id")) if routed.state.get("stakes_id") is not None else None,
                    solver_version=str(routed.state.get("solver_version")) if routed.state.get(
                        "solver_version") else None,
                    extras={
                        "policy_kind": "root",
                        "bet_sizing_id": routed.state.get("bet_sizing_id"),
                        "ctx": routed.state.get("ctx"),
                        "topology": routed.state.get("topology"),
                    },
                ),
            )

        # facing
        net = self.facing
        if routed.faced_size_pct is None:
            raise ValueError("facing inference requires state['faced_size_pct'] (percent-of-pot)")

        x = encode_postflop_state_dense(state=routed.state, artifacts=net.artifacts).to(self.device)
        logits = net.net(x.unsqueeze(0)).squeeze(0)
        probs = F.softmax(logits, dim=-1).cpu().tolist()

        aps = [ActionProb(action=a, p=float(p)) for a, p in zip(net.action_vocab, probs)]
        aps = normalize_action_probs(aps)

        return SignalBundle(
            kind="policy_postflop",
            street=_state_street_id(routed.state),
            action_probs=aps,
            confidence=1.0,
            meta=Meta(
                model_name="PostflopPolicyFacing",
                model_version=(getattr(net.artifacts, "model_version", None) or None),
                ckpt_path=str(getattr(net.artifacts, "ckpt_path", None) or Path("")),
                sidecar_path=str(getattr(net.artifacts, "sidecar_path", None) or Path("")),
                stakes_id=int(routed.state.get("stakes_id")) if routed.state.get("stakes_id") is not None else None,
                solver_version=str(routed.state.get("solver_version")) if routed.state.get("solver_version") else None,
                extras={
                    "policy_kind": "facing",
                    "bet_sizing_id": routed.state.get("bet_sizing_id"),
                    "ctx": routed.state.get("ctx"),
                    "topology": routed.state.get("topology"),
                },
            ),
        )