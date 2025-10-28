from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Sequence, Mapping, Any, Dict
import torch
from ml.inference.policy.types import PolicyResponse, PolicyRequest
from ml.inference.postflop_infer_single import PostflopPolicyInferSingle


class PostflopPolicyRouter:
    """
    Holds two single-side inferencers:
      - root:   CHECK / BET_* / (DONK_* if OOP)
      - facing: FOLD / CALL / RAISE_* / ALLIN

    Selection rule (default):
      use `facing` if req.facing_bet is truthy, else `root`.

    You can override via `side` arg in predict().
    """

    def __init__(
        self,
        *,
        root: PostflopPolicyInferSingle,
        facing: PostflopPolicyInferSingle,
        device: Optional[torch.device] = None,
    ):
        self.root = root
        self.facing = facing
        self.device = device or self.root.device  # keep consistent

        # quick sanity: vocab disjointness hints
        assert "CHECK" in self.root.action_vocab, "root model should include CHECK"
        assert "CALL" in self.facing.action_vocab and "FOLD" in self.facing.action_vocab, \
            "facing model should include CALL/FOLD"
        self.board_cluster_feat = (
                self.root.board_cluster_feat or self.facing.board_cluster_feat
        )

    # -------- loaders --------

    @classmethod
    def from_dirs(
        cls,
        *,
        root_dir: Union[str, Path],
        facing_dir: Union[str, Path],
        device: str = "auto",
    ) -> "PostflopPolicyRouter":

        r = PostflopPolicyInferSingle.from_dir(root_dir, device=device)
        f = PostflopPolicyInferSingle.from_dir(facing_dir, device=device)
        dev = r.device  # unify
        f.model.to(dev)
        return cls(root=r, facing=f, device=dev)

    @classmethod
    def from_checkpoints(
        cls,
        *,
        root_ckpt: Union[str, Path], root_sidecar: Union[str, Path],
        facing_ckpt: Union[str, Path], facing_sidecar: Union[str, Path],
        device: str = "auto",
    ) -> "PostflopPolicyRouter":
        r = PostflopPolicyInferSingle.from_checkpoint(root_ckpt, root_sidecar, device=device)
        f = PostflopPolicyInferSingle.from_checkpoint(facing_ckpt, facing_sidecar, device=device)
        dev = r.device
        f.model.to(dev)
        return cls(root=r, facing=f, device=dev)

    # -------- predict (single) --------

    @torch.no_grad()
    def predict(
            self,
            req: PolicyRequest,
            *,
            actor: str = "ip",
            temperature: float = 1.0,
            side: Optional[str] = None,
    ) -> "PolicyResponse":
        """
        If `side` is provided, route explicitly.
        Else, prefer req.facing_bet; if missing, infer from actions_hist using the FACING model's helper.
        """
        side_norm = (side or "").strip().lower()
        if side_norm not in ("root", "facing", ""):
            raise ValueError("side must be 'root', 'facing', or None")

        # 1) Explicit override
        if side_norm == "root":
            return self.root.predict(req, actor=actor, temperature=temperature)
        if side_norm == "facing":
            return self.facing.predict(req, actor=actor, temperature=temperature)

        # 2) Auto — prefer explicit flag on the request
        if bool(getattr(req, "facing_bet", False)):
            return self.facing.predict(req, actor=actor, temperature=temperature)

        # 3) Auto — infer from action history
        try:
            hero_is_ip = PolicyRequest.is_hero_ip(req.hero_pos or "", req.villain_pos or "")
        except Exception:
            hero_is_ip = True
        facing_flag, _ = self.facing.infer_facing_and_size(req, hero_is_ip=hero_is_ip)

        if facing_flag:
            return self.facing.predict(req, actor=actor, temperature=temperature)
        else:
            return self.root.predict(req, actor=actor, temperature=temperature)

    @torch.no_grad()
    def predict_batch(
            self,
            reqs: Sequence["PolicyRequest"],
            *,
            actor: str = "ip",
            temperature: float = 1.0,
    ) -> Sequence["PolicyResponse"]:
        """
        Splits into root/facing groups per-request (explicit req.facing_bet wins; else infer),
        runs each side once, then restores input order.
        """
        if not reqs:
            return []

        # Decide side per request
        decisions: list[str] = []
        for r in reqs:
            if bool(getattr(r, "facing_bet", False)):
                decisions.append("facing")
                continue
            try:
                hero_is_ip = PolicyRequest.is_hero_ip(r.hero_pos or "", r.villain_pos or "")
            except Exception:
                hero_is_ip = True
            facing_flag, _ = self.facing.infer_facing_and_size(r, hero_is_ip=hero_is_ip)
            decisions.append("facing" if facing_flag else "root")

        # Partition indices
        idx_root = [i for i, d in enumerate(decisions) if d == "root"]
        idx_facing = [i for i, d in enumerate(decisions) if d == "facing"]

        out: Dict[int, PolicyResponse] = {}

        # Root group
        for i in idx_root:
            out[i] = self.root.predict(reqs[i], actor=actor, temperature=temperature)

        # Facing group
        for i in idx_facing:
            out[i] = self.facing.predict(reqs[i], actor=actor, temperature=temperature)

        return [out[i] for i in range(len(reqs))]