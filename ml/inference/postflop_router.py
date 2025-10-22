# ml/policy/postflop_router.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Sequence, Mapping, Any, Dict

import torch

from ml.inference.policy.types import PolicyResponse
from ml.inference.postflop_infer_single import PostflopPolicyInferSingle


# expect these available in your project:
# - PolicyRequest, PolicyResponse
# - PostflopPolicyInferSingle (the class we just wrote)
# - _to_device

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
        req: "PolicyRequest",
        *,
        actor: str = "ip",        # for ROOT, decides DONK legality (OOP only)
        temperature: float = 1.0,
        side: Optional[str] = None,  # "root" | "facing" | None (auto from req.facing_bet)
    ) -> "PolicyResponse":
        side_norm = (side or "").strip().lower()
        if side_norm not in ("root", "facing", ""):
            raise ValueError("side must be 'root', 'facing', or None")

        use_facing = req.facing_bet if side_norm == "" else (side_norm == "facing")
        if use_facing:
            return self.facing.predict(req, actor=actor, temperature=temperature)
        else:
            return self.root.predict(req, actor=actor, temperature=temperature)

    # -------- predict (batch convenience) --------

    @torch.no_grad()
    def predict_batch(
        self,
        reqs: Sequence["PolicyRequest"],
        *,
        actor: str = "ip",
        temperature: float = 1.0,
    ) -> Sequence["PolicyResponse"]:
        """
        Small helper: splits into root/facing groups and calls underlying
        models per group to avoid head mismatch; preserves input order.
        """
        if not reqs:
            return []

        # partition
        idx_root, idx_facing = [], []
        for i, r in enumerate(reqs):
            (idx_facing if r.facing_bet else idx_root).append(i)

        out: Dict[int, PolicyResponse] = {}

        # root group
        for i in idx_root:
            out[i] = self.root.predict(reqs[i], actor=actor, temperature=temperature)

        # facing group
        for i in idx_facing:
            out[i] = self.facing.predict(reqs[i], actor=actor, temperature=temperature)

        # restore order
        return [out[i] for i in range(len(reqs))]