from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, Union, Sequence, Mapping, Any, Dict, Tuple, List
import torch
from ml.features.boards import BoardClusterer
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
        clusterer: Optional[BoardClusterer] = None,
    ):
        self.root = root
        self.facing = facing
        self.device = device or self.root.device  # keep consistent
        self.clusterer: Optional["BoardClusterer"] = None

        # quick sanity: vocab disjointness hints
        assert "CHECK" in self.root.action_vocab, "root model should include CHECK"
        assert "CALL" in self.facing.action_vocab and "FOLD" in self.facing.action_vocab, \
            "facing model should include CALL/FOLD"
        self.board_cluster_feat = (
                self.root.board_cluster_feat or self.facing.board_cluster_feat
        )
        if clusterer is not None:
            self.set_clusterer(clusterer)

    def set_clusterer(self, clusterer: Optional["BoardClusterer"]) -> None:
        """Propagate a board clusterer to both singles."""
        # Why: singles compute board_cluster_id internally; must share same artifact as training.
        self.clusterer = clusterer
        if hasattr(self.root, "clusterer"):
            self.root.clusterer = clusterer
        if hasattr(self.facing, "clusterer"):
            self.facing.clusterer = clusterer

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

    @staticmethod
    def _coerce_side(side: Optional[str]) -> Optional[str]:
        """Only 'root'/'facing' allowed; everything else -> None (auto)."""
        if side is None:
            return None
        s = str(side).strip().lower()
        if s in ("root", "facing"):
            return s
        # tolerate accidental values like 'ip'/'oop'/'hero' etc.
        return None

    @staticmethod
    def _street_name(street: int) -> str:
        return {0: "pre", 1: "flop", 2: "turn", 3: "river"}.get(int(street), "flop")

    @classmethod
    def _derive_side(cls, req) -> Tuple[str, Dict[str, Any]]:
        """
        Decide root/facing from request. Returns (side, debug).
        Precedence:
          1) explicit facing_bet
          2) explicit faced_size_* present
          3) parse actions_hist on current street
          4) default root
        """
        dbg: Dict[str, Any] = {}

        # 1) explicit flag
        fb = getattr(req, "facing_bet", None)
        if isinstance(fb, bool):
            dbg["source"] = "explicit_flag"
            return ("facing" if fb else "root", dbg)

        # 2) explicit faced size
        fsf = getattr(req, "faced_size_frac", None)
        fsp = getattr(req, "faced_size_pct", None)
        if fsf is not None or fsp is not None:
            dbg["source"] = "faced_size_present"
            return ("facing", dbg)

        # 3) parse history (coarse but robust)
        actions: List[str] = getattr(req, "actions_hist", None) or []
        street = int(getattr(req, "street", 1) or 1)
        vpos = str(getattr(req, "villain_pos", "") or "").upper()
        sname = cls._street_name(street)
        bet_like = re.compile(r"\b(bet|raise|donk)\b", re.I)
        pass_like = re.compile(r"\b(check|call|fold)\b", re.I)

        for line in reversed(actions):
            t = str(line or "")
            if not t.lower().startswith(sname + ":"):
                continue
            is_villain = (vpos.lower() in t.lower())
            if is_villain and bet_like.search(t):
                dbg["source"] = "history"
                dbg["line"] = t
                return ("facing", dbg)
            if is_villain and pass_like.search(t):
                dbg["source"] = "history"
                dbg["line"] = t
                return ("root", dbg)
            if (not is_villain) and bet_like.search(t):
                dbg["source"] = "history"
                dbg["line"] = t
                return ("root", dbg)

        # 4) default
        dbg["source"] = "default_root"
        return ("root", dbg)

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
        side_norm = self._coerce_side(side)

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