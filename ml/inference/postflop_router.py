from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, Union, Sequence, Mapping, Any, Dict, Tuple, List
import torch
from ml.features.boards import BoardClusterer
from ml.inference.policy.types import PolicyResponse, PolicyRequest
from ml.inference.postflop_infer_single import PostflopPolicyInferSingle


# postflop/router.py
from typing import Optional, Union, List, Tuple

class PostflopPolicyRouter:
    """
    Holds two single-side inferencers:
      - root   : CHECK / BET_* / DONK_* (OOP)
      - facing : FOLD / CALL / RAISE_* / ALLIN

    Selection:
      - If side override provided → respect it.
      - Else if req.facing_bet is not None → respect it.
      - Else infer from actions_hist (strict).
    """

    def __init__(
        self,
        *,
        root,
        facing,
        device=None,
        clusterer=None,
    ):
        self.root = root
        self.facing = facing
        self.device = device or self.root.device
        self.clusterer = None

        assert "CHECK" in self.root.action_vocab, "root model should include CHECK"
        assert "CALL"  in self.facing.action_vocab and "FOLD" in self.facing.action_vocab, \
            "facing model should include CALL and FOLD"

        self.board_cluster_feat = self.root.board_cluster_feat or self.facing.board_cluster_feat
        if clusterer is not None:
            self.set_clusterer(clusterer)

    def set_clusterer(self, clusterer) -> None:
        self.clusterer = clusterer
        if hasattr(self.root, "clusterer"):   self.root.clusterer = clusterer
        if hasattr(self.facing, "clusterer"): self.facing.clusterer = clusterer

    @classmethod
    def from_dirs(cls, *, root_dir, facing_dir, device: str = "auto"):
        r = PostflopPolicyInferSingle.from_dir(root_dir, device=device)
        f = PostflopPolicyInferSingle.from_dir(facing_dir, device=device)
        dev = r.device
        f.model.to(dev)
        return cls(root=r, facing=f, device=dev)

    @classmethod
    def from_checkpoints(
        cls, *,
        root_ckpt, root_sidecar,
        facing_ckpt, facing_sidecar,
        device: str = "auto",
    ):
        r = PostflopPolicyInferSingle.from_checkpoint(root_ckpt, root_sidecar, device=device)
        f = PostflopPolicyInferSingle.from_checkpoint(facing_ckpt, facing_sidecar, device=device)
        dev = r.device
        f.model.to(dev)
        return cls(root=r, facing=f, device=dev)

    # ---------- side selection helpers ----------
    @staticmethod
    def _coerce_side(side: Optional[str]) -> Optional[str]:
        if side is None: return None
        s = str(side).strip().lower()
        return s if s in ("root", "facing") else None

    @staticmethod
    def _explicit_facing(req) -> Tuple[Optional[bool], Optional[float]]:
        """
        Return (facing_bet, faced_size_frac) if explicitly provided by orchestrator.
        """
        fb = getattr(req, "facing_bet", None)
        sz = getattr(req, "faced_size_frac", None)
        try:
            sz = float(sz) if sz is not None else None
        except Exception:
            sz = None
        return (bool(fb) if fb is not None else None), sz

    @staticmethod
    def _is_villain(entry, villain_id: Optional[str]) -> bool:
        if villain_id is not None:
            return entry.player_id == villain_id
        return str(entry.player_id or "").upper() not in {"", "HERO"}

    def _infer_from_history(self, req) -> Tuple[bool, Optional[float]]:
        """
        Strict flop-only inference:
          - Look at actions_hist for street==1 (flop).
          - If the latest non-check villain action is BET/RAISE
            and hero hasn't responded with CALL/RAISE afterward → facing.
          - Otherwise not facing.
        Size is unknown here (None). We do NOT guess size.
        """
        hist = getattr(req, "actions_hist", None) or []
        flop = [e for e in hist if int(getattr(e, "street", 1) or 1) == 1]
        if not flop:
            return False, None

        vill_id = getattr(req, "villain_id", None)

        for idx in range(len(flop) - 1, -1, -1):
            a = flop[idx]
            A = str(getattr(a, "action", "")).upper()
            if A in {"BET", "RAISE"} and self._is_villain(a, vill_id):
                # Did hero respond after this?
                responded = any(
                    (not self._is_villain(flop[j], vill_id)) and
                    str(getattr(flop[j], "action", "")).upper() in {"CALL", "RAISE"}
                    for j in range(idx + 1, len(flop))
                )
                if not responded:
                    return True, None
                break
            if A in {"CALL", "RAISE"} and (not self._is_villain(a, vill_id)):
                break
        return False, None

    # ---------- public shims ----------
    def infer_facing_and_size(self, req, *, hero_is_ip: bool):
        fb, sz = self._explicit_facing(req)
        if fb is not None:
            return fb, (sz if fb else None)
        return self._infer_from_history(req)

    def infer_root_menu(self, req, *, hero_is_ip: bool):
        fb, _ = self._explicit_facing(req)
        if fb is None:
            fb, _ = self._infer_from_history(req)
        is_root = not fb

        bet_menu: Optional[List[float]] = None
        if is_root:
            if isinstance(getattr(req, "bet_sizes", None), (list, tuple)) and len(req.bet_sizes) > 0:
                try:
                    bet_menu = [float(x) for x in req.bet_sizes]
                except Exception:
                    bet_menu = None
            if bet_menu is None:
                try:
                    bet_menu = sorted({
                        float(int(tok.split("_", 1)[1]))
                        for tok in self.root.action_vocab
                        if tok.startswith("BET_") and tok.split("_", 1)[1].isdigit()
                    })
                except Exception:
                    bet_menu = None
            if not bet_menu:
                bet_menu = [0.33, 0.66]  # fractions only
        return is_root, bet_menu

    # ---------- main predict ----------
    @torch.no_grad()
    def predict(self, req: "PolicyRequest", *, actor: str = "ip", temperature: float = 1.0, side: Optional[str] = None):
        side_norm = self._coerce_side(side)
        if side_norm == "root":
            # normalize request for root
            req.facing_bet = False
            req.faced_size_frac = None
            return self.root.predict(req, actor=actor, temperature=temperature)
        if side_norm == "facing":
            # ensure size present (default 0.33 if missing)
            req.facing_bet = True
            req.faced_size_frac = float(req.faced_size_frac) if req.faced_size_frac is not None else 0.33
            return self.facing.predict(req, actor=actor, temperature=temperature)

        # AUTO
        try:
            hero_is_ip = PolicyRequest.is_hero_ip(req.hero_pos or "", req.villain_pos or "")
        except Exception:
            hero_is_ip = True

        fb, sz = self._explicit_facing(req)
        if fb is None:
            fb, sz = self._infer_from_history(req)

        # Normalize req for downstream
        req.facing_bet = bool(fb)
        if fb:
            req.faced_size_frac = float(sz) if sz is not None else 0.33
        else:
            req.faced_size_frac = None

        engine = self.facing if fb else self.root
        return engine.predict(req, actor=actor, temperature=temperature)