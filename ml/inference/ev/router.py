from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from pathlib import Path
import torch

from ml.inference.ev.infer_single import EVInferSingle, EVOutput


class EVRouter:
    """
    Routes to the correct EV model:
      - street==0 -> preflop
      - street>0  -> root/facing chosen like your policy router
    """

    def __init__(
        self,
        *,
        preflop: Optional[EVInferSingle],
        root: EVInferSingle,
        facing: EVInferSingle,
        device: Optional[torch.device] = None,
        clusterer: Optional[Any] = None,
    ):
        self.preflop = preflop
        self.root = root
        self.facing = facing
        self.device = device or self.root.device
        if clusterer is not None:
            self.set_clusterer(clusterer)

    def set_clusterer(self, clusterer: Any) -> None:
        if self.root:    self.root.clusterer = clusterer
        if self.facing:  self.facing.clusterer = clusterer
        if self.preflop: self.preflop.clusterer = clusterer

    @staticmethod
    def _is_facing(req, hero_is_ip: bool) -> Tuple[bool, Optional[float]]:
        try:
            from ml.inference.postflop_single.facing import infer_facing_and_size
            f, s, _ = infer_facing_and_size(req, hero_is_ip=hero_is_ip)
            return bool(f), s
        except Exception:
            fb = bool(getattr(req, "facing_bet", False))
            frac = getattr(req, "faced_size_frac", None)
            return fb, (float(frac) if fb and frac is not None else None)

    @staticmethod
    def _hero_is_ip(req) -> bool:
        h = (getattr(req, "hero_pos", "") or "").upper()
        v = (getattr(req, "villain_pos", "") or "").upper()
        street = int(getattr(req, "street", 0) or 0)
        if street == 0:
            return False
        if h == "BTN" and v in ("SB", "BB"):
            return True
        if {h, v} == {"SB", "BB"}:
            return h == "BB"
        order = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
        try:
            return order.index(h) > order.index(v)
        except ValueError:
            return True

    # ml/inference/ev/router.py
    @torch.no_grad()
    def predict(self, req, *, side: str | None = None, tokens: Sequence[str] | None = None) -> EVOutput:
        street = int(getattr(req, "street", 0) or 0)

        if street == 0:
            if not self.preflop:
                return EVOutput(False, [], [], {}, {"err": "no_preflop"}, "no_preflop")
            out = self.preflop.predict(req, tokens=tokens)
            return out

        # choose split
        if side is None:
            hero_is_ip = self._hero_is_ip(req)
            facing, _ = self._is_facing(req, hero_is_ip=hero_is_ip)
            side = "facing" if facing else "root"

        inf = self.facing if side == "facing" else self.root
        return inf.predict(req, tokens=tokens)

    # Convenience constructors
    @classmethod
    def from_checkpoints(
        cls,
        *,
        preflop_ckpt: Optional[Union[str, Path]],
        preflop_sidecar: Optional[Union[str, Path]],
        root_ckpt: Union[str, Path],
        root_sidecar: Union[str, Path],
        facing_ckpt: Union[str, Path],
        facing_sidecar: Union[str, Path],
        device: str = "auto",
        clusterer: Optional[Any] = None,
    ) -> "EVRouter":
        pre = None
        if preflop_ckpt and preflop_sidecar:
            pre = EVInferSingle.from_checkpoint(
                checkpoint_path=preflop_ckpt,
                sidecar_path=preflop_sidecar,
                mode="preflop",
                device=device,
                clusterer=clusterer,
            )
        root = EVInferSingle.from_checkpoint(
            checkpoint_path=root_ckpt,
            sidecar_path=root_sidecar,
            mode="root",
            device=device,
            clusterer=clusterer,
        )
        face = EVInferSingle.from_checkpoint(
            checkpoint_path=facing_ckpt,
            sidecar_path=facing_sidecar,
            mode="facing",
            device=device,
            clusterer=clusterer,
        )
        dev = root.device
        return cls(preflop=pre, root=root, facing=face, device=dev, clusterer=clusterer)