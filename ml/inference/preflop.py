from __future__ import annotations
from pathlib import Path
from typing import Union, Sequence, Dict, Optional, Any, List
import torch.nn.functional as F
import numpy as np
import torch
from ml.inference.policy.types import PolicyRequest
from ml.inference.preflop_seq import infer_preflop_action_seq
from ml.models.preflop_rangenet import RangeNetLit
from ml.utils.sidecar import load_sidecar

DeviceLike = Union[str, torch.device]

def _to_device(device: DeviceLike = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))

class PreflopPolicy:
    """
    Single, self-contained preflop policy that:
      - loads the trained RangeNet (Lightning) from ckpt(+sidecar)
      - encodes a single inference row using sidecar id_maps/cards
      - predicts a simple action distribution (with optional equity nudge)
    """

    def __init__(
        self,
        *,
        model: RangeNetLit,
        feature_order: Sequence[str],
        cards: Dict[str, int],
        id_maps: Dict[str, Dict[str, int]] | None,
        device: torch.device,
    ):
        self.model = model.eval().to(device)
        self.device = device

        self.feature_order = list(feature_order)
        self.cards = {k: int(v) for k, v in cards.items()}
        self.id_maps = {k: {str(a): int(b) for a, b in m.items()} for k, m in (id_maps or {}).items()}

    # ---------- loaders ----------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        sidecar_path: Union[str, Path],
        *,
        device: DeviceLike = "auto",
    ) -> "PreflopPolicy":
        dev = _to_device(device)
        sc = load_sidecar(sidecar_path)
        feature_order = sc["feature_order"]
        cards = {str(k): int(v) for k, v in sc["cards"].items()}
        id_maps = sc.get("id_maps") or {}

        # ⚠️ RangeNetLit needs cards/feature_order passed to constructor
        model = RangeNetLit.load_from_checkpoint(
            str(checkpoint_path),
            map_location=dev,
            cards=cards,
            feature_order=feature_order,
        )
        model.eval().to(dev)

        return cls(
            model=model,
            feature_order=feature_order,
            cards=cards,
            id_maps=id_maps,
            device=dev,
        )

    @classmethod
    def from_dir(
        cls,
        ckpt_dir: Union[str, Path],
        ckpt_name: Optional[str] = None,
        device: DeviceLike = "auto",
    ) -> "PreflopPolicy":
        d = Path(ckpt_dir)
        if ckpt_name is None:
            cands = sorted(d.glob("range_preflop-*-*.ckpt"))
            ckpt = cands[0] if cands else (d / "best.ckpt")
        else:
            ckpt = d / ckpt_name
        sidecar = d / "best_sidecar.json"
        return cls.from_checkpoint(ckpt, sidecar, device=device)

    def _encode_row(self, row: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for feat in self.feature_order:
            enc = self.id_maps.get(feat, {})
            card = int(self.cards.get(feat, max(len(enc), 1)))
            unk = min(max(len(enc), 1) - 1, card - 1) if enc else self._unknown_idx(feat)

            val = row.get(feat, None)
            key = "__NONE__" if val is None else str(val).upper()
            idx = enc.get(key, unk)
            if idx >= card:
                idx = card - 1
            out[feat] = torch.tensor([int(idx)], dtype=torch.long, device=self.device)
        return out


    def _bucket_stack(self, val: float) -> float:
        # snap to nearest categorical key from sidecar (e.g., 25.0, 60.0, 100.0, 150.0)
        mp = self.id_maps.get("stack_bb", {}) or {}
        try:
            keys = [float(k) for k in mp.keys()]
            return min(keys, key=lambda x: abs(x - float(val))) if keys else float(val)
        except Exception:
            return float(val)

    @torch.no_grad()
    def predict(self, req: "PolicyRequest", *, quiet: bool = True) -> np.ndarray:
        """
        PURE RangeNet inference (169-d). Uses sidecar’s exact feature_order:
          ["stack_bb","hero_pos","villain_pos","action_seq_1","action_seq_2","action_seq_3"]
        """
        stack = float(getattr(req, "eff_stack_bb", None) or getattr(req, "pot_bb", None) or 100.0)
        hero = (getattr(req, "hero_pos", "") or "").upper()
        vill = (getattr(req, "villain_pos", "") or "").upper()
        seq = infer_preflop_action_seq(getattr(req, "actions_hist", None) or [], hero)

        row_raw = {
            "stack_bb": self._bucket_stack(stack),  # will stringify inside encoder
            "hero_pos": hero,
            "villain_pos": vill,
            "action_seq_1": seq["action_seq_1"],
            "action_seq_2": seq["action_seq_2"],
            "action_seq_3": seq["action_seq_3"],
        }

        xb = self._encode_row(row_raw)
        logits = self.model(xb)  # [1,169]
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        s = float(probs.sum())
        if s <= 1e-8:
            probs = (np.ones(169, dtype="float32") / 169.0)
        else:
            probs = (probs / s).astype("float32")

        if not quiet:
            print("[PreflopPolicy] row_raw:", row_raw)
        return probs