# ml/populationnet/infer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from ml.models.population_net import PopulationNetLit

DeviceLike = Union[str, torch.device]


def _to_device(device: Optional[DeviceLike]) -> torch.device:
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        # guard MPS import/availability
        try:
            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")
    return torch.device(device)


def _load_sidecar(sidecar: Union[dict, str, Path]) -> dict:
    if isinstance(sidecar, (str, Path)):
        return json.loads(Path(sidecar).read_text())
    return sidecar


class PopulationNetInfer:
    """
    Lightweight inference wrapper for PopulationNet.

    Expects sidecar with keys:
      - feature_order: List[str]
      - id_maps: Dict[str, Dict[str, int]]     # stringified tokens -> IDs
      - cards: Dict[str, int]                  # embedding cardinalities per feature
    """

    def __init__(
        self,
        *,
        model: PopulationNetLit,
        feature_order: Sequence[str],
        id_maps: Dict[str, Dict[str, int]],
        cards: Dict[str, int],
        device: Optional[torch.device] = None,
    ):
        self.model = model.eval()
        self.feature_order = list(feature_order)
        self.id_maps = id_maps
        self.cards = cards
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        sidecar: Union[dict, str, Path],
        device: DeviceLike = "auto",
    ) -> "PopulationNetInfer":
        dev = _to_device(device)
        sc = _load_sidecar(sidecar)

        model = PopulationNetLit.load_from_checkpoint(checkpoint_path, map_location=dev)
        model.eval().to(dev)

        return cls(
            model=model,
            feature_order=sc["feature_order"],
            id_maps=sc["id_maps"],
            cards=sc["cards"],
            device=dev,
        )

    # ---------- encoding helpers ----------

    def _encode_column(self, feat: str, values: List[Any]) -> torch.Tensor:
        enc = self.id_maps[feat]
        card = int(self.cards[feat])  # embedding size
        # If training reserved an UNK bucket, it should be the last index.
        # If not, clamp unseen to last valid index to avoid OOB.
        unk_idx = card - 1 if card > len(enc) else max(len(enc) - 1, 0)

        ids: List[int] = []
        for v in values:
            key = str(v)
            idx = enc.get(key, None)
            if idx is None:
                # unseen → UNK (safe)
                idx = unk_idx
            # also clamp for safety if card shrank (shouldn't happen, but guard anyway)
            if idx >= card:
                idx = card - 1
            ids.append(int(idx))
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _encode_batch(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        cols: Dict[str, List[Any]] = {k: [] for k in self.feature_order}
        for r in rows:
            for k in self.feature_order:
                cols[k].append(r[k])
        return {k: self._encode_column(k, v) for k, v in cols.items()}

    # ---------- public inference API ----------

    @torch.no_grad()
    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        """
        rows: list of dicts with keys matching feature_order.
        returns: [B, 3] probs for actions [fold, call, raise]
        """
        if not rows:
            return torch.empty(0, 3, device=self.device)
        x_dict = self._encode_batch(rows)
        logits = self.model(x_dict)        # [B, 3]
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, rows: Sequence[Mapping[str, Any]]) -> List[int]:
        """Hard decisions: argmax over probs. Returns list[int] of size B."""
        probs = self.predict_proba(rows)
        if probs.numel() == 0:
            return []
        return probs.argmax(dim=-1).tolist()