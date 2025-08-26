from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional
import torch
import numpy as np

from ml.models.rangenet import RangeNetLit  # your Lightning module
from ml.utils.sidecar import load_sidecar


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _normalize_device(device: Optional[object]) -> torch.device:
    """
    Accepts: None | torch.device | "cpu" | "cuda" | "auto"
    Returns: torch.device
    """
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    # default
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RangeNetInfer:
    """
    Unified inference for RangeNet (preflop & postflop).
    Expects a checkpoint directory containing:
      - feature_order.json
      - id_maps.json
      - cards.json
      - (and the model checkpoint itself)
    """

    def __init__(
        self,
        model: RangeNetLit,
        feature_order: Sequence[str],
        id_maps: Dict[str, Dict[str, int]],
        cards: Dict[str, int],
        device: Optional[torch.device] = None,
    ):
        self.model = model.eval()
        self.feature_order = list(feature_order)
        self.id_maps = id_maps
        self.cards = cards
        self.device = device or _best_device()
        self.model.to(self.device)

    @classmethod
    def from_checkpoint(
            cls,
            checkpoint_path: str,
            sidecar_path: str | Path,
            device: Optional[torch.device] = None,
    ):
        dev = _normalize_device(device)
        model = RangeNetLit.load_from_checkpoint(checkpoint_path, map_location=dev)
        model.eval().to(dev)
        sc = load_sidecar(sidecar_path)
        return cls(
            model=model,
            feature_order=sc["feature_order"],
            id_maps=sc["id_maps"],
            cards=sc["cards"],
            device=dev,
        )

    def _encode_row(self, row: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Map raw categorical features → integer IDs using saved id_maps.
        Unseen values go to an implicit 'unknown' bucket at index = cards[col]-1 if needed.
        """
        out: Dict[str, torch.Tensor] = {}
        for col in self.feature_order:
            val = row.get(col, None)
            enc = self.id_maps.get(col, {})
            # id_maps keys are strings (JSON); normalize to str for lookup
            key = str(val) if val is not None else "__NONE__"
            if key in enc:
                idx = enc[key]
            else:
                # unknown bucket: last index (cards[col]-1), if available
                C = int(self.cards.get(col, 0))
                idx = C - 1 if C > 0 else 0
            out[col] = torch.tensor(idx, dtype=torch.long, device=self.device)
        return out

    def _collate(self, batch_rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        cols = self.feature_order
        tensors = {c: [] for c in cols}
        for r in batch_rows:
            enc = self._encode_row(r)
            for c in cols:
                tensors[c].append(enc[c])
        for c in cols:
            tensors[c] = torch.stack(tensors[c], dim=0)  # [B]
        return tensors

    @torch.no_grad()
    def predict_one(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Returns a (169,) numpy array of probabilities over the 13x13 hand grid ordering.
        """
        batch = self._collate([features])
        logits_or_probs = self.model(batch)  # model.forward returns [B,169]
        probs = torch.softmax(logits_or_probs, dim=-1) if logits_or_probs.dtype.is_floating_point else logits_or_probs
        return probs[0].detach().cpu().numpy()

    @torch.no_grad()
    def predict_batch(self, rows: List[Dict[str, Any]], as_numpy: bool = True) -> np.ndarray | torch.Tensor:
        """
        rows: list of raw feature dicts
        Returns: [B,169] array/tensor of probabilities.
        """
        if not rows:
            return np.zeros((0, 169), dtype=np.float32) if as_numpy else torch.zeros((0, 169))
        batch = self._collate(rows)
        logits_or_probs = self.model(batch)
        probs = torch.softmax(logits_or_probs, dim=-1) if logits_or_probs.dtype.is_floating_point else logits_or_probs
        if as_numpy:
            return probs.detach().cpu().numpy()
        return probs