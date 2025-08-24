# ml/inference/population.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Sequence, Optional, Union

import json
import torch
import torch.nn.functional as F

from ml.models.population_net import PopulationNetLit  # your LightningModule

DeviceLike = Union[str, torch.device]

def _to_device(device: Optional[DeviceLike]) -> torch.device:
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")
    return torch.device(device)

class PopulationNetInfer:
    """
    Simple inference wrapper for PopulationNet.
    Requires:
      - checkpoint (.ckpt)
      - sidecar JSON with:
          { "feature_order": ["stakes_id","street_id",...],
            "cards": {"stakes_id":4,...},
            "encoders": {"stakes_id":{"NL10":0,...}, ...} }   # raw -> id
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        sidecar_json: str | Path,
        device: DeviceLike = "auto",
    ):
        self.device = _to_device(device)

        meta = json.loads(Path(sidecar_json).read_text())
        self.feature_order: List[str] = list(meta["feature_order"])
        self.cards: Dict[str, int] = dict(meta["cards"])
        self.encoders: Dict[str, Dict[Any, int]] = {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in meta["encoders"].items()}

        # Load model
        self.model: PopulationNetLit = PopulationNetLit(cards=self.cards, feature_order=self.feature_order)
        state = torch.load(str(ckpt_path), map_location="cpu")
        # Lightning puts weights under 'state_dict'
        self.model.load_state_dict(state["state_dict"])
        self.model.eval().to(self.device)

    def _encode_row(self, raw: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        raw values -> categorical IDs using encoders; unseen → last ID (UNK bucket).
        """
        x: Dict[str, torch.Tensor] = {}
        for name in self.feature_order:
            enc = self.encoders[name]
            # enc keys are strings; coerce raw to str for lookup
            raw_key = str(raw[name])
            if raw_key in enc:
                idx = enc[raw_key]
            else:
                # unseen bucket: append index = len(enc)
                idx = len(enc)
            x[name] = torch.tensor([idx], dtype=torch.long, device=self.device)
        return x

    @torch.no_grad()
    def predict_proba(self, rows: Sequence[Dict[str, Any]]) -> torch.Tensor:
        """
        rows: list of raw feature dicts (keys match feature_order).
        returns: probs [B, 3] for [fold, call, raise]
        """
        if not rows:
            return torch.empty(0, 3)
        # Stack encoded columns
        cols: Dict[str, List[int]] = {k: [] for k in self.feature_order}
        for r in rows:
            for k in self.feature_order:
                enc = self.encoders[k]
                raw_key = str(r[k])
                cols[k].append(enc.get(raw_key, len(enc)))  # UNK if unseen

        x_dict = {k: torch.tensor(v, dtype=torch.long, device=self.device) for k, v in cols.items()}
        logits = self.model(x_dict)                          # [B,3]
        probs = F.softmax(logits, dim=-1)                    # [B,3]
        return probs

    @torch.no_grad()
    def predict(self, rows: Sequence[Dict[str, Any]]) -> List[int]:
        """
        returns hard action ids (0=fold,1=call,2=raise)
        """
        probs = self.predict_proba(rows)
        if probs.numel() == 0:
            return []
        return probs.argmax(dim=-1).tolist()