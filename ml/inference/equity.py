# ml/inference/equity.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import json
import torch
import torch.nn.functional as F

from ml.models.equity_net import EquityNetLit  # your LightningModule

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

class EquityNetInfer:
    """
    Generic inference for EquityNet (preflop OR postflop).
    Needs a sidecar JSON:
      {
        "feature_order": ["stack_bb","hero_pos","opener_action", ...],
        "cards": {"stack_bb":X,...},
        "encoders": {"stack_bb":{"12":0,"15":1,...}, "hero_pos":{"BB":0,...}, ...}
      }
    Output: probs [p_win, p_tie, p_lose]
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
        self.encoders: Dict[str, Dict[str, int]] = {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in meta["encoders"].items()}

        self.model: EquityNetLit = EquityNetLit(cards=self.cards, cat_order=self.feature_order)
        state = torch.load(str(ckpt_path), map_location="cpu")
        self.model.load_state_dict(state["state_dict"])
        self.model.eval().to(self.device)

    @torch.no_grad()
    def predict_proba(self, rows: List[Dict[str, Any]]) -> torch.Tensor:
        """
        rows: list of raw feature dicts (keys must match sidecar feature_order)
        returns: probs [B,3] = [p_win, p_tie, p_lose]
        """
        if not rows:
            return torch.empty(0, 3)
        cols: Dict[str, List[int]] = {k: [] for k in self.feature_order}
        for r in rows:
            for k in self.feature_order:
                enc = self.encoders[k]
                raw_key = str(r[k])
                cols[k].append(enc.get(raw_key, len(enc)))  # UNK bucket if unseen

        x_dict = {k: torch.tensor(v, dtype=torch.long, device=self.device) for k, v in cols.items()}
        logits = self.model(x_dict)                       # [B,3]
        probs = F.softmax(logits, dim=-1)                 # [B,3]
        return probs

    @torch.no_grad()
    def predict(self, rows: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Convenience wrapper → Python lists
        """
        probs = self.predict_proba(rows)
        return probs.tolist()