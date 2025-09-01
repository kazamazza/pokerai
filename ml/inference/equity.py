from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import torch
import torch.nn.functional as F
from ml.models.equity_net import EquityNetLit  # your LightningModule
from ml.utils.device import DeviceLike, to_device
from ml.utils.sidecar import load_sidecar


class EquityNetInfer:
    """
    Inference wrapper for EquityNet (preflop OR postflop).

    Sidecar JSON schema:
      {
        "feature_order": ["stack_bb","hero_pos","opener_action", ...],
        "cards": {"stack_bb":X,...},
        "encoders": {"stack_bb":{"12":0,"15":1,...}, "hero_pos":{"BB":0,...}, ...}
      }

    Model output: [p_win, p_tie, p_lose]
    """

    def __init__(
        self,
        *,
        model: EquityNetLit,
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
        sidecar_path: str | Path,
        device: DeviceLike = "auto",
    ) -> "EquityNetInfer":
        dev = to_device(device)
        sc = load_sidecar(sidecar_path)

        model = EquityNetLit.load_from_checkpoint(checkpoint_path, map_location=dev)
        model.eval().to(dev)

        return cls(
            model=model,
            feature_order=sc["feature_order"],
            id_maps=sc["encoders"],  # 👈 sidecar uses "encoders"
            cards=sc["cards"],
            device=dev,
        )

    def _encode_column(self, feat: str, values: List[Any]) -> torch.Tensor:
        enc = self.id_maps[feat]  # e.g. {"BTN":0,"SB":1,...}
        card = int(self.cards[feat])  # total categorical size
        unk_idx = max(card - 1, 0)  # always reserve last as UNK

        ids = []
        for v in values:
            key = str(v) if v is not None else "__NONE__"
            idx = enc.get(key, unk_idx)
            if idx >= card:  # clamp to valid range
                idx = unk_idx
            ids.append(int(idx))
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _x_num_placeholder(self, batch_size: int) -> torch.Tensor:
        """
        Build the numeric feature tensor to pass to the model.
        If the model was configured with num_in_dim == 0, pass an empty tensor [B,0].
        Otherwise pass zeros [B, num_in_dim].
        """
        # Try to read num_in_dim from the LightningModule hparams
        num_in_dim = 0
        try:
            num_in_dim = int(getattr(self.model.hparams, "num_in_dim", 0))
        except Exception:
            pass

        if num_in_dim and num_in_dim > 0:
            return torch.zeros((batch_size, num_in_dim), dtype=torch.float32, device=self.device)
        else:
            return torch.empty((batch_size, 0), dtype=torch.float32, device=self.device)

    def _encode_batch(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        cols: Dict[str, List[Any]] = {k: [] for k in self.feature_order}
        for r in rows:
            for k in self.feature_order:
                cols[k].append(r[k])
        return {k: self._encode_column(k, v) for k, v in cols.items()}

    @torch.no_grad()
    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        """
        rows: list of dicts with keys matching feature_order.
        returns: [B,3] = [p_win, p_tie, p_lose]
        """
        if not rows:
            return torch.empty(0, 3, device=self.device)

        x_cat = self._encode_batch(rows)
        x_num = self._x_num_placeholder(batch_size=len(rows))

        logits = self.model(x_cat, x_num)   # <-- pass BOTH inputs
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, rows: Sequence[Mapping[str, Any]]) -> List[List[float]]:
        return self.predict_proba(rows).tolist()