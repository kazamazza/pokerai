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

    Sidecar JSON schema must include:
      - feature_order: list[str]
      - cards: dict[str,int]
      - id_maps (preferred) or encoders: dict[str, dict[str,int]]
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

        # Normalize id_maps to {col: {str(raw): int(id)}}
        self.id_maps = {
            col: {str(k): int(v) for k, v in (mapping or {}).items()}
            for col, mapping in (id_maps or {}).items()
        }
        self.cards = {k: int(v) for k, v in (cards or {}).items()}

        # device
        self.device = device or to_device("auto")
        self.model.to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        sidecar_path: str | Path,
        device: DeviceLike = "auto",
    ) -> "EquityNetInfer":
        dev = to_device(device)
        sc = load_sidecar(sidecar_path)  # expects feature_order, cards, (id_maps|encoders)

        model = EquityNetLit.load_from_checkpoint(str(checkpoint_path), map_location=dev)
        model.eval().to(dev)

        # Accept both keys; prefer id_maps
        id_maps = sc.get("id_maps") or sc.get("encoders") or {}

        return cls(
            model=model,
            feature_order=sc["feature_order"],
            id_maps=id_maps,
            cards=sc["cards"],
            device=dev,
        )

    # ---------- helpers ----------

    def _unknown_idx(self, col: str) -> int:
        """Return a safe 'unknown' index for a column (last bucket)."""
        C = int(self.cards.get(col, 0))
        return max(C - 1, 0)

    def _x_num_placeholder(self, batch_size: int) -> torch.Tensor:
        """
        Build the numeric feature tensor to pass to the model.
        If the model has num_in_dim==0, pass [B,0]; else zeros [B,num_in_dim].
        """
        num_in_dim = 0
        try:
            num_in_dim = int(getattr(self.model.hparams, "num_in_dim", 0))
        except Exception:
            pass

        if num_in_dim and num_in_dim > 0:
            return torch.zeros((batch_size, num_in_dim), dtype=torch.float32, device=self.device)
        else:
            return torch.empty((batch_size, 0), dtype=torch.float32, device=self.device)

    def _maybe_fill_defaults(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill unified-but-optional features if missing.
        - board_cluster_id: map to UNK so preflop requests (street=0) work out-of-the-box.
        """
        out = dict(row)
        if "board_cluster_id" not in out:
            out["board_cluster_id"] = self._unknown_idx("board_cluster_id")
        return out

    def _encode_column(self, feat: str, values: List[Any]) -> torch.Tensor:
        enc = self.id_maps.get(feat, {})
        card = int(self.cards.get(feat, max(len(enc), 1)))
        unk_idx = min(max(len(enc), 1) - 1, card - 1) if enc else self._unknown_idx(feat)

        ids: List[int] = []
        for v in values:
            key = str(v) if v is not None else "__NONE__"
            idx = enc.get(key, unk_idx)
            if idx >= card:
                idx = card - 1
            ids.append(int(idx))
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _encode_batch(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        prepared: List[Dict[str, Any]] = [self._maybe_fill_defaults(dict(r)) for r in rows]

        cols: Dict[str, List[Any]] = {k: [] for k in self.feature_order}
        for r in prepared:
            for k in self.feature_order:
                if k not in r:
                    raise KeyError(f"Missing feature '{k}' in inference row: {r.keys()}")
                cols[k].append(r[k])
        return {k: self._encode_column(k, v) for k, v in cols.items()}

    # ---------- public API ----------

    @torch.no_grad()
    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        """
        rows: list of dicts with keys matching feature_order (board_cluster_id may be omitted for preflop).
        returns: [B,3] probabilities (p_win, p_tie, p_lose)
        """
        if not rows:
            return torch.empty(0, 3, device=self.device)

        x_cat = self._encode_batch(rows)
        x_num = self._x_num_placeholder(batch_size=len(rows))
        logits = self.model(x_cat, x_num)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, rows: Sequence[Mapping[str, Any]]) -> List[List[float]]:
        return self.predict_proba(rows).tolist()