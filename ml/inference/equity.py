# ml/infer/equity_infer.py
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Union, List
import torch
import torch.nn.functional as F

from ml.models.equity_net import EquityNetLit

try:
    from ml.utils.device import to_device, DeviceLike  # your helper
except Exception:
    DeviceLike = Union[str, torch.device]
    def to_device(wish: DeviceLike = "auto") -> torch.device:
        if wish == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


from ml.utils.sidecar import load_sidecar  # your existing helper

class EquityNetInfer:
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

        self.device = device or to_device("auto")
        self.model.to(self.device)

        # Sanity: model has embeddings for the same categorical features we’ll feed.
        emb_keys = set(getattr(self.model, "emb_layers", {}).keys())
        want_keys = set(self.feature_order)
        if emb_keys and emb_keys != want_keys:
            missing = want_keys - emb_keys
            extra   = emb_keys - want_keys
            raise ValueError(
                f"feature_order ↔ model.emb_layers mismatch. "
                f"Missing in model: {sorted(missing)}  Extra in model: {sorted(extra)}"
            )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        sidecar_path: Union[str, Path],
        device: DeviceLike = "auto",
    ) -> "EquityNetInfer":
        dev = to_device(device)
        sc = load_sidecar(sidecar_path)  # expects feature_order, cards, (id_maps|encoders)

        model = EquityNetLit.load_from_checkpoint(str(checkpoint_path), map_location=dev)
        model.eval().to(dev)

        # Accept both keys; prefer id_maps
        id_maps = sc.get("id_maps") or sc.get("encoders") or {}
        feature_order = sc["feature_order"]
        cards = sc["cards"]

        return cls(
            model=model,
            feature_order=feature_order,
            id_maps=id_maps,
            cards=cards,
            device=dev,
        )

    # ---------- helpers ----------

    def _unknown_idx(self, col: str) -> int:
        """Return a safe 'unknown' index for a column (last bucket)."""
        C = int(self.cards.get(col, 1))
        return max(C - 1, 0)

    def _x_num_placeholder(self, batch_size: int) -> torch.Tensor:
        """If the model expects numeric features, provide zeros; else [B,0]."""
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
        - board_cluster_id: map to UNK so preflop requests (street=0) work.
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

    @torch.no_grad()
    def predict_one(self, row: Mapping[str, Any]) -> List[float]:
        y = self.predict([row])
        return y[0] if y else [0.0, 0.0, 1.0]