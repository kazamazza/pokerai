from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, List, Union
import torch
import torch.nn.functional as F
from ml.models.preflop_rangenet import RangeNetLit


def _load_json(p: Union[str, Path]) -> Any:
    return json.loads(Path(p).read_text())


def _to_device(device: Optional[str | torch.device]) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class RangeNetPreflopInfer:
    """
    Inference wrapper for preflop RangeNet.
    Expects a classification head with 169 outputs (one per canonical hand class).
    Categorical features only (x_num empty).
    """

    def __init__(
        self,
        *,
        model: RangeNetLit,
        feature_order: Sequence[str],
        id_maps: Dict[str, Dict[str, int]],
        cards: Dict[str, int],
        device: Optional[torch.device] = None,
    ):
        self.model = model.eval()
        self.feature_order = list(feature_order)
        # normalize id_maps to {col: {str(raw): int_id}}
        self.id_maps = {c: {str(k): int(v) for k, v in (m or {}).items()} for c, m in (id_maps or {}).items()}
        self.cards = {k: int(v) for k, v in (cards or {}).items()}
        self.device = device or _to_device("auto")
        self.model.to(self.device)

        # infer numeric input size from model hparams (should be 0 for this dataset)
        try:
            self.num_in_dim = int(getattr(self.model.hparams, "num_in_dim", 0))
        except Exception:
            self.num_in_dim = 0

    # ---------- construction helpers ----------

    @classmethod
    def from_checkpoint_dir(
            cls,
            ckpt_path: Union[str, Path],
            ckpt_dir: Optional[Union[str, Path]] = None,
            device: Optional[str | torch.device] = "auto",
    ) -> "RangeNetPreflopInfer":
        ckpt_path = Path(ckpt_path)
        sidecar_dir = Path(ckpt_dir) if ckpt_dir else ckpt_path.parent

        # Load sidecar FIRST so we can pass required ctor args
        feature_order = _load_json(sidecar_dir / "feature_order.json")
        id_maps = _load_json(sidecar_dir / "id_maps.json")
        cards = _load_json(sidecar_dir / "cards.json")

        dev = _to_device(device)

        # Preferred: let Lightning restore and pass required args explicitly
        try:
            model = RangeNetLit.load_from_checkpoint(
                str(ckpt_path),
                map_location=dev,
                cards=cards,
                feature_order=feature_order,
            )
        except TypeError:
            # Fallback: construct, then load state_dict manually (strict=False to be resilient to minor key diffs)
            model = RangeNetLit(cards=cards, feature_order=feature_order)
            ckpt_obj = torch.load(str(ckpt_path), map_location=dev)
            state = ckpt_obj.get("state_dict", ckpt_obj)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[warn] load_state_dict: missing={missing} unexpected={unexpected}")

        model.eval().to(dev)

        return cls(
            model=model,
            feature_order=feature_order,
            id_maps=id_maps,
            cards=cards,
            device=dev,
        )

    # ---------- encoding ----------

    def _unknown_idx(self, col: str) -> int:
        """Return a safe 'unknown' bucket id = last index for that feature."""
        C = int(self.cards.get(col, 0))
        return max(C - 1, 0)

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
        cols: Dict[str, List[Any]] = {k: [] for k in self.feature_order}
        for r in rows:
            for k in self.feature_order:
                if k not in r:
                    raise KeyError(f"Missing feature '{k}' in inference row: {list(r.keys())}")
                cols[k].append(r[k])
        return {k: self._encode_column(k, v) for k, v in cols.items()}

    def _x_num_placeholder(self, batch_size: int) -> torch.Tensor:
        if self.num_in_dim and self.num_in_dim > 0:
            return torch.zeros((batch_size, self.num_in_dim), dtype=torch.float32, device=self.device)
        return torch.empty((batch_size, 0), dtype=torch.float32, device=self.device)

    # ---------- public API ----------

    @torch.no_grad()
    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        """
        rows: list[dict] with keys exactly matching feature_order
              e.g. {"stack_bb":100, "hero_pos":"BB", "opener_pos":"BTN", "opener_action":"RAISE", "ctx":"SRP"}
        returns: FloatTensor [B,169] — probabilities over the 169 preflop hand classes (row sums = 1)
        """
        if not rows:
            return torch.empty(0, 169, device=self.device)

        x_dict = self._encode_batch(rows)  # already produces the dict of tensors
        logits = self.model(x_dict)  # ✅ forward expects one arg
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, rows: Sequence[Mapping[str, Any]]) -> List[List[float]]:
        return self.predict_proba(rows).tolist()