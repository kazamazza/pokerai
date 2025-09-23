# tools/rangenet/infer/rangenet_infer.py

import json, glob, random, time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ml.datasets.postflop_rangenet import PostflopPolicyDatasetParquet, postflop_policy_collate_fn
from ml.models.preflop_rangenet import RangeNetLit


# ----------------- small utils -----------------
def _to_device(device: Union[str, torch.device] = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def _find_ckpt(ckpt_or_dir: Union[str, Path]) -> Path:
    p = Path(ckpt_or_dir)
    if p.is_file():
        return p
    # prefer best.ckpt > last.ckpt > lexicographically last
    best = p / "best.ckpt"
    if best.exists(): return best
    last = p / "last.ckpt"
    if last.exists(): return last
    cks = sorted(Path(x) for x in glob.glob(str(p / "*.ckpt")))
    if not cks:
        raise FileNotFoundError(f"No checkpoints found under {p}")
    return cks[-1]

def _load_sidecar_any(sidecar_or_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Accept either:
      - single file: *.sidecar.json  ({"feature_order": [...], "id_maps": {...}, "cards": {...}})
      - directory containing feature_order.json, id_maps.json, cards.json
      - checkpoint path => infer sibling *.sidecar.json or directory
    """
    p = Path(sidecar_or_dir)

    # If a checkpoint was passed, try its .sidecar.json next to it
    if p.suffix == ".ckpt":
        sc = p.with_suffix(p.suffix + ".sidecar.json")
        if sc.exists():
            return json.loads(sc.read_text())
        # else try the parent dir triplet
        p = p.parent

    # Single .sidecar.json
    if p.is_file() and p.suffix.endswith(".json"):
        return json.loads(p.read_text())

    # Triplet files in a directory
    fo = p / "feature_order.json"
    im = p / "id_maps.json"
    cd = p / "cards.json"
    if fo.exists() and im.exists() and cd.exists():
        return {
            "feature_order": json.loads(fo.read_text()),
            "id_maps": json.loads(im.read_text()),
            "cards": json.loads(cd.read_text()),
        }

    raise FileNotFoundError(
        f"Could not find sidecar at {sidecar_or_dir} "
        "(expected *.sidecar.json or {feature_order,id_maps,cards}.json)"
    )


# ----------------- encoder helpers -----------------
def _unknown_idx(cards: Dict[str, int], col: str) -> int:
    """Safe 'unknown' index for a column: last bucket."""
    C = int(cards.get(col, 1))
    return max(C - 1, 0)

def _encode_column(values: Sequence[Any], enc: Dict[str, int], card: int, unk: int) -> torch.Tensor:
    ids: List[int] = []
    for v in values:
        key = "__NONE__" if v is None else str(v)
        idx = enc.get(key, unk)
        # clamp to card size just in case
        if idx >= card:
            idx = card - 1
        ids.append(int(idx))
    return torch.tensor(ids, dtype=torch.long)


# ===========================================================
#                 Preflop Inference Wrapper
# ===========================================================
class RangeNetPreflopInfer:
    """
    Inference wrapper for RangeNet (preflop).
    Output: [B,169] probabilities over the 13x13 preflop grid.
    Sidecar must provide: feature_order, id_maps, cards.
    """

    def __init__(
        self,
        *,
        model: RangeNetLit,
        feature_order: Sequence[str],
        id_maps: Dict[str, Dict[str, int]],
        cards: Dict[str, int],
        device: torch.device | None = None,
    ):
        self.model = model.eval()
        self.feature_order = list(feature_order)
        # normalize maps to {str -> int}
        self.id_maps = {c: {str(k): int(v) for k, v in (m or {}).items()} for c, m in (id_maps or {}).items()}
        self.cards = {k: int(v) for k, v in (cards or {}).items()}
        self.device = device or _to_device("auto")
        self.model.to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_or_dir: Union[str, Path],
        sidecar_or_dir: Union[str, Path] | None = None,
        device: Union[str, torch.device] = "auto",
    ) -> "RangeNetPreflopInfer":
        ckpt = _find_ckpt(ckpt_or_dir)
        dev = _to_device(device)
        sc = _load_sidecar_any(sidecar_or_dir or ckpt)

        model = RangeNetLit.load_from_checkpoint(str(ckpt), map_location=dev)
        model.eval().to(dev)

        return cls(
            model=model,
            feature_order=sc["feature_order"],
            id_maps=sc["id_maps"],
            cards=sc["cards"],
            device=dev,
        )

    # -------------- encoding --------------
    def _encode_batch(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        if not rows:
            return {k: torch.empty(0, dtype=torch.long, device=self.device) for k in self.feature_order}

        cols: Dict[str, List[Any]] = {k: [] for k in self.feature_order}
        for r in rows:
            for k in self.feature_order:
                if k not in r:
                    raise KeyError(f"Missing feature '{k}' in row: keys={list(r.keys())}")
                cols[k].append(r[k])

        out: Dict[str, torch.Tensor] = {}
        for feat in self.feature_order:
            enc = self.id_maps.get(feat, {})
            card = int(self.cards.get(feat, max(len(enc), 1)))
            unk = _unknown_idx(self.cards, feat) if not enc else min(max(len(enc), 1) - 1, card - 1)
            out[feat] = _encode_column(cols[feat], enc, card, unk).to(self.device)
        return out

    # -------------- public API --------------
    @torch.no_grad()
    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        """
        rows: list of dicts with keys matching feature_order.
        returns: [B,169] probabilities.
        """
        x = self._encode_batch(rows)
        logits = self.model(x)           # [B,169] unnormalized
        return F.softmax(logits, dim=-1) # [B,169]

    @torch.no_grad()
    def predict(self, rows: Sequence[Mapping[str, Any]]) -> List[List[float]]:
        return self.predict_proba(rows).tolist()
