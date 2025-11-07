from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence
import torch

from ml.utils.board_mask import make_board_mask_52


def _safe_board_mask_52(board: str) -> List[float]:
    """Fallback 52-mask if project util isn't available. Rank-major order 13x4 (s,h,d,c)."""
    try:
        return list(make_board_mask_52(board))
    except Exception:
        pass
    ranks = "23456789TJQKA"
    suits = "shdc"  # NOTE: may differ from your training; prefer your project util if available
    mask = [0.0] * 52
    if not board:
        return mask
    cards = [board[i:i+2] for i in range(0, len(board), 2)]
    for c in cards:
        if len(c) != 2: continue
        r, s = c[0].upper(), c[1].lower()
        try:
            ri = ranks.index(r); si = suits.index(s)
            idx = ri * 4 + si
            mask[idx] = 1.0
        except ValueError:
            continue
    return mask

def encode_cats(feature_order: Sequence[str],
                cards: Mapping[str, int],
                id_maps: Mapping[str, Mapping[str, int]],
                rows: Sequence[Mapping[str, Any]],
                device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, List[int]] = {c: [] for c in feature_order}
    for r in rows:
        for c in feature_order:
            mapping = id_maps.get(c) or {}
            vocab = int(cards.get(c, 0))
            key = "__NA__" if r.get(c) is None else str(r.get(c))
            if mapping:
                out[c].append(mapping.get(key, max(vocab - 1, 0)))
            else:
                try:
                    out[c].append(int(r.get(c)))
                except Exception:
                    out[c].append(0)
    return {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in out.items()}

def encode_cont(cont_features: Sequence[str],
                rows: Sequence[Mapping[str, Any]],
                device: torch.device) -> Dict[str, torch.Tensor]:
    B = len(rows)
    out: Dict[str, torch.Tensor] = {}
    for name in cont_features:
        if name == "board_mask_52":
            masks = [torch.tensor(_safe_board_mask_52(r.get("board", "")), dtype=torch.float32, device=device) for r in rows]
            out["board_mask_52"] = torch.stack(masks, dim=0)
        else:
            vals = [float(r.get(name, 0.0) or 0.0) for r in rows]
            out[name] = torch.tensor(vals, dtype=torch.float32, device=device).view(B, 1)
    return out

def compute_cluster_id(board: str,
                       clusterer: Optional[Any],
                       sidecar_maps: Mapping[str, Mapping[str, int]],
                       feat_name: Optional[str]) -> Optional[int]:
    """Predict raw cluster id and remap via sidecar id_map if present."""
    if not feat_name:
        return None
    if not board or not clusterer:
        return None
    try:
        raw = int(clusterer.predict_one(board))
    except Exception:
        return None
    # sidecar keys can be "27.0" → str-compare
    mp = sidecar_maps.get(feat_name) or {}
    key = str(raw)
    if key in mp:
        return int(mp[key])
    keyf = f"{float(raw):.1f}"
    return int(mp[keyf]) if keyf in mp else raw