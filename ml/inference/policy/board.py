# File: policy/board.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from ml.config.types_hands import SUITS, RANKS


def parse_board_str(board: Optional[str]) -> List[str]:
    """Return ["Ts","5c","Kd","__","__"] length 5."""
    if not board:
        return ["__","__","__","__","__"]
    s = board.strip()
    cards = [s[i:i+2] for i in range(0, len(s), 2)]
    out = (cards + ["__","__","__","__","__"])[:5]
    return out

def make_board_mask_52(cards5: List[str]) -> List[float]:
    """52-hot mask; '__' ignored."""
    idx = {r+s: (i_s*13 + i_r) for i_s, s in enumerate(SUITS) for i_r, r in enumerate(RANKS)}
    mask = [0.0] * 52
    for c in cards5:
        if c and c != "__" and c in idx:
            mask[idx[c]] = 1.0
    return mask

def map_cluster_id(raw_cid: int, sidecar_idmap: Dict[str, Any]) -> int:
    """Map raw cluster → model's categorical index using sidecar id_maps; default 0."""
    # sidecar stores keys as strings like "27.0"
    key1 = str(raw_cid)
    key2 = f"{float(raw_cid):.1f}"
    if key1 in sidecar_idmap: return int(sidecar_idmap[key1])
    if key2 in sidecar_idmap: return int(sidecar_idmap[key2])
    return 0  # why: stable fallback