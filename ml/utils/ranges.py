# ml/utils/ranges.py
from typing import Dict, List
from .constants import HAND2ID

def opp_range_vec169(opp_map: Dict[str, float]) -> List[float]:
    vec = [0.0] * 169
    for h, w in opp_map.items():
        idx = HAND2ID.get(h)
        if idx is not None:
            vec[idx] = float(w)
    return vec