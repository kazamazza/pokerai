# ml/etl/rangenet/utils/range_format.py
from typing import List
from ml.config.types_hands import RANKS

def _hand_label(i: int, j: int) -> str:
    r1, r2 = RANKS[i], RANKS[j]
    if i == j:
        return r1 + r2
    return (r1 + r2 + "s") if i < j else (r2 + r1 + "o")

def vec169_to_monker_string(v169: List[float]) -> str:
    if len(v169) != 169:
        raise ValueError("Expected 169-length vector")
    parts = []
    k = 0
    for i in range(13):
        for j in range(13):
            parts.append(f"{_hand_label(i,j)}:{float(v169[k]):.6f}")
            k += 1
    return ",".join(parts)