# ml/utils/constants.py
from typing import List, Dict

RANKS = ['A','K','Q','J','T','9','8','7','6','5','4','3','2']  # high -> low
SUITS = ['s','h','d','c']                                      # fixed suit order
R2I   = {r:i for i,r in enumerate(RANKS)}                      # A=0 .. 2=12
S2I   = {s:i for i,s in enumerate(SUITS)}                      # s=0 .. c=3

def hand_id_table() -> List[str]:
    ids: List[str] = []
    # pairs
    ids += [r+r for r in RANKS]
    # suited (above diag): high-first, e.g., AKs
    ids += [RANKS[i]+RANKS[j]+'s' for i in range(len(RANKS)) for j in range(i+1, len(RANKS))]
    # offsuit (below diag): high-first, e.g., AKo
    ids += [RANKS[i]+RANKS[j]+'o' for i in range(len(RANKS)) for j in range(i+1, len(RANKS))]
    assert len(ids) == 169
    return ids

ID2HAND: List[str] = hand_id_table()
HAND2ID: Dict[str, int] = {h: i for i, h in enumerate(ID2HAND)}