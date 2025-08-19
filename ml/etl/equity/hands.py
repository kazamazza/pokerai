from typing import List, Tuple

RANKS = ['A','K','Q','J','T','9','8','7','6','5','4','3','2']

def hand_id_table() -> List[str]:
    ids = []
    # pairs
    ids += [r+r for r in RANKS]
    # suited (above diag): high-first, e.g., AKs
    ids += [RANKS[i]+RANKS[j]+'s' for i in range(len(RANKS)) for j in range(i+1, len(RANKS))]
    # offsuit (below diag): high-first, e.g., AKo
    ids += [RANKS[i]+RANKS[j]+'o' for i in range(len(RANKS)) for j in range(i+1, len(RANKS))]
    assert len(ids) == 169
    return ids

ID2HAND = hand_id_table()
HAND2ID = {h:i for i,h in enumerate(ID2HAND)}

def canon_hand_from_id(hid: int) -> str:
    return ID2HAND[hid]  # 'AKs', 'AQo', 'TT'