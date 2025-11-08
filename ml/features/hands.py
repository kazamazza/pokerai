# ml/features/hands.py
from __future__ import annotations
from typing import List, Tuple
import eval7  # pip install eval7

from ml.config.types_hands import ALL_HANDS, SUITS, RANKS

HANDS_169: List[str] = ALL_HANDS

HAND169_TO_ID: dict[str, int] = {lab: i for i, lab in enumerate(HANDS_169)}

def hand_to_169_label(cards: str) -> str:
    """Return canonical 169 label like 'AKs', 'QJo', '77'."""
    s = cards.strip().upper()
    if len(s) != 4:  # e.g. "AsKh"
        return ""
    r1, s1, r2, s2 = s[0], s[1], s[2], s[3]
    if r1 == r2:
        return r1 + r2
    # order by rank (A high)
    i1, i2 = RANKS.index(r1), RANKS.index(r2)
    hi, lo, suited = (r1, r2, s1 == s2) if i1 < i2 else (r2, r1, s1 == s2)
    return f"{hi}{lo}{'s' if suited else 'o'}"

def hand169_label_to_id(label: str) -> int | None:
    """Map 'AKs'/'QJo'/'77' → 0..168 as defined by HANDS_169 order."""
    return HAND169_TO_ID.get(label)

def hand_code_from_id(hid: int) -> str:
    """
    Inverse of HAND_TO_ID: 0..168 -> canonical hand code ("AA", "AKs", "AKo", ...).
    """
    if hid < 0 or hid >= len(ALL_HANDS):
        raise IndexError(f"hand id out of range: {hid}")
    return ALL_HANDS[hid]

def enumerate_suited_combos(code: str) -> List[Tuple[eval7.Card, eval7.Card]]:
    """
    Expand a 169 hand code to all real 2-card combos (as eval7.Card pairs).
      - Pairs: 6 combos (choose 2 of 4 suits)
      - Suited: 4 combos (same suit)
      - Offsuit: 12 combos (different suits)

    Returns list of tuples [(Card(rank1+s), Card(rank2+s2)), ...].
    The first card in each tuple corresponds to the first rank in `code`.
    """
    # Normalize & basic parse
    code = code.strip()
    if len(code) not in (2, 3):
        raise ValueError(f"Invalid hand code: {code}")

    # Determine ranks and suitedness
    if len(code) == 2:  # pair like "AA"
        r1 = r2 = code[0]
        typ = "pair"
    else:
        r1, r2, tag = code[0], code[1], code[2].lower()
        if tag == "s":
            typ = "suited"
        elif tag == "o":
            typ = "offsuit"
        else:
            raise ValueError(f"Invalid suitedness tag in {code}")

    # Build cards by suit
    def make(rank: str, suit: str) -> eval7.Card:
        return eval7.Card(rank + suit)

    combos: List[Tuple[eval7.Card, eval7.Card]] = []

    if typ == "pair":
        # choose 2 suits out of 4 (6 combos)
        for i, s1 in enumerate(SUITS):
            for s2 in SUITS[i+1:]:
                c1 = make(r1, s1)
                c2 = make(r2, s2)
                combos.append((c1, c2))
    elif typ == "suited":
        # both cards same suit (4 combos)
        for s in SUITS:
            c1 = make(r1, s)
            c2 = make(r2, s)
            combos.append((c1, c2))
    else:  # offsuit
        # different suits (4*3 = 12 combos)
        for s1 in SUITS:
            for s2 in SUITS:
                if s1 == s2:
                    continue
                c1 = make(r1, s1)
                c2 = make(r2, s2)
                combos.append((c1, c2))

    return combos