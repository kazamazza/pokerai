from typing import List

RANKS = "AKQJT98765432"
SUITS = "cdhs"
RANK_TO_I = {r: i for i, r in enumerate(RANKS)}

def _pair(code: str) -> bool:
    return len(code) == 2 and code[0] == code[1]

def all_169_hands() -> List[str]:
    """
    Canonical 169 preflop classes in a stable order:
      - Pairs: AA, KK, ..., 22
      - Suited: AKs, AQs, ..., 32s (upper triangle)
      - Offsuit: AKo, AQo, ..., 32o (upper triangle)
    """
    hands: List[str] = []
    # pairs
    for r in RANKS:
        hands.append(r + r)
    # suited & offsuit (only i<j to avoid duplicates)
    for i, hi in enumerate(RANKS):
        for j in range(i + 1, len(RANKS)):
            lo = RANKS[j]
            hands.append(hi + lo + "s")
    for i, hi in enumerate(RANKS):
        for j in range(i + 1, len(RANKS)):
            lo = RANKS[j]
            hands.append(hi + lo + "o")
    return hands

ALL_HANDS = all_169_hands()
HAND_TO_ID = {h: i for i, h in enumerate(ALL_HANDS)}  # AA -> 0, ..., 32o -> 168