import random
from typing import Tuple, List, Dict
import eval7
from ml.config.types_hands import SUITS, RANKS

CardStr = str         # e.g. "As"
Combo   = Tuple[str,str]  # ("As","Kd")

def _to_ev7(card: CardStr) -> eval7.Card:
    return eval7.Card(card)

def _sample_discrete(weights: List[float], rng: random.Random) -> int:
    """Return index sampled according to weights (sum ~1)."""
    r = rng.random()
    cum = 0.0
    for i, w in enumerate(weights):
        cum += w
        if r <= cum:
            return i
    return len(weights) - 1  # numeric safety

def _is_class_code(code: str) -> bool:
    # 2-char pairs like "AA" or 3-char like "AKs"/"AJo"
    if len(code) == 2:
        return code[0] in RANKS and code[1] in RANKS and code[0] == code[1]
    if len(code) == 3:
        hi, lo, t = code[0], code[1], code[2]
        return (hi in RANKS and lo in RANKS and hi != lo and t in ("s", "o"))
    return False

def _is_exact_combo(code: str) -> bool:
    # 4-char like "AhAd" or "7h8h"
    if len(code) != 4:
        return False
    r1, s1, r2, s2 = code[0], code[1], code[2], code[3]
    if r1 not in RANKS or r2 not in RANKS or s1 not in SUITS or s2 not in SUITS:
        return False
    # cannot be the same physical card, suits can be equal only if ranks differ
    # (two identical cards is impossible)
    return not (r1 == r2 and s1 == s2)

def expand_hand_code_to_combos(code: str) -> List[Combo]:
    """
    Expand a canonical code (e.g., 'AA', 'AKs', 'AKo') into all concrete 2-card combos.
      - Pairs: 6 combos (choose 2 suits out of 4)
      - Suited: 4 combos (same suit)
      - Offsuit: 12 combos (different suits)
    Returns list of tuples like ("As","Ah"), ("Ad","Ac"), ...
    """

    code = code.strip()

    if _is_exact_combo(code):
        return [(code[:2], code[2:])]

    if not _is_class_code(code):
        raise ValueError(f"Unexpected hand code: {code}")

    # Pair, e.g., "AA"
    if len(code) == 2 and code[0] == code[1]:
        r = code[0]
        out: List[Combo] = []
        # choose 2 suits from 4 → 6 combos
        s_list = list(SUITS)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                s1, s2 = s_list[i], s_list[j]
                out.append((r + s1, r + s2))
        return out

    # Two different ranks, with 's' or 'o'
    assert len(code) == 3 and code[2] in ("s", "o"), f"Expected suited/offsuit suffix in {code}"
    hi, lo, suit_flag = code[0], code[1], code[2]

    if suit_flag == "s":
        # same suit (4 combos)
        return [(hi + s, lo + s) for s in SUITS]

    # offsuit: different suits (12 combos)
    out: List[Combo] = []
    for s1 in SUITS:
        for s2 in SUITS:
            if s1 == s2:
                continue
            out.append((hi + s1, lo + s2))
    return out


def cards_overlap(a: Combo, b: Combo) -> bool:
    """True if the two combos share a physical card."""
    return (a[0] in b) or (a[1] in b) or (b[0] in a) or (b[1] in a)


# ---------- Build villain COMBO distribution given hero blockers ----------

def villain_combo_distribution(vill_range: Dict[str, float], hero_combo: Combo) -> List[Tuple[Combo, float]]:
    """
    Expand a villain canonical range { 'AA':p, 'AKs':q, ... } into a list of concrete
    combos [(('As','Ah'), prob), ...] after removing combos blocked by hero cards.
    Probabilities are renormalized to sum to 1.0.
    """
    tmp: List[Tuple[Combo, float]] = []

    for code, p_code in vill_range.items():
        if p_code <= 0:
            continue
        combos = expand_hand_code_to_combos(code)
        # Keep only combos that don't collide with hero cards
        valid = [c for c in combos if not cards_overlap(hero_combo, c)]
        if not valid:
            continue
        # Split the hand-level prob equally among its remaining combos
        p_each = p_code / len(valid)
        for c in valid:
            tmp.append((c, p_each))

    total = sum(p for _, p in tmp)
    if total <= 0:
        return []  # empty / invalid range after blocking
    # Renormalize
    return [(c, p / total) for c, p in tmp]