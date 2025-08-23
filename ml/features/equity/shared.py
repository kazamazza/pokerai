import random
from functools import lru_cache
from typing import Tuple, List, Dict
import eval7
from ml.config.types_hands import SUITS, RANKS

CardStr = str         # e.g. "As"
Combo   = Tuple[str,str]  # ("As","Kd")
EquityTuple = Tuple[float, float, float]  # (p_win, p_tie, p_lose)

@lru_cache(maxsize=256)
def _to_ev7(card_str: str) -> eval7.Card:
    return eval7.Card(card_str)

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

@lru_cache(maxsize=512)
def expand_hand_code_to_combos(hand_code: str) -> Tuple[Combo, ...]:
    """
    Return all concrete 2-card combos (as ('Ah','Kd') strings) for a canonical
    169 code ('AA','AKs','AKo',...). Sorted for determinism.
    """
    ranks = "AKQJT98765432"
    suits = "cdhs"

    def combos_pair(r: str) -> List[Combo]:
        # pairs: choose any 2 suits without order
        out = []
        for i, s1 in enumerate(suits):
            for s2 in suits[i+1:]:
                out.append((r + s1, r + s2))
        return out

    def combos_suited(hi: str, lo: str) -> List[Combo]:
        return [(hi + s, lo + s) for s in suits]

    def combos_offsuit(hi: str, lo: str) -> List[Combo]:
        out = []
        for s1 in suits:
            for s2 in suits:
                if s1 == s2:
                    continue
                out.append((hi + s1, lo + s2))
        return out

    code = hand_code.strip()
    if len(code) == 2 and code[0] == code[1]:
        # pair
        combos = combos_pair(code[0])
    elif len(code) == 3 and code[2] in ("s", "o"):
        hi, lo, t = code[0], code[1], code[2]
        if t == "s":
            combos = combos_suited(hi, lo)
        else:
            combos = combos_offsuit(hi, lo)
    else:
        # treat as exact combo like "AhAd" or "AhKd"
        if len(code) == 4:
            combos = [(code[:2], code[2:])]
        else:
            combos = []

    # Normalize ordering (lexicographic) for deterministic cache keys
    combos = [tuple(sorted(c)) for c in combos]
    combos.sort()
    return tuple(combos)

def cards_overlap(a: Combo, b: Combo) -> bool:
    """True if the two combos share a physical card."""
    return (a[0] in b) or (a[1] in b) or (b[0] in a) or (b[1] in a)

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

def expand_range_to_combos_weighted(vill_range: Dict[str, float]) -> Tuple[List[Combo], List[float]]:
    combos: List[Combo] = []
    weights: List[float] = []

    for code, p in vill_range.items():
        if p <= 0:
            continue
        cs = expand_hand_code_to_combos(code)  # e.g. "AKs" -> [("As","Ks"), ("Ah","Kh"), ...]
        if not cs:
            continue
        w_each = p / len(cs)
        for c in cs:
            combos.append(c)
            weights.append(w_each)

    s = sum(weights)
    if s > 0:
        weights = [w / s for w in weights]
    return combos, weights

def _sample_index(weights: List[float], rng: random.Random) -> int:
    r, acc = rng.random(), 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1  # numerical tail

def _normalize_probs(win: int, tie: int, lose: int) -> EquityTuple:
    total = win + tie + lose
    if total <= 0:
        # fallback if no trials succeeded
        return (0.0, 1.0, 0.0)
    return (win / total, tie / total, lose / total)

def _remaining_board_cards_needed(board_len: int) -> int:
    """How many community cards are left to deal to reach 5."""
    if board_len < 0 or board_len > 5:
        raise ValueError(f"board must have 0..5 cards; got {board_len}")
    return 5 - board_len


@lru_cache(maxsize=1)
def all_52_eval7() -> List[eval7.Card]:
    ranks = "AKQJT98765432"
    suits = "cdhs"
    return [eval7.Card(r + s) for r in ranks for s in suits]

def _overlaps_board(combo: Combo, board_cards: List[CardStr]) -> bool:
    a, b = combo
    used = set(board_cards)
    return (a in used) or (b in used)


def _build_remaining_deck(used_cards: List[eval7.Card]) -> List[eval7.Card]:
    """All 52 minus used_cards."""
    used_set = set(used_cards)
    return [c for c in all_52_eval7() if c not in used_set]


def _draw_board(board_cards: List[CardStr], rng: random.Random) -> List[eval7.Card]:
    """
    Given some fixed board_cards (flop/turn/river),
    draw the missing ones until length 5.
    """
    board_ev7 = [_to_ev7(c) for c in board_cards]
    n_to_draw = _remaining_board_cards_needed(len(board_ev7))

    used = set(board_ev7)
    remaining = [c for c in all_52_eval7() if c not in used]

    if n_to_draw > 0:
        drawn = rng.sample(remaining, k=n_to_draw)
    else:
        drawn = []
    return board_ev7 + drawn


def _evaluate_hand(hero: Combo, vill: Combo, board: List[eval7.Card]) -> EquityTuple:
    """Evaluate hero vs vill on this board once; return (win,tie,lose) as 0/1."""
    h1, h2 = _to_ev7(hero[0]), _to_ev7(hero[1])
    v1, v2 = _to_ev7(vill[0]), _to_ev7(vill[1])

    hero_score = eval7.evaluate([h1, h2] + board)
    vill_score = eval7.evaluate([v1, v2] + board)

    if hero_score > vill_score:
        return (1, 0, 0)
    elif hero_score < vill_score:
        return (0, 0, 1)
    else:
        return (0, 1, 0)