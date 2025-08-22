from __future__ import annotations
from typing import Dict, List, Tuple
import random
import eval7

from ml.features.equity.shared import (
    expand_hand_code_to_combos,
    villain_combo_distribution,
    _sample_discrete,
    _to_ev7,
)

CardStr = str  # e.g. "As", "Td"
Combo = Tuple[CardStr, CardStr]
EquityTuple = Tuple[float, float, float]  # (p_win, p_tie, p_lose)


def _normalize_probs(win: int, tie: int, lose: int) -> EquityTuple:
    total = win + tie + lose
    if total <= 0:
        # If nothing evaluated (e.g., filtering removed all combos), fallback to neutral tie.
        return (0.0, 1.0, 0.0)
    return (win / total, tie / total, lose / total)


def _remaining_board_cards_needed(board_len: int) -> int:
    """How many community cards are left to deal to reach 5."""
    if board_len < 0 or board_len > 5:
        raise ValueError(f"board must have 0..5 cards; got {board_len}")
    return 5 - board_len


def _all_52_eval7() -> List[eval7.Card]:
    """Return all 52 cards as eval7.Card objects."""
    deck = []
    ranks = "AKQJT98765432"
    suits = "cdhs"
    for r in ranks:
        for s in suits:
            deck.append(eval7.Card(r + s))
    return deck


def _pick_hero_combo(hero_code: str, board_cards: List[CardStr], rng: random.Random) -> Combo | None:
    """Uniformly pick a concrete hero combo from the code that does not overlap the board."""
    combos = [c for c in expand_hand_code_to_combos(hero_code)
              if not _overlaps_board(c, board_cards)]
    if not combos:
        return None
    return rng.choice(combos)


def _overlaps_board(combo: Combo, board_cards: List[CardStr]) -> bool:
    a, b = combo
    used = set(board_cards)
    return (a in used) or (b in used)


def _build_remaining_deck(used_cards: List[eval7.Card]) -> List[eval7.Card]:
    """All 52 minus used_cards."""
    used_set = {c for c in used_cards}
    return [c for c in _all_52_eval7() if c not in used_set]


def equity_postflop_vs_range(
    board_cards: List[CardStr],
    hero_code: str,
    vill_range: Dict[str, float],
    n_samples: int = 20000,
    seed: int = 42,
    max_resample: int = 30,
) -> EquityTuple:
    """
    Monte Carlo estimate of hero equity (win/tie/lose) postflop vs a villain range.

    Args:
      board_cards: list of visible community cards as strings, length 3/4/5 (e.g., ["As","Kd","7h"])
      hero_code: canonical 169 code ("AA","AKs","AKo",...)
      vill_range: dict {hand_code -> prob} that sums ~1.0 (will be renormalized per-blockers)
      n_samples: number of Monte Carlo trials
      seed: RNG seed
      max_resample: attempts to draw a non-overlapping villain combo

    Returns:
      (p_win, p_tie, p_lose) as floats summing to 1.0
    """
    rng = random.Random(seed)

    # Convert board once
    board_ev7 = [_to_ev7(c) for c in board_cards]
    n_to_draw = _remaining_board_cards_needed(len(board_cards))

    wins = ties = losses = 0

    for _ in range(n_samples):
        # 1) Pick a concrete hero combo that doesn't overlap board
        hero_combo = _pick_hero_combo(hero_code, board_cards, rng)
        if hero_combo is None:
            # No legal hero combo (e.g., board has blockers that remove all combos)
            continue
        h1, h2 = hero_combo
        h1c, h2c = _to_ev7(h1), _to_ev7(h2)

        # 2) Build villain combo distribution given hero blockers
        #    EXPECTS: list of (combo_tuple, weight_float)
        vill_dist = villain_combo_distribution(vill_range, hero_combo)
        if not vill_dist:
            # Nothing left after blockers — skip sample
            continue

        # --- NEW: split into parallel lists for sampling by weights ---
        vill_combos: List[Tuple[CardStr, CardStr]] = [c for (c, w) in vill_dist]
        vill_weights: List[float] = [w for (c, w) in vill_dist]
        s = sum(vill_weights)
        if s <= 0:
            continue
        vill_weights = [w / s for w in vill_weights]
        # ----------------------------------------------------------------

        # 3) Sample a villain combo that does not overlap the board (and hero)
        v_combo = None
        for _try in range(max_resample):
            # sample index by weights, then pick combo
            idx = _sample_discrete(vill_weights, rng=rng)
            v_combo = vill_combos[idx]
            va, vb = _to_ev7(v_combo[0]), _to_ev7(v_combo[1])
            # overlap checks vs hero and board
            if va in (h1c, h2c) or vb in (h1c, h2c):
                v_combo = None
                continue
            if va in board_ev7 or vb in board_ev7:
                v_combo = None
                continue
            break
        if v_combo is None:
            # Could not draw a valid villain combo this round
            continue
        v1c, v2c = _to_ev7(v_combo[0]), _to_ev7(v_combo[1])

        # 4) Build remaining deck and draw missing community cards
        used = [h1c, h2c, v1c, v2c] + board_ev7
        remaining = _build_remaining_deck(used)
        if n_to_draw > 0:
            if len(remaining) < n_to_draw:
                continue
            drawn = rng.sample(remaining, k=n_to_draw)
        else:
            drawn = []

        full_board = board_ev7 + drawn

        # 5) Evaluate hands
        hero_score = eval7.evaluate([h1c, h2c] + full_board)
        vill_score = eval7.evaluate([v1c, v2c] + full_board)

        if hero_score > vill_score:
            wins += 1
        elif hero_score < vill_score:
            losses += 1
        else:
            ties += 1

    return _normalize_probs(wins, ties, losses)