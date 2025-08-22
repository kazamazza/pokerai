from __future__ import annotations
from typing import Dict, List, Tuple
import random
import eval7
from ml.config.types_hands import SUITS, RANKS
from ml.features.equity.shared import expand_hand_code_to_combos, villain_combo_distribution, _to_ev7, \
    _sample_discrete


def equity_preflop_vs_range(
    hero_code: str,
    vill_range: Dict[str, float],
    n_samples: int = 20000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Monte Carlo estimate of preflop (p_win, p_tie, p_lose) for a HERO canonical hand code vs a villain range.

    - For each hero concrete combo (e.g., AKs → 4 suited combos), we:
        * Build villain **combo** distribution with blockers taken into account.
        * Sample villain combos according to that distribution.
        * Sample 5-card boards from the remaining deck.
        * Evaluate win/tie/lose using eval7.
    - Average across hero combos uniformly.

    Returns probabilities that sum ~1.
    """
    # Expand hero code to concrete combos (average uniformly across them)
    hero_combos = expand_hand_code_to_combos(hero_code)
    if not hero_combos:
        return (0.0, 0.0, 1.0)

    rng = random.Random(seed)

    # Prebuild full deck of eval7.Card
    deck = [eval7.Card(r + s) for r in RANKS for s in SUITS]

    total_win = 0.0
    total_tie = 0.0
    total_lose = 0.0

    for hc in hero_combos:
        h1, h2 = _to_ev7(hc[0]), _to_ev7(hc[1])

        # Build villain combo distribution with blockers, then arrays for fast sampling
        vill_dist = villain_combo_distribution(vill_range, hc)
        if not vill_dist:
            # No valid villain hands given blockers: treat as auto-win (or skip).
            # We'll skip contribution here.
            continue

        vill_combos = [(_to_ev7(c[0]), _to_ev7(c[1])) for c, _ in vill_dist]
        weights = [p for _, p in vill_dist]

        # Build remaining deck for this hero combo (remove hero cards once)
        remaining_base = [c for c in deck if c != h1 and c != h2]

        win = tie = lose = 0

        for _ in range(n_samples // len(hero_combos)):
            # Sample a villain combo index
            vi = _sample_discrete(weights, rng)
            v1, v2 = vill_combos[vi]

            # Remove villain cards from deck (respecting blockers)
            # If overlap ever slipped in (shouldn't), resample
            if v1 == h1 or v1 == h2 or v2 == h1 or v2 == h2 or v1 == v2:
                continue

            # Build deck without h1,h2,v1,v2
            rem = [c for c in remaining_base if c != v1 and c != v2]

            # Sample 5-card board
            board = rng.sample(rem, 5)

            # Evaluate
            hero_rank = eval7.evaluate([h1, h2] + board)
            vill_rank = eval7.evaluate([v1, v2] + board)

            if hero_rank > vill_rank:
                win += 1
            elif hero_rank < vill_rank:
                lose += 1
            else:
                tie += 1

        n = win + tie + lose
        if n == 0:
            continue
        total_win += win / n
        total_tie += tie / n
        total_lose += lose / n

    m = len(hero_combos)
    if m == 0:
        return (0.0, 0.0, 1.0)

    # Average across hero combos
    p_win = total_win / m
    p_tie = total_tie / m
    p_lose = total_lose / m

    # numerical safety to ensure they sum to 1.0
    s = p_win + p_tie + p_lose
    if s > 0:
        p_win /= s; p_tie /= s; p_lose /= s
    return (p_win, p_tie, p_lose)