from __future__ import annotations
from typing import Dict, List, Tuple, Any
import random
import eval7
from ml.config.types_hands import SUITS, RANKS
from ml.features.equity.shared import expand_hand_code_to_combos, villain_combo_distribution, _to_ev7, \
    _sample_discrete, EquityTuple, _build_remaining_deck, _evaluate_hand, _normalize_probs

# Module-level cache
_EQ_PREFLOP_CACHE: Dict[str, Any] = {}


def _cache_key_vill(vill_range: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    """Stable key for villain range dict."""
    return tuple(sorted(vill_range.items()))

def equity_preflop_vs_range_cached(
    hero_code: str,
    vill_range: Dict[str, float],
    n_samples: int = 20000,
    seed: int = 42,
    max_resample: int = 50,
) -> EquityTuple:
    """
    Monte Carlo estimate of hero preflop equity vs a villain range (no board).
    Uses caching for:
      - hero_code -> concrete combos
      - vill_range -> expanded base villain combo list + weights

    Notes:
      - Villain is sampled with rejection to respect hero blockers (no overlap).
      - Board is drawn uniformly from the remaining 52 minus {hero,villain}.
    """
    rng = random.Random(seed)

    # ---- Cache hero combos (uniform over concrete combos of the code) ----
    hero_key = f"hero:{hero_code}"
    if hero_key not in _EQ_PREFLOP_CACHE:
        combos = expand_hand_code_to_combos(hero_code)  # List[Tuple[str,str]]
        if not combos:
            # If the code is invalid or fully blocked somehow, return neutral tie.
            return (0.0, 1.0, 0.0)
        _EQ_PREFLOP_CACHE[hero_key] = combos
    hero_combos: List[Tuple[str, str]] = _EQ_PREFLOP_CACHE[hero_key]

    # ---- Cache villain base combos & weights (before hero blockers) ----
    vill_key = ("vill:", _cache_key_vill(vill_range))
    if vill_key not in _EQ_PREFLOP_CACHE:
        # Expand range into concrete combos with weights proportional to hand prob / #combos
        base_combos: List[Tuple[str, str]] = []
        base_weights: List[float] = []
        for code, p in vill_range.items():
            if p <= 0:
                continue
            c_list = expand_hand_code_to_combos(code)
            if not c_list:
                continue
            w = float(p) / float(len(c_list))
            for c in c_list:
                base_combos.append(c)
                base_weights.append(w)
        # Normalize weights (defensive)
        s = float(sum(base_weights))
        if s > 0:
            base_weights = [w / s for w in base_weights]
        _EQ_PREFLOP_CACHE[vill_key] = (base_combos, base_weights)
    vill_base_combos, vill_base_weights = _EQ_PREFLOP_CACHE[vill_key]

    # If villain has nothing, neutral tie
    if not vill_base_combos:
        return (0.0, 1.0, 0.0)

    wins = ties = losses = 0

    # Preconvert all 52 once for speed (tiny micro-opt)
    # (We’ll still call _to_ev7 for the two drawn hero/villain cards)
    for _ in range(n_samples):
        # 1) Draw concrete hero combo uniformly
        h = rng.choice(hero_combos)
        h1c, h2c = _to_ev7(h[0]), _to_ev7(h[1])

        # 2) Sample villain combo with rejection until no overlap with hero
        v_combo = None
        for _try in range(max_resample):
            idx = _sample_discrete(vill_base_weights, rng)
            cand = vill_base_combos[idx]
            v1c, v2c = _to_ev7(cand[0]), _to_ev7(cand[1])
            # No overlap with hero cards
            if v1c != h1c and v1c != h2c and v2c != h1c and v2c != h2c:
                v_combo = (v1c, v2c)
                break
        if v_combo is None:
            # Couldn’t draw a non-overlapping villain hand this round
            continue
        v1c, v2c = v_combo

        # 3) Build remaining deck and sample a full 5-card board
        used = [h1c, h2c, v1c, v2c]
        remaining = _build_remaining_deck(used)
        # Defensive: ensure we have at least 5 cards left
        if len(remaining) < 5:
            continue
        board = rng.sample(remaining, k=5)

        # 4) Evaluate once
        w, t, l = _evaluate_hand((h[0], h[1]), (str(v1c), str(v2c)), board)
        wins += w
        ties += t
        losses += l

    return _normalize_probs(wins, ties, losses)

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