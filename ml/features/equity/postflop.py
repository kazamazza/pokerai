from __future__ import annotations

import hashlib
import json
import threading
from typing import Dict, List, Tuple
import random
import eval7

from ml.features.equity.shared import (
    expand_hand_code_to_combos,
    villain_combo_distribution,
    _sample_discrete,
    _to_ev7, EquityTuple, CardStr, Combo, _overlaps_board, _normalize_probs, _build_remaining_deck,
    _remaining_board_cards_needed,
)

_POSTFLOP_EQ_CACHE: Dict[str, EquityTuple] = {}
_POSTFLOP_EQ_LOCK = threading.Lock()

def _hash_villain_range(vill_range: Dict[str, float]) -> str:
    """
    Stable SHA1 for a range dict. Sort by key to be order-independent.
    """
    payload = json.dumps(sorted(vill_range.items()), separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def _cache_key_postflop(
    board_cards: List[CardStr],
    hero_code: str,
    vill_range_hash: str,
    n_samples: int,
) -> str:
    """
    Compact string key for the cache.
    """
    # normalize board as e.g. "As|Kd|7h"
    bkey = "|".join(board_cards)
    return f"pf:{bkey}|hero:{hero_code}|vr:{vill_range_hash}|n:{n_samples}"

def equity_postflop_vs_range_cached(
    board_cards: List[CardStr],
    hero_code: str,
    vill_range: Dict[str, float],
    n_samples: int = 20000,
    seed: int = 42,
    max_resample: int = 50,
) -> EquityTuple:
    """
    Thin cache around equity_postflop_vs_range.
    Caches by (board, hero_code, vill_range_hash, n_samples).
    """
    vr_hash = _hash_villain_range(vill_range)
    key = _cache_key_postflop(board_cards, hero_code, vr_hash, n_samples)

    with _POSTFLOP_EQ_LOCK:
        hit = _POSTFLOP_EQ_CACHE.get(key)
    if hit is not None:
        return hit

    # call your existing simulator
    win, tie, lose = equity_postflop_vs_range(
        board_cards=board_cards,
        hero_code=hero_code,
        vill_range=vill_range,
        n_samples=n_samples,
        seed=seed,
        max_resample=max_resample,
    )

    with _POSTFLOP_EQ_LOCK:
        _POSTFLOP_EQ_CACHE[key] = (win, tie, lose)

    return (win, tie, lose)

def _pick_hero_combo(hero_code: str, board_cards: List[CardStr], rng: random.Random) -> Combo | None:
    """Uniformly pick a concrete hero combo from the code that does not overlap the board."""
    combos = [c for c in expand_hand_code_to_combos(hero_code)
              if not _overlaps_board(c, board_cards)]
    if not combos:
        return None
    return rng.choice(combos)

def equity_postflop_vs_range(
    board_cards: List[CardStr],
    hero_code: str,
    vill_range: Dict[str, float],
    n_samples: int = 20000,
    seed: int = 42,
    max_resample: int = 30,
) -> EquityTuple:
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


# --- Small cache for hero combos filtered by board (fast path) ---
_HERO_CB_CACHE: Dict[tuple, tuple] = {}
def _hero_combos_for_board_cached(key: tuple) -> tuple | None:
    return _HERO_CB_CACHE.get(key)


def equity_postflop_vs_range_using_combos(
    board_cards: List[CardStr],
    hero_code: str,
    vill_combos: List[Combo],
    vill_weights: List[float],
    n_samples: int = 20000,
    seed: int = 42,
    max_resample: int = 30,
) -> EquityTuple:
    """
    Monte Carlo hero equitynet vs *pre-expanded* villain combos.
    Assumes helpers exist:
      - _to_ev7, _remaining_board_cards_needed, _build_remaining_deck
      - expand_hand_code_to_combos, _overlaps_board
      - _sample_discrete, _normalize_probs
    """
    rng = random.Random(seed)

    # Board convert once
    board_ev7 = [_to_ev7(c) for c in board_cards]
    board_ev7_set = set(board_ev7)
    n_to_draw = _remaining_board_cards_needed(len(board_cards))

    # Normalize villain weights once
    s = float(sum(vill_weights))
    if s <= 0.0:
        return (0.0, 1.0, 0.0)
    base_w = [w / s for w in vill_weights]

    wins = ties = losses = 0

    # Cache hero's legal concrete combos w.r.t. the visible board
    hk = (hero_code, tuple(board_cards))
    hero_cb = _hero_combos_for_board_cached(hk)
    if hero_cb is None:
        combos = [c for c in expand_hand_code_to_combos(hero_code)
                  if not _overlaps_board(c, board_cards)]
        hero_cb = tuple(combos) if combos else tuple()
        _HERO_CB_CACHE[hk] = hero_cb
    if not hero_cb:
        return _normalize_probs(0, 0, 0)

    for _ in range(n_samples):
        # sample one hero combo uniformly among legal ones
        h1, h2 = rng.choice(hero_cb)
        h1c, h2c = _to_ev7(h1), _to_ev7(h2)

        # sample villain combo by weights; reject overlaps
        v1c = v2c = None
        for _try in range(max_resample):
            idx = _sample_discrete(base_w, rng)
            va, vb = vill_combos[idx]
            v1c, v2c = _to_ev7(va), _to_ev7(vb)
            # overlap with hero or board?
            if (v1c == h1c) or (v1c == h2c) or (v2c == h1c) or (v2c == h2c):
                v1c = v2c = None
                continue
            if (v1c in board_ev7_set) or (v2c in board_ev7_set):
                v1c = v2c = None
                continue
            break
        if v1c is None:
            # failed to sample a legal villain combo this round
            continue

        # deal the remaining community cards
        used = [h1c, h2c, v1c, v2c] + board_ev7
        remaining = _build_remaining_deck(used)
        drawn = rng.sample(remaining, k=n_to_draw) if n_to_draw > 0 else []
        full_board = board_ev7 + drawn

        # evaluate
        hs = eval7.evaluate([h1c, h2c] + full_board)
        vs = eval7.evaluate([v1c, v2c] + full_board)

        if hs > vs: wins += 1
        elif hs < vs: losses += 1
        else: ties += 1

    return _normalize_probs(wins, ties, losses)


def equity_postflop_vs_range_cached_combos(
    board_cards: List[CardStr],
    hero_code: str,
    vill_combos: List[Combo],
    vill_weights: List[float],
    n_samples: int = 20000,
    seed: int = 42,
    max_resample: int = 50,
) -> EquityTuple:
    """
    Cached wrapper of the combo-based simulator. Caches by:
      (board_cards, hero_code, SHA1(combos+weights), n_samples)
    Assumes globals:
      - _POSTFLOP_EQ_CACHE: Dict[str, EquityTuple]
      - _POSTFLOP_EQ_LOCK: threading.Lock
      - _cache_key_postflop(board_cards, hero_code, vr_hash, n_samples) -> str
    """
    import hashlib, json

    # Build a stable hash of villain combos + weights
    # (round weights a bit to keep hash stable across tiny float diffs)
    payload = json.dumps(
        {"c": vill_combos, "w": [round(float(w), 9) for w in vill_weights]},
        separators=(",", ":"),
    )
    vr_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()

    key = _cache_key_postflop(board_cards, hero_code, vr_hash, n_samples)
    with _POSTFLOP_EQ_LOCK:
        hit = _POSTFLOP_EQ_CACHE.get(key)
    if hit is not None:
        return hit

    win, tie, lose = equity_postflop_vs_range_using_combos(
        board_cards=board_cards,
        hero_code=hero_code,
        vill_combos=vill_combos,
        vill_weights=vill_weights,
        n_samples=n_samples,
        seed=seed,
        max_resample=max_resample,
    )

    with _POSTFLOP_EQ_LOCK:
        _POSTFLOP_EQ_CACHE[key] = (win, tie, lose)
    return (win, tie, lose)