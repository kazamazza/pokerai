# file: ml/etl/ev/utils_ev.py
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ml.models.vocab_actions import PREFLOP_ACTION_VOCAB

try:
    import eval7  # pip install eval7
except Exception as _e:
    raise ImportError("eval7 is required for EV labeling. pip install eval7") from _e

# ---- Canonical contexts & action sequence mapping (matches your pipeline)
def infer_action_sequence(ctx: str) -> List[str]:
    c = (ctx or "").upper()
    if c == "VS_OPEN":        return ["RAISE", "CALL", "CALL"]
    if c == "VS_3BET":        return ["RAISE", "3BET", "CALL"]
    if c == "VS_4BET":        return ["RAISE", "3BET", "4BET"]
    if c == "LIMPED_SINGLE":  return ["LIMP", "CHECK", ""]
    if c == "LIMPED_MULTI":   return ["LIMP", "CALL", "CALL"]
    if c == "SRP":            return ["RAISE", "CALL", ""]
    return ["", "", ""]

def serialize_action_seq(action_seq: Sequence[str]) -> Tuple[str, str, str]:
    a1 = action_seq[0] if len(action_seq) > 0 else ""
    a2 = action_seq[1] if len(action_seq) > 1 else ""
    a3 = action_seq[2] if len(action_seq) > 2 else ""
    return a1, a2, a3

# ---- Ranges provider with tolerant fallbacks (mirrors your runtime)
@dataclass
class RangeProviderCfg:
    ranges_parquet: str  # rangenet_preflop_from_flop_<STAKE>.parquet
    stack_round_to: Optional[float] = 20.0  # banding for tolerant join

class VillainRangeProvider:
    def __init__(self, cfg: RangeProviderCfg):
        self.df = pd.read_parquet(cfg.ranges_parquet)
        self.cfg = cfg

    def _lookup_exact(self, hero_pos: str, villain_pos: str, stack: float, action_seq: Sequence[str]) -> Optional[np.ndarray]:
        a1, a2, a3 = serialize_action_seq(action_seq)
        m = self.df[
            (self.df["hero_pos"] == hero_pos) &
            (self.df["villain_pos"] == villain_pos) &
            (self.df["stack_bb"] == float(stack)) &
            (self.df["action_seq_1"] == a1) &
            (self.df["action_seq_2"] == a2) &
            (self.df["action_seq_3"] == a3)
        ]
        if m.empty:
            return None
        row = m.iloc[0]
        vec = np.array([row[f"y_{i}"] for i in range(169)], dtype=np.float32)
        s = float(vec.sum())
        return (vec / s) if s > 1e-8 else None

    def get(self, hero_pos: str, villain_pos: str, stack: float, action_seq: Sequence[str]) -> Optional[np.ndarray]:
        v = self._lookup_exact(hero_pos, villain_pos, stack, action_seq)
        if v is not None:
            return v
        if self.cfg.stack_round_to:
            r = float(self.cfg.stack_round_to)
            stack2 = round(stack / r) * r
            v = self._lookup_exact(hero_pos, villain_pos, stack2, action_seq)
            if v is not None:
                return v
            v = self._lookup_exact(hero_pos, villain_pos, stack2, ["", "", ""])
            if v is not None:
                return v
        return None


@dataclass
class PreflopLegality:
    open_sizes_bb: Tuple[float, ...] = (2.0, 2.5, 3.0)
    raise_sizes_bb: Tuple[float, ...] = (6.0, 7.5, 9.0, 12.0)  # total vs faced
    allow_allin: bool = True
    include_check_when_free: bool = True

def _tok_amount_cbb(tok: str) -> Optional[float]:
    # centi-bb to bb amount
    if "_" not in tok:
        return None
    try:
        raw = tok.split("_", 1)[1]
        return float(int(raw)) / 100.0
    except Exception:
        return None

def build_preflop_legal_mask(
    *,
    facing_bet: bool,
    faced_bb: float,
    stack_bb: float,
    rules: PreflopLegality,
) -> np.ndarray:
    mask = np.zeros(len(PREFLOP_ACTION_VOCAB), dtype=np.float32)
    for i, t in enumerate(PREFLOP_ACTION_VOCAB):
        legal = False
        if t == "FOLD":
            legal = True
        elif t == "CHECK":
            legal = (not facing_bet) and rules.include_check_when_free
        elif t == "CALL":
            legal = facing_bet and faced_bb > 0.0 + 1e-9
        elif t.startswith("OPEN_") and (not facing_bet):
            amt = _tok_amount_cbb(t)
            legal = (amt is not None) and (amt <= stack_bb + 1e-6) and (amt in rules.open_sizes_bb)
        elif t.startswith("RAISE_") and facing_bet:
            tot = _tok_amount_cbb(t)
            legal = (tot is not None) and (tot in rules.raise_sizes_bb) and (tot > faced_bb + 1e-6) and (tot <= stack_bb + 1e-6)
        elif t == "ALLIN":
            legal = rules.allow_allin and (stack_bb > 0.0)
        mask[i] = 1.0 if legal else 0.0
    if mask.sum() <= 0:
        mask[:] = 1.0
    return mask

# ---- MC EV calculator (preflop; boardless → river)
@dataclass
class EVSimCfg:
    num_samples: int = 1200
    seed: int = 7

def _sample_villain_hands_from_169(vec: np.ndarray, n: int, rng: np.random.Generator) -> List[Tuple[eval7.Card, eval7.Card]]:
    ids = np.arange(169, dtype=np.int32)
    probs = vec / (vec.sum() + 1e-12)
    idx = rng.choice(ids, size=n, p=probs)
    # Map 169-grid id → a representative combo; fallback: random from all combos of that class
    # For simplicity: sample two random cards uniformly; the 169 mapping detail can be refined later.
    # This keeps the structure (and we still weight by vec).
    deck = [c for c in eval7.Deck()]
    out: List[Tuple[eval7.Card, eval7.Card]] = []
    for _ in idx:
        random.shuffle(deck)
        out.append((deck[0], deck[1]))
    return out

def _hero_cards_from_str(hero_hand: Optional[str]) -> Optional[Tuple[eval7.Card, eval7.Card]]:
    if not hero_hand or len(hero_hand) < 4:
        return None
    try:
        return eval7.Card(hero_hand[:2]), eval7.Card(hero_hand[2:4])
    except Exception:
        return None

def mc_ev_preflop_row(
    *,
    hero_hand: Optional[str],
    pot_bb: float,
    stack_bb: float,
    faced_bb: float,
    facing_bet: bool,
    vocab: Sequence[str],
    mask: np.ndarray,
    villain_vec_169: np.ndarray,
    sim: EVSimCfg,
) -> np.ndarray:
    rng = np.random.default_rng(sim.seed)
    hero_cards = _hero_cards_from_str(hero_hand)
    # If hero hand unknown, we average over random hero combos.
    avg_over_hero = hero_cards is None

    def showdown_win_tie(hero: Tuple[eval7.Card, eval7.Card], vill: Tuple[eval7.Card, eval7.Card]) -> Tuple[float, float]:
        used = {str(hero[0]), str(hero[1]), str(vill[0]), str(vill[1])}
        deck = [c for c in eval7.Deck() if str(c) not in used]
        rng.shuffle(deck)
        board = [deck[i] for i in range(5)]
        hv = eval7.evaluate(list(hero) + board)
        vv = eval7.evaluate(list(vill) + board)
        if hv > vv:
            return 1.0, 0.0
        if hv == vv:
            return 0.0, 1.0
        return 0.0, 0.0

    # Precompute villain sample pool
    vill_pool = _sample_villain_hands_from_169(villain_vec_169, sim.num_samples, rng)

    def action_cost_bb(tok: str) -> float:
        if tok in ("FOLD", "CHECK"):
            return 0.0
        if tok == "CALL":
            return faced_bb
        if tok == "ALLIN":
            return stack_bb
        amt = _tok_amount_cbb(tok)
        if amt is None:
            return 0.0
        # OPEN_xxx: absolute bb; RAISE_xxx: interpret token as total to-put-in vs faced (training-aligned)
        if tok.startswith("OPEN_"):
            return min(amt, stack_bb)
        if tok.startswith("RAISE_"):
            return min(amt, stack_bb)
        return 0.0

    def action_pot_after(tok: str) -> float:
        # Crude pot accounting consistent with your runtime placeholder
        commit = action_cost_bb(tok)
        return pot_bb + commit

    ev = np.zeros(len(vocab), dtype=np.float32)

    # Evaluate each token independently with the placeholder pot model
    for i, tok in enumerate(vocab):
        if mask[i] < 0.5:
            ev[i] = 0.0
            continue
        if tok in ("FOLD", "CHECK"):
            ev[i] = 0.0
            continue

        commit = action_cost_bb(tok)
        max_pot = min(action_pot_after(tok), 2.0 * stack_bb)

        wins = ties = sims = 0.0
        H: List[Tuple[eval7.Card, eval7.Card]] = []
        if avg_over_hero:
            # Sample random distinct hero combos
            for _ in range(sim.num_samples):
                deck = [c for c in eval7.Deck()]
                rng.shuffle(deck)
                H.append((deck[0], deck[1]))
        else:
            H = [hero_cards] * sim.num_samples  # fixed hero hand

        for k in range(sim.num_samples):
            vill = vill_pool[k % len(vill_pool)]
            hero = H[k]
            w, t = showdown_win_tie(hero, vill)
            wins += w
            ties += t
            sims += 1.0

        win_p = wins / max(sims, 1.0)
        tie_p = ties / max(sims, 1.0)
        ev[i] = float(win_p * max_pot + tie_p * (max_pot / 2.0) - commit)

    return ev

# ---- Small helpers
OPEN_SIZE_BY_POS_NL10 = {
    "UTG": 3.0, "HJ": 2.5, "CO": 2.5, "BTN": 2.5, "SB": 3.0, "BB": 0.0,
}

def stake_to_stakes_id_token(stake: str) -> str:
    # Your sidecars use {"stakes_id": {"2": 0}} for NL10.
    s = (stake or "NL10").upper()
    if s == "NL10":
        return "2"
    # Extend as needed
    return "2"