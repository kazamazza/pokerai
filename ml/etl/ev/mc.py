# ml/etl/ev/mc.py

from __future__ import annotations
import random
from typing import List, Sequence, Tuple
import eval7

# EVMC: Monte-Carlo EV baseline (units: bb)
# - Uses hero hand + known board cards
# - Optional 169d villain range vector is accepted (currently fallback → uniform)
# - Consistent token semantics across preflop/postflop

from typing import List, Sequence, Optional, Tuple
import random, numpy as np
import eval7

def _cards_from_str(s: str) -> list[eval7.Card]:
    s = (s or "").replace(" ", "")
    if not s:
        return []
    return [eval7.Card(s[i:i+2]) for i in range(0, len(s), 2)]

class EVMC:
    """
    EV(action) ≈ equity * final_pot_if_called - hero_contribution
    (No fold-equity model — this is just a consistent target generator.)

    Conventions:
      - pot_bb: pot *before* the hero acts (and before a faced bet is added).
      - faced_size_bb:
          * PRE/PF facing-open: cost-to-call (open_bb - posted_bb).
          * POSTFLOP facing-bet: villain bet amount (b = frac * pot_before_bet).
      - posted_bb: chips hero has already posted this street (preflop blinds).
          SB=0.5, BB=1.0, otherwise 0.0. Postflop: 0.0.
    """

    def __init__(self, samples: int = 10000, seed: int = 42):
        self.samples = int(samples)
        self.rng = random.Random(int(seed))
        self.npr = np.random.default_rng(int(seed))

    # ---------- equity engine (conditional on known cards) ----------
    def _sample_villain(self, deck: list[eval7.Card],
                        weights169: Optional[np.ndarray]) -> Tuple[eval7.Card, eval7.Card]:
        # TODO: weight by 169 grid if you wire a mapper. Uniform fallback is fine for now.
        i = self.rng.randrange(len(deck)); c1 = deck.pop(i)
        j = self.rng.randrange(len(deck)); c2 = deck.pop(j)
        return c1, c2

    def _equity(self,
                hero_cards: list[eval7.Card],
                board_cards: list[eval7.Card],
                villain_vec_169: Optional[np.ndarray]) -> Tuple[float, float]:
        wins = ties = 0
        base = list(iter(eval7.Deck()))
        used = set(hero_cards + board_cards)
        deck0 = [c for c in base if c not in used]
        need = 5 - len(board_cards)

        for _ in range(self.samples):
            deck = deck0[:]
            self.rng.shuffle(deck)
            v1, v2 = self._sample_villain(deck, villain_vec_169)
            runout = board_cards[:]
            for _ in range(need):
                runout.append(deck.pop())
            he = eval7.evaluate(hero_cards + runout)
            ve = eval7.evaluate([v1, v2] + runout)
            if he > ve: wins += 1
            elif he == ve: ties += 1

        n = float(self.samples)
        return wins / n, ties / n

    # ---------- helpers ----------
    @staticmethod
    def _posted_blind_bb(hero_pos: Optional[str]) -> float:
        p = (hero_pos or "").upper()
        return 1.0 if p == "BB" else (0.5 if p == "SB" else 0.0)

    @staticmethod
    def _clamp_bet(x: float, stack_bb: float) -> float:
        return max(0.0, min(float(x), float(stack_bb)))

    # ---------- main ----------
    def compute_ev_vector(
        self,
        tokens: Sequence[str],
        *,
        hero_hand: str,
        board: str,
        stack_bb: float,
        pot_bb: float,
        faced_size_bb: float = 0.0,          # see convention above
        villain_vec: Optional[np.ndarray] = None,
        posted_bb: float = 0.0,              # preflop blinds already posted by hero
        hero_pos: Optional[str] = None,      # if you prefer, pass this & omit posted_bb
    ) -> List[float]:

        if posted_bb == 0.0 and hero_pos:
            posted_bb = self._posted_blind_bb(hero_pos)

        hero_cards = _cards_from_str(hero_hand)
        board_cards = _cards_from_str(board)
        p_win, p_tie = self._equity(hero_cards, board_cards, villain_vec)

        def show_ev(hero_bet: float, final_pot_if_called: float) -> float:
            # Expected value at showdown (no FE)
            return (p_win * final_pot_if_called +
                    p_tie * (final_pot_if_called * 0.5) -
                    hero_bet)

        P = float(pot_bb)
        B = float(self._clamp_bet(faced_size_bb, stack_bb))  # faced bet (postflop) or cost-to-call (preflop)
        S = float(stack_bb)
        posted = float(posted_bb)

        evs: List[float] = []

        for raw in tokens:
            tok = str(raw).upper()

            # ----- passives we anchor outside anyway -----
            if tok == "FOLD":
                evs.append(0.0); continue
            if tok == "CHECK":
                evs.append(0.0); continue  # baseline; model/policy add dynamics elsewhere

            # ----- preflop absolute totals (centi-bb) -----
            if tok.startswith("OPEN_") or tok.startswith("RAISE_"):
                # total target size in bb
                try:
                    total_bb = float(tok.split("_", 1)[1]) / 100.0
                except Exception:
                    total_bb = 0.0
                # increment we put in now (subtract what we already posted)
                invest = self._clamp_bet(total_bb - posted, S)
                # final pot if called:
                # unopened:   P + 2*total_bb - posted
                # facing open: P + 2*total_bb - posted   (uses call/raise arithmetic; opener's bet implicit)
                final_pot = P + 2.0 * min(total_bb, S) - posted
                evs.append(show_ev(invest, final_pot))
                continue

            if tok == "CALL":
                # PRE: B = cost-to-call; final pot ≈ P + posted + 2*B
                # POST: B = villain bet; final pot = P + 2*B
                # To keep one rule, use: P + 2*B + posted_adjust, where posted_adjust=posted for preflop, else 0
                posted_adjust = posted if board_cards == [] else 0.0
                invest = self._clamp_bet(B, S)
                final_pot = P + 2.0 * invest + posted_adjust
                evs.append(show_ev(invest, final_pot))
                continue

            if tok == "ALLIN":
                invest = self._clamp_bet(S, S)
                # crude symmetry
                final_pot = P + 2.0 * invest - (posted if board_cards == [] else 0.0)
                evs.append(show_ev(invest, final_pot))
                continue

            # ----- postflop root bets (percent of pot before betting) -----
            if tok.startswith("BET_") or tok.startswith("DONK_"):
                # treat DONK_* same as BET_* numerically; legality is handled via y_mask
                try:
                    pct = float(tok.split("_", 1)[1])
                except Exception:
                    pct = 0.0
                frac = max(0.0, pct / 100.0)
                b = self._clamp_bet(frac * P, S)
                final_pot = P + 2.0 * b
                evs.append(show_ev(b, final_pot))
                continue

            # ----- postflop facing raises (multiples of faced bet B) -----
            if tok.startswith("RAISE_"):
                try:
                    mult = float(tok.split("_", 1)[1]) / 100.0  # e.g., 150 → 1.5x
                except Exception:
                    mult = 0.0
                raise_to = mult * B
                invest = self._clamp_bet(raise_to, S)
                final_pot = P + 2.0 * invest
                evs.append(show_ev(invest, final_pot))
                continue

            # fallback
            evs.append(0.0)

        return evs