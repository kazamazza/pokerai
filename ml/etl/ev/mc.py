# ml/etl/ev/mc.py

from __future__ import annotations
import random
from typing import List, Sequence, Tuple
import eval7

class EVMC:
    """
    Simple Monte Carlo baseline:
    EV(action) ≈ p_win * max_pot + p_tie * (max_pot/2) - hero_contribution
    """
    def __init__(self, samples: int = 20000, seed: int = 42):
        self.samples = int(samples)
        self.rng = random.Random(seed)

    def _rand_combo(self, deck: list[eval7.Card]) -> tuple[eval7.Card, eval7.Card]:
        i = self.rng.randrange(len(deck))
        c1 = deck.pop(i)
        j = self.rng.randrange(len(deck))
        c2 = deck.pop(j)
        return c1, c2

    def _pwin_ptie_vs_uniform(self) -> Tuple[float, float]:
        wins = ties = 0
        base = list(eval7.Deck())
        for _ in range(self.samples):
            deck = base[:]
            self.rng.shuffle(deck)
            h1, h2 = self._rand_combo(deck)
            v1, v2 = self._rand_combo(deck)
            board = deck[:5]
            he = eval7.evaluate([h1, h2] + board)
            ve = eval7.evaluate([v1, v2] + board)
            if he > ve:
                wins += 1
            elif he == ve:
                ties += 1
        n = float(self.samples)
        return wins / n, ties / n

    def compute_ev_vector(
            self,
            tokens: Sequence[str],
            *,
            hero_hand: str,
            board: str,
            stack_bb: float,
            pot_bb: float,
            faced_size_bb: float = 0.0,
            villain_vec: Optional[np.ndarray] = None,
    ) -> List[float]:
        p_win, p_tie = self._pwin_ptie_vs_uniform()
        cap = 2.0 * float(stack_bb)
        evs: List[float] = []
        for tok in tokens:
            t = tok.upper()
            hero_bet = 0.0
            if t in ("FOLD", "CHECK"):
                hero_bet = 0.0
            elif t == "CALL":
                hero_bet = float(faced_size_bb)
            elif t.startswith(("OPEN_", "RAISE_")):
                try:
                    centi = float(t.split("_", 1)[1])
                except Exception:
                    centi = 0.0
                hero_bet = centi / 100.0  # <-- centi-bb → bb
            elif t == "ALLIN":
                hero_bet = float(stack_bb)
            total_pot = float(pot_bb) + hero_bet
            max_pot = min(total_pot, cap)
            evs.append(0.0 if (t in ("FOLD", "CHECK") and hero_bet == 0.0)
                       else (p_win * max_pot + p_tie * (max_pot / 2.0) - hero_bet))
        return evs