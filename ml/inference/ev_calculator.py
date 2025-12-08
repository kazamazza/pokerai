from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np

from ml.features.hands import hand_code_from_id
from ml.inference.villain_range_provider import VillainRangeProvider


@dataclass
class EVSignal:
    evs: Dict[str, float]
    best_ev: Optional[float]
    available: bool

class EVCalculator:
    """
    Monte Carlo EV calculator (approx). Interprets tokens consistently:
      - Root:  BET_33 means 33% of current pot_bb. DONK_33 same as BET_33.
      - Facing: CALL pays faced bet; RAISE_150 raises **to** 1.5× faced bet (invest = call + increment).
      - ALLIN invests min(stack_bb, pot_bb + stack_bb - already_in) (simplified).
    """

    def __init__(self, num_simulations: int = 1000):
        self.num_simulations = int(num_simulations)
        # optional: provider cache if you ever move it inside the calculator
        self._provider_cache: Dict[str, VillainRangeProvider] = {}

    # ---------- sampling helpers ----------
    def _sample_hands_from_vec(self, vec: np.ndarray, n: int) -> List[str]:
        ids = np.arange(vec.shape[0], dtype=int)
        p = vec / max(vec.sum(), 1e-9)
        idx = np.random.choice(ids, size=n, p=p)
        # You already have a combo id→code fn in your codebase
        return [hand_code_from_id(int(i)) for i in idx]  # must exist in your project

    @staticmethod
    def _parse_board(board_str: str) -> List[str]:
        s = (board_str or "").replace(" ", "")
        return [s[i:i+2] for i in range(0, len(s), 2) if i + 2 <= len(s)]

    # ---------- token → amount (bb) ----------
    @staticmethod
    def _faced_bet_bb(req) -> float:
        fs = getattr(req, "faced_size_frac", None)
        pot = float(getattr(req, "pot_bb", 0.0) or 0.0)
        stack = float(getattr(req, "eff_stack_bb", 0.0) or 0.0)
        if fs is None:
            return 0.0
        # Heuristic: if fs <= 1 treat as pot fraction; else treat as absolute bb.
        return float(fs) * pot if float(fs) <= 1.0 else float(fs)

    @staticmethod
    def _bet_amount_root(token: str, pot_bb: float, stack_bb: float) -> float:
        # BET_33 → 0.33 * pot; DONK_33 same; BET_100 → pot; BET_75 → 0.75 * pot
        try:
            kind, amt = token.split("_", 1)
            pct = float(amt)
            frac = (pct / 100.0) if pct > 1.0 else pct
        except Exception:
            return 0.0
        invest = max(0.0, min(stack_bb, frac * pot_bb))
        return invest

    @staticmethod
    def _bet_amount_facing(token: str, faced_bb: float, stack_bb: float) -> float:
        t = token.upper()
        if t == "CALL":
            return max(0.0, min(stack_bb, faced_bb))
        if t == "FOLD":
            return 0.0
        if t == "ALLIN":
            return max(0.0, stack_bb)
        if t.startswith("RAISE_"):
            # RAISE_150: raise-to 1.5x faced; invest = call + (raise_to - faced)
            try:
                mult = float(t.split("_", 1)[1])
            except Exception:
                return 0.0
            raise_to = (mult / 100.0) * faced_bb if mult > 10.0 else mult * faced_bb
            increment = max(0.0, raise_to - faced_bb)
            invest = min(stack_bb, faced_bb + increment)
            return invest
        return 0.0

    # ---------- preflop EV (unchanged except guard) ----------
    def _compute_preflop(self, req, tokens: Optional[List[str]], villain_range_vec: Optional[np.ndarray]) -> EVSignal:
        if not req.hero_hand or not req.eff_stack_bb:
            return EVSignal({}, None, False)
        if villain_range_vec is None:
            return EVSignal({}, None, False)

        import eval7, random
        hero_cards = [eval7.Card(req.hero_hand[:2]), eval7.Card(req.hero_hand[2:])]
        known = {str(c) for c in hero_cards}

        # tokens
        actions = tokens or []
        if not actions:
            actions = ["OPEN_2.5", "OPEN_3.0", "OPEN_4.0"]  # or your generator

        # sample villain combos
        villain_combos = self._sample_hands_from_vec(villain_range_vec, self.num_simulations)

        results: Dict[str, float] = {}
        for action in actions:
            # Very rough preflop EV toy model; keep as-is or replace with your solver outputs
            try:
                hero_bet = float(action.split("_", 1)[1])
            except Exception:
                hero_bet = 0.0

            pot_bb = float(req.pot_bb or 0.0) + hero_bet
            stack_bb = float(req.eff_stack_bb or 100.0)
            max_pot = min(pot_bb, 2 * stack_bb)

            win = tie = 0
            itr = 0
            import random
            for _ in range(self.num_simulations):
                # pick first non-conflicting villain combo
                for code in random.sample(villain_combos, k=len(villain_combos)):
                    v1 = eval7.Card(code[:2]); v2 = eval7.Card(code[2:])
                    if str(v1) in known or str(v2) in known:  # collision
                        continue
                    vill = [v1, v2]; break
                else:
                    continue

                used = {str(c) for c in hero_cards + vill}
                deck = [c for c in eval7.Deck() if str(c) not in used]
                random.shuffle(deck)
                full_board = deck[:5]
                hero_eval = eval7.evaluate(hero_cards + full_board)
                vill_eval = eval7.evaluate(vill + full_board)
                if hero_eval > vill_eval: win += 1
                elif hero_eval == vill_eval: tie += 1
                itr += 1

            if itr == 0:
                results[action] = 0.0
            else:
                wp = win / itr; tp = tie / itr
                results[action] = wp * max_pot + tp * (max_pot / 2.0) - hero_bet

        best_ev = max(results.values()) if results else None
        return EVSignal(results, best_ev, bool(results))

    # ---------- postflop EV (token semantics fixed) ----------
    def compute(self, req, tokens: Optional[List[str]] = None, villain_range: Optional[np.ndarray] = None) -> EVSignal:
        if int(getattr(req, "street", 1) or 1) == 0:
            return self._compute_preflop(req, tokens, villain_range)

        if not req.hero_hand or not req.eff_stack_bb or villain_range is None:
            return EVSignal({}, None, False)

        import eval7, random
        hero = [eval7.Card(req.hero_hand[:2]), eval7.Card(req.hero_hand[2:])]
        board_codes = self._parse_board(getattr(req, "board", "") or "")
        board = [eval7.Card(c) for c in board_codes]
        known = {str(c) for c in hero + board}

        actions = tokens or []
        if not actions:
            if getattr(req, "facing_bet", False):
                rb = getattr(req, "raise_buckets", None) or []
                actions = ["FOLD", "CALL"] + [f"RAISE_{int(b)}" for b in rb]
            else:
                bs = getattr(req, "bet_sizes", None) or []
                actions = ["CHECK"] + [f"BET_{int(x*100) if x <= 1.0 else int(x)}" for x in bs]

        pot_bb = float(getattr(req, "pot_bb", 0.0) or 0.0)
        stack_bb = float(getattr(req, "eff_stack_bb", 100.0) or 100.0)
        faced_bb = self._faced_bet_bb(req)

        # pre-sample villain combos
        villain_combos = self._sample_hands_from_vec(villain_range, self.num_simulations)

        results: Dict[str, float] = {}
        for tok in actions:
            T = tok.upper()
            if T in ("FOLD", "CHECK"):
                results[T] = 0.0
                continue

            if T.startswith("BET_") or T.startswith("DONK_"):
                invest = self._bet_amount_root(T, pot_bb=pot_bb, stack_bb=stack_bb)
            elif T == "CALL" or T.startswith("RAISE_") or T == "ALLIN":
                invest = self._bet_amount_facing(T, faced_bb=faced_bb, stack_bb=stack_bb)
            else:
                invest = 0.0

            total_pot = pot_bb + invest
            max_pot = min(total_pot, 2 * stack_bb)

            win = tie = 0
            itr = 0
            for _ in range(self.num_simulations):
                # pick first non-conflicting villain combo
                for code in random.sample(villain_combos, k=len(villain_combos)):
                    v1 = eval7.Card(code[:2]); v2 = eval7.Card(code[2:])
                    if str(v1) in known or str(v2) in known:
                        continue
                    vill = [v1, v2]; break
                else:
                    continue

                used = {str(c) for c in hero + vill + board}
                deck = [c for c in eval7.Deck() if str(c) not in used]
                random.shuffle(deck)
                full_board = board + deck[:5 - len(board)]
                hero_eval = eval7.evaluate(hero + full_board)
                vill_eval = eval7.evaluate(vill + full_board)
                if hero_eval > vill_eval: win += 1
                elif hero_eval == vill_eval: tie += 1
                itr += 1

            if itr == 0:
                results[T] = 0.0
            else:
                wp = win / itr; tp = tie / itr
                results[T] = wp * max_pot + tp * (max_pot / 2.0) - invest

        best_ev = max(results.values()) if results else None
        return EVSignal(results, best_ev, bool(results))