from typing import List, Dict, Optional
from dataclasses import dataclass
import eval7
import random

from ml.inference.policy.types import PolicyRequest


@dataclass
class EVSignal:
    evs: Dict[str, float]
    best_ev: Optional[float]
    available: bool

class EVCalculator:
    def __init__(self, num_simulations: int = 1000):
        self.num_simulations = num_simulations

    def compute(self, req: "PolicyRequest") -> EVSignal:
        """
        Computes EV per action based on hero hand, board, pot, and villain range.

        :param req: PolicyRequest object
        :return: EVSignal with per-action EVs, best EV, and availability
        """
        hero_hand = req.hero_hand
        board = list(req.board or "")
        pot_size = req.pot_bb
        stack_size = req.eff_stack_bb
        actions: List[str] = req.raw.get("actions", [])
        bet_sizes: Dict[str, float] = req.raw.get("bet_sizes", {})
        villain_range: List[str] = req.raw.get("villain_range", [])

        if not hero_hand or not villain_range or not actions:
            return EVSignal(evs={}, best_ev=None, available=False)

        hero_cards = [eval7.Card(hero_hand[:2]), eval7.Card(hero_hand[2:])]
        known = set(str(c) for c in hero_cards + [eval7.Card(c) for c in board])

        results = {}

        for action in actions:
            if action == "FOLD":
                results[action] = 0.0
                continue

            bet = bet_sizes.get(action, 0.0)
            hero_bet = min(bet, stack_size)
            total_pot = pot_size + hero_bet
            max_pot = min(total_pot, 2 * stack_size)

            win, tie = 0, 0

            for _ in range(self.num_simulations):
                # Sample villain hand
                random.shuffle(villain_range)
                for combo in villain_range:
                    vill_cards = [eval7.Card(combo[:2]), eval7.Card(combo[2:])]
                    if any(str(c) in known for c in vill_cards):
                        continue
                    break
                else:
                    continue  # no valid villain hand found

                used = set(str(c) for c in hero_cards + vill_cards + [eval7.Card(c) for c in board])
                deck = [c for c in eval7.Deck() if str(c) not in used]
                random.shuffle(deck)
                full_board = [eval7.Card(c) for c in board] + deck[:5 - len(board)]

                hero_eval = eval7.evaluate(hero_cards + full_board)
                vill_eval = eval7.evaluate(vill_cards + full_board)

                if hero_eval > vill_eval:
                    win += 1
                elif hero_eval == vill_eval:
                    tie += 1

            win_prob = win / self.num_simulations
            tie_prob = tie / self.num_simulations

            ev = win_prob * max_pot + tie_prob * (max_pot / 2.0) - hero_bet
            results[action] = ev

        return EVSignal(
            evs=results,
            best_ev=max(results.values()) if results else None,
            available=bool(results),
        )