from dataclasses import dataclass
from typing import Optional, Dict, List
import eval7
import random

@dataclass
class EVSignal:
    evs: Dict[str, float]
    best_ev: Optional[float]
    available: bool

class EVCalculator:
    def __init__(self, num_simulations: int = 1000):
        self.num_simulations = num_simulations

    @staticmethod
    def _parse_cards(s: Optional[str]) -> List[str]:
        """Convert 'Td9s3c' → ['Td','9s','3c']"""
        if not s:
            return []
        return [s[i:i+2] for i in range(0, len(s), 2)]

    def compute(self, req: "PolicyRequest") -> EVSignal:
        # Validate required fields
        if not req.hero_hand or not req.eff_stack_bb:
            return EVSignal(evs={}, best_ev=None, available=False)

        # Parse hero and board
        hero_hand = req.hero_hand
        hero_cards = [eval7.Card(hero_hand[:2]), eval7.Card(hero_hand[2:])]

        board_tokens = self._parse_cards(req.board)
        board_cards = [eval7.Card(c) for c in board_tokens]

        known = {str(c) for c in hero_cards + board_cards}

        # Build action list from request
        actions = []
        if req.facing_bet:
            # Example: CALL + RAISE_200 / RAISE_300
            actions = ["FOLD", "CALL"] + [f"RAISE_{b}" for b in (req.raise_buckets or [])]
        else:
            # Example: CHECK + BET_33 / BET_66
            actions = ["CHECK"] + [f"BET_{b}" for b in (req.bet_sizes or [])]

        if not actions:
            return EVSignal(evs={}, best_ev=None, available=False)

        pot_size = req.pot_bb
        stack_size = req.eff_stack_bb

        # Dummy villain range (replace later if needed)
        dummy_range = ["AhKd", "QhJs", "KcQc", "Ts9d", "8h7c", "AdQc"]

        results = {}

        for action in actions:
            # Non-betting actions
            if action in ("FOLD", "CHECK"):
                results[action] = 0.0
                continue

            # Determine hero investment
            if action == "CALL":
                hero_bet = (req.faced_size_frac or 0) * stack_size
            else:
                # Extract numeric part from RAISE_300 or BET_66
                try:
                    bet = float(action.split("_")[1])
                except:
                    results[action] = 0.0
                    continue
                hero_bet = min(bet, stack_size)

            total_pot = pot_size + hero_bet
            max_pot = min(total_pot, 2 * stack_size)

            win = tie = 0

            # Monte Carlo simulation
            for _ in range(self.num_simulations):
                random.shuffle(dummy_range)

                # sample villain combo avoiding known cards
                for combo in dummy_range:
                    vill_cards = [eval7.Card(combo[:2]), eval7.Card(combo[2:])]
                    if any(str(c) in known for c in vill_cards):
                        continue
                    break
                else:
                    continue

                used = {str(c) for c in hero_cards + vill_cards + board_cards}
                deck = [c for c in eval7.Deck() if str(c) not in used]
                random.shuffle(deck)

                full_board = board_cards + deck[:5 - len(board_cards)]

                hero_eval = eval7.evaluate(hero_cards + full_board)
                vill_eval = eval7.evaluate(vill_cards + full_board)

                if hero_eval > vill_eval:
                    win += 1
                elif hero_eval == vill_eval:
                    tie += 1

            win_prob = win / self.num_simulations
            tie_prob = tie / self.num_simulations

            ev = win_prob * max_pot + tie_prob * (max_pot / 2) - hero_bet
            results[action] = ev

        best_ev = max(results.values()) if results else None
        return EVSignal(evs=results, best_ev=best_ev, available=bool(results))