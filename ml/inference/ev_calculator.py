from dataclasses import dataclass
from typing import Optional, Dict, List
import eval7
import random

from ml.inference.preflop_legal_action_generator import PreflopLegalActionGenerator


DEFAULT_POSITION_MATCHUP_RANGES = {
    # OOP vs IP (e.g. BB vs UTG)
    ("BB", "UTG"): ["AA", "KK", "QQ", "AKs", "AQs", "JJ", "TT", "AKo", "KQs"],
    ("BB", "BTN"): ["AA", "KK", "QQ", "JJ", "TT", "AKs", "AQs", "AJs", "KQs", "AKo"],

    # BTN vs CO
    ("BTN", "CO"): ["AA", "KK", "QQ", "AKs", "AQs", "AJs", "KQs", "JJ", "AKo"],

    # Default fallback
    ("ANY", "ANY"): ["AA", "KK", "QQ", "AKs", "AQs", "JJ", "TT", "AKo"],
}

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

    def _expand_hand_abbreviations(self, hands: list[str]) -> list[str]:
        """
        Converts combos like 'AKs' into representative 4-char combos like 'AsKs', 'AdKd', etc.
        You can keep this simple for now.
        """
        # For simplicity, just expand to one fixed suit combo
        mapping = {
            "s": lambda r1, r2: f"{r1}s{r2}s",  # suited
            "o": lambda r1, r2: f"{r1}s{r2}d",  # offsuit
            "": lambda r1, r2: f"{r1}s{r2}h",  # pair
        }

        out = []
        for h in hands:
            if len(h) == 2:  # pair like AA
                out.append(mapping[""](h[0], h[1]))
            elif len(h) == 3:
                r1, r2, t = h[0], h[1], h[2]
                out.append(mapping[t](r1, r2))
        return out

    def default_villain_range(self, req: "PolicyRequest") -> list[str]:
        hero_pos = req.hero_pos or "ANY"
        vill_pos = req.villain_pos or "ANY"

        # Try exact match first
        key = (vill_pos, hero_pos)  # Villain is IP vs hero
        if key in DEFAULT_POSITION_MATCHUP_RANGES:
            hands = DEFAULT_POSITION_MATCHUP_RANGES[key]
        else:
            hands = DEFAULT_POSITION_MATCHUP_RANGES[("ANY", "ANY")]

        return self._expand_hand_abbreviations(hands)

    def _compute_preflop(self, req: "PolicyRequest", tokens: Optional[list[str]] = None) -> EVSignal:
        if not req.hero_hand or not req.eff_stack_bb:
            return EVSignal(evs={}, best_ev=None, available=False)

        hero_cards = [eval7.Card(req.hero_hand[:2]), eval7.Card(req.hero_hand[2:])]
        known = {str(c) for c in hero_cards}
        board_cards = []  # preflop: empty board

        # Generate legal preflop actions
        action_gen = PreflopLegalActionGenerator()
        actions = tokens or action_gen.generate(
            stack_bb=req.eff_stack_bb,
            facing_bet=req.facing_bet,
            faced_frac=req.faced_size_frac
        )

        # Load default villain range (you can replace this with a smarter positional lookup)
        villain_range = self.default_villain_range(req)

        results = {}
        for action in actions:
            if action == "FOLD":
                results[action] = 0.0
                continue

            # Convert action into investment
            try:
                kind, amt_str = action.split("_", 1)
                hero_bet = float(amt_str)
            except:
                results[action] = 0.0
                continue

            pot = (req.pot_bb or 0) + hero_bet
            max_pot = min(pot, 2 * req.eff_stack_bb)

            win = tie = 0
            for _ in range(self.num_simulations):
                # Sample villain hand
                random.shuffle(villain_range)
                for hand in villain_range:
                    vill_cards = [eval7.Card(hand[:2]), eval7.Card(hand[2:])]
                    if any(str(c) in known for c in vill_cards):
                        continue
                    break
                else:
                    continue

                used = {str(c) for c in hero_cards + vill_cards}
                deck = [c for c in eval7.Deck() if str(c) not in used]
                random.shuffle(deck)
                full_board = deck[:5]

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

    def compute(self, req: "PolicyRequest", tokens: Optional[list[str]] = None) -> EVSignal:
        if req.street == 0:
            return self._compute_preflop(req, tokens)
        # Validate required fields
        if not req.hero_hand or not req.eff_stack_bb:
            return EVSignal(evs={}, best_ev=None, available=False)

        # Parse hero and board
        hero_hand = req.hero_hand
        hero_cards = [eval7.Card(hero_hand[:2]), eval7.Card(hero_hand[2:])]

        board_tokens = self._parse_cards(req.board)
        board_cards = [eval7.Card(c) for c in board_tokens]
        known = {str(c) for c in hero_cards + board_cards}

        # If tokens are provided, use them. Otherwise infer from context
        actions = tokens or []
        if not actions:
            if req.facing_bet:
                actions = ["FOLD", "CALL"] + [f"RAISE_{b}" for b in (req.raise_buckets or [])]
            else:
                actions = ["CHECK"] + [f"BET_{b}" for b in (req.bet_sizes or [])]

        if not actions:
            return EVSignal(evs={}, best_ev=None, available=False)

        pot_size = req.pot_bb or 0
        stack_size = req.eff_stack_bb or 100

        # Dummy villain range (replace with real one later)
        dummy_range = ["AhKd", "QhJs", "KcQc", "Ts9d", "8h7c", "AdQc"]
        results = {}

        for action in actions:
            if action in ("FOLD", "CHECK"):
                results[action] = 0.0
                continue

            # Determine hero investment
            try:
                kind, amt_str = action.split("_", 1)
                amt = float(amt_str)
            except Exception:
                results[action] = 0.0
                continue

            if kind in ("RAISE", "BET", "OPEN"):
                hero_bet = min(amt, stack_size)
            elif kind == "CALL":
                hero_bet = (req.faced_size_frac or 0) * stack_size
            else:
                results[action] = 0.0
                continue

            total_pot = pot_size + hero_bet
            max_pot = min(total_pot, 2 * stack_size)

            win = tie = 0
            for _ in range(self.num_simulations):
                random.shuffle(dummy_range)

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