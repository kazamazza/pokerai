import math
from typing import List, Optional
from ml.inference.policy.types import ActionHistoryEntry, PolicyRequest


class ActionContextClassifier:
    """
    Analyzes villain's action history to derive exploit-relevant traits
    such as fold tendency, raise aggression, and exploitability score.
    """

    def __init__(self, history: Optional[List[ActionHistoryEntry]] = None):
        self.history = history or []
        self._villain_actions = [h for h in self.history if h.actor == "villain"]

    @staticmethod
    def from_request(req: PolicyRequest, side: str) -> "ActionContextClassifier":
        """
        Factory to create ActionContextClassifier from PolicyRequest.
        Uses villain action history directly (already filtered upstream).
        """
        hist = getattr(req, "actions_hist", []) or []
        if not hist:
            return ActionContextClassifier([])

        postflop_ints = {1, 2, 3}  # Flop, Turn, River
        relevant = [a for a in hist if a.street in postflop_ints]
        return ActionContextClassifier(relevant)

    def fold_tendency(self) -> float:
        """
        Proportion of villain actions that were folds postflop.
        """
        decisions = [a.action for a in self._villain_actions]
        total = len(decisions)
        if total == 0:
            return 0.0
        folds = sum(1 for a in decisions if a.upper() == "FOLD")
        return folds / total

    def fold_bias_score(self, midpoint: float = 0.5, sharpness: float = 4.0) -> float:
        """
        Smooth scoring function:
        - Returns ~ +1.0 for tight players (fold > midpoint)
        - Returns ~ -1.0 for sticky players (fold < midpoint)
        - Returns  0.0 when near midpoint
        """
        f = self.fold_tendency()
        return math.tanh((f - midpoint) * sharpness)

    def raise_response_aggression(self) -> float:
        """
        Aggression when facing a bet — postflop only.
        """
        responses = [
            a.action.upper()
            for a in self._villain_actions
            if a.facing_bet and a.street in ("FLOP", "TURN", "RIVER")
        ]
        total = len(responses)
        if total == 0:
            return 0.0
        raises = sum(1 for a in responses if a.startswith("RAISE"))
        return raises / total

    def stab_opportunity_rate(self) -> float:
        """
        How often villain stabs when checked to (facing_check=True) on flop/turn.
        """
        stab_opps = [
            a for a in self._villain_actions
            if a.facing_check and a.street in ("FLOP", "TURN")
        ]
        total = len(stab_opps)
        if total == 0:
            return 0.0
        bets = sum(1 for a in stab_opps if a.action.upper().startswith("BET"))
        return bets / total

    def is_exploitable(self, fold_threshold: float = 0.7, min_count: int = 3) -> bool:
        """
        Binary exploitable classification based on fold ratio and sample size.
        """
        fold_ratio = self.fold_tendency()
        count = len(self._villain_actions)
        return count >= min_count and fold_ratio >= fold_threshold

    def summary(self) -> dict:
        return {
            "fold_tendency": round(self.fold_tendency(), 2),
            "fold_bias_score": round(self.fold_bias_score(), 2),
            "raise_response_aggression": round(self.raise_response_aggression(), 2),
            "stab_opportunity_rate": round(self.stab_opportunity_rate(), 2),
            "actions_seen": len(self._villain_actions),
        }