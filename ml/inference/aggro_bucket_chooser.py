from typing import List, Optional
import torch


class AggroBucketChooser:
    """
    Chooses the best raise or bet action from a set of legal buckets.
    Uses both logits (GTO policy) and EVs to determine best raise.
    """

    def __init__(
        self,
        actions: List[str],
        logits: torch.Tensor,
        hero_mask: torch.Tensor,
        evs: Optional[List[float]] = None,
        ev_threshold: float = 0.1,  # min EV delta to prefer higher EV raise
    ):
        self.actions = actions
        self.logits = logits
        self.hero_mask = hero_mask
        self.evs = evs
        self.ev_threshold = ev_threshold

        self.legal_idx = [i for i, m in enumerate(hero_mask) if m > 0.5]
        self.raise_idx = [i for i in self.legal_idx if actions[i].startswith("RAISE_") or actions[i].startswith("BET_")]

    def choose_best(self) -> Optional[int]:
        """
        Picks best raise/bet bucket:
        - If EVs available: prefer highest EV with meaningful gap
        - Break ties using GTO logits
        - Else fall back to best logit
        """
        if not self.raise_idx:
            return None

        if self.evs:
            # Find EV-max among raises
            raise_evs = [(i, self.evs[self.actions[i]]) for i in self.raise_idx if self.actions[i] in self.evs]
            raise_evs.sort(key=lambda x: x[1], reverse=True)

            best_ev_idx, best_ev = raise_evs[0]
            if len(raise_evs) > 1:
                second_best_ev = raise_evs[1][1]
                delta = best_ev - second_best_ev

                if delta < self.ev_threshold:
                    # Tie-break using logits if EV gap small
                    best_ev_idx = max(
                        [i for i, ev in raise_evs if abs(ev - best_ev) < self.ev_threshold],
                        key=lambda i: float(self.logits[0][i])
                    )
            return best_ev_idx

        # Fallback to highest logit if no EVs
        return max(self.raise_idx, key=lambda i: float(self.logits[0][i]))

    def debug_info(self) -> dict:
        idx = self.choose_best()

        # Convert EVs safely: only include actions that exist in the EV dict
        evs_map = (
            {
                self.actions[i]: round(self.evs[self.actions[i]], 3)
                for i in self.raise_idx
                if self.actions[i] in self.evs
            }
            if self.evs else None
        )

        return {
            "legal_raise_actions": [self.actions[i] for i in self.raise_idx],
            "evs": evs_map,
            "best_action": self.actions[idx] if idx is not None else None,
            "ev_threshold": self.ev_threshold,
        }