from typing import List, Optional
import torch


class AggroBucketChooser:
    """
    Chooses the best raise or bet action from a set of legal buckets.
    Uses both logits (GTO policy) and EVs to determine best raise.
    Includes safety: avoids raise if EV worse than CALL or CHECK.
    Prefers passive action if EVs are very close.
    """

    def __init__(
        self,
        actions: List[str],
        logits: torch.Tensor,
        hero_mask: torch.Tensor,
        evs: Optional[dict] = None,
        ev_threshold: float = 0.1,         # Tie-break: min EV gap to prefer higher EV
        min_raise_ev_delta: float = 0.05,  # Sanity: avoid raise if EV worse than passive
        stability_ev_delta: float = 0.02,  # Soft preference: prefer call/check if EV very close
        prefer_stability: bool = True,     # Enable call preference override
    ):
        self.actions = actions
        self.logits = logits
        self.hero_mask = hero_mask
        self.evs = evs or {}
        self.ev_threshold = ev_threshold
        self.min_raise_ev_delta = min_raise_ev_delta
        self.stability_ev_delta = stability_ev_delta
        self.prefer_stability = prefer_stability

        self.legal_idx = [i for i, m in enumerate(hero_mask) if m > 0.5]
        self.raise_idx = [i for i in self.legal_idx if actions[i].startswith("RAISE_") or actions[i].startswith("BET_")]
        self.passive_idx = [i for i in self.legal_idx if actions[i] in ("CALL", "CHECK")]

        self.best_idx = self._choose()

    def _choose(self) -> Optional[int]:
        if not self.raise_idx:
            return None

        if self.evs:
            raise_evs = [(i, self.evs.get(self.actions[i], float("-inf"))) for i in self.raise_idx]
            raise_evs = [pair for pair in raise_evs if pair[1] != float("-inf")]
            if not raise_evs:
                return None

            raise_evs.sort(key=lambda x: x[1], reverse=True)
            best_ev_idx, best_ev = raise_evs[0]

            # Tie-break with logits
            if len(raise_evs) > 1:
                second_best_ev = raise_evs[1][1]
                delta = best_ev - second_best_ev
                if delta < self.ev_threshold:
                    best_ev_idx = max(
                        [i for i, ev in raise_evs if abs(ev - best_ev) < self.ev_threshold],
                        key=lambda i: float(self.logits[0][i])
                    )
                    best_ev = self.evs.get(self.actions[best_ev_idx], best_ev)

            # Check vs passive action EV
            if self.passive_idx:
                best_passive_ev = max(
                    [self.evs.get(self.actions[i], float("-inf")) for i in self.passive_idx],
                    default=float("-inf")
                )

                ev_gap = best_ev - best_passive_ev
                if ev_gap < self.min_raise_ev_delta:
                    return None  # Not good enough

                # Soft override: prefer passive if EVs are close
                if self.prefer_stability and ev_gap < self.stability_ev_delta:
                    return self.passive_idx[0]  # Prefer CALL or CHECK

            return best_ev_idx

        # Fallback: logits only
        return max(self.raise_idx, key=lambda i: float(self.logits[0][i]))

    def choose_best(self) -> Optional[int]:
        return self.best_idx

    def debug_info(self) -> dict:
        idx = self.choose_best()
        return {
            "legal_raise_actions": [self.actions[i] for i in self.raise_idx],
            "evs": {
                self.actions[i]: round(self.evs[self.actions[i]], 3)
                for i in self.raise_idx if self.actions[i] in self.evs
            } if self.evs else None,
            "best_action": self.actions[idx] if idx is not None else None,
            "ev_threshold": self.ev_threshold,
            "min_raise_ev_delta": self.min_raise_ev_delta,
            "stability_ev_delta": self.stability_ev_delta,
            "prefer_stability": self.prefer_stability,
        }