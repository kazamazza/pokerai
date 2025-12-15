from dataclasses import dataclass
from typing import List, Optional, Set, Dict, Any

from ml.inference.exploit.infer_actions import ActionInferrer, ActionRecord

# Reuse your ActionRecord from ActionInferrer
# action in {"BET","RAISE","CALL","CHECK","FOLD"}

Aggressive = {"BET", "RAISE"}

@dataclass
class VillainPick:
    villain_id: Optional[str]
    reason: str
    confidence: float
    candidates: List[str]

class VillainResolver:
    """
    Picks the 'villain' (primary opponent) for the current decision,
    using only request + inferred action stream.
    """

    def __init__(self) -> None:
        self._inferrer = ActionInferrer()

    def resolve(self, req) -> VillainPick:
        hero_id = getattr(req, "hero_id", None)
        street  = int(getattr(req, "street", 0) or 0)
        facing  = bool(getattr(req, "facing_bet", False))

        # 1) Infer normalized actions from the raw stack/pot streams.
        events: List[ActionRecord] = self._inferrer.infer(
            req, exclude_hero=True, target_player_id=None
        )

        # 2) Partition helpers
        by_street: Dict[int, List[ActionRecord]] = {}
        for e in events:
            by_street.setdefault(int(e.street or 0), []).append(e)

        current: List[ActionRecord] = sorted(by_street.get(street, []), key=lambda x: (x.tick, x.when_ms or 0))

        # Active players = anyone not folded on/before current street
        folded: Set[str] = set()
        for s, evs in by_street.items():
            if s > street:  # ignore future noise
                continue
            for e in evs:
                if e.action == "FOLD":
                    folded.add(e.player_id)

        # Candidates are non-hero, not folded
        candidates: Set[str] = set(e.player_id for e in events if e.player_id != hero_id) - folded

        # 3) Decision rules (ordered)
        # A) If we face a bet/raise now → bettor/raiser is the villain
        if facing:
            for e in reversed(current):
                if e.action in Aggressive:
                    vid = e.player_id
                    if vid in candidates:
                        return VillainPick(vid, "facing_bet_last_aggressor", 1.0, sorted(candidates))

        # B) If exactly one opponent is still active → that opponent
        if len(candidates) == 1:
            vid = next(iter(candidates))
            return VillainPick(vid, "heads_up_only_opponent", 0.85, [vid])

        # C) Last street aggressor (walk back from current → flop/pre) who is still active
        for s in range(street, -1, -1):
            evs = sorted(by_street.get(s, []), key=lambda x: (x.tick, x.when_ms or 0))
            for e in reversed(evs):
                if e.action in Aggressive and e.player_id in candidates:
                    return VillainPick(e.player_id, "last_street_aggressor", 0.7, sorted(candidates))

        # D) Fallback: most recent non-hero actor on current street
        if current:
            for e in reversed(current):
                if e.player_id in candidates:
                    return VillainPick(e.player_id, "recent_actor_current_street", 0.55, sorted(candidates))

        # E) Total fallback: unknown
        return VillainPick(None, "no_candidate", 0.0, sorted(candidates))