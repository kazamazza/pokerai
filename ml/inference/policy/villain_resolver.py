from dataclasses import dataclass
from typing import List, Optional, Set, Dict, Any

from ml.inference.exploit.infer_actions import ActionInferrer, ActionRecord

Aggressive = {"BET", "RAISE"}

@dataclass
class VillainPick:
    villain_id: Optional[str]
    villain_pos: Optional[str]          # ← add
    reason: str
    confidence: float
    candidates: List[str]

class VillainResolver:
    def __init__(self) -> None:
        self._inferrer = ActionInferrer()

    def resolve(self, req) -> VillainPick:
        hero_id = getattr(req, "hero_id", None)
        street  = int(getattr(req, "street", 0) or 0)
        facing  = bool(getattr(req, "facing_bet", False))

        events: List[ActionRecord] = self._inferrer.infer(
            req, exclude_hero=True, target_player_id=None
        )

        # map last-known seat_label per player_id for this hand
        last_seat: dict[str, str] = {}
        by_street: dict[int, list[ActionRecord]] = {}
        for e in events:
            by_street.setdefault(int(e.street or 0), []).append(e)
            if getattr(e, "seat_label", None):
                last_seat[e.player_id] = e.seat_label

        current = sorted(by_street.get(street, []), key=lambda x: (x.tick, x.when_ms or 0))

        folded: set[str] = set()
        for s, evs in by_street.items():
            if s > street:
                continue
            for e in evs:
                if e.action == "FOLD":
                    folded.add(e.player_id)

        candidates = {e.player_id for e in events if e.player_id != hero_id} - folded

        def _pick(vid: Optional[str], reason: str, conf: float) -> VillainPick:
            return VillainPick(
                villain_id=vid,
                villain_pos=(last_seat.get(vid) if vid else None),
                reason=reason,
                confidence=conf,
                candidates=sorted(candidates),
            )

        if facing:
            for e in reversed(current):
                if e.action in Aggressive and e.player_id in candidates:
                    return _pick(e.player_id, "facing_bet_last_aggressor", 1.0)

        if len(candidates) == 1:
            vid = next(iter(candidates))
            return _pick(vid, "heads_up_only_opponent", 0.85)

        for s in range(street, -1, -1):
            for e in reversed(sorted(by_street.get(s, []), key=lambda x: (x.tick, x.when_ms or 0))):
                if e.action in Aggressive and e.player_id in candidates:
                    return _pick(e.player_id, "last_street_aggressor", 0.7)

        if current:
            for e in reversed(current):
                if e.player_id in candidates:
                    return _pick(e.player_id, "recent_actor_current_street", 0.55)

        return _pick(None, "no_candidate", 0.0)