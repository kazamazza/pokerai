# file: ml/inference/policy/facing_resolver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ml.inference.exploit.infer_actions import ActionInferrer

# Treat ALLIN as aggressive (important for shoves)
AGGRO: Set[str] = {"BET", "RAISE", "ALLIN"}

# ActionInferrer is monkeypatched in tests; keep a soft reference
# from ml.inference.exploit.infer_actions import ActionInferrer  # real import in production


@dataclass
class FacingPick:
    facing_bet: bool
    faced_size_bb: Optional[float]
    size_frac: Optional[float]
    aggressor_id: Optional[str]
    aggressor_seat: Optional[str]
    tick: Optional[int]
    reason: str
    confidence: float
    debug: Dict[str, object]


class FacingBetResolver:
    """Why: centralizes 'am I facing a bet right now?' logic; removes UI burden."""

    def __init__(self) -> None:
        self._inferrer = ActionInferrer()  # monkeypatched in tests

    # ---------- helpers (pure) ----------

    @staticmethod
    def _group_by_street(events: List[Any]) -> Dict[int, List[Any]]:
        by: Dict[int, List[Any]] = {}
        for e in events:
            s = int(getattr(e, "street", 0) or 0)
            by.setdefault(s, []).append(e)
        for s in by:
            by[s].sort(key=lambda x: (int(getattr(x, "tick", 0)), int(getattr(x, "when_ms", 0) or 0)))
        return by

    @staticmethod
    def _pot_before_at_tick(req: Any, street: int, tick: int) -> float:
        """Why: robust fraction calc even with sparse pot stream."""
        ps = [p for p in getattr(req, "pot_stream", []) if int(getattr(p, "street", 0) or 0) == int(street)]
        if not ps:
            return float(getattr(req, "pot_bb", 0.0) or 0.0)

        # exact tick -> pot_before_bb
        same = [p for p in ps if int(getattr(p, "tick", -1)) == int(tick)]
        if same:
            v = getattr(same[-1], "pot_before_bb", None)
            return float(v) if v is not None else float(getattr(req, "pot_bb", 0.0) or 0.0)

        # last prior -> pot_after_bb
        prior = [p for p in ps if int(getattr(p, "tick", -1)) < int(tick)]
        if prior:
            v = getattr(prior[-1], "pot_after_bb", None)
            return float(v) if v is not None else float(getattr(req, "pot_bb", 0.0) or 0.0)

        return float(getattr(req, "pot_bb", 0.0) or 0.0)

    @staticmethod
    def _amount_from_event(req: Any, ev: Any) -> Optional[float]:
        """Why: tolerate missing amount_bb by falling back to raw stack deltas."""
        amt = getattr(ev, "amount_bb", None)
        if amt is not None:
            try:
                v = float(amt)
                if v > 0:
                    return v
            except Exception:
                pass

        # fallback: stack delta at same tick for same player/street
        for sc in getattr(req, "stack_stream", []) or []:
            try:
                if (
                    int(getattr(sc, "street", 0) or 0) == int(getattr(ev, "street", 0) or 0)
                    and int(getattr(sc, "tick", -1)) == int(getattr(ev, "tick", -1))
                    and getattr(sc, "player_id", None) == getattr(ev, "player_id", None)
                ):
                    d = getattr(sc, "delta_bb", None)
                    if d is not None:
                        v = -float(d)  # delta is after-before; negative means money in
                        if v > 0:
                            return v
            except Exception:
                continue
        return None

    @staticmethod
    def _seat_from_stacks(req: Any, player_id: str, *, street: int, up_to_tick: int) -> Optional[str]:
        """Why: maintain aggressor_seat when ActionRecord lacks seat_label."""
        best: Tuple[int, int] = (-1, -1)
        seat: Optional[str] = None
        for sc in getattr(req, "stack_stream", []) or []:
            try:
                if getattr(sc, "player_id", None) != player_id:
                    continue
                if int(getattr(sc, "street", 0) or 0) != int(street):
                    continue
                tk = int(getattr(sc, "tick", -1))
                if tk > up_to_tick:
                    continue
                wm = int(getattr(sc, "when_ms", 0) or 0)
                s = getattr(sc, "seat_label", None)
                if s and (tk, wm) > best:
                    best = (tk, wm)
                    seat = s
            except Exception:
                continue
        return seat

    @staticmethod
    def _hero_acted_after(cur_events: List[Any], hero_id: Optional[str], aggr_tick: int, aggr_ms: int) -> bool:
        """Why: once hero acts after aggressor, hero is no longer facing."""
        if not hero_id:
            return False
        for e in reversed(cur_events):
            et = int(getattr(e, "tick", 0))
            em = int(getattr(e, "when_ms", 0) or 0)
            if (et, em) <= (aggr_tick, aggr_ms):
                break
            if getattr(e, "player_id", None) == hero_id:
                return True
        return False

    # ---------- public ----------

    def resolve(self, req: Any, *, hero_is_ip: Optional[bool] = None) -> FacingPick:
        hero_id = getattr(req, "hero_id", None)
        street = int(getattr(req, "street", 0) or 0)

        events: List[Any] = self._inferrer.infer(req, exclude_hero=False, target_player_id=None)
        by_street = self._group_by_street(events)
        cur: List[Any] = by_street.get(street, [])

        # active candidates (not folded up to current street)
        folded: Set[str] = set()
        for s, evs in by_street.items():
            if int(s) > street:
                continue
            for e in evs:
                if getattr(e, "action", None) == "FOLD":
                    folded.add(getattr(e, "player_id", None))

        # last non-hero aggressor on current street
        last_agg: Optional[Any] = None
        for e in reversed(cur):
            pid = getattr(e, "player_id", None)
            if pid == hero_id:
                continue
            if getattr(e, "action", None) in AGGRO and pid not in folded:
                last_agg = e
                break

        if last_agg is None:
            return FacingPick(
                facing_bet=False,
                faced_size_bb=None,
                size_frac=None,
                aggressor_id=None,
                aggressor_seat=None,
                tick=None,
                reason="no_aggressor_current_street",
                confidence=0.8,
                debug={"street": street, "count_cur_events": len(cur)},
            )

        agt_tick = int(getattr(last_agg, "tick", 0))
        agt_ms = int(getattr(last_agg, "when_ms", 0) or 0)

        # hero already acted after aggressor → not facing
        if self._hero_acted_after(cur, hero_id, agt_tick, agt_ms):
            return FacingPick(
                facing_bet=False,
                faced_size_bb=None,
                size_frac=None,
                aggressor_id=None,
                aggressor_seat=None,
                tick=None,
                reason="hero_acted_after_aggressor",
                confidence=1.0,
                debug={"street": street, "aggressor": getattr(last_agg, "player_id", None)},
            )

        amt_bb = self._amount_from_event(req, last_agg)
        pot_before = self._pot_before_at_tick(req, street, agt_tick)
        size_frac = (float(amt_bb) / float(pot_before)) if (amt_bb is not None and pot_before > 1e-9) else None

        seat = getattr(last_agg, "seat_label", None) or self._seat_from_stacks(
            req, getattr(last_agg, "player_id", None), street=street, up_to_tick=agt_tick
        )

        return FacingPick(
            facing_bet=True,
            faced_size_bb=(float(amt_bb) if amt_bb is not None else None),
            size_frac=(float(size_frac) if size_frac is not None else None),
            aggressor_id=getattr(last_agg, "player_id", None),
            aggressor_seat=seat,
            tick=agt_tick,
            reason="last_non_hero_aggressor_pending_hero_response",
            confidence=1.0 if amt_bb is not None else 0.85,
            debug={
                "street": street,
                "pot_before_bb": pot_before,
                "event": {
                    "action": getattr(last_agg, "action", None),
                    "amount_bb": getattr(last_agg, "amount_bb", None),
                    "prior_bet_bb": getattr(last_agg, "prior_bet_bb", None),
                    "tick": agt_tick,
                },
            },
        )