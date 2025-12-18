# ml/inference/policy/facing_resolver.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Set, Tuple

from ml.inference.exploit.infer_actions import ActionInferrer, ActionRecord

AGGRO = {"BET", "RAISE", "ALLIN"}

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
    """
    Resolves whether hero is currently *facing a bet/raise* on the current street,
    and, if so, the absolute faced size in BB and its fraction of the pot.

    Inputs: PolicyRequest with stack_stream, pot_stream, hero_id, street.
    """

    def __init__(self) -> None:
        self._inferrer = ActionInferrer()

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def _group_by_street(events: List[ActionRecord]) -> Dict[int, List[ActionRecord]]:
        by: Dict[int, List[ActionRecord]] = {}
        for e in events:
            s = int(getattr(e, "street", 0) or 0)
            by.setdefault(s, []).append(e)
        for s in by:
            by[s].sort(key=lambda x: (x.tick, x.when_ms or 0))
        return by

    @staticmethod
    def _pot_before_at_tick(req, street: int, tick: int) -> Optional[float]:
        """
        Best-effort pot-before lookup from pot_stream on same street.
        - Prefer PotChangeEvent at same tick → pot_before_bb
        - Else last event BEFORE tick → pot_after_bb of that event
        - Else fall back to req.pot_bb (may be slightly stale)
        """
        ps = [p for p in getattr(req, "pot_stream", []) if int(p.street or 0) == street]
        if not ps:
            return float(getattr(req, "pot_bb", 0.0) or 0.0)
        # exact tick
        same = [p for p in ps if int(p.tick) == int(tick)]
        if same:
            return float(getattr(same[-1], "pot_before_bb", None) or getattr(req, "pot_bb", 0.0) or 0.0)
        # last prior
        prior = [p for p in ps if int(p.tick) < int(tick)]
        if prior:
            return float(getattr(prior[-1], "pot_after_bb", None) or getattr(req, "pot_bb", 0.0) or 0.0)
        return float(getattr(req, "pot_bb", 0.0) or 0.0)

    @staticmethod
    def _amount_from_event(req, ev: ActionRecord) -> Optional[float]:
        """
        Prefer ev.amount_bb if provided by the inferrer.
        Fallback: use the aggressor's stack delta at the same tick on this street (negative → money in).
        """
        amt = getattr(ev, "amount_bb", None)
        if amt is not None:
            try:
                v = float(amt)
                if v > 0:
                    return v
            except Exception:
                pass

        # fallback via stack_stream delta at same tick
        ss = getattr(req, "stack_stream", None) or []
        for sc in ss:
            try:
                if (int(sc.street or 0) == int(ev.street or 0)
                    and int(sc.tick) == int(ev.tick)
                    and sc.player_id == ev.player_id):
                    if sc.delta_bb is not None:
                        v = -float(sc.delta_bb)  # delta is after - before; negative when chips go in
                        if v > 0:
                            return v
            except Exception:
                continue

        return None

    # ---- public -------------------------------------------------------------

    def resolve(self, req, *, hero_is_ip: Optional[bool] = None) -> FacingPick:
        hero_id = getattr(req, "hero_id", None)
        street = int(getattr(req, "street", 0) or 0)

        # 1) Infer normalized actions on the current hand (excluding hero by default)
        events: List[ActionRecord] = self._inferrer.infer(
            req, exclude_hero=False, target_player_id=None
        )
        by_street = self._group_by_street(events)
        cur: List[ActionRecord] = by_street.get(street, [])

        # Active (not folded up to current street)
        folded: Set[str] = set()
        for s, evs in by_street.items():
            if s > street:
                continue
            for e in evs:
                if e.action == "FOLD":
                    folded.add(e.player_id)

        # 2) Find the last non-hero aggressive action on this street
        last_agg: Optional[ActionRecord] = None
        for e in reversed(cur):
            if e.player_id == hero_id:
                # once hero has acted *after* a previous aggressor, we aren't facing anymore
                if last_agg is not None and (e.tick, e.when_ms or 0) > (last_agg.tick, last_agg.when_ms or 0):
                    return FacingPick(
                        facing_bet=False, faced_size_bb=None, size_frac=None,
                        aggressor_id=None, aggressor_seat=None, tick=None,
                        reason="hero_already_acted_after_aggressor", confidence=1.0,
                        debug={"street": street}
                    )
                continue
            if e.action in AGGRO and e.player_id not in folded:
                last_agg = e
                break

        if last_agg is None:
            # No aggressor on current street → not facing
            return FacingPick(
                facing_bet=False, faced_size_bb=None, size_frac=None,
                aggressor_id=None, aggressor_seat=None, tick=None,
                reason="no_aggressor_current_street", confidence=0.8,
                debug={"street": street, "count_cur_events": len(cur)}
            )

        # 3) Ensure hero has NOT acted after the aggressor
        for e in reversed(cur):
            if (e.tick, e.when_ms or 0) <= (last_agg.tick, last_agg.when_ms or 0):
                break
            if e.player_id == hero_id:
                return FacingPick(
                    facing_bet=False, faced_size_bb=None, size_frac=None,
                    aggressor_id=None, aggressor_seat=None, tick=None,
                    reason="hero_acted_after_aggressor", confidence=1.0,
                    debug={"street": street, "aggressor": last_agg.player_id}
                )

        # 4) Compute faced size (BB) and fraction of pot at that moment
        amt_bb = self._amount_from_event(req, last_agg)
        pot_before = self._pot_before_at_tick(req, street, last_agg.tick)
        size_frac = (float(amt_bb) / float(pot_before)) if (amt_bb is not None and pot_before > 1e-9) else None

        return FacingPick(
            facing_bet=True,
            faced_size_bb=(float(amt_bb) if amt_bb is not None else None),
            size_frac=(float(size_frac) if size_frac is not None else None),
            aggressor_id=last_agg.player_id,
            aggressor_seat=getattr(last_agg, "seat_label", None),
            tick=int(last_agg.tick),
            reason="last_non_hero_aggressor_pending_hero_response",
            confidence=1.0 if amt_bb is not None else 0.85,
            debug={
                "street": street,
                "pot_before_bb": pot_before,
                "event": {
                    "action": last_agg.action,
                    "amount_bb": getattr(last_agg, "amount_bb", None),
                    "prior_bet_bb": getattr(last_agg, "prior_bet_bb", None),
                    "tick": last_agg.tick,
                },
            },
        )