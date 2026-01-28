from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple
from ml.core.types import Ctx, Pos  # adjust
from types import PreflopSpot
from normalizer import PreflopSpotNormalizer

CTX_TO_LOOKUP = {
    Ctx.OPEN.value: "SRP",                 # 👈 important: OPEN -> SRP
    Ctx.VS_OPEN.value: "VS_OPEN",
    Ctx.VS_3BET.value: "VS_3BET",
    Ctx.VS_4BET.value: "VS_4BET",
    Ctx.BLIND_VS_STEAL.value: "VS_OPEN",   # or "BvS" if you actually have that key in charts
    Ctx.LIMPED_SINGLE.value: "LIMPED_SINGLE",
    Ctx.LIMPED_MULTI.value: "LIMPED_MULTI",
    # postflop ctxs shouldn't hit preflop engine, but safe fallback:
    Ctx.VS_CBET.value: "VS_CBET",
    Ctx.VS_CBET_TURN.value: "VS_CBET_TURN",
    Ctx.VS_CHECK_RAISE.value: "VS_CHECK_RAISE",
    Ctx.VS_DONK.value: "VS_DONK",
}

def ctx_id_to_lookup(ctx_id: int) -> str:
    return CTX_TO_LOOKUP.get(int(ctx_id), "SRP")


@dataclass
class PreflopPolicyEngine:
    """
    Produces a baseline preflop "range-based policy" by looking up
    (range_ip, range_oop) given spot context.

    Uses ints internally; converts to strings for the existing lookup API.
    """
    lookup: Any  # PreflopRangeLookup
    normalizer: PreflopSpotNormalizer = PreflopSpotNormalizer()

    def _pos_name(self, pos_id: int) -> str:
        return Pos(pos_id).name  # e.g. 3 -> "BTN"

    def _ctx_name(self, ctx_id: int) -> str:
        # Your lookup default is "SRP"; ensure we map enum ids -> names.
        # If your enum name differs from lookup keys, add a small mapping here.
        return Ctx(ctx_id).name

    def ranges_for_state(self, state: Mapping[str, Any], *, strict: bool = False) -> Tuple[str, str, Dict[str, object]]:
        spot = self.normalizer.normalize(state)
        return self.ranges_for_spot(spot, strict=strict)

    def ranges_for_spot(self, spot: PreflopSpot, *, strict: bool = False) -> Tuple[str, str, Dict[str, object]]:
        ip = self._pos_name(spot.ip_pos_id)
        oop = self._pos_name(spot.oop_pos_id)
        ctx = ctx_id_to_lookup(spot.ctx_id)

        # Call your existing lookup (string-based)
        range_ip, range_oop, meta = self.lookup.ranges_for_pair(
            stack_bb=float(spot.stack_bb),
            ip=ip,
            oop=oop,
            ctx=ctx,
            strict=bool(strict),
        )
        return range_ip, range_oop, meta