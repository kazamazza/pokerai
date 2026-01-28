# ml/infer/preflop/normalizer.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ml.core.types import Ctx, Pos  # adjust
from types import PreflopSpot, _as_ctx_id, _as_pos_id


@dataclass(frozen=True)
class PreflopSpotNormalizer:
    """
    Accepts flexible inputs (ints/strings/enums) and outputs a canonical PreflopSpot.
    """

    def normalize(self, state: Mapping[str, Any]) -> PreflopSpot:
        # required
        stack_bb = float(state["stack_bb"])
        ip_pos_id = _as_pos_id(state["ip_pos_id"] if "ip_pos_id" in state else state["ip_pos"])
        oop_pos_id = _as_pos_id(state["oop_pos_id"] if "oop_pos_id" in state else state["oop_pos"])

        # ctx can come in as ctx_id int or ctx str
        if "ctx_id" in state:
            ctx_id = _as_ctx_id(state["ctx_id"])
        else:
            ctx_id = _as_ctx_id(state.get("ctx", "SRP"))

        return PreflopSpot(
            street="preflop",
            stack_bb=stack_bb,
            ip_pos_id=ip_pos_id,
            oop_pos_id=oop_pos_id,
            ctx_id=ctx_id,
            stake_id=state.get("stake_id"),
            topology=state.get("topology"),
        )