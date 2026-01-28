# ml/infer/preflop/types.py (or wherever you keep core types)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Any

# import your enums
from ml.core.types import Ctx, Pos  # adjust import path if different

StreetName = Literal["preflop", "flop", "turn", "river"]


@dataclass(frozen=True)
class PreflopSpot:
    """
    Minimal spot representation for preflop range lookup + policy baseline.

    Keep internal IDs as ints (Ctx/Pos ids) and convert to strings ONLY when
    calling legacy lookup tables.
    """
    street: StreetName
    stack_bb: float

    ip_pos_id: int
    oop_pos_id: int
    ctx_id: int

    # optional metadata
    stake_id: Optional[int] = None
    topology: Optional[str] = None


def _as_ctx_id(ctx: int | str | Ctx) -> int:
    if isinstance(ctx, Ctx):
        return int(ctx.value)
    if isinstance(ctx, int):
        return int(ctx)
    # string -> enum
    return int(Ctx[ctx.upper()].value)


def _as_pos_id(pos: int | str | Pos) -> int:
    if isinstance(pos, Pos):
        return int(pos.value)
    if isinstance(pos, int):
        return int(pos)
    return int(Pos[pos.upper()].value)