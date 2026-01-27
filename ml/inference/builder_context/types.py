from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

# keep aligned with ResolvedState + YAML scenarios
Street = Literal[1, 2, 3]
NodeType = Literal["ROOT", "FACING"]

Ctx = Literal["VS_OPEN", "BLIND_VS_STEAL", "VS_3BET", "VS_4BET", "LIMPED_SINGLE"]
Topology = Literal["SRP", "BVS", "3BP", "4BP", "LIMP"]
Role = Literal["AGGRESSOR", "CALLER", "ANY"]

SPRBin = Literal["SPR_0_2", "SPR_2_5", "SPR_5_10", "SPR_10_PLUS"]


@dataclass(frozen=True)
class BuilderContext:
    # identity
    stakes: str
    hand_id: Optional[str]

    # node info
    street: Street
    node_type: NodeType
    size_frac: float  # 0.0 for ROOT else faced_size_frac

    # seats
    hero_pos: str
    villain_pos: str
    ip_pos: str
    oop_pos: str

    # scenario binding
    ctx: Ctx
    topology: Topology
    role: Role
    bet_sizing_id: str  # must match stake bet menu keys

    # numerics
    pot_bb: float
    eff_stack_bb: float
    spr: float
    spr_bin: SPRBin