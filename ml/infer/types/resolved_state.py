from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from ml.infer.types.observed_request import StackChangeEvent, PotChangeEvent, StreetTransition

Street = Literal[0, 1, 2, 3]
NodeType = Literal["ROOT", "FACING"]

# Keep these aligned to your YAML scenarios
Ctx = Literal["VS_OPEN", "BLIND_VS_STEAL", "VS_3BET", "VS_4BET", "LIMPED_SINGLE"]
Topology = Literal["SRP", "BVS", "3BP", "4BP", "LIMP"]
Role = Literal["AGGRESSOR", "CALLER", "ANY"]


@dataclass(frozen=True)
class ResolvedState:
    # ----- raw observed (safe inputs) -----
    stakes: str
    street: Street
    hero_id: str
    hero_pos: str

    hero_hand: Optional[str] = None
    board: Optional[str] = None
    hand_id: Optional[str] = None

    pot_bb: float = 0.0
    eff_stack_bb: float = 100.0

    stack_stream: List[StackChangeEvent] = field(default_factory=list)
    pot_stream: List[PotChangeEvent] = field(default_factory=list)
    street_transitions: List[StreetTransition] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    # ----- inferred / resolved -----
    villain_id: Optional[str] = None
    villain_pos: Optional[str] = None

    ip_pos: Optional[str] = None
    oop_pos: Optional[str] = None

    node_type: NodeType = "ROOT"
    faced_size_frac: float = 0.0  # only meaningful when node_type == FACING

    ctx: Optional[Ctx] = None
    topology: Optional[Topology] = None
    role: Optional[Role] = None

    spr: Optional[float] = None
    spr_bin: Optional[str] = None

    # Confidence + reasons (for debugging + unit tests)
    confidence: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)