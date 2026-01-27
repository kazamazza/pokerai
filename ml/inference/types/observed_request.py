# ml/inference/types/observed_request.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Position = str
SeatLabel = Literal["UTG", "MP", "CO", "BTN", "SB", "BB"]
Street = Literal[0, 1, 2, 3]  # 0=pre,1=flop,2=turn,3=river


@dataclass(frozen=True)
class StackChangeEvent:
    tick: int
    when_ms: Optional[int]
    street: Street
    player_id: str
    seat_label: SeatLabel
    stack_before_bb: float
    stack_after_bb: float
    delta_bb: float
    source: Literal["vision", "derived"]
    conf: Optional[float] = None


@dataclass(frozen=True)
class PotChangeEvent:
    tick: int
    when_ms: Optional[int]
    street: Street
    pot_before_bb: float
    pot_after_bb: float
    delta_bb: float
    source: Literal["vision", "derived"]


@dataclass(frozen=True)
class StreetTransition:
    to_street: Literal[1, 2, 3]
    tick: int
    when_ms: Optional[int]
    reason: Literal["card_seen", "timer", "inferred"] = "card_seen"


@dataclass(frozen=True)
class ObservedRequest:
    # --- always available / safe ---
    stakes: str                    # e.g. "NL10"
    street: Street                 # current street
    hero_pos: SeatLabel            # user-selected hero seat label (required)

    # --- optional but user-facing ---
    hero_id: Optional[str] = None
    hero_hand: Optional[str] = None        # "AhKh"
    board: Optional[str] = None            # "Ts5cKd" / "Ts5cKd2h" / "Ts5cKd2h9d"

    # --- numeric state (required for postflop inference/use) ---
    pot_bb: float = 0.0
    eff_stack_bb: float = 100.0

    # --- toggles (user-facing) ---
    allow_allin: Optional[bool] = None

    # --- streams (authoritative observation) ---
    stack_stream: List[StackChangeEvent] = field(default_factory=list)
    pot_stream: List[PotChangeEvent] = field(default_factory=list)
    street_transitions: List[StreetTransition] = field(default_factory=list)

    # --- ids / misc ---
    hand_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    debug: bool = False