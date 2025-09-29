from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Any

# ---- Normalized action model ----
ActionKind = Literal["FOLD", "CHECK", "CALL", "BET", "RAISE", "ALLIN"]

@dataclass(frozen=True)
class Action:
    """
    Normalized action. Only one of size_* should be used at a time.
      - size_bb:   absolute bet/raise-to size in big blinds (preflop this is common)
      - size_pct:  percent of pot (postflop common)
      - size_mult: multiplier of the previous bet (e.g., 3.0 = 3x)
    """
    kind: ActionKind
    size_bb: Optional[float] = None
    size_pct: Optional[float] = None
    size_mult: Optional[float] = None

# ---- Normalized request/response ----
@dataclass
class PolicyRequest:
    """
    Minimal, normalized inputs the policy layer needs.
    Add fields as your models require; keep token forms (strings) here.
    """
    # core
    street: int = 0                                 # 0=pre,1=flop,2=turn,3=river
    hero_pos: Optional[str] = None                  # "UTG","HJ","CO","BTN","SB","BB"
    villain_pos: Optional[str] = None
    ctx: Optional[str] = None                       # e.g., "SRP","VS_OPEN","VS_3BET"
    pot_bb: float = 0.0
    eff_stack_bb: float = 100.0
    stack_bb: Optional[float] = None                # fallback if eff_stack_bb missing
    facing_bet: bool = False
    facing_open: bool = False

    # preflop specifics
    opener_pos: Optional[str] = None
    opener_action: Optional[str] = None             # typically "RAISE" or "LIMP"

    # optional cards/board
    hero_hand: Optional[str] = None                 # "AsKs"
    board: Optional[str] = None                     # e.g. "AhKd2c" (optional)

    # optional ids for popnet / routing
    stakes_id: Optional[int] = None
    ctx_id: Optional[int] = None
    hero_pos_id: Optional[int] = None
    villain_pos_id: Optional[int] = None

    # passthrough (seats/actions/etc.) for sub-models that need richer context
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicyResponse:
    actions: List[str]
    probs: List[float]
    evs: List[float]
    debug: Dict[str, Any] = field(default_factory=dict)  # <- Any so you can stuff nested dicts here
    notes: List[str] = field(default_factory=list)