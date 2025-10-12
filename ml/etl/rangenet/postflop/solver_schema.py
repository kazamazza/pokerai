from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

ActionMix = Dict[str, float]  # maps ACTION_VOCAB token -> prob (must sum≈1.0)

@dataclass
class SolverExtraction:
    # Identifiers/context (copied from manifest or inferred)
    ctx: str
    ip_pos: str
    oop_pos: str
    board: str
    pot_bb: float
    stack_bb: float
    bet_sizing_id: str

    # What we require from the JSON
    root_mix: ActionMix = field(default_factory=dict)     # IP root action mix (CHECK/BET_xx)
    facing_mix: ActionMix = field(default_factory=dict)   # OOP facing action mix (FOLD/CALL/RAISE_xxx)
    facing_bet_bb: Optional[float] = None                # numeric facing bet size at the facing node

    # Diagnostics for auditability
    meta: Dict[str, Any] = field(default_factory=dict)   # e.g., paths taken, raw labels seen, parser branch
    ok: bool = False                                     # did we satisfy minimum contract?
    reason: Optional[str] = None                         # if not ok, why