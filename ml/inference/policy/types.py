from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
import numpy as np

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

@dataclass
class PolicyRequest:
    street: int = 0
    hero_pos: Optional[str] = None
    villain_pos: Optional[str] = None
    hero_hand: Optional[str] = None
    board: Optional[str] = None
    pot_bb: float = 0.0
    eff_stack_bb: float = 100.0
    facing_bet: bool = False
    villain_id: Optional[str] = None
    actions_hist: Optional[List[str]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyResponse:
    actions: List[str]
    probs: List[float]
    evs: List[float]
    notes: List[str] = field(default_factory=list)
    debug: Optional[Dict[str, Any]] = field(default_factory=dict)

    def top_action(self) -> str:
        """Return the action with the highest probability."""
        if not self.actions or not self.probs:
            return "NONE"
        idx = int(np.argmax(self.probs))
        return self.actions[idx]

    def as_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for logging or JSON output."""
        return {
            "actions": self.actions,
            "probs": self.probs,
            "evs": self.evs,
            "notes": self.notes,
            "debug": self.debug,
            "top_action": self.top_action(),
        }

    def __repr__(self) -> str:
        top = self.top_action()
        top_p = max(self.probs) if self.probs else 0.0
        return f"<PolicyResponse top={top} ({top_p:.2f}), len={len(self.actions)}>"