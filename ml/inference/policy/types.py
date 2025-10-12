from __future__ import annotations

import re
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

    _POSITION_ORDER = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]

    @staticmethod
    def is_hero_ip(hero_pos: str, villain_pos: str) -> bool:
        try:
            h_i = PolicyRequest._POSITION_ORDER.index(hero_pos)
            v_i = PolicyRequest._POSITION_ORDER.index(villain_pos)
            # In position if hero acts later (larger index preflop)
            return h_i > v_i
        except ValueError:
            return True  # default safe

    # --- Validation + legalize logic ---
    def legalize(self) -> "PolicyRequest":
        """
        Normalize and ensure this request is logically valid for poker.
        Auto-deduce IP/OOP and correct street/board consistency.
        Raises ValueError if request cannot be fixed.
        """
        # --- Normalize positions ---
        def norm_pos(p: Optional[str]) -> Optional[str]:
            return str(p).strip().upper() if p else None

        self.hero_pos = norm_pos(self.hero_pos)
        self.villain_pos = norm_pos(self.villain_pos)

        # --- Validate positions ---
        if self.hero_pos not in self._POSITION_ORDER:
            raise ValueError(f"Illegal hero_pos '{self.hero_pos}'")
        if self.villain_pos not in self._POSITION_ORDER:
            raise ValueError(f"Illegal villain_pos '{self.villain_pos}'")

        # --- Deduce in-position/out-of-position from order ---
        hero_i = self._POSITION_ORDER.index(self.hero_pos)
        vill_i = self._POSITION_ORDER.index(self.villain_pos)
        if hero_i < vill_i:
            # hero acts earlier → OOP
            self.oop_pos, self.ip_pos = self.hero_pos, self.villain_pos
        else:
            # hero acts later → IP
            self.ip_pos, self.oop_pos = self.hero_pos, self.villain_pos

        # --- Validate street vs board consistency ---
        if self.street == 0 and self.board:
            raise ValueError("Preflop (street=0) cannot have a board")
        n_cards = len(re.findall(r"[AKQJT98765432][shdc]", self.board or ""))
        expected = {0: 0, 1: 3, 2: 4, 3: 5}.get(self.street, 0)
        if self.street > 0 and n_cards != expected:
            raise ValueError(f"Street {self.street} expects {expected} board cards, got {n_cards}")

        # --- Facing-bet sanity ---
        # Hero cannot face a bet preflop without villain opening first.
        if self.street == 0 and self.facing_bet and hero_i < vill_i:
            raise ValueError("Hero cannot face a bet preflop if hero acts before villain")

        # --- Effective stack/pot sanity ---
        if self.pot_bb <= 0 or self.eff_stack_bb <= 0:
            raise ValueError(f"Invalid pot_bb={self.pot_bb} or eff_stack_bb={self.eff_stack_bb}")

        # --- Clean raw context ---
        self.raw = dict(self.raw or {})
        self.raw.setdefault("ctx", "VS_OPEN")
        self.raw.setdefault("ip_pos", self.ip_pos)
        self.raw.setdefault("oop_pos", self.oop_pos)

        return self

    def validate(self) -> bool:
        """Return True if request is valid; False otherwise."""
        try:
            _ = self.legalize()
            return True
        except Exception:
            return False


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