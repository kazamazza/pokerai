from __future__ import annotations
from typing import Literal
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re

Position = str
_POSITION_ORDER_PRE  = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
_POSITION_ORDER_POST = ["SB", "BB", "UTG", "MP", "CO", "BTN"]

Street = Literal[0, 1, 2, 3]  # 0=pre, 1=flop, 2=turn, 3=river

@dataclass
class StackChangeEvent:
    tick: int
    when_ms: Optional[int]
    street: Street
    player_id: str
    seat_label: str
    stack_before_bb: float
    stack_after_bb: float
    delta_bb: float
    source: Literal["vision", "derived"]
    conf: Optional[float] = None

@dataclass
class PotChangeEvent:
    """Optional but very handy for inference sanity checks."""
    tick: int
    when_ms: Optional[int]
    street: Street
    pot_before_bb: float
    pot_after_bb: float
    delta_bb: float
    source: Literal["vision","derived"]

@dataclass
class StreetTransition:
    """Marks when we saw the street advance (e.g., new board card)."""
    to_street: Street                 # 1,2,3 when entering flop/turn/river
    tick: int
    when_ms: Optional[int]
    reason: Literal["card_seen","timer","inferred"] = "card_seen"

@dataclass
class PolicyRequest:
    stakes: str = "NL10"
    street: int = 1
    ctx: Optional[str] = None
    hero_id: Optional[str] = None
    hero_pos: Optional[Position] = None
    villain_pos: Optional[Position] = None   # CHANGED: allow None
    hero_hand: Optional[str] = None
    board: Optional[str] = None

    pot_bb: float = 0.0
    eff_stack_bb: float = 100.0

    facing_bet: Optional[bool] = None
    faced_size: Optional[float] = None       # CHANGED: renamed field (formerly faced_size_frac)

    # Postflop defaults: pot fractions
    bet_sizes: Optional[List[float]] = field(default_factory=lambda: [0.33, 0.66])
    # Preflop defaults: raise multipliers
    raise_buckets: Optional[List[float]] = field(default_factory=lambda: [1.5, 2.0, 3.0])

    allow_allin: Optional[bool] = False
    villain_id: Optional[str] = None
    stack_stream: List["StackChangeEvent"] = field(default_factory=list)
    pot_stream: List["PotChangeEvent"] = field(default_factory=list)
    street_transitions: List["StreetTransition"] = field(default_factory=list)

    hand_id: Optional[str] = None

    raw: Dict[str, Any] = field(default_factory=dict)
    debug: bool = False

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def is_hero_ip(hero_pos: Optional[str], villain_pos: Optional[str], street: int = 0) -> bool:
        """
        True if hero acts later than villain for the given street.
        Safe defaults if positions missing.
        """
        order = PolicyRequest._order_for_street(street)
        try:
            h_i = order.index((hero_pos or "").upper())
            v_i = order.index((villain_pos or "").upper())
            return h_i > v_i
        except ValueError:
            return True  # safe default when unknown

    @staticmethod
    def _order_for_street(street: int) -> List[str]:
        try:
            s = int(street)
        except Exception:
            s = 0
        return _POSITION_ORDER_POST if s > 0 else _POSITION_ORDER_PRE

    @staticmethod
    def _norm_pos(p: Optional[str]) -> Optional[str]:
        if p is None:
            return None
        s = str(p).strip().upper()
        if s in {"", "NONE", "NULL"}:
            return None
        return s

    @staticmethod
    def _count_board_cards(s: Optional[str]) -> int:
        return len(re.findall(r"[AKQJT98765432][shdc]", s or ""))

    # --- main normalizer -----------------------------------------------------

    def legalize(
            self,
            *,
            allow_missing_villain: bool = False,  # ← NEW: allow villain_pos to be None/invalid without raising
    ) -> "PolicyRequest":
        """Normalize + validate fields; derive ip/oop only when possible.

        If allow_missing_villain=True, we won't raise for missing/invalid villain_pos.
        """

        # ---- Street ----
        try:
            self.street = int(self.street)
        except Exception:
            self.street = 0
        if self.street < 0 or self.street > 3:
            raise ValueError(f"Illegal street '{self.street}'")

        order = self._order_for_street(self.street)

        # ---- Positions ----
        self.hero_pos = self._norm_pos(self.hero_pos)
        self.villain_pos = self._norm_pos(self.villain_pos)

        if self.hero_pos and self.hero_pos not in order:
            raise ValueError(f"Illegal hero_pos '{self.hero_pos}'")

        if self.villain_pos and self.villain_pos not in order:
            # invalid provided villain seat
            if allow_missing_villain:
                self.villain_pos = None
            else:
                raise ValueError(f"Illegal villain_pos '{self.villain_pos}'")

        if not self.villain_pos and not allow_missing_villain:
            # villain required in strict mode
            raise ValueError("villain_pos is required")

        # ---- IP/OOP (only when both seats known) ----
        if self.hero_pos in order and self.villain_pos in order:
            h_i = order.index(self.hero_pos)
            v_i = order.index(self.villain_pos)
            if h_i < v_i:
                setattr(self, "oop_pos", self.hero_pos)
                setattr(self, "ip_pos", self.villain_pos)
            else:
                setattr(self, "ip_pos", self.hero_pos)
                setattr(self, "oop_pos", self.villain_pos)
        else:
            setattr(self, "ip_pos", None)
            setattr(self, "oop_pos", None)

        # ---- Board vs street (relaxed when board omitted) ----
        n_cards = self._count_board_cards(self.board)
        if self.street == 0:
            if n_cards != 0:
                raise ValueError("Preflop (street=0) cannot have a board")
        else:
            if self.board:
                expected = {1: 3, 2: 4, 3: 5}.get(self.street, 0)
                if n_cards != expected:
                    raise ValueError(f"Street {self.street} expects {expected} board cards, got {n_cards}")

        # ---- Numerics (soft defaults) ----
        try:
            self.pot_bb = float(self.pot_bb or 0.0)
        except Exception:
            self.pot_bb = 0.0
        try:
            self.eff_stack_bb = float(self.eff_stack_bb or 0.0)
        except Exception:
            self.eff_stack_bb = 0.0
        if self.eff_stack_bb <= 0.0:
            self.eff_stack_bb = 100.0

        # ---- Menus to floats ----
        if self.bet_sizes is not None:
            self.bet_sizes = [float(x) for x in self.bet_sizes]
        if self.raise_buckets is not None:
            self.raise_buckets = [float(x) for x in self.raise_buckets]

        # ---- Hero id placeholders → None ----
        if isinstance(self.hero_id, str) and self.hero_id.strip().upper() in {"", "SELECT", "UNKNOWN", "NONE", "NULL"}:
            self.hero_id = None

        # ---- Raw passthrough hints ----
        self.raw = dict(self.raw or {})
        self.raw.setdefault("ctx", self.ctx or "VS_OPEN")
        if getattr(self, "ip_pos", None) is not None:
            self.raw.setdefault("ip_pos", getattr(self, "ip_pos"))
        if getattr(self, "oop_pos", None) is not None:
            self.raw.setdefault("oop_pos", getattr(self, "oop_pos"))

        return self

@dataclass
class PolicyResponse:
    actions: List[str]
    probs: List[float]
    evs: List[float]

    notes: List[str] = field(default_factory=list)
    debug: Optional[Dict[str, Any]] = field(default_factory=dict)
    # Optional extras (for introspection / debugging)
    logits: Optional[List[float]] = None  # pre-softmax, temperature-free, masked only by model (not router)
    mask: Optional[List[float]] = None  # single-side legality mask aligned to `actions` (0/1 floats)

    # (optional) which branch produced this, e.g. "root" or "facing"
    side: Optional[str] = None
    best_action: Optional[str] = None
