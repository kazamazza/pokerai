from __future__ import annotations
from typing import Literal
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re

Position = str
_POSITION_ORDER_PRE  = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
_POSITION_ORDER_POST = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]

Street = Literal[0, 1, 2, 3]  # 0=pre, 1=flop, 2=turn, 3=river

@dataclass
class StackChangeEvent:
    """Raw stack delta captured by the orchestrator (no semantics)."""
    tick: int                         # monotonic per hand (0,1,2,...)
    when_ms: Optional[int]            # wall clock (optional)
    street: Street
    player_id: str                    # stable id you already use
    seat_label: str                   # e.g. "BTN","SB","BB","UTG","HJ","CO"
    stack_before_bb: float
    stack_after_bb: float
    delta_bb: float                   # after - before (typically negative when they put chips in)
    source: Literal["vision","derived"]
    conf: Optional[float] = None      # confidence from vision, if available

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
    villain_pos: Optional[Position] = None
    hero_hand: Optional[str] = None  # e.g., "AhKh"
    board: Optional[str] = None  # e.g., "Ts5cKd" or None

    pot_bb: float = 0.0
    eff_stack_bb: float = 100.0

    facing_bet: Optional[bool] = None  # ← was bool=False
    faced_size: Optional[float] = None

    # ✅ Postflop defaults: fractions of pot
    bet_sizes: Optional[List[float]] = field(default_factory=lambda: [0.33, 0.66])

    # ✅ Preflop defaults: raise multipliers
    raise_buckets: Optional[List[float]] = field(default_factory=lambda: [1.5, 2.0, 3.0])

    allow_allin: Optional[bool] = False
    villain_id: Optional[str] = None
    stack_stream: List[StackChangeEvent] = field(default_factory=list)
    pot_stream: List[PotChangeEvent] = field(default_factory=list)
    street_transitions: List[StreetTransition] = field(default_factory=list)

    hand_id: Optional[str] = None

    raw: Dict[str, Any] = field(default_factory=dict)
    debug: bool = False

    @staticmethod
    def is_hero_ip(hero_pos: str, villain_pos: str, street: int = 0) -> bool:
        """
        True if hero acts later than villain for the given street.
        Preflop: UTG ... BTN SB BB (open-first order).
        Postflop: SB BB UTG HJ CO BTN (button acts last).
        """
        order = _POSITION_ORDER_POST if int(street) > 0 else _POSITION_ORDER_PRE
        try:
            h_i = order.index((hero_pos or "").upper())
            v_i = order.index((villain_pos or "").upper())
            return h_i > v_i
        except ValueError:
            return True  # safe default

    def _norm_pos(self, p: Optional[str]) -> Optional[str]:
        return str(p).strip().upper() if p else None

    def _count_board_cards(self, s: Optional[str]) -> int:
        return len(re.findall(r"[AKQJT98765432][shdc]", s or ""))

    @staticmethod
    def _order_for_street(street: int) -> list[str]:
        """Preflop vs postflop position order."""
        try:
            s = int(street)
        except Exception:
            s = 0
        return _POSITION_ORDER_POST if s > 0 else _POSITION_ORDER_PRE


    def legalize(self) -> "PolicyRequest":
        """Normalize + validate fields; derive ip/oop using street-aware order."""
        # Normalize positions
        self.hero_pos = self._norm_pos(self.hero_pos)
        self.villain_pos = self._norm_pos(self.villain_pos)
        order = self._order_for_street(int(self.street or 0))

        if self.hero_pos not in order:
            raise ValueError(f"Illegal hero_pos '{self.hero_pos}'")
        if self.villain_pos not in order:
            raise ValueError(f"Illegal villain_pos '{self.villain_pos}'")

        # Street-aware IP/OOP
        h_i = order.index(self.hero_pos)
        v_i = order.index(self.villain_pos)
        if h_i < v_i:
            self.oop_pos, self.ip_pos = self.hero_pos, self.villain_pos  # type: ignore[attr-defined]
        else:
            self.ip_pos, self.oop_pos = self.hero_pos, self.villain_pos  # type: ignore[attr-defined]

        # Board vs street consistency
        expected = {0: 0, 1: 3, 2: 4, 3: 5}.get(int(self.street), 0)
        n_cards = self._count_board_cards(self.board)
        if self.street == 0 and n_cards != 0:
            raise ValueError("Preflop (street=0) cannot have a board")
        if self.street > 0 and n_cards != expected:
            raise ValueError(f"Street {self.street} expects {expected} board cards, got {n_cards}")

        # Basic numeric sanity
        if self.pot_bb <= 0 or self.eff_stack_bb <= 0:
            raise ValueError(f"Invalid pot_bb={self.pot_bb} or eff_stack_bb={self.eff_stack_bb}")

        # Raw passthrough hints for downstream
        self.raw = dict(self.raw or {})
        self.raw.setdefault("ctx", self.ctx or "VS_OPEN")
        self.raw.setdefault("ip_pos", getattr(self, "ip_pos", None))
        self.raw.setdefault("oop_pos", getattr(self, "oop_pos", None))
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
