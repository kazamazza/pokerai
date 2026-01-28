# ml/core/contracts.py
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple


# ---------------------------
# Common types
# ---------------------------

Street = Literal["preflop", "flop", "turn", "river"]
PolicyKind = Literal["root", "facing"]
SignalKind = Literal[
    "policy_postflop",
    "preflop_engine",
    "equity",
    "ev",
    "exploit",
]

# Keep actions as strings so they match your vocab directly:
# e.g. "CHECK", "BET_33", "FOLD", "CALL", "RAISE_TO_300", "ALLIN"
Action = str  # keep as string

@dataclass(frozen=True)
class ActionProb:
    action: Action
    p: float
    value: Optional[float] = None
    note: Optional[str] = None

@dataclass(frozen=True)
class Meta:
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    ckpt_path: Optional[str] = None
    sidecar_path: Optional[str] = None
    stakes_id: Optional[int] = None
    solver_version: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


# ---------------------------
# Request context (what we infer FROM)
# ---------------------------

@dataclass(frozen=True)
class PolicyRequest:
    # minimal “routing” context
    street: int
    stakes_id: int

    # heads-up assumed for now, but keep positions explicit
    ip_pos: str
    oop_pos: str

    node_type: Literal["root", "facing"]

    # game state
    pot_bb: float
    effective_stack_bb: float
    board: Optional[str] = None               # e.g. "AsKd7h" or "As Kd 7h"
    board_cluster_id: Optional[int] = None

    # sizing context
    size_pct: Optional[float] = None          # for root (your manifest feature)
    faced_size_pct: Optional[float] = None    # for facing

    # scenario/context tokens
    ctx: Optional[str] = None                 # e.g. "VS_OPEN", "VS_3BET", "VS_CBET" etc
    topology: Optional[str] = None            # SRP/3BP/4BP etc (optional)
    bet_sizing_id: Optional[str] = None       # your menu ID (good to keep)
    allow_allin: bool = True

    # optional: free-form for debugging / future extension
    debug: Dict[str, Any] = field(default_factory=dict)


# ---------------------------
# Signal bundles (what components OUTPUT)
# ---------------------------

@dataclass(frozen=True)
class SignalBundle:
    kind: str
    street: int
    action_probs: List[ActionProb] = field(default_factory=list)
    scalars: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    meta: Meta = field(default_factory=Meta)

    @staticmethod
    def empty(*, street: int, stakes_id: Optional[int] = None) -> "SignalBundle":
        return SignalBundle(
            kind="empty",
            street=street,
            action_probs=[],
            scalars={},
            confidence=1.0,
            meta=Meta(model_name="runtime", model_version="v1", stakes_id=stakes_id),
        )

    def with_kind(self, kind: str) -> "SignalBundle":
        return replace(self, kind=kind)

    def merge(self, other: "SignalBundle") -> "SignalBundle":
        if self.street != other.street:
            raise ValueError(f"Cannot merge bundles across streets: {self.street} vs {other.street}")

        out_kind = other.kind if self.kind == "empty" else self.kind

        def pick(a, b):
            return b if b is not None else a

        out_meta = Meta(
            model_name=pick(self.meta.model_name, other.meta.model_name),
            model_version=pick(self.meta.model_version, other.meta.model_version),
            ckpt_path=pick(self.meta.ckpt_path, other.meta.ckpt_path),
            sidecar_path=pick(self.meta.sidecar_path, other.meta.sidecar_path),
            stakes_id=pick(self.meta.stakes_id, other.meta.stakes_id),
            solver_version=pick(self.meta.solver_version, other.meta.solver_version),
            extras={**(self.meta.extras or {}), **(other.meta.extras or {})},
        )

        return SignalBundle(
            kind=out_kind,
            street=self.street,
            action_probs=[*self.action_probs, *other.action_probs],
            scalars={**self.scalars, **other.scalars},
            confidence=min(self.confidence, other.confidence),
            meta=out_meta,
        )


# Convenience helpers
def probs_sum_to_1(action_probs: Sequence[ActionProb], eps: float = 1e-4) -> bool:
    s = sum(ap.p for ap in action_probs)
    return abs(s - 1.0) <= eps


def normalize_action_probs(action_probs: Sequence[ActionProb]) -> List[ActionProb]:
    s = sum(max(0.0, ap.p) for ap in action_probs)
    if s <= 0:
        n = len(action_probs)
        if n == 0:
            return []
        uni = 1.0 / n
        return [ActionProb(ap.action, uni, ap.value, ap.note) for ap in action_probs]
    out: List[ActionProb] = []
    for ap in action_probs:
        p = max(0.0, ap.p) / s
        out.append(ActionProb(ap.action, p, ap.value, ap.note))
    return out


# ---------------------------
# Final decision output
# ---------------------------

@dataclass(frozen=True)
class DecisionBundle:
    street: Street
    # final distribution after blending
    action_probs: List[ActionProb]

    recommendation: Optional[Action] = None
    summary: Optional[str] = None

    # raw bundles used + any combiner diagnostics
    inputs: List[SignalBundle] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PolicyResponse:
    street: int  # 0=preflop,1=flop,2=turn,3=river
    node_type: Literal["root","facing"]
    action_probs: List["ActionProb"]                      # sorted desc
    recommendation: Optional[str] = None
    summary: Optional[str] = None
    signals: Optional["SignalBundle"] = None
    debug: Dict[str, Any] = field(default_factory=dict)