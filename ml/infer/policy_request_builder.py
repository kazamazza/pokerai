from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

from ml.infer.types.resolved_state import ResolvedState
from ml.core.contracts import PolicyRequest  # <-- adjust to your actual contracts import

# Keep using your strict validator/router context
from ml.infer.builder_context.factory import BuilderContextFactory  # <-- adjust import path
from ml.infer.builder_context.types import NodeType, Street


# -----------------------------
# Helpers
# -----------------------------
def _stakes_str_to_id(stakes: str) -> int:
    """
    Map "NL10" -> 10 (or whatever your stakes_id convention is).
    If your system uses an enum id (0..N), swap this implementation
    to use your Stakes IntEnum instead.
    """
    s = (stakes or "").upper().strip()
    # common forms: "NL10", "nl10", "NL10_FAST"
    if s.startswith("NL"):
        digits = "".join(ch for ch in s[2:] if ch.isdigit())
        if digits:
            return int(digits)
    raise ValueError(f"Cannot map stakes='{stakes}' to stakes_id.")


def _frac_to_pct(x: float) -> float:
    """
    Accept either fraction (0..1) or pct (0..100).
    Heuristic: if <= 1.5 assume fraction.
    """
    v = float(x)
    if v <= 1.5:
        return v * 100.0
    return v


def _extract_root_size_pct(meta: Dict[str, Any]) -> Optional[float]:
    """
    Root needs a sizing feature in your training setup.
    We support a few keys to avoid brittle coupling.
    """
    if not isinstance(meta, dict):
        return None

    # prefer explicit pct
    for k in ("size_pct", "bet_size_pct", "root_size_pct"):
        if k in meta and meta[k] is not None:
            return float(meta[k])

    # accept fractions
    for k in ("size_frac", "bet_size_frac", "root_size_frac"):
        if k in meta and meta[k] is not None:
            return _frac_to_pct(float(meta[k]))

    return None


# -----------------------------
# Builder
# -----------------------------
@dataclass(frozen=True)
class PolicyRequestBuilder:
    """
    Turns ResolvedState into the canonical PolicyRequest contract.

    Intentionally:
      - Calls BuilderContextFactory for strict validation + derived routing fields
      - Keeps all "guessing" limited to small extraction helpers (e.g. size_pct keys)
    """

    def build(self, st: ResolvedState) -> PolicyRequest:
        # 1) strict validation + derived routing fields (ctx/topo/role/ip/oop/size_frac/spr/bin/etc.)
        ctx = BuilderContextFactory.from_resolved_state(st)

        # 2) stakes id
        stakes_id = _stakes_str_to_id(ctx.stakes)

        # 3) board cluster (optional)
        board_cluster_id = None
        if isinstance(st.meta, dict):
            bc = st.meta.get("board_cluster_id")
            board_cluster_id = int(bc) if bc is not None else None

        # 4) sizing
        size_pct: Optional[float] = None
        faced_size_pct: Optional[float] = None

        if ctx.node_type == "ROOT":
            # Your root model usually expects a size feature.
            size_pct = _extract_root_size_pct(st.meta if isinstance(st.meta, dict) else {})
        else:
            # facing uses the faced size fraction from resolved/builder context
            faced_size_pct = _frac_to_pct(float(ctx.size_frac))

        # 5) street: keep consistent with contracts
        # If your PolicyRequest.street expects a Street enum (preflop/flop/turn/river),
        # ensure ctx.street is that type. If it's int, convert here.
        street = ctx.street
        node_type: Literal["root", "facing"] = (
            "root" if ctx.node_type == "ROOT" else "facing"
        )

        # 6) build request
        return PolicyRequest(
            street=street,
            node_type=node_type,
            stakes_id=stakes_id,
            ip_pos=ctx.ip_pos,
            oop_pos=ctx.oop_pos,
            pot_bb=float(ctx.pot_bb),
            effective_stack_bb=float(ctx.eff_stack_bb),
            board=st.board,
            board_cluster_id=board_cluster_id,
            size_pct=size_pct,
            faced_size_pct=faced_size_pct,
            ctx=str(ctx.ctx) if ctx.ctx is not None else None,
            topology=str(ctx.topology) if ctx.topology is not None else None,
            bet_sizing_id=ctx.bet_sizing_id,
            debug={
                "node_type": ctx.node_type,
                "hero_pos": ctx.hero_pos,
                "villain_pos": ctx.villain_pos,
                "spr": ctx.spr,
                "spr_bin": ctx.spr_bin,
                "confidence": dict(st.confidence or {}),
                "reasons": list(st.reasons or []),
                "meta": dict(st.meta or {}),
            },
        )