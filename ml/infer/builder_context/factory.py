from __future__ import annotations

from dataclasses import replace
from typing import Dict, Set, Tuple, Optional, Literal
from ml.infer.builder_context.types import (
    BuilderContext, Street, NodeType,
    Ctx, Topology, Role, SPRBin,
)
from ml.infer.types.resolved_state import ResolvedState

POS_SET: Set[str] = {"UTG", "HJ", "MP", "CO", "BTN", "SB", "BB"}

# exact (ctx, topo) mapping is already enforced by ResolvedState, but keep a guard
_CTX_TO_TOPO: Dict[str, str] = {
    "VS_OPEN": "SRP",
    "BLIND_VS_STEAL": "BVS",
    "VS_3BET": "3BP",
    "VS_4BET": "4BP",
    "LIMPED_SINGLE": "LIMP",
}

# Scenario legality (this is your YAML baked into code for v1 strictness)
# We validate (ctx, role, ip_pos, oop_pos) against these.
_SCENARIOS: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {
    # VS_OPEN
    ("VS_OPEN", "AGGRESSOR"): {
        ("UTG", "BB"), ("UTG", "SB"),
        ("HJ", "BB"), ("HJ", "SB"),
        ("CO", "BB"), ("CO", "SB"),
        ("BTN", "BB"), ("BTN", "SB"),
    },
    ("VS_OPEN", "CALLER"): {
        ("UTG", "BB"), ("UTG", "SB"),
        ("HJ", "BB"), ("HJ", "SB"),
        ("CO", "BB"), ("CO", "SB"),
        ("BTN", "BB"), ("BTN", "SB"),
    },

    # BVS
    ("BLIND_VS_STEAL", "ANY"): {
        ("BTN", "BB"), ("BTN", "SB"),
        ("CO", "BB"), ("CO", "SB"),
    },

    # 3BP (aggressor only)
    ("VS_3BET", "AGGRESSOR"): {
        ("BTN", "BB"), ("BTN", "SB"),
        ("CO", "BB"), ("CO", "SB"),
        # NOTE: if you later support caller 3bet pots, add ("VS_3BET","CALLER") here
    },

    # 4BP (aggressor only)
    ("VS_4BET", "AGGRESSOR"): {
        ("BTN", "BB"), ("BTN", "SB"),
        ("CO", "BB"), ("CO", "SB"),
    },

    # Limp
    ("LIMPED_SINGLE", "ANY"): {
        ("BB", "SB"),
    },
}

Role = Literal["AGGRESSOR", "CALLER", "ANY"]
Ctx  = Literal["VS_OPEN", "BLIND_VS_STEAL", "VS_3BET", "VS_4BET", "LIMPED_SINGLE"]

def _coerce_role_for_v1(ctx: Ctx, role: Optional[Role]) -> Role:
    # ctx-specific rules (v1)
    if ctx in ("VS_3BET", "VS_4BET"):
        return "AGGRESSOR"   # collapse CALLER -> AGGRESSOR for v1 scenarios
    if ctx in ("BLIND_VS_STEAL", "LIMPED_SINGLE"):
        return "ANY"
    # VS_OPEN must stay explicit
    return role or "ANY"

# bet_sizing_id selection to match your stake config keys exactly.
# IMPORTANT: For VS_3BET/VS_4BET you explicitly have IP and OOP aggressor menus.
# Since ResolvedState.role might be CALLER in those, we hard-fail (v1) unless AGGRESSOR.
def _bet_sizing_id(ctx: str, role: str, ip_pos: str, oop_pos: str, *, last_raiser_seat: Optional[str]) -> str:
    c = ctx.upper()
    r = role.upper()
    ip = ip_pos.upper()
    oop = oop_pos.upper()
    last = (last_raiser_seat or "").upper()

    if c == "VS_OPEN":
        if r == "AGGRESSOR":
            return "srp_hu.PFR_IP"
        if r == "CALLER":
            return "srp_hu.Caller_OOP"
        raise ValueError(f"VS_OPEN: unsupported role={role}")

    if c == "BLIND_VS_STEAL":
        return "bvs.Any"

    if c == "VS_3BET":
        # choose the aggressor tree by who last raised preflop
        if last == ip:
            return "3bet_hu.Aggressor_IP"
        if last == oop:
            return "3bet_hu.Aggressor_OOP"
        # fallback: if last missing, choose by oop blind heuristic
        return "3bet_hu.Aggressor_OOP" if oop in {"SB", "BB"} else "3bet_hu.Aggressor_IP"

    if c == "VS_4BET":
        if last == ip:
            return "4bet_hu.Aggressor_IP"
        if last == oop:
            return "4bet_hu.Aggressor_OOP"
        return "4bet_hu.Aggressor_OOP" if oop in {"SB", "BB"} else "4bet_hu.Aggressor_IP"

    if c == "LIMPED_SINGLE":
        return "limped_single.BB_IP"

    raise ValueError(f"Unknown ctx={ctx}")


def _spr_bin(spr: float) -> SPRBin:
    x = float(spr)
    if x < 2.0:
        return "SPR_0_2"
    if x < 5.0:
        return "SPR_2_5"
    if x < 10.0:
        return "SPR_5_10"
    return "SPR_10_PLUS"


class BuilderContextFactory:
    @staticmethod
    def from_resolved_state(st: ResolvedState) -> BuilderContext:
        # --- hard requirements (no guessing) ---
        if st.street not in (1, 2, 3):
            raise ValueError(f"BuilderContext is postflop-only; got street={st.street}")

        if st.ctx is None or st.topology is None or st.role is None:
            raise ValueError("ResolvedState must have ctx/topology/role inferred before building context.")

        if not st.villain_pos or not st.ip_pos or not st.oop_pos:
            raise ValueError("ResolvedState must have villain_pos + ip_pos/oop_pos before building context.")

        if st.pot_bb <= 0.0 or st.eff_stack_bb <= 0.0:
            raise ValueError("ResolvedState must have pot_bb>0 and eff_stack_bb>0.")

        # seats sanity
        hero_pos = (st.hero_pos or "").upper()
        villain_pos = (st.villain_pos or "").upper()
        ip_pos = (st.ip_pos or "").upper()
        oop_pos = (st.oop_pos or "").upper()

        for p in (hero_pos, villain_pos, ip_pos, oop_pos):
            if p not in POS_SET:
                raise ValueError(f"Illegal position '{p}' in ResolvedState.")

        # derived: node_type + size_frac
        node_type: NodeType = st.node_type  # already Literal["ROOT","FACING"]
        if node_type == "ROOT":
            size_frac = 0.0
        else:
            # facing must have a positive faced fraction
            if st.faced_size_frac <= 0.0:
                raise ValueError("FACING requires faced_size_frac > 0.")
            size_frac = float(st.faced_size_frac)

        # enforce ctx<->topology mapping
        ctx: Ctx = st.ctx
        topo: Topology = st.topology
        expected_topo = _CTX_TO_TOPO.get(ctx)
        if expected_topo is None or expected_topo != topo:
            raise ValueError(f"ctx/topology mismatch: ctx={ctx} topo={topo} expected={expected_topo}")

        role_raw = st.role
        role_use = _coerce_role_for_v1(ctx, role_raw)

        # scenario legality uses role_use (because YAML v1 is role-scoped)
        key = (ctx, role_use)
        legal_pairs = _SCENARIOS.get(key)
        if not legal_pairs:
            raise ValueError(f"Unsupported scenario: ctx={ctx} role={role_use} (raw role was {role_raw})")

        pair = (ip_pos, oop_pos)
        if pair not in legal_pairs:
            raise ValueError(f"Illegal (ip,oop) pair for ctx={ctx} role={role_use}: {pair}")

        last_raiser_seat = None
        if isinstance(st.meta, dict):
            last_raiser_seat = st.meta.get("last_raiser_seat")

        bet_sizing_id = _bet_sizing_id(ctx, role_use, ip_pos, oop_pos, last_raiser_seat=last_raiser_seat)

        # spr
        spr = float(st.eff_stack_bb) / max(float(st.pot_bb), 1e-9)
        spr_bin = _spr_bin(spr)

        street: Street = st.street  # type: ignore

        return BuilderContext(
            stakes=str(st.stakes).upper(),
            hand_id=st.hand_id,

            street=street,
            node_type=node_type,
            size_frac=float(size_frac),

            hero_pos=hero_pos,
            villain_pos=villain_pos,
            ip_pos=ip_pos,
            oop_pos=oop_pos,

            ctx=ctx,
            topology=topo,
            role=role_use,
            bet_sizing_id=bet_sizing_id,

            pot_bb=float(st.pot_bb),
            eff_stack_bb=float(st.eff_stack_bb),
            spr=spr,
            spr_bin=spr_bin,
        )