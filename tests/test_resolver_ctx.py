# tests/test_resolved_state_resolver.py
# ==========================================================
# ResolvedStateResolver test suite (v1)
# Focus: ctx/topology/role/villain + ip/oop + node_type/faced_size_frac
# ==========================================================

import pytest
from typing import List, Optional

from ml.inference.policy.types import PolicyRequest, StackChangeEvent, PotChangeEvent, StreetTransition
from ml.inference.resolve.resolver import ResolvedStateResolver


# -------------------------
# Helpers
# -------------------------
def e(
    tick: int,
    street: int,
    pid: str,
    seat: str,
    delta_bb: float,
    before: float = 100.0,
) -> StackChangeEvent:
    after = before + float(delta_bb)
    return StackChangeEvent(
        tick=tick,
        when_ms=None,
        street=street,
        player_id=pid,
        seat_label=seat,
        stack_before_bb=before,
        stack_after_bb=after,
        delta_bb=float(delta_bb),
        source="derived",
        conf=1.0,
    )


def p(
    tick: int,
    street: int,
    pot_before: float,
    pot_after: float,
) -> PotChangeEvent:
    return PotChangeEvent(
        tick=tick,
        when_ms=None,
        street=street,
        pot_before_bb=float(pot_before),
        pot_after_bb=float(pot_after),
        delta_bb=float(pot_after - pot_before),
        source="derived",
    )


def mk_req(
    *,
    street: int,
    hero_id: str,
    hero_pos: str,
    stack_stream: List[StackChangeEvent],
    pot_stream: Optional[List[PotChangeEvent]] = None,
    board: Optional[str] = None,
    pot_bb: float = 6.0,
    eff_stack_bb: float = 100.0,
    villain_pos=None,  # explicitly unused; resolver infers
) -> PolicyRequest:
    # Board defaults per street
    if board is None:
        if street == 1:
            board = "Ts5cKd"
        elif street == 2:
            board = "Ts5cKd2h"
        elif street == 3:
            board = "Ts5cKd2h9d"
        else:
            board = None

    return PolicyRequest(
        stakes="NL10",
        street=street,
        hero_id=hero_id,
        hero_pos=hero_pos,
        villain_pos=villain_pos,
        board=board,
        pot_bb=pot_bb,
        eff_stack_bb=eff_stack_bb,
        stack_stream=stack_stream,
        pot_stream=pot_stream or [],
        street_transitions=[],
    )


# ==========================================================
# A) Preflop-world inference: ctx/topology/role/villain
# ==========================================================

@pytest.mark.parametrize(
    "name, pre_stream, exp_ctx, exp_topo, exp_role, exp_villain_pos",
    [
        # Limped single: SB completes, BB checks (0). Two players still present.
        (
            "limped_single",
            [e(0, 0, "h", "SB", -0.5), e(1, 0, "v", "BB", 0.0)],
            "LIMPED_SINGLE",
            "LIMP",
            "ANY",
            "BB",
        ),
        # VS_OPEN: UTG opens, BB calls.
        (
            "vs_open_utg_bb_hero_is_opener",
            [e(0, 0, "h", "UTG", -2.5), e(1, 0, "v", "BB", -2.5)],
            "VS_OPEN",
            "SRP",
            "AGGRESSOR",
            "BB",
        ),
        # VS_OPEN: UTG opens, BB calls, hero is caller.
        (
            "vs_open_utg_bb_hero_is_caller",
            [e(0, 0, "v", "UTG", -2.5), e(1, 0, "h", "BB", -2.5)],
            "VS_OPEN",
            "SRP",
            "CALLER",
            "UTG",
        ),
        # BVS: BTN opens, BB calls.
        (
            "bvs_btn_bb",
            [e(0, 0, "h", "BTN", -2.5), e(1, 0, "v", "BB", -2.5)],
            "BLIND_VS_STEAL",
            "BVS",
            "ANY",
            "BB",
        ),
        # VS_3BET: BTN open, BB 3b, BTN calls.
        (
            "vs_3bet_btn_bb",
            [e(0, 0, "h", "BTN", -2.5), e(1, 0, "v", "BB", -9.0), e(2, 0, "h", "BTN", -6.5)],
            "VS_3BET",
            "3BP",
            "CALLER",  # last raiser is villain
            "BB",
        ),
        # VS_4BET: BTN open, BB 3b, BTN 4b, BB calls.
        (
            "vs_4bet_btn_bb",
            [e(0, 0, "h", "BTN", -2.5), e(1, 0, "v", "BB", -9.0), e(2, 0, "h", "BTN", -21.5), e(3, 0, "v", "BB", -15.0)],
            "VS_4BET",
            "4BP",
            "AGGRESSOR",  # last raiser is hero (the 4bettor)
            "BB",
        ),
    ],
)
def test_preflop_world_inference(name, pre_stream, exp_ctx, exp_topo, exp_role, exp_villain_pos):
    req = mk_req(
        street=1,
        hero_id="h",
        hero_pos=next(x.seat_label for x in pre_stream if x.player_id == "h"),
        stack_stream=pre_stream,
    )
    st = ResolvedStateResolver().resolve(req)

    assert st.ctx == exp_ctx
    assert st.topology == exp_topo
    assert st.role == exp_role
    assert st.villain_id == "v"
    assert st.villain_pos == exp_villain_pos
    assert "ctx_from_preflop_stream" in st.reasons
    assert st.confidence.get("ctx", 0.0) > 0.0


def test_preflop_missing_stream_raises():
    req = mk_req(street=1, hero_id="h", hero_pos="BTN", stack_stream=[])
    with pytest.raises(ValueError):
        ResolvedStateResolver().resolve(req)


def test_preflop_not_hu_raises_three_players():
    pre = [
        e(0, 0, "h", "BTN", -2.5),
        e(1, 0, "v", "SB", -2.5),
        e(2, 0, "x", "BB", -2.5),
    ]
    req = mk_req(street=1, hero_id="h", hero_pos="BTN", stack_stream=pre)
    with pytest.raises(ValueError):
        ResolvedStateResolver().resolve(req)


def test_preflop_hero_not_in_stream_raises():
    pre = [e(0, 0, "v", "BTN", -2.5), e(1, 0, "x", "BB", -2.5)]
    req = mk_req(street=1, hero_id="h", hero_pos="SB", stack_stream=pre)
    with pytest.raises(ValueError):
        ResolvedStateResolver().resolve(req)


# ==========================================================
# B) IP/OOP inference (requires villain_pos)
# ==========================================================

@pytest.mark.parametrize(
    "street, hero_pos, villain_pos, exp_ip, exp_oop",
    [
        # Postflop order: SB,BB,UTG,MP,CO,BTN
        (1, "BTN", "BB", "BTN", "BB"),
        (1, "BB", "SB", "BB", "SB"),
        (2, "CO", "SB", "CO", "SB"),
        (3, "UTG", "BB", "UTG", "BB"),
    ],
)
def test_ip_oop_inference(street, hero_pos, villain_pos, exp_ip, exp_oop):
    pre = [e(0, 0, "h", hero_pos, -2.5), e(1, 0, "v", villain_pos, -2.5)]
    req = mk_req(street=street, hero_id="h", hero_pos=hero_pos, stack_stream=pre)
    st = ResolvedStateResolver().resolve(req)

    assert st.ip_pos == exp_ip
    assert st.oop_pos == exp_oop


# ==========================================================
# C) Node type inference (ROOT vs FACING) + faced_size_frac
# ==========================================================

def test_node_type_root_when_no_current_street_actions():
    pre = [e(0, 0, "h", "BTN", -2.5), e(1, 0, "v", "BB", -2.5)]
    req = mk_req(street=1, hero_id="h", hero_pos="BTN", stack_stream=pre, pot_stream=[])
    st = ResolvedStateResolver().resolve(req)
    assert st.node_type == "ROOT"
    assert st.faced_size_frac == 0.0


def test_node_type_facing_when_villain_bets_and_hero_not_responded_with_pot_stream():
    pre = [e(0, 0, "h", "BTN", -2.5), e(1, 0, "v", "BB", -2.5)]
    # villain bets 2bb into 6bb pot at tick=10
    post = [
        e(10, 1, "v", "BB", -2.0),
    ]
    pot_stream = [p(10, 1, pot_before=6.0, pot_after=8.0)]
    req = mk_req(
        street=1,
        hero_id="h",
        hero_pos="BTN",
        stack_stream=pre + post,
        pot_stream=pot_stream,
        pot_bb=6.0,
        eff_stack_bb=100.0,
    )
    st = ResolvedStateResolver().resolve(req)
    assert st.node_type == "FACING"
    assert pytest.approx(st.faced_size_frac, rel=1e-6) == (2.0 / 6.0)


def test_node_type_facing_when_no_pot_stream_falls_back_to_request_pot_bb():
    pre = [e(0, 0, "h", "BTN", -2.5), e(1, 0, "v", "BB", -2.5)]
    post = [e(10, 1, "v", "BB", -2.0)]
    req = mk_req(
        street=1,
        hero_id="h",
        hero_pos="BTN",
        stack_stream=pre + post,
        pot_stream=[],
        pot_bb=6.0,
        eff_stack_bb=100.0,
    )
    st = ResolvedStateResolver().resolve(req)
    assert st.node_type == "FACING"
    assert pytest.approx(st.faced_size_frac, rel=1e-6) == (2.0 / 6.0)


def test_node_type_root_when_hero_already_responded_after_villain_bet():
    pre = [e(0, 0, "h", "BTN", -2.5), e(1, 0, "v", "BB", -2.5)]
    post = [
        e(10, 1, "v", "BB", -2.0),  # villain bets
        e(11, 1, "h", "BTN", -2.0), # hero calls
    ]
    req = mk_req(
        street=1,
        hero_id="h",
        hero_pos="BTN",
        stack_stream=pre + post,
        pot_stream=[],
        pot_bb=6.0,
        eff_stack_bb=100.0,
    )
    st = ResolvedStateResolver().resolve(req)
    assert st.node_type == "ROOT"
    assert st.faced_size_frac == 0.0


# ==========================================================
# D) SPR bins
# ==========================================================

@pytest.mark.parametrize(
    "pot_bb, eff_stack_bb, exp_prefix",
    [
        (10.0, 15.0, "SPR_0_2"),      # spr=1.5
        (10.0, 30.0, "SPR_2_5"),      # spr=3.0
        (10.0, 70.0, "SPR_5_10"),     # spr=7.0
        (10.0, 200.0, "SPR_10_PLUS"), # spr=20.0
    ],
)
def test_spr_bins(pot_bb, eff_stack_bb, exp_prefix):
    pre = [e(0, 0, "h", "BTN", -2.5), e(1, 0, "v", "BB", -2.5)]
    req = mk_req(
        street=1,
        hero_id="h",
        hero_pos="BTN",
        stack_stream=pre,
        pot_bb=pot_bb,
        eff_stack_bb=eff_stack_bb,
    )
    st = ResolvedStateResolver().resolve(req)
    assert st.spr is not None
    assert st.spr_bin is not None
    assert st.spr_bin.startswith(exp_prefix)


def test_postflop_requires_pot_and_stack_positive():
    pre = [e(0, 0, "h", "BTN", -2.5), e(1, 0, "v", "BB", -2.5)]
    req = mk_req(
        street=1,
        hero_id="h",
        hero_pos="BTN",
        stack_stream=pre,
        pot_bb=0.0,
        eff_stack_bb=100.0,
    )
    with pytest.raises(ValueError):
        ResolvedStateResolver().resolve(req)