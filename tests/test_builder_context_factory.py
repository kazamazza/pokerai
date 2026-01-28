import pytest

from ml.inference.policy.types import PolicyRequest
from ml.inference.resolve.resolver import ResolvedStateResolver
from ml.inference.builder_context.factory import BuilderContextFactory
from ml.inference.types.observed_request import StackChangeEvent


def e(tick, street, pid, seat, delta_bb, before=100.0):
    after = before + float(delta_bb)
    return StackChangeEvent(
        tick=tick, when_ms=None, street=street,
        player_id=pid, seat_label=seat,
        stack_before_bb=before, stack_after_bb=after,
        delta_bb=float(delta_bb),
        source="derived", conf=1.0,
    )


@pytest.mark.parametrize(
    "name, pre_stream, exp_ctx, exp_topo, exp_role, exp_ip, exp_oop, exp_menu",
    [
        # SRP aggressor (PFR IP): UTG vs BB, hero=UTG
        ("srp_pfr_ip", [e(0,0,"h","UTG",-2.5), e(1,0,"v","BB",-2.5)],
         "VS_OPEN","SRP","AGGRESSOR","UTG","BB","srp_hu.PFR_IP"),

        # SRP caller (Caller OOP): UTG vs BB, hero=BB (caller)
        ("srp_caller_oop", [e(0,0,"v","UTG",-2.5), e(1,0,"h","BB",-2.5)],
         "VS_OPEN","SRP","CALLER","UTG","BB","srp_hu.Caller_OOP"),

        # BVS any: BTN vs BB
        ("bvs_any", [e(0,0,"h","BTN",-2.5), e(1,0,"v","BB",-2.5)],
         "BLIND_VS_STEAL","BVS","ANY","BTN","BB","bvs.Any"),

        # 3BP aggressor IP: (menu selected by IP=BTN/CO)
        # hero is last raiser? in your resolver role logic, last raiser determines AGGRESSOR.
        ("3bp_agg_ip",
         [e(0, 0, "h", "BTN", -2.5), e(1, 0, "v", "BB", -9.0), e(2, 0, "h", "BTN", -6.5)],
         "VS_3BET", "3BP", "AGGRESSOR", "BTN", "BB", "3bet_hu.Aggressor_OOP"),

        # 3BP aggressor OOP: make OOP the last raiser
        ("3bp_agg_oop",
         [e(0, 0, "h", "BTN", -2.5), e(1, 0, "v", "BB", -9.0), e(2, 0, "h", "BTN", -6.5)],
         "VS_3BET", "3BP", "AGGRESSOR", "BTN", "BB", "3bet_hu.Aggressor_OOP"),

        # 4BP aggressor IP: hero last raiser
        ("4bp_agg_ip", [e(0,0,"h","BTN",-2.5), e(1,0,"v","BB",-9.0), e(2,0,"h","BTN",-21.5)],
         "VS_4BET","4BP","AGGRESSOR","BTN","BB","4bet_hu.Aggressor_IP"),

        # 4BP aggressor OOP: OOP last raiser (still chooses by IP seat in mapping)
        ("4bp_agg_oop", [e(0,0,"h","BTN",-2.5), e(1,0,"v","BB",-9.0), e(2,0,"h","BTN",-21.5), e(3,0,"v","BB",-10.0)],
         "VS_4BET","4BP","AGGRESSOR","BTN","BB","4bet_hu.Aggressor_IP"),

        # Limp BB IP
        ("limp_bb_ip", [e(0,0,"h","SB",-0.5), e(1,0,"v","BB",0.0)],
         "LIMPED_SINGLE","LIMP","ANY","BB","SB","limped_single.BB_IP"),
    ],
)
def test_builder_context_from_resolved_state(
    name, pre_stream, exp_ctx, exp_topo, exp_role, exp_ip, exp_oop, exp_menu
):
    req = PolicyRequest(
        stakes="NL10",
        street=1,
        hero_id="h",
        hero_pos=next(x.seat_label for x in pre_stream if x.player_id == "h"),
        board="Ts5cKd",
        pot_bb=6.0,
        eff_stack_bb=100.0,
        stack_stream=pre_stream,
        pot_stream=[],
        street_transitions=[],
    )
    st = ResolvedStateResolver().resolve(req)
    ctx = BuilderContextFactory.from_resolved_state(st)

    assert ctx.ctx == exp_ctx
    assert ctx.topology == exp_topo
    assert ctx.role == exp_role
    assert ctx.ip_pos == exp_ip
    assert ctx.oop_pos == exp_oop
    assert ctx.bet_sizing_id == exp_menu