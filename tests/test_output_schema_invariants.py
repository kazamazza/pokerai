# file: tests/test_output_schema_invariants.py
import pytest
from tests.conftest import mk_stack, mk_pot, mk_req, pick, assert_ar


def test_size_frac_none_when_pot_before_zero(inf):
    stacks = [ mk_stack(tick=5, street=1, pid="BTN", seat="BTN", before=100, after=97) ]
    pots = []  # no pot → pot_before fallback uses req.pot_bb (0.0 default) => size_frac None
    req = mk_req(street=1, stack_stream=stacks, pot_stream=pots, pot_bb=0.0)
    ev = inf.infer(req)
    act = pick(ev, pid="BTN")
    assert act.size_frac is None

def test_pot_after_fallback_equals_before_plus_contrib(inf):
    stacks = [ mk_stack(tick=5, street=1, pid="BTN", seat="BTN", before=100, after=97) ] # 3
    pots = [ mk_pot(tick=3, street=1, before=6.0, after=6.0) ]  # no exact tick=5 entry
    req = mk_req(street=1, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    act = pick(ev, pid="BTN")
    assert act.pot_before_bb == 6.0
    assert act.pot_after_bb == pytest.approx(9.0)

def test_raise_level_progression(inf):
    stacks = [
        mk_stack(tick=5,  street=0, pid="UTG", seat="UTG", before=100, after=97.5),  # 2.5
        mk_stack(tick=8,  street=0, pid="BTN", seat="BTN", before=100, after=92.0),  # 8.0
        mk_stack(tick=12, street=0, pid="UTG", seat="UTG", before=97.5, after=80.0), # large raise (4bet)
    ]
    pots = [
        mk_pot(tick=4, street=0, before=1.5, after=1.5),
        mk_pot(tick=6, street=0, before=1.5, after=4.0),
        mk_pot(tick=9, street=0, before=4.0,  after=12.0),
        mk_pot(tick=13,street=0, before=12.0, after=29.5),
    ]
    req = mk_req(street=0, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    levels = { (e.player_id, e.tick): e.raise_level for e in ev if e.action in ("BET","RAISE","ALLIN") }
    assert levels.get(("UTG",5), None) == 0
    assert levels.get(("BTN",8), None) == 1
    assert levels.get(("UTG",12), None) >= 2