# file: tests/test_postflop_lines.py
import pytest
from tests.conftest import mk_stack, mk_pot, mk_req, pick, assert_ar

def test_flop_cbet_fold_and_call(inf):
    stacks = [
        mk_stack(tick=101, street=1, pid="BTN", seat="BTN", before=100, after=97),   # bet 3
        mk_stack(tick=105, street=1, pid="SB",  seat="SB",  before=100, after=100), # 0-delta -> seen -> fold
        mk_stack(tick=109, street=1, pid="BB",  seat="BB",  before=100, after=97),  # call 3
    ]
    pots = [
        mk_pot(tick=100, street=1, before=6.0,  after=6.0),
        mk_pot(tick=102, street=1, before=6.0,  after=9.0),  # +3
        mk_pot(tick=110, street=1, before=9.0,  after=12.0), # +3
    ]
    req = mk_req(street=1, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    assert_ar(pick(ev, pid="BTN"), pid="BTN", action="BET", street=1, tick=101, raise_level=0, contrib=3.0, prior=0.0, faced=False)
    assert_ar(pick(ev, pid="BB"),  pid="BB",  action="CALL", street=1, tick=109, contrib=3.0, prior=3.0, faced=True)
    f = pick(ev, pid="SB")
    assert f.action == "FOLD" and f.faced_bet and f.prior_bet_bb == pytest.approx(3.0)

def test_turn_check_through(inf):
    stacks = [
        mk_stack(tick=200, street=2, pid="SB",  seat="SB",  before=95, after=95),
        mk_stack(tick=201, street=2, pid="BTN", seat="BTN", before=95, after=95),
    ]
    pots = [ mk_pot(tick=199, street=2, before=12.0, after=12.0) ]
    req = mk_req(street=2, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    assert len(ev) == 2
    assert set(e.action for e in ev) == {"CHECK"}

def test_river_donk_and_call(inf):
    stacks = [
        mk_stack(tick=305, street=3, pid="SB",  seat="SB",  before=95, after=90),   # bet 5
        mk_stack(tick=309, street=3, pid="BTN", seat="BTN", before=95, after=90),   # call 5
    ]
    pots = [
        mk_pot(tick=300, street=3, before=20.0, after=20.0),
        mk_pot(tick=306, street=3, before=20.0, after=25.0),
        mk_pot(tick=310, street=3, before=25.0, after=30.0),
    ]
    req = mk_req(street=3, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    assert_ar(pick(ev, pid="SB"),  pid="SB",  action="BET",  street=3, tick=305, contrib=5.0, prior=0.0, faced=False)
    assert_ar(pick(ev, pid="BTN"), pid="BTN", action="CALL", street=3, tick=309, contrib=5.0, prior=5.0, faced=True)

def test_check_raise_sequence(inf):
    stacks = [
        mk_stack(tick=10, street=1, pid="SB",  seat="SB",  before=100, after=100),  # check (seen)
        mk_stack(tick=12, street=1, pid="BTN", seat="BTN", before=100, after=97),   # bet 3
        mk_stack(tick=14, street=1, pid="SB",  seat="SB",  before=100, after=92),   # raise to 8 (adds 8)
    ]
    pots = [
        mk_pot(tick=9,  street=1, before=6.0, after=6.0),
        mk_pot(tick=13, street=1, before=6.0, after=9.0),
        mk_pot(tick=15, street=1, before=9.0, after=17.0),
    ]
    req = mk_req(street=1, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    sb_raise = pick(ev, pid="SB", tick=14)
    assert sb_raise.action in ("RAISE","ALLIN")
    assert sb_raise.raise_level >= 1