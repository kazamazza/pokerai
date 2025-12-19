# file: tests/test_generated_records.py
import pytest
from tests.conftest import mk_stack, mk_pot, mk_req, pick, assert_ar

def test_generated_fold_for_seen_zero_delta(inf):
    stacks = [
        mk_stack(tick=20, street=1, pid="BTN", seat="BTN", before=100, after=97),   # bet 3
        mk_stack(tick=22, street=1, pid="SB",  seat="SB",  before=100, after=100), # seen but no contrib
    ]
    pots = [
        mk_pot(tick=19, street=1, before=6.0, after=6.0),
        mk_pot(tick=21, street=1, before=6.0, after=9.0),
    ]
    req = mk_req(street=1, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    f = pick(ev, pid="SB")
    assert f.action == "FOLD" and f.faced_bet and f.prior_bet_bb == pytest.approx(3.0)

def test_generated_checks_when_no_contributions(inf):
    stacks = [
        mk_stack(tick=50, street=2, pid="SB",  seat="SB",  before=100, after=100),
        mk_stack(tick=51, street=2, pid="BTN", seat="BTN", before=100, after=100),
    ]
    pots = [ mk_pot(tick=49, street=2, before=12.0, after=12.0) ]
    req = mk_req(street=2, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    assert set(e.action for e in ev) == {"CHECK"}