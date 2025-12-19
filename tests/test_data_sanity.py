# file: tests/test_data_sanity.py
import math
import pytest
from tests.conftest import mk_stack, mk_pot, mk_req, pick, assert_ar, EPS


def test_ignore_non_negative_deltas(inf):
    stacks = [
        mk_stack(tick=10, street=1, pid="BTN", seat="BTN", before=100, after=100), # 0
        mk_stack(tick=12, street=1, pid="BB",  seat="BB",  before=100, after=97),  # 3
    ]
    pots = [
        mk_pot(tick=9, street=1, before=6.0, after=6.0),
        mk_pot(tick=13, street=1, before=6.0, after=9.0),
    ]
    req = mk_req(street=1, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    assert any(e.player_id == "BB" for e in ev)
    assert all(e.player_id != "BTN" or e.action != "BET" for e in ev)

def test_eps_tolerance_small_noise(inf):
    tiny = EPS / 10.0
    stacks = [
        mk_stack(tick=10, street=1, pid="BTN", seat="BTN", before=100, after=100 + tiny),  # +tiny (should be ignored)
        mk_stack(tick=12, street=1, pid="BTN", seat="BTN", before=100, after=97),          # -3 (action)
    ]
    pots = [
        mk_pot(tick=9, street=1, before=6.0, after=6.0),
        mk_pot(tick=13, street=1, before=6.0, after=9.0),
    ]
    req = mk_req(street=1, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    assert any(e.player_id == "BTN" and e.contrib_bb == pytest.approx(3.0) for e in ev)

def test_missing_when_ms_handled(inf):
    stacks = [ mk_stack(tick=10, street=1, pid="BTN", seat="BTN", before=100, after=97, when_ms=None) ]
    pots = [ mk_pot(tick=9, street=1, before=6.0, after=9.0, when_ms=None) ]
    req = mk_req(street=1, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    assert ev and ev[0].tick == 10