# file: tests/test_timestamps_ordering.py
import pytest
from tests.conftest import mk_stack, mk_pot, mk_req, pick, assert_ar

def test_equal_tick_ordered_by_when_ms(inf):
    stacks = [
        mk_stack(tick=10, when_ms=1000, street=1, pid="BTN", seat="BTN", before=100, after=97),  # bet first
        mk_stack(tick=10, when_ms=1005, street=1, pid="BB",  seat="BB",  before=100, after=97),  # call next
    ]
    pots = [
        mk_pot(tick=9, street=1, before=6.0, after=6.0),
        mk_pot(tick=11, street=1, before=6.0, after=12.0),
    ]
    req = mk_req(street=1, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    assert ev[0].player_id == "BTN" and ev[0].action in ("BET","RAISE")

def test_equal_tick_equal_when_ms_aggro_first_in_output_sort(inf):
    stacks = [
        mk_stack(tick=10, when_ms=1000, street=1, pid="BTN", seat="BTN", before=100, after=97),  # bet
        mk_stack(tick=10, when_ms=1000, street=1, pid="BB",  seat="BB",  before=100, after=97),  # call
    ]
    pots = [
        mk_pot(tick=9, street=1, before=6.0, after=6.0),
        mk_pot(tick=11, street=1, before=6.0, after=12.0),
    ]
    req = mk_req(street=1, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    # Aggro-first tie-breaker in final sort
    assert ev[0].action in ("BET","RAISE","ALLIN")