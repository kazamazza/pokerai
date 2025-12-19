# file: tests/test_table_state_blinds_straddles.py
import pytest
from tests.conftest import mk_stack, mk_pot, mk_req, pick, assert_ar


def test_preflop_limp_is_call_when_blinds_seeded(inf):
    # Simulate blinds in pot (pot_before=1.5) but no blind deltas; UTG puts 1bb (limp) which should be CALL vs 1bb price.
    stacks = [
        mk_stack(tick=5, street=0, pid="UTG", seat="UTG", before=100, after=99),   # 1bb limp
    ]
    pots = [
        mk_pot(tick=4, street=0, before=1.5, after=2.5),  # +1
    ]
    req = mk_req(street=0, stack_stream=stacks, pot_stream=pots, pot_bb=1.5)
    ev = inf.infer(req)
    utg = pick(ev, pid="UTG")
    assert utg.action == "CALL"


def test_straddle_sets_prior_price(inf):
    # Pot has 3bb before any action due to straddle; first to act overcalls 3bb => CALL
    stacks = [
        mk_stack(tick=5, street=0, pid="HJ", seat="HJ", before=100, after=97),  # 3bb
    ]
    pots = [
        mk_pot(tick=4, street=0, before=3.0, after=6.0),
    ]
    req = mk_req(street=0, stack_stream=stacks, pot_stream=pots, pot_bb=3.0)
    ev = inf.infer(req)
    assert pick(ev, pid="HJ").action == "CALL"