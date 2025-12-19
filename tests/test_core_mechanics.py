# file: tests/test_core_mechanics.py
import pytest

from tests.conftest import mk_stack, mk_pot, mk_req, pick


def test_street_filtering_only_current(inf):
    stacks = [
        mk_stack(tick=1, street=0, pid="UTG", seat="UTG", before=100, after=97.5),
        mk_stack(tick=2, street=1, pid="BTN", seat="BTN", before=100, after=97.0),
    ]
    pots = [
        mk_pot(tick=0, street=0, before=1.5, after=4.0),
        mk_pot(tick=3, street=1, before=6.0, after=9.0),
    ]
    req0 = mk_req(street=0, stack_stream=stacks, pot_stream=pots)
    ev0 = inf.infer(req0)
    assert any(e.player_id == "UTG" for e in ev0)
    assert all(e.street == 0 for e in ev0)

    req1 = mk_req(street=1, stack_stream=stacks, pot_stream=pots)
    ev1 = inf.infer(req1)
    assert any(e.player_id == "BTN" for e in ev1)
    assert all(e.street == 1 for e in ev1)

def test_ordering_aggro_first_on_tie(inf):
    stacks = [
        mk_stack(tick=10, street=1, pid="BTN", seat="BTN", before=100, after=97),  # bet 3
        mk_stack(tick=10, street=1, pid="BB",  seat="BB",  before=100, after=97),  # call 3
    ]
    pots = [ mk_pot(tick=9, street=1, before=6.0, after=6.0),
             mk_pot(tick=11, street=1, before=6.0, after=12.0) ]
    req = mk_req(street=1, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    assert ev[0].action in ("BET","RAISE","ALLIN")
    assert ev[1].action in ("CALL","FOLD","CHECK")

def test_pot_fallback_and_size_frac_from_req_pot(inf):
    stacks = [ mk_stack(tick=5, street=1, pid="BTN", seat="BTN", before=100, after=97) ] # +3
    pots = []  # missing pot stream
    req = mk_req(street=1, stack_stream=stacks, pot_stream=pots, pot_bb=12.0)
    ev = inf.infer(req)
    act = pick(ev, pid="BTN")
    assert act.size_frac is not None
    assert pytest.approx(act.size_frac, rel=1e-6) == pytest.approx(3.0/12.0, rel=1e-6)
    assert act.pot_before_bb == 12.0
    assert act.pot_after_bb == pytest.approx(15.0)

def test_amount_mirror_and_prior_progression(inf):
    stacks = [
        mk_stack(tick=5, street=0, pid="UTG", seat="UTG", before=100, after=97.5),  # 2.5
        mk_stack(tick=8, street=0, pid="BTN", seat="BTN", before=100, after=92.0),  # 8.0
    ]
    pots = [
        mk_pot(tick=4, street=0, before=1.5, after=1.5),
        mk_pot(tick=6, street=0, before=1.5, after=4.0),
        mk_pot(tick=9, street=0, before=4.0,  after=12.0),
    ]
    req = mk_req(street=0, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    utg = pick(ev, pid="UTG")
    btn = pick(ev, pid="BTN")
    assert utg.amount_bb == pytest.approx(2.5)
    assert btn.prior_bet_bb == pytest.approx(2.5)