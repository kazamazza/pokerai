# file: tests/test_preflop_lines.py
import pytest
from tests.conftest import mk_stack, mk_pot, mk_req, pick, assert_ar


def test_open_and_call(inf):
    stacks = [
        mk_stack(tick=5, street=0, pid="UTG", seat="UTG", before=100, after=97.5),
        mk_stack(tick=8, street=0, pid="BTN", seat="BTN", before=100, after=97.5),
    ]
    pots = [
        mk_pot(tick=4, street=0, before=1.5, after=1.5),
        mk_pot(tick=6, street=0, before=1.5, after=4.0),
        mk_pot(tick=9, street=0, before=4.0, after=6.5),
    ]
    req = mk_req(street=0, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    assert_ar(pick(ev, pid="UTG"), pid="UTG", action="RAISE", street=0, tick=5, raise_level=0, contrib=2.5, prior=0.0, faced=False)
    assert_ar(pick(ev, pid="BTN"), pid="BTN", action="CALL", street=0, tick=8, contrib=2.5, prior=2.5, faced=True)

def test_threebet_line(inf):
    stacks = [
        mk_stack(tick=5,  street=0, pid="CO",  seat="CO",  before=100, after=97.5),  # 2.5
        mk_stack(tick=9,  street=0, pid="BTN", seat="BTN", before=100, after=92.0),  # 8.0
        mk_stack(tick=12, street=0, pid="CO",  seat="CO",  before=97.5, after=92.0), # 5.5
    ]
    pots = [
        mk_pot(tick=4,  street=0, before=1.5, after=1.5),
        mk_pot(tick=6,  street=0, before=1.5, after=4.0),    # +2.5
        mk_pot(tick=10, street=0, before=4.0, after=12.0),   # +8.0
        mk_pot(tick=13, street=0, before=12.0, after=17.5),  # +5.5
    ]
    req = mk_req(street=0, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    # opener
    assert_ar(pick(ev, pid="CO", tick=5), pid="CO", action="RAISE", street=0, tick=5, raise_level=0, contrib=2.5, prior=0.0, faced=False)
    # 3bet
    assert_ar(pick(ev, pid="BTN", tick=9), pid="BTN", action="RAISE", street=0, tick=9, raise_level=1, contrib=8.0, prior=2.5, faced=True)
    # caller to 3bet
    assert_ar(pick(ev, pid="CO", tick=12), pid="CO", action="CALL", street=0, tick=12, contrib=5.5, prior=8.0, faced=True)

def test_squeeze_threebet(inf):
    stacks = [
        mk_stack(tick=5, street=0, pid="UTG", seat="UTG", before=100, after=97.5),  # open 2.5
        mk_stack(tick=7, street=0, pid="CO",  seat="CO",  before=100, after=97.5),  # cold call 2.5
        mk_stack(tick=10,street=0, pid="BB",  seat="BB",  before=100, after=90.0),  # squeeze 10
    ]
    pots = [
        mk_pot(tick=4, street=0, before=1.5, after=1.5),
        mk_pot(tick=6, street=0, before=1.5, after=4.0),
        mk_pot(tick=8, street=0, before=4.0, after=6.5),
        mk_pot(tick=11,street=0, before=6.5, after=16.5),
    ]
    req = mk_req(street=0, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    assert_ar(pick(ev, pid="BB", tick=10), pid="BB", action="RAISE", street=0, tick=10, raise_level=1, contrib=10.0, prior=2.5, faced=True)

def test_backraise(inf):
    stacks = [
        mk_stack(tick=5,  street=0, pid="UTG", seat="UTG", before=100,  after=97.5),  # open 2.5
        mk_stack(tick=8,  street=0, pid="BTN", seat="BTN", before=100,  after=92.0),  # 3bet 8
        mk_stack(tick=12, street=0, pid="UTG", seat="UTG", before=97.5, after=80.0),  # 4bet 17.5 (adds 15.0? here cumulative is 20, but test focuses level)
    ]
    pots = [
        mk_pot(tick=6, street=0, before=1.5, after=4.0),
        mk_pot(tick=9, street=0, before=4.0,  after=12.0),
        mk_pot(tick=13,street=0, before=12.0, after=29.5),
    ]
    req = mk_req(street=0, stack_stream=stacks, pot_stream=pots)
    ev = inf.infer(req)
    # back-raiser should be a raise with next level
    utg4 = pick(ev, pid="UTG", tick=12)
    assert utg4.action in ("RAISE","ALLIN")
    assert utg4.raise_level >= 2