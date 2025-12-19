# tests/test_action_inferrer.py
import pytest
from types import SimpleNamespace as NS

from ml.inference.exploit.infer_actions import ActionInferrer

# ⬇️ Adjust this path if your file is named differently.


EPS = 1e-6

# ---------- tiny duck-typed builders (no project imports) ---------------------
def mk_stack(*, tick, street, pid, seat, before, after, when_ms=None, source="derived", conf=None):
    return NS(
        tick=int(tick),
        when_ms=when_ms,
        street=int(street),
        player_id=str(pid),
        seat_label=str(seat),
        stack_before_bb=float(before),
        stack_after_bb=float(after),
        delta_bb=float(after) - float(before),   # negative when chips go in
        source=source,
        conf=conf,
    )

def mk_pot(*, tick, street, before, after, when_ms=None, source="derived"):
    return NS(
        tick=int(tick),
        when_ms=when_ms,
        street=int(street),
        pot_before_bb=float(before),
        pot_after_bb=float(after),
        delta_bb=float(after) - float(before),
        source=source,
    )

def mk_tr(*, to_street, tick, when_ms=None, reason="card_seen"):
    return NS(
        to_street=int(to_street),
        tick=int(tick),
        when_ms=when_ms,
        reason=reason,
    )

def mk_req(
    *,
    street,
    stack_stream=None,
    pot_stream=None,
    street_transitions=None,
    hero_id=None,
    hand_id="H1",
    pot_bb=0.0,
):
    return NS(
        street=int(street),
        stack_stream=list(stack_stream or []),
        pot_stream=list(pot_stream or []),
        street_transitions=list(street_transitions or []),
        hero_id=hero_id,
        hand_id=hand_id,
        pot_bb=float(pot_bb),
    )

# handy assertion
def assert_ar(ev, *, pid, action, street, tick, raise_level=None, contrib=None, prior=None, faced=None):
    assert ev.player_id == pid
    assert ev.action == action
    assert int(ev.street) == street
    assert int(ev.tick) == tick
    if raise_level is not None:
        assert getattr(ev, "raise_level", None) == raise_level
    if contrib is not None:
        assert pytest.approx(getattr(ev, "contrib_bb", 0.0), rel=1e-6) == pytest.approx(contrib, rel=1e-6)
    if prior is not None:
        assert pytest.approx(getattr(ev, "prior_bet_bb", 0.0), rel=1e-6) == pytest.approx(prior, rel=1e-6)
    if faced is not None:
        assert bool(getattr(ev, "faced_bet", False)) is bool(faced)

# ---------- tests -------------------------------------------------------------

def test_preflop_open_and_call():
    """
    Preflop: UTG opens 2.5bb at tick 5; BTN calls at tick 8.
    Expect: UTG RAISE(lvl=0, contrib=2.5), BTN CALL (prior=2.5, contrib=2.5, faced=True).
    """
    stacks = [
        mk_stack(tick=5, street=0, pid="UTG", seat="UTG", before=100, after=97.5),
        mk_stack(tick=8, street=0, pid="BTN", seat="BTN", before=100, after=97.5),
    ]
    pots = [
        mk_pot(tick=4, street=0, before=1.5, after=1.5),
        mk_pot(tick=6, street=0, before=1.5, after=4.0),   # +2.5
        mk_pot(tick=9, street=0, before=4.0, after=6.5),   # +2.5
    ]
    req = mk_req(street=0, stack_stream=stacks, pot_stream=pots)

    ev = ActionInferrer().infer(req, exclude_hero=True, target_player_id=None)
    # Should at least have the two actions
    assert len(ev) >= 2
    # First: opener
    assert_ar(ev[0], pid="UTG", action="RAISE", street=0, tick=5, raise_level=0, contrib=2.5, prior=0.0, faced=False)
    # Second: caller
    assert_ar(ev[1], pid="BTN", action="CALL", street=0, tick=8, raise_level=0, contrib=2.5, prior=2.5, faced=True)

def test_preflop_threebet_line():
    """
    Preflop: CO opens 2.5bb (t=5), BTN 3bets to 8.0 (t=9), CO calls 5.5 (t=12).
    Expect sequence: CO RAISE(lvl=0, 2.5), BTN RAISE(lvl=1, 8.0), CO CALL(prior=8.0, 5.5).
    """
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

    ev = ActionInferrer().infer(req, exclude_hero=True, target_player_id=None)
    # Pull only the two players in order
    seq = [(e.player_id, e.action, e.tick, getattr(e, "raise_level", None), getattr(e, "contrib_bb", 0.0), getattr(e, "prior_bet_bb", 0.0)) for e in ev]
    # First two should be open (lvl0) then 3bet (lvl1)
    p0 = seq[0]
    assert p0[0] == "CO" and p0[1] == "RAISE" and p0[2] == 5 and p0[3] == 0 and pytest.approx(p0[4], rel=1e-6) == 2.5
    p1 = seq[1]
    assert p1[0] == "BTN" and p1[1] == "RAISE" and p1[2] == 9 and p1[3] == 1 and pytest.approx(p1[4], rel=1e-6) == 8.0
    # Caller (CO)
    co_call = [e for e in ev if e.player_id == "CO" and e.tick == 12][0]
    assert_ar(co_call, pid="CO", action="CALL", street=0, tick=12, contrib=5.5, prior=8.0, faced=True)

def test_flop_cbet_fold_and_call():
    """
    Flop: PFR bets 3bb into 6bb; one opponent folds (no delta → 0-delta event), one calls 3bb.
    Expect: bettor BET(lvl=0), a FOLD record for the zero-delta seat, a CALL for the caller.
    """
    stacks = [
        mk_stack(tick=101, street=1, pid="BTN", seat="BTN", before=100, after=97),   # bet 3
        mk_stack(tick=105, street=1, pid="SB",  seat="SB",  before=100, after=100), # 0-delta → present -> fold
        mk_stack(tick=109, street=1, pid="BB",  seat="BB",  before=100, after=97),  # call 3
    ]
    pots = [
        mk_pot(tick=100, street=1, before=6.0,  after=6.0),
        mk_pot(tick=102, street=1, before=6.0,  after=9.0),  # +3
        mk_pot(tick=110, street=1, before=9.0,  after=12.0), # +3
    ]
    req = mk_req(street=1, stack_stream=stacks, pot_stream=pots)

    ev = ActionInferrer().infer(req, exclude_hero=True, target_player_id=None)
    # bettor
    b = [e for e in ev if e.player_id == "BTN"][0]
    assert_ar(b, pid="BTN", action="BET", street=1, tick=101, raise_level=0, contrib=3.0, prior=0.0, faced=False)
    # caller
    c = [e for e in ev if e.player_id == "BB"][0]
    assert_ar(c, pid="BB", action="CALL", street=1, tick=109, contrib=3.0, prior=3.0, faced=True)
    # folder (generated)
    f = [e for e in ev if e.player_id == "SB"][0]
    assert_ar(f, pid="SB", action="FOLD", street=1, tick=101, contrib=0.0, prior=3.0, faced=True)

def test_turn_check_through_and_river_donk_bet():
    """
    Turn: everyone checks (no contributions) -> inferrer emits CHECKs for seen players.
    River: OOP (SB) donk-bets 5bb; IP (BTN) calls 5bb.
    """
    # TURN street=2: zero-delta events so players are "seen"
    stacks_turn = [
        mk_stack(tick=200, street=2, pid="SB",  seat="SB",  before=95, after=95),
        mk_stack(tick=201, street=2, pid="BTN", seat="BTN", before=95, after=95),
    ]
    pots_turn = [ mk_pot(tick=199, street=2, before=12.0, after=12.0) ]
    req_turn = mk_req(street=2, stack_stream=stacks_turn, pot_stream=pots_turn)
    ev_t = ActionInferrer().infer(req_turn, exclude_hero=True, target_player_id=None)
    assert len(ev_t) == 2
    assert any(e.player_id == "SB" and e.action == "CHECK" for e in ev_t)
    assert any(e.player_id == "BTN" and e.action == "CHECK" for e in ev_t)

    # RIVER street=3: donk 5, call 5
    stacks_riv = [
        mk_stack(tick=305, street=3, pid="SB",  seat="SB",  before=95, after=90),   # bet 5
        mk_stack(tick=309, street=3, pid="BTN", seat="BTN", before=95, after=90),   # call 5
    ]
    pots_riv = [
        mk_pot(tick=300, street=3, before=20.0, after=20.0),
        mk_pot(tick=306, street=3, before=20.0, after=25.0),
        mk_pot(tick=310, street=3, before=25.0, after=30.0),
    ]
    req_riv = mk_req(street=3, stack_stream=stacks_riv, pot_stream=pots_riv, street_transitions=[mk_tr(to_street=3, tick=300)])
    ev_r = ActionInferrer().infer(req_riv, exclude_hero=True, target_player_id=None)
    sb = [e for e in ev_r if e.player_id == "SB"][0]
    btn = [e for e in ev_r if e.player_id == "BTN"][0]
    assert_ar(sb, pid="SB", action="BET", street=3, tick=305, contrib=5.0, prior=0.0, faced=False)
    assert_ar(btn, pid="BTN", action="CALL", street=3, tick=309, contrib=5.0, prior=5.0, faced=True)

def test_allin_detection_and_amount():
    """
    River: CO shoves remaining 25bb; detect ALLIN and contribution=25.
    """
    stacks = [ mk_stack(tick=401, street=3, pid="CO", seat="CO", before=25.0, after=0.0) ]
    pots   = [ mk_pot(tick=400, street=3, before=40.0, after=65.0) ]  # +25
    req = mk_req(street=3, stack_stream=stacks, pot_stream=pots, street_transitions=[mk_tr(to_street=3, tick=300)])

    ev = ActionInferrer().infer(req, exclude_hero=True, target_player_id=None)
    assert len(ev) >= 1
    shove = ev[-1]
    assert shove.player_id == "CO"
    assert shove.action in ("ALLIN", "RAISE", "BET")
    assert pytest.approx(getattr(shove, "contrib_bb", 0.0), rel=1e-6) == 25.0

def test_hero_excluded_and_target_filter():
    """
    Ensure exclude_hero removes hero entries and target_player_id filters to one player.
    """
    stacks = [
        mk_stack(tick=5, street=0, pid="HERO", seat="BB",  before=100, after=97),  # hero contributed
        mk_stack(tick=6, street=0, pid="BTN",  seat="BTN", before=100, after=98),  # villain contributed
    ]
    pots = [
        mk_pot(tick=4, street=0, before=1.5, after=1.5),
        mk_pot(tick=7, street=0, before=1.5, after=4.5),
    ]
    req = mk_req(street=0, stack_stream=stacks, pot_stream=pots, hero_id="HERO")

    # exclude_hero=True: only BTN remains
    ev1 = ActionInferrer().infer(req, exclude_hero=True, target_player_id=None)
    assert all(e.player_id != "HERO" for e in ev1)
    assert any(e.player_id == "BTN" for e in ev1)

    # target_player_id narrows further
    ev2 = ActionInferrer().infer(req, exclude_hero=False, target_player_id="BTN")
    assert all(e.player_id == "BTN" for e in ev2)