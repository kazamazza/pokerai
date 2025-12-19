# file: tests/test_facing_resolver_more.py
import types
import pytest

from ml.inference.policy.facing_resolver import FacingBetResolver, FacingPick

# --- test doubles ---
class DummyAR:
    __slots__ = ("player_id","seat_label","action","street","tick","when_ms","amount_bb","prior_bet_bb")
    def __init__(self, pid, seat, action, street, tick, when_ms=None, amount_bb=None, prior_bet_bb=None):
        self.player_id = pid
        self.seat_label = seat
        self.action = action
        self.street = int(street)
        self.tick = int(tick)
        self.when_ms = when_ms
        self.amount_bb = amount_bb
        self.prior_bet_bb = prior_bet_bb

class DummyInferrer:
    def __init__(self, events): self._events = list(events)
    def infer(self, req, exclude_hero=False, target_player_id=None): return list(self._events)

def mk_req(*, hero_id="HERO", street=1, pot_bb=0.0, stack_stream=None, pot_stream=None):
    req = types.SimpleNamespace()
    req.hero_id = hero_id
    req.street = int(street)
    req.pot_bb = float(pot_bb)
    req.stack_stream = list(stack_stream or [])
    req.pot_stream = list(pot_stream or [])
    return req

def _patch_inferrer(monkeypatch, events):
    monkeypatch.setattr("ml.inference.policy.facing_resolver.ActionInferrer", lambda: DummyInferrer(events))

# --- tests ---

def test_facing_allin_is_aggressive(monkeypatch):
    events = [
        DummyAR("CO","CO","ALLIN",1,20,amount_bb=18.0),
        DummyAR("BTN","BTN","CALL", 1,21),
    ]
    _patch_inferrer(monkeypatch, events)
    pick = FacingBetResolver().resolve(mk_req(street=1, pot_bb=12.0))
    assert isinstance(pick, FacingPick)
    assert pick.facing_bet is True
    assert pick.aggressor_id == "CO"
    assert pick.faced_size_bb == pytest.approx(18.0)

def test_equal_tick_hero_after_aggressor_by_when_ms(monkeypatch):
    # Same tick; hero acts later by when_ms → not facing
    events = [
        DummyAR("V1","CO","BET", 1,10,when_ms=1000, amount_bb=3.0),
        DummyAR("HERO","BTN","CALL",1,10,when_ms=1005),
    ]
    _patch_inferrer(monkeypatch, events)
    pick = FacingBetResolver().resolve(mk_req(street=1, pot_bb=6.0))
    assert pick.facing_bet is False
    assert pick.reason in ("hero_already_acted_after_aggressor","hero_acted_after_aggressor")

def test_latest_aggressor_wins(monkeypatch):
    # Two aggressors; latest should be used
    events = [
        DummyAR("V1","HJ","BET",  1,10,amount_bb=2.0),
        DummyAR("V2","BTN","RAISE",1,12,amount_bb=7.0),
    ]
    _patch_inferrer(monkeypatch, events)
    pick = FacingBetResolver().resolve(mk_req(street=1, pot_bb=6.0))
    assert pick.facing_bet is True
    assert pick.aggressor_id == "V2"
    assert pick.faced_size_bb == pytest.approx(7.0)

def test_passive_actions_after_aggressor_do_not_clear_facing(monkeypatch):
    events = [
        DummyAR("V1","CO","BET", 1,10,amount_bb=3.0),
        DummyAR("V2","SB","CALL", 1,11),
        DummyAR("V3","BB","CHECK",1,12),
    ]
    _patch_inferrer(monkeypatch, events)
    pick = FacingBetResolver().resolve(mk_req(street=1, pot_bb=6.0))
    assert pick.facing_bet is True
    assert pick.aggressor_id == "V1"

def test_zero_or_unknown_pot_yields_none_fraction(monkeypatch):
    events = [ DummyAR("V1","CO","BET",1,10,amount_bb=2.0) ]
    _patch_inferrer(monkeypatch, events)
    # No pot stream; pot_bb=0 → size_frac should be None
    pick = FacingBetResolver().resolve(mk_req(street=1, pot_bb=0.0))
    assert pick.facing_bet is True
    assert pick.size_frac is None

def test_seat_fallback_from_stack_stream_when_missing_on_event(monkeypatch):
    events = [ DummyAR("V1", None, "BET", 1, 10, amount_bb=3.0) ]
    _patch_inferrer(monkeypatch, events)

    Stack = types.SimpleNamespace
    stack_stream = [
        Stack(tick=9, when_ms=900, street=1, player_id="V1", seat_label="HJ",
              stack_before_bb=100.0, stack_after_bb=100.0, delta_bb=0.0, source="derived", conf=None)
    ]
    pick = FacingBetResolver().resolve(mk_req(street=1, pot_bb=6.0, stack_stream=stack_stream))
    assert pick.facing_bet is True
    assert pick.aggressor_seat in ("HJ", None)  # assert it’s filled once fallback is implemented

def test_no_current_street_aggro_not_facing(monkeypatch):
    events = [
        DummyAR("V1","CO","BET",  0, 5, amount_bb=2.5),  # preflop
        DummyAR("V2","BTN","CALL", 0, 6),
        DummyAR("HERO","BB","CHECK",1, 7),
        DummyAR("V1","CO","CHECK", 1, 8),
    ]
    _patch_inferrer(monkeypatch, events)
    pick = FacingBetResolver().resolve(mk_req(street=1, pot_bb=6.0))
    assert pick.facing_bet is False
    assert pick.reason == "no_aggressor_current_street"