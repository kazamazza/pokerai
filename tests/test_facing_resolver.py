import types
import math
import pytest

from ml.inference.policy.facing_resolver import FacingBetResolver, FacingPick

# Reuse a test-double ActionRecord
class DummyAR:
    __slots__ = ("player_id","seat_label","action","street","tick","when_ms",
                 "amount_bb","prior_bet_bb")
    def __init__(self, player_id, seat_label, action, street, tick,
                 when_ms=None, amount_bb=None, prior_bet_bb=None):
        self.player_id = player_id
        self.seat_label = seat_label
        self.action = action
        self.street = street
        self.tick = tick
        self.when_ms = when_ms
        self.amount_bb = amount_bb
        self.prior_bet_bb = prior_bet_bb

class DummyInferrer:
    def __init__(self, events):
        self._events = events
    def infer(self, req, exclude_hero=False, target_player_id=None):
        return list(self._events)

def mk_req(*, hero_id="HERO", street=1, pot_bb=6.0, stack_stream=None, pot_stream=None):
    req = types.SimpleNamespace()
    req.hero_id = hero_id
    req.street = street
    req.pot_bb = pot_bb
    req.stack_stream = stack_stream or []
    req.pot_stream = pot_stream or []
    return req

def test_facing_true_with_amount_and_fraction(monkeypatch):
    # Villain bets 3bb into pot_before 6bb → size_frac ≈ 0.5
    events = [
        DummyAR("BTN","BTN","BET",1,10,amount_bb=3.0),
    ]
    monkeypatch.setattr(
        "ml.inference.policy.facing_resolver.ActionInferrer",
        lambda: DummyInferrer(events)
    )
    req = mk_req(street=1, pot_bb=6.0)
    pick = FacingBetResolver().resolve(req)
    assert isinstance(pick, FacingPick)
    assert pick.facing_bet is True
    assert pick.faced_size_bb == pytest.approx(3.0)
    assert pick.size_frac == pytest.approx(0.5, abs=1e-6)
    assert pick.aggressor_id == "BTN"

def test_facing_false_if_hero_acted_after_aggressor(monkeypatch):
    # BTN bets (tick 10); HERO acts after (tick 11) → not facing
    events = [
        DummyAR("BTN","BTN","BET", 1,10,amount_bb=2.0),
        DummyAR("HERO","BB","CALL",1,11),
    ]
    monkeypatch.setattr(
        "ml.inference.policy.facing_resolver.ActionInferrer",
        lambda: DummyInferrer(events)
    )
    req = mk_req(street=1, pot_bb=6.0)
    pick = FacingBetResolver().resolve(req)
    assert pick.facing_bet is False
    assert pick.reason in ("hero_already_acted_after_aggressor","hero_acted_after_aggressor")

def test_facing_false_when_no_aggressor(monkeypatch):
    events = [
        DummyAR("CO","CO","CHECK",1,1),
        DummyAR("BTN","BTN","CHECK",1,2),
    ]
    monkeypatch.setattr(
        "ml.inference.policy.facing_resolver.ActionInferrer",
        lambda: DummyInferrer(events)
    )
    req = mk_req(street=1, pot_bb=6.0)
    pick = FacingBetResolver().resolve(req)
    assert pick.facing_bet is False
    assert pick.faced_size_bb is None
    assert pick.size_frac is None

def test_amount_falls_back_to_stack_delta(monkeypatch):
    # No amount_bb on action; use stack_stream delta (negative when chips go in)
    events = [
        DummyAR("BTN","BTN","BET",1,5,amount_bb=None),
    ]
    monkeypatch.setattr(
        "ml.inference.policy.facing_resolver.ActionInferrer",
        lambda: DummyInferrer(events)
    )

    # At tick 5, BTN stack goes -2.5bb → amount inferred as +2.5
    Stack = types.SimpleNamespace  # mimics StackChangeEvent
    stack_stream = [
        Stack(tick=5, when_ms=None, street=1, player_id="BTN", seat_label="BTN",
              stack_before_bb=100.0, stack_after_bb=97.5, delta_bb=-2.5,
              source="derived", conf=None)
    ]

    req = mk_req(street=1, pot_bb=5.0, stack_stream=stack_stream)
    pick = FacingBetResolver().resolve(req)
    assert pick.facing_bet is True
    assert pick.faced_size_bb == pytest.approx(2.5)

def test_pot_fraction_from_pot_stream_prior(monkeypatch):
    # Use pot_stream “prior” event to compute fraction
    events = [ DummyAR("SB","SB","BET",1,12,amount_bb=1.5) ]
    monkeypatch.setattr(
        "ml.inference.policy.facing_resolver.ActionInferrer",
        lambda: DummyInferrer(events)
    )

    Pot = types.SimpleNamespace  # mimics PotChangeEvent
    pot_stream = [
        Pot(tick=8, when_ms=None, street=1, pot_before_bb=2.0, pot_after_bb=4.0, delta_bb=2.0, source="derived"),
        Pot(tick=11, when_ms=None, street=1, pot_before_bb=4.0, pot_after_bb=6.0, delta_bb=2.0, source="derived"),
        # bet at tick 12 → pot_before should be 6.0
    ]

    req = mk_req(street=1, pot_bb=0.0, pot_stream=pot_stream)
    pick = FacingBetResolver().resolve(req)
    assert pick.facing_bet is True
    assert pick.size_frac == pytest.approx(1.5 / 6.0, abs=1e-6)