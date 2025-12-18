import types
import pytest

# Adjust these imports if your paths differ
from ml.inference.policy.villain_resolver import VillainResolver, VillainPick

# --- Minimal ActionRecord test double ---
class DummyAR:
    __slots__ = ("player_id", "seat_label", "action", "street", "tick", "when_ms")
    def __init__(self, player_id, seat_label, action, street, tick, when_ms=None):
        self.player_id = player_id
        self.seat_label = seat_label
        self.action = action  # "BET","RAISE","CALL","CHECK","FOLD","ALLIN"
        self.street = street  # 0 pre, 1 flop, 2 turn, 3 river
        self.tick = tick
        self.when_ms = when_ms

# --- Helper: fake ActionInferrer for monkeypatching ---
class DummyInferrer:
    def __init__(self, events):
        self._events = events
    def infer(self, req, exclude_hero=True, target_player_id=None):
        # ignore args; return the scripted timeline
        return list(self._events)

# --- Helper: tiny req object for tests ---
def mk_req(*, hero_id="HERO", street=1, facing_bet=False):
    req = types.SimpleNamespace()
    req.hero_id = hero_id
    req.street = street
    req.facing_bet = facing_bet
    # streams are not used by VillainResolver directly; they’re used by ActionInferrer,
    # which we are stubbing out anyway.
    req.stack_stream = []
    req.pot_stream = []
    return req

def test_villain_resolver_facing_bet_last_aggressor(monkeypatch):
    # On flop: BTN bets last → we’re facing → villain=BTN
    events = [
        DummyAR("BTN", "BTN", "BET", 1, 10),
        DummyAR("CO",  "CO",  "FOLD", 1, 11),
    ]
    def _factory():
        return DummyInferrer(events)

    # Patch the ActionInferrer used inside VillainResolver
    monkeypatch.setattr(
        "ml.inference.policy.villain_resolver.ActionInferrer",
        lambda: _factory()
    )

    req = mk_req(street=1, facing_bet=True)
    pick = VillainResolver().resolve(req)
    assert isinstance(pick, VillainPick)
    assert pick.villain_id == "BTN"
    assert pick.reason == "facing_bet_last_aggressor"
    assert pick.confidence == pytest.approx(1.0)

def test_villain_resolver_heads_up_only_opponent(monkeypatch):
    # Only one non-hero remains active → pick them
    events = [
        DummyAR("HERO", "BB",  "CHECK", 1, 1),
        DummyAR("V1",   "BTN", "CHECK", 1, 2),
        DummyAR("V2",   "SB",  "FOLD",  1, 3),
    ]
    monkeypatch.setattr(
        "ml.inference.policy.villain_resolver.ActionInferrer",
        lambda: DummyInferrer(events)
    )
    req = mk_req(street=1, facing_bet=False)
    pick = VillainResolver().resolve(req)
    assert pick.villain_id == "V1"
    assert pick.reason == "heads_up_only_opponent"
    assert pick.confidence == pytest.approx(0.85)

def test_villain_resolver_last_street_aggressor(monkeypatch):
    # No facing now; last aggressor on flop was CO → choose CO
    events = [
        DummyAR("UTG", "UTG", "CHECK", 1, 1),
        DummyAR("CO",  "CO",  "BET",   1, 2),
        DummyAR("UTG", "UTG", "CALL",  1, 3),
        DummyAR("HERO","BTN", "CHECK", 2, 4),
        DummyAR("CO",  "CO",  "CHECK", 2, 5),
    ]
    monkeypatch.setattr(
        "ml.inference.policy.villain_resolver.ActionInferrer",
        lambda: DummyInferrer(events)
    )
    req = mk_req(street=2, facing_bet=False)
    pick = VillainResolver().resolve(req)
    assert pick.villain_id == "CO"
    assert pick.reason == "last_street_aggressor"

def test_villain_resolver_recent_actor_fallback(monkeypatch):
    # No aggressor anywhere; pick most recent non-hero actor this street
    events = [
        DummyAR("HERO", "BB",  "CHECK", 1, 1),
        DummyAR("MP",   "MP",  "CHECK", 1, 2),
        DummyAR("CO",   "CO",  "CHECK", 1, 3),
    ]
    monkeypatch.setattr(
        "ml.inference.policy.villain_resolver.ActionInferrer",
        lambda: DummyInferrer(events)
    )
    req = mk_req(street=1, facing_bet=False)
    pick = VillainResolver().resolve(req)
    assert pick.villain_id == "CO"
    assert pick.reason == "recent_actor_current_street"

def test_villain_resolver_no_candidate(monkeypatch):
    # Everyone folded earlier; no candidate non-hero active
    events = [
        DummyAR("V1", "SB",   "FOLD", 1, 1),
        DummyAR("V2", "CO",   "FOLD", 1, 2),
        DummyAR("HERO","BTN", "CHECK", 1, 3),
    ]
    monkeypatch.setattr(
        "ml.inference.policy.villain_resolver.ActionInferrer",
        lambda: DummyInferrer(events)
    )
    req = mk_req(street=1, facing_bet=False)
    pick = VillainResolver().resolve(req)
    assert pick.villain_id is None
    assert pick.reason == "no_candidate"