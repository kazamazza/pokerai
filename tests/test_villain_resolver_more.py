# file: tests/test_villain_resolver_more.py
import types
import pytest

# Adjust import path if needed
from ml.inference.policy.villain_resolver import VillainResolver, VillainPick

class DummyAR:
    __slots__ = ("player_id", "seat_label", "action", "street", "tick", "when_ms")
    def __init__(self, player_id, seat_label, action, street, tick, when_ms=None):
        self.player_id = player_id
        self.seat_label = seat_label
        self.action = action
        self.street = int(street)
        self.tick = int(tick)
        self.when_ms = when_ms

class DummyInferrer:
    def __init__(self, events): self._events = list(events)
    def infer(self, req, exclude_hero=True, target_player_id=None):
        return list(self._events)

def mk_req(*, hero_id="HERO", street=1, facing_bet=False):
    req = types.SimpleNamespace()
    req.hero_id = hero_id
    req.street = int(street)
    req.facing_bet = bool(facing_bet)
    req.stack_stream = []
    req.pot_stream = []
    return req

def _patch(monkeypatch, events):
    monkeypatch.setattr(
        "ml.inference.policy.villain_resolver.ActionInferrer",
        lambda: DummyInferrer(events)
    )

def test_facing_allin_picks_shover(monkeypatch):
    # Flop: CO shoves, BTN calls → facing → villain must be CO (the shover)
    events = [
        DummyAR("CO",  "CO",  "ALLIN", 1, 10, 1000),
        DummyAR("BTN", "BTN", "CALL",  1, 11, 1010),
    ]
    _patch(monkeypatch, events)
    req = mk_req(street=1, facing_bet=True)
    pick = VillainResolver().resolve(req)
    assert isinstance(pick, VillainPick)
    assert pick.villain_id == "CO"
    assert pick.reason == "facing_bet_last_aggressor"

def test_facing_last_aggro_not_trailing_caller(monkeypatch):
    # Flop: BTN bets, CO calls → facing → pick BTN (last aggro), not CO
    events = [
        DummyAR("BTN", "BTN", "BET",  1, 10, 1000),
        DummyAR("CO",  "CO",  "CALL", 1, 11, 1010),
    ]
    _patch(monkeypatch, events)
    req = mk_req(street=1, facing_bet=True)
    pick = VillainResolver().resolve(req)
    assert pick.villain_id == "BTN"
    assert pick.reason == "facing_bet_last_aggressor"

def test_folded_aggro_is_not_candidate(monkeypatch):
    # Flop: CO bets then folds later (same street); BTN checks → not facing
    # Last street aggressor is CO but CO folded → resolver must skip CO
    events = [
        DummyAR("CO",  "CO",  "BET",  1, 10, 1000),
        DummyAR("CO",  "CO",  "FOLD", 1, 12, 1020),
        DummyAR("BTN", "BTN", "CHECK",1, 13, 1030),
    ]
    _patch(monkeypatch, events)
    req = mk_req(street=1, facing_bet=False)
    pick = VillainResolver().resolve(req)
    assert pick.villain_id == "BTN"
    assert pick.reason in ("recent_actor_current_street","heads_up_only_opponent")

def test_uses_seat_label_when_available(monkeypatch):
    events = [
        DummyAR("V1", "HJ", "BET", 1, 10),
        DummyAR("HERO", "BTN", "CALL", 1, 11),  # even if present, hero is excluded at return time
    ]
    _patch(monkeypatch, events)
    req = mk_req(street=1, facing_bet=True)
    pick = VillainResolver().resolve(req)
    assert pick.villain_id == "V1"
    assert pick.villain_pos == "HJ"

def test_hero_excluded_by_inferrer_still_ok(monkeypatch):
    # Simulate inferrer already excluded hero; only V1 events present
    events = [
        DummyAR("V1", "SB", "CHECK", 1, 5),
    ]
    _patch(monkeypatch, events)
    req = mk_req(street=1, facing_bet=False)
    pick = VillainResolver().resolve(req)
    assert pick.villain_id == "V1"
    assert pick.reason == "heads_up_only_opponent"

def test_equal_tick_when_ms_tie_prefers_last_aggro(monkeypatch):
    # Same tick & when_ms: order-insensitive; ensure aggro found walking backward
    events = [
        DummyAR("V1", "CO",  "BET",  1, 10, 1000),
        DummyAR("V2", "BTN", "CALL", 1, 10, 1000),
    ]
    _patch(monkeypatch, events)
    req = mk_req(street=1, facing_bet=True)
    pick = VillainResolver().resolve(req)
    assert pick.villain_id == "V1"