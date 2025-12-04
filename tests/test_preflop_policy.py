import pytest
from ml.inference.policy.policy import PolicyInfer
from ml.inference.policy.policy_factory import PolicyInferFactory
from ml.inference.policy.types import PolicyRequest


@pytest.fixture
def infer():
    policy_engine = PolicyInferFactory().create()
    return policy_engine

def test_predict_preflop_basic(infer):
    req = PolicyRequest(
        street=0,
        hero_hand="AhKh",
        hero_pos="BB",
        villain_pos="UTG",
        eff_stack_bb=25,
        pot_bb=2.5,
        facing_bet=True,
        faced_size_frac=0.1,
    )
    res = infer._predict_preflop(req)

    assert res is not None
    assert res.actions
    assert res.probs
    assert abs(sum(res.probs) - 1.0) < 1e-5
    assert all(p >= 0 for p in res.probs)
    assert len(res.actions) == len(res.probs) == len(res.evs)
    assert res.best_action in res.actions
    assert res.debug.get("hero_range") is not None
    assert isinstance(res.debug["hero_range"], list)
    assert len(res.debug["hero_range"]) == 169

def test_predict_preflop_srp_open(infer):
    req = PolicyRequest(
        street=0,
        hero_hand="QsJh",
        hero_pos="BTN",
        villain_pos="None",
        eff_stack_bb=40,
        pot_bb=0,
        facing_bet=False,
    )
    res = infer._predict_preflop(req)

    assert res is not None
    assert "OPEN_200" in res.actions or "OPEN_300" in res.actions
    assert abs(sum(res.probs) - 1.0) < 1e-5

def test_predict_preflop_short_stack(infer):
    req = PolicyRequest(
        street=0,
        hero_hand="9c9d",
        hero_pos="SB",
        villain_pos="BTN",
        eff_stack_bb=5,
        pot_bb=1.0,
        facing_bet=True,
        faced_size_frac=0.4,
    )
    res = infer._predict_preflop(req)

    assert res is not None
    assert any("RAISE" in a or "CALL" in a for a in res.actions)
    assert abs(sum(res.probs) - 1.0) < 1e-5

def test_predict_preflop_deep_stack(infer):
    req = PolicyRequest(
        street=0,
        hero_hand="AsQs",
        hero_pos="CO",
        villain_pos="MP",
        eff_stack_bb=300,
        pot_bb=4.5,
        facing_bet=True,
        faced_size_frac=0.15,
    )
    res = infer._predict_preflop(req)

    assert res is not None
    assert len(res.actions) >= 2
    assert abs(sum(res.probs) - 1.0) < 1e-5

def test_predict_preflop_limped_pot(infer):
    req = PolicyRequest(
        street=0,
        hero_hand="Td9d",
        hero_pos="BB",
        villain_pos="SB",
        eff_stack_bb=100,
        pot_bb=1.5,
        facing_bet=True,
        faced_size_frac=0.02,  # Limp-sized
    )
    res = infer._predict_preflop(req)

    assert res is not None
    assert "RAISE_200" in res.actions or "CALL" in res.actions
    assert abs(sum(res.probs) - 1.0) < 1e-5