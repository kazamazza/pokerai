import pytest

from ml.inference.preflop_legal_action_generator import PreflopLegalActionGenerator


@pytest.fixture
def gen():
    return PreflopLegalActionGenerator()

def test_unopened_actions(gen):
    actions = gen.generate(stack_bb=100, facing_bet=False)
    assert "FOLD" in actions
    assert any(a.startswith("OPEN_") for a in actions)
    assert "CALL" not in actions

def test_facing_actions(gen):
    actions = gen.generate(stack_bb=100, facing_bet=True, faced_frac=0.25)
    assert "FOLD" in actions
    assert "CALL" in actions
    assert any(a.startswith("RAISE_") for a in actions)

def test_short_stack_raises(gen):
    actions = gen.generate(stack_bb=20, facing_bet=True, faced_frac=0.5)
    raise_sizes = [int(a.split("_")[1]) for a in actions if a.startswith("RAISE_")]
    assert all(rs <= 40 for rs in raise_sizes)  # max 2x stack