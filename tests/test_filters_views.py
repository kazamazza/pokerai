# file: tests/test_filters_views.py
import pytest
from tests.conftest import mk_stack, mk_pot, mk_req, pick, assert_ar

def test_exclude_hero_applies_to_generated_and_real(inf):
    stacks = [
        mk_stack(tick=5, street=0, pid="HERO", seat="BB",  before=100, after=97),  # hero contributed
        mk_stack(tick=6, street=0, pid="BTN",  seat="BTN", before=100, after=98),  # villain contributed
        mk_stack(tick=7, street=0, pid="CO",   seat="CO",  before=100, after=100), # seen → may fold/check
    ]
    pots = [
        mk_pot(tick=4, street=0, before=1.5, after=1.5),
        mk_pot(tick=7, street=0, before=1.5, after=4.5),
    ]
    req = mk_req(street=0, stack_stream=stacks, pot_stream=pots, hero_id="HERO")
    ev = inf.infer(req, exclude_hero=True)
    assert all(e.player_id != "HERO" for e in ev)
    assert any(e.player_id == "BTN" for e in ev)

def test_target_player_only_returns_that_player(inf):
    stacks = [
        mk_stack(tick=5, street=0, pid="HERO", seat="BB",  before=100, after=97),
        mk_stack(tick=6, street=0, pid="BTN",  seat="BTN", before=100, after=98),
    ]
    pots = [
        mk_pot(tick=4, street=0, before=1.5, after=1.5),
        mk_pot(tick=7, street=0, before=1.5, after=4.5),
    ]
    req = mk_req(street=0, stack_stream=stacks, pot_stream=pots, hero_id="HERO")
    ev = inf.infer(req, exclude_hero=False, target_player_id="BTN")
    assert all(e.player_id == "BTN" for e in ev)