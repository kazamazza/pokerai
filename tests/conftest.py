# file: tests/conftest.py
import math
from types import SimpleNamespace as NS
import pytest

from ml.inference.exploit.infer_actions import ActionInferrer
from ml.inference.policy.types import PolicyRequest

# Import your implementation module name here:
# If your module is different, adjust this import.
EPS = 1e-6

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
    # Use the real PolicyRequest so req.pot_bb defaults are honored
    return PolicyRequest(
        street=int(street),
        stack_stream=list(stack_stream or []),
        pot_stream=list(pot_stream or []),
        street_transitions=list(street_transitions or []),
        hero_id=hero_id,
        hand_id=hand_id,
        pot_bb=float(pot_bb),
    )

def assert_ar(ev, *, pid, action, street, tick, raise_level=None, contrib=None, prior=None, faced=None, size_frac=None):
    assert ev.player_id == pid
    assert ev.action == action
    assert int(ev.street) == street
    assert int(ev.tick) == tick
    if raise_level is not None:
        assert getattr(ev, "raise_level", None) == raise_level
    if contrib is not None:
        assert pytest.approx(getattr(ev, "contrib_bb", 0.0), rel=1e-6) == pytest.approx(contrib, rel=1e-6)
        # mirror field invariant
        assert pytest.approx(getattr(ev, "amount_bb", 0.0), rel=1e-6) == pytest.approx(contrib, rel=1e-6)
    if prior is not None:
        assert pytest.approx(getattr(ev, "prior_bet_bb", 0.0), rel=1e-6) == pytest.approx(prior, rel=1e-6)
    if faced is not None:
        assert bool(getattr(ev, "faced_bet", False)) is bool(faced)
    if size_frac is not None:
        if size_frac is None:
            assert getattr(ev, "size_frac") is None
        else:
            assert ev.size_frac is not None
            assert pytest.approx(ev.size_frac, rel=1e-6) == pytest.approx(size_frac, rel=1e-6)

def pick(ev_list, *, pid=None, tick=None, action=None):
    out = [e for e in ev_list if (pid is None or e.player_id == pid)
                              and (tick is None or int(e.tick) == int(tick))
                              and (action is None or e.action == action)]
    assert out, f"no action matched pid={pid} tick={tick} action={action}"
    return out[0]

@pytest.fixture
def inf():
    return ActionInferrer()