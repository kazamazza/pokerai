import pytest

from ml.infer.resolve.resolver import ResolvedStateResolver
from ml.infer.types.observed_request import ObservedRequest, StackChangeEvent
from ml.infer.types.resolved_state import ResolvedState
from ml.infer.policy_request_builder import PolicyRequestBuilder            # <- adjust path
from ml.core.contracts import PolicyRequest                                 # <- adjust path


def test_observed_to_policy_request_smoke_postflop_root():
    obs = ObservedRequest(
        stakes="NL10",
        street=1,  # flop
        hero_pos="BB",
        hero_id="HERO",
        board="AsKd7h",
        pot_bb=6.0,
        eff_stack_bb=100.0,
        hand_id="hand_1",
        raw={"note": "unit_test"},
        debug=True,
        stack_stream=[
            StackChangeEvent(
                tick=0, when_ms=None, street=0,
                player_id="HERO", seat_label="BB",
                stack_before_bb=100.0, stack_after_bb=100.0, delta_bb=0.0,
                source="derived", conf=1.0,
            ),
            StackChangeEvent(
                tick=1, when_ms=None, street=0,
                player_id="VILLAIN", seat_label="SB",  # ✅ key change
                stack_before_bb=100.0, stack_after_bb=100.0, delta_bb=0.0,
                source="derived", conf=1.0,
            ),
        ],
    )

    resolver = ResolvedStateResolver()
    st = resolver.resolve(obs)

    assert isinstance(st, ResolvedState)
    assert st.stakes == "NL10"
    assert st.street in (1, 2, 3)          # ✅ postflop only
    assert st.ip_pos is not None and st.oop_pos is not None
    assert st.hero_pos is not None

    builder = PolicyRequestBuilder()
    req = builder.build(st)

    assert isinstance(req, PolicyRequest)

    # Core invariants
    assert req.street == st.street
    assert isinstance(req.stakes_id, int)
    assert req.pot_bb > 0
    assert req.effective_stack_bb > 0

    allowed_pos = {"UTG", "HJ", "MP", "CO", "BTN", "SB", "BB"}
    assert req.ip_pos in allowed_pos
    assert req.oop_pos in allowed_pos

    # Root vs facing sizing invariants
    if getattr(st, "node_type", None) == "ROOT":
        assert req.faced_size_pct in (None, 0.0)
        # Only assert size_pct if your root model requires it:
        # assert req.size_pct is not None and req.size_pct > 0
    else:
        assert req.faced_size_pct is not None and req.faced_size_pct > 0

    assert isinstance(req.debug, dict)