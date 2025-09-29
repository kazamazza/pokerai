# tools/policy/smoke_test.py

from typing import Any, Dict
import numpy as np

from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.policy import PolicyInfer


# --- import your real classes ---
# If your project paths differ, tweak these imports accordingly.


# ---------- Minimal stubs ----------

class RangePreStub:
    """Returns a uniform 169-vector so preflop path can run."""
    def __init__(self):
        # optional mapping; ok to omit for smoke test
        self.hand_to_id = {}

    def predict_proba(self, rows):
        # rows is a list of dicts; return [1,169] np array
        out = np.ones((1, 169), dtype=np.float32) / 169.0
        return out

class PopStub:
    """Population prior: 1/3 each for fold/call/raise."""
    def predict(self, req: Dict[str, Any]) -> Dict[str, float]:
        return {"p_fold": 1/3, "p_call": 1/3, "p_raise": 1/3}

# ---------- Build the policy infer with stubs ----------
deps = PolicyInferDeps(
    pop=PopStub(),
    exploit=None,            # neutral mix will be used
    equity=None,             # neutral equity will be used
    range_pre=RangePreStub(),
    policy_post=None,        # postflop falls back to stub probs
    clusterer=None,
    params={
        # optional guardrail/config knobs; safe defaults
        "exploit_full_weight": 200,
        "min_prob": 1e-4,
    },
)

pi = PolicyInfer(deps)

# ---------- PRE-FLOP smoke ----------
pre_req = {
    "street": 0,
    "stack_bb": 100,
    "hero_pos": "BB",
    "opener_pos": "BTN",
    "opener_action": "RAISE",
    "ctx": "SRP",
    # no hero_hand -> avoids hand_to_169_label path, fine for smoke
}
pre_out = pi.predict(pre_req)
print("\n=== PRE-FLOP ===")
print("actions:", pre_out["actions"])
print("probs:  ", [round(x, 4) for x in pre_out["probs"]])
print("evs:    ", [round(x, 4) for x in pre_out["evs"]])
print("debug keys:", list(pre_out["debug"].keys()))

# ---------- POST-FLOP smoke (flop) ----------
post_req = {
    "street": 1,                 # flop
    "pot_bb": 6.0,
    "eff_stack_bb": 100.0,
    "facing_bet": False,         # change to True to test facing-bet legality
    "villain_actor": "Bob",
    "position_by_player": {"Bob": {"name": "BTN", "player_id": "p2"}},
    # "board": "AsKd2c" (optional; not needed for smoke)
}
post_out = pi.predict(post_req)
print("\n=== POST-FLOP ===")
print("actions:", post_out["actions"])
print("probs:  ", [round(x, 4) for x in post_out["probs"]])
print("evs:    ", [round(x, 4) for x in post_out["evs"]])
print("debug keys:", list(post_out["debug"].keys()))