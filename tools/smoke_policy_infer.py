from __future__ import annotations
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from ml.inference.policy import PolicyInfer, PolicyInferDeps

# --- (1) monkeypatch the postflop ACTION_VOCAB module the PolicyInfer expects ---
# Your PolicyInfer calls: from ml.models.postflop_policy_net import ACTION_VOCAB
fake_mod = types.ModuleType("ml.models.postflop_policy_net")
fake_mod.ACTION_VOCAB = [
    "FOLD","CHECK","CALL",
    "BET_33","BET_66","BET_100",
    "RAISE_150","RAISE_200","RAISE_300",
    "ALLIN",
]
sys.modules["ml"] = types.ModuleType("ml")
sys.modules["ml.models"] = types.ModuleType("ml.models")
sys.modules["ml.models.postflop_policy_net"] = fake_mod

# --- (2) tiny helpers used by PolicyInfer (hand encoding) ---
def hand_to_169(h: str) -> int:
    # ultra-dumb encoding: bucket AKs -> 0, otherwise 1
    return 0 if h in ("AsKs","AsKd","AhKh","AdKd","AcKc") else 1

def hand_to_id(h: str) -> int:
    # pretend card-id space; not used in this smoke
    return 123

# --- (3) mock dependency inferencers ----------------------------------------
class MockRangePreflopInfer:
    def predict(self, rows: Sequence[Mapping[str, Any]]) -> List[List[float]]:
        out = []
        for r in rows:
            v = np.ones(169, dtype=np.float32) / 169.0
            # give a tiny bump to AKs bucket
            v[0] = 0.10
            v = v / v.sum()
            out.append(v.tolist())
        return out

class MockPostflopPolicyInfer:
    def predict_proba(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        # produce a simple distribution biased by actor (more CHECK for oop)
        V = len(fake_mod.ACTION_VOCAB)
        p = np.ones(V, dtype=np.float32)
        act = str(rows[0].get("actor","ip")).lower()
        if act == "ip":
            # IP: more bets
            for i, a in enumerate(fake_mod.ACTION_VOCAB):
                if a.startswith("BET"): p[i] += 2.0
        else:
            # OOP: more check/call
            for i, a in enumerate(fake_mod.ACTION_VOCAB):
                if a in ("CHECK","CALL"): p[i] += 2.0
        p = p / p.sum()
        return np.stack([p], axis=0)

class MockEquityNetInfer:
    def predict(self, **kwargs) -> Tuple[float,float,float]:
        street = int(kwargs.get("street", 0))
        if street == 0:
            return (0.45, 0.05, 0.50)  # preflop: meh
        else:
            return (0.60, 0.05, 0.35)  # postflop: decent equity

class MockExploitNetInfer:
    def predict(self, feat=None, context=None) -> Tuple[float,float,float,float]:
        # fold/call/raise mix + weight (sample size)
        return (0.30, 0.50, 0.20, 150.0)

class MockPopulationNetInfer:
    def predict(self, **kwargs) -> Tuple[float,float,float]:
        return (1/3, 1/3, 1/3)

class MockBoardClusterer:
    def predict(self, boards: Sequence[str]) -> List[int]:
        return [42 for _ in boards]

# --- (4) bring in your PolicyInfer (import from your codebase if available) ---
# If your class lives at ml.infer.policy_infer import it instead of redefining.
# Below we import the class as you integrated it earlier.


# ---- wire in the fallback helpers if your PolicyInfer expects them at module level
# (Only needed if your PolicyInfer file uses these names without importing.)
policy_infer_module = sys.modules.get("policy_infer")
if policy_infer_module:
    if not hasattr(policy_infer_module, "hand_to_169"):
        setattr(policy_infer_module, "hand_to_169", hand_to_169)
    if not hasattr(policy_infer_module, "hand_to_id"):
        setattr(policy_infer_module, "hand_to_id", hand_to_id)

# --- (5) build the PolicyInfer with mocks ------------------------------------
def build_policy_infer() -> PolicyInfer:
    deps = PolicyInferDeps(
        pop=MockPopulationNetInfer(),
        exploit=MockExploitNetInfer(),
        equity=MockEquityNetInfer(),
        range_pre=MockRangePreflopInfer(),
        policy_post=MockPostflopPolicyInfer(),
        clusterer=MockBoardClusterer(),
        params=dict(alpha=0.35, beta=0.35, gamma=0.30, exploit_full_weight=200, spr_shove_threshold=1.5),
    )
    return PolicyInfer(deps)

# --- (6) minimal requests -----------------------------------------------------
def preflop_req() -> Dict[str, Any]:
    return {
        "street": 0,
        "stack_bb": 100.0,
        "pot_bb": 1.5,
        "hero_pos": "CO",
        "opener_pos": "UTG",
        "opener_action": "RAISE",
        "ctx": "6max",
        "stakes_id": "nl50",
        "villain_pos": "UTG",
        "hero_hand": "AsKd",
    }

def postflop_req() -> Dict[str, Any]:
    return {
        "street": 1,
        "stack_bb": 100.0,
        "eff_stack_bb": 80.0,
        "pot_bb": 6.5,
        "hero_pos": "BTN",
        "ip_pos": "BTN",
        "oop_pos": "BB",
        "actor": "ip",
        "ctx": "SRP_IP",
        "stakes_id": "nl50",
        "villain_pos": "BB",
        "hero_hand": "AsKd",
        # Optional if your policy uses it:
        "board_mask_52": [0.0]*52,
    }

# --- (7) run ------------------------------------------------------------------
def main():
    pi = build_policy_infer()

    print("=== PREFLOP ===")
    out_pre = pi.predict(preflop_req())
    print("actions:", out_pre["actions"])
    print("probs  :", [round(p,3) for p in out_pre["probs"]])
    print("evs    :", [round(x,3) for x in out_pre["evs"]])
    print("debug.equity:", out_pre["debug"]["equity"])

    print("\n=== POSTFLOP ===")
    out_post = pi.predict(postflop_req())
    print("actions:", out_post["actions"][:8], "...")
    print("probs  :", [round(p,3) for p in out_post["probs"][:8]], "...")
    print("evs    :", [round(x,3) for x in out_post["evs"][:8]], "...")
    print("debug.equity:", out_post["debug"]["equity"])

if __name__ == "__main__":
    main()