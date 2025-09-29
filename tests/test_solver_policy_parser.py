import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


from ml.etl.rangenet.postflop.solver_policy_parser import (
    SolverPolicyParser, PolicyParseConfig, ACTION_VOCAB
)

def load_payload(path):
    with open(path, "r") as f:
        return json.load(f)

def test_ip_root_bets(tmp_path):
    payload = {
        # 5.0 BB on a 10 BB pot = 50% pot → buckets to BET_50
        "actions": ["CHECK", "BET 5.0"],
        "strategy": {"strategy": {"hand1":[0.3,0.7], "hand2":[0.3,0.7]}},
    }
    cfg = PolicyParseConfig(pot_bb=10.0, stack_bb=100.0, role="PFR_IP")
    out = SolverPolicyParser().parse(payload, cfg)
    assert out.ok
    assert out.vec[ACTION_VOCAB.index("BET_50")] > 0

def test_oop_vs_bet_with_raise():
    payload = {
        "actions": ["BET 4.0"],
        "strategy": {"strategy": {"h":[1.0]}},
        "childrens": {
            "BET 4.0": {
                "actions": ["CALL","RAISE 8.0","FOLD"],
                "strategy": {"strategy": {"h":[0.5,0.3,0.2]}},
                "childrens": {}
            }
        }
    }
    cfg = PolicyParseConfig(pot_bb=10.0, stack_bb=100.0, role="CALLER_OOP")
    out = SolverPolicyParser().parse(payload, cfg)
    assert out.ok
    assert out.vec[ACTION_VOCAB.index("RAISE_200")] > 0  # 8/4 = 2.0
    assert out.vec[ACTION_VOCAB.index("CALL")] > 0
    assert out.vec[ACTION_VOCAB.index("FOLD")] > 0

def test_oop_after_check_donk():
    payload = {
        "actions": ["CHECK"],
        "strategy": {"strategy": {"h":[1.0]}},
        "childrens": {
            "CHECK": {
                "actions": ["CHECK","BET 3.0"],
                "strategy": {"strategy": {"h":[0.6,0.4]}},
                "childrens": {}
            }
        }
    }
    cfg = PolicyParseConfig(pot_bb=12.0, stack_bb=100.0, role="CALLER_OOP")
    out = SolverPolicyParser().parse(payload, cfg)
    assert out.ok
    assert out.vec[ACTION_VOCAB.index("CHECK")] > 0
    assert out.vec[ACTION_VOCAB.index("BET_25")] > 0  # 3/12 = 25%