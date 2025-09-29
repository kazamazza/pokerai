import gzip, json, os
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from ml.etl.rangenet.postflop.solver_policy_parser import PolicyParseConfig, SolverPolicyParser
from ml.models.policy_consts import ACTION_VOCAB

def _load_json(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def _first_existing(*paths: str) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# ---- Real SRP HU CALLER_OOP (or whatever you placed at data/raw/real.json) ----
_REAL_SRP_PATH = _first_existing(
    "data/raw/real.json",
    "data/raw/real_solve_srp_caller_oop.json",
    "data/raw/real_solve_srp_caller_oop.json.gz",
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")  # keep output clean

@pytest.mark.skipif(
    _REAL_SRP_PATH is None,
    reason="Put a real SRP HU CALLER_OOP flop file at data/raw/real.json (or *_caller_oop.json[.gz])",
)
def test_real_srp_caller_oop_has_raises():
    payload = _load_json(_REAL_SRP_PATH)

    # Adjust these two numbers to match the file you downloaded if needed.
    cfg = PolicyParseConfig(
        pot_bb=6.0,           # typical SRP flop pot ~6bb; set to your file’s pot
        stack_bb=100.0,       # set to effective stack of the solve
        role="CALLER_OOP",    # from bet_sizing_id suffix: srp_hu.CALLER_OOP
    )

    out = SolverPolicyParser().parse(payload, cfg)
    assert out.ok, f"parse failed: {out.diag}"

    idx = {a: i for i, a in enumerate(ACTION_VOCAB)}
    v = out.vec

    # Expect at least one of CALL/FOLD when facing c-bet
    assert v[idx["CALL"]] > 0.0 or v[idx["FOLD"]] > 0.0, "Expected at least CALL or FOLD mass"

    # Raises are common in many SRP caller spots; if your specific file has none, this will show you.
    raise_mass = sum(v[idx[a]] for a in ("RAISE_150","RAISE_200","RAISE_300","ALLIN") if a in idx)
    assert raise_mass > 0.0, "Expected some raise mass for OOP vs IP cbet in SRP (choose a file with raises)."

# ---- Limped single SB_IP: allowed to have no raises; we just require a valid normalized vector ----
_REAL_LIMP_PATH = _first_existing(
    "data/raw/real_solve_limped_sb_ip.json",
    "data/raw/real_solve_limped_sb_ip.json.gz",
)

@pytest.mark.skipif(
    _REAL_LIMP_PATH is None,
    reason="Optionally put a limped_single.SB_IP flop file at data/raw/real_solve_limped_sb_ip.json[.gz]",
)
def test_real_limped_sb_ip_may_have_no_raises_and_should_not_crash():
    payload = _load_json(_REAL_LIMP_PATH)

    cfg = PolicyParseConfig(
        pot_bb=2.0,           # typical limp pot at flop
        stack_bb=150.0,       # match your file
        role="SB_IP",         # from bet_sizing_id suffix: limped_single.SB_IP
    )

    out = SolverPolicyParser().parse(payload, cfg)
    assert out.ok, f"parse failed: {out.diag}"
    s = float(sum(out.vec))
    assert abs(s - 1.0) < 1e-6, f"Vector not normalized: sum={s}"