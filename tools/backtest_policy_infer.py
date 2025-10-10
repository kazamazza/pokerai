# tools/backtest_policy_infer.py
import itertools
import sys
import time, random, json, math
from pathlib import Path
from typing import List
from dataclasses import replace
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from ml.features.boards import load_board_clusterer
from ml.inference.equity import EquityNetInfer
from ml.inference.player_exploit_store import PlayerExploitStore, ExploitConfig
from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.policy import PolicyInfer
from ml.inference.policy.policy_blend_config import PolicyBlendConfig
from ml.inference.population import PopulationNetInference
from ml.inference.postflop import PostflopPolicyInfer
from ml.inference.preflop import PreflopPolicy
from ml.inference.policy.types import PolicyRequest

# --- hard paths (edit if needed) ---
POSTFLOP_DIR = Path("checkpoints/postflop_policy")
PREFLOP_DIR  = Path("checkpoints/range_pre")
EQUITY_DIR   = Path("checkpoints/equitynet")
POP_DIR      = Path("checkpoints/popnet")
CLUSTER_JSON = Path("data/artifacts/board_clusters_kmeans_128.json")

ACTIONS = None  # filled from infer.action_vocab after load
BET_BUCKETS   = {"BET_25","BET_33","BET_50","BET_66","BET_75","BET_100","DONK_33"}
RAISE_BUCKETS = {"RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500","ALLIN"}

def load_dependency(cls, arg):
    try:
        return cls.from_dir(str(arg))
    except Exception as e:
        print(f"[warn] failed to load {arg}: {e}")
        return None

# ---- deterministic request builder (single source of truth) ----
def make_request(
    *,
    street: int,
    hero_pos: str,
    villain_pos: str,
    board: str | None,
    pot_bb: float,
    eff_stack_bb: float,
    facing_bet: bool,
    villain_id: str | None,
    hero_hand: str | None = None,
    ctx: str = "VS_OPEN",
    equity_nudge: float | None = None,
) -> PolicyRequest:
    raw = {"ctx": str(ctx).upper(), "facing_open": bool(facing_bet)}
    if equity_nudge is not None:
        raw["equity_nudge"] = float(equity_nudge)

    req = PolicyRequest(
        street=int(street),
        hero_pos=(hero_pos or "").upper(),
        villain_pos=(villain_pos or "").upper(),
        hero_hand=(str(hero_hand) if hero_hand else None),
        board=board,
        pot_bb=float(pot_bb),
        eff_stack_bb=float(eff_stack_bb),
        facing_bet=bool(facing_bet),
        villain_id=(str(villain_id) if villain_id is not None else None),
        actions_hist=None,
        raw=raw,
    )

    # Normalize & ensure legality at the source
    if hasattr(req, "legalize"):
        req = req.legalize()  # expected to return a (possibly new) PolicyRequest
    elif hasattr(req, "validate"):
        req.validate()
    return req


# ---- random board helper (no duplicates) ----
def random_board(n_cards: int) -> str:
    R, S = list("AKQJT98765432"), list("shdc")
    all_cards = [r + s for r, s in itertools.product(R, S)]
    board = random.sample(all_cards, n_cards)
    return "".join(board)


# ---- consistent random request (1 sample) ----
def make_consistent_req(street: int) -> PolicyRequest:
    # plausible HU seat pairs (hero on left, villain on right)
    pairs = [("BTN", "BB"), ("CO", "BTN"), ("SB", "BB"), ("HJ", "BTN")]
    hero_pos, villain_pos = random.choice(pairs)

    if street == 0:
        # Preflop: hero doesn't "face a bet" in this simple generator
        return make_request(
            street=0,
            hero_pos=hero_pos,
            villain_pos=villain_pos,
            board=None,
            pot_bb=random.choice([2.5, 3.0, 6.0]),
            eff_stack_bb=random.choice([40.0, 80.0, 160.0]),
            facing_bet=False,
            villain_id=random.choice([None, "vill_A", "vill_B"]),
            hero_hand=random.choice(["AsAh", "KcKd", "AdQs"]),
            ctx="VS_OPEN",
            equity_nudge=(0.08 if random.random() < 0.25 else None),
        )
    else:
        n_board = 3 if street == 1 else (4 if street == 2 else 5)
        return make_request(
            street=street,
            hero_pos=hero_pos,
            villain_pos=villain_pos,
            board=random_board(n_board),
            pot_bb=random.choice([6.0, 12.0, 24.0, 48.0]),
            eff_stack_bb=random.choice([40.0, 80.0, 160.0]),
            facing_bet=random.choice([False, True]),
            villain_id=random.choice([None, "vill_A", "vill_B"]),
            hero_hand=random.choice(["AsAh", "KcKd", "AdQs"]),
            ctx="VS_OPEN",
        )


# ---- batch sampler ----
def sample_requests(n: int = 1000, *, include_preflop: bool = True) -> List[PolicyRequest]:
    reqs: List[PolicyRequest] = []
    for _ in range(n):
        st = random.choice([0, 1, 2, 3]) if include_preflop else random.choice([1, 2, 3])
        reqs.append(make_consistent_req(st))
    return reqs


def run_backtest(n=1, exploit_probe=True, equity_probe=True):
    # --- Load model dependencies ---
    postflop = load_dependency(PostflopPolicyInfer, POSTFLOP_DIR)
    preflop  = load_dependency(PreflopPolicy, PREFLOP_DIR)
    equity   = load_dependency(EquityNetInfer, EQUITY_DIR)
    pop      = load_dependency(PopulationNetInference, POP_DIR)
    clusterer = None
    try:
        clusterer = load_board_clusterer({"board_clustering": {"type": "kmeans", "artifact": str(CLUSTER_JSON)}})
    except Exception as e:
        print("[warn] clusterer:", e)
    expl = PlayerExploitStore(ExploitConfig())

    deps = PolicyInferDeps(
        pop=pop,
        exploit=expl,
        equity=equity,
        range_pre=preflop,
        policy_post=postflop,
        clusterer=clusterer,
        params={}
    )
    infer = PolicyInfer(deps, blend_cfg=PolicyBlendConfig.default())
    global ACTIONS
    ACTIONS = infer.action_vocab

    # --- Metrics ---
    t0 = time.time()
    reqs = sample_requests(n=n, include_preflop=True)
    n_ok, n_nan = 0, 0
    cluster_used = []
    exploit_hits = 0
    equity_shifts = 0
    lat_sum = 0.0

    # --- Optional exploit pre-seed ---
    if exploit_probe:
        player_id = "vill_B"
        scen = "1:VS_OPEN:BTN:BB"
        CALL_IDX = ACTIONS.index("CALL") if "CALL" in ACTIONS else None
        if CALL_IDX is not None:
            for _ in range(50):
                expl.observe(player_id, scen, CALL_IDX, weight=1.0)

    # --- Main loop ---
    for req in reqs:  # req is already a PolicyRequest (legalized in make_request)
        # (Optional) re-legalize if you really want belt & braces:
        # req = req.legalize()

        # --- exploit A/B probe (postflop only) ---
        if exploit_probe and req.street in (1, 2, 3) and req.villain_id:
            old_lambda = infer.blend.lambda_expl
            infer.blend.lambda_expl = 0.0
            p0 = infer.predict(req).probs
            infer.blend.lambda_expl = old_lambda

        ts = time.time()
        out = infer.predict(req)
        dbg = out.debug or {}
        lat_sum += (time.time() - ts)

        probs = np.asarray(out.probs, dtype=float)
        if not np.isfinite(probs).all():
            n_nan += 1
            continue

        probs = probs / max(probs.sum(), 1e-8)
        n_ok += 1

        if out.debug and "board_cluster_id" in out.debug and req.street > 0:
            cluster_used.append(int(out.debug["board_cluster_id"] or 0))

        if exploit_probe and req.street in (1, 2, 3) and req.villain_id:
            p1 = np.asarray(out.probs, dtype=float)
            if len(p1) == len(p0) and np.max(np.abs(p1 - p0)) > 0.03:
                exploit_hits += 1

        if equity_probe and req.street == 0 and infer.eq is not None and req.hero_hand:
            out0 = out

            # clone & override nudge
            r2 = replace(req, raw=dict(req.raw))
            r2.raw["equity_nudge"] = 0.10  # bump for visibility in tests

            out1 = infer.predict(r2)

            if len(out1.probs) == len(out0.probs):
                p0 = np.asarray(out0.probs, dtype=float)
                p1 = np.asarray(out1.probs, dtype=float)
                delta = float(np.max(np.abs(p1 - p0)))
                # Also consider argmax flips as a “shift”
                flip = (int(p1.argmax()) != int(p0.argmax()))
                if delta > 0.01 or flip:
                    equity_shifts += 1

    # --- Summary metrics ---
    dt = time.time() - t0
    qps = n / max(dt, 1e-6)
    lat_ms = (lat_sum / max(n_ok, 1)) * 1000.0

    report = {
        "n": n,
        "ok_rate": round(n_ok / max(n, 1), 4),
        "nan_rate": round(n_nan / max(n, 1), 4),
        "throughput_qps": round(qps, 1),
        "avg_latency_ms": round(lat_ms, 2),
        "cluster_coverage": len(set(cluster_used)),
        "exploit_effect_hits": exploit_hits,
        "equity_nudge_shifts": equity_shifts,
    }
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    run_backtest(n=5000, exploit_probe=True, equity_probe=True)