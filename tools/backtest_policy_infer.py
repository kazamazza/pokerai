# tools/backtest_policy_infer.py
import sys
import time, random, json, math
from pathlib import Path
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
    board: str | None,
    pot_bb: float,
    eff_stack_bb: float,
    facing_bet: bool,
    villain_id: str | None,
    hero_hand: str | None = None,
    ctx: str = "VS_OPEN",
    equity_nudge: float | None = None,  # optional hint for smoke
) -> dict:

    raw = {
        "ctx": str(ctx).upper(),
        "facing_open": bool(facing_bet),
    }
    if equity_nudge is not None:
        raw["equity_nudge"] = float(equity_nudge)

    return {
        "street": int(street),
        "board": board,
        "pot_bb": float(pot_bb),
        "eff_stack_bb": float(eff_stack_bb),
        "facing_bet": bool(facing_bet),
        "villain_id": (str(villain_id) if villain_id is not None else None),
        "hero_hand": (str(hero_hand) if hero_hand else None),
        "raw": raw,
    }

# ---- random board helper ----
def random_board(n_cards: int) -> str:
    R, S = list("AKQJT98765432"), list("shdc")
    seen, out = set(), []
    while len(out) < n_cards:
        c = random.choice(R) + random.choice(S)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return "".join(out)

# ---- consistent random request (1 sample) ----
def make_consistent_req(street: int) -> dict:
    # plausible HU pairs
    pairs = [("BTN","BB"), ("CO","BTN"), ("SB","BB"), ("HJ","BTN")]
    actor = random.choice(["ip","oop"])
    facing = random.choice([False, True])

    if street == 0:
        # preflop: no board
        return make_request(
            street=0,
            board=None,
            pot_bb=random.choice([2.5, 3.0, 6.0]),
            eff_stack_bb=random.choice([40.0, 80.0, 160.0]),
            facing_bet=facing,
            villain_id=random.choice([None, "vill_A", "vill_B"]),
            hero_hand=random.choice(["AsAh", "KcKd", "AdQs", None]),
            ctx="VS_OPEN",
            equity_nudge=(0.08 if random.random() < 0.25 else None),
        )
    else:
        n_board = 3 if street == 1 else (4 if street == 2 else 5)
        return make_request(
            street=street,
            board=random_board(n_board),
            pot_bb=random.choice([6.0, 12.0, 24.0, 48.0]),
            eff_stack_bb=random.choice([40.0, 80.0, 160.0]),
            facing_bet=facing,
            villain_id=random.choice([None, "vill_A", "vill_B"]),
            ctx="VS_OPEN",
        )

# ---- batch sampler ----
def sample_requests(n: int = 1000, *, include_preflop: bool = True) -> list[dict]:
    reqs = []
    for _ in range(n):
        st = random.choice([0,1,2,3]) if include_preflop else random.choice([1,2,3])
        reqs.append(make_consistent_req(st))
    return reqs

# ---- legality helper mirror (useful for sanity stats) ----
def legal_mask(hero_side: str, facing_bet: bool) -> set[str]:
    if facing_bet:
        return {"FOLD","CALL","RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500","ALLIN"}
    legal = {"CHECK","BET_25","BET_33","BET_50","BET_66","BET_75","BET_100"}
    if hero_side == "oop":
        legal.add("DONK_33")
    return legal

def run_backtest(n=5000, exploit_probe=True, equity_probe=True):
    # --- load components ---
    postflop = load_dependency(PostflopPolicyInfer, POSTFLOP_DIR)
    preflop  = load_dependency(PreflopPolicy, PREFLOP_DIR)
    equity   = load_dependency(EquityNetInfer, EQUITY_DIR)
    pop      = load_dependency(PopulationNetInference, POP_DIR)
    clusterer = None
    try:
        clusterer = load_board_clusterer({"board_clustering":{"type":"kmeans","artifact":str(CLUSTER_JSON)}})
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

    # --- metrics ---
    t0 = time.time()
    reqs = sample_requests(n=n, include_preflop=True)
    n_ok = 0
    n_illegal = 0
    n_nan = 0
    cluster_used = []
    exploit_delta_hits = 0
    equity_shifts = 0
    lat_sum = 0.0

    # optional: pre-seed some exploit observations to make sure signal fires
    if exploit_probe:
        # Pick a scenario_key consistent with PolicyInfer (_build_postflop_row-family).
        # We’ll bias "CALL" to be high for villain_B on FLOP BTNvsBB.
        player_id = "vill_B"
        scen = "1:VS_OPEN:BTN:BB"  # street:ctx:ip:oop
        CALL_IDX = ACTIONS.index("CALL") if "CALL" in ACTIONS else None
        if CALL_IDX is not None:
            for _ in range(50):
                expl.observe(player_id, scen, CALL_IDX, weight=1.0)

    for r in reqs:
        # A/B for exploit
        if exploit_probe and r["street"] in (1,2,3) and r["villain_id"]:
            # snapshot without exploit (temporarily zero out lambda_expl)
            old_lambda = infer.blend.lambda_expl
            infer.blend.lambda_expl = 0.0
            p0 = infer.predict(r).probs
            infer.blend.lambda_expl = old_lambda

        ts = time.time()
        out = infer.predict(r)
        lat_sum += (time.time() - ts)

        # sanity
        probs = np.asarray(out.probs, dtype=float)
        if not np.isfinite(probs).all():
            n_nan += 1
            continue
        if abs(probs.sum() - 1.0) > 1e-3:
            # normalize defensively for checking
            ps = probs / max(probs.sum(), 1e-8)
        else:
            ps = probs

        # legality check
        actor = (r.get("raw", {}).get("actor") or "ip").lower()
        facing = bool(r.get("facing_bet", False))
        legal = legal_mask(actor, facing)
        chosen = out.actions[int(ps.argmax())] if len(out.actions) == len(ps) else None
        if chosen and chosen in legal:
            n_ok += 1
        else:
            n_illegal += 1

        # cluster coverage
        if out.debug and "board_cluster_id" in out.debug and r["street"]>0:
            cluster_used.append(int(out.debug["board_cluster_id"] or 0))

        # probe exploit effect
        if exploit_probe and r["street"] in (1,2,3) and r["villain_id"]:
            p1 = np.asarray(out.probs, dtype=float)
            if len(p1) == len(p0) and np.max(np.abs(p1 - p0)) > 0.03:
                exploit_delta_hits += 1

        # --- preflop equity nudge probe (owned by PolicyInfer) ---
        if equity_probe and r["street"] == 0 and infer.eq is not None and r.get("hero_hand"):
            # baseline
            out0 = out  # already computed above via infer.predict(r)

            # ask PolicyInfer to apply a small equity nudge via raw hint
            r2 = dict(r)
            r2_raw = dict(r2.get("raw", {}))
            r2_raw["equity_nudge"] = 0.05  # <-- tiny tilt; PolicyInfer should honor this in preflop
            r2["raw"] = r2_raw

            out1 = infer.predict(r2)

            if len(out1.probs) == len(out0.probs):
                delta = np.max(np.abs(np.asarray(out1.probs) - np.asarray(out0.probs)))
                if delta > 0.02:
                    equity_shifts += 1

    dt = time.time() - t0
    qps = n / max(dt, 1e-6)
    lat_ms = (lat_sum / max(n,1)) * 1000.0

    report = {
        "n": n,
        "ok_rate": round(n_ok / max(n,1), 4),
        "illegal_rate": round(n_illegal / max(n,1), 4),
        "nan_rate": round(n_nan / max(n,1), 4),
        "throughput_qps": round(qps, 1),
        "avg_latency_ms": round(lat_ms, 2),
        "cluster_coverage": len(set(cluster_used)),
        "exploit_effect_hits": int(exploit_delta_hits),
        "equity_nudge_shifts": int(equity_shifts),
    }
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    run_backtest(n=10000, exploit_probe=True, equity_probe=True)