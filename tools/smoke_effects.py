import sys
from pathlib import Path

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



from dataclasses import replace
import numpy as np
from ml.inference.policy.types import PolicyRequest

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

def smoke_equity_effect(infer):
    print("\n--- SMOKE: equity effect ---")
    old_lambda_eq = getattr(infer.blend, "lambda_eq", 0.0)
    # amplify so we can visibly detect effect
    infer.blend.lambda_eq = 1.5

    base = PolicyRequest(
        street=1,  # flop
        hero_pos="BTN",
        villain_pos="BB",
        board="TdTs8s",  # paired/connected — good for mix
        pot_bb=12.0,
        eff_stack_bb=100.0,
        facing_bet=True,  # <-- not facing; allows CHECK + BET_*
        villain_id="vill_B",  # set to test exploit too (optional)
        hero_hand="3d8s",  # a middling hand so equity can swing either way
        raw={
            "ctx": "VS_OPEN",
            "bet_sizes": [0.33, 0.50, 0.66]  # <-- ensures multiple BET_* are legal
        }
    ).legalize()

    weak = replace(base, hero_hand="7c2d").legalize()

    out_strong = infer.predict(base)
    out_weak   = infer.predict(weak)

    p_strong = np.asarray(out_strong.probs, dtype=float)
    p_weak   = np.asarray(out_weak.probs, dtype=float)
    delta_L1 = float(np.abs(p_strong - p_weak).sum())

    top_s = out_strong.actions[int(p_strong.argmax())] if out_strong.actions else "NONE"
    top_w = out_weak.actions[int(p_weak.argmax())] if out_weak.actions else "NONE"

    print(" strong hand:", base.hero_hand, "=>", top_s)
    print("  weak hand:", weak.hero_hand, "=>", top_w)
    print(" L1 Δ(probs):", round(delta_L1, 4))
    print(" equity_debug (strong):", out_strong.debug.get("equity_debug"))

    infer.blend.lambda_eq = old_lambda_eq


def smoke_exploit_effect(infer):
    print("\n--- SMOKE: exploit effect ---")
    old_lambda = getattr(infer.blend, "lambda_expl", 0.0)
    infer.blend.lambda_expl = 2.0

    # seed exploit store for villain 'vill_B' on a canonical scenario
    villain = "vill_B"
    scen_key = "1:VS_OPEN:BTN:BB"
    try:
        call_idx = infer.action_vocab.index("CALL")
    except Exception:
        call_idx = None

    if call_idx is not None:
        # add lots of observations to make the effect clear
        for _ in range(80):
            infer.expl.observe(villain, scen_key, call_idx, weight=1.0)

    req = PolicyRequest(
        street=1,
        hero_pos="BTN",
        villain_pos="BB",
        board="Td9c2h",
        pot_bb=10.0,
        eff_stack_bb=100.0,
        facing_bet=False,
        villain_id=villain,
        hero_hand="AsKs",
        raw={"ctx": "VS_OPEN"}
    ).legalize()

    # A/B: with exploit disabled vs enabled
    infer.blend.lambda_expl = 0.0
    p0 = np.asarray(infer.predict(req).probs, dtype=float)

    infer.blend.lambda_expl = old_lambda or 2.0
    p1 = np.asarray(infer.predict(req).probs, dtype=float)

    delta_L1 = float(np.abs(p1 - p0).sum())
    top0 = infer.action_vocab[int(p0.argmax())] if len(p0) == len(infer.action_vocab) else "NONE"
    top1 = infer.action_vocab[int(p1.argmax())] if len(p1) == len(infer.action_vocab) else "NONE"

    print(" top w/out exploit:", top0)
    print(" top    with exploit:", top1)
    print(" L1 Δ(probs):", round(delta_L1, 4))

    infer.blend.lambda_expl = old_lambda


if __name__ == "__main__":
    # minimal loader (same approach as your smoke/backtest tool)
    print("Loading dependencies (best-effort)...")
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

    try:
        infer = PolicyInfer(deps, blend_cfg=PolicyBlendConfig.default())
    except Exception as e:
        print("Failed to construct PolicyInfer:", e)
        raise

    # run the tiny checks
    smoke_equity_effect(infer)
    #smoke_exploit_effect(infer)

    print("\nSMOKE TESTS DONE.")