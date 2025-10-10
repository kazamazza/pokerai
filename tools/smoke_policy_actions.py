import sys
from pathlib import Path



ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from ml.inference.population import PopulationNetInference
from ml.inference.postflop import PostflopPolicyInfer
from ml.inference.preflop import PreflopPolicy
from ml.features.boards import load_board_clusterer
from ml.inference.equity import EquityNetInfer
from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.policy import PolicyInfer
from ml.inference.policy.policy_blend_config import PolicyBlendConfig
from ml.inference.policy.types import PolicyRequest
from ml.inference.player_exploit_store import PlayerExploitStore, ExploitConfig

# Paths (adjust if needed)
POSTFLOP_DIR = Path("checkpoints/postflop_policy")
PREFLOP_DIR  = Path("checkpoints/range_pre")
EQUITY_DIR   = Path("checkpoints/equitynet")
POP_DIR      = Path("checkpoints/popnet")
EXPLOIT_DIR = Path("checkpoints/exploit")
CLUSTER_JSON = Path("data/artifacts/board_clusters_kmeans_128.json")

# --- quick dependency loader
def load_dependency(cls, arg):
    try:
        return cls.from_dir(str(arg))
    except Exception as e:
        print(f"[warn] failed to load {arg}: {e}")
        return None

def build_infer():
    postflop = load_dependency(PostflopPolicyInfer, POSTFLOP_DIR)
    preflop  = load_dependency(PreflopPolicy, PREFLOP_DIR)
    equity   = load_dependency(EquityNetInfer, EQUITY_DIR)
    pop      = load_dependency(PopulationNetInference, POP_DIR)
    clusterer = None
    try:

        clusterer = load_board_clusterer({
            "board_clustering": {"type": "kmeans", "artifact": str(CLUSTER_JSON)}
        })
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
    return PolicyInfer(deps, blend_cfg=PolicyBlendConfig.default())


def make_cases():
    """Each case is (expected_action, PolicyRequest)."""
    cases = []

    # --- CHECK ---
    cases.append((
        "CHECK",
        PolicyRequest(
            street=1, hero_pos="BTN", villain_pos="BB",
            board="As7d2c", pot_bb=12.0, eff_stack_bb=100.0,
            facing_bet=False, hero_hand="Kc9c",
            raw={"ctx":"VS_OPEN"}
        ).legalize()
    ))

    # --- BET_33 ---
    cases.append((
        "BET_33",
        PolicyRequest(
            street=1, hero_pos="BTN", villain_pos="BB",
            board="9h5d2s", pot_bb=12.0, eff_stack_bb=100.0,
            facing_bet=False, hero_hand="Jh9h",
            raw={"ctx":"VS_OPEN"}
        ).legalize()
    ))

    # --- BET_66 ---
    cases.append((
        "BET_66",
        PolicyRequest(
            street=1, hero_pos="BTN", villain_pos="BB",
            board="Ts9s8d", pot_bb=12.0, eff_stack_bb=100.0,
            facing_bet=False, hero_hand="JdJh",
            raw={"ctx":"VS_OPEN"}
        ).legalize()
    ))

    # --- FOLD ---
    cases.append((
        "FOLD",
        PolicyRequest(
            street=3, hero_pos="BTN", villain_pos="BB",
            board="AsKhQhTh2s", pot_bb=60.0, eff_stack_bb=80.0,
            facing_bet=True, hero_hand="7c2d",
            raw={"ctx":"VS_OPEN"}
        ).legalize()
    ))

    # --- CALL ---
    cases.append((
        "CALL",
        PolicyRequest(
            street=2, hero_pos="BTN", villain_pos="BB",
            board="Ts9s3d8h", pot_bb=28.0, eff_stack_bb=100.0,
            facing_bet=True, hero_hand="JcTd",
            raw={"ctx":"VS_OPEN"}
        ).legalize()
    ))

    # --- RAISE_300 ---
    cases.append((
        "RAISE_300",
        PolicyRequest(
            street=2, hero_pos="BTN", villain_pos="BB",
            board="8h7h2s3c", pot_bb=28.0, eff_stack_bb=100.0,
            facing_bet=True, hero_hand="AhKh",
            raw={"ctx":"VS_OPEN"}
        ).legalize()
    ))

    # --- RAISE_400 ---
    cases.append((
        "RAISE_400",
        PolicyRequest(
            street=1, hero_pos="BTN", villain_pos="BB",
            board="TsTh8s", pot_bb=12.0, eff_stack_bb=100.0,
            facing_bet=True, hero_hand="TdTc",
            raw={"ctx":"VS_OPEN"}
        ).legalize()
    ))

    return cases


def run_smoke(infer):
    cases = make_cases()
    seen = set()

    print("\n--- ACTION SMOKE TEST (with raw logits) ---")
    for expected, req in cases:
        # 1) Build row for the postflop model
        row = infer._build_postflop_row(req)

        # 2) Decide actor from the built row (hero is IP if hero_pos == ip_pos)
        hero_is_ip = str(row.get("hero_pos", "")).upper() == str(row.get("ip_pos", "")).upper()
        actor = "ip" if hero_is_ip else "oop"

        # 3) Get raw logits from the postflop model (before masks/blend)
        out = infer.pol_post.predict_proba([row], actor=actor, return_logits=True)
        z_ip, z_oop = out["logits_ip"], out["logits_oop"]  # [1, V] each
        z_hero = z_ip if hero_is_ip else z_oop

        # Be defensive about shapes
        if z_hero is None or z_hero.ndim != 2 or z_hero.size(0) != 1:
            print(f"{expected:>10} | got: <logit shape err> | logits_shape={tuple(z_hero.shape) if z_hero is not None else None}")
            continue

        logits = z_hero.squeeze(0).detach().cpu().numpy()  # [V]

        # 4) Top-3 raw logits (pre-mask)
        top3_idx = logits.argsort()[-3:][::-1]
        top3 = [(infer.action_vocab[i], round(float(logits[i]), 3)) for i in top3_idx]

        # 5) Full policy path (with masks/blend) to see final chosen action
        out_full = infer.predict(req)
        if out_full.probs and len(out_full.probs) == len(out_full.actions):
            top_idx = int(max(range(len(out_full.probs)), key=lambda i: out_full.probs[i]))
            top = out_full.actions[top_idx]
            seen.add(top)
        else:
            top = "<bad_probs>"

        print(f"{expected:>10} | got: {top:8} | top3_logits: {top3}")

    print("\nSeen actions:", seen)
    print("Total unique:", len(seen), "of", len(infer.action_vocab))


if __name__ == "__main__":
    print("Loading policy inference...")
    infer = build_infer()
    run_smoke(infer)
    print("\n✅ Smoke test complete.")