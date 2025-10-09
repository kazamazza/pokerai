#!/usr/bin/env python3
import argparse, json
import sys
from pathlib import Path

from ml.inference.equity import EquityNetInfer
from ml.inference.exploit import ExploitNetInfer
from ml.inference.population import PopulationNetInference

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from ml.features.boards.board_clusterers.kmeans import KMeansBoardClusterer
from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.policy import PolicyInfer
from ml.inference.policy.policy_blend_config import PolicyBlendConfig
from ml.inference.policy.types import PolicyRequest
from ml.inference.postflop import PostflopPolicyInfer
from ml.inference.preflop import PreflopPolicy


def main():
    print("🚀 PolicyInfer smoke test starting...")

    postflop = PostflopPolicyInfer.from_dir("checkpoints/postflop_policy")
    preflop = PreflopPolicy.from_dir("checkpoints/range_preflop")
    equity = EquityNetInfer.from_dir("checkpoints/equitynet")        # stub or trained model
    exploit = ExploitNetInfer.from_dir()      # stub or trained model
    pop = PopulationNetInference.from_dir()              # stub or trained model
    clusterer = KMeansBoardClusterer.load("data/artifacts/board_clusters_kmeans_128.json")

    deps = PolicyInferDeps(
        policy_post=postflop,
        range_pre=preflop,
        equity=equity,
        exploit=exploit,
        pop=pop,
        clusterer=clusterer,
        params={},
    )

    blend_cfg = PolicyBlendConfig.default()
    infer = PolicyInfer(deps, blend_cfg)

    # === 2️⃣ Make a test request ===
    req = PolicyRequest(
        board="AsKh2d",
        ip_pos="BTN",
        oop_pos="BB",
        ctx="VS_OPEN",
        pot_bb=18.0,
        eff_stack_bb=100.0,
        street=1,
    )

    # === 3️⃣ Run inference ===
    out = infer.predict(req.__dict__)

    # === 4️⃣ Print output ===
    print("\n✅ PolicyInfer smoke test complete")
    print(f"Board: {req.board}  ctx={req.ctx}  actor={req.actor}")
    print("Top actions:")
    top = sorted(zip(out.actions, out.probs), key=lambda x: x[1], reverse=True)[:10]
    for a, p in top:
        print(f"  {a:<10} {p:6.3f}")
    print("\nNotes:", out.notes)
    print("Debug:", {k: v for k, v in out.debug.items() if k in ('actor','ctx','street','board_cluster_id')})

if __name__ == "__main__":
    main()