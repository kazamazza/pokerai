import json
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from features.types import STACK_BUCKETS
from flop.clustering.cluster_config import FlopClusterGranularity
from infra.storage.s3_uploader import S3Uploader
from postflop.generate.generate_cluster_strategy import generate_cluster_strategy
from preflop.matchups import MATCHUPS

# === Load environment ===
load_dotenv()

# === Load flop cluster map ===
with open("data/flop/flop_cluster_map.json", "r") as f:
    FLOP_CLUSTER_MAP = json.load(f)

# === Prepare cluster lookups ===
valid_clusters = set(FLOP_CLUSTER_MAP.values())
cluster_to_board = {
    cid: board for board, cid in FLOP_CLUSTER_MAP.items()
    if cid in valid_clusters and cid not in locals().get("cluster_to_board", {})
}

# === Setup ===
OUTPUT_DIR = Path("postflop/strategy_templates")
s3 = S3Uploader()

def generate_all_cluster_strategies():
    for cluster_id in range(FlopClusterGranularity.HIGH.value):
        if cluster_id not in valid_clusters:
            print(f"[SKIP] Cluster {cluster_id} has no mapped board.")
            continue

        for ip_position, oop_position in MATCHUPS:
            for stack_bb in STACK_BUCKETS:
                # Optional: minimize other permutations for now
                villain_profile = "GTO"
                exploit_setting = "GTO"
                multiway_context = "HU"
                population_type = "REGULAR"
                action_context = "OPEN"

                print(f"\n🧠 Cluster {cluster_id} | {ip_position} vs {oop_position} @ {stack_bb}bb")
                print(
                    f"    → {villain_profile}/{exploit_setting}/{multiway_context}/{population_type}/{action_context}")

                # Build path to required preflop file
                s3_prefix = f"preflop/ranges/profile={villain_profile}/exploit={exploit_setting}/multiway={multiway_context}/pop={population_type}/action={action_context}"
                file_name = f"{ip_position}_vs_{oop_position}_{stack_bb}bb.json"
                s3_key = f"{s3_prefix}/{file_name}"
                local_path = Path(s3_key)

                # Pull from S3 if needed
                s3.download_file_if_missing(s3_key, local_path)

                try:
                    strategy = generate_cluster_strategy(
                        cluster_id=cluster_id,
                        stack_bb=stack_bb,
                        ip_position=ip_position,
                        oop_position=oop_position,
                        villain_profile=villain_profile,
                        exploit_setting=exploit_setting,
                        multiway_context=multiway_context,
                        population_type=population_type,
                        action_context=action_context
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to generate strategy: {e}")
                    continue

                # === Save locally ===
                out_dir = OUTPUT_DIR / villain_profile / exploit_setting / multiway_context / population_type / action_context
                out_dir.mkdir(parents=True, exist_ok=True)

                file_out = f"{ip_position}_vs_{oop_position}_{stack_bb}bb_cluster_{cluster_id}.json"
                local_out_path = out_dir / file_out

                with open(local_out_path, "w") as f:
                    json.dump(strategy.model_dump(), f, indent=2)

                print(f"✅ Saved: {local_out_path}")

                # === Upload to S3 ===
                s3_out_key = f"postflop/strategy_templates/profile={villain_profile}/exploit={exploit_setting}/multiway={multiway_context}/pop={population_type}/action={action_context}/{file_out}"
                s3.upload_file(local_out_path, s3_out_key)

# === Entry Point ===
if __name__ == "__main__":
    generate_all_cluster_strategies()