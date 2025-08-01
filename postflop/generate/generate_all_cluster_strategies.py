import json
from pathlib import Path

from features.types import STACK_BUCKETS, VILLAIN_PROFILES, EXPLOIT_SETTINGS, MULTIWAY_CONTEXTS, POPULATION_TYPES, \
    ACTION_CONTEXTS
from flop.clustering.cluster_config import FlopClusterGranularity
from postflop.generate.generate_cluster_strategy import generate_cluster_strategy, FLOP_CLUSTER_MAP
from postflop.schema.cluster_strategy_schema import ClusterStrategy
from preflop.matchups import MATCHUPS

# === Output directory ===
OUTPUT_DIR = Path("postflop/strategy_templates")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Invert cluster map for quick lookup ===
valid_clusters = set(FLOP_CLUSTER_MAP.values())

# === Iterate over all postflop contexts + cluster IDs ===
for cluster_id in range(FlopClusterGranularity.HIGH.value):
    if cluster_id not in valid_clusters:
        print(f"[SKIP] Cluster {cluster_id} has no mapped board.")
        continue

    for (
            ip_position,
            oop_position
    ) in MATCHUPS:
        for stack_bb in STACK_BUCKETS:
            for villain_profile in VILLAIN_PROFILES:
                for exploit_setting in EXPLOIT_SETTINGS:
                    for multiway_context in MULTIWAY_CONTEXTS:
                        for population_type in POPULATION_TYPES:
                            for action_context in ACTION_CONTEXTS:

                                print(f"\n🧠 Cluster {cluster_id} | {ip_position} vs {oop_position} @ {stack_bb}bb")
                                print(
                                    f"    → {villain_profile}/{exploit_setting}/{multiway_context}/{population_type}/{action_context}")

                                try:
                                    strategy: ClusterStrategy = generate_cluster_strategy(
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
                                    print(f"[ERROR] Skipped: {e}")
                                    continue

                                # === Build file path ===
                                out_dir = OUTPUT_DIR / villain_profile / exploit_setting / multiway_context / population_type / action_context
                                out_dir.mkdir(parents=True, exist_ok=True)

                                file_name = f"{ip_position}_vs_{oop_position}_{stack_bb}bb_cluster_{cluster_id}.json"
                                out_path = out_dir / file_name

                                with open(out_path, "w") as f:
                                    json.dump(strategy.model_dump(), f, indent=2)

                                print(f"✅ Saved strategy: {out_path}")