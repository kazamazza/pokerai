from pathlib import Path
import json
from typing import Generator
from ml.datasets.range_net_dataset import RangeNetDataset
from ml.schema.range_net_schema import RangeNetLabel, RangeNetFeatures
from postflop.schema.cluster_strategy_schema import ClusterStrategy


# === Configuration ===
SOLVED_STRATEGY_DIR = Path("s3://your-bucket/postflop/solved/")  # Update if working locally
OUTPUT_PATH = Path("ml/datasets/generated/range_net_dataset.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# === Example Generator Function ===
def load_solved_strategies() -> Generator[ClusterStrategy, None, None]:
    """
    Stream ClusterStrategy objects from solved postflop JSONs.
    """
    # TODO: Replace with S3 integration logic
    for path in Path("postflop/solved").rglob("*.json"):
        with open(path, "r") as f:
            data = json.load(f)
            yield ClusterStrategy(**data)

# === Main Dataset Build Logic ===
def build_range_net_dataset() -> None:
    count = 0

    with open(OUTPUT_PATH, "w") as out_file:
        for strategy in load_solved_strategies():
            # TODO: Sample hands from strategy.combos + board
            # TODO: Extract contextual + texture features
            # TODO: Build labels from IP or OOP actions

            features = RangeNetFeatures(
                cluster_id=strategy.cluster_id,
                ip_position=strategy.meta.ip_position,
                oop_position=strategy.meta.oop_position,
                stack_bb=strategy.meta.stack_bb,
                # ... add others later
            )

            label = RangeNetLabel(
                range_vector=strategy.ip_strategy.combos  # Placeholder — refine later
            )

            sample = RangeNetDataset(features=features, label=label)
            out_file.write(sample.model_dump_json() + "\n")

            count += 1
            if count % 100 == 0:
                print(f"✅ Processed {count} samples...")

    print(f"\n🎯 Done! Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_range_net_dataset()