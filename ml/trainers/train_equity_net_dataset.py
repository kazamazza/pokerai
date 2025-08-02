from pathlib import Path
import json
from typing import List

from ml.datasets.equitynet_dataset import EquityNetDataset
from ml.schema.equity_net_schema import EquityNetFeatures, EquityNetLabel

# Example input directory (simulated hands with equities)
INPUT_DIR = Path("data/equity_simulations")


def parse_equity_file(path: Path) -> List[tuple[EquityNetFeatures, EquityNetLabel]]:
    with open(path, "r") as f:
        data = json.load(f)

    output = []
    for entry in data:
        features = EquityNetFeatures(
            hero_hand=entry["hero_hand"],
            board=entry["board"],
            position=entry["position"],
            stack_bb=entry["stack_bb"],
            pot_size=entry["pot_size"],
            num_players=entry["num_players"],
            has_initiative=entry["has_initiative"],
        )
        label = EquityNetLabel(
            raw_equity=entry["raw_equity"],
            normalized_equity=entry["normalized_equity"]
        )
        output.append((features, label))

    return output


def build_equity_dataset(input_dir: Path = INPUT_DIR) -> EquityNetDataset:
    all_pairs = []

    for file in input_dir.glob("*.json"):
        print(f"📄 Parsing: {file}")
        pairs = parse_equity_file(file)
        all_pairs.extend(pairs)

    features, labels = zip(*all_pairs)
    dataset = EquityNetDataset(list(features), list(labels))
    print(f"✅ Built EquityNetDataset with {len(dataset)} samples")
    return dataset


if __name__ == "__main__":
    dataset = build_equity_dataset()