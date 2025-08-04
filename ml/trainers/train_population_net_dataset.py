# Example input directory (aggregated parsed population stats)
import json
from pathlib import Path
from typing import List, Tuple
from ml.datasets.population_net_dataset import PopulationNetDataset
from ml.schema.population_net_schema import PopulationNetFeatures, PopulationNetLabel

INPUT_DIR = Path("data/processed")

def parse_population_file(path: Path) -> List[Tuple[PopulationNetFeatures, PopulationNetLabel]]:
    with path.open("r") as f:
        data = [json.loads(line) for line in f]

    pairs = []
    for record in data:
        features = PopulationNetFeatures(**record["features"])
        label = PopulationNetLabel(**record["label"])
        pairs.append((features, label))
    return pairs


def build_population_dataset(stake: str, input_dir: Path = INPUT_DIR) -> PopulationNetDataset:
    input_file = input_dir / f"population_{stake}.jsonl"
    print(f"📄 Loading: {input_file}")

    samples = parse_population_file(input_file)
    features, labels = zip(*samples)
    dataset = PopulationNetDataset(list(features), list(labels))

    print(f"✅ Built PopulationNetDataset with {len(dataset)} samples for {stake}")
    return dataset


if __name__ == "__main__":
    # Example: build and inspect dataset
    dataset = build_population_dataset("NL10")