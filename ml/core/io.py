import json
from pathlib import Path
from typing import Sequence, Dict, Any


def save_bins_sidecar(dataset_parquet: Path, spr_bins: Sequence[float], rate_bins: Sequence[float]) -> Path:
    sidecar = dataset_parquet.with_suffix(dataset_parquet.suffix + ".bins.json")
    payload = {
        "spr_bins": list(spr_bins),
        "rate_bins": list(rate_bins),
        "notes": "ExploitNet bucket boundaries used for training.",
    }
    sidecar.write_text(json.dumps(payload, indent=2))
    return sidecar

def load_bins_sidecar(dataset_parquet_or_sidecar: Path) -> Dict[str, Any]:
    p = Path(dataset_parquet_or_sidecar)
    sidecar = p if p.suffix.endswith(".json") else p.with_suffix(p.suffix + ".bins.json")
    data = json.loads(Path(sidecar).read_text())
    if "spr_bins" not in data or "rate_bins" not in data:
        raise ValueError(f"Bins sidecar missing keys: {sidecar}")
    return data