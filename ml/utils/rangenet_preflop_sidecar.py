import json
from pathlib import Path
from typing import Sequence, Dict, Any

def write_preflop_policy_sidecar(
    ckpt_dir: Path,
    *,
    feature_order: Sequence[str],
    id_maps: Dict[str, Dict[str, int]],
    cards: Dict[str, int],
    extra: Dict[str, Any] | None = None,
) -> Path:
    """
    Saves a unified `sidecar.json` metadata file alongside model checkpoints.

    This file is essential for inference: it tells the loader how to encode
    categorical features into IDs exactly the same way the model was trained.

    Args:
        ckpt_dir: Directory where checkpoints and logs are stored.
        feature_order: Ordered list of categorical feature names.
        id_maps: Mapping of raw category strings -> integer IDs per feature.
        cards: Vocabulary size (cardinality) per feature.
        extra: Optional dictionary of additional metadata (e.g., dataset info,
               label smoothing, normalization flags, or timestamp).

    Returns:
        Path to the written `sidecar.json` file.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    sidecar = {
        "feature_order": list(feature_order),
        "id_maps": id_maps,
        "cards": cards,
    }

    if extra and isinstance(extra, dict):
        sidecar.update(extra)

    path = ckpt_dir / "sidecar.json"
    path.write_text(json.dumps(sidecar, indent=2))
    print(f"[sidecar] saved → {path}")
    return path