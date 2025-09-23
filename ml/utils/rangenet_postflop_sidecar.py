# ml/utils/sidecar.py

import json
from pathlib import Path
from typing import Any, Dict, Optional

def write_rangenet_postflop_sidecar(
    best_ckpt: Path,
    ds,          # PostflopRangeDatasetParquet instance
    model,       # RangeNetLit (already loaded)
    model_name: str = "RangeNetPostflop",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Create a sidecar JSON next to `best_ckpt` capturing the input schema
    used for training/eval so inference can reproduce feature ordering.

    Expects the dataset to expose:
      - feature_order : List[str]
      - cards_info.cards : Dict[str, int]
      - id_maps()       : Dict[str, Dict[str,int]]
    """

    best_ckpt = Path(best_ckpt)
    sidecar_path = best_ckpt.with_suffix(best_ckpt.suffix + ".sidecar.json")

    # --- core schema ---
    try:
        feature_order = list(ds.feature_order)
    except Exception:
        feature_order = []

    try:
        cards = ds.cards_info.cards
    except Exception:
        cards = None

    try:
        id_maps = ds.id_maps()
    except Exception:
        id_maps = None

    # --- model info ---
    model_info: Dict[str, Any] = {
        "class": type(model).__name__,
        "name": model_name,
    }
    for attr in ("hidden_dims", "dropout", "num_actions", "input_dim"):
        if hasattr(model, attr):
            try:
                model_info[attr] = getattr(model, attr)
            except Exception:
                pass

    payload: Dict[str, Any] = {
        "model_name": model_name,
        "feature_order": feature_order,
        "cards": cards,
        "id_maps": id_maps,
        "num_actions": getattr(model, "num_actions", 169),
        "model_info": model_info,
    }
    if extra:
        payload.update(extra)

    sidecar_path.write_text(json.dumps(payload, indent=2))
    return sidecar_path