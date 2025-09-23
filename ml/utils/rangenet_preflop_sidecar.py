# ml/utils/sidecar_rangenet_preflop.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

def write_rangenet_preflop_sidecar(
    best_ckpt: Path,
    ds,          # PreflopRangeDatasetParquet
    model,       # RangeNetLit
    model_name: str = "RangeNetPreflop",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    best_ckpt = Path(best_ckpt)
    sidecar_path = best_ckpt.with_suffix(best_ckpt.suffix + ".sidecar.json")

    try:
        cards = ds.cards()
    except Exception:
        cards = None

    try:
        feature_order = list(ds.feature_order)
    except Exception:
        feature_order = []

    # NEW: pull exact categorical encoders
    try:
        id_maps = ds.id_maps()  # {col: {category_str: int_id}}
    except Exception:
        id_maps = None

    model_info = {"class": type(model).__name__, "name": model_name}
    for attr in ("hidden_sizes", "dropout", "num_actions", "input_dim"):
        if hasattr(model, attr):
            try:
                model_info[attr] = getattr(model, attr)
            except Exception:
                pass

    payload: Dict[str, Any] = {
        "model_name": model_name,
        "feature_order": feature_order,
        "cards": cards,
        "id_maps": id_maps,                    # <—— NEW: persist encoders
        "num_actions": getattr(model, "num_actions", 169),
        "model_info": model_info,
    }
    if extra:
        payload.update(extra)

    sidecar_path.write_text(json.dumps(payload, indent=2))
    return sidecar_path