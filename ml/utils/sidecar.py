# ml/utils/sidecar.py
from __future__ import annotations
from pathlib import Path
import json, time
from typing import Any, Dict

def save_sidecar_json(
    ckpt_path: str | Path,
    *,
    model_name: str,
    feature_order: list[str],
    cards: Dict[str, int],
    id_maps: Dict[str, Dict[Any, int]] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Path:
    """
    Write <checkpoint>.sidecar.json next to the checkpoint.
    """
    ckpt_path = Path(ckpt_path)
    payload = {
        "model_name": model_name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_order": feature_order,
        "cards": cards,                # embedding cardinalities per feature
        "id_maps": id_maps or None,    # optional: raw_value -> id per feature
    }
    if extra:
        payload.update(extra)
    out = ckpt_path.with_suffix(ckpt_path.suffix + ".sidecar.json")
    out.write_text(json.dumps(payload, indent=2))
    return out