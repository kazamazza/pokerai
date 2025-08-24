# ml/utils/save_sidecar.py
from __future__ import annotations
from pathlib import Path
import json

def save_sidecar_for_inference(
    out_dir: str | Path,
    feature_order: list[str],
    cards: dict[str, int],
    encoders: dict[str, dict],
    filename: str = "sidecar.json",
):
    out = {
        "feature_order": list(feature_order),
        "cards": {k: int(v) for k, v in cards.items()},
        # stringify keys for JSON stability
        "encoders": {col: {str(k): int(v) for k, v in enc.items()} for col, enc in encoders.items()},
    }
    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"✅ wrote inference sidecar → {out_path}")