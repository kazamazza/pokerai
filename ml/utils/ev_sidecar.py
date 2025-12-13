# utils/sidecar_evnet.py
from __future__ import annotations
import json, os
from pathlib import Path
from datetime import datetime
from typing import Mapping, Any, Dict, List, Optional

def write_ev_sidecar(
    best_ckpt: Optional[str],
    meta: Mapping[str, Any],
    *,
    filename: str = "best_sidecar.json",
    duplicate_stem_copy: bool = False,
) -> Optional[str]:
    if not best_ckpt:
        print("write_ev_sidecar: no checkpoint path provided")
        return None

    ckpt_path = Path(best_ckpt)
    out_dir = ckpt_path.parent if ckpt_path.suffix else ckpt_path
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = str(meta.get("model_name", "EVNet"))
    action_vocab: List[str] = list(meta.get("action_vocab", []))
    x_cols: List[str]        = list(meta.get("x_cols", []))
    cont_cols: List[str]     = list(meta.get("cont_cols", []))
    id_maps_raw: Dict[str, Dict[str, Any]] = dict(meta.get("id_maps") or {})
    id_maps: Dict[str, Dict[str, int]] = {
        str(k): {str(a): int(b) for a, b in (mp or {}).items()} for k, mp in id_maps_raw.items()
    }

    payload: Dict[str, Any] = {
        "schema_version": "ev_sidecar_v1",
        "model_name": model_name,
        "action_vocab": action_vocab,
        "vocab_index": {a: i for i, a in enumerate(action_vocab)},
        "cat_feature_order": x_cols,
        "cont_feature_order": cont_cols,
        "x_cols": x_cols,
        "cont_cols": cont_cols,
        "id_maps": id_maps,
        "cat_cardinalities": {c: len(id_maps.get(c, {})) for c in x_cols},
        "notes": meta.get("notes", ""),
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "checkpoint_file": os.path.basename(str(ckpt_path)) if ckpt_path.suffix else None,
        # ✅ NEW:
        "units": str(meta.get("units", "bb")),
        "split": meta.get("split"),  # "preflop" | "root" | "facing" | None
    }

    best_path = out_dir / filename
    best_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"💾 wrote EV sidecar → {best_path}")

    if duplicate_stem_copy and ckpt_path.suffix:
        stem_copy = out_dir / f"{ckpt_path.stem}_sidecar.json"
        stem_copy.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"↳ mirrored sidecar   → {stem_copy}")

    return str(best_path)