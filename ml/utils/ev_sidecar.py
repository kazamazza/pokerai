# ml/etl/ev/sidecar.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Optional, Dict, List
from datetime import datetime
import json
import os


def write_ev_sidecar(best_ckpt: Optional[str], meta: Mapping[str, Any]) -> Optional[str]:
    """
    Write an EVNet sidecar JSON next to the checkpoint file.

    Args:
        best_ckpt: Path to a .ckpt file (will write <stem>_sidecar.json beside it).
        meta: Dict with keys like:
              - model_name: str
              - action_vocab: List[str]
              - x_cols: List[str]
              - cont_cols: List[str]
              - id_maps: Dict[str, Dict[str,int]]
              - notes: str (optional)

    Returns:
        The sidecar path as string, or None if writing failed.
    """
    try:
        if not best_ckpt:
            print("write_ev_sidecar: no checkpoint path provided")
            return None

        ckpt_path = Path(best_ckpt)
        out_dir = ckpt_path.parent if ckpt_path.suffix else ckpt_path
        out_dir.mkdir(parents=True, exist_ok=True)

        # Resolve names/columns
        model_name: str = str(meta.get("model_name", "EVNet"))
        action_vocab: List[str] = list(meta.get("action_vocab", []))
        x_cols: List[str] = list(meta.get("x_cols", []))
        cont_cols: List[str] = list(meta.get("cont_cols", []))
        id_maps: Dict[str, Dict[str, int]] = {str(k): {str(a): int(b) for a, b in (v or {}).items()}
                                              for k, v in (meta.get("id_maps") or {}).items()}

        vocab_index = {a: i for i, a in enumerate(action_vocab)}
        cat_cardinalities = {c: len(id_maps.get(c, {})) for c in x_cols}

        payload: Dict[str, Any] = {
            "model_name": model_name,
            "action_vocab": action_vocab,
            "vocab_index": vocab_index,
            "cat_feature_order": x_cols,
            "cont_feature_order": cont_cols,
            "id_maps": id_maps,
            "cat_cardinalities": cat_cardinalities,
            "notes": meta.get("notes", ""),
            "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "checkpoint_file": os.path.basename(str(ckpt_path)) if ckpt_path.suffix else None,
        }

        sidecar_name = (ckpt_path.stem + "_sidecar.json") if ckpt_path.suffix else "evnet_sidecar.json"
        sidecar_path = out_dir / sidecar_name

        with sidecar_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

        return str(sidecar_path)

    except Exception as e:
        print(f"write_ev_sidecar: failed to write sidecar: {e}")
        return None