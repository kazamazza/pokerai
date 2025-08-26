from __future__ import annotations
from pathlib import Path
import json, time
from typing import Any, Dict, Optional


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

DEFAULT_SIDECAR_NAME = "sidecar.json"

def resolve_sidecar_path(
    explicit: Optional[str | Path],
    checkpoint_path: Optional[str | Path] = None,
    default_name: str = DEFAULT_SIDECAR_NAME,
) -> Path:
    """
    If explicit provided -> Path(explicit).
    Else if checkpoint provided -> <ckpt_dir>/<default_name>.
    Else -> raise.
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Sidecar not found at explicit path: {p}")
        return p

    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        ckpt_dir = ckpt.parent if ckpt.suffix else ckpt  # allow passing a directory
        p = ckpt_dir / default_name
        if not p.exists():
            raise FileNotFoundError(f"Sidecar not found next to checkpoint: {p}")
        return p

    raise ValueError("resolve_sidecar_path needs either explicit sidecar path or a checkpoint path")

def load_sidecar(path: str | Path) -> Dict[str, Any]:
    """
    Load and lightly validate a sidecar JSON produced by training.
    Expected keys (minimum): feature_order (list[str]), cards (dict[str,int] or similar)
    Returns the JSON as a dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sidecar missing: {p}")
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        raise ValueError(f"Failed to parse sidecar JSON at {p}: {e}")

    # soft validation
    if "feature_order" not in data or not isinstance(data["feature_order"], list):
        raise ValueError(f"Sidecar {p} missing 'feature_order' list")
    if "cards" not in data or not isinstance(data["cards"], dict):
        # some models might not need 'cards'; keep warning but not fatal if you prefer
        raise ValueError(f"Sidecar {p} missing 'cards' dict")

    return data