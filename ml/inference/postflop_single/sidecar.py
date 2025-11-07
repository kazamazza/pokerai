from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

def load_sidecar(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sidecar not found: {p}")
    with p.open("r") as f:
        sc = json.load(f)
    # Basic checks
    feature_order = sc.get("feature_order") or sc.get("cat_feature_order") or []
    cards = sc.get("cards") or sc.get("card_sizes") or {}
    action_vocab = sc.get("action_vocab") or []
    if not feature_order:
        raise ValueError(f"sidecar missing 'feature_order' in {p}")
    if not cards:
        raise ValueError(f"sidecar missing 'cards' in {p}")
    if not action_vocab:
        raise ValueError(f"sidecar missing 'action_vocab' in {p}")
    return sc