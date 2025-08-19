# ml/utils/enums.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import os, yaml

@dataclass(frozen=True)
class Types:
    positions: List[str]
    action_contexts: List[str]
    multiway_contexts: List[str]
    villain_profiles: List[str]
    streets: List[str]

ALIAS_CTX: Dict[str, str] = {
    # normalize extra names your code may emit → canonical settings types
    "VS_SQUEEZE": "VS_3BET",      # if you don’t plan a separate head yet
    "CHECK_TO": "VS_OPEN",        # example: map “check_to” buckets to a base node
    # add more once you decide policy
}

def load_types(yaml_path: str = "ml/config/settings.yaml") -> Types:
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    t = cfg.get("types", {}) or {}
    return Types(
        positions=t.get("positions", []),
        action_contexts=t.get("action_contexts", []),
        multiway_contexts=t.get("multiway_contexts", []),
        villain_profiles=t.get("villain_profiles", []),
        streets=t.get("streets", []),
    )

def normalize_ctx(raw: str, types: Types) -> str:
    if raw in types.action_contexts:
        return raw
    raw_up = (raw or "").upper()
    mapped = ALIAS_CTX.get(raw_up, raw_up)
    if mapped not in types.action_contexts:
        raise ValueError(f"Unknown action context '{raw}' (mapped '{mapped}'). "
                         f"Allowed: {types.action_contexts}")
    return mapped

def require_member(name: str, value: str, allowed: List[str]) -> str:
    v = (value or "").upper()
    if v not in allowed:
        raise ValueError(f"Invalid {name}='{value}'. Allowed: {allowed}")
    return v