# tools/rangenet/postflop/inspect_raise_nodes.py
# -*- coding: utf-8 -*-
"""
Probe solver trees and dump pot/facing candidates around RAISE edges.
Why: discover the exact field names in your TexasSolver exports to wire the parser.
"""

from __future__ import annotations
import argparse
import gzip
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

Json = Dict[str, Any]

RAISE_RE = re.compile(r"\braise\b", re.IGNORECASE)
BET_RE   = re.compile(r"\bbet\b", re.IGNORECASE)

# Broad key lists; we’ll see which ones exist in your dumps.
POT_KEYS = [
    "pot_bb","total_pot_bb","pot","total_pot","round_pot","potSize","pot_size","current_pot",
    "pot_after","street_pot","pot_before"
]
FACING_KEYS = [
    "to_call","call_amount","facing_bet_bb","facing_bet","amount_to_call","last_bet","bet_to_call",
    "needed","min_call","commit","callsize","ip_to_call","oop_to_call"
]

def load_json(path: str) -> Json:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def root_node(payload: Json) -> Json:
    for k in ("root","node","game_tree","tree","payload"):
        v = payload.get(k)
        if isinstance(v, dict):
            return v
    return payload

def _child_container(node: Json):
    for key in ("children","childrens","nodes"):
        v = node.get(key)
        if isinstance(v, list): return "list", v
        if isinstance(v, dict): return "dict", v
    return None, None

def get_children(node: Json) -> Dict[str, Json]:
    kind, cont = _child_container(node)
    if kind == "list":
        out: Dict[str, Json] = {}
        for ch in cont:  # type: ignore
            if isinstance(ch, dict):
                label = ch.get("action") or ch.get("label") or ch.get("name")
                if isinstance(label, str):
                    out[label] = ch
        return out
    if kind == "dict":
        return {str(k): v for k, v in cont.items() if isinstance(v, dict)}  # type: ignore
    return {}

def _try_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def _extract_candidates(obj: Dict[str, Any], keys: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in keys:
        if k in obj:
            v = _try_float(obj[k])
            if v is not None:
                out[k] = v
    return out

def _gather_state_views(node: Json) -> List[Dict[str, Any]]:
    views: List[Dict[str, Any]] = [node]
    for k in ("state","info","stats","meta","ctx","context","data"):
        v = node.get(k)
        if isinstance(v, dict):
            views.append(v)
    return views

def _snapshot_state(node: Json) -> Dict[str, Any]:
    views = _gather_state_views(node)
    pots, faces = {}, {}
    for v in views:
        pots.update(_extract_candidates(v, POT_KEYS))
        faces.update(_extract_candidates(v, FACING_KEYS))
    return {"pot_candidates": pots, "facing_candidates": faces, "keys_present": sorted(set(list(pots.keys()) + list(faces.keys())))}

def _shorten(d: Dict[str, float], topn: int = 4) -> Dict[str, float]:
    items = sorted(d.items(), key=lambda kv: (-abs(kv[1]), kv[0]))  # arbitrary stable order
    return {k: v for k, v in items[:topn]}

def _dfs_collect_raises(node: Json, path: List[str], limit: int, max_depth: int, out: List[Dict[str, Any]]) -> None:
    if len(out) >= limit or len(path) > max_depth:
        return
    kids = get_children(node)
    if not kids:
        return
    for label, ch in kids.items():
        new_path = path + [label]
        if RAISE_RE.search(label):
            parent_snap = _snapshot_state(node)
            child_snap  = _snapshot_state(ch)
            out.append({
                "path": " / ".join(new_path[-6:]),
                "label": label,
                "parent": {
                    "pots": _shorten(parent_snap["pot_candidates"]),
                    "faces": _shorten(parent_snap["facing_candidates"]),
                    "keys": parent_snap["keys_present"],
                },
                "child": {
                    "pots": _shorten(child_snap["pot_candidates"]),
                    "faces": _shorten(child_snap["facing_candidates"]),
                    "keys": child_snap["keys_present"],
                },
            })
            if len(out) >= limit:
                return
        _dfs_collect_raises(ch, new_path, limit, max_depth, out)

def inspect_file(path: str, limit: int, max_depth: int) -> Dict[str, Any]:
    payload = load_json(path)
    root = root_node(payload)
    hits: List[Dict[str, Any]] = []
    _dfs_collect_raises(root, [], limit, max_depth, hits)
    return {"file": os.path.basename(path), "count": len(hits), "samples": hits}

def iter_paths(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        out: List[str] = []
        for root, _, files in os.walk(input_path):
            for fn in files:
                n = fn.lower()
                if n.endswith(".json") or n.endswith(".json.gz"):
                    out.append(os.path.join(root, fn))
        out.sort()
        return out
    else:
        return [input_path]

def main():
    ap = argparse.ArgumentParser(description="Inspect RAISE edges and print pot/facing candidates.")
    ap.add_argument("--input", default="data/debug_samples", help="Dir or file (json/json.gz)")
    ap.add_argument("--limit", type=int, default=4, help="Max RAISE samples per file")
    ap.add_argument("--max-depth", type=int, default=24, help="DFS depth cutoff")
    ap.add_argument("--out", default="data/debug_raise_inspect.json", help="Where to write JSON report")
    args = ap.parse_args()

    paths = iter_paths(args.input)
    if not paths:
        print(f"[err] no inputs in {args.input}")
        return

    report: Dict[str, Any] = {}
    print("=== RAISE node inspection (pot/facing candidates) ===")
    for p in paths:
        res = inspect_file(p, args.limit, args.max_depth)
        report[os.path.basename(p)] = res
        print(f"\n-- {res['file']}  (samples={res['count']})")
        for i, s in enumerate(res["samples"], 1):
            print(f"  [{i}] {s['path']}")
            print(f"      label: {s['label']}")
            # Parent
            if s["parent"]["pots"] or s["parent"]["faces"]:
                print(f"      parent.pots   : {s['parent']['pots']}")
                print(f"      parent.facing : {s['parent']['faces']}")
            # Child
            if s["child"]["pots"] or s["child"]["faces"]:
                print(f"      child.pots    : {s['child']['pots']}")
                print(f"      child.facing  : {s['child']['faces']}")
            # Keys seen (debug)
            keys = sorted(set(s["parent"]["keys"] + s["child"]["keys"]))
            if keys:
                print(f"      keys_present  : {keys[:10]}{' ...' if len(keys) > 10 else ''}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] wrote {args.out}")

if __name__ == "__main__":
    main()