#!/usr/bin/env python3
from __future__ import annotations
import json, gzip, io, sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))


from ml.etl.rangenet.postflop.solver_policy_parser import actions_and_mix, root_node, get_children, resolve_child, \
    _has_any, bucket_raise_label, parse_root_bet_size_bb, parse_raise_to_bb


# tools/rangenet/postflop/build_solve_maps.py
# PATCH: make pot_bb/stack_bb optional for build_map_for_file

import argparse, gzip, json, os, re
from typing import Any, Dict, Iterable, List, Optional, Tuple

Json = Dict[str, Any]

RAISE_RE = re.compile(r"\braise\b", re.IGNORECASE)
CALL_RE  = re.compile(r"\bcall\b",  re.IGNORECASE)
FOLD_RE  = re.compile(r"\bfold\b",  re.IGNORECASE)
ALLIN_RE = re.compile(r"\ball[-\s]*in\b|\bjam\b", re.IGNORECASE)
BET_RE   = re.compile(r"\bbet\b",   re.IGNORECASE)
CHECK_RE = re.compile(r"\bcheck\b", re.IGNORECASE)

def load_solver(path: str) -> Json:
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def root_node(payload: Json) -> Json:
    for key in ("root", "node", "game_tree", "tree", "payload"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key]
    return payload

def _children_listlike(node: Json):
    for key in ("children", "childrens", "nodes"):
        v = node.get(key)
        if isinstance(v, list):
            return v
    return None

def get_children(node: Json) -> Dict[str, Json]:
    lst = _children_listlike(node)
    if isinstance(lst, list):
        out: Dict[str, Json] = {}
        for ch in lst:
            if not isinstance(ch, dict):
                continue
            label = ch.get("action") or ch.get("label") or ch.get("name")
            if isinstance(label, str):
                out[label] = ch
        return out
    for key in ("children", "childrens"):
        v = node.get(key)
        if isinstance(v, dict):
            return v
    return {}

def action_list(node: Json) -> List[str]:
    for key in ("actions", "available_actions", "menu"):
        v = node.get(key)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return v  # type: ignore
    return list(get_children(node).keys())

def resolve_child(node: Json, wanted: str) -> Optional[Json]:
    kids = get_children(node)
    if wanted in kids:
        return kids[wanted]
    wl = wanted.lower()
    for k, v in kids.items():
        kl = k.lower()
        if kl == wl or kl.startswith(wl) or wl.startswith(kl):
            return v
    if "raise" in wl:
        for k, v in kids.items():
            if "raise" in k.lower():
                return v
    return None

def _has_any(node: Json, pred) -> bool:
    if any(pred(a) for a in action_list(node)):
        return True
    for ch in get_children(node).values():
        if _has_any(ch, pred):
            return True
    return False

def is_raise_label(s: str) -> bool: return bool(RAISE_RE.search(s))
def is_call_label(s: str)  -> bool: return bool(CALL_RE.search(s))
def is_fold_label(s: str)  -> bool: return bool(FOLD_RE.search(s))
def is_allin_label(s: str) -> bool: return bool(ALLIN_RE.search(s))
def is_bet_label(s: str)   -> bool: return bool(BET_RE.search(s))
def is_check_label(s: str) -> bool: return bool(CHECK_RE.search(s))

def _any_raise_in_subtree(node: Json, max_depth: int = 12) -> bool:
    if max_depth < 0:
        return False
    if any(is_raise_label(a) for a in action_list(node)):
        return True
    for ch in get_children(node).values():
        if _any_raise_in_subtree(ch, max_depth - 1):
            return True
    return False

def _pattern_root_ip_bets_deep(root: Json) -> bool:
    for bet in (a for a in action_list(root) if is_bet_label(a)):
        ch = resolve_child(root, bet)
        if ch and _any_raise_in_subtree(ch):
            return True
    return False

def _pattern_oop_check_then_ip_bet_deep(root: Json) -> bool:
    for chk in (a for a in action_list(root) if is_check_label(a)):
        ch = resolve_child(root, chk)
        if not ch:
            continue
        for ipb in (a for a in action_list(ch) if is_bet_label(a)):
            ch2 = resolve_child(ch, ipb)
            if ch2 and _any_raise_in_subtree(ch2):
                return True
    return False

def _pattern_root_oop_donk_deep(root: Json) -> bool:
    for bet in (a for a in action_list(root) if is_bet_label(a)):
        ch = resolve_child(root, bet)
        if ch and _any_raise_in_subtree(ch):
            return True
    return False

def _pattern_root_ip_bets_shallow(root: Json) -> bool:
    for bet in (a for a in action_list(root) if is_bet_label(a)):
        ch = resolve_child(root, bet)
        if ch and any(is_raise_label(a) for a in action_list(ch)):
            return True
    return False

def _pattern_oop_check_then_ip_bet_shallow(root: Json) -> bool:
    for chk in (a for a in action_list(root) if is_check_label(a)):
        ch = resolve_child(root, chk)
        if not ch:
            continue
        for ipb in (a for a in action_list(ch) if is_bet_label(a)):
            ch2 = resolve_child(ch, ipb)
            if ch2 and any(is_raise_label(a) for a in action_list(ch2)):
                return True
    return False

def _pattern_root_oop_donk_shallow(root: Json) -> bool:
    for bet in (a for a in action_list(root) if is_bet_label(a)):
        ch = resolve_child(root, bet)
        if ch and any(is_raise_label(a) for a in action_list(ch)):
            return True
    return False

def build_entry(pattern_id: str, captured: bool) -> Dict[str, Any]:
    return {"pattern_id": pattern_id, "captures_raise": bool(captured)}

def build_map_for_file(path: str, *, pot_bb: float | None = None, stack_bb: float | None = None) -> Dict[str, Any]:
    payload = load_solver(path)
    root = root_node(payload)

    presence = {
        "has_raise": _has_any(root, is_raise_label),
        "has_call":  _has_any(root, is_call_label),
        "has_fold":  _has_any(root, is_fold_label),
        "has_allin": _has_any(root, is_allin_label),
    }

    entries: List[Dict[str, Any]] = []
    # Shallow
    entries.append({"pattern_id":"ROOT_OOP_DONK","facing_source":None,"nodes":"root",
                    "captures_raise": _pattern_root_oop_donk_shallow(root)})
    entries.append({"pattern_id":"ROOT_IP_BETS","facing_source":"root_ip_bet","nodes":"root->bet_child",
                    "captures_raise": _pattern_root_ip_bets_shallow(root)})
    entries.append({"pattern_id":"OOP_CHECK_THEN_IP_BET","facing_source":"check->ip_bet","nodes":"root->check_child->bet_child",
                    "captures_raise": _pattern_oop_check_then_ip_bet_shallow(root)})

    # Deep
    entries.append({"pattern_id":"ROOT_OOP_DONK_DEEP","facing_source":"root_oop_donk_deep","nodes":"root->bet_child(deep)",
                    "captures_raise": _pattern_root_oop_donk_deep(root)})
    entries.append({"pattern_id":"ROOT_IP_BETS_DEEP","facing_source":"root_ip_bet_deep","nodes":"root->bet_child(deep)",
                    "captures_raise": _pattern_root_ip_bets_deep(root)})
    entries.append({"pattern_id":"OOP_CHECK_THEN_IP_BET_DEEP","facing_source":"check->ip_bet_deep","nodes":"root->check_child->bet_child(deep)",
                    "captures_raise": _pattern_oop_check_then_ip_bet_deep(root)})

    # ✅ Generic fallback: if the tree has raises but none of the above captured them
    captured_any = any(e["captures_raise"] for e in entries)
    if presence["has_raise"] and not captured_any:
        entries.append({"pattern_id":"GLOBAL_DEEP","facing_source":"any","nodes":"root(deep)","captures_raise": True})

    return {"presence": presence, "entries": entries}

def find_solver_files(rootdir: str) -> List[str]:
    out: List[str] = []
    for dp, _, files in os.walk(rootdir):
        for fn in files:
            name = fn.lower()
            if name.endswith(".json") or name.endswith(".json.gz"):
                if name.startswith("solve_maps"):
                    continue
                out.append(os.path.join(dp, fn))
    out.sort(key=lambda p: os.path.basename(p).lower())
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/debug_samples",
                    help="Directory containing solver *.json(.gz) files")
    ap.add_argument("--output", default="solve_maps.json",
                    help="Path to write combined solve map JSON")
    args = ap.parse_args()

    files = find_solver_files(args.input)
    if not files:
        print(f"[err] no solver files found in {args.input}")
        with open(args.output, "w", encoding="utf-8") as fo:
            json.dump({}, fo)
        return

    maps: Dict[str, Any] = {}
    for f in sorted(files):
        key = os.path.splitext(os.path.basename(f))[0]
        try:
            maps[key] = build_map_for_file(f)  # pot/stack optional
            print(f"[ok] mapped {key}")
        except Exception as e:
            print(f"[fail] {f}: {e}")

    with open(args.output, "w", encoding="utf-8") as fo:
        json.dump(maps, fo, indent=2, sort_keys=True)
    print(f"[OK] wrote {args.output} with {len(maps)} entries")

if __name__ == "__main__":
    main()