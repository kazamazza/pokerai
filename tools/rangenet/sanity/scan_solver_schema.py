# file: tools/rangenet/sanity/scan_solver_schema.py
from __future__ import annotations
import argparse, json, io, os, re, sys, gzip
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple, Set, Union

# Optional deps are used only if needed.
def _read_uri(uri: str) -> bytes:
    uri = str(uri)
    if uri.startswith("s3://"):
        try:
            import s3fs  # type: ignore
        except Exception as e:
            raise SystemExit("Install s3fs to read s3:// URIs (pip install s3fs)") from e
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(uri, "rb") as f:
            return f.read()
    if uri.startswith("http://") or uri.startswith("https://"):
        try:
            import requests  # type: ignore
        except Exception as e:
            raise SystemExit("Install requests to read http(s):// URIs (pip install requests)") from e
        r = requests.get(uri, timeout=60)
        r.raise_for_status()
        return r.content
    # local path
    return Path(uri).read_bytes()

_NUM = re.compile(r"([-+]?\d+(?:\.\d+)?)")

def _load_json_any(uri: str) -> Dict[str, Any]:
    b = _read_uri(uri)
    # detect gzip by magic header or .gz suffix
    if (len(b) >= 2 and b[:2] == b"\x1f\x8b") or uri.endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(b)) as gz:
            txt = gz.read().decode("utf-8")
    else:
        txt = b.decode("utf-8")
    return json.loads(txt)

def _type_name(x: Any) -> str:
    if x is None: return "null"
    if isinstance(x, bool): return "bool"
    if isinstance(x, int): return "int"
    if isinstance(x, float): return "float"
    if isinstance(x, str): return "str"
    if isinstance(x, list): return "list"
    if isinstance(x, dict): return "dict"
    return type(x).__name__

def _classify_label(s: str) -> str:
    u = str(s).strip().upper()
    if u.startswith("CHECK"): return "CHECK"
    if u.startswith("CALL"):  return "CALL"
    if u.startswith("FOLD"):  return "FOLD"
    if u.startswith("ALLIN") or "ALL-IN" in u or "ALL IN" in u: return "ALLIN"
    if u.startswith("RAISE") or "RE-RAISE" in u or "RERAISE" in u: return "RAISE"
    if u.startswith("BET") or "DONK" in u: return "BET"  # DONK often encoded as BET by OOP
    return "OTHER"

def _extract_number(s: str) -> float | None:
    m = _NUM.findall(str(s))
    if not m: return None
    # Use the last number in label; some labels have both "RAISE TO X" where X is last token
    try:
        return float(m[-1])
    except Exception:
        return None

def _paths_collect(d: Any, prefix: str, paths: Dict[str, Set[str]], limit_lists: bool = True):
    """Collect key paths -> observed types (sets)."""
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else k
            paths.setdefault(p, set()).add(_type_name(v))
            _paths_collect(v, p, paths, limit_lists)
    elif isinstance(d, list):
        if not d:
            paths.setdefault(prefix + "[]", set()).add("list_empty")
            return
        # Record element type
        paths.setdefault(prefix + "[]", set()).add(_type_name(d[0]))
        # Only descend into first dict element to avoid explosion
        if isinstance(d[0], dict):
            _paths_collect(d[0], prefix + "[]", paths, limit_lists)

def _traverse_collect(node: Dict[str, Any],
                      stats: Dict[str, Any],
                      depth: int = 0):
    # node_type histogram
    nt = str(node.get("node_type", "")).lower() if isinstance(node, dict) else ""
    if nt:
        stats["node_type_hist"][nt] += 1

    # actions present at this node
    actions = node.get("actions") or []
    strat = node.get("strategy") or {}
    strat_actions = strat.get("actions") or []
    childrens = node.get("childrens") or {}

    # record counters
    for a in actions:
        stats["node_actions"][str(a)] += 1
        cls = _classify_label(a)
        stats["node_action_classes"][cls] += 1
        if cls in ("BET", "RAISE", "ALLIN"):
            v = _extract_number(a)
            if v is not None:
                stats["label_num_samples"][cls].append(v)

    for a in strat_actions:
        stats["strategy_actions"][str(a)] += 1
        cls = _classify_label(a)
        stats["strategy_action_classes"][cls] += 1
        if cls in ("BET", "RAISE", "ALLIN"):
            v = _extract_number(a)
            if v is not None:
                stats["label_num_samples_strat"][cls].append(v)

    for k in childrens.keys():
        stats["children_keys"][str(k)] += 1
        cls = _classify_label(k)
        stats["children_key_classes"][cls] += 1
        if cls in ("BET", "RAISE", "ALLIN"):
            v = _extract_number(k)
            if v is not None:
                stats["label_num_samples_child"][cls].append(v)

    # strategy vector length diagnostics
    strat_map = strat.get("strategy") or {}
    if isinstance(strat_map, dict) and strat_actions:
        # take first example to avoid huge memory use
        it = next(iter(strat_map.values()), None)
        if isinstance(it, list):
            stats["strategy_vec_len_hist"][len(it)] += 1
        stats["strategy_hand_count"] += len(strat_map)

    # recurse
    if isinstance(childrens, dict):
        # Avoid pathological breadth; still visit all keys but only descend into dict-like children
        for _, child in childrens.items():
            if isinstance(child, dict):
                _traverse_collect(child, stats, depth + 1)

def main():
    ap = argparse.ArgumentParser("Scan solver JSON schema & action label variants")
    ap.add_argument("input", help="Path/URL to solver JSON (.json or .json.gz). Supports file, http(s), s3.")
    ap.add_argument("--out", default="solver_schema_report.json", help="Where to write structured JSON report")
    ap.add_argument("--topk", type=int, default=30, help="How many distinct labels to print per section")
    args = ap.parse_args()

    js = _load_json_any(args.input)
    root = js.get("root") or js
    if not isinstance(root, dict):
        print("ERROR: root is not a dict", file=sys.stderr)
        sys.exit(2)

    # 1) Key paths & types
    paths: Dict[str, Set[str]] = {}
    _paths_collect(root, "", paths)

    # 2) Deep stats
    stats: Dict[str, Any] = dict(
        node_type_hist=Counter(),
        node_actions=Counter(),
        strategy_actions=Counter(),
        children_keys=Counter(),
        node_action_classes=Counter(),
        strategy_action_classes=Counter(),
        children_key_classes=Counter(),
        label_num_samples=defaultdict(list),            # from node.actions
        label_num_samples_strat=defaultdict(list),      # from strategy.actions
        label_num_samples_child=defaultdict(list),      # from childrens keys
        strategy_vec_len_hist=Counter(),
        strategy_hand_count=0,
    )
    _traverse_collect(root, stats)

    # 3) Compute union/mismatches for awareness
    node_lbls = set(stats["node_actions"].keys())
    strat_lbls = set(stats["strategy_actions"].keys())
    child_lbls = set(stats["children_keys"].keys())

    only_in_node = sorted(node_lbls - strat_lbls)[:args.topk]
    only_in_strat = sorted(strat_lbls - node_lbls)[:args.topk]
    only_in_child = sorted(child_lbls - (node_lbls | strat_lbls))[:args.topk]

    # ---- HUMAN SUMMARY ----
    print(f"\n=== KEYS & TYPES (sampled) ===")
    for p in sorted(paths.keys()):
        types = ",".join(sorted(paths[p]))
        print(f"{p} : {types}")

    print("\n=== node_type histogram ===")
    for k, v in stats["node_type_hist"].most_common():
        print(f"{k:15s} : {v}")

    def _print_counter(title: str, c: Counter):
        print(f"\n=== {title} (top {args.topk}) ===")
        for k, v in c.most_common(args.topk):
            print(f"{k:30s} : {v}")

    _print_counter("node.actions labels", stats["node_actions"])
    _print_counter("strategy.actions labels", stats["strategy_actions"])
    _print_counter("childrens keys", stats["children_keys"])

    print("\n=== Class breakdown (counts) ===")
    for name in ("node_action_classes", "strategy_action_classes", "children_key_classes"):
        print(f"{name:28s} : {dict(stats[name])}")

    print("\n=== Label numeric samples (first few) ===")
    for src, dd in [("node.actions", stats["label_num_samples"]),
                    ("strategy.actions", stats["label_num_samples_strat"]),
                    ("childrens keys", stats["label_num_samples_child"])]:
        print(f"- {src}:")
        for cls, nums in dd.items():
            show = ", ".join(str(n) for n in nums[:10])
            print(f"    {cls:7s} -> [{show}]")

    print("\n=== Strategy vector lens & hand-count ===")
    print("length histogram:", dict(stats["strategy_vec_len_hist"]))
    print("strategy_hand_count:", stats["strategy_hand_count"])

    print("\n=== Label set mismatches (first few) ===")
    print("only_in_node.actions         :", only_in_node)
    print("only_in_strategy.actions     :", only_in_strat)
    print("only_in_childrens_keys       :", only_in_child)

    # ---- MACHINE-READABLE REPORT ----
    report = {
        "input": args.input,
        "paths_types": {p: sorted(list(t)) for p, t in paths.items()},
        "node_type_hist": dict(stats["node_type_hist"]),
        "node_actions_top": stats["node_actions"].most_common(args.topk),
        "strategy_actions_top": stats["strategy_actions"].most_common(args.topk),
        "children_keys_top": stats["children_keys"].most_common(args.topk),
        "class_counts": {
            "node": dict(stats["node_action_classes"]),
            "strategy": dict(stats["strategy_action_classes"]),
            "children": dict(stats["children_key_classes"]),
        },
        "label_numeric_examples": {
            "node_actions": {k: v[:10] for k, v in stats["label_num_samples"].items()},
            "strategy_actions": {k: v[:10] for k, v in stats["label_num_samples_strat"].items()},
            "children_keys": {k: v[:10] for k, v in stats["label_num_samples_child"].items()},
        },
        "strategy_vec_len_hist": dict(stats["strategy_vec_len_hist"]),
        "strategy_hand_count": stats["strategy_hand_count"],
        "mismatches": {
            "only_in_node_actions": only_in_node,
            "only_in_strategy_actions": only_in_strat,
            "only_in_children_keys": only_in_child,
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n✅ Wrote report: {args.out}")

if __name__ == "__main__":
    main()