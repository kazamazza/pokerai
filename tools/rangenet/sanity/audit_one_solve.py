#!/usr/bin/env python3
import argparse, json, gzip, io, sys, re
from typing import Any, Dict, List, Tuple, Optional
import boto3

def load_json_from_s3_gz(s3_uri: str) -> Dict[str, Any]:
    # s3://bucket/key...
    if not s3_uri.startswith("s3://"):
        raise SystemExit("s3_uri must start with s3://")
    _, rest = s3_uri.split("s3://", 1)
    bucket, key = rest.split("/", 1)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    by = obj["Body"].read()
    try:
        data = gzip.decompress(by)
    except OSError:
        # already plain json
        data = by
    return json.loads(data.decode("utf-8"))

# ----- schema-tolerant helpers -----
def children_of(node: Dict[str, Any]) -> Dict[str, Any]:
    # some dumps use "children", others "childrens"
    return (node.get("children") or node.get("childrens") or {}) if isinstance(node, dict) else {}

def node_actions(node: Dict[str, Any]) -> List[str]:
    if not isinstance(node, dict): return []
    # try node["actions"], else strategy.actions
    acts = node.get("actions")
    if isinstance(acts, list) and acts: return [str(a) for a in acts]
    strat = node.get("strategy") or {}
    acts = strat.get("actions")
    if isinstance(acts, list) and acts: return [str(a) for a in acts]
    # some dumps embed at strategy.strategy.actions (rare)
    strat2 = strat.get("strategy") or {}
    acts = strat2.get("actions")
    if isinstance(acts, list) and acts: return [str(a) for a in acts]
    return []

def node_mix(node: Dict[str, Any]) -> List[float]:
    strat = node.get("strategy") or {}
    probs = strat.get("strategy")  # usually a dict of combos->vec or, at root, per-action mix not present
    # we’ll skip extracting the exact mix here; for the auditor we mainly want actions presence.
    return []

def label_num(s: str) -> Optional[float]:
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else None

# ----- traversal to find facing-bet with raises -----
def find_first_facing_bet_with_children(root: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """Return (bet_label_at_root, child_actions) if root had a BET and that child lists actions; else (None, [])."""
    ch = children_of(root)
    acts = node_actions(root)
    # fall back to keys if actions missing
    root_action_labels = acts if acts else list(ch.keys())
    bet_label = None
    for lab in root_action_labels:
        up = str(lab).upper()
        if up.startswith("BET"):
            bet_label = lab
            break
    if bet_label is None:
        return None, []
    node = ch.get(bet_label) or ch.get(str(bet_label)) or {}
    return str(bet_label), node_actions(node) or list(children_of(node).keys())

def scan_for_raises_anywhere(node: Dict[str, Any], limit_nodes: int = 2000) -> Tuple[bool, int]:
    """BFS up to limit_nodes; return (found_raise, visited_count)."""
    q = [node]; seen = 0
    while q and seen < limit_nodes:
        cur = q.pop(0); seen += 1
        acts = node_actions(cur)
        if any(str(a).upper().startswith(("RAISE","ALLIN")) for a in acts):
            return True, seen
        ch = children_of(cur)
        for k, nxt in ch.items():
            if isinstance(nxt, dict):
                q.append(nxt)
    return False, seen

def summarize_top(root_like: Dict[str, Any]) -> Dict[str, Any]:
    root = root_like.get("root") if isinstance(root_like, dict) and "root" in root_like else root_like
    top = {
        "top_level_keys": sorted(list(root_like.keys())) if isinstance(root_like, dict) else type(root_like).__name__,
        "root_type": type(root).__name__,
        "root_keys": sorted(list(root.keys())) if isinstance(root, dict) else [],
        "root_actions": node_actions(root),
        "root_children_keys": sorted(list(children_of(root).keys()))[:10],
    }
    bet_at_root, child_actions = find_first_facing_bet_with_children(root)
    top["bet_at_root"] = bet_at_root
    top["child_actions_for_that_bet"] = child_actions[:10]
    found_raise, visited = scan_for_raises_anywhere(root)
    top["any_raise_found_anywhere"] = found_raise
    top["nodes_scanned"] = visited
    return top

def main():
    ap = argparse.ArgumentParser(description="Audit a single solver JSON(.gz) from S3 and print structural facts.")
    ap.add_argument("s3_uri", help="e.g. s3://bucket/path/output_result.json.gz")
    args = ap.parse_args()

    js = load_json_from_s3_gz(args.s3_uri)
    rep = summarize_top(js)

    print("\n=== STRUCTURE ===")
    print("top_level_keys:", rep["top_level_keys"])
    print("root_keys     :", rep["root_keys"])
    print("root_actions  :", rep["root_actions"][:12])
    print("root_children :", rep["root_children_keys"])

    print("\n=== ROOT BET BRANCH ===")
    print("bet_at_root   :", rep["bet_at_root"])
    print("child_actions :", rep["child_actions_for_that_bet"])

    print("\n=== GLOBAL RAISE PRESENCE ===")
    print("any_raise_found_anywhere:", rep["any_raise_found_anywhere"])
    print("nodes_scanned           :", rep["nodes_scanned"])

    # quick verdicts
    problems = []
    if not rep["root_actions"] and not rep["root_children_keys"]:
        problems.append("missing root actions and children")
    if (rep["bet_at_root"] is None):
        problems.append("no BET action at root")
    if not rep["any_raise_found_anywhere"]:
        problems.append("no RAISE found anywhere")

    print("\n=== VERDICT ===")
    if problems:
        for p in problems:
            print(" -", p)
        print("\nConclusion: output schema/tree is not what a normal flop tree should look like; likely solver-side or schema drift.")
    else:
        print("Looks structurally fine (root has actions, has a BET with follow-up, and raises exist).")

if __name__ == "__main__":
    main()