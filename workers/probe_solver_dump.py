import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


# ----------------------------
# IO helpers
# ----------------------------
def open_json_any(path: str) -> Any:
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def short(x: Any, n: int = 600) -> str:
    try:
        s = json.dumps(x, ensure_ascii=False)
    except Exception:
        s = str(x)
    return s[:n] + ("…" if len(s) > n else "")


# ----------------------------
# Node resolution
# ----------------------------
def resolve_root(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    TexasSolver dumps in your case are often "payload is the node".
    Sometimes they can be wrapped in {"root": node}. Your probe showed root=None
    but top-level is already a node. So:
      - if payload["root"] is dict -> use it
      - elif payload looks like a node (has actions/childrens/strategy) -> use payload
    """
    r = payload.get("root")
    if isinstance(r, dict):
        return r, "payload['root']_is_node"

    # if payload itself looks like a node
    if any(k in payload for k in ("actions", "childrens", "strategy", "node_type", "player")):
        return payload, "payload_is_node"

    return None, "unresolved"


# ----------------------------
# Children iteration (CRITICAL)
# ----------------------------
def iter_children(node: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Yield (action_label, child_node) for TexasSolver shapes.

    Supported shapes:
      A) node["childrens"] is dict keyed by action label
      B) node["childrens"] is dict keyed by index (0..len(actions)-1)
      C) node["childrens"] is list aligned with node["actions"] list
      D) fallback: node["children"] dict (other dumps)
    """
    if not isinstance(node, dict):
        return

    # fallback alternate key
    ch_alt = node.get("children")
    if isinstance(ch_alt, dict):
        for a, sub in ch_alt.items():
            if isinstance(sub, dict):
                yield str(a), sub
        return

    actions = node.get("actions")
    childrens = node.get("childrens")

    # C) list aligned with actions
    if isinstance(actions, list) and isinstance(childrens, list) and len(actions) == len(childrens):
        for a, sub in zip(actions, childrens):
            if isinstance(sub, dict):
                yield str(a), sub
        return

    # A/B) dict keyed by label or index
    if isinstance(actions, list) and isinstance(childrens, dict):
        # A) by action label
        for a in actions:
            if a in childrens and isinstance(childrens[a], dict):
                yield str(a), childrens[a]

        # B) by index
        for i, a in enumerate(actions):
            if i in childrens and isinstance(childrens[i], dict):
                yield str(a), childrens[i]
            elif str(i) in childrens and isinstance(childrens[str(i)], dict):
                yield str(a), childrens[str(i)]

        # single-entry fallback
        if len(actions) == 1 and len(childrens) == 1:
            only_child = next(iter(childrens.values()))
            if isinstance(only_child, dict):
                yield str(actions[0]), only_child
        return

    # If childrens exists but no actions list, still try dict values
    if isinstance(childrens, dict):
        for k, sub in childrens.items():
            if isinstance(sub, dict):
                yield str(k), sub


def children_keys(node: Dict[str, Any], limit: int = 50) -> List[str]:
    out: List[str] = []
    for a, _ in iter_children(node):
        out.append(a)
        if len(out) >= limit:
            break
    return out


def find_child_by_action(node: Dict[str, Any], want: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    w = want.strip().upper()
    for a, sub in iter_children(node):
        if str(a).strip().upper() == w:
            return sub, str(a)
    return None, None


# ----------------------------
# Bet label parsing / pick
# ----------------------------
def parse_bet_size_frac(label: str) -> Optional[float]:
    """
    Extract bet sizing fraction from action label.
    Examples handled:
      "BET 33%", "BET_33", "BET 0.33", "BET(0.33)", "B33"
    Returns fraction like 0.33
    """
    s = str(label).strip().upper()

    # must look bet-like
    if "BET" not in s and not s.startswith("B"):
        return None
    if "ALLIN" in s or "ALL-IN" in s:
        return None

    # percent
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        pct = float(m.group(1))
        return pct / 100.0

    # decimal
    m = re.search(r"(\d+\.\d+)", s)
    if m:
        fr = float(m.group(1))
        if 0.0 < fr <= 2.0:
            return fr

    # integer token (BET_33, B33)
    m = re.search(r"\b(\d{1,3})\b", s)
    if m:
        pct = float(m.group(1))
        if 1 <= pct <= 200:
            return pct / 100.0

    return None


def pick_bet_child(node: Dict[str, Any], size_pct: Optional[int]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Choose the bet child closest to requested size_pct.
    If size_pct is None, return the first bet-like edge found.
    """
    target = (float(size_pct) / 100.0) if size_pct is not None else None
    best: Optional[Tuple[float, str, Dict[str, Any]]] = None

    for a, sub in iter_children(node):
        fr = parse_bet_size_frac(a)
        if fr is None:
            continue

        if target is None:
            return str(a), sub

        d = abs(fr - target)
        if best is None or d < best[0]:
            best = (d, str(a), sub)

    if best is None:
        return None, None
    return best[1], best[2]


# ----------------------------
# Summaries
# ----------------------------
def node_summary(node: Dict[str, Any]) -> Dict[str, Any]:
    actions = node.get("actions")
    childrens = node.get("childrens")
    return {
        "keys": sorted(list(node.keys()))[:30],
        "player": node.get("player"),
        "node_type": node.get("node_type"),
        "has_strategy": isinstance(node.get("strategy"), dict),
        "num_actions": len(actions) if isinstance(actions, list) else None,
        "actions_head": actions[:10] if isinstance(actions, list) else None,
        "childrens_type": str(type(childrens)),
        "child_edge_labels_head": children_keys(node, 20),
    }


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser("Deep probe TexasSolver dump (supports actions+childrens)")
    ap.add_argument("--path", required=True)
    ap.add_argument("--size-pct", type=int, default=None, help="e.g. 33")
    ap.add_argument("--dump-strategy-keys", action="store_true")
    args = ap.parse_args()

    payload = open_json_any(args.path)
    if not isinstance(payload, dict):
        print("TOP TYPE:", type(payload))
        return

    print("\nTOP KEYS:", sorted(payload.keys()))
    if "root" in payload:
        print("payload['root'] type:", type(payload.get("root")))

    root, how = resolve_root(payload)
    print("\nRESOLVE ROOT:", how)
    if root is None:
        print("❌ Could not resolve root node.")
        return

    print("\nROOT SUMMARY:", short(node_summary(root), 1200))

    # Explicit inspection
    if isinstance(root.get("actions"), list):
        print("\nROOT actions (first 30):", root["actions"][:30])

    ch = root.get("childrens")
    if isinstance(ch, dict):
        print("ROOT childrens is dict. keys (first 30):", list(ch.keys())[:30])
    elif isinstance(ch, list):
        print("ROOT childrens is list. len:", len(ch))
    else:
        print("ROOT childrens type:", type(ch))

    print("ROOT child actions via iter_children (first 50):", children_keys(root, 50))

    if isinstance(root.get("strategy"), dict):
        print("ROOT strategy keys (sample):", list(root["strategy"].keys())[:10])

    # Step 1: find CHECK child
    check_child, check_label = find_child_by_action(root, "CHECK")
    print("\nCHECK EDGE FOUND:", bool(check_child), "label=", check_label)

    if check_child is None:
        print("❌ No CHECK child at root.")
        return

    print("\nAFTER CHECK NODE SUMMARY:", short(node_summary(check_child), 1200))
    if isinstance(check_child.get("actions"), list):
        print("CHECK child actions (first 40):", check_child["actions"][:40])
    print("CHECK child child actions via iter_children (first 80):", children_keys(check_child, 80))

    # Step 2: find BET after check (IP node)
    bet_label, bet_child = pick_bet_child(check_child, args.size_pct)
    print("\nBET EDGE FOUND:", bool(bet_child), "label=", bet_label, "requested_size_pct=", args.size_pct)

    if bet_child is None:
        print("❌ No bet child found after CHECK.")
        avail = children_keys(check_child, 200)
        print("Available child actions after CHECK:", avail)

        print("\nParsed bet-like candidates after CHECK:")
        for a in avail:
            fr = parse_bet_size_frac(a)
            if fr is not None:
                print(" ", a, "->", fr)
        return

    print("\nAFTER BET NODE SUMMARY (this should be OOP response):", short(node_summary(bet_child), 1200))
    if isinstance(bet_child.get("actions"), list):
        print("BET child actions (first 50):", bet_child["actions"][:50])
    print("BET child child actions via iter_children (first 80):", children_keys(bet_child, 80))

    if args.dump_strategy_keys and isinstance(bet_child.get("strategy"), dict):
        print("\nBET child strategy keys sample:", list(bet_child["strategy"].keys())[:40])

    print("\n✅ Done. If 'BET child actions' includes FOLD/CALL/RAISE, your limp facing node is correct.")


if __name__ == "__main__":
    main()