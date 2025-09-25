#!/usr/bin/env python
import json, gzip, sys, os
from pathlib import Path

def load_json(path: str) -> dict:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def print_node(node: dict, depth=0, max_depth=2):
    """Recursively print actions up to max_depth."""
    pad = "  " * depth
    actions = node.get("actions") or list((node.get("childrens") or {}).keys())
    if actions:
        print(f"{pad}actions: {actions}")
    else:
        print(f"{pad}(no actions)")

    if depth < max_depth:
        for act, child in (node.get("childrens") or {}).items():
            print(f"{pad}→ {act}")
            print_node(child, depth+1, max_depth)

def main():
    if len(sys.argv) < 2:
        print("Usage: python peek_strategy.py <output_result.json[.gz]>")
        sys.exit(1)

    path = sys.argv[1]
    if not Path(path).exists():
        print(f"File not found: {path}")
        sys.exit(1)

    data = load_json(path)

    # Some files have root at top, others under key 'root'
    root = data.get("root", data)
    print("=== ROOT ===")
    print_node(root, depth=0, max_depth=2)

if __name__ == "__main__":
    main()