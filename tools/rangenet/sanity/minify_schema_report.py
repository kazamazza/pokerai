# tools/rangenet/sanity/minify_schema_report.py
import json, sys
from pathlib import Path

def take_top(counter_list, k=30):
    # input like [["BET 33.000000", 512], ["CHECK", 341], ...]
    return counter_list[:k]

def main():
    if len(sys.argv) < 2:
        print("usage: python minify_schema_report.py schema_report.json [topk]", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    rep = json.loads(path.read_text())

    out = {
        "node_type_hist": rep.get("node_type_hist", {}),
        "node_actions_top": take_top(rep.get("node_actions_top", []), topk),
        "strategy_actions_top": take_top(rep.get("strategy_actions_top", []), topk),
        "children_keys_top": take_top(rep.get("children_keys_top", []), topk),
        "mismatches": rep.get("mismatches", {}),
        "label_numeric_examples": rep.get("label_numeric_examples", {}),
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()