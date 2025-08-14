# tools/preflop/select_regen_from_audit.py
import json, re
from pathlib import Path
from collections import defaultdict

ISSUES = Path("preflop_audit_issues.jsonl")

def parse_key_to_config(s3_key: str) -> dict:
    # e.g. preflop/ranges/profile=GTO/exploit=GTO/multiway=HU/pop=REGULAR/action=VS_OPEN/BTN_vs_CO_100bb.json.gz
    parts = dict(p.split("=", 1) for p in s3_key.split("/") if "=" in p)
    fname = s3_key.rsplit("/", 1)[-1]
    m = re.match(r"(?P<ip>[A-Z]+)_vs_(?P<oop>[A-Z]+)_(?P<stack>\d+)bb\.json\.gz$", fname)
    if not m:
        raise ValueError(f"Unparsable filename: {fname}")
    return {
        "ip_position": m.group("ip"),
        "oop_position": m.group("oop"),
        "stack_bb": int(m.group("stack")),
        "villain_profile": parts["profile"],
        "exploit_setting": parts["exploit"],
        "multiway_context": parts["multiway"],
        "population_type": parts["pop"],
        "action_context": parts["action"],
    }

def main():
    missing, corrupt = set(), set()
    with ISSUES.open() as f:
        for line in f:
            row = json.loads(line)
            key = row.get("key")
            if not key:
                continue
            err = row.get("err", "")
            cfg = parse_key_to_config(key)
            tup = tuple(sorted(cfg.items()))
            if err in {"missing_file", "vso_load_fail"}:
                missing.add(tup)
            else:
                # bad_hand_token, illegal_bucket, empty_bucket, etc.
                corrupt.add(tup)

    def to_list(dict_tuples):
        return [dict(items) for items in dict_tuples]

    print(f"Missing: {len(missing)} | Corrupt: {len(corrupt)}")
    Path("preflop_to_regen_missing.json").write_text(json.dumps(to_list(missing), indent=2))
    Path("preflop_to_regen_corrupt.json").write_text(json.dumps(to_list(corrupt), indent=2))
    print("Wrote: preflop_to_regen_missing.json, preflop_to_regen_corrupt.json")

if __name__ == "__main__":
    main()