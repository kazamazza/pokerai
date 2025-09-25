#!/usr/bin/env python3
import argparse, gzip, json, re
from pathlib import Path
import boto3

def load_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def parse_menu_from_key(key: str) -> str:
    m = re.search(r"sizes=([^/]+)/", key)
    return m.group(1) if m else "unknown"

def root_node(p: dict) -> dict:
    return p.get("root", p)

def children(n: dict) -> dict:
    ch = n.get("childrens") or n.get("children") or {}
    return ch if isinstance(ch, dict) else {}

def actions_and_mix(n: dict):
    acts = list(n.get("actions") or [])
    strat = n.get("strategy") or {}
    if not acts and isinstance(strat, dict):
        acts = list(strat.get("actions") or [])
    m = []
    smap = strat.get("strategy") if isinstance(strat, dict) else None
    if acts and isinstance(smap, dict) and smap:
        m = [0.0] * len(acts); cnt = 0
        for probs in smap.values():
            if isinstance(probs, list):
                L = min(len(probs), len(acts))
                for i in range(L):
                    v = probs[i]
                    if v is not None and v >= 0:
                        m[i] += float(v)
                cnt += 1
        if cnt:
            s = sum(m)
            if s > 0:
                m = [x / s for x in m]
    return acts, m

def first_bet_child_with_mass(n: dict):
    acts, mix = actions_and_mix(n)
    ch = children(n)
    best = (-1.0, None, None)
    for i, a in enumerate(acts):
        if str(a).upper().startswith("BET"):
            w = mix[i] if i < len(mix) and mix else 0.0
            nxt = ch.get(a) or ch.get(str(a))
            if isinstance(nxt, dict) and w >= best[0]:
                best = (w, str(a), nxt)
    if best[1] is not None:
        return best[1], best[2], best[0]
    # fallback: any BET child
    for k, v in ch.items():
        if str(k).upper().startswith("BET") and isinstance(v, dict):
            return str(k), v, 0.0
    return None

def has_raise_anywhere(n: dict, max_nodes=10000) -> bool:
    stack = [n]; seen = 0
    while stack and seen < max_nodes:
        cur = stack.pop(); seen += 1
        acts, _ = actions_and_mix(cur)
        for a in acts:
            up = str(a).upper()
            if up.startswith("RAISE") or up == "ALLIN":
                return True
        for v in children(cur).values():
            if isinstance(v, dict):
                stack.append(v)
    return False

def drill_one(s3, bucket: str, key: str):
    menu = parse_menu_from_key(key)
    tmp = Path("/tmp") / key.replace("/", "_")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(tmp))
    data = load_json_gz(tmp)
    root = root_node(data)
    r_acts, r_mix = actions_and_mix(root)
    pick = first_bet_child_with_mass(root)
    any_raise = has_raise_anywhere(root)

    print(f"\n== {menu} :: {key}")
    print(f"root_actions: {r_acts}")
    print(f"root_mix    : {['%.3f'%x for x in r_mix] if r_mix else '[]'}")

    if pick:
        bet_lab, child, w = pick
        c_acts, c_mix = actions_and_mix(child)
        print(f"chosen BET child: {bet_lab}  (mass≈{w:.3f})")
        print(f"child_actions   : {c_acts}")
        print(f"child_mix       : {['%.3f'%x for x in c_mix] if c_mix else '[]'}")
        print(f"child_has_raise?: {any(str(a).upper().startswith('RAISE') or str(a).upper()=='ALLIN' for a in c_acts)}")
    else:
        print("no BET child found at root")

    print(f"raise_anywhere  : {any_raise}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--keys", nargs="+", required=True, help="S3 keys to drill (space-separated)")
    args = ap.parse_args()

    s3 = boto3.client("s3")
    for k in args.keys:
        drill_one(s3, args.bucket, k)

if __name__ == "__main__":
    main()