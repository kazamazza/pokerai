# tools/rangenet/sanity/probe_facing_bet_stats.py
from __future__ import annotations
import argparse, json, gzip, io, re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, List
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client
from ml.etl.rangenet.postflop.build_rangenet_postflop_dataset import _cache_path_for_key, _retry, _role_from_menu
from ml.utils.config import load_model_config



NUM = re.compile(r"([-+]?\d+(?:\.\d+)?)")

def _extract_last_number(s: str) -> Optional[float]:
    m = NUM.findall(str(s));
    return float(m[-1]) if m else None

def _norm(s: str) -> Tuple[str, Optional[float]]:
    u = str(s).strip().upper()
    return (u.split()[0] if u else u), _extract_last_number(u)

def _resolve_child(children: Mapping[str, Any], action_label: str, eps: float = 1e-3) -> Mapping[str, Any]:
    if not children: return {}
    if action_label in children: return children[action_label]
    vt, nt = _norm(action_label)
    for k, v in children.items():
        vk, nk = _norm(k)
        if vk != vt: continue
        if (nt is None and nk is None) or (nt is not None and nk is not None and abs(nt - nk) <= eps):
            return v
    return {}

def _avg_mix_over(ordered_actions: List[str], strat_map: Mapping[str, List[float]]) -> np.ndarray:
    if not ordered_actions or not strat_map:
        return np.zeros(0, dtype=np.float64)
    k = len(ordered_actions)
    mass = np.zeros(k, dtype=np.float64); n = 0
    for probs in strat_map.values():
        arr = np.asarray(probs, dtype=np.float64)
        if arr.size == 0: continue
        L = min(arr.size, k)
        mass[:L] += np.clip(arr[:L], 0.0, None); n += 1
    if n == 0: return np.zeros(0, dtype=np.float64)
    s = float(mass.sum())
    return mass / s if s > 0 else mass

def _actions_and_mix_union(node: Mapping[str, Any]) -> Tuple[List[str], np.ndarray]:
    strat = node.get("strategy") or {}
    s_actions = [str(a) for a in (strat.get("actions") or [])]
    n_actions = [str(a) for a in (node.get("actions") or [])]
    if not s_actions and not n_actions:
        n_actions = list((node.get("childrens") or {}).keys())
    union = list(s_actions); seen = set(s_actions)
    for a in n_actions:
        if a not in seen:
            union.append(a); seen.add(a)
    mix = _avg_mix_over(s_actions, strat.get("strategy") or {})
    if not union:
        return [], np.zeros(0, dtype=np.float64)
    if mix.size < len(union):
        pad = np.zeros(len(union), dtype=np.float64)
        if mix.size: pad[:mix.size] = mix
        mix = pad
    return union, mix

def _load_solver_json_local_or_s3(cfg: Mapping[str, Any], s3c: S3Client, s3_key: str) -> dict:
    local_path = _cache_path_for_key(cfg, s3_key)
    if not local_path.is_file():
        _retry(lambda: s3c.download_file_if_missing(s3_key, local_path))
    b = local_path.read_bytes()
    if local_path.suffix == ".gz" or (len(b) >= 2 and b[:2] == b"\x1f\x8b"):
        with gzip.GzipFile(fileobj=io.BytesIO(b)) as gz:
            text = gz.read().decode("utf-8")
        return json.loads(text)
    return json.loads(b.decode("utf-8"))

def main():
    ap = argparse.ArgumentParser("Stats + first OOP facing-bet example")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-scan", type=int, default=10000)
    ap.add_argument("--random", action="store_true", help="random sample instead of head")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    df = pd.read_parquet(args.manifest)
    if args.random and len(df) > args.max_scan:
        df = df.sample(args.max_scan, random_state=args.seed).reset_index(drop=True)
    elif len(df) > args.max_scan:
        df = df.head(args.max_scan).reset_index(drop=True)

    s3c = S3Client()

    total = len(df)
    oop_rows = 0
    oop_root_has_bet = 0
    oop_root_bet_mass_rows = 0

    for _, r in df.iterrows():
        s3_key = str(r["s3_key"])
        role = _role_from_menu(str(r.get("bet_sizing_id","") or "")).upper()
        actor = "ip" if role.endswith("_IP") else ("oop" if role.endswith("_OOP") else "ip")
        if actor != "oop":
            continue
        oop_rows += 1

        js = _load_solver_json_local_or_s3(cfg, s3c, s3_key)
        root = js.get("root") or js
        ch = root.get("childrens") or {}

        root_actions, root_mix = _actions_and_mix_union(root)
        if root_mix.size == 0 or float(root_mix.sum()) <= 0:
            continue

        has_bet_any = any(str(a).upper().startswith("BET") for a in root_actions)
        if has_bet_any: oop_root_has_bet += 1

        bet_idxs = [i for i, lab in enumerate(root_actions) if str(lab).upper().startswith("BET") and root_mix[i] > 0]
        if not bet_idxs:
            continue
        oop_root_bet_mass_rows += 1

        i = bet_idxs[0]
        root_lab = root_actions[i]
        w_root = float(root_mix[i])

        node = _resolve_child(ch, root_lab)
        node_actions, node_mix = _actions_and_mix_union(node)

        print("\n=== FIRST OOP FACING-BET ===")
        print("s3_key:", s3_key)
        print("role:", role, "actor:", actor)
        print("root_actions:", root_actions)
        print("root_mix    :", [float(x) for x in root_mix])
        print("chosen root BET:", root_lab, "mix:", w_root)
        print("child actions:", node_actions)
        print("child mix    :", [float(x) for x in node_mix] if node_mix.size else [])
        has_raise = any(("RAISE" in str(a).upper()) or ("RE-RAISE" in str(a).upper()) for a in node_actions)
        print("has RAISE in child?:", has_raise)
        return

    print("\n=== STATS (no example found) ===")
    print(f"rows scanned         : {total}")
    print(f"rows with actor==OOP : {oop_rows}")
    print(f"OOP rows root has BET label(s) : {oop_root_has_bet}")
    print(f"OOP rows with BET mass > 0     : {oop_root_bet_mass_rows}")
    print("Interpretation: if the last number is 0, your sample doesn't contain an OOP-facing-bet state; widen sample or select menus that guarantee IP bets at root (e.g., PFR_IP vs CALLER_OOP in SRP/3BP).")

if __name__ == "__main__":
    main()