# tools/rangenet/sanity/probe_one_case.py
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
from ml.etl.rangenet.postflop.build_rangenet_postflop_dataset import _cache_path_for_key, _retry, _role_from_menu, \
    ACTION_VOCAB, VOCAB_INDEX, bucket_raise_label, bucket_bet_label

# ------- import project helpers -------
# Assumes these are importable in your env:
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
    mass = np.zeros(k, dtype=np.float64)
    n = 0
    for probs in strat_map.values():
        arr = np.asarray(probs, dtype=np.float64)
        if arr.size == 0: continue
        L = min(arr.size, k)
        mass[:L] += np.clip(arr[:L], 0.0, None)
        n += 1
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
    ap = argparse.ArgumentParser("Probe a single OOP-facing-bet node to debug missing RAISE")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-scan", type=int, default=2000, help="Rows to scan looking for OOP facing bet")
    args = ap.parse_args()

    from ml.utils.config import load_model_config
    cfg = load_model_config(args.config)
    df = pd.read_parquet(args.manifest)
    if len(df) > args.max_scan:
        df = df.head(args.max_scan)

    s3c = S3Client()
    found = False

    for _, r in df.iterrows():
        s3_key = str(r["s3_key"])
        menu_id = str(r.get("bet_sizing_id","") or "")
        role = _role_from_menu(menu_id).upper()
        actor = "ip" if role.endswith("_IP") else ("oop" if role.endswith("_OOP") else "ip")
        if actor != "oop":
            continue  # we need OOP cases

        js = _load_solver_json_local_or_s3(cfg, s3c, s3_key)
        root = js.get("root") or js
        ch = root.get("childrens") or {}

        root_actions, root_mix = _actions_and_mix_union(root)
        if root_mix.size == 0 or float(root_mix.sum()) <= 0:
            continue

        # pick the first BET at root with nonzero mix
        bet_idxs = [i for i, lab in enumerate(root_actions) if str(lab).upper().startswith("BET") and root_mix[i] > 0]
        if not bet_idxs:
            continue

        i = bet_idxs[0]
        root_lab = root_actions[i]
        w_root = float(root_mix[i])
        up_root = str(root_lab).upper()

        node = _resolve_child(ch, root_lab)
        print("\n=== PROBE ===")
        print("s3_key:", s3_key)
        print("role:", role, "actor:", actor)
        print("root_actions:", root_actions)
        print("root_mix    :", root_mix.tolist())
        print("chosen root BET:", root_lab, "mix:", w_root)

        if not node:
            print("!! child_resolve_fail; children keys (first 20):", list(ch.keys())[:20])
            return

        print("child node_type:", node.get("node_type"))
        node_actions, node_mix = _actions_and_mix_union(node)
        print("node_actions:", node_actions)
        print("node_mix    :", node_mix.tolist() if node_mix.size else [])
        print("strategy.actions at child:", (node.get("strategy", {}) or {}).get("actions"))

        # classify a bit
        has_raise = any("RAISE" in str(a).upper() or "RE-RAISE" in str(a).upper() for a in node_actions)
        print("has RAISE in node_actions?:", has_raise)

        if node_mix.size == 0:
            print("!! zero_mass_child")
            return

        # compute contribution into our vocab to verify bucketing
        pot_bb = float(r["pot_bb"])
        vec = np.zeros(len(ACTION_VOCAB), dtype=np.float64)
        curr = _extract_last_number(up_root) or 0.0
        for j, a_lab in enumerate(node_actions):
            mass = float(w_root * node_mix[j])
            if mass <= 0: continue
            aup = str(a_lab).upper()
            if aup.startswith("CALL"):
                vec[VOCAB_INDEX["CALL"]] += mass
            elif aup.startswith("FOLD"):
                vec[VOCAB_INDEX["FOLD"]] += mass
            elif aup.startswith("ALLIN") or aup.startswith("RAISE") or "RE-RAISE" in aup:
                rb = bucket_raise_label(aup, current_bet_bb=(curr or 1.0))
                vec[VOCAB_INDEX[rb]] += mass
            elif aup.startswith("BET"):
                b = bucket_bet_label(aup, pot_bb=pot_bb, actor="oop")
                vec[VOCAB_INDEX[b]] += mass
            elif aup.startswith("CHECK"):
                vec[VOCAB_INDEX["CHECK"]] += mass

        s = vec.sum()
        print("bucketed vec sum:", s, "vec (nonzeros):", {ACTION_VOCAB[i]: float(v) for i, v in enumerate(vec) if v > 0})
        found = True
        break

    if not found:
        print("No OOP facing-bet case found in the scanned rows. Either the sample is IP-only at root or bets had zero mass.")

if __name__ == "__main__":
    main()