#!/usr/bin/env python3
# tools/rangenet/sanity/build_postflop_smoke.py
"""
Build a small postflop parquet from a short list of solved S3 keys (no manifest).
- Reuses the production extraction logic (root action -> 169 policy vector).
- Emits a minimal parquet aligned with the postflop dataset (using 'board', not clusters).
- Prints a PASS/FAIL summary with simple sanity checks.

Usage:
  python tools/rangenet/sanity/build_postflop_smoke.py \
    --bucket pokeraistore \
    --keys-file data/pilots_15.txt \
    --out data/artifacts/postflop_smoke.parquet
"""

from __future__ import annotations
import os, io, json, gzip, sys, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import boto3
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.range.solvers.utils.solver_extract import pick_root, range_to_vec169
from ml.range.solvers.utils.solver_json_extract import hand_to_index_169

def _get(cfg: Dict[str, Any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _s3_client(cfg: Dict[str, Any]):
    region = _get(cfg, "aws.region") or os.getenv("AWS_REGION") or "eu-central-1"
    return boto3.client("s3", region_name=region)

def _read_json_bytes(b: bytes, key_hint: str) -> dict:
    if key_hint.endswith(".gz"):
        b = gzip.GzipFile(fileobj=io.BytesIO(b)).read()
    return json.loads(b)

def _load_solver_json(cfg: Dict[str, Any], bucket: str, s3_key: str) -> dict:
    s3 = _s3_client(cfg)
    body = s3.get_object(Bucket=bucket, Key=s3_key)["Body"].read()
    return _read_json_bytes(body, s3_key)

# ----------------- actor/action logic (mirrors prod) -----------------
def _role_from_menu(menu_id: str) -> str:
    m = (menu_id or "").strip()
    return m.split(".", 1)[1] if "." in m else ""

def _group_from_menu(menu_id: str) -> str:
    m = (menu_id or "").strip()
    return m.split(".", 1)[0] if "." in m else m

def _donk_available(menu_id: str, actor: str) -> bool:
    grp = _group_from_menu(menu_id)
    role = _role_from_menu(menu_id)
    if grp.startswith("limped_multi"):
        return False
    if (actor or "").lower() != "oop":
        return False
    return ("Caller_OOP" in role) or grp.startswith("limped_single")

def _infer_actor_from_menu(menu_id: str) -> str:
    role = _role_from_menu(menu_id).upper()
    if role.endswith("_IP"):  return "ip"
    if role.endswith("_OOP"): return "oop"
    return "ip"

def _split_positions(positions: str) -> Tuple[str, str]:
    s = str(positions).upper()
    if "V" in s:
        a, b = s.split("V", 1)
        return a, b
    return ("IP","OOP")

def _extract_best_policy_169(js: dict, *, node_key: str) -> tuple[Optional[np.ndarray], str]:
    root = pick_root(js, node_key=node_key)
    childrens = root.get("childrens") or {}
    child_labels = [str(k).upper() for k in childrens.keys()]
    strat_map = (root.get("strategy") or {}).get("strategy") or {}

    def _take(prefix: str) -> Optional[np.ndarray]:
        if not strat_map or not child_labels:
            return None
        want = prefix.upper()
        idxs = [i for i, lab in enumerate(child_labels) if lab.startswith(want)]
        if not idxs:
            return None
        v = np.zeros(169, dtype=np.float32)
        for hand, probs in strat_map.items():
            hi = hand_to_index_169(str(hand))
            if hi is None:
                continue
            try:
                s = sum(float(probs[i]) for i in idxs)
                v[hi] = max(0.0, min(1.0, s))
            except Exception:
                pass
        return v if np.any(v) else None

    # Try actions in this order
    for act in ("BET", "DONK", "CHECK"):
        v = _take(act)
        if v is not None:
            return v, act

    # Fallback: root ranges (rare)
    rmap = {}
    if isinstance(root.get("ranges"), dict):
        rmap = (root["ranges"].get("ip") or root["ranges"].get("oop") or {})
    elif isinstance(root.get("actors"), dict):
        act = root["actors"].get("ip") or root["actors"].get("oop") or {}
        rmap = act.get("range") or act.get("ranges") or {}
    if rmap:
        return range_to_vec169(rmap), "RANGE"

    return None, "NONE"

# ----------------- key parsing -----------------
def _parts_from_keypath(k: str) -> dict:
    # expects .../street=1/pos=BTNvSB/stack=25/pot=2/board=8dQdQc/acc=0.02/sizes=limped_multi.Any/.../output_result.json.gz
    parts = {}
    for seg in Path(k).parts:
        if "=" in seg:
            a, b = seg.split("=", 1)
            parts[a] = b
    return parts

def _infer_ctx_from_sizes(s: str) -> str:
    g = _group_from_menu(s)
    if   g.startswith("srp"):           return "SRP"
    elif g.startswith("3bet"):          return "3BET"
    elif g.startswith("4bet"):          return "4BET"
    elif g.startswith("limped_single"): return "LIMP_HU"
    elif g.startswith("limped_multi"):  return "LIMP_MULTI"
    return "UNKNOWN"

# ----------------- main builder -----------------
def main():
    ap = argparse.ArgumentParser("Build a tiny postflop parquet from a few solved S3 keys (smoke test).")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--keys-file", required=True, help="Text file with one S3 key per line")
    ap.add_argument("--out", default="data/artifacts/postflop_smoke.parquet")
    ap.add_argument("--normalize", action="store_true", help="L1-normalize label vectors")
    args = ap.parse_args()

    cfg: Dict[str, Any] = {}

    # collect keys
    keys: List[str] = []
    with open(args.keys_file, "r", encoding="utf-8") as f:
        for line in f:
            k = line.strip()
            if k and not k.startswith("#"):
                keys.append(k)
    if not keys:
        print("No keys to process from --keys-file.")
        sys.exit(2)

    rows: List[Dict[str, Any]] = []
    skipped_empty = 0
    degenerate = 0

    for key in keys:
        try:
            js = _load_solver_json(cfg, args.bucket, key)
        except Exception as e:
            sys.stderr.write(f"[skip] {key}: {e}\n")
            continue

        meta = _parts_from_keypath(key)
        positions = meta.get("pos", "")
        ip_pos, oop_pos = _split_positions(positions)
        stack_bb = int(round(float(meta.get("stack", "0") or "0")))
        pot_bb   = float(meta.get("pot", "0") or "0")
        street   = int(meta.get("street", "1") or "1")
        board    = meta.get("board", "")
        sizes_id = meta.get("sizes", "")
        ctx      = _infer_ctx_from_sizes(sizes_id)
        node_key = "root"

        actor = _infer_actor_from_menu(sizes_id)
        action = "DONK" if _donk_available(sizes_id, actor) else "BET"

        # preferred: action-conditioned policy at root
        vec, src = _extract_best_policy_169(js, node_key=node_key)
        if vec is None or not np.any(vec):
            skipped_empty += 1
            continue

        if args.normalize:
            s = float(vec.sum())
            if s > 0:
                vec = vec / s

        hero_pos = ip_pos if actor == "ip" else oop_pos

        row = {
            "stack_bb": stack_bb,
            "pot_bb": pot_bb,
            "hero_pos": hero_pos,
            "ip_pos": ip_pos,
            "oop_pos": oop_pos,
            "street": street,
            "ctx": ctx,
            "board": board,
            "bet_sizing_id": sizes_id,
            "actor": actor,
            "action": src,  # record actual extraction source: BET/DONK/CHECK/RANGE
            "weight": 1.0,
        }
        for i, v in enumerate(vec.tolist()):
            row[f"y_{i}"] = float(v)

        # quick degeneracy check (too spiky or near-zero mass)
        mean_mass = float(np.mean(vec))
        if mean_mass < 1e-6 or mean_mass > 0.9:
            degenerate += 1

        rows.append(row)

    # write parquet
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)

    # ---- summary / PASS-FAIL ----
    n = len(df)
    print(f"✅ wrote {out} with {n} rows  (skipped empty: {skipped_empty}, degenerate: {degenerate})")

    # Heuristics: ≥10 rows, ≤20% empty, ≤50% degenerate → PASS
    pass_rows = (n >= 10)
    pass_empty = (skipped_empty <= max(1, int(0.2 * (n + skipped_empty))))
    pass_degen = (degenerate <= max(2, int(0.5 * n)))

    verdict = "PASS" if (pass_rows and pass_empty and pass_degen) else "FAIL"
    print(f"[SMOKE] {verdict} | rows_ok={pass_rows} empty_ok={pass_empty} degenerate_ok={pass_degen}")

if __name__ == "__main__":
    main()