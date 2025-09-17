from __future__ import annotations
import os, io, json, gzip
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

import boto3
import numpy as np
import pandas as pd

# ------------- project imports (adjust paths if needed) -------------
# Assumes this file sits under tools/rangenet/
ROOT_DIR = Path(__file__).resolve().parents[2]
import sys; sys.path.append(str(ROOT_DIR))
from ml.range.solvers.utils.range_utils import hand_to_index
from ml.utils.config import load_model_config


# -------------------------- util helpers ----------------------------

def _get(cfg: Mapping[str, Any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _s3_client(cfg: Mapping[str, Any]):
    region = _get(cfg, "aws.region") or os.getenv("AWS_REGION") or "eu-central-1"
    return boto3.client("s3", region_name=region)

def _local_cache_path(cfg: Mapping[str, Any], s3_key: str) -> Path:
    root = Path(_get(cfg, "worker.local_cache_dir", "data/solver_cache"))
    return (root / s3_key).resolve()

def _read_json_bytes(b: bytes, key_hint: str) -> dict:
    if key_hint.endswith(".gz"):
        b = gzip.GzipFile(fileobj=io.BytesIO(b)).read()
    return json.loads(b)

def _load_solver_json_local_or_s3(cfg: Mapping[str, Any], s3_key: str) -> dict:
    """
    Prefer local mirror of the S3 key; on miss, fetch from S3 and cache it locally
    (creating parent dirs), then return parsed JSON.
    """
    p = _local_cache_path(cfg, s3_key)
    if p.is_file():
        raw = p.read_bytes()
        return _read_json_bytes(raw, p.name)

    bucket = _get(cfg, "aws.bucket") or os.getenv("AWS_BUCKET_NAME")
    if not bucket:
        raise FileNotFoundError("No local cache file and aws.bucket not configured")

    s3 = _s3_client(cfg)
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    body = obj["Body"].read()

    # mirror to local cache
    p.parent.mkdir(parents=True, exist_ok=True)
    if s3_key.endswith(".gz"):
        p.write_bytes(body)  # keep gz intact
    else:
        p.write_bytes(body)

    return _read_json_bytes(body, s3_key)

def _extract_actor_range(js: dict, *, actor: str, node_key: str) -> Optional[Dict[str, float]]:
    """
    Extract a hand->weight map for the requested actor at node_key.
    Supports a couple of common dump shapes. Adjust if your solver differs.
    """
    actor_l = actor.lower()

    # node-keyed first (most explicit)
    nodes = js.get("nodes")
    if isinstance(nodes, dict) and node_key in nodes:
        node = nodes[node_key]
        ranges = node.get("ranges") or node.get("range") or {}
        if isinstance(ranges, dict) and actor_l in ranges:
            return _normalize_range_payload(ranges[actor_l])
        actors = node.get("actors")
        if isinstance(actors, dict) and actor_l in actors:
            r = actors[actor_l].get("range") or actors[actor_l].get("ranges")
            if r is not None:
                return _normalize_range_payload(r)

    # root-style
    for root_k in ("root", "tree"):
        root = js.get(root_k)
        if isinstance(root, dict):
            ranges = root.get("ranges") or root.get("range") or {}
            if isinstance(ranges, dict) and actor_l in ranges:
                return _normalize_range_payload(ranges[actor_l])
            actors = root.get("actors")
            if isinstance(actors, dict) and actor_l in actors:
                r = actors[actor_l].get("range") or actors[actor_l].get("ranges")
                if r is not None:
                    return _normalize_range_payload(r)
    return None

def _normalize_range_payload(payload) -> Dict[str, float]:
    """
    Accepts:
      - monker string "AA:1.0,AKs:0.5,..."
      - dict of { 'AA':1.0, ... }
    Returns a dict[str->float].
    """
    if isinstance(payload, str):
        out: Dict[str, float] = {}
        for tok in payload.replace(" ", "").split(","):
            if not tok or ":" not in tok: continue
            h, v = tok.split(":", 1)
            try:
                out[h] = float(v)
            except Exception:
                pass
        return out
    if isinstance(payload, dict):
        # ensure floats
        return {str(k): float(v) for k, v in payload.items()}
    # unknown
    return {}

def _range_to_vec169(rmap: Dict[str, float]) -> np.ndarray:
    """
    Convert a hand->weight map into a 169-length vector (row-major 13x13).
    Requires your `hand_to_index` that matches the solver’s encoding.
    """
    vec = np.zeros(169, dtype=np.float32)
    for hand, w in rmap.items():
        try:
            vec[hand_to_index(hand)] = float(w)
        except Exception:
            # skip unknown hands silently
            pass
    return vec

def _infer_actor(row: pd.Series) -> str:
    """
    Decide whose range we train on for this row.
    Priority:
      - row['villain_pos'] in {'IP','OOP'}
      - row['actor'] in {'ip','oop'}
      - positions like 'IPvOOP' or 'OOPvIP'
      - fallback = 'ip'
    """
    vpos = str(row.get("villain_pos", "")).upper()
    if vpos in ("IP", "OOP"):
        return "ip" if vpos == "IP" else "oop"

    actor = str(row.get("actor", "")).lower()
    if actor in ("ip", "oop"):
        return actor

    pos = str(row.get("positions", "")).upper()
    if pos.startswith("IPV"):  # "IPvOOP"
        return "ip"
    if pos.startswith("OOPV"):
        return "oop"

    # last resort (prefer IP as target)
    return "ip"

# ---------------------- main builder logic --------------------------

def build_rangenet_postflop(
    manifest_path: Path,
    out_parquet: Path,
    cfg: Mapping[str, Any],
) -> None:
    """
    Pipeline:
      - read manifest rows (already solved or to be solved)
      - for each row, load result JSON (local-or-S3) by s3_key
      - extract actor range at node_key
      - average duplicates into scenario buckets
      - write parquet with y_0..y_168 and weights
    """
    df = pd.read_parquet(manifest_path)

    # we’ll support either exact board or clusters
    use_clusters = bool(_get(cfg, "worker.use_board_clusters", True))

    # ensure node_key column
    if "node_key" not in df.columns:
        df = df.assign(node_key="root")

    # group key (what the trainer conditions on)
    if use_clusters:
        need_cols = ["effective_stack_bb", "positions", "street", "board_cluster_id", "node_key"]
        # fallbacks if missing
        if "board_cluster_id" not in df.columns:
            # if clusters weren’t embedded, require board (string) and your clusterer here
            raise RuntimeError("Manifest missing board_cluster_id but worker.use_board_clusters=True")
    else:
        need_cols = ["effective_stack_bb", "positions", "street", "board", "node_key"]

    missing_need = [c for c in need_cols if c not in df.columns]
    if missing_need:
        raise RuntimeError(f"Manifest missing required columns: {missing_need}")

    # storage
    buckets: Dict[Tuple, List[np.ndarray]] = {}
    weights: Dict[Tuple, float] = {}

    skipped: Dict[str, int] = {}

    for _, r in df.iterrows():
        s3_key = str(r.get("s3_key") or "")
        if not s3_key:
            skipped["no_s3_key"] = skipped.get("no_s3_key", 0) + 1
            continue

        # infer actor
        actor = _infer_actor(r)
        node_key = str(r.get("node_key") or "root")

        # load result json (local-or-S3)
        try:
            js = _load_solver_json_local_or_s3(cfg, s3_key)
        except FileNotFoundError:
            skipped["cache_miss"] = skipped.get("cache_miss", 0) + 1
            continue
        except Exception:
            skipped["s3_fetch_fail"] = skipped.get("s3_fetch_fail", 0) + 1
            continue

        # extract that actor’s range at this node
        rmap = _extract_actor_range(js, actor=actor, node_key=node_key)
        if not rmap:
            skipped["parse_fail"] = skipped.get("parse_fail", 0) + 1
            continue

        vec = _range_to_vec169(rmap)

        # build bucket key
        if use_clusters:
            key = (
                int(round(float(r["effective_stack_bb"]))),
                str(r["positions"]),
                int(r["street"]),
                int(r["board_cluster_id"]),
                node_key,
            )
        else:
            key = (
                int(round(float(r["effective_stack_bb"]))),
                str(r["positions"]),
                int(r["street"]),
                str(r["board"]),
                node_key,
            )

        buckets.setdefault(key, []).append(vec)
        weights[key] = weights.get(key, 0.0) + 1.0

    # materialize rows
    rows: List[Dict[str, Any]] = []
    for key, vecs in buckets.items():
        y = np.mean(np.stack(vecs, axis=0), axis=0)
        row = {
            "stack_bb": key[0],
            "positions": key[1],
            "street": key[2],
            "node_key": key[4],
            "weight": float(weights[key]),
        }
        if use_clusters:
            row["board_cluster_id"] = key[3]
        else:
            row["board_str"] = key[3]

        for i, v in enumerate(y.tolist()):
            row[f"y_{i}"] = float(v)

        rows.append(row)

    out = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)

    print(f"✅ wrote {out_parquet} with {len(out):,} rows")
    if skipped:
        tot = sum(skipped.values())
        print("   Skipped:", ", ".join(f"{k}={v}" for k, v in skipped.items()), f"(total {tot})")
    if not len(out):
        print("   (No rows — likely cache misses or schema mismatch)")

# ------------------------------ CLI ---------------------------------

def run_from_config(cfg: Mapping[str, Any]) -> None:
    manifest = Path(_get(cfg, "inputs.manifest", "data/artifacts/rangenet_postflop_flop_manifest.parquet"))
    out_pq   = Path(_get(cfg, "outputs.parquet", "data/datasets/rangenet_postflop.parquet"))
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    build_rangenet_postflop(manifest, out_pq, cfg)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Build RangeNet Postflop parquet from solved charts (local-or-S3)")
    ap.add_argument("--config", type=str, default="rangenet/postflop",
                    help="Model name or YAML path (resolved by load_model_config)")
    ap.add_argument("--manifest", type=str, default=None,
                    help="Override manifest parquet path")
    ap.add_argument("--out", type=str, default=None,
                    help="Override output parquet path")
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    if args.manifest:
        d = cfg.setdefault("inputs", {})
        d["manifest"] = args.manifest
    if args.out:
        d = cfg.setdefault("outputs", {})
        d["parquet"] = args.out

    run_from_config(cfg)