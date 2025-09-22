#!/usr/bin/env python3
import os, io, gzip, json, random, argparse, hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import boto3
from botocore.exceptions import ClientError

load_dotenv()
# ---------------- helpers (mirrors builder logic) ----------------

def hand_to_index_169(hand: str) -> Optional[int]:
    # Expect keys like "AhAd", "AKs", "QJo", etc. If your JSON uses a different key,
    # replace this with your existing helper.
    from collections import defaultdict
    # Precompute static 13x13 map once
    if not hasattr(hand_to_index_169, "_map"):
        ranks = "AKQJT98765432"
        idx = {}
        k = 0
        for i, r1 in enumerate(ranks):
            for j, r2 in enumerate(ranks):
                suited = (i < j)
                off = (i > j)
                if i == j:
                    key = r1 + r2
                elif suited:
                    key = r1 + r2 + "s"
                else:
                    key = r2 + r1 + "o"
                idx[key] = k
                k += 1
        hand_to_index_169._map = idx
    # Normalize a few common formats:
    h = hand.strip()
    h = h.replace("10", "T")
    h = h.replace("t", "T")
    h = h.replace("-", "")
    # Already condensed (AKs/QQ/QJo) → direct
    if len(h) in (2,3):
        return hand_to_index_169._map.get(h)
    # 2-card explicit (e.g., "AhAd","AsKd") → reduce to class
    if len(h) == 4:
        r1, s1, r2, s2 = h[0], h[1], h[2], h[3]
        r1 = r1.upper(); r2 = r2.upper()
        suited = (s1 == s2)
        if r1 == r2:
            key = r1 + r2
        else:
            # rank order in our grid is high→low on rows, and columns mirror;
            # we map to canonical AKs/AQo etc.
            ranks = "AKQJT98765432"
            i1, i2 = ranks.index(r1), ranks.index(r2)
            if i1 < i2:   # r1 stronger
                key = r1 + r2 + ("s" if suited else "o")
            else:
                key = r2 + r1 + ("s" if suited else "o")
        return hand_to_index_169._map.get(key)
    return None

def _group_from_menu(menu_id: str) -> str:
    m = (menu_id or "").strip()
    return m.split(".", 1)[0] if "." in m else m

def _role_from_menu(menu_id: str) -> str:
    m = (menu_id or "").strip()
    return m.split(".", 1)[1] if "." in m else ""

def _parse_child_label(label: str) -> Tuple[str, Optional[int], str]:
    raw = str(label).strip()
    up  = raw.upper()
    toks = up.split()
    if not toks:
        return ("UNKNOWN", None, raw)
    act = toks[0]
    size = None
    if len(toks) >= 2:
        try:
            size = int(round(float(toks[1])))
        except Exception:
            size = None
    return (act, size, raw)

def _extract_child_vec_169(root: dict, child_index: int) -> Optional[np.ndarray]:
    strat_map = (root.get("strategy") or {}).get("strategy") or {}
    if not strat_map:
        return None
    v = np.zeros(169, dtype=np.float32)
    any_set = False
    for hand, probs in strat_map.items():
        idx = hand_to_index_169(str(hand))
        if idx is None:
            continue
        try:
            p = float(probs[child_index])
            if p < 0.0: p = 0.0
            if p > 1.0: p = 1.0
            v[idx] = p
            any_set = True
        except Exception:
            pass
    return v if any_set and np.any(v) else None

def pick_root(js: dict, node_key: str = "root") -> dict:
    # If your solver JSON uses a different structure, adapt this to your existing pick_root.
    # Fallback: assume js is already the root node.
    if isinstance(js, dict) and "childrens" in js and "strategy" in js:
        return js
    # Sometimes wrapped under "tree" or similar; try obvious places
    for k in ("tree","root","node"):
        if isinstance(js.get(k), dict) and "childrens" in js[k]:
            return js[k]
    return js

# ---------------- IO (local cache + S3) ----------------

def _s3_client(region: Optional[str]=None):
    return boto3.client("s3", region_name=region or os.getenv("AWS_REGION") or "eu-central-1")

def _local_cache_path(cache_root: Path, s3_key: str) -> Path:
    return (cache_root / s3_key).resolve()

def _read_json_bytes(b: bytes, key_hint: str) -> dict:
    if key_hint.endswith(".gz"):
        b = gzip.GzipFile(fileobj=io.BytesIO(b)).read()
    return json.loads(b)

def load_solver_json_local_or_s3(s3_bucket: str, s3_key: str, cache_root: Path, region: Optional[str]=None) -> dict:
    p = _local_cache_path(cache_root, s3_key)
    if p.is_file():
        return _read_json_bytes(p.read_bytes(), p.name)
    s3 = _s3_client(region)
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    body = obj["Body"].read()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return _read_json_bytes(body, s3_key)

# ---------------- validator ----------------

def validate_sample(
    manifest_path: Path,
    *,
    sample_size: int = 50,
    dataset_path: Optional[Path] = None,
    cache_root: Path = Path("data/solver_cache"),
    region: Optional[str] = None,
    bucket_env: str = "pokeraistore",
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    df = pd.read_parquet(manifest_path)
    if "node_key" not in df.columns:
        df = df.assign(node_key="root")

    # de-dupe like builder
    df = df.drop_duplicates(subset=["s3_key","node_key"]).reset_index(drop=True)

    if len(df) == 0:
        print("Empty manifest.")
        return

    # optional post-build parquet to cross-check
    built = None
    if dataset_path and Path(dataset_path).exists():
        built = pd.read_parquet(dataset_path)

    bucket = os.getenv(bucket_env)
    if not bucket:
        raise SystemExit(f"{bucket_env} not set")

    rows = df.sample(n=min(sample_size, len(df)), random_state=42).to_dict("records")

    problems = []
    ok_count = 0

    for r in tqdm(rows, desc="Validating extraction"):
        s3_key   = str(r["s3_key"])
        node_key = str(r.get("node_key") or "root")

        js = load_solver_json_local_or_s3(bucket, s3_key, cache_root, region=region)
        root = pick_root(js, node_key=node_key)
        childrens = root.get("childrens") or {}
        child_labels = list(childrens.keys())

        if not child_labels:
            problems.append((s3_key, "no_children"))
            continue

        # Build all child vectors, check normalization if your builder normalizes per-action Y
        child_vecs = []
        for ci, lab in enumerate(child_labels):
            vec = _extract_child_vec_169(root, ci)
            if vec is None or not np.any(vec):
                # builder skips empty actions; so do we
                continue
            s = float(vec.sum())
            if s > 0:
                vec = vec / s
            child_vecs.append((ci, lab, vec))

        if not child_vecs:
            problems.append((s3_key, "all_children_empty"))
            continue

        # Cross-check against built parquet (optional)
        if built is not None:
            # match rows for this file/node at root by action label
            # your builder stores human action and bet_size_pct; use parsed tuple
            # we tolerate multiple rows per child if you emit weights; we compare any match
            subset = built[built["node_key"].fillna("root") == node_key].copy()
            # you may also have s3_key in the dataset; if so, uncomment to narrow:
            # subset = subset[subset["s3_key"] == s3_key]

            # parse child labels of built rows
            def key_row(row):
                a = str(row.get("action","")).upper()
                bs = row.get("bet_size_pct")
                return (a, int(bs) if pd.notna(bs) else None)

            built_map: Dict[Tuple[str, Optional[int]], List[np.ndarray]] = {}
            ycols = [c for c in subset.columns if c.startswith("y_")]
            for _, br in subset.iterrows():
                k = key_row(br)
                v = br[ycols].to_numpy(dtype=np.float32, copy=False)
                s = float(v.sum())
                if s > 0:
                    v = v / s
                built_map.setdefault(k, []).append(v)

            for ci, lab, vec in child_vecs:
                act, size_pct, _ = _parse_child_label(lab)
                k = (act, size_pct)
                cand = built_map.get(k)
                if not cand:
                    # allow the case where this S3 file wasn’t included due to sharding/dupes
                    continue
                # compare against any candidate row; flag if none match within tolerances
                matched = any(np.allclose(vec, v2, rtol=rtol, atol=atol) for v2 in cand)
                if not matched:
                    problems.append((s3_key, f"mismatch for child '{lab}'"))
                    break
            else:
                ok_count += 1
        else:
            ok_count += 1

    total = len(rows)
    print(f"\nChecked {total} samples. OK: {ok_count}. Problems: {len(problems)}")
    if problems:
        # group brief summary
        from collections import Counter
        kinds = Counter(reason for _, reason in problems)
        print("Problem summary:", dict(kinds))
        # print first few examples
        for s3_key, reason in problems[:10]:
            print(" -", reason, "→", s3_key)

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser("Validate that postflop targets are extracted correctly from solver JSON.")
    ap.add_argument("--manifest", required=True, help="Path to flop manifest parquet")
    ap.add_argument("--dataset", default=None, help="Optional built parquet to cross-check (post-build)")
    ap.add_argument("--cache-root", default="data/solver_cache", help="Local cache dir for downloaded .json.gz")
    ap.add_argument("--samples", type=int, default=50, help="Number of manifest rows to validate")
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "eu-central-1"))
    ap.add_argument("--bucket-env", default="AWS_BUCKET_NAME", help="Env var holding the bucket name")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args()

    validate_sample(
        Path(args.manifest),
        sample_size=args.samples,
        dataset_path=(Path(args.dataset) if args.dataset else None),
        cache_root=Path(args.cache_root),
        region=args.region,
        bucket_env=args.bucket_env,
        rtol=args.rtol,
        atol=args.atol,
    )

if __name__ == "__main__":
    main()