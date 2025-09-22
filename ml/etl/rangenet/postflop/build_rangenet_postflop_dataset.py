from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from ml.utils.config import load_model_config
from infra.storage.s3_client import S3Client
from ml.range.solvers.utils.solver_json_extract import hand_to_index_169
import io
import gzip
import json
import time
import random
from pathlib import Path
from typing import Mapping, Any, Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from botocore.exceptions import ClientError, BotoCoreError

# --------- tiny cfg + retry helpers ----------
def _get(cfg: Mapping[str, Any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _retry(fn, *, tries: int = 5, jitter: float = 0.25, base_sleep: float = 0.6):
    last = None
    for i in range(tries):
        try:
            return fn()
        except (ClientError, BotoCoreError) as e:
            last = e
            sleep = base_sleep * (2 ** i) + random.random() * jitter
            time.sleep(sleep)
    if last:
        raise last

# --------- local cache + loader ----------
def _cache_root(cfg: Mapping[str, Any]) -> Path:
    return Path(_get(cfg, "worker.local_cache_dir", "data/solver_cache"))

def _cache_path_for_key(cfg: Mapping[str, Any], s3_key: str) -> Path:
    return (_cache_root(cfg) / s3_key).resolve()

def _read_json_file_allow_gz(p: Path) -> dict:
    b = p.read_bytes()
    if p.suffix == ".gz":
        with gzip.GzipFile(fileobj=io.BytesIO(b)) as gz:
            text = gz.read().decode("utf-8")
        return json.loads(text)
    else:
        return json.loads(b.decode("utf-8"))

def _load_solver_json_local_or_s3(cfg: Mapping[str, Any], s3c: S3Client, s3_key: str) -> tuple[dict, bool]:
    """
    Returns (json_obj, cache_hit).
    1) Try local cache; 2) download from S3 with retries; 3) read again from cache.
    """
    local_path = _cache_path_for_key(cfg, s3_key)
    if local_path.is_file():
        return _read_json_file_allow_gz(local_path), True

    def _dl():
        s3c.download_file_if_missing(s3_key, local_path)
        return True

    _retry(_dl, tries=5, jitter=0.3, base_sleep=0.8)
    return _read_json_file_allow_gz(local_path), False

# --------- minimal menu/actor/labels helpers ----------
def _role_from_menu(menu_id: str) -> str:
    m = (menu_id or "").strip()
    return m.split(".", 1)[1] if "." in m else ""

def _split_positions(positions: str) -> Tuple[str, str]:
    s = str(positions).upper()
    if "V" in s:
        a, b = s.split("V", 1)
        return a, b
    return ("IP","OOP")

def _parse_child_label(label: str) -> tuple[str, Optional[int], str]:
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
        idx = hand_to_index_169(str(hand))  # assumes available in your codebase
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

def _dedupe_manifest(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    before = len(df)
    if "node_key" not in df.columns:
        df = df.assign(node_key="root")
    df2 = df.drop_duplicates(subset=["s3_key", "node_key"]).reset_index(drop=True)
    after = len(df2)
    return df2, before, after

# --------- pick root (delegates to your schema) ----------
def pick_root(js: dict, node_key: str = "root") -> dict:
    # If your JSON schema differs, point this to your existing util.
    # Common pattern:
    # - Monker-like export: js["nodes"][node_key]
    # - Flat: js if contains "childrens", else js["root"]
    if isinstance(js, dict) and "childrens" in js and "strategy" in js:
        return js
    if "root" in js:
        return js["root"]
    # fallback to first dict that looks like a node
    for v in js.values():
        if isinstance(v, dict) and "childrens" in v:
            return v
    return js

# --------- core builder (sharded, streaming-to-parts) ----------
def build_rangenet_postflop(
    manifest_path: Path,
    out_parquet: Optional[Path],            # kept for compatibility; not used in sharded mode
    cfg: Mapping[str, Any],
    *,
    shard_index: Optional[int] = None,
    shard_count: Optional[int] = None,
    part_rows: int = 2000,                  # flush a parquet part every N rows
    parts_local_dir: str = "data/datasets/postflop_parts",
    parts_s3_prefix: Optional[str] = None,  # e.g. "datasets/rangenet_postflop/parts"
) -> None:
    import hashlib

    def _stable_shard(s3_key: str, node_key: str, m: int) -> int:
        h = hashlib.sha1(f"{s3_key}|{node_key}".encode("utf-8")).hexdigest()
        return int(h[:8], 16) % m

    df = pd.read_parquet(manifest_path)

    use_clusters = bool(_get(cfg, "worker.use_board_clusters", True))
    normalize_y = bool(_get(cfg, "dataset.normalize_labels", True))

    need = ["s3_key", "positions", "street", "effective_stack_bb", "pot_bb", "bet_sizing_id", "ctx"]
    need += ["board_cluster_id"] if use_clusters else ["board"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {missing}")

    df, m_before, m_after = _dedupe_manifest(df)
    if m_before != m_after:
        print(f"⚠️ manifest deduped: {m_before} → {m_after}")

    # Optional sharding
    if shard_count is not None and shard_index is not None:
        if not (0 <= shard_index < shard_count):
            raise ValueError("--shard-index must be in [0, shard_count)")
        if "node_key" not in df.columns:
            df = df.assign(node_key="root")
        mask = df.apply(lambda r: _stable_shard(str(r["s3_key"]), str(r["node_key"]), shard_count) == shard_index, axis=1)
        df = df.loc[mask].reset_index(drop=True)
        print(f"🧩 shard {shard_index}/{shard_count}: {len(df)} rows")

    s3c = S3Client()
    local_parts_dir = Path(parts_local_dir)
    local_parts_dir.mkdir(parents=True, exist_ok=True)

    rows_buffer: List[Dict[str, Any]] = []
    parts_written = 0
    skipped_empty = 0
    skipped_no_children = 0
    cache_hits = 0
    fetched = 0

    # deterministic part prefix per shard
    shard_tag = (f"s{shard_index:02d}" if shard_index is not None else "s00")

    def flush_part():
        nonlocal rows_buffer, parts_written
        if not rows_buffer:
            return
        part_df = pd.DataFrame(rows_buffer)
        y_cols = [c for c in part_df.columns if c.startswith("y_")]
        if y_cols:
            part_df[y_cols] = part_df[y_cols].fillna(0.0)

        part_name = f"part-{shard_tag}-{parts_written:05d}.parquet"
        part_path = local_parts_dir / part_name
        part_df.to_parquet(part_path, index=False)
        print(f"💾 wrote {part_path}  (rows={len(part_df)})")
        if parts_s3_prefix:
            s3_key_ck = f"{parts_s3_prefix}/{part_name}"
            _retry(lambda: s3c.upload_file(part_path, s3_key_ck), tries=4, base_sleep=0.6, jitter=0.2)
        parts_written += 1
        rows_buffer = []  # clear

    total = len(df)
    for idx, (_, r) in enumerate(tqdm(df.iterrows(), total=total, desc="Building postflop (sharded→parts)")):
        s3_key  = str(r["s3_key"])
        node_key = str(r.get("node_key") or "root")

        js, cache_hit = _load_solver_json_local_or_s3(cfg, s3c, s3_key)
        cache_hits += int(cache_hit)
        fetched    += int(not cache_hit)

        root = pick_root(js, node_key=node_key)
        childrens = root.get("childrens") or {}
        child_labels = list(childrens.keys())
        if not child_labels:
            skipped_no_children += 1
            continue

        ip_pos, oop_pos = _split_positions(str(r["positions"]))
        stack_bb  = int(round(float(r["effective_stack_bb"])))
        pot_bb    = float(r["pot_bb"])
        street    = int(r["street"])
        ctx       = str(r["ctx"])
        board_key = (int(r["board_cluster_id"]) if use_clusters else str(r["board"]))
        menu_id   = str(r["bet_sizing_id"])

        role = _role_from_menu(menu_id).upper()
        actor = ("ip" if role.endswith("_IP") else
                 "oop" if role.endswith("_OOP") else "ip")
        hero_pos = ip_pos if actor == "ip" else oop_pos

        for ci, lab in enumerate(child_labels):
            action, size_pct, _ = _parse_child_label(lab)
            vec = _extract_child_vec_169(root, ci)
            if (vec is None) or (not np.any(vec)):
                skipped_empty += 1
                continue

            if normalize_y:
                s = float(vec.sum())
                if s > 0:
                    vec = vec / s

            row = {
                "stack_bb": stack_bb,
                "pot_bb": pot_bb,
                "hero_pos": hero_pos,
                "ip_pos": ip_pos,
                "oop_pos": oop_pos,
                "street": street,
                "ctx": ctx,
                "bet_sizing_id": menu_id,
                "actor": actor,
                "action": action,
                "bet_size_pct": size_pct,
                "node_key": node_key,
                "weight": 1.0,
            }
            if use_clusters:
                row["board_cluster"] = int(board_key)
                row["board_cluster_id"] = int(board_key)
            else:
                row["board"] = str(board_key)

            # y_0..y_168
            yv = vec.tolist()
            for i, v in enumerate(yv):
                row[f"y_{i}"] = float(v)

            rows_buffer.append(row)

            # rolling flush to keep memory flat
            if len(rows_buffer) >= part_rows:
                flush_part()

    # final flush
    flush_part()

    print(
        f"✅ shard done: parts={parts_written}, "
        f"skipped no-children={skipped_no_children}, skipped empty-actions={skipped_empty}, "
        f"cache_hits={cache_hits}, fetched={fetched}"
    )

# --------- runner ----------
def run_from_config(
    cfg: Mapping[str, Any],
    *,
    shard_index: Optional[int],
    shard_count: Optional[int],
    part_rows: int,
    parts_local_dir: str,
    parts_s3_prefix: Optional[str],
) -> None:
    manifest = Path(_get(cfg, "inputs.manifest", "data/artifacts/rangenet_postflop_flop_manifest.dedup.parquet"))
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    # We purposefully do NOT write a single monolithic parquet here.
    # Each shard writes rolling parts to `parts_local_dir` (and optionally S3).
    build_rangenet_postflop(
        manifest_path=manifest,
        out_parquet=None,
        cfg=cfg,
        shard_index=shard_index,
        shard_count=shard_count,
        part_rows=part_rows,
        parts_local_dir=parts_local_dir,
        parts_s3_prefix=parts_s3_prefix,
    )

# --------- CLI ----------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("Build RangeNet Postflop (sharded, streaming parquet parts)")
    ap.add_argument("--config", type=str, default="rangenet/postflop", help="Model name or YAML path")
    ap.add_argument("--shard-index", type=int, default=None, help="This worker’s shard index (0-based)")
    ap.add_argument("--shard-count", type=int, default=None, help="Total number of shards")
    ap.add_argument("--part-rows", type=int, default=2000, help="Flush a parquet part every N rows")
    ap.add_argument("--parts-local-dir", type=str, default="data/datasets/postflop_parts", help="Where to write local parts")
    ap.add_argument("--parts-s3-prefix", type=str, default=None, help='If set, upload parts to this S3 prefix (e.g., "datasets/rangenet_postflop/parts")')
    args = ap.parse_args()

    cfg = load_model_config(args.config)

    run_from_config(
        cfg,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        part_rows=args.part_rows,
        parts_local_dir=args.parts_local_dir,
        parts_s3_prefix=args.parts_s3_prefix,
    )