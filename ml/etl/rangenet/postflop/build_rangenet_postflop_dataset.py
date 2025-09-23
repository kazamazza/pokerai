from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import boto3
import dotenv
import requests

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client
import io, gzip, json, random, time, hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping, Tuple
from tqdm import tqdm
from botocore.exceptions import BotoCoreError, ClientError

dotenv.load_dotenv()

# -------------------- vocab --------------------
ACTION_VOCAB = [
    "FOLD","CHECK","CALL",
    "BET_25","BET_33","BET_50","BET_66","BET_75","BET_100",
    "DONK_33",
    "RAISE_150","RAISE_200","RAISE_300","ALLIN",
]
VOCAB_INDEX = {a:i for i,a in enumerate(ACTION_VOCAB)}
VOCAB_SIZE = len(ACTION_VOCAB)

def _get_instance_id(timeout=1.5) -> str | None:
    try:
        r = requests.get("http://169.254.169.254/latest/meta-data/instance-id", timeout=timeout)
        return r.text.strip() if r.ok else None
    except Exception:
        return None

def _detect_region(timeout=1.5) -> str | None:
    try:
        r = requests.get("http://169.254.169.254/latest/dynamic/instance-identity/document", timeout=timeout)
        if r.ok:
            return r.json().get("region")
    except Exception:
        pass
    return None

def shutdown_ec2_instance(mode: str = "stop", wait_seconds: int = 5) -> None:
    iid = _get_instance_id()
    if not iid:
        print("ℹ️ Not on EC2 (or IMDS unreachable); skipping shutdown.")
        return
    reg = _detect_region() or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not reg:
        print("⚠️ Could not determine AWS region; skipping shutdown.")
        return

    print(f"🕘 Shutting down in {wait_seconds}s (mode={mode}) for {iid} in {reg}")
    time.sleep(wait_seconds)

    ec2 = boto3.client("ec2", region_name=reg)
    try:
        if mode == "terminate":
            ec2.terminate_instances(InstanceIds=[iid])
            print(f"🗑️ Terminate API sent for {iid}")
        else:
            ec2.stop_instances(InstanceIds=[iid])
            print(f"🛑 Stop API sent for {iid}")
    except Exception as e:
        print(f"❌ Failed to {mode} instance: {e}")

def _extract_pct(label: str) -> Optional[float]:
    """Extract a percentage from 'BET 33', 'BET 66%', etc."""
    m = re.search(r"(\d+)", label)
    return float(m.group(1)) if m else None

def _extract_raise_x(label: str) -> Optional[float]:
    """Extract raise multiple from 'RAISE 2.5x', 'RAISE 3', etc."""
    m = re.search(r"([\d\.]+)", label)
    return float(m.group(1)) if m else None

def bucket_bet_pct(p: Optional[float]) -> str:
    if p is None: return "BET_33"  # safe fallback
    if p < 30:  return "BET_25"
    if p < 42:  return "BET_33"
    if p < 58:  return "BET_50"
    if p < 71:  return "BET_66"
    if p < 90:  return "BET_75"
    return "BET_100"

def bucket_raise_x(x: Optional[float]) -> str:
    if x is None: return "RAISE_200"
    if x < 1.75: return "RAISE_150"
    if x < 2.5:  return "RAISE_200"
    return "RAISE_300"

# -------------------- unified mapper --------------------
def map_child_to_bucket(label: str, actor: str) -> str:
    up = str(label).upper().strip()

    if up.startswith("CHECK"): return "CHECK"
    if up.startswith("CALL"):  return "CALL"
    if up.startswith("FOLD"):  return "FOLD"
    if up.startswith("ALL"):   return "ALLIN"
    if up.startswith("DONK"):  return "DONK_33"

    if up.startswith("BET"):
        pct = _extract_pct(up)
        return bucket_bet_pct(pct)

    if up.startswith("RAISE"):
        x = _extract_raise_x(up)
        return bucket_raise_x(x)

    return "CHECK"  # ultimate fallback

# -------------------- helpers --------------------
def _get(cfg: Mapping[str, any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _stable_shard_index(s3_key: str, node_key: str, m: int) -> int:
    # stable, deterministic shard from (s3_key, node_key)
    h = hashlib.sha1(f"{s3_key}|{node_key}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % m

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

def _load_solver_json_local_or_s3(cfg: Mapping[str, Any], s3c: S3Client, s3_key: str) -> dict:
    local_path = _cache_path_for_key(cfg, s3_key)
    if not local_path.is_file():
        _retry(lambda: s3c.download_file_if_missing(s3_key, local_path))
    return _read_json_file_allow_gz(local_path)


def _split_positions(positions: str) -> Tuple[str, str]:
    s = str(positions).upper()
    if "V" in s:
        a, b = s.split("V", 1)
        return a, b
    return ("IP","OOP")


def build_postflop_policy(
    manifest_path: Path,
    cfg: Mapping[str, Any],
    *,
    part_rows: int = 2000,
    parts_local_dir: str = "data/datasets/postflop_policy_parts",
    parts_s3_prefix: Optional[str] = None,
) -> None:
    """
    Build postflop *policy* parquet parts by aggregating root child probabilities
    over ACTION_VOCAB. We compute child mass from the node's strategy map, not
    from child["weight"] (which is usually absent/non-probabilistic).
    """
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(manifest_path)
    s3c = S3Client()

    use_clusters = bool(_get(cfg, "worker.use_board_clusters", True))

    local_parts_dir = Path(parts_local_dir)
    local_parts_dir.mkdir(parents=True, exist_ok=True)

    rows_buffer: List[Dict[str, Any]] = []
    parts_written = 0
    skipped_no_children = 0
    skipped_zero_mass = 0

    def flush_part():
        nonlocal rows_buffer, parts_written
        if not rows_buffer:
            return
        part_df = pd.DataFrame(rows_buffer)

        # Ensure every ACTION_VOCAB column exists
        for a in ACTION_VOCAB:
            if a not in part_df.columns:
                part_df[a] = 0.0

        part_name = f"part-{parts_written:05d}.parquet"
        part_path = local_parts_dir / part_name
        part_df.to_parquet(part_path, index=False)
        print(f"💾 wrote {part_path} (rows={len(part_df)})")
        if parts_s3_prefix:
            s3_key = f"{parts_s3_prefix}/{part_name}"
            _retry(lambda: s3c.upload_file(part_path, s3_key))
        parts_written += 1
        rows_buffer = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Building postflop policy"):
        # --- feature fields needed to infer 'actor' & write row ---
        ip_pos, oop_pos = _split_positions(str(r["positions"]))
        menu_id = str(r.get("bet_sizing_id", "") or "")
        role = _role_from_menu(menu_id).upper()  # e.g., "SRP_HU.PFR_IP" -> "PFR_IP"
        actor = ("ip" if role.endswith("_IP") else
                 "oop" if role.endswith("_OOP") else "ip")  # default to IP if unknown
        hero_pos = ip_pos if actor == "ip" else oop_pos

        stack_bb = int(round(float(r["effective_stack_bb"])))
        pot_bb   = float(r["pot_bb"])
        street   = int(r["street"])
        ctx      = str(r["ctx"])

        board_cluster: Optional[int] = None
        board: Optional[str] = None
        if use_clusters and "board_cluster_id" in r:
            board_cluster = int(r["board_cluster_id"])
        elif not use_clusters and "board" in r:
            board = str(r["board"])

        # --- fetch JSON once per manifest row & pick root node ---
        js = _load_solver_json_local_or_s3(cfg, s3c, str(r["s3_key"]))
        root = js.get("root") or js

        childrens = root.get("childrens") or {}
        if not childrens:
            skipped_no_children += 1
            continue
        child_labels = list(childrens.keys())

        # --- pull strategy map and accumulate per-child mass over hands ---
        strat_map = (root.get("strategy") or {}).get("strategy") or {}
        if not strat_map:
            skipped_zero_mass += 1
            continue

        # child_mass[i] = total (or average) probability for child i across hands
        child_mass = np.zeros(len(child_labels), dtype=np.float64)
        num_hands = 0

        for probs in strat_map.values():
            # probs should be an iterable of per-child probabilities
            try:
                L = len(probs)
            except Exception:
                L = 0
            if L == 0:
                continue
            take = min(L, len(child_labels))
            # accumulate only non-negative probabilities
            for i in range(take):
                try:
                    p = float(probs[i])
                except Exception:
                    p = 0.0
                if p > 0.0:
                    child_mass[i] += p
            num_hands += 1

        if num_hands == 0:
            skipped_zero_mass += 1
            continue

        # Average across hands (you can also keep sum; we'll normalize later anyway)
        child_mass /= float(num_hands)

        # --- map child labels -> ACTION_VOCAB buckets & aggregate ---
        action_probs = np.zeros(VOCAB_SIZE, dtype=np.float32)
        for i, lab in enumerate(child_labels):
            bucket = map_child_to_bucket(lab, actor)  # uses actor for DONK vs BET, etc.
            idx = VOCAB_INDEX[bucket]
            action_probs[idx] += float(child_mass[i])

        total = float(action_probs.sum())
        if total <= 0.0:
            skipped_zero_mass += 1
            continue
        action_probs /= total  # normalize to sum≈1

        # easy-to-read argmax (sanity/debug)
        argmax_idx = int(action_probs.argmax())
        argmax_action = ACTION_VOCAB[argmax_idx]

        row: Dict[str, Any] = {
            "stack_bb": stack_bb,
            "pot_bb": pot_bb,
            "hero_pos": hero_pos,
            "ip_pos": ip_pos,
            "oop_pos": oop_pos,
            "street": street,
            "ctx": ctx,
            "actor": actor,
            "action": argmax_action,     # stored for quick checks; training uses prob cols
            "bet_size_pct": np.nan,      # N/A at root aggregation level
            "weight": 1.0,
            "bet_sizing_id": menu_id,
        }
        if board_cluster is not None:
            row["board_cluster"] = int(board_cluster)
        if board is not None:
            row["board"] = board

        # add probability columns
        for a, prob in zip(ACTION_VOCAB, action_probs.tolist()):
            row[a] = float(prob)

        rows_buffer.append(row)
        if len(rows_buffer) >= part_rows:
            flush_part()

    flush_part()
    print(
        f"✅ done. parts={parts_written} "
        f"(skipped: no_children={skipped_no_children}, zero_mass={skipped_zero_mass})"
    )

def _role_from_menu(menu_id: str) -> str:
    """
    e.g. 'srp_hu.PFR_IP' -> 'PFR_IP'
         '3bp_hu.CALLER_OOP' -> 'CALLER_OOP'
    Falls back to upper(menu_id) if no dot.
    """
    m = (menu_id or "").strip()
    if "." in m:
        return m.split(".", 1)[1].upper()
    return m.upper()

def run_from_config(
    cfg: Mapping[str, any],
    *,
    shard_index: Optional[int],
    shard_count: Optional[int],
    part_rows: int,
    parts_local_dir: str,
    parts_s3_prefix: Optional[str],
    sample_n: Optional[int] = None,
    sample_random: bool = False,
    sample_seed: int = 42,
) -> None:
    """
    Sharded builder with optional small-sample mode.

    - If shard_count & shard_index are provided, only the assigned shard rows are processed.
    - If sample_n is provided, only that many manifest rows are built (after sharding, if any).
    """
    import pandas as pd
    import numpy as np

    manifest = Path(_get(cfg, "inputs.manifest", "data/artifacts/rangenet_postflop_flop_manifest.dedup.parquet"))
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    df = pd.read_parquet(manifest)

    # Ensure node_key exists for stable sharding
    if "node_key" not in df.columns:
        df = df.assign(node_key="root")

    # Optional sharding
    if shard_count is not None and shard_index is not None:
        if not (0 <= shard_index < shard_count):
            raise ValueError("--shard-index must be in [0, shard_count)")
        mask = df.apply(
            lambda r: _stable_shard_index(str(r["s3_key"]), str(r["node_key"]), shard_count) == shard_index,
            axis=1,
        )
        df = df.loc[mask].reset_index(drop=True)
        print(f"🧩 shard {shard_index}/{shard_count} → {len(df)} manifest rows")

    # Optional sampling (after sharding to keep per-shard test fast)
    if sample_n is not None and sample_n > 0 and len(df) > 0:
        if sample_random:
            rng = np.random.default_rng(sample_seed)
            take = min(sample_n, len(df))
            idx = rng.choice(len(df), size=take, replace=False)
            df = df.iloc[idx].reset_index(drop=True)
        else:
            df = df.head(sample_n).reset_index(drop=True)
        print(f"🔎 sample mode: using {len(df)} row(s)")

    # Hand the filtered frame to your builder (no re-read inside)
    # Your build_postflop_policy should accept either a path or a DataFrame;
    # if it only accepts a path, we’ll write a temp parquet.
    tmp_manifest = manifest.with_suffix(f".shard.tmp.parquet")
    df.to_parquet(tmp_manifest, index=False)

    build_postflop_policy(
        manifest_path=tmp_manifest,
        cfg=cfg,
        part_rows=part_rows,
        parts_local_dir=parts_local_dir,
        parts_s3_prefix=parts_s3_prefix,
    )

    print("✅ run_from_config complete.")


if __name__ == "__main__":
    import argparse
    from ml.utils.config import load_model_config  # adjust import to your project

    ap = argparse.ArgumentParser("Build Postflop Policy (sharded, streaming parquet parts)")
    ap.add_argument("--config", type=str, default="rangenet/postflop",
                    help="Model name or YAML path resolved by load_model_config")
    ap.add_argument("--shard-index", type=int, default=None,
                    help="This worker’s shard index (0-based)")
    ap.add_argument("--shard-count", type=int, default=None,
                    help="Total number of shards")
    ap.add_argument("--part-rows", type=int, default=2000,
                    help="Flush a parquet part every N rows")
    ap.add_argument("--parts-local-dir", type=str, default="data/datasets/postflop_policy_parts",
                    help="Where to write local parts")
    ap.add_argument("--parts-s3-prefix", type=str, default=None,
                    help='If set, upload parts to this S3 prefix (e.g., "datasets/rangenet_postflop/policy_parts")')

    # sampling / smoke test
    ap.add_argument("--sample-n", type=int, default=None,
                    help="If set, build only this many rows (after sharding).")
    ap.add_argument("--sample-random", action="store_true",
                    help="If set with --sample-n, sample randomly (else use head).")
    ap.add_argument("--sample-seed", type=int, default=42,
                    help="Seed for random sampling.")

    args = ap.parse_args()
    cfg = load_model_config(args.config)

    try:
        run_from_config(
            cfg,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            part_rows=args.part_rows,
            parts_local_dir=args.parts_local_dir,
            parts_s3_prefix=args.parts_s3_prefix,
            sample_n=args.sample_n,
            sample_random=args.sample_random,
            sample_seed=args.sample_seed,
        )
    finally:
        shutdown_ec2_instance(mode="stop", wait_seconds=10)