from __future__ import annotations
import sys
from pathlib import Path

from ml.policy.policy_rows import make_facing_policy_payload, make_root_policy_payload
from ml.policy.solver_action_mapping import oop_root_kind_for_bet_sizing_id, map_root_mix_to_root_vocab, \
    map_facing_mix_to_facing_vocab

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Tuple

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm

from ml.etl.rangenet.postflop.texas_solver_extractor import TexasSolverExtractor
from ml.models.vocab_actions import ROOT_ACTION_VOCAB, FACING_ACTION_VOCAB
from ml.policy.extractor_invariants import validate_extractor_output
# ----------------------------
# YAML loader (no legacy)
# ----------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}

# ----------------------------
# S3 (direct boto3)
# ----------------------------
_S3CFG = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    connect_timeout=10,
    read_timeout=120,
    tcp_keepalive=True,
)

def s3_client(region: str):
    return boto3.client("s3", region_name=region, config=_S3CFG)

def s3_download_to(s3, bucket: str, key: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(dst))

def s3_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code"))
        http = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        if code in ("404", "NotFound") or http == 404:
            return False
        raise

# ----------------------------
# Manifest bet_sizes parsing
# (matches your schema: array<struct<element: double>>)
# ----------------------------
from decimal import Decimal, ROUND_HALF_UP

def parse_bet_sizes_cell(cell: Any) -> List[int]:
    """
    Manifest 'bet_sizes' can arrive as:
      - None
      - list[float] like [0.33, 0.66] or [33, 66]
      - list[dict] like [{"element":0.33}, ...]
      - pyarrow scalars/arrays, numpy arrays
    Returns ordered unique integer percents: [33, 67]
    """
    if cell is None:
        return []

    # unwrap pyarrow/numpy if present
    try:
        import pyarrow as pa
        if isinstance(cell, (pa.Array, pa.ChunkedArray, pa.Scalar)):
            cell = cell.as_py()
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(cell, np.ndarray):
            cell = cell.tolist()
    except Exception:
        pass

    seq = cell if isinstance(cell, list) else [cell]
    out: List[int] = []
    seen = set()

    for it in seq:
        if it is None:
            continue
        v = it.get("element") if isinstance(it, dict) else it
        try:
            f = float(v)
        except Exception:
            continue

        # fractions (<=3.0) interpreted as pot-fractions
        if f <= 3.0:
            pct = int(Decimal(str(f * 100.0)).quantize(0, rounding=ROUND_HALF_UP))
        else:
            pct = int(Decimal(str(f)).quantize(0, rounding=ROUND_HALF_UP))

        if 1 <= pct <= 200 and pct not in seen:
            out.append(pct)
            seen.add(pct)

    return out

def size_key_for(base_s3_key: str, size_pct: int) -> str:
    base = str(base_s3_key).rstrip("/")
    return f"{base}/size={int(size_pct)}p/output_result.json.gz"

# ----------------------------
# Optional: 52-card board mask (kept simple)
# ----------------------------
_RANKS = "23456789TJQKA"
_SUITS = "cdhs"

def make_board_mask_52(board: str) -> List[float]:
    mask = [0.0] * 52
    s = str(board or "").strip()
    if len(s) % 2 != 0:
        return mask
    for i in range(0, len(s), 2):
        r, u = s[i].upper(), s[i + 1].lower()
        try:
            ri = _RANKS.index(r)
            si = _SUITS.index(u)
            mask[ri * 4 + si] = 1.0
        except ValueError:
            continue
    return mask

# ----------------------------
# Stable shard index
# ----------------------------
import hashlib
def stable_shard_index(sha1: str, shard_count: int) -> int:
    h = hashlib.sha1(str(sha1).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % int(shard_count)

# ----------------------------
# Writing parquet parts
# ----------------------------
def write_part(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")

# ============================================================
# Main builder
# ============================================================
def build_postflop_policy(
    *,
    manifest_path: str,
    solver_yaml_path: str,
    stake_key: str,
    s3_bucket: str,
    s3_region: str,
    out_root_dir: str,
    out_facing_dir: str,
    part_rows: int = 50_000,
    shard_index: Optional[int] = None,
    shard_count: Optional[int] = None,
    strict: Literal["fail", "skip", "emit_sentinel"] = "emit_sentinel",
    local_cache_dir: str = "data/solver_cache",
    debug_jsonl: Optional[str] = None,
) -> None:
    """
    Reads manifest parquet, downloads solver outputs, extracts + maps to action vocabs,
    writes two datasets:
      - root (CHECK/BET_*)
      - facing (FOLD/CALL/RAISE_TO_*/ALLIN)

    strict:
      - fail: raise on any missing/invalid output
      - skip: drop bad rows
      - emit_sentinel: emit valid=0 sentinel rows (recommended for stability)
    """
    solver_yaml = load_yaml(solver_yaml_path)
    stake_cfg = solver_yaml.get(stake_key) or {}
    raise_mults = stake_cfg.get("raise_mult") or []
    if not isinstance(raise_mults, list) or not raise_mults:
        raise ValueError(f"{stake_key}.raise_mult missing/invalid in solver.yaml")

    raise_mults_f = [float(x) for x in raise_mults]

    df = pd.read_parquet(manifest_path)
    if len(df) == 0:
        raise ValueError("manifest is empty")

    # Shard by sha1 (stable)
    if shard_index is not None or shard_count is not None:
        if shard_index is None or shard_count is None:
            raise ValueError("both shard_index and shard_count must be set together")
        si, sc = int(shard_index), int(shard_count)
        df = df[df["sha1"].apply(lambda x: stable_shard_index(str(x), sc) == si)].reset_index(drop=True)
        print(f"🧩 shard {si}/{sc} → manifest rows={len(df):,}")

    s3 = s3_client(s3_region)

    out_root_dir = str(out_root_dir)
    out_facing_dir = str(out_facing_dir)
    Path(out_root_dir).mkdir(parents=True, exist_ok=True)
    Path(out_facing_dir).mkdir(parents=True, exist_ok=True)

    cache_dir = Path(local_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    extractor = TexasSolverExtractor()

    rows_root: List[Dict[str, Any]] = []
    rows_facing: List[Dict[str, Any]] = []
    part_root_i = 0
    part_facing_i = 0

    dbg_fh = open(debug_jsonl, "a", encoding="utf-8") if debug_jsonl else None

    def emit_sentinel(base: Dict[str, Any], *, target: Literal["root", "facing"], reason: str) -> None:
        if dbg_fh:
            dbg_fh.write(json.dumps({"target": target, "reason": reason, **base}) + "\n")
        if strict == "fail":
            raise RuntimeError(f"{target} sentinel: {reason} :: {base.get('solver_key')}")
        if strict == "skip":
            return
        # emit_sentinel
        if target == "root":
            payload = {**base, "valid": 0, "weight": 0.0, "action": "CHECK"}
            for a in ROOT_ACTION_VOCAB:
                payload[a] = 0.0
            rows_root.append(payload)
        else:
            payload = {**base, "valid": 0, "weight": 0.0, "action": "CALL"}
            for a in FACING_ACTION_VOCAB:
                payload[a] = 0.0
            rows_facing.append(payload)

    def flush(kind: Literal["root", "facing"]) -> None:
        nonlocal rows_root, rows_facing, part_root_i, part_facing_i
        if kind == "root":
            if not rows_root:
                return
            out = Path(out_root_dir) / f"root-part-{part_root_i:05d}.parquet"
            write_part(pd.DataFrame(rows_root), out)
            print(f"💾 wrote {out} rows={len(rows_root):,}")
            part_root_i += 1
            rows_root = []
        else:
            if not rows_facing:
                return
            out = Path(out_facing_dir) / f"facing-part-{part_facing_i:05d}.parquet"
            write_part(pd.DataFrame(rows_facing), out)
            print(f"💾 wrote {out} rows={len(rows_facing):,}")
            part_facing_i += 1
            rows_facing = []
        gc.collect()

    # ----------------------------------------
    # Main loop
    # ----------------------------------------
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="postflop_policy"):
        base_s3_key = str(getattr(r, "s3_key"))
        sha1 = str(getattr(r, "sha1"))
        street = int(getattr(r, "street"))
        board = str(getattr(r, "board") or "")
        ip_pos = str(getattr(r, "ip_pos") or "")
        oop_pos = str(getattr(r, "oop_pos") or "")
        ctx = str(getattr(r, "ctx") or "")
        bet_sizing_id = str(getattr(r, "bet_sizing_id") or "")

        pot_bb = float(getattr(r, "pot_bb") or 0.0)
        stack_bb = float(getattr(r, "effective_stack_bb") or 0.0)

        if pot_bb <= 0 or stack_bb <= 0:
            emit_sentinel({"sha1": sha1, "base_s3_key": base_s3_key}, target="root", reason="bad_pot_or_stack")
            emit_sentinel({"sha1": sha1, "base_s3_key": base_s3_key}, target="facing", reason="bad_pot_or_stack")
            continue

        # menu sizes for this manifest row
        sizes_pct = parse_bet_sizes_cell(getattr(r, "bet_sizes", None))
        if not sizes_pct:
            emit_sentinel({"sha1": sha1, "base_s3_key": base_s3_key}, target="root", reason="no_bet_sizes")
            emit_sentinel({"sha1": sha1, "base_s3_key": base_s3_key}, target="facing", reason="no_bet_sizes")
            continue

        root_kind = oop_root_kind_for_bet_sizing_id(bet_sizing_id)

        # base row fields shared by outputs
        common = {
            "sha1": sha1,
            "stake": getattr(r, "stake", None),
            "scenario": getattr(r, "scenario", None),
            "street": street,
            "ctx": ctx,
            "ip_pos": ip_pos,
            "oop_pos": oop_pos,
            "board": board,
            "board_cluster_id": getattr(r, "board_cluster_id", None),
            "pot_bb": pot_bb,
            "effective_stack_bb": stack_bb,
            "bet_sizing_id": bet_sizing_id,
            "board_mask_52": make_board_mask_52(board),
        }

        for size_pct in sizes_pct:
            solver_key = size_key_for(base_s3_key, int(size_pct))
            local_path = (cache_dir / solver_key).resolve()

            try:
                # ensure local cached file exists
                if not local_path.is_file():
                    # download from S3
                    if not s3_exists(s3, s3_bucket, solver_key):
                        emit_sentinel({**common, "solver_key": solver_key, "size_pct": size_pct},
                                      target="root", reason="missing_solver_output_s3")
                        emit_sentinel({**common, "solver_key": solver_key, "size_pct": size_pct},
                                      target="facing", reason="missing_solver_output_s3")
                        continue
                    s3_download_to(s3, s3_bucket, solver_key, local_path)

                ex = extractor.extract(
                    str(local_path),
                    ctx=ctx,
                    ip_pos=ip_pos,
                    oop_pos=oop_pos,
                    board=board,
                    pot_bb=pot_bb,
                    stack_bb=stack_bb,
                    bet_sizing_id=bet_sizing_id,
                    size_pct=int(size_pct),
                    root_actor="oop",
                    root_bet_kind=root_kind,
                    raise_mults=raise_mults_f,
                )

                # invariants are non-negotiable (your new file)
                validate_extractor_output(ex)

                solver_version = str(getattr(r, "solver_version", "v1") or "v1")

                # --- ROOT ---
                if not ex.root_mix:
                    emit_sentinel(
                        {**common, "solver_key": solver_key, "size_pct": int(size_pct)},
                        target="root",
                        reason="empty_root_mix",
                    )
                else:
                    root_probs = map_root_mix_to_root_vocab(
                        ex.root_mix,
                        root_kind=root_kind,
                        size_pct=int(size_pct),
                    )
                    rows_root.append(
                        make_root_policy_payload(
                            common=common,
                            solver_key=solver_key,
                            solver_version=solver_version,
                            size_pct=int(size_pct),
                            probs=root_probs,
                            weight=1.0,
                            valid=True,
                        )
                    )

                # --- FACING ---
                if not ex.facing_mix:
                    emit_sentinel(
                        {**common, "solver_key": solver_key, "faced_size_pct": int(size_pct)},
                        target="facing",
                        reason="empty_facing_mix",
                    )
                else:
                    facing_probs = map_facing_mix_to_facing_vocab(
                        ex.facing_mix,
                        raise_mults=raise_mults_f,
                    )
                    rows_facing.append(
                        make_facing_policy_payload(
                            common=common,
                            solver_key=solver_key,
                            solver_version=solver_version,
                            faced_size_pct=int(size_pct),
                            probs=facing_probs,
                            weight=1.0,
                            valid=True,
                        )
                    )

            except Exception as e:
                emit_sentinel({**common, "solver_key": solver_key, "size_pct": size_pct},
                              target="root", reason=f"exception:{type(e).__name__}:{e}")
                emit_sentinel({**common, "solver_key": solver_key, "faced_size_pct": size_pct},
                              target="facing", reason=f"exception:{type(e).__name__}:{e}")

            if len(rows_root) >= part_rows:
                flush("root")
            if len(rows_facing) >= part_rows:
                flush("facing")

    flush("root")
    flush("facing")
    if dbg_fh:
        dbg_fh.close()

    print("✅ done",
          f"root_parts={part_root_i}",
          f"facing_parts={part_facing_i}",
          sep="  ")


def main():
    import argparse

    ap = argparse.ArgumentParser("Build postflop policy parquet from solver outputs")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--solver-yaml", required=True)
    ap.add_argument("--stake-key", default="Stakes.NL10")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "eu-central-1"))

    ap.add_argument("--out-root", default="data/datasets/postflop_policy_root_parts")
    ap.add_argument("--out-facing", default="data/datasets/postflop_policy_facing_parts")
    ap.add_argument("--part-rows", type=int, default=50_000)

    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--shard-count", type=int, default=None)

    ap.add_argument("--strict", choices=["fail", "skip", "emit_sentinel"], default="emit_sentinel")
    ap.add_argument("--local-cache", default="data/solver_cache")
    ap.add_argument("--debug-jsonl", default=None)

    args = ap.parse_args()

    build_postflop_policy(
        manifest_path=args.manifest,
        solver_yaml_path=args.solver_yaml,
        stake_key=args.stake_key,
        s3_bucket=args.bucket,
        s3_region=args.region,
        out_root_dir=args.out_root,
        out_facing_dir=args.out_facing,
        part_rows=int(args.part_rows),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        strict=args.strict,  # type: ignore[arg-type]
        local_cache_dir=args.local_cache,
        debug_jsonl=args.debug_jsonl,
    )

if __name__ == "__main__":
    main()