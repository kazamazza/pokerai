from __future__ import annotations
from functools import lru_cache
import json
import sys
from pathlib import Path
import dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client
from pathlib import Path
from typing import Any, Optional, Mapping
from ml.etl.utils.ec2_shutdown import shutdown_ec2_instance
from ml.etl.utils.postflop import _retry, _cache_path_for_key, _split_positions, \
     _stable_shard_index, _get
from ml.models.policy_consts import ACTION_VOCAB
from ml.config.bet_menus import BET_SIZE_MENUS, DEFAULT_MENU
from ml.utils.board_mask import make_board_mask_52
from ml.etl.rangenet.postflop.solver_policy_parser import parse_solver_simple
dotenv.load_dotenv()


def _normalize_s3_key(s3_key: Any) -> str:
    # Unwrap dicts until we get a string
    while isinstance(s3_key, dict):
        if "path" in s3_key:
            s3_key = s3_key["path"]
        elif "key" in s3_key:
            s3_key = s3_key["key"]
        else:
            # just take first value
            s3_key = next(iter(s3_key.values()))
    return str(s3_key)

def _ensure_float_list(v) -> list[float]:
    """Fast, schema-tolerant cast to List[float]."""
    if v is None:
        return []

    # Fast paths
    if isinstance(v, (list, tuple)):
        if v and isinstance(v[0], dict) and "element" in v[0]:
            # Parquet array-of-records: [{"element": 0.33}, ...]
            out = []
            for d in v:
                try:
                    out.append(float(d.get("element")))
                except Exception:
                    continue
            return out
        # Normal list/tuple of numbers/strings
        out = []
        for x in v:
            try:
                out.append(float(x))
            except Exception:
                # tolerate junk quietly
                continue
        return out

    if isinstance(v, (int, float)):
        return [float(v)]

    if isinstance(v, str):
        s = v.strip()
        # Only JSON-decode if it *looks* like JSON array/number
        if s and (s[0] in "[{" or s[0].isdigit() or s[0] in "+-."):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, (list, tuple)):
                    return [float(x) for x in parsed if _is_number_like(x)]
                if isinstance(parsed, (int, float)):
                    return [float(parsed)]
            except Exception:
                pass
        return []

    return []

def _is_number_like(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

@lru_cache(maxsize=50000)
def _mask52_cached(board: str) -> list[float]:
    return make_board_mask_52(board)

def build_postflop_policy(
    manifest_path: Path,
    cfg: Mapping[str, Any],
    *,
    part_rows: int = 2000,
    parts_local_dir: str = "data/datasets/postflop_policy_parts",
    parts_s3_prefix: Optional[str] = None,
    shard_label: Optional[str] = None,
    strict_mode: str = "fail",                 # 'fail' | 'emit_sentinel' | 'skip'
    debug_dump: Optional[str] = None,          # path to .jsonl for problematic rows
) -> None:
    """
    Postflop-policy builder using the new solver parser.
    Produces action probability vectors aligned with ACTION_VOCAB.
    """
    s3c = S3Client()
    df = pd.read_parquet(manifest_path)

    out_dir = Path(parts_local_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows_buffer: list[dict] = []
    parts_written = 0

    diag = {"ok": 0, "sentinel": 0, "zero_mass": 0}
    dump_fh = open(debug_dump, "a", encoding="utf-8") if debug_dump else None

    def _emit_sentinel(base_row: dict, reason: str):
        if dump_fh:
            dump_fh.write(json.dumps({"reason": reason, **base_row}) + "\n")
        if strict_mode == "fail":
            raise RuntimeError(f"{reason}: {str(base_row.get('s3_key', '?'))}")
        elif strict_mode in {"emit_sentinel", "skip"}:
            row = {**base_row, "valid": 0, "weight": 0.0, "action": "CHECK"}
            for a in ACTION_VOCAB:
                row[a] = 0.0
            rows_buffer.append(row)
            diag["sentinel"] += 1

    def part_name(i: int) -> str:
        pref = f"shard-{shard_label}-" if shard_label else ""
        return f"{pref}part-{i:05d}.parquet"

    def flush_part():
        nonlocal rows_buffer, parts_written
        if not rows_buffer: return
        part_df = pd.DataFrame(rows_buffer)
        for a in ACTION_VOCAB:
            if a not in part_df.columns:
                part_df[a] = 0.0
        path = out_dir / part_name(parts_written)
        part_df.to_parquet(path, index=False)
        print(f"💾 wrote {path} (rows={len(part_df)})")
        if parts_s3_prefix:
            _retry(lambda: s3c.upload_file(path, f"{parts_s3_prefix}/{path.name}"))
        parts_written += 1
        rows_buffer = []

    def _resolve_solver_path(cfg: Mapping[str, Any], s3_key: Any) -> Path:
        """Return a local filesystem path for the given s3_key.
        1) Try inputs.local_solver_dir + s3_key
        2) Fall back to cache path (which may download if missing, depending on your cache impl)
        """
        s3_key = _normalize_s3_key(s3_key)
        local_dir = _get(cfg, "inputs.local_solver_dir", None)
        if local_dir:
            p = Path(local_dir) / Path(s3_key)
            if p.is_file():
                return p
            # Optional: warn once if not found locally
            print(f"[warn] not found locally, will fall back to cache: {p}")

        return _cache_path_for_key(cfg, s3_key)

    # === Loop rows ===
    # === Loop rows ===
    for r in tqdm(df.itertuples(index=False, name="Row"), total=len(df), desc="Building postflop policy"):
        base_row = {}
        try:
            # --- read from namedtuple safely ---
            raw_key = getattr(r, "s3_key", None)
            s3_key = _normalize_s3_key(raw_key)
            node_key = str(getattr(r, "node_key", "root"))
            positions = str(getattr(r, "positions", "") or "")
            ip_pos, oop_pos = _split_positions(positions)

            stack_bb = float(getattr(r, "effective_stack_bb", 0.0) or 0.0)
            pot_bb = float(getattr(r, "pot_bb", 0.0) or 0.0)
            street = int(getattr(r, "street", 1) or 1)
            ctx = str(getattr(r, "ctx", "") or "")
            menu_id = str(getattr(r, "bet_sizing_id", "") or "")

            solver_path = _resolve_solver_path(cfg, s3_key)
            board = str(getattr(r, "board", "") or "")
            cluster_id = getattr(r, "board_cluster_id", None)
            hero_pos = ip_pos

            # --- sizes: prefer manifest column if present ---
            raw_sizes = getattr(r, "bet_sizes", None)
            sizes_manifest = _ensure_float_list(raw_sizes)
            menu_pcts = tuple(sizes_manifest if sizes_manifest else BET_SIZE_MENUS.get(menu_id, DEFAULT_MENU))

            # --- assemble base_row (trainer-ready) ---
            base_row = {
                "s3_key": _normalize_s3_key(s3_key),
                "node_key": node_key,
                "stack_bb": int(round(stack_bb)),
                "effective_stack_bb": float(stack_bb),
                "pot_bb": float(pot_bb),

                "ip_pos": ip_pos,
                "oop_pos": oop_pos,
                "hero_pos": hero_pos,

                "street": int(street),
                "ctx": ctx,

                "board": board,
                "board_cluster_id": int(cluster_id) if (cluster_id is not None and pd.notna(cluster_id)) else None,
                "board_mask_52": _mask52_cached(board) if board else [0.0] * 52,

                "bet_sizing_id": menu_id,
                "bet_sizes": list(menu_pcts),

                "valid": 1,
            }

            # --- parse with the new universal parser: try facing, then root ---
            probs_root, meta_root = parse_solver_simple(str(solver_path), facing_bet=False)
            probs_face, meta_face = parse_solver_simple(str(solver_path), facing_bet=True)

            if probs_face and sum(probs_face.values()) > 0:
                probs = probs_face
                meta = meta_face
                facing_bet = 1
            elif probs_root and sum(probs_root.values()) > 0:
                probs = probs_root
                meta = meta_root
                facing_bet = 0
            else:
                _emit_sentinel(base_row, "zero_mass")
                continue

            # actor: root actor is IP in our generation; facing_bet row is a response side
            actor = "ip" if facing_bet == 0 else "oop"

            # --- vectorize to ACTION_VOCAB ---
            vec = np.array([probs.get(a, 0.0) for a in ACTION_VOCAB], dtype=np.float32)
            mass = float(vec.sum())
            if mass <= 1e-12:
                _emit_sentinel(base_row, "zero_mass")
                continue
            vec /= mass

            argmax_action = ACTION_VOCAB[int(vec.argmax())]

            out_row = {
                **base_row,
                "actor": actor,
                "facing_bet": int(facing_bet),
                "action": argmax_action,
                "weight": 1.0,
            }
            for a, p in zip(ACTION_VOCAB, vec.tolist()):
                out_row[a] = float(p)

            rows_buffer.append(out_row)
            diag["ok"] += 1

            if len(rows_buffer) >= part_rows:
                flush_part()

        except Exception as e:
            # ensure dict for sentinel path
            if not isinstance(base_row, dict):
                base_row = {}
            _emit_sentinel(base_row, f"exception: {e}")

    flush_part()
    if dump_fh: dump_fh.close()

    print("✅ done.", f"parts={parts_written}", f"ok={diag['ok']}", f"sentinel={diag['sentinel']}", sep="  ")


# === Shard runner ===
def run_from_config(
    cfg: Mapping[str, Any],
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
    manifest = Path(_get(cfg, "inputs.manifest", "data/artifacts/rangenet_postflop_flop_manifest.parquet"))
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    df = pd.read_parquet(manifest)

    if "node_key" not in df.columns:
        df = df.assign(node_key="root")

    # optional sharding
    shard_label = "solo"
    if shard_count is not None and shard_index is not None:
        mask = df.apply(lambda r: _stable_shard_index(str(r["s3_key"]), str(r["node_key"]), shard_count) == shard_index, axis=1)
        df = df.loc[mask].reset_index(drop=True)
        print(f"🧩 shard {shard_index}/{shard_count} → {len(df)} manifest rows")
        shard_label = f"{shard_index:02d}of{shard_count:02d}"

    # optional sampling
    if sample_n is not None and sample_n > 0 and len(df) > 0:
        if sample_random:
            rng = np.random.default_rng(sample_seed)
            idx = rng.choice(len(df), size=min(sample_n, len(df)), replace=False)
            df = df.iloc[idx].reset_index(drop=True)
        else:
            df = df.head(sample_n).reset_index(drop=True)
        print(f"🔎 sample mode: using {len(df)} row(s)")

    tmp_manifest = manifest.with_suffix(".shard.tmp.parquet")
    df.to_parquet(tmp_manifest, index=False)

    debug_dump = Path(parts_local_dir) / f"debug-{shard_label}.jsonl"

    build_postflop_policy(
        manifest_path=tmp_manifest,
        cfg=cfg,
        part_rows=part_rows,
        parts_local_dir=parts_local_dir,
        parts_s3_prefix=parts_s3_prefix,
        shard_label=shard_label,
        strict_mode=_get(cfg, "builder.strict_mode", "fail"),
        debug_dump=str(debug_dump),
    )
    print("✅ run_from_config complete.")


if __name__ == "__main__":
    import argparse
    from ml.utils.config import load_model_config

    ap = argparse.ArgumentParser("Build Postflop Policy (sharded, streaming parquet parts)")
    ap.add_argument("--config", type=str, default="rangenet/postflop")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--shard-count", type=int, default=None)
    ap.add_argument("--part-rows", type=int, default=1000)
    ap.add_argument("--parts-local-dir", type=str, default="data/datasets/postflop_policy_parts")
    ap.add_argument("--parts-s3-prefix", type=str, default=None)
    ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--sample-random", action="store_true")
    ap.add_argument("--sample-seed", type=int, default=42)

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