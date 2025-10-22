from __future__ import annotations
from functools import lru_cache
import json
import sys
from pathlib import Path
import dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client
from pathlib import Path
from typing import Any, Optional, Mapping
from ml.etl.utils.ec2_shutdown import shutdown_ec2_instance
from ml.etl.utils.postflop import _retry, _cache_path_for_key, _split_positions, \
     _stable_shard_index, _get
from ml.models.policy_consts import ACTION_VOCAB
from ml.utils.board_mask import make_board_mask_52
from ml.etl.rangenet.postflop.texas_solver_extractor import TexasSolverExtractor
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
    parts_local_dir_root: str = "data/datasets/postflop_policy_parts_root",
    parts_local_dir_facing: str = "data/datasets/postflop_policy_parts_facing",
    parts_s3_prefix_root: Optional[str] = None,
    parts_s3_prefix_facing: Optional[str] = None,
    shard_label: Optional[str] = None,
    strict_mode: str = "fail",
    debug_dump: Optional[str] = None,
) -> None:
    s3c = S3Client()
    df = pd.read_parquet(manifest_path)

    out_root   = Path(parts_local_dir_root);   out_root.mkdir(parents=True, exist_ok=True)
    out_facing = Path(parts_local_dir_facing); out_facing.mkdir(parents=True, exist_ok=True)

    rows_root:   list[dict] = []
    rows_facing: list[dict] = []

    parts_written_root   = 0
    parts_written_facing = 0
    x = TexasSolverExtractor()

    diag = {"ok": 0, "sentinel": 0, "zero_mass": 0}
    dump_fh = open(debug_dump, "a", encoding="utf-8") if debug_dump else None

    def _emit_sentinel(base_row: dict, reason: str, *, target: str):
        if dump_fh:
            dump_fh.write(json.dumps({"reason": reason, "target": target, **base_row}) + "\n")
        if strict_mode == "fail":
            raise RuntimeError(f"{reason}: {str(base_row.get('s3_key', '?'))} [{target}]")
        elif strict_mode in {"emit_sentinel", "skip"}:
            row = {**base_row, "valid": 0, "weight": 0.0, "action": "CHECK"}
            for a in ACTION_VOCAB: row[a] = 0.0
            if target == "root":
                rows_root.append(row)
            else:
                rows_facing.append(row)
            diag["sentinel"] += 1

    def _part_name(kind: str, i: int) -> str:
        pref = f"shard-{shard_label}-" if shard_label else ""
        return f"{pref}{kind}-part-{i:05d}.parquet"

    def _ensure_vocab_cols(df: pd.DataFrame) -> pd.DataFrame:
        for a in ACTION_VOCAB:
            if a not in df.columns:
                df[a] = 0.0
        return df

    def flush_root():
        nonlocal rows_root, parts_written_root
        if not rows_root: return
        part_df = _ensure_vocab_cols(pd.DataFrame(rows_root))
        path = out_root / _part_name("root", parts_written_root)
        part_df.to_parquet(path, index=False, engine="pyarrow")
        if parts_s3_prefix_root:
            _retry(lambda: s3c.upload_file(path, f"{parts_s3_prefix_root}/{path.name}"))
        print(f"💾 wrote {path} (rows={len(part_df)})")
        parts_written_root += 1
        rows_root = []
        del part_df;
        gc.collect()

    def flush_facing():
        nonlocal rows_facing, parts_written_facing
        if not rows_facing: return
        part_df = _ensure_vocab_cols(pd.DataFrame(rows_facing))
        path = out_facing / _part_name("facing", parts_written_facing)
        part_df.to_parquet(path, index=False, engine="pyarrow")
        if parts_s3_prefix_facing:
            _retry(lambda: s3c.upload_file(path, f"{parts_s3_prefix_facing}/{path.name}"))
        print(f"💾 wrote {path} (rows={len(part_df)})")
        parts_written_facing += 1
        rows_facing = []
        del part_df;
        gc.collect()

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
            stake = getattr(r, "stake", 0)
            hero_pos = ip_pos

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
                "stake": stake,  # 👈 keep for lineage

                "board": board,
                "board_cluster_id": int(cluster_id) if (cluster_id is not None and pd.notna(cluster_id)) else None,
                "board_mask_52": _mask52_cached(board) if board else [0.0] * 52,

                "bet_sizing_id": menu_id,

                "valid": 1,
            }

            # --- parse with the new extractor (robust to schema/layout) ---
            ex = x.extract(
                str(solver_path),
                ctx=ctx,
                ip_pos=ip_pos,
                oop_pos=oop_pos,
                board=board,
                pot_bb=pot_bb,
                stack_bb=stack_bb,
                bet_sizing_id=menu_id,
            )

            emitted_any = False

            def _emit_row(probs: dict, *, actor: str, facing_flag: int, target: str):
                if not probs:
                    _emit_sentinel(base_row, "empty_probs", target=target)
                    return
                vec = np.array([probs.get(a, 0.0) for a in ACTION_VOCAB], dtype=np.float32)
                s = float(vec.sum())
                if s <= 1e-12:
                    _emit_sentinel(base_row, "zero_mass", target=target)
                    return
                vec /= s
                argmax_action = ACTION_VOCAB[int(vec.argmax())]

                out_row = {
                    **base_row,
                    "actor": actor,  # "ip" for root, "oop" for facing (convention)
                    "facing_bet": facing_flag,  # 0/1
                    "action": argmax_action,
                    "weight": 1.0,
                }
                for a, p in zip(ACTION_VOCAB, vec.tolist()):
                    out_row[a] = float(p)

                if target == "root":
                    rows_root.append(out_row)
                else:
                    rows_facing.append(out_row)

                diag["ok"] += 1

            if ex.ok:
                _emit_row(ex.root_mix, actor="ip", facing_flag=0, target="root")
                _emit_row(ex.facing_mix, actor="oop", facing_flag=1, target="facing")
            else:
                # emit a sentinel to BOTH targets so downstream counts match
                _emit_sentinel(base_row, ex.reason or "extract_failed", target="root")
                _emit_sentinel(base_row, ex.reason or "extract_failed", target="facing")

            # flush each independently when it hits the threshold
            if len(rows_root) >= part_rows: flush_root()
            if len(rows_facing) >= part_rows: flush_facing()

        except Exception as e:
            # ensure dict for sentinel path
            if not isinstance(base_row, dict):
                base_row = {}
            _emit_sentinel(base_row, f"exception: {e}")

    flush_root()
    flush_facing()
    if dump_fh: dump_fh.close()

    print("✅ done.",
          f"root_parts={parts_written_root}",
          f"facing_parts={parts_written_facing}",
          f"ok={diag['ok']}",
          f"sentinel={diag['sentinel']}",
          sep="  ")


# === Shard runner ===
def run_from_config(
    cfg: Mapping[str, Any],
    *,
    manifest_path: str,
    shard_index: Optional[int],
    shard_count: Optional[int],
    part_rows: int,
    parts_local_dir: str,
    sample_n: Optional[int] = None,
    sample_random: bool = False,
    sample_seed: int = 42,
) -> None:
    from pathlib import Path
    import numpy as np
    import pandas as pd

    mpath = Path(manifest_path).expanduser().resolve()
    if not mpath.exists():
        raise FileNotFoundError(f"Manifest not found: {mpath}")

    # Load manifest once
    df = pd.read_parquet(mpath)
    if "node_key" not in df.columns:
        df = df.assign(node_key="root")

    # Shard bookkeeping
    shard_label = "solo"
    if shard_count is not None and shard_index is not None:
        # stable sharding by (s3_key, node_key)
        mask = df.apply(
            lambda r: _stable_shard_index(
                str(r["s3_key"]), str(r["node_key"]), shard_count
            ) == shard_index,
            axis=1,
        )
        df = df.loc[mask].reset_index(drop=True)
        print(f"🧩 shard {shard_index}/{shard_count} → {len(df)} manifest rows")
        shard_label = f"{shard_index:02d}of{shard_count:02d}"

    # Optional sampling
    if sample_n is not None and sample_n > 0 and len(df) > 0:
        if sample_random:
            rng = np.random.default_rng(sample_seed)
            idx = rng.choice(len(df), size=min(sample_n, len(df)), replace=False)
            df = df.iloc[idx].reset_index(drop=True)
        else:
            df = df.head(sample_n).reset_index(drop=True)
        print(f"🔎 sample mode: using {len(df)} row(s)")

    # Persist a filtered temp manifest for the builder
    out_dir = Path(parts_local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_manifest = out_dir / f"__tmp_manifest_{shard_label}.parquet"
    df.to_parquet(tmp_manifest, index=False)

    debug_dump = out_dir / f"debug-{shard_label}.jsonl"

    root_dir = _get(cfg, "builder.policy_parts_dir_root", "data/datasets/postflop_policy_parts_root")
    facing_dir = _get(cfg, "builder.policy_parts_dir_facing", "data/datasets/postflop_policy_parts_facing")

    # ensure both exist
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    Path(facing_dir).mkdir(parents=True, exist_ok=True)

    build_postflop_policy(
        manifest_path=tmp_manifest,
        cfg=cfg,
        part_rows=part_rows,
        parts_local_dir_root=root_dir,
        parts_local_dir_facing=facing_dir,
        parts_s3_prefix_root=_get(cfg, "builder.parts_s3_prefix_root", None),
        parts_s3_prefix_facing=_get(cfg, "builder.parts_s3_prefix_facing", None),
        shard_label=shard_label,
        strict_mode=_get(cfg, "builder.strict_mode", "fail"),
        debug_dump=str(debug_dump),
    )

    print("✅ run_from_config complete.")


if __name__ == "__main__":
    import argparse
    from ml.utils.config import load_model_config

    ap = argparse.ArgumentParser("Build Postflop Policy (sharded, streaming parquet parts)")
    ap.add_argument("--manifest", type=str, required=True,  # <— NEW: explicit parquet path
                    help="Path to the manifest parquet to consume")
    ap.add_argument("--config", type=str, default="rangenet/postflop")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--shard-count", type=int, default=None)
    ap.add_argument("--part-rows", type=int, default=500)
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
            manifest_path=args.manifest,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            part_rows=args.part_rows,
            parts_local_dir=args.parts_local_dir,
            sample_n=args.sample_n,
            sample_random=args.sample_random,
            sample_seed=args.sample_seed,
        )
    finally:
        shutdown_ec2_instance(mode="stop", wait_seconds=10)