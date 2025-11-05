from __future__ import annotations

import os
import tempfile
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

from ml.config.bet_menus import _parse_menu_id
from infra.storage.s3_client import S3Client
from pathlib import Path
from typing import Any, Optional, Mapping, cast, Literal
from ml.etl.utils.ec2_shutdown import shutdown_ec2_instance
from ml.etl.utils.postflop import _retry, _cache_path_for_key, _split_positions, \
     _stable_shard_index, _get
from ml.models.policy_consts import ACTION_VOCAB
from ml.utils.board_mask import make_board_mask_52
from ml.etl.rangenet.postflop.texas_solver_extractor import TexasSolverExtractor
dotenv.load_dotenv()

def _oop_is_caller(menu_id: str) -> bool:
    g, r = _parse_menu_id(str(menu_id))  # e.g. ("3bet_hu", "Aggressor_OOP")
    if g.startswith("limped_single"):
        return True
    if g == "srp_hu":
        return r.endswith("PFR_IP") or r.endswith("Caller_OOP")
    if g == "3bet_hu":
        return r.endswith("Aggressor_IP")
    if g == "4bet_hu":
        return r.endswith("Aggressor_IP")
    return True  # safe default

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

def _parse_sizes_to_pct(cell) -> list[int]:
    # Accepts None, [0.33, 0.66], [{"element": 0.33}, ...], [33, 66]
    if cell is None:
        return []
    vals = []
    if isinstance(cell, list):
        for it in cell:
            v = it.get("element") if isinstance(it, dict) else it
            if v is None:
                continue
            f = float(v)
            vals.append(int(round(100.0 * f)) if f <= 3.0 else int(round(f)))
    else:
        f = float(cell)
        vals.append(int(round(100.0 * f)) if f <= 3.0 else int(round(f)))
    # keep sane range and dedup
    return sorted({x for x in vals if 1 <= x <= 200})

def _norm_size_pct(x: Any) -> int:
    """Return 1..200. Accepts 0.33/0.66 or 33/66. Fixes double-scaled 3300."""
    f = float(x)
    if f <= 3.0:         # fraction of pot
        f *= 100.0
    if f > 200.0:        # likely double-scaled (e.g., 3300)
        f /= 100.0
    f = max(1.0, min(200.0, f))
    return int(round(f))

def s3_key_for_size(base_key: str, size_pct: int) -> str:
    """
    Builds a concrete S3 key for a given size, using compressed solver results (.json.gz).
    Example:
      base_key: solver/outputs/v1/street=1/.../Ah9hAs
      size_pct: 50
      → solver/outputs/v1/street=1/.../Ah9hAs/size=50p/output_result.json.gz
    """
    base = str(base_key).rstrip("/")
    return f"{base}/size={int(size_pct)}p/output_result.json.gz"


def _filter_root_only(m: dict, *, root_bet_kind: str, size_pct: int) -> dict:
    """Keep only CHECK + the single size token (BET_S or DONK_S) + optional ALLIN."""
    keep = {"CHECK", "ALLIN"}
    size_tok = f'{"DONK" if root_bet_kind=="donk" else "BET"}_{int(size_pct)}'
    keep.add(size_tok)
    return {k: float(m.get(k, 0.0)) if k in keep else 0.0 for k in ACTION_VOCAB}

def _filter_facing_only(m: dict) -> dict:
    """Keep only FOLD/CALL/RAISE_* and ALLIN for facing responses."""
    return {k: float(m.get(k, 0.0)) if (k in ("FOLD","CALL","ALLIN") or k.startswith("RAISE_")) else 0.0
            for k in ACTION_VOCAB}

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
    strict_mode: str = "emit_sentinel",
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
    diag = {"ok": 0, "sentinel": 0}
    dump_fh = open(debug_dump, "a", encoding="utf-8") if debug_dump else None

    def _emit_sentinel(base_row: dict, reason: str, *, target: str):
        if dump_fh:
            dump_fh.write(json.dumps({"reason": reason, "target": target, **base_row}) + "\n")
        if strict_mode == "fail":
            raise RuntimeError(f"{reason}: {str(base_row.get('s3_key', '?'))} [{target}]")
        elif strict_mode in {"emit_sentinel", "skip"}:
            row = {**base_row, "valid": 0, "weight": 0.0, "action": "CHECK"}
            for a in ACTION_VOCAB:
                row[a] = 0.0
            (rows_root if target == "root" else rows_facing).append(row)
            diag["sentinel"] += 1

    def _part_name(kind: str, i: int) -> str:
        pref = f"shard-{shard_label}-" if shard_label else ""
        return f"{pref}{kind}-part-{i:05d}.parquet"

    def _ensure_schema_cols(df: pd.DataFrame) -> pd.DataFrame:
        # add missing vocab cols + the new size columns + bet_sizes menu
        for a in ACTION_VOCAB:
            if a not in df.columns:
                df[a] = 0.0
        for c in ("bet_sizes", "size_pct", "faced_size_pct"):
            if c not in df.columns:
                df[c] = None
        return df

    def flush_root():
        nonlocal rows_root, parts_written_root
        if not rows_root:
            return
        part_df = _ensure_schema_cols(pd.DataFrame(rows_root))
        path = out_root / _part_name("root", parts_written_root)
        part_df.to_parquet(path, index=False, engine="pyarrow")
        if parts_s3_prefix_root:
            _retry(lambda: s3c.upload_file(path, f"{parts_s3_prefix_root}/{path.name}"))
        print(f"💾 wrote {path} (rows={len(part_df)})")
        parts_written_root += 1
        rows_root = []
        del part_df; gc.collect()

    def flush_facing():
        nonlocal rows_facing, parts_written_facing
        if not rows_facing:
            return
        part_df = _ensure_schema_cols(pd.DataFrame(rows_facing))
        path = out_facing / _part_name("facing", parts_written_facing)
        part_df.to_parquet(path, index=False, engine="pyarrow")
        if parts_s3_prefix_facing:
            _retry(lambda: s3c.upload_file(path, f"{parts_s3_prefix_facing}/{path.name}"))
        print(f"💾 wrote {path} (rows={len(part_df)})")
        parts_written_facing += 1
        rows_facing = []
        del part_df; gc.collect()

    def _resolve_solver_path(cfg: Mapping[str, Any], s3_key: Any, s3_client: Any) -> tuple[Path, bool]:
        """
        Returns (local_path, ephemeral). If 'ephemeral' is True, caller should unlink after use.
        """
        s3_key = _normalize_s3_key(s3_key)

        local_dir = _get(cfg, "solver.local_solver_dir", None)
        if local_dir:
            p = Path(local_dir) / Path(s3_key)
            if p.is_file():
                return p, False  # not ephemeral
            print(f"[warn] not found locally, will fall back to S3 download: {p}")

        # Prefix with configured s3_prefix only if base key doesn't already include it
        s3_prefix = _get(cfg, "solver.s3_prefix", "").strip("/")
        key_norm = s3_key.lstrip("/")

        if s3_prefix and not key_norm.startswith(f"{s3_prefix}/") and key_norm != s3_prefix:
            s3_key = f"{s3_prefix}/{key_norm}"
        else:
            s3_key = key_norm  # ensure no leading slash

        # Build a temp destination under the cache dir if present, else system temp.
        cache_dir = Path(_get(cfg, "solver.local_cache_dir", tempfile.gettempdir()))
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Keep original filename to preserve .json / .json.gz for the extractor.
        filename = Path(s3_key).name
        tmp_path = Path(tempfile.mkstemp(prefix=".dl_", suffix=f".{filename}", dir=str(cache_dir))[1])
        print(f"[warn] downloading {s3_key} to {tmp_path}")
        try:
            s3_client.download_file(s3_key=s3_key, local_path=tmp_path)
            return tmp_path, True
        except Exception as e:
            # Best-effort cleanup of partials
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(f"download failed for {s3_key}: {e}") from e

    for r in tqdm(df.itertuples(index=False, name="Row"), total=len(df), desc="Building postflop policy"):
        base_row = {}
        try:
            base_key  = _normalize_s3_key(getattr(r, "s3_key", None))
            node_key  = str(getattr(r, "node_key", "root"))
            positions = str(getattr(r, "positions", "") or "")
            ip_pos, oop_pos = _split_positions(positions)

            stack_bb = float(getattr(r, "effective_stack_bb", 0.0) or 0.0)
            pot_bb   = float(getattr(r, "pot_bb", 0.0) or 0.0)
            street   = int(getattr(r, "street", 1) or 1)
            ctx      = str(getattr(r, "ctx", "") or "")
            menu_id  = str(getattr(r, "bet_sizing_id", "") or "")

            board     = str(getattr(r, "board", "") or "")
            cluster_id= getattr(r, "board_cluster_id", None)
            stake     = getattr(r, "stake", 0)

            raw_bets = getattr(r, "bet_sizes", None)
            if hasattr(raw_bets, "to_pylist"):
                raw_bets = raw_bets.to_pylist()
            if isinstance(raw_bets, np.ndarray):
                raw_bets = raw_bets.tolist()

            menu_sizes_pct: list[int] = []
            if raw_bets is None:
                menu_sizes_pct = []
            elif isinstance(raw_bets, list):
                for it in raw_bets:
                    v = it.get("element") if isinstance(it, dict) else it
                    if v is not None:
                        menu_sizes_pct.append(_norm_size_pct(v))
            else:
                menu_sizes_pct = [_norm_size_pct(raw_bets)]

            menu_sizes_pct = sorted(set(menu_sizes_pct))
            if not menu_sizes_pct:
                _emit_sentinel({"s3_key": base_key}, "no_bet_sizes", target="root")
                _emit_sentinel({"s3_key": base_key}, "no_bet_sizes", target="facing")
                continue

            base_row = {
                "s3_key": base_key,
                "node_key": node_key,
                "stack_bb": int(round(stack_bb)),
                "effective_stack_bb": float(stack_bb),
                "pot_bb": float(pot_bb),

                "ip_pos": ip_pos,
                "oop_pos": oop_pos,
                "hero_pos": ip_pos,

                "street": int(street),
                "ctx": ctx,
                "stakes_id": stake,

                "board": board,
                "board_cluster_id": int(cluster_id) if (cluster_id is not None and pd.notna(cluster_id)) else None,
                "board_mask_52": _mask52_cached(board) if board else [0.0] * 52,

                "bet_sizing_id": menu_id,
                "bet_sizes": menu_sizes_pct,         # keep full menu for gating later
                "size_pct": None,                # filled per-root emit
                "faced_size_pct": None,          # filled per-facing emit

                "valid": 1,
            }

            for size_pct in menu_sizes_pct:
                s3_key_sz = s3_key_for_size(base_key, int(size_pct))
                solver_path, ephemeral = _resolve_solver_path(cfg, s3_key_sz, s3c)

                root_bet_kind = "donk" if _oop_is_caller(menu_id) else "bet"

                ex = x.extract(
                    str(solver_path),
                    ctx=ctx,
                    ip_pos=ip_pos,
                    oop_pos=oop_pos,
                    board=board,
                    pot_bb=pot_bb,
                    stack_bb=stack_bb,
                    bet_sizing_id=menu_id,
                    size_pct=int(size_pct),
                    root_actor="oop",
                    root_bet_kind=cast(Literal["donk", "bet"], root_bet_kind),
                    raise_mults = cfg.get("solver", {}).get("raise_mult", [1.5, 2.0, 3.0])
                )

                if not ex.ok and ex.reason in {"zero_mass", "empty_root", "root_allin_only"}:
                    if strict_mode == "emit_sentinel":
                        _emit_sentinel(base_row, f"extract_failed: {ex.reason}", target="root")
                    elif strict_mode == "skip":
                        diag["sentinel"] += 1  # count & drop
                    else:
                        # strict_mode == "fail"
                        raise RuntimeError(f"extract_failed: {ex.reason}: {s3_key_sz}")

                    # cleanup temp if any, then continue to next size
                    if ephemeral:
                        try:
                            os.unlink(solver_path)
                        except FileNotFoundError:
                            pass
                        except Exception as _cleanup_err:
                            print(f"[warn] failed to unlink temp file {solver_path}: {_cleanup_err}")
                    continue

                def _emit_row(probs: dict, *, actor: str, facing_flag: int, target: str):
                    if not probs:
                        _emit_sentinel({**base_row, "s3_key": s3_key_sz}, "empty_probs", target=target);
                        return
                    vec = np.array([probs.get(a, 0.0) for a in ACTION_VOCAB], dtype=np.float32)
                    ssum = float(vec.sum())
                    if ssum <= 1e-12:
                        _emit_sentinel({**base_row, "s3_key": s3_key_sz}, "zero_mass", target=target);
                        return
                    vec /= ssum
                    argmax_action = ACTION_VOCAB[int(vec.argmax())]

                    out_row = {
                        **base_row,
                        "s3_key": s3_key_sz,
                        "actor": actor,  # "oop" for root; "oop"/"ip" for facing as derived
                        "facing_bet": facing_flag,  # 0 for root, 1 for facing
                        "action": argmax_action,
                        "weight": 1.0,
                        "hero_pos": oop_pos if target in {"root", "facing"} else base_row.get("hero_pos"),
                    }
                    # write probs
                    for a, p in zip(ACTION_VOCAB, vec.tolist()):
                        out_row[a] = float(p)

                    if target == "root":
                        out_row["size_pct"] = size_pct
                        out_row["faced_size_pct"] = None
                        rows_root.append(out_row)
                    else:
                        out_row["size_pct"] = None
                        out_row["faced_size_pct"] = size_pct
                        rows_facing.append(out_row)

                    diag["ok"] += 1

                if ex.ok:
                    emitted_any = False
                    if ex.root_mix and len(ex.root_mix) > 0:
                        root_filtered = _filter_root_only(ex.root_mix,
                                                          root_bet_kind=root_bet_kind,
                                                          size_pct=size_pct)
                        _emit_row(root_filtered, actor="oop", facing_flag=0, target="root")
                        emitted_any = True
                    elif strict_mode in {"emit_sentinel"}:
                        _emit_sentinel({**base_row, "s3_key": s3_key_sz}, "empty_root_mix", target="root")

                    fp = (ex.meta.get("facing_path") or [])
                    if fp and isinstance(fp, list) and len(fp) >= 1:
                        first_lbl = str(fp[0]).lower()
                        facing_actor = "ip" if first_lbl.startswith("bet") else "oop"
                    else:
                        facing_actor = "oop"  # conservative default: OOP faces IP c-bet

                    if ex.facing_mix and len(ex.facing_mix) > 0:
                        facing_filtered = _filter_facing_only(ex.facing_mix)
                        _emit_row(facing_filtered, actor=facing_actor, facing_flag=1, target="facing")
                        emitted_any = True
                    elif strict_mode in {"emit_sentinel"}:
                        _emit_sentinel({**base_row, "s3_key": s3_key_sz}, "empty_facing_mix", target="facing")

                    if not emitted_any and strict_mode == "fail":
                        raise RuntimeError(f"no_applicable_mix: {s3_key_sz}")
                else:
                    if strict_mode == "fail":
                        raise RuntimeError(f"extract_failed: {ex.reason or 'unknown'}: {s3_key_sz}")
                    elif strict_mode == "emit_sentinel":
                        _emit_sentinel({**base_row, "s3_key": s3_key_sz}, ex.reason or "extract_failed", target="root")
                        _emit_sentinel({**base_row, "s3_key": s3_key_sz}, ex.reason or "extract_failed",
                                       target="facing")
                    # strict_mode == "skip": emit nothing

                if len(rows_root) >= part_rows: flush_root()
                if len(rows_facing) >= part_rows: flush_facing()

        except Exception as e:
            if not isinstance(base_row, dict):
                base_row = {}
            _emit_sentinel(base_row, f"exception: {e}", target="root")
            _emit_sentinel(base_row, f"exception: {e}", target="facing")

    flush_root()
    flush_facing()
    if dump_fh: dump_fh.close()

    print("✅ done.",
          f"root_parts={parts_written_root}",
          f"facing_parts={parts_written_facing}",
          f"ok={diag['ok']}",
          f"sentinel={diag['sentinel']}",
          sep="  ")


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
        parts_s3_prefix_root=_get(cfg, "build.parts_s3_prefix_root", None),
        parts_s3_prefix_facing=_get(cfg, "build.parts_s3_prefix_facing", None),
        shard_label=shard_label,
        strict_mode="emit_sentinel",
        debug_dump=str(debug_dump),
    )

    print("✅ run_from_config complete.")


if __name__ == "__main__":
    import argparse
    from ml.utils.config import load_model_config

    ap = argparse.ArgumentParser("Build Postflop Policy (sharded, streaming parquet parts)")
    ap.add_argument("--manifest", type=str, required=True,
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