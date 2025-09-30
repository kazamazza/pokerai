from __future__ import annotations
import sys
from pathlib import Path
import dotenv


ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from ml.etl.rangenet.postflop.solver_policy_parser import PolicyParseConfig, SolverPolicyParser
from infra.storage.s3_client import S3Client
from pathlib import Path
from typing import Any, Optional, Mapping
from ml.etl.utils.ec2_shutdown import shutdown_ec2_instance
from ml.etl.utils.postflop import _retry, _cache_path_for_key, _split_positions, \
     _stable_shard_index, _get
from ml.models.policy_consts import ACTION_VOCAB

dotenv.load_dotenv()

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
    Universal postflop-policy builder (refactored to use shared utils).
    Captures IP (CHECK/BET%) and OOP (FOLD/CALL/RAISE*/ALLIN or CHECK/DONK).
    Robust to label variants.
    """
    import json, re, gzip, io
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_parquet(manifest_path)
    s3c = S3Client()
    use_clusters = bool(_get(cfg, "worker.use_board_clusters", True))

    out_dir = Path(parts_local_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows_buffer: list[dict] = []
    parts_written = 0

    diag = {
        "ok": 0,
        "sentinel": 0,
        "zero_mass_root": 0,
        "no_children": 0,
        "child_resolve_fail": 0,
        "child_not_action": 0,
        "zero_mass_child": 0,
    }
    counters = {
        "root_has_any_bet": 0,
        "oop_facing_bet_nodes": 0,
        "oop_raises_present_in_actions": 0,
        "oop_raises_present_in_strat": 0,
        "rows_with_nonzero_raise_mass": 0,
    }
    dump_fh = open(debug_dump, "a", encoding="utf-8") if debug_dump else None

    def _emit_or_handle_sentinel(base_row: dict, reason_key: str, extra: dict):
        diag[reason_key] = diag.get(reason_key, 0) + 1
        if dump_fh:
            dump_fh.write(json.dumps({"reason": reason_key, **extra}) + "\n")
        if strict_mode == "fail":
            raise RuntimeError(f"{reason_key}: {extra.get('s3_key','?')}")
        elif strict_mode == "emit_sentinel":
            row = {**base_row, "valid": 0, "weight": 0.0, "action": "CHECK"}
            for a in ACTION_VOCAB: row[a] = 0.0
            rows_buffer.append(row); diag["sentinel"] += 1

    def part_name(i: int) -> str:
        pref = f"shard-{shard_label}-" if shard_label else ""
        return f"{pref}part-{i:05d}.parquet"

    def flush_part():
        nonlocal rows_buffer, parts_written
        if not rows_buffer: return
        part_df = pd.DataFrame(rows_buffer)
        for a in ACTION_VOCAB:
            if a not in part_df.columns: part_df[a] = 0.0
        path = out_dir / part_name(parts_written)
        part_df.to_parquet(path, index=False)
        print(f"💾 wrote {path} (rows={len(part_df)})")
        if parts_s3_prefix:
            _retry(lambda: s3c.upload_file(path, f"{parts_s3_prefix}/{path.name}"))
        parts_written += 1
        rows_buffer = []

    def _load_solver_json_local_or_s3(cfg: Mapping[str, Any], s3c, s3_key: str) -> dict:
        local_path = _cache_path_for_key(cfg, s3_key)
        if not local_path.is_file():
            _retry(lambda: s3c.download_file_if_missing(s3_key, local_path))
        b = local_path.read_bytes()
        if local_path.suffix == ".gz" or (len(b) >= 2 and b[:2] == b"\x1f\x8b"):
            with gzip.GzipFile(fileobj=io.BytesIO(b)) as gz:
                text = gz.read().decode("utf-8")
            return json.loads(text)
        return json.loads(b.decode("utf-8"))

    parser = SolverPolicyParser()
    parser = SolverPolicyParser()

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Building postflop policy"):
        # keep base_row defined outside try so it's available in except
        base_row = None
        try:
            s3_key = str(r["s3_key"])
            node_key = str(r.get("node_key", "root"))

            ip_pos, oop_pos = _split_positions(str(r.get("positions", "")))
            menu_id = str(r.get("bet_sizing_id", "") or "")
            role = _role_from_menu(menu_id).upper()
            actor = "ip" if role.endswith("_IP") else ("oop" if role.endswith("_OOP") else "ip")
            hero_pos = ip_pos if actor == "ip" else oop_pos

            stack_bb = float(r["effective_stack_bb"])
            pot_bb = float(r["pot_bb"])
            street = int(r["street"])
            ctx = str(r["ctx"])

            board_cluster = None
            if use_clusters and "board_cluster_id" in r:
                board_cluster = int(r["board_cluster_id"])

            js = _load_solver_json_local_or_s3(cfg, s3c, s3_key)

            base_row = {
                "s3_key": s3_key, "node_key": node_key,
                "stack_bb": int(round(stack_bb)), "pot_bb": pot_bb,
                "hero_pos": hero_pos, "ip_pos": ip_pos, "oop_pos": oop_pos,
                "street": street, "ctx": ctx, "actor": actor,
                "bet_sizing_id": menu_id, "valid": 1,
            }
            if board_cluster is not None:
                base_row["board_cluster"] = board_cluster

            # ---- parse with the centralized parser ----
            cfg_parser = PolicyParseConfig(pot_bb=pot_bb, stack_bb=stack_bb, role=role)
            result = parser.parse(js, cfg_parser)

            if not result.ok:
                _emit_or_handle_sentinel(base_row, result.debug.get("reason", "parse_fail"), {"s3_key": s3_key})
                continue

            vec = np.asarray(result.vec, dtype=np.float32)
            s = float(vec.sum())
            if s <= 0:
                _emit_or_handle_sentinel(base_row, "zero_mass_child", {"s3_key": s3_key})
                continue

            argmax_action = ACTION_VOCAB[int(vec.argmax())]
            out_row = {
                **base_row,
                "action": argmax_action,
                "bet_size_pct": np.nan,
                "weight": 1.0,
            }
            for a, p in zip(ACTION_VOCAB, vec.tolist()):
                out_row[a] = float(p)

            rows_buffer.append(out_row)
            diag["ok"] += 1
            if len(rows_buffer) >= part_rows:
                flush_part()

            # optional counters fed by parser.debug (no duplicate logic)
            dbg = result.debug or {}
            pres = dbg.get("presence", {})
            if dbg.get("any_raise"): counters["rows_with_nonzero_raise_mass"] += 1
            if pres.get("has_raise"): counters["oop_raises_present_in_actions"] += 1

        except Exception:
            if strict_mode == "fail":
                raise
            elif strict_mode == "emit_sentinel":
                # minimal sentinel row
                row = {
                    "s3_key": (base_row or {}).get("s3_key", str(r.get("s3_key", "?"))),
                    "node_key": (base_row or {}).get("node_key", str(r.get("node_key", "root"))),
                    "stack_bb": int(round(float(r.get("effective_stack_bb", 0)))),
                    "pot_bb": float(r.get("pot_bb", 0.0)),
                    "hero_pos": (base_row or {}).get("hero_pos", ""),
                    "ip_pos": (base_row or {}).get("ip_pos", ""),
                    "oop_pos": (base_row or {}).get("oop_pos", ""),
                    "street": int(r.get("street", 1)),
                    "ctx": str(r.get("ctx", "")),
                    "actor": (base_row or {}).get("actor", ""),
                    "bet_sizing_id": str(r.get("bet_sizing_id", "")),
                    "valid": 0,
                    "weight": 0.0,
                    "action": "CHECK",
                }
                for a in ACTION_VOCAB:
                    row[a] = 0.0
                rows_buffer.append(row)
                diag["sentinel"] += 1
            # strict_mode == 'skip' → just drop the row and continue

    flush_part()
    if dump_fh: dump_fh.close()

    print("✅ done.",
          f"parts={parts_written}",
          f"ok={diag['ok']}",
          f"sentinel={diag['sentinel']}",
          f"zero_mass_root={diag['zero_mass_root']}",
          f"no_children={diag['no_children']}",
          f"child_resolve_fail={diag['child_resolve_fail']}",
          f"child_not_action={diag['child_not_action']}",
          f"zero_mass_child={diag['zero_mass_child']}",
          sep="  ")
    print("RAISE visibility:",
          f"root_has_any_bet={counters['root_has_any_bet']}",
          f"oop_facing_bet_nodes={counters['oop_facing_bet_nodes']}",
          f"raises_in_actions={counters['oop_raises_present_in_actions']}",
          f"raises_in_strat={counters['oop_raises_present_in_strat']}",
          f"rows_with_nonzero_raise_mass={counters['rows_with_nonzero_raise_mass']}",
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
    import pandas as pd
    import numpy as np

    manifest = Path(_get(cfg, "inputs.manifest", "data/artifacts/rangenet_postflop_flop_manifest.parquet"))
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
        shard_label = f"{shard_index:02d}of{shard_count:02d}"
    else:
        shard_label = "solo"

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

    tmp_manifest = manifest.with_suffix(".shard.tmp.parquet")
    df.to_parquet(tmp_manifest, index=False)

    # (optional) per-shard debug log to avoid collisions
    debug_dump = Path(parts_local_dir) / f"debug-{shard_label}.jsonl"

    build_postflop_policy(
        manifest_path=tmp_manifest,
        cfg=cfg,
        part_rows=part_rows,
        parts_local_dir=parts_local_dir,
        parts_s3_prefix=parts_s3_prefix,
        shard_label=shard_label,           # <-- unique per worker
        strict_mode=_get(cfg, "builder.strict_mode", "fail"),
        debug_dump=str(debug_dump),
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
    ap.add_argument("--shard-label", type=str, default=None,
                    help="String tag to prefix part filenames, e.g. 'test10' or shard index")
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