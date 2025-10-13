from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from ml.etl.utils.positions import canon_pos
from ml.config.solver_profiles import profile_for
from typing import List, Dict, Optional, Tuple
from ml.range.solvers.utils.sanitize_pairs import sanitize_position_pairs
from infra.storage.s3_client import S3Client
from ml.features.boards import load_board_clusterer
from ml.utils.config import load_model_config
from ml.etl.rangenet.preflop.range_lookup import PreflopRangeLookup
from ml.range.solvers.keying import s3_key_for_solve, solve_sha1
from ml.etl.rangenet.postflop.helpers_topology import _infer_topology_and_roles, _menu_for, \
    _ctx_for_lookup, compute_pot_bb, bet_menu_for
from ml.features.boards.representatives import discover_representative_flops
from ml.core.types import Stakes
from ml.config.solver import STAKE_CFG

def parse_stake(value) -> Stakes:
    """
    Convert a string or int into a Stakes enum value.
    Accepts 'nl10', 'NL10', 2, Stakes.NL10, etc.
    Defaults to Stakes.NL10 if unrecognized.
    """
    if isinstance(value, Stakes):
        return value
    if isinstance(value, int):
        return Stakes(value)
    if isinstance(value, str):
        key = value.strip().upper()
        if key.startswith("NL") and key[2:].isdigit():
            try:
                return Stakes[key]  # e.g. Stakes["NL10"]
            except KeyError:
                pass
    return Stakes.NL10  # fallback default

def build_manifest(cfg: dict, *, stake: Stakes = Stakes.NL10) -> pd.DataFrame:
    mb      = cfg.get("manifest_build", {}) or {}
    sv      = cfg.get("solver", {}) or {}
    inputs  = cfg.get("inputs", {}) or {}
    rake_tier = str(sv.get("rake_tier", "nl10_5pct_1bbcap"))
    mw_cfg  = cfg.get("multiway", {}) or {}

    # --- NEW: stake-specific tables (centralized) ---
    stake_cfg       = STAKE_CFG[stake]
    board_clusters  = stake_cfg.get("board_clusters", 64)

    multiway_enabled = bool(mw_cfg.get("enable", False))
    multiway_max_players = int(mw_cfg.get("max_players", 3))
    multiway_menu_id = str(mw_cfg.get("default_menu_id", "limped_multi.Any"))
    multiway_default_pot = float(mw_cfg.get("default_flop_pot_bb", 3.0))
    multiway_allow_scen = bool(mw_cfg.get("allow_in_scenarios", True))

    scenarios = mb.get("scenarios") or [{
        "name": "SRP",
        "ctx": "SRP",
        "stacks_bb": [100],
        "position_pairs": [("BTN", "BB")],
    }]

    # --- stake-aware sampling knobs ---
    # Use stake default for board cluster count unless explicitly overridden in config
    n_clusters_limit = int(mb.get("board_clusters_limit", board_clusters))
    boards_per_cluster = int(mb.get("boards_per_cluster", 2))
    sample_pool = int(mb.get("sample_pool", 50000))
    seed = int(cfg.get("seed", 42))
    allow_pair_subs = bool(mb.get("allow_pair_subs", False))
    include_missing = bool(mb.get("include_missing", False))

    raw_delta = mb.get("lookup_max_stack_delta")
    try:
        max_stack_delta: Optional[int] = int(raw_delta) if raw_delta is not None else None
    except (TypeError, ValueError):
        max_stack_delta = None

    # Preflop range lookup (Monker-only)
    lookup = PreflopRangeLookup(
        monker_manifest_parquet=inputs.get("monker_manifest", "data/artifacts/monker_manifest.parquet"),
        sph_manifest_parquet=inputs.get("sph_manifest", "data/artifacts/sph_manifest.parquet"),
        s3_client=S3Client(),
        s3_vendor=inputs.get("vendor_s3_prefix", "data/vendor"),
        cache_dir=sv.get("local_cache_dir", "data/vendor_cache"),
        allow_pair_subs=allow_pair_subs,
        max_stack_delta=max_stack_delta,
    )

    clusterer = load_board_clusterer(cfg)
    boards_by_cluster = discover_representative_flops(
        clusterer=clusterer,
        n_clusters_limit=n_clusters_limit,
        boards_per_cluster=boards_per_cluster,
        seed=seed,
        sample_pool=sample_pool,
    )
    cluster_ids_sorted = sorted(boards_by_cluster.keys(), key=int)

    rows: List[dict] = []
    total_jobs = kept = missing_rows = skipped = 0
    per_scenario_counts: Dict[str, Dict[str, int]] = {}

    for sc in scenarios:
        scenario_name = str(sc.get("name") or sc.get("ctx") or "SCENARIO").upper()
        ctx = str(sc.get("ctx") or scenario_name).upper()
        stacks = [float(x) for x in sc.get("stacks_bb", [100])]
        raw_pairs: List[Tuple[str, str]] = [(str(a), str(b)) for (a, b) in sc.get("position_pairs", [("BTN", "BB")])]
        ctx_up = str(ctx).upper()
        norm_ctx = {
            "LIMPED_SINGLE": "LIMPED_SINGLE",
            "LIMP_SINGLE": "LIMPED_SINGLE",
        }.get(ctx_up, ctx_up)

        pairs = sanitize_position_pairs(raw_pairs, ctx=norm_ctx)
        if not pairs:
            print(f"[warn] scenario {scenario_name} produced no legal (IP,OOP) pairs for ctx={ctx}")
            per_scenario_counts[scenario_name] = {"jobs": 0, "kept": 0, "missing": 0, "skipped": 0}
            continue

        planned = len(stacks) * len(pairs) * len(cluster_ids_sorted) * boards_per_cluster
        if planned > 200_000:
            print(f"[warn] {scenario_name}: ~{planned:,} jobs planned. Consider reducing boards/menus/stacks.")

        sc_jobs = sc_kept = sc_missing = sc_skipped = 0

        # Pick stacks based on context first, then fall back to generic stake stacks
        stacks_for_ctx = stake_cfg.get("stacks_by_ctx", {}).get(norm_ctx)
        if not stacks_for_ctx:
            stacks_for_ctx = stake_cfg.get("stacks", [])

        for stack in stacks_for_ctx:
            for (ip_pos, oop_pos) in pairs:
                ip_pos = canon_pos(ip_pos)
                oop_pos = canon_pos(oop_pos)
                if not ip_pos or not oop_pos or ip_pos == oop_pos:
                    continue

                # --- roles + deterministic pot (stake-aware) ---
                topo, opener, three_bettor = _infer_topology_and_roles(norm_ctx, ip_pos, oop_pos)

                pot_bb = compute_pot_bb(
                    ctx=norm_ctx,
                    opener=opener,
                    ip=ip_pos,
                    three_bettor=three_bettor,
                    stake=stake,  # uses STAKE_CFG[stake] internally
                )

                # --- bet menu id + sizes from stake config ---
                menu_tag = (sc.get("bet_menus") or [None])[0]
                menu_id, bet_sizes = _menu_for(
                    norm_ctx, ip_pos, oop_pos, opener, three_bettor,
                    menu_tag=menu_tag, stake=stake
                )

                try:
                    rng_ip, rng_oop, meta = lookup.ranges_for_pair(
                        stack_bb=stack, ip=ip_pos, oop=oop_pos, ctx=_ctx_for_lookup(norm_ctx), strict=False
                    )
                except Exception as e:
                    rng_ip = rng_oop = None
                    meta = {"error": str(e)}

                if (rng_ip is None or rng_oop is None) and not include_missing:
                    miss = len(cluster_ids_sorted) * boards_per_cluster
                    sc_skipped += miss;
                    skipped += miss
                    continue

                for cluster_id in cluster_ids_sorted:
                    for board_tuple in boards_by_cluster[cluster_id]:
                        board_str = "".join(board_tuple)
                        knobs = profile_for(menu_id)  # {"accuracy","max_iter","allin_threshold"}

                        params = {
                            "street": 1,
                            "scenario": scenario_name,
                            "ctx": norm_ctx,
                            "topology": topo,
                            "rake_tier": rake_tier,

                            "positions": f"{ip_pos}v{oop_pos}",
                            "ip_actor_flop": ip_pos,
                            "oop_actor_flop": oop_pos,

                            "opener": opener,
                            "three_bettor": three_bettor,

                            "board_cluster_id": int(cluster_id),
                            "board": board_str,

                            "pot_bb": float(pot_bb),
                            "effective_stack_bb": float(stack),

                            "bet_sizing_id": menu_id,
                            "bet_sizes": list(bet_sizes),

                            "range_ip": rng_ip or "",
                            "range_oop": rng_oop or "",

                            "accuracy": knobs["accuracy"],
                            "max_iter": knobs["max_iter"],
                            "allin_threshold": knobs["allin_threshold"],
                            "solver_version": "v1",

                            "range_ip_source_stack": meta.get("range_ip_source_stack"),
                            "range_oop_source_stack": meta.get("range_oop_source_stack"),
                            "range_ip_stack_delta": meta.get("range_ip_stack_delta"),
                            "range_oop_stack_delta": meta.get("range_oop_stack_delta"),
                            "range_ip_fallback_level": meta.get("range_ip_fallback_level"),
                            "range_oop_fallback_level": meta.get("range_oop_fallback_level"),
                            "range_pair_substituted": bool(meta.get("range_pair_substituted", False)),
                            "range_ip_source_pair": meta.get("range_ip_source_pair"),
                            "range_oop_source_pair": meta.get("range_oop_source_pair"),
                            "range_source": meta.get("source", "unknown"),
                        }

                        sha = solve_sha1(params)
                        s3_key = s3_key_for_solve(params, sha1=sha, prefix="solver/outputs/v1")

                        rows.append({
                            **params,
                            "sha1": sha,
                            "s3_key": s3_key,
                            "node_key": "root",
                            "weight": 1.0,
                        })
                        total_jobs += 1
                        sc_jobs += 1
                        if not rng_ip or not rng_oop:
                            sc_missing += 1
                            missing_rows += 1
                        else:
                            sc_kept += 1
                            kept += 1

        per_scenario_counts[scenario_name] = {
            "jobs": sc_jobs, "kept": sc_kept, "missing": sc_missing, "skipped": sc_skipped
        }

    df = pd.DataFrame(rows)

    # stake should already be a column in df at this point
    dedup_keys_cfg = (cfg.get("manifest_build", {}) or {}).get("dedup_keys") or []
    if not dedup_keys_cfg:
        # conservative default if config omitted
        dedup_keys_cfg = [
            "street", "ctx", "topology", "ip_actor_flop", "oop_actor_flop",
            "effective_stack_bb", "pot_bb", "bet_sizing_id", "board",
            "board_cluster_id", "range_ip", "range_oop", "stake"
        ]

    before = len(df)
    if "sha1" in df.columns:
        df = df.drop_duplicates(subset=["sha1"]).reset_index(drop=True)
    else:
        # only keep keys present in the dataframe
        key_cols = [c for c in dedup_keys_cfg if c in df.columns]
        df = df.drop_duplicates(subset=key_cols).reset_index(drop=True)
    removed = before - len(df)

    # summary
    print(f"[dbg] scenarios={len(scenarios)} → jobs={total_jobs:,} "
          f"(kept={kept:,}, missing_rows={missing_rows:,}, skipped={skipped:,})")
    print(f"      deduped: removed {removed:,} duplicate jobs; final={len(df):,}")
    if per_scenario_counts:
        print("   per-scenario:")
        for name in sorted(per_scenario_counts.keys()):
            c = per_scenario_counts[name]
            print(f"     - {name}: jobs={c['jobs']:,} kept={c['kept']:,} "
                  f"missing={c['missing']:,} skipped={c['skipped']:,}")

    # quick peek
    if not df.empty:
        cols = ["ctx","topology","ip_actor_flop","oop_actor_flop","effective_stack_bb","pot_bb","bet_sizing_id","range_source"]
        print(df[cols].head(12).to_string(index=False))

    return df


def _parse_stake_arg(s: str) -> Stakes:
    s_norm = (s or "NL10").strip().upper()
    try:
        return Stakes[s_norm]
    except KeyError:
        raise ValueError(f"Unsupported --stake '{s}'. Use one of: {', '.join([e.name for e in Stakes])}")

def _append_stake_suffix(path_str: str, stake: Stakes) -> str:
    p = Path(path_str)
    stem = p.stem
    suf = "".join(p.suffixes) or ".parquet"
    return str(p.with_name(f"{stem}_{stake.name}{suf}"))

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build flop-only manifest for RangeNet Postflop (Monker-first).")
    ap.add_argument("--config", type=str, default="rangenet/postflop",
                    help="model[/variant]/profile in your config loader")
    ap.add_argument("--out", type=str,
                    default="data/artifacts/rangenet_postflop_flop_manifest.parquet")
    ap.add_argument("--scenario", type=str, default=None,
                    help="Optional: only build a single scenario name (case-insensitive)")
    ap.add_argument("--stake", type=str, default="NL10",
                    help="Stake level (e.g., NL10, NL25)")
    args = ap.parse_args()

    # parse stake → enum
    stake = _parse_stake_arg(args.stake)

    cfg = load_model_config(model=args.config)

    # optional single-scenario filter
    if args.scenario:
        mb = dict(cfg.get("manifest_build", {}) or {})
        scenarios = [s for s in (mb.get("scenarios") or [])
                     if str(s.get("name","")).upper() == args.scenario.upper()]
        if not scenarios:
            print(f"[warn] no scenario named '{args.scenario}' found; building all.")
        else:
            mb["scenarios"] = scenarios
            cfg = {**cfg, "manifest_build": mb}

    # >>> IMPORTANT: pass stake into your builder <<<
    # Make sure build_manifest signature is: build_manifest(cfg, stake: Stakes)
    df = build_manifest(cfg, stake=stake)

    # append stake suffix to filename
    out_path = _append_stake_suffix(args.out, stake)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # persist + log
    df.to_parquet(out, index=False)
    print(f"✅ wrote FLOP manifest: {out} rows={len(df):,} stake={stake.name}")

if __name__ == "__main__":
    main()