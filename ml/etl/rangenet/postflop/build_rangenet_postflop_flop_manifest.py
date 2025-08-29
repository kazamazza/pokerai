from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client
from ml.features.boards.board_clusterers.utils import discover_representative_flops
from ml.range.solvers.keying import solve_sha1, s3_key_for_solve
from ml.features.boards import load_board_clusterer
from ml.utils.config import load_model_config
from ml.etl.rangenet.preflop.range_lookup import PreflopRangeLookup


def build_manifest(cfg: dict) -> pd.DataFrame:
    # ----- read config (top-level only, per your setup) -----
    mb  = cfg.get("manifest_build", {}) or {}
    sv  = cfg.get("solver", {}) or {}
    inputs = cfg.get("inputs", {}) or {}

    acc   = float(sv.get("accuracy", 0.75))
    iters = int(sv.get("max_iterations", 100))
    a_th  = float(sv.get("allin_threshold", 0.67))
    ver   = str(sv.get("version", "v1"))
    s3_prefix_outputs = str(sv.get("s3_prefix", f"solver/outputs/{ver}"))

    stacks  = [float(x) for x in mb.get("stacks_bb", [100])]
    pots    = [float(x) for x in mb.get("pots_bb",   [20])]
    position_pairs = [tuple(x) for x in mb.get("position_pairs", [("BTN","BB")])]
    bet_menu_ids   = [str(x)   for x in mb.get("bet_menus", ["std"])]

    n_clusters_limit   = int(mb.get("board_clusters_limit", 24))
    boards_per_cluster = int(mb.get("boards_per_cluster", 2))
    sample_pool        = int(mb.get("sample_pool", 20000))
    seed               = int(cfg.get("seed", 42))

    # where vendor files live in S3 and where to cache locally
    monker_manifest_path = inputs.get("monker_manifest", "data/artifacts/monker_manifest.parquet")
    vendor_s3_prefix     = inputs.get("vendor_s3_prefix", "data/vendor/monker")  # e.g. s3://bucket/<this>/*
    cache_dir            = sv.get("local_cache_dir", "data/vendor_cache")

    # Toggle whether to include rows with missing ranges as placeholders (False = skip)
    include_missing = bool(mb.get("include_missing", False))
    allow_pair_subs = bool(mb.get("allow_pair_subs", False))  # conservative default False

    # ----- ranges lookup (once) -----
    s3c = S3Client()  # reads creds/env automatically
    lookup = PreflopRangeLookup(
        monker_manifest_path,
        s3_client=s3c,
        s3_prefix=vendor_s3_prefix,
        cache_dir=cache_dir,
        allow_pair_subs=allow_pair_subs,
    )

    # ----- clusterer + representative flops -----
    clusterer = load_board_clusterer(cfg)
    boards_by_cluster = discover_representative_flops(
        clusterer=clusterer,
        n_clusters_limit=n_clusters_limit,
        boards_per_cluster=boards_per_cluster,
        seed=seed,
        sample_pool=sample_pool,
    )

    rows: List[Dict[str, Any]] = []
    skipped, kept, missing_rows = 0, 0, 0
    miss_counts: Dict[str, int] = {}  # per pair diagnostics

    for stack in stacks:
        for pot in pots:
            for (ip_pos, oop_pos) in position_pairs:
                print(f"{stack} {pot} {ip_pos}:{oop_pos}")
                rng_ip, rng_oop, meta = lookup.ranges_for_pair(
                    stack_bb=stack, ip=ip_pos, oop=oop_pos, strict=False
                )
                print(rng_ip, rng_oop, meta)
                if rng_ip is None or rng_oop is None:
                    key = f"{ip_pos}v{oop_pos}@{int(stack)}"
                    miss_counts[key] = miss_counts.get(key, 0) + 1
                    if not include_missing:
                        skipped += len(boards_by_cluster) * boards_per_cluster * len(bet_menu_ids)
                        continue  # drop this scenario entirely
                    # else: include placeholder row(s) flagged as missing (rare; mainly for audits)

                for menu in bet_menu_ids:
                    for cluster_id, boards in boards_by_cluster.items():
                        for b in boards:
                            board_str = "".join(b)

                            params = {
                                "street": 1,  # flop-only manifest
                                "pot_bb": pot,
                                "effective_stack_bb": stack,
                                "board": board_str,
                                "board_cluster_id": int(cluster_id),
                                "range_ip": rng_ip if rng_ip is not None else "",
                                "range_oop": rng_oop if rng_oop is not None else "",
                                "positions": f"{ip_pos}v{oop_pos}",
                                "bet_sizing_id": menu,
                                "accuracy": acc,
                                "max_iter": iters,
                                "allin_threshold": a_th,
                                "solver_version": ver,
                                # provenance
                                "range_ip_source_stack": meta.get("range_ip_source_stack"),
                                "range_oop_source_stack": meta.get("range_oop_source_stack"),
                                "range_ip_stack_delta": meta.get("range_ip_stack_delta"),
                                "range_oop_stack_delta": meta.get("range_oop_stack_delta"),
                                "range_ip_fallback_level": meta.get("range_ip_fallback_level"),
                                "range_oop_fallback_level": meta.get("range_oop_fallback_level"),
                                "range_pair_substituted": bool(meta.get("range_pair_substituted", False)),
                                "range_ip_source_pair": meta.get("range_ip_source_pair"),
                                "range_oop_source_pair": meta.get("range_oop_source_pair"),
                                "ranges_missing": bool(rng_ip is None or rng_oop is None),
                            }

                            sha = solve_sha1(params)
                            s3k = s3_key_for_solve(params, sha1=sha, prefix=s3_prefix_outputs)

                            rows.append({
                                **params,
                                "sha1": sha,
                                "s3_key": s3k,
                                "node_key": "root",
                                "weight": 1.0,
                            })
                            kept += 1 if not params["ranges_missing"] else 0
                            missing_rows += 1 if params["ranges_missing"] else 0

    df = pd.DataFrame(rows)

    # compact debug summary
    total_jobs = len(df)
    print(
        f"[dbg] stacks={len(stacks)} pots={len(pots)} pairs={len(position_pairs)} "
        f"clusters_used={len(boards_by_cluster)} boards/cluster≈{boards_per_cluster} "
        f"→ jobs={total_jobs} (kept={kept}, missing_rows={missing_rows}, skipped={skipped})"
    )
    if miss_counts:
        top = list(sorted(miss_counts.items(), key=lambda kv: -kv[1]))[:8]
        print("   Missing scenarios (top):", ", ".join(f"{k}×{v}" for k,v in top))

    return df


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Build flop-only manifest for RangeNet Postflop"
    )
    ap.add_argument("--config", type=str, default="rangenet/postflop",
                    help="model[/variant]/profile")
    ap.add_argument("--out", type=str,
                    default="data/artifacts/rangenet_postflop_flop_manifest.parquet")
    args = ap.parse_args()

    cfg = load_model_config(model=args.config)
    df = build_manifest(cfg)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ wrote FLOP manifest: {out} rows={len(df):,}")


if __name__ == "__main__":
    main()