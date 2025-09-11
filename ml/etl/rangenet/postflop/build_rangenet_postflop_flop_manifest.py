from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from ml.etl.utils.positions import sanitize_position_pairs
from infra.storage.s3_client import S3Client
from ml.features.boards.board_clusterers.utils import discover_representative_flops
from ml.range.solvers.keying import solve_sha1, s3_key_for_solve
from ml.features.boards import load_board_clusterer
from ml.utils.config import load_model_config
from ml.etl.rangenet.preflop.range_lookup import PreflopRangeLookup


def build_manifest(cfg: dict) -> pd.DataFrame:
    mb  = cfg.get("manifest_build", {}) or {}
    sv  = cfg.get("worker", {}) or {}
    inputs = cfg.get("inputs", {}) or {}

    # Solver knobs
    acc   = float(sv.get("accuracy", 0.75))
    iters = int(sv.get("max_iterations", 100))
    a_th  = float(sv.get("allin_threshold", 0.67))
    ver   = str(sv.get("version", "v1"))
    s3_prefix_outputs = str(sv.get("s3_prefix", f"worker/outputs/{ver}"))

    # Build scope
    ctx    = str(mb.get("ctx", "SRP")).upper()  # <--- make context explicit
    stacks = [float(x) for x in mb.get("stacks_bb", [100])]
    pots   = [float(x) for x in mb.get("pots_bb",   [20])]
    raw_pairs = [tuple(x) for x in mb.get("position_pairs", [("BTN","BB")])]
    bet_menu_ids = [str(x) for x in mb.get("bet_menus", ["std"])]

    n_clusters_limit   = int(mb.get("board_clusters_limit", 24))
    boards_per_cluster = int(mb.get("boards_per_cluster", 2))
    sample_pool        = int(mb.get("sample_pool", 50000))
    seed               = int(cfg.get("seed", 42))

    include_missing = bool(mb.get("include_missing", False))
    allow_pair_subs = bool(mb.get("allow_pair_subs", False))

    # ✅ sanitize pairs for this context
    position_pairs = sanitize_position_pairs(raw_pairs, ctx)
    if not position_pairs:
        raise SystemExit(f"No legal position pairs for ctx={ctx} from {raw_pairs}")

    # ----- ranges lookup -----
    lookup = PreflopRangeLookup(
        monker_manifest_parquet=inputs.get("monker_manifest", "data/artifacts/monker_manifest.parquet"),
        sph_manifest_parquet=inputs.get("sph_manifest", "data/artifacts/sph_manifest.parquet"),
        s3_client=S3Client(),
        s3_vendor=inputs.get("vendor_s3_prefix", "data/vendor"),
        cache_dir=sv.get("local_cache_dir", "data/vendor_cache"),
        allow_pair_subs=allow_pair_subs,
    )

    # ----- flops -----
    clusterer = load_board_clusterer(cfg)
    boards_by_cluster = discover_representative_flops(
        clusterer=clusterer,
        n_clusters_limit=n_clusters_limit,
        boards_per_cluster=boards_per_cluster,
        seed=seed,
        sample_pool=sample_pool,
    )

    rows: list[dict] = []
    skipped = kept = missing_rows = 0
    miss_counts: dict[str, int] = {}

    for stack in stacks:
        for pot in pots:
            for (ip_pos, oop_pos) in position_pairs:
                # pull ranges with correct context
                try:
                    rng_ip, rng_oop, meta = lookup.ranges_for_pair(
                        stack_bb=stack, ip=ip_pos, oop=oop_pos, ctx=ctx, strict=False
                    )
                except Exception as e:
                    rng_ip = rng_oop = None
                    meta = {"error": str(e)}

                if rng_ip is None or rng_oop is None:
                    key = f"{ctx}:{ip_pos}v{oop_pos}@{int(stack)}"
                    miss_counts[key] = miss_counts.get(key, 0) + 1
                    if not include_missing:
                        skipped += len(boards_by_cluster) * boards_per_cluster * len(bet_menu_ids)
                        continue

                for menu in bet_menu_ids:
                    for cluster_id, boards in boards_by_cluster.items():
                        for b in boards:
                            board_str = "".join(b)
                            params = {
                                "street": 1,
                                "pot_bb": pot,
                                "effective_stack_bb": stack,
                                "board": board_str,
                                "board_cluster_id": int(cluster_id),
                                "range_ip": rng_ip or "",
                                "range_oop": rng_oop or "",
                                "positions": f"{ip_pos}v{oop_pos}",
                                "bet_sizing_id": menu,
                                "accuracy": acc,
                                "max_iter": iters,
                                "allin_threshold": a_th,
                                "solver_version": ver,
                                # provenance
                                "ctx": ctx,
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

                            rows.append({**params, "sha1": sha, "s3_key": s3k, "node_key": "root", "weight": 1.0})
                            kept += 1 if not params["ranges_missing"] else 0
                            missing_rows += 1 if params["ranges_missing"] else 0

    df = pd.DataFrame(rows)

    total_jobs = len(df)
    print(
        f"[dbg] ctx={ctx} stacks={len(stacks)} pots={len(pots)} pairs={len(position_pairs)} "
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
    print("config:", cfg)
    df = build_manifest(cfg)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ wrote FLOP manifest: {out} rows={len(df):,}")


if __name__ == "__main__":
    main()