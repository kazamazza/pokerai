from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from ml.features.boards.board_clusterers.utils import discover_representative_flops
from ml.range.solvers.keying import solve_sha1, s3_key_for_solve
from ml.range.solvers.utils.preflop_ranges import get_ranges_for_pair
from ml.features.boards import load_board_clusterer
from ml.utils.config import load_model_config


def build_manifest(cfg: dict) -> pd.DataFrame:
    # allow both nested and top-level config
    rpf = cfg.get("rangenet_postflop", {}) or {}

    # 🔧 read manifest_build from nested OR top-level
    mb = (rpf.get("manifest_build")
          or cfg.get("manifest_build")
          or {})

    # 🔧 board_clustering can be nested OR top-level
    bc_cfg = (rpf.get("board_clustering")
              or cfg.get("board_clustering")
              or {})

    # 🔧 solver can be nested OR top-level
    sv = (rpf.get("solver")
          or cfg.get("solver")
          or {})

    # -------- solver knobs --------
    acc   = float(sv.get("accuracy", 0.75))
    iters = int(sv.get("max_iterations", 100))
    a_th  = float(sv.get("allin_threshold", 0.67))
    ver   = str(sv.get("version", "v1"))
    s3_prefix = str(sv.get("s3_prefix", f"solver/outputs/{ver}"))

    # -------- manifest knobs (now pulled from your prod settings) --------
    stacks  = [float(x) for x in mb.get("stacks_bb", [100])]
    pots    = [float(x) for x in mb.get("pots_bb",   [20])]
    # flop-only builder; no `streets` here

    # accept concrete pairs only here (BTN/CO/SB/BB etc)
    position_pairs = [tuple(x) for x in mb.get("position_pairs", [("BTN", "BB")])]
    bet_menu_ids   = [str(x)   for x in mb.get("bet_menus", ["std"])]

    n_clusters_limit   = int(mb.get("board_clusters_limit", 24))
    boards_per_cluster = int(mb.get("boards_per_cluster", 2))
    sample_pool        = int(mb.get("sample_pool", 20000))
    seed               = int(cfg.get("seed", 42))

    # -------- clusterer + representative flops --------
    clusterer = load_board_clusterer(cfg)  # already supports nested/top-level
    boards_by_cluster = discover_representative_flops(
        clusterer=clusterer,
        n_clusters_limit=n_clusters_limit,
        boards_per_cluster=boards_per_cluster,
        seed=seed,
        sample_pool=sample_pool,
    )

    rows: List[Dict[str, Any]] = []
    for stack in stacks:
        for pot in pots:
            for (ip_pos, oop_pos) in position_pairs:
                rng_ip, rng_oop = get_ranges_for_pair(
                    stack_bb=stack, ip=ip_pos, oop=oop_pos, cfg=cfg
                )
                for menu in bet_menu_ids:
                    for cluster_id, boards in boards_by_cluster.items():
                        for b in boards:
                            board_str = "".join(b)
                            params = {
                                "street": 1,  # flop-only
                                "pot_bb": pot,
                                "effective_stack_bb": stack,
                                "board": board_str,
                                "board_cluster_id": int(cluster_id),
                                "range_ip": rng_ip,
                                "range_oop": rng_oop,
                                "positions": f"{ip_pos}v{oop_pos}",
                                "bet_sizing_id": menu,
                                "accuracy": acc,
                                "max_iter": iters,
                                "allin_threshold": a_th,
                                "solver_version": ver,
                            }
                            sha = solve_sha1(params)
                            s3k = s3_key_for_solve(params, sha1=sha, prefix=s3_prefix)
                            rows.append({**params, "sha1": sha, "s3_key": s3k, "node_key": "root", "weight": 1.0})

    df = pd.DataFrame(rows)

    # (optional) quick debug print so this never surprises you again
    print(f"[dbg] stacks={len(stacks)} pots={len(pots)} pairs={len(position_pairs)} "
          f"clusters_used={len(boards_by_cluster)} boards/cluster≈{boards_per_cluster} → jobs={len(df)}")

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
    print(cfg)
    print(cfg)
    df = build_manifest(cfg)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✅ wrote FLOP manifest: {out} rows={len(df):,}")


if __name__ == "__main__":
    main()