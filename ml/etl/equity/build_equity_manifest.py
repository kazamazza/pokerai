from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd
import sys


ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.utils.config import load_model_config
from ml.etl.utils.hand import HAND_COUNT
from ml.utils.board_mask import make_board_mask_52
from ml.features.boards import load_board_clusterer, BoardClusterer
from ml.features.boards.representatives import discover_representative_flops

def _street_name_to_id(name: str) -> int:
    name = name.strip().lower()
    if name in ("flop", "f", "1", "street1"):
        return 1
    if name in ("turn", "t", "2", "street2"):
        return 2
    if name in ("river", "r", "3", "street3"):
        return 3
    raise ValueError(f"Unknown street: {name}")

def _parse_streets(arg: str) -> List[int]:
    """
    Accepts comma-separated list like: 'flop,turn,river' or '1,2,3'
    """
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return [_street_name_to_id(p) for p in parts]


def build_preflop_manifest(seed: int | None = None) -> pd.DataFrame:
    """
    One row per hand_id (0..168). Mark as street=0 and add sentinel columns
    so downstream counting & joins work uniformly.
    """
    return pd.DataFrame({
        "street": [0] * HAND_COUNT,
        "board_cluster_id": [-1] * HAND_COUNT,  # sentinel
        "hand_id": list(range(HAND_COUNT)),
        "samples": [1] * HAND_COUNT,            # trivial
        "seed": [int(seed) if seed is not None else -1] * HAND_COUNT,
        "weight": [1.0] * HAND_COUNT,           # (optional) uniform
    })


def build_postflop_manifest(
    *,
    streets: List[int] | List[str],
    board_clusters_limit: int,
    samples_per_cluster: int,
    seed: int | None = None,
    boards_per_cluster: int = 64,
    clusterer: BoardClusterer,                 # pass your Rule/KMeans clusterer
    default_ctx: str = "VS_OPEN",
    default_ip_pos: str = "BTN",
    default_oop_pos: str = "BB",
) -> pd.DataFrame:
    """
    Equity manifest (postflop part):
    emits rows with (street, board_cluster_id, board, board_mask_52, hand_id, samples, weight, seed, ctx, ip_pos, oop_pos).
    """

    if clusterer is None:
        raise ValueError("build_postflop_manifest requires a clusterer to materialize concrete boards")

    def _norm_street(s):
        s = str(s).lower()
        if s in {"1","flop"}: return 1
        if s in {"2","turn"}: return 2
        if s in {"3","river"}: return 3
        return int(s)

    norm_streets = [_norm_street(s) for s in streets]

    boards_by_cluster = discover_representative_flops(
        clusterer=clusterer,
        n_clusters_limit=board_clusters_limit,
        boards_per_cluster=boards_per_cluster,
        seed=(seed or 42),
    )

    rows: list[dict] = []
    for st in norm_streets:
        for bc in range(int(board_clusters_limit)):
            boards = list(boards_by_cluster.get(int(bc), []))
            for board in boards:
                bm = make_board_mask_52(board)
                for hand_id in range(HAND_COUNT):
                    rows.append({
                        "street": int(st),
                        "board_cluster_id": int(bc),
                        "board": board,
                        "board_mask_52": bm,           # 52-d float list
                        "hand_id": int(hand_id),
                        "samples": int(samples_per_cluster),
                        "weight": float(samples_per_cluster),
                        "seed": int(seed) if seed is not None else -1,
                        # lightweight context in case you later condition on villain priors
                        "ctx": str(default_ctx).upper(),
                        "ip_pos": str(default_ip_pos).upper(),
                        "oop_pos": str(default_oop_pos).upper(),
                    })

    cols = ["street","board_cluster_id","board","board_mask_52","hand_id",
            "samples","weight","seed","ctx","ip_pos","oop_pos"]
    return pd.DataFrame(rows, columns=cols)


def _coerce_list(x):
    if x is None: return []
    if isinstance(x, str): return [s.strip() for s in x.split(",") if s.strip()]
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

def _street_key(s):
    # Accept "flop"/"turn"/"river" or 1/2/3
    s = str(s).strip().lower()
    if s in {"1","flop"}: return "flop"
    if s in {"2","turn"}: return "turn"
    if s in {"3","river"}: return "river"
    raise ValueError(f"Unknown street: {s}")

def _plan_from_cfg(cfg, default_streets="flop,turn,river", default_clusters=128, default_samples=64):
    name2street = {"flop": 1, "turn": 2, "river": 3}
    build = cfg.get("build", {})
    dataset = cfg.get("dataset", {})

    streets_cfg = build.get("streets")
    if not streets_cfg:
        streets_cfg = [s.strip() for s in str(default_streets).split(",") if s.strip()]

    clusters_map = build.get("clusters", {})
    samples_map  = build.get("samples_per_cluster", {})
    base_seed    = int(dataset.get("seed", 42))

    plan = []
    for i, st in enumerate(streets_cfg):
        st_key = st if isinstance(st, str) else str(st)
        street_int = name2street.get(st_key.lower(), None)
        if street_int is None:
            # allow numeric streets too
            try:
                street_int = int(st_key)
            except Exception:
                continue

        clusters = int(clusters_map.get(st_key, clusters_map.get(str(street_int), default_clusters)))
        samples  = int(samples_map.get(st_key,  samples_map.get(str(street_int),  default_samples)))
        seed     = base_seed + i  # simple per-street variation
        plan.append({"street": street_int, "clusters": clusters, "samples": samples, "seed": seed})
    return plan

def _get(cfg, path, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def main():
    import argparse
    from pathlib import Path
    import pandas as pd

    ap = argparse.ArgumentParser(
        description="Build unified Equity manifest (preflop + postflop) using config-driven sampling."
    )
    ap.add_argument("--config", type=str, default="equitynet",
                    help="Model name or YAML path, resolved by load_model_config (e.g. equity/base or configs/equity_base.yaml)")
    args = ap.parse_args()

    cfg = load_model_config(args.config)

    # ---- Output path from unified config ----
    out_path = Path(cfg["paths"]["manifest_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []

    # ---- Preflop (always trivial; controlled by build.include.preflop) ----
    include = cfg.get("build", {}).get("include", {})
    if include.get("preflop", True):
        df_pre = build_preflop_manifest()
        frames.append(df_pre)

    # ---- Postflop per-street ----
    if include.get("postflop", True):
        # plan comes from cfg.build.{streets,clusters,samples_per_cluster} (+ dataset.seed)
        plan = _plan_from_cfg(
            cfg,
            default_streets="flop,turn,river",
            default_clusters=128,
            default_samples=64,
        )

        # NEW: shared knobs for materializing concrete boards
        boards_per_cluster = int(_get(cfg, "build.boards_per_cluster", 64))
        clusterer = load_board_clusterer(cfg)

        # Optional light context defaults (carried into rows for future conditioning)
        default_ctx = str(_get(cfg, "build.default_ctx", "VS_OPEN"))
        default_ip_pos = str(_get(cfg, "build.default_ip_pos", "BTN"))
        default_oop_pos = str(_get(cfg, "build.default_oop_pos", "BB"))

        for step in plan:
            street = step["street"]
            clusters = step["clusters"]
            samples = step["samples"]
            seed = step["seed"]

            df_s = build_postflop_manifest(
                streets=[street],
                board_clusters_limit=int(clusters),
                samples_per_cluster=int(samples),
                seed=int(seed),
                boards_per_cluster=boards_per_cluster,  # <-- NEW
                clusterer=clusterer,  # <-- NEW
                default_ctx=default_ctx,  # <-- NEW
                default_ip_pos=default_ip_pos,  # <-- NEW
                default_oop_pos=default_oop_pos,  # <-- NEW
            )
            frames.append(df_s)

    # ---- Concatenate & write ----
    if frames:
        df_all = pd.concat(frames, ignore_index=True)
    else:
        df_all = pd.DataFrame(columns=["street", "board_cluster_id", "hand_id", "samples"])

    df_all.to_parquet(out_path, index=False)

    # ---- Logs ----
    n_rows = len(df_all)
    pre_ct = (df_all["street"] == 0).sum() if n_rows else 0
    post_ct = n_rows - pre_ct
    scen_ct = df_all[["street", "board_cluster_id"]].drop_duplicates().shape[0] if n_rows else 0
    print(f"✅ Wrote unified equity manifest → {out_path}")
    print(f"   rows: {n_rows:,}  (preflop: {pre_ct:,}, postflop: {post_ct:,})  scenarios: {scen_ct:,}")


if __name__ == "__main__":
    main()