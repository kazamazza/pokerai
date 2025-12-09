# file: ml/etl/ev/build_ev_postflop_facing.py
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.etl.ev.mc import EVMC
from ml.etl.ev.ranges import VillainRangeProvider
from ml.etl.ev.sampling import sample_random_hand_excluding, sample_random_flop_excluding
from ml.etl.ev.utils_ev import infer_action_sequence
from ml.features.boards import load_board_clusterer
from ml.inference.postflop_ctx import ALLOWED_PAIRS
from ml.utils.board_mask import make_board_mask_52
from ml.utils.config import load_model_config
from ml.models.vocab_actions import ROOT_ACTION_VOCAB as ACTION_TOKENS
from ml.features.hands import hand169_id_from_hand_code
from ml.etl.ev.common import pairs_from_cfg, stakes_id_from_cfg, write_outputs

def _default_pot_by_ctx(ctx: str) -> float:
    c = (ctx or "").upper()
    if c == "LIMPED_SINGLE": return 3.0
    if c == "VS_OPEN":       return 7.5
    if c == "VS_3BET":       return 22.5
    if c == "VS_4BET":       return 50.0
    return 7.5


def _action_seq_for_ctx(ctx: str) -> List[str]:
    ctx = (ctx or "").upper()
    if ctx == "VS_OPEN":        return ["RAISE", "CALL", ""]
    if ctx == "VS_3BET":        return ["RAISE", "3BET", "CALL"]
    if ctx == "VS_4BET":        return ["RAISE", "3BET", "4BET"]
    if ctx == "LIMPED_SINGLE":  return ["LIMP", "CHECK", ""]
    return ["", "", ""]


def build_ev_postflop_root(cfg: Dict[str, Any]) -> pd.DataFrame:
    dataset = cfg.get("dataset", {}) or {}
    build    = cfg.get("build", {}) or {}
    compute  = cfg.get("compute", {}) or {}
    stake    = cfg.get("stake", {}) or {}
    paths    = cfg.get("paths", {}) or {}

    # Repro
    seed = int(dataset.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    # Knobs
    stacks_bb: List[float] = [float(s) for s in build.get("stacks_bb", [25, 60, 100, 150])]
    samples_per_board: int = int(build.get("samples_per_board", 8))
    ctxs: List[str] = [str(x).upper() for x in build.get("contexts", ["VS_OPEN", "VS_3BET", "VS_4BET", "LIMPED_SINGLE"])]
    pot_overrides: Dict[str, float] = {k.upper(): float(v) for k, v in (build.get("pot_by_ctx", {}) or {}).items()}

    # Stake artifacts
    stakes_token = str(stake.get("token", "NL10")).upper().replace(" ", "")
    stakes_id    = str(stake.get("stakes_id", "2"))
    ranges_parquet = stake.get("ranges_parquet") or f"data/datasets/rangenet_preflop_from_flop_{stakes_token}.parquet"

    # Engines / artifacts
    vrp       = VillainRangeProvider(ranges_parquet)
    clusterer = load_board_clusterer(cfg)  # needs board_clustering in cfg
    evmc      = EVMC(samples=int(compute.get("root_samples", 5000)), seed=seed)

    # Progress
    total_pairs = sum(len(ALLOWED_PAIRS.get(c, ())) for c in ctxs)
    total = total_pairs * max(1, len(stacks_bb)) * samples_per_board
    pbar = tqdm(total=total, desc=f"Building EV (postflop:root)[{stakes_token}]", unit="row", ncols=100)

    rows: List[Dict[str, Any]] = []

    for ctx in ctxs:
        allowed_pairs = sorted(ALLOWED_PAIRS.get(ctx, set()))
        if not allowed_pairs:
            continue

        action_seq = _action_seq_for_ctx(ctx)
        base_pot   = float(pot_overrides.get(ctx, _default_pot_by_ctx(ctx)))

        for (ip_seat, oop_seat) in allowed_pairs:
            # Root sidecar is OOP → hero_pos="OOP" (actor at root)
            hero_pos_role = "OOP"
            ip_pos, oop_pos = "IP", "OOP"

            for stack in stacks_bb:
                # Root: not facing a bet
                size_frac = 0.0
                faced_bb  = 0.0

                for _ in range(samples_per_board):
                    # Sample hero & board (avoid collisions)
                    hero  = sample_random_hand_excluding(exclude=[])
                    board = sample_random_flop_excluding(exclude=[hero[:2], hero[2:]])
                    hand_id = int(hand169_id_from_hand_code(hero))

                    # Board features
                    bmask = make_board_mask_52(board)
                    try:
                        cluster_id = int(clusterer.predict(board))
                    except Exception:
                        cluster_id = 0

                    # Preflop villain 169-d vector
                    vvec = vrp.get_vector(
                        hero_pos=ip_seat,
                        villain_pos=oop_seat,
                        stack=stack,
                        action_seq=action_seq,
                    )

                    # EVs for ROOT vocab (CHECK/BET_xx/DONK_33)
                    evs = evmc.compute_ev_vector(
                        ACTION_TOKENS,
                        hero_hand=hero,
                        board=board,
                        stack_bb=stack,
                        pot_bb=base_pot,
                        faced_size_bb=faced_bb,
                        villain_vec=vvec,
                    )

                    rows.append({
                        # Categorical inputs (match dataset.x_cols)
                        "hero_pos": hero_pos_role,     # "OOP"
                        "ip_pos": ip_pos,              # "IP"
                        "oop_pos": oop_pos,            # "OOP"
                        "ctx": ctx,
                        "street": 1,
                        "board_cluster_id": int(cluster_id),
                        "stakes_id": stakes_id,
                        "hand_id": hand_id,

                        # Continuous inputs (match dataset.cont_cols)
                        "board_mask_52": bmask,
                        "pot_bb": float(base_pot),
                        "stack_bb": float(stack),
                        "size_frac": float(size_frac),  # 0.0 at root

                        # Targets (one column per action in vocab)
                        **{f"ev_{tok}": float(val) for tok, val in zip(ACTION_TOKENS, evs)},

                        # Weight = MC sims count
                        "weight": int(evmc.samples),

                        # Optional audit fields
                        "board": board,
                        "hero_hand": hero,
                        "ip_seat": ip_seat,
                        "oop_seat": oop_seat,
                    })

                    pbar.update(1)

    pbar.close()
    df = pd.DataFrame(rows)

    # Persist via your helper
    write_outputs(
        df,
        cfg,
        manifest_key="paths.manifest_path",
        parquet_key="paths.parquet_path",
    )
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="ml/config/ev/postflop_root/base.yaml",
                    help="Path to YAML, or 'ev/postflop_root/base' triple.")
    args = ap.parse_args()

    cfg_arg = args.config.strip()
    if cfg_arg.endswith(".yaml"):
        cfg = load_model_config(path=cfg_arg)
    else:
        parts = cfg_arg.split("/")
        if len(parts) == 3:
            model, variant, profile = parts
            cfg = load_model_config(model=model, variant=variant, profile=profile)
        elif len(parts) == 2:
            model, variant = parts
            cfg = load_model_config(model=model, variant=variant, profile="base")
        else:
            cfg = load_model_config(path=cfg_arg)

    df = build_ev_postflop_root(cfg)
    print(f"✅ ev_postflop_root → {cfg.get('paths',{}).get('parquet_path','<unknown>')}  rows={len(df):,}")


if __name__ == "__main__":
    main()