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

def _default_pot_by_ctx(ctx: str) -> float:
    c = (ctx or "").upper()
    if c == "LIMPED_SINGLE": return 3.0
    if c == "VS_OPEN":       return 7.5
    if c == "VS_3BET":       return 22.5
    if c == "VS_4BET":       return 50.0
    return 7.5


def build_ev_postflop_root(cfg_path: Optional[str] = None) -> pd.DataFrame:
    """
    Build EV parquet for POSTFLOP ROOT (first-to-act, OOP model in your sidecar).
    - Uses ROOT action vocab (CHECK, BET_xx, DONK_33).
    - EVMC must understand BET_xx/DONK_33 as pot-fraction bets.
    """
    cfg = (
        load_model_config(path=cfg_path)
        if (cfg_path and cfg_path.endswith(".yaml"))
        else load_model_config(model="ev", variant="postflop_root", profile="base")
    )

    dataset = cfg.get("dataset", {}) or {}
    build    = cfg.get("build", {}) or {}
    compute  = cfg.get("compute", {}) or {}
    stake    = cfg.get("stake", {}) or {}
    paths    = cfg.get("paths", {}) or {}

    # Repro
    seed = int(dataset.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    # Build knobs
    stacks_bb: List[float]  = [float(s) for s in build.get("stacks_bb", [25, 60, 100, 150])]
    samples_per_combo: int  = int(build.get("samples_per_combination", 64))
    ctxs: List[str]         = [str(x).upper() for x in build.get("ctxs", ["VS_OPEN", "VS_3BET", "VS_4BET", "LIMPED_SINGLE"])]
    pot_overrides: Dict[str, float] = {k.upper(): float(v) for k, v in (build.get("pot_by_ctx", {}) or {}).items()}

    # Stake artifacts
    stakes_token = str(stake.get("token", "NL10")).upper().replace(" ", "")
    stakes_id    = str(stake.get("stakes_id", "2"))
    ranges_parquet = stake.get("ranges_parquet") or f"data/datasets/rangenet_preflop_from_flop_{stakes_token}.parquet"

    # Engines / artifacts
    vrp       = VillainRangeProvider(ranges_parquet)
    clusterer = load_board_clusterer(cfg)
    evmc      = EVMC(samples=int(compute.get("root_samples", 5000)), seed=seed)

    rows: List[Dict[str, float | int | str | None]] = []

    # Progress estimation
    total_pairs = sum(len(ALLOWED_PAIRS.get(c, ())) for c in ctxs)
    total = total_pairs * max(1, len(stacks_bb)) * samples_per_combo
    pbar = tqdm(total=total, desc="ev_postflop_root", ncols=100)

    for ctx in ctxs:
        allowed_pairs = sorted(ALLOWED_PAIRS.get(ctx, set()))
        if not allowed_pairs:
            continue

        action_seq = infer_action_sequence(ctx)
        base_pot   = pot_overrides.get(ctx, _default_pot_by_ctx(ctx))

        for (ip_seat, oop_seat) in allowed_pairs:
            # Root sidecar is OOP in your dump → hero_pos="OOP"
            hero_pos = "OOP"

            for stack in stacks_bb:
                # faced_size for root is 0 (no bet faced); size_frac not needed
                faced_bb = 0.0

                for _ in range(samples_per_combo):
                    # Sample hero & flop with collision guards in helpers
                    hero  = sample_random_hand_excluding(exclude=[])
                    board = sample_random_flop_excluding(exclude=[hero[:2], hero[2:]])

                    # Features
                    bmask = make_board_mask_52(board)
                    try:
                        cluster_id = int(clusterer.predict(board))
                    except Exception:
                        cluster_id = 0

                    # Preflop villain range vec (169-d) keyed by action_seq
                    vvec = vrp.get_vector(
                        hero_pos=ip_seat,
                        villain_pos=oop_seat,
                        stack=stack,
                        action_seq=action_seq,
                    )

                    # EVs aligned to ROOT vocab order
                    evs = evmc.compute_ev_vector(
                        ACTION_TOKENS,
                        hero_hand=hero,
                        board=board,
                        stack_bb=stack,
                        pot_bb=base_pot,
                        faced_size_bb=faced_bb,
                        villain_vec=vvec,
                    )

                    # Row
                    row = {
                        "stakes_id": stakes_id,
                        "street": 1,
                        "ctx": ctx,
                        "ip_pos": "IP",
                        "oop_pos": "OOP",
                        "hero_pos": hero_pos,          # "OOP"
                        "board": board,
                        "board_cluster_id": cluster_id,
                        "pot_bb": float(base_pot),
                        "stack_bb": float(stack),
                        # board mask (bm0..bm51) to keep loader simple
                        **{f"bm{i}": float(bmask[i]) for i in range(52)},
                        # EV targets one column per action token
                        **{f"ev_{tok}": float(e) for tok, e in zip(ACTION_TOKENS, evs)},
                        # weight = number of sims used
                        "weight": int(evmc.samples),
                        # optional for audit/debug
                        "hero_hand": hero,
                        "ip_seat": ip_seat,
                        "oop_seat": oop_seat,
                    }
                    rows.append(row)
                    pbar.update(1)

    pbar.close()
    df = pd.DataFrame(rows)

    out_path = paths.get("parquet_path") or f"data/datasets/ev_postflop_root_{stakes_token}.parquet"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"✅ ev_postflop_root → {out_path}  rows={len(df):,}")

    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="", help="Optional direct YAML path.")
    args = ap.parse_args()
    build_ev_postflop_root(cfg_path=args.config.strip() or None)