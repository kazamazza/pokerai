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
from ml.features.boards import load_board_clusterer
from ml.inference.postflop_ctx import ALLOWED_PAIRS
from ml.utils.board_mask import make_board_mask_52
from ml.utils.config import load_model_config
from ml.models.vocab_actions import FACING_ACTION_VOCAB as ACTION_TOKENS
from ml.features.hands import hand169_id_from_hand_code
from ml.etl.ev.common import write_outputs


def _default_pot_by_ctx(ctx: str) -> float:
    c = (ctx or "").upper()
    if c == "LIMPED_SINGLE": return 3.0
    if c == "VS_OPEN":       return 7.5
    if c == "VS_3BET":       return 22.5
    if c == "VS_4BET":       return 50.0
    return 7.5

def build_ev_postflop_facing(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Build EV parquet for POSTFLOP FACING (IP responds to a bet).
    - Targets columns: ev_<TOKEN> for each token in FACING_ACTION_VOCAB.
    - Categorical/continuous columns align with your postflop-facing dataset YAML.
    """
    # ---- Config sections ----
    dataset = cfg.get("dataset", {}) or {}
    build    = cfg.get("build", {}) or {}
    compute  = cfg.get("compute", {}) or {}
    stake    = cfg.get("stake", {}) or {}
    # paths handled at end by write_outputs

    # ---- Repro ----
    seed = int(dataset.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    # ---- Knobs (aligned to NL10 menus; accept new & legacy keys) ----
    stacks_bb: List[float]   = [float(s) for s in build.get("stacks_bb", [25, 60, 100, 150])]
    size_fracs: List[float]  = [float(x) for x in (build.get("size_fracs")
                                                   or build.get("faced_size_fracs")
                                                   or [0.33, 0.66])]
    samples_per_board: int   = int(build.get("samples_per_board", build.get("samples_per_combination", 64)))
    ctxs: List[str]          = [str(x).upper() for x in (build.get("contexts") or build.get("ctxs") or
                                                         ["VS_OPEN", "VS_3BET", "VS_4BET", "LIMPED_SINGLE"])]
    pot_overrides: Dict[str, float] = {str(k).upper(): float(v) for k, v in (build.get("pot_by_ctx", {}) or {}).items()}

    # ---- Stake artifacts ----
    stakes_token = str(stake.get("token", "NL10")).upper().replace(" ", "")
    stakes_id    = str(stake.get("stakes_id", "2"))
    ranges_parquet = stake.get("ranges_parquet") or f"data/datasets/rangenet_preflop_from_flop_{stakes_token}.parquet"

    # ---- Engines / artifacts ----
    vrp       = VillainRangeProvider(ranges_parquet)
    clusterer = load_board_clusterer(cfg)  # expects board_clustering in cfg
    evmc      = EVMC(samples=int(compute.get("facing_samples", 5000)), seed=seed)

    # ---- Action vocab (facing) ----
    try:
        from ml.models.vocab_actions import FACING_ACTION_VOCAB as ACTION_TOKENS
    except Exception:
        # fallback: legality set (order becomes sorted)
        from ml.inference.postflop_single.legality import FACING_TOKENS as _TOK_SET
        ACTION_TOKENS = sorted(_TOK_SET)

    # ---- Pairs per ctx: prefer YAML override; fallback to ALLOWED_PAIRS ----
    pairs_cfg = build.get("pairs_ip_oop") or {}

    def pairs_for_ctx(ctx: str) -> List[Tuple[str, str]]:
        if pairs_cfg:
            return [tuple(p) for p in pairs_cfg.get(ctx, [])]
        return sorted(ALLOWED_PAIRS.get(ctx, set()))

    # ---- Action sequence for preflop ranges lookup ----
    def action_seq_for_ctx(ctx: str) -> List[str]:
        c = (ctx or "").upper()
        if c == "VS_OPEN":       return ["RAISE", "CALL", ""]
        if c == "VS_3BET":       return ["RAISE", "3BET", "CALL"]
        if c == "VS_4BET":       return ["RAISE", "3BET", "4BET"]
        if c == "LIMPED_SINGLE": return ["LIMP", "CHECK", ""]
        return ["", "", ""]

    # ---- Progress total ----
    total_iters = sum(len(pairs_for_ctx(ctx)) * len(stacks_bb) * len(size_fracs) * samples_per_board
                      for ctx in ctxs)

    rows: List[Dict[str, Any]] = []
    with tqdm(total=total_iters, desc=f"Building EV (postflop:facing)[{stakes_token}]", unit="row", ncols=100) as pbar:
        for ctx in ctxs:
            pairs = pairs_for_ctx(ctx)
            if not pairs:
                continue

            base_pot = float(pot_overrides.get(ctx, _default_pot_by_ctx(ctx)))
            seq = action_seq_for_ctx(ctx)

            for (ip_seat, oop_seat) in pairs:
                # Model expects role tokens "IP"/"OOP" (seats kept only for audit)
                ip_pos, oop_pos = "IP", "OOP"
                hero_pos_role = "IP"  # facing model predicts IP actions

                for stack in stacks_bb:
                    for frac in size_fracs:
                        faced_bb = float(frac) * float(stack)

                        for _ in range(samples_per_board):
                            # --- Sample hero & flop; avoid collisions
                            hero  = sample_random_hand_excluding(exclude=[])
                            board = sample_random_flop_excluding(exclude=[hero[:2], hero[2:]])
                            hand_id = hand169_id_from_hand_code(hero)
                            if hand_id is None:
                                # extremely rare; skip row but keep progress moving
                                pbar.update(1)
                                continue

                            # --- Features
                            bmask = make_board_mask_52(board)
                            try:
                                cluster_id = int(clusterer.predict(board))
                            except Exception:
                                cluster_id = 0

                            # --- Villain range vec (169-d) from preflop parquet
                            vvec = vrp.get_vector(
                                hero_pos=ip_seat,
                                villain_pos=oop_seat,
                                stack=stack,
                                action_seq=seq,
                            )

                            # --- EVs aligned to ACTION_TOKENS order
                            evs = evmc.compute_ev_vector(
                                ACTION_TOKENS,
                                hero_hand=hero,
                                board=board,
                                stack_bb=stack,
                                pot_bb=base_pot,
                                faced_size_bb=faced_bb,
                                villain_vec=vvec,
                            )

                            # --- Row
                            rows.append({
                                # Categoricals (match dataset.x_cols)
                                "hero_pos": hero_pos_role,    # "IP"
                                "ip_pos": ip_pos,             # "IP"
                                "oop_pos": oop_pos,           # "OOP"
                                "ctx": ctx,
                                "street": 1,
                                "board_cluster_id": int(cluster_id),
                                "stakes_id": stakes_id,
                                "hand_id": int(hand_id),

                                # Continuous (match dataset.cont_cols)
                                "board_mask_52": bmask,
                                "pot_bb": float(base_pot),
                                "stack_bb": float(stack),
                                "size_frac": float(frac),

                                # Targets: one column per action token
                                **{f"ev_{tok}": float(val) for tok, val in zip(ACTION_TOKENS, evs)},

                                # Weight = MC sims
                                "weight": int(evmc.samples),

                                # Audit (not used by dataset)
                                "board": board,
                                "hero_hand": hero,
                                "ip_seat": ip_seat,
                                "oop_seat": oop_seat,
                            })

                            pbar.update(1)

    df = pd.DataFrame(rows)

    # Persist via your standard helper (manifest/parquet from cfg.paths)
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
    ap.add_argument(
        "--config",
        type=str,
        default="ml/config/ev/postflop_facing/base.yaml",
        help="Path to YAML (recommended). Also accepts 'model/variant/profile' triple.",
    )
    args = ap.parse_args()

    # Resolve config
    cfg_arg = args.config.strip()
    if cfg_arg.endswith(".yaml"):
        cfg = load_model_config(path=cfg_arg)
    else:
        # supports: ev/postflop_facing/base
        parts = cfg_arg.split("/")
        if len(parts) == 3:
            model, variant, profile = parts
            cfg = load_model_config(model=model, variant=variant, profile=profile)
        elif len(parts) == 2:
            model, variant = parts
            cfg = load_model_config(model=model, variant=variant, profile="base")
        else:
            cfg = load_model_config(path=cfg_arg)  # last-ditch

    df = build_ev_postflop_facing(cfg)

    # Persist using your standard helper
    write_outputs(
        df,
        cfg,
        manifest_key="paths.manifest_path",
        parquet_key="paths.parquet_path",
    )
    print(f"✅ Wrote {len(df):,} rows to {cfg.get('paths',{}).get('parquet_path','<unknown>')}")


if __name__ == "__main__":
    main()