from __future__ import annotations
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.etl.ev.common import pairs_from_cfg, stakes_id_from_cfg, write_outputs
from ml.etl.ev.mc import EVMC
from ml.inference.preflop_legal_action_generator import PreflopLegalActionGenerator
from ml.utils.config import load_model_config
from ml.etl.ev.sampling import sample_random_hand_excluding
from ml.features.hands import hand169_id_from_hand_code


def _estimate_total_iters(
    *,
    pairs: List[tuple[str, str]],
    stacks_bb: List[float],
    faced_fracs_by_stack: Dict[float, List[float]],
    include_facing: bool,
    include_unopened: bool,
    samples_per_combo: int,
) -> int:
    total = 0
    for stack in stacks_bb:
        for _hp, _vp in pairs:
            for ff in faced_fracs_by_stack.get(stack, []):
                facing_bet = ff > 0.0
                if facing_bet and not include_facing:
                    continue
                if (not facing_bet) and not include_unopened:
                    continue
                total += samples_per_combo
    return total


def build_ev_preflop(cfg_path: Optional[str] = None) -> pd.DataFrame:
    cfg = (
        load_model_config(path=cfg_path)
        if (cfg_path and cfg_path.endswith(".yaml"))
        else load_model_config(model="ev", variant="postflop_root", profile="base")
    )

    dataset, build, compute, paths = (cfg.get(k, {}) or {} for k in ("dataset","build","compute","paths"))
    seed = int(dataset.get("seed", 42)); random.seed(seed); np.random.seed(seed)

    stacks   = [float(s) for s in build.get("stacks_bb", [25,60,100,150])]
    faced_fs = [float(x) for x in build.get("faced_fracs", [0.0, 0.25, 0.33, 0.5])]
    pairs    = pairs_from_cfg(cfg)                      # [(hero_pos, villain_pos), ...]
    pre_pot  = float(build.get("pre_pot_bb", 1.5))
    stakes_id = stakes_id_from_cfg(cfg, default="2")
    samples  = int(build.get("samples_per_combination", 64))

    # action vocab from cfg (keeps trainer/builder aligned)
    action_vocab = cfg["model"]["action_vocab"]

    # EV engine + generator
    evmc = EVMC(samples=int(compute.get("preflop_samples", 20000)), seed=seed)
    pg = (cfg.get("preflop_generator") or {})
    gen = PreflopLegalActionGenerator(
        open_sizes_cbb=tuple(int(x) for x in (pg.get("open_sizes_cbb") or (200, 250, 300))),
        raise_totals_cbb=tuple(int(x) for x in (pg.get("raise_totals_cbb") or (600, 750, 900, 1200))),
        allow_allin=bool(pg.get("allow_allin", False)),
        max_open_cbb=(int(pg["max_open_cbb"]) if pg.get("max_open_cbb") is not None else None),
    )

    total = len(pairs) * len(stacks) * len(faced_fs) * samples
    pbar = tqdm(total=total, desc="ev_preflop", ncols=100)
    rows = []

    for hp, vp in pairs:
        hp = hp.upper(); vp = vp.upper()
        for stack in stacks:
            for frac in faced_fs:
                facing = frac > 0.0
                free_check = (not facing) and (hp == "BB")
                faced_bb = frac * stack
                for _ in range(samples):
                    hero = sample_random_hand_excluding(exclude=[])
                    hand_id = hand169_id_from_hand_code(hero)

                    toks = gen.generate(
                        stack_bb=stack,
                        facing_bet=facing,
                        faced_size_bb=(faced_bb if facing else None),
                        free_check=free_check,
                    )

                    evs = evmc.compute_ev_vector(
                        toks,
                        hero_hand=hero,
                        board="",
                        stack_bb=stack,
                        pot_bb=pre_pot,
                        faced_size_bb=faced_bb,
                        villain_vec=None,
                    )

                    rows.append({
                        "stakes_id": stakes_id,
                        "street": 0,
                        "hero_pos": hp,
                        "villain_pos": vp,
                        "stack_bb": float(stack),
                        "pot_bb": float(pre_pot),
                        "faced_frac": float(frac),
                        "facing_flag": int(facing),
                        "free_check": int(free_check),
                        "hand_id": int(hand_id) if hand_id is not None else -1,
                        "hero_hand": hero,
                        # targets
                        **{f"ev_{tok}": float(e) for tok, e in zip(action_vocab, evs)},
                        "weight": int(evmc.samples),
                    })
                    pbar.update(1)
    pbar.close()

    df = pd.DataFrame(rows)
    write_outputs(df, cfg, manifest_key="paths.manifest_path", parquet_key="paths.parquet_path")
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="ml/config/ev/preflop/base.yaml",   # path by default
        help="Path to YAML (or shorthand like 'ev/preflop/base').",
    )
    args = ap.parse_args()

    # Allow shorthand "ev/preflop/base" → path
    cfg_path = args.config
    if not cfg_path.endswith(".yaml"):
        cfg_path = f"ml/config/{cfg_path}.yaml"

    # Load cfg once for outputs; pass PATH to builder
    cfg = load_model_config(path=cfg_path)
    df = build_ev_preflop(cfg_path)

    write_outputs(
        df,
        cfg,
        manifest_key="paths.manifest_path",
        parquet_key="paths.parquet_path",
    )

if __name__ == "__main__":
    main()