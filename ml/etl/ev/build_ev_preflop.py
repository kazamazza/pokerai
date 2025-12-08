from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.etl.ev.common import pairs_from_cfg, stakes_id_from_cfg, faced_fracs_for_stack, sanitize_ev_col, \
    fill_missing_ev_cols, write_outputs
from ml.etl.ev.mc import EVMC
from ml.inference.preflop_legal_action_generator import PreflopLegalActionGenerator
from ml.utils.config import load_model_config
from ml.etl.ev.sampling import sample_random_hand_excluding
from ml.features.hands import hand169_label_to_id, hand169_id_from_hand_code


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


def build_preflop(cfg: Dict[str, Any]) -> pd.DataFrame:
    dataset = cfg.get("dataset") or {}
    build = cfg.get("build") or {}
    compute = cfg.get("compute") or {}

    # --- seeds ---
    seed = int(dataset.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    # --- knobs ---
    stacks_bb: List[float] = [float(s) for s in build.get("stacks_bb", [25, 60, 100, 150])]
    include_facing: bool = bool(build.get("include_facing", True))
    include_unopened: bool = bool(build.get("include_unopened", True))
    samples_per_combo: int = int(build.get("samples_per_combination", 64))  # ✅ now used
    pre_pot_bb: float = float(build.get("pre_pot_bb", 1.5))
    stakes_id: str = stakes_id_from_cfg(cfg, default="2")
    pairs: List[tuple[str, str]] = pairs_from_cfg(cfg)

    # faced frac menu per stack (re-uses your helper)
    faced_fracs_by_stack: Dict[float, List[float]] = {
        s: faced_fracs_for_stack(cfg, s) for s in stacks_bb
    }

    # --- EV engine ---
    n_sims = int(compute.get("preflop_samples", 20000))
    evmc = EVMC(samples=n_sims, seed=seed)

    # --- finalized preflop generator (reads optional cfg overrides) ---
    pg = (cfg.get("preflop_generator") or {})
    gen = PreflopLegalActionGenerator(
        open_sizes_cbb=tuple(int(x) for x in (pg.get("open_sizes_cbb") or (200, 250, 300))),
        raise_totals_cbb=tuple(int(x) for x in (pg.get("raise_totals_cbb") or (600, 750, 900, 1200))),
        allow_allin=bool(pg.get("allow_allin", False)),
        max_open_cbb=(int(pg["max_open_cbb"]) if pg.get("max_open_cbb") is not None else None),
    )

    # --- progress meter ---
    total_iters = _estimate_total_iters(
        pairs=pairs,
        stacks_bb=stacks_bb,
        faced_fracs_by_stack=faced_fracs_by_stack,
        include_facing=include_facing,
        include_unopened=include_unopened,
        samples_per_combo=samples_per_combo,
    )
    pbar = tqdm(total=total_iters, desc="Building EV (preflop)", unit="row")

    rows: List[Dict[str, Any]] = []

    for (hero_pos, villain_pos) in pairs:
        hp = str(hero_pos or "").upper()
        vp = str(villain_pos or "").upper()

        for stack in stacks_bb:
            for faced_frac in faced_fracs_by_stack.get(stack, []):
                facing_bet = faced_frac > 0.0
                if facing_bet and not include_facing:
                    continue
                if (not facing_bet) and not include_unopened:
                    continue

                free_check = (not facing_bet) and (hp == "BB")
                faced_bb = (float(faced_frac) * float(stack)) if faced_frac else 0.0

                # Repeat per-combination to add stochasticity (hand sampling etc.)
                for _ in range(samples_per_combo):
                    # Sample hero hand (we also store hand_id; dataset can choose to use it or ignore it)
                    hero_hand = sample_random_hand_excluding(exclude=[])
                    hand_id = hand169_id_from_hand_code(hero_hand)

                    # Generate legal tokens
                    toks = gen.generate(
                        stack_bb=stack,
                        facing_bet=facing_bet,
                        faced_size_bb=(faced_bb if faced_bb > 0 else None),
                        free_check=free_check,
                    )

                    # EV vector (uses uniform p(win/tie) in current EVMC placeholder;
                    # can be swapped later to equity-conditioned EVMC)
                    evs = evmc.compute_ev_vector(
                        toks,
                        hero_hand=hero_hand,
                        board="",  # preflop
                        stack_bb=stack,
                        pot_bb=pre_pot_bb,
                        faced_size_bb=faced_bb,
                        villain_vec=None,
                    )

                    rows.append({
                        # identifiers / features
                        "stakes_id": stakes_id,
                        "street": 0,
                        "hero_pos": hp,
                        "villain_pos": vp,
                        "stack_bb": float(stack),
                        "pot_bb": float(pre_pot_bb),
                        "faced_frac": float(faced_frac),
                        "facing_flag": int(facing_bet),
                        "hand_id": int(hand_id),
                        "hero_hand": hero_hand,
                        "board": "",

                        # targets & aux
                        "tokens": toks,     # order of actions used to align EVs
                        "evs": evs,         # EV per token (same order)
                        "weight": n_sims,   # useful for training sampler
                    })

                    pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="ev/preflop/base",
        help="model[/variant]/profile or full path (see load_model_config).",
    )
    args = ap.parse_args()

    parts = args.config.strip().split("/")
    if len(parts) == 3:
        model, variant, profile = parts
        cfg = load_model_config(model=model, variant=variant, profile=profile)
    elif len(parts) == 2:
        model, variant = parts
        cfg = load_model_config(model=model, variant=variant, profile="base")
    else:
        cfg = load_model_config(path=f"ml/config/{args.config}.yaml")

    df = build_preflop(cfg)
    write_outputs(
        df,
        cfg,
        manifest_key="paths.manifest_path",
        parquet_key="paths.parquet_path",
    )


if __name__ == "__main__":
    main()