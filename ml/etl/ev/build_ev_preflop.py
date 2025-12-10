from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _import_symbol(path: str):
    """Import 'pkg.mod:SYM' → object."""
    mod, sym = path.split(":")
    return getattr(importlib.import_module(mod), sym)

def _resolve_preflop_tokens(cfg: Dict[str, Any]) -> List[str]:
    """Resolve preflop action vocab (prefer explicit import path)."""
    m = cfg.get("model", {}) or {}
    if m.get("action_vocab_import"):
        toks = list(_import_symbol(m["action_vocab_import"]))
        if not toks:
            raise ValueError("model.action_vocab_import resolved to empty list")
        return toks
    if m.get("action_vocab"):
        toks = [str(x) for x in m["action_vocab"]]
        if not toks:
            raise ValueError("model.action_vocab is empty")
        return toks
    # Fallback to canonical module
    try:
        from ml.models.vocab_actions import PREFLOP_ACTION_VOCAB
        return list(PREFLOP_ACTION_VOCAB)
    except Exception as e:
        raise ValueError("Cannot resolve preflop action vocab; set model.action_vocab_import or model.action_vocab") from e


def build_ev_preflop(cfg_path: Optional[str] = None) -> pd.DataFrame:
    """
    Build EV parquet for PREFLOP.
    - Targets: ev_<TOKEN> for each token in preflop action vocab.
    - Writes only the configured dataset parquet via write_outputs(..., parquet_key="paths.parquet_path").
    """
    # ---- Load config (path or shorthand already resolved by caller) ----
    cfg = (
        load_model_config(path=cfg_path)
        if (cfg_path and cfg_path.endswith(".yaml"))
        else load_model_config(model="ev", variant="preflop", profile="base")
    )

    dataset, build, compute, stake = (cfg.get(k, {}) or {} for k in ("dataset", "build", "compute", "stake"))

    # ---- Repro ----
    seed = int(dataset.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    # ---- Vocab: keep trainer/builder aligned ----
    action_vocab: List[str] = _resolve_preflop_tokens(cfg)

    # ---- Knobs ----
    stacks: List[float]   = [float(s) for s in build.get("stacks_bb", [25, 60, 100, 150])]
    faced_fs: List[float] = [float(x) for x in build.get("faced_fracs", [0.0, 0.25, 0.33, 0.5])]
    pairs: List[Tuple[str, str]] = pairs_from_cfg(cfg)  # [(hero_pos, villain_pos), ...]
    pre_pot: float        = float(build.get("pre_pot_bb", 1.5))
    stakes_id: str        = stakes_id_from_cfg(cfg, default="2")
    stakes_token: str     = str(stake.get("token", "NL10")).upper().replace(" ", "")
    samples: int          = int(build.get("samples_per_combination", 64))

    # ---- Engines ----
    evmc = EVMC(samples=int(compute.get("preflop_samples", 20000)), seed=seed)
    pg_cfg = (cfg.get("preflop_generator") or {})
    gen = PreflopLegalActionGenerator(
        open_sizes_cbb=tuple(int(x) for x in (pg_cfg.get("open_sizes_cbb") or (200, 250, 300))),
        raise_totals_cbb=tuple(int(x) for x in (pg_cfg.get("raise_totals_cbb") or (600, 750, 900, 1200))),
        allow_allin=bool(pg_cfg.get("allow_allin", False)),
        max_open_cbb=(int(pg_cfg["max_open_cbb"]) if pg_cfg.get("max_open_cbb") is not None else None),
    )

    # ---- Progress ----
    total = len(pairs) * len(stacks) * len(faced_fs) * samples
    rows: List[Dict[str, Any]] = []

    with tqdm(total=total, desc=f"Building EV (preflop)[{stakes_token}]", unit="row", ncols=100) as pbar:
        for hp, vp in pairs:
            hp = (hp or "").upper()
            vp = (vp or "").upper()

            for stack in stacks:
                for frac in faced_fs:
                    facing = frac > 0.0
                    free_check = (not facing) and (hp == "BB")
                    faced_bb = float(frac) * float(stack)

                    for _ in range(samples):
                        # --- Sample hero
                        hero = sample_random_hand_excluding(exclude=[])
                        hand_id = hand169_id_from_hand_code(hero)
                        if hand_id is None:
                            pbar.update(1)
                            continue

                        # --- Legal tokens for this state
                        toks = gen.generate(
                            stack_bb=stack,
                            facing_bet=facing,
                            faced_size_bb=(faced_bb if facing else None),
                            free_check=free_check,
                        )

                        # --- EVs for produced tokens
                        evs = evmc.compute_ev_vector(
                            toks,
                            hero_hand=hero,
                            board="",            # preflop
                            stack_bb=stack,
                            pot_bb=pre_pot,
                            faced_size_bb=faced_bb,
                            villain_vec=None,    # optional later
                        )

                        # Map only produced tokens, then project onto full vocab with 0.0 fill
                        ev_map = {t: float(e) for t, e in zip(toks, evs)}

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
                            "hand_id": int(hand_id),
                            "hero_hand": hero,

                            # Targets in declared action_vocab order
                            **{f"ev_{t}": ev_map.get(t, 0.0) for t in action_vocab},

                            # training weight = MC sims used
                            "weight": int(evmc.samples),
                        })
                        pbar.update(1)

    df = pd.DataFrame(rows)

    # Persist (dataset parquet only; manifest optional elsewhere)
    write_outputs(df, cfg, parquet_key="paths.parquet_path")
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="ml/config/evnet/preflop/base.yaml",   # path by default
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
        parquet_key="paths.parquet_path",
    )

if __name__ == "__main__":
    main()