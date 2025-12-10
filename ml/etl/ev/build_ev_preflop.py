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
    - Uses absolute open sizes in bb via build.faced_open_bbs.
    - Produces unopened rows (facing=False) and BB free-check rows.
    - Anchors EV targets: FOLD=0.0; CHECK=0.0 when free_check==1.
    """
    # ---- Load config ----
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

    # ---- Vocab ----
    action_vocab: List[str] = _resolve_preflop_tokens(cfg)

    # ---- Knobs ----
    # ---- Knobs ----
    # stacks_bb: unique, sorted, >0
    stacks_cfg = build.get("stacks_bb", [25, 40, 60, 80, 100, 150])
    stacks: List[float] = sorted({float(s) for s in stacks_cfg if float(s) > 0.0})
    if not stacks:
        raise ValueError("build.stacks_bb is empty or invalid")

    # absolute open sizes in bb for the facing branch (not fraction of stack)
    faced_open_bbs_cfg = build.get("faced_open_bbs", [2.0, 2.5, 3.0, 3.5])
    faced_open_bbs: List[float] = [float(x) for x in faced_open_bbs_cfg if 0.5 <= float(x) <= 20.0]
    if not faced_open_bbs:
        raise ValueError("build.faced_open_bbs is empty or out-of-range")

    include_unopened: bool = bool(build.get("include_unopened", True))
    include_facing: bool = bool(build.get("include_facing", True))

    pairs: List[Tuple[str, str]] = pairs_from_cfg(cfg)  # [(hero_pos, villain_pos), ...]
    # must include hero=BB to generate free-check rows
    if include_unopened and not any((hp or "").upper() == "BB" for hp, _ in pairs):
        raise ValueError("preflop: include_unopened=True but no hero=BB pair found in build.pairs")

    pre_pot: float = float(build.get("pre_pot_bb", 1.5))
    stakes_id: str = stakes_id_from_cfg(cfg, default="2")
    stakes_token: str = str(stake.get("token", "NL10")).upper().replace(" ", "")
    samples: int = max(1, int(build.get("samples_per_combination", 64)))

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
    per_state = (int(include_unopened) + (len(faced_open_bbs) if include_facing else 0))
    total = len(pairs) * len(stacks) * per_state * samples
    rows: List[Dict[str, Any]] = []

    with tqdm(total=total, desc=f"Building EV (preflop)[{stakes_token}]", unit="row", ncols=100) as pbar:
        for hp, vp in pairs:
            hp = (hp or "").upper()
            vp = (vp or "").upper()

            for stack in stacks:
                # --- UNOPENED branch ---
                if include_unopened:
                    facing = False
                    free_check = (hp == "BB")
                    faced_bb = 0.0
                    faced_frac = 0.0

                    for _ in range(samples):
                        hero = sample_random_hand_excluding(exclude=[])
                        hand_id = hand169_id_from_hand_code(hero)
                        if hand_id is None:
                            pbar.update(1)
                            continue

                        toks = gen.generate(
                            stack_bb=stack,
                            facing_bet=facing,
                            faced_size_bb=None,
                            faced_frac=faced_frac,
                            free_check=free_check,
                        )
                        # compute EVs for produced tokens (preflop; no board)
                        evs = evmc.compute_ev_vector(
                            toks,
                            hero_hand=hero,
                            board="",
                            stack_bb=stack,
                            pot_bb=pre_pot,
                            faced_size_bb=faced_bb,
                            villain_vec=None,
                        )
                        ev_map = {t: float(e) for t, e in zip(toks, evs)}
                        # Anchor passives
                        ev_map["FOLD"] = 0.0
                        if free_check:
                            ev_map["CHECK"] = 0.0

                        rows.append({
                            "stakes_id": stakes_id,
                            "street": 0,
                            "hero_pos": hp,
                            "villain_pos": vp,
                            "stack_bb": float(stack),
                            "pot_bb": float(pre_pot),
                            "faced_frac": float(faced_frac),   # kept for compatibility
                            "facing_flag": int(facing),
                            "free_check": int(free_check),
                            "hand_id": int(hand_id),
                            "hero_hand": hero,
                            **{f"ev_{t}": ev_map.get(t, 0.0) for t in action_vocab},
                            "weight": int(evmc.samples),
                        })
                        pbar.update(1)

                # --- FACING branch (absolute open sizes) ---
                if include_facing:
                    for open_bb in faced_open_bbs:
                        facing = True
                        free_check = False
                        faced_bb = float(open_bb)
                        # keep faced_frac as a small scalar feature: open_bb / stack
                        faced_frac = float(faced_bb) / float(stack if stack > 0 else 1.0)

                        for _ in range(samples):
                            hero = sample_random_hand_excluding(exclude=[])
                            hand_id = hand169_id_from_hand_code(hero)
                            if hand_id is None:
                                pbar.update(1)
                                continue

                            toks = gen.generate(
                                stack_bb=stack,
                                facing_bet=facing,
                                faced_size_bb=faced_bb,  # important for raise caps
                                faced_frac=None,         # not used when faced_size_bb is provided
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
                            ev_map = {t: float(e) for t, e in zip(toks, evs)}
                            # Anchor fold (no chips invested by hero preflop before facing)
                            ev_map["FOLD"] = 0.0

                            rows.append({
                                "stakes_id": stakes_id,
                                "street": 0,
                                "hero_pos": hp,
                                "villain_pos": vp,
                                "stack_bb": float(stack),
                                "pot_bb": float(pre_pot),
                                "faced_frac": float(faced_frac),
                                "facing_flag": int(facing),
                                "free_check": int(free_check),
                                "hand_id": int(hand_id),
                                "hero_hand": hero,
                                **{f"ev_{t}": ev_map.get(t, 0.0) for t in action_vocab},
                                "weight": int(evmc.samples),
                            })
                            pbar.update(1)

    df = pd.DataFrame(rows)

    # ---- Safety asserts: must have both categories & BB present ----
    try:
        vc_face = df["facing_flag"].value_counts().to_dict()
        vc_free = df["free_check"].value_counts().to_dict()
        vc_hero = df["hero_pos"].value_counts().to_dict()
        assert 0 in vc_face and 1 in vc_face, f"facing_flag coverage missing: {vc_face}"
        assert 0 in vc_free and 1 in vc_free, f"free_check coverage missing: {vc_free}"
        assert "BB" in vc_hero and vc_hero["BB"] > 0, f"hero_pos 'BB' missing: {vc_hero}"
    except Exception as e:
        # Fail fast to avoid training a biased head
        raise RuntimeError(f"Preflop EV parquet failed coverage checks: {e}")

    # Persist
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