from __future__ import annotations

import importlib
import os
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


def build_ev_preflop(
    cfg_path: Optional[str] = None,
    *,
    shards: int = 1,
    shard_idx: int = 0,
) -> pd.DataFrame:
    """
    Build EV parquet for PREFLOP (units: bb).

    Emits BOTH branches:
      • unopened (OPEN_*) — CHECK allowed only when hero=BB (free_check==1)
      • facing-open (CALL / RAISE_*), where faced_size_bb = cost-to-call (open - posted)

    Anchors: FOLD=0.0; CHECK=0.0 when free_check==1.

    Sharding: pass shards>1 and shard_idx∈[0..shards-1] to split the cartesian task list
              evenly (round-robin) across processes. Each shard writes its own parquet file:
              <stem>.shard{idx}of{shards}.parquet
    """
    # ---- Load cfg ----
    cfg = (
        load_model_config(path=cfg_path)
        if (cfg_path and cfg_path.endswith(".yaml"))
        else load_model_config(model="ev", variant="preflop", profile="base")
    )
    dataset, build, compute, stake = (cfg.get(k, {}) or {} for k in ("dataset", "build", "compute", "stake"))

    # ---- Repro (offset per-shard, deterministic but distinct) ----
    base_seed = int(dataset.get("seed", 42))
    seed = base_seed + int(shard_idx) * 1000
    random.seed(seed)
    np.random.seed(seed)

    # ---- Vocab ----
    action_vocab: List[str] = _resolve_preflop_tokens(cfg)

    # ---- Knobs (validated) ----
    stacks_cfg = build.get("stacks_bb", [25, 40, 60, 80, 100, 150])
    stacks: List[float] = sorted({float(s) for s in stacks_cfg if float(s) > 0.0})
    if not stacks:
        raise ValueError("build.stacks_bb is empty or invalid")

    faced_open_bbs_cfg = build.get("faced_open_bbs", [2.0, 2.5, 3.0, 3.5])  # absolute bb
    faced_open_bbs: List[float] = [float(x) for x in faced_open_bbs_cfg if 0.5 <= float(x) <= 20.0]
    if not faced_open_bbs:
        raise ValueError("build.faced_open_bbs is empty or out-of-range")

    include_unopened: bool = bool(build.get("include_unopened", True))
    include_facing:   bool = bool(build.get("include_facing",   True))

    # pairs: list of [hero_pos, villain_pos]; must include hero=BB for free-check coverage
    pairs_cfg = build.get("pairs") or []
    pairs: List[Tuple[str, str]] = [(str(a).upper(), str(b).upper()) for a, b in pairs_cfg]
    if include_unopened and not any(h == "BB" for h, _ in pairs):
        raise ValueError("preflop: include_unopened=True but no hero=BB pair found in build.pairs")

    pre_pot: float = float(build.get("pre_pot_bb", 1.5))
    stakes_id: str = (stake.get("stakes_id") or "2")
    stakes_token: str = str(stake.get("token", "NL10")).upper().replace(" ", "")
    samples: int = max(1, int(build.get("samples_per_combination", 512)))

    # ---- Engines ----
    evmc = EVMC(samples=int(compute.get("preflop_samples", 5000)), seed=seed)
    pg_cfg = (cfg.get("preflop_generator") or {})
    gen = PreflopLegalActionGenerator(
        open_sizes_cbb=tuple(int(x) for x in (pg_cfg.get("open_sizes_cbb") or (200, 250, 300))),
        raise_totals_cbb=tuple(int(x) for x in (pg_cfg.get("raise_totals_cbb") or (600, 750, 900, 1200))),
        allow_allin=bool(pg_cfg.get("allow_allin", True)),
        max_open_cbb=(int(pg_cfg["max_open_cbb"]) if pg_cfg.get("max_open_cbb") is not None else None),
    )

    # ---- Task list + sharding ----
    Task = Tuple[str, str, str, float, Optional[float]]  # (kind, hero_pos, villain_pos, stack, open_bb_or_None)
    all_tasks: List[Task] = []
    for hp, vp in pairs:
        for stack in stacks:
            if include_unopened:
                all_tasks.append(("unopened", hp, vp, float(stack), None))
            if include_facing:
                for open_bb in faced_open_bbs:
                    all_tasks.append(("facing", hp, vp, float(stack), float(open_bb)))

    shards = max(1, int(shards))
    shard_idx = int(shard_idx)
    if not (0 <= shard_idx < shards):
        raise ValueError(f"shard_idx {shard_idx} out of range [0..{shards-1}]")

    tasks: List[Task] = [t for i, t in enumerate(all_tasks) if i % shards == shard_idx]

    # ---- Build rows ----
    def _posted_by_hero(pos: str) -> float:
        pos = (pos or "").upper()
        return 1.0 if pos == "BB" else (0.5 if pos == "SB" else 0.0)

    rows: List[Dict[str, Any]] = []
    total = len(tasks) * samples
    desc = f"Building EV (preflop)[{stakes_token}] shard {shard_idx+1}/{shards}" if shards > 1 else f"Building EV (preflop)[{stakes_token}]"
    with tqdm(total=total, desc=desc, unit="row", ncols=100) as pbar:
        for kind, hp, vp, stack, open_bb in tasks:
            posted = _posted_by_hero(hp)

            if kind == "unopened":
                facing = 0
                free_check = 1 if hp == "BB" else 0
                faced_bb = 0.0
                faced_frac = 0.0

                for _ in range(samples):
                    hero = sample_random_hand_excluding(exclude=[])
                    hand_id = hand169_id_from_hand_code(hero)
                    if hand_id is None:
                        pbar.update(1); continue

                    toks = [t for t in gen.generate(
                        stack_bb=float(stack),
                        facing_bet=False,
                        faced_size_bb=None,
                        faced_frac=faced_frac,
                        free_check=bool(free_check),
                    ) if t in action_vocab]
                    if not toks:
                        pbar.update(1); continue

                    evs = evmc.compute_ev_vector(
                        toks,
                        hero_hand=hero, board="",
                        stack_bb=float(stack), pot_bb=float(pre_pot),
                        faced_size_bb=faced_bb,
                        villain_vec=None,
                        hero_pos=hp,  # account for posted blinds internally if needed
                    )
                    ev_map = {t: float(e) for t, e in zip(toks, evs)}
                    ev_map["FOLD"] = 0.0
                    if free_check and "CHECK" in action_vocab:
                        ev_map["CHECK"] = 0.0

                    rows.append({
                        "stakes_id": str(stakes_id),
                        "street": 0,
                        "hero_pos": hp,
                        "villain_pos": vp,
                        "stack_bb": float(stack),
                        "pot_bb": float(pre_pot),
                        "faced_frac": 0.0,
                        "facing_flag": facing,
                        "free_check": free_check,
                        "hand_id": int(hand_id),
                        "hero_hand": hero,
                        **{f"ev_{t}": ev_map.get(t, 0.0) for t in action_vocab},
                        "weight": int(evmc.samples),
                    })
                    pbar.update(1)

            else:  # facing
                assert open_bb is not None
                facing = 1
                free_check = 0
                call_cost = max(0.0, float(open_bb) - posted)   # cost-to-call
                faced_bb = call_cost
                faced_frac = round(faced_bb / float(stack if stack > 0 else 1.0), 6)

                for _ in range(samples):
                    hero = sample_random_hand_excluding(exclude=[])
                    hand_id = hand169_id_from_hand_code(hero)
                    if hand_id is None:
                        pbar.update(1); continue

                    toks = [t for t in gen.generate(
                        stack_bb=float(stack),
                        facing_bet=True,
                        faced_size_bb=faced_bb,  # legal caps use cost-to-call
                        faced_frac=None,
                        free_check=False,
                    ) if t in action_vocab]
                    if not toks:
                        pbar.update(1); continue

                    evs = evmc.compute_ev_vector(
                        toks,
                        hero_hand=hero, board="",
                        stack_bb=float(stack), pot_bb=float(pre_pot),
                        faced_size_bb=faced_bb,
                        villain_vec=None,
                        hero_pos=hp,
                    )
                    ev_map = {t: float(e) for t, e in zip(toks, evs)}
                    ev_map["FOLD"] = 0.0

                    rows.append({
                        "stakes_id": str(stakes_id),
                        "street": 0,
                        "hero_pos": hp,
                        "villain_pos": vp,
                        "stack_bb": float(stack),
                        "pot_bb": float(pre_pot),
                        "faced_frac": float(faced_frac),
                        "facing_flag": facing,
                        "free_check": free_check,
                        "hand_id": int(hand_id),
                        "hero_hand": hero,
                        **{f"ev_{t}": ev_map.get(t, 0.0) for t in action_vocab},
                        "weight": int(evmc.samples),
                    })
                    pbar.update(1)

    df = pd.DataFrame(rows)

    # ---- Coverage checks ----
    vc_face = df["facing_flag"].value_counts().to_dict()
    vc_free = df["free_check"].value_counts().to_dict()
    vc_hero = df["hero_pos"].value_counts().to_dict()
    if not (0 in vc_face and 1 in vc_face):
        raise RuntimeError(f"facing_flag coverage: {vc_face}")
    if not (0 in vc_free and 1 in vc_free):
        raise RuntimeError(f"free_check coverage: {vc_free}")
    if "BB" not in vc_hero or vc_hero["BB"] <= 0:
        raise RuntimeError(f"hero_pos 'BB' missing: {vc_hero}")

    # ---- Write outputs ----
    out = (cfg.get("paths", {}) or {}).get("parquet_path") or "data/datasets/evnet_preflop.parquet"
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if shards > 1:
        shard_out = outp.with_name(f"{outp.stem}.shard{shard_idx}of{shards}{outp.suffix}")
        df.to_parquet(shard_out, index=False)
        print(f"💾 wrote preflop shard → {shard_out}")
    else:
        write_outputs(df, cfg, parquet_key="paths.parquet_path")  # single file
    return df


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="ml/config/evnet/preflop/base.yaml",
        help='Path to YAML (or shorthand like "ev/preflop/base").',
    )
    ap.add_argument("--shards", type=int, default=int(os.getenv("EV_SHARDS", "1")))
    ap.add_argument("--shard-idx", type=int, default=int(os.getenv("EV_SHARD_IDX", "0")))
    args = ap.parse_args()

    # Support shorthand "ev/preflop/base" → repo path
    cfg_path = args.config.strip()
    if not cfg_path.endswith(".yaml"):
        cfg_path = f"ml/config/{cfg_path}.yaml"

    _ = build_ev_preflop(cfg_path, shards=args.shards, shard_idx=args.shard_idx)


if __name__ == "__main__":
    main()