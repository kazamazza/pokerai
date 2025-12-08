# file: ml/etl/ev/build_ev_preflop.py
from __future__ import annotations
import argparse
from typing import Dict, Iterable, List, Tuple
import pandas as pd

from ml.etl.ev.schema_ev import SCHEMA_PREFLOP
from ml.etl.ev.utils_ev import (
    EVSimCfg,
    PreflopLegality,
    VillainRangeProvider,
    RangeProviderCfg,
    infer_action_sequence,
    build_preflop_legal_mask,
    mc_ev_preflop_row,
    OPEN_SIZE_BY_POS_NL10,
    stake_to_stakes_id_token,
)
from ml.models.vocab_actions import PREFLOP_ACTION_VOCAB

ALLOWED_VS_OPEN_PAIRS: List[Tuple[str, str]] = [
    ("UTG", "SB"), ("UTG", "BB"),
    ("HJ", "SB"),  ("HJ", "BB"),
    ("CO", "SB"),  ("CO", "BB"),
    ("BTN", "SB"), ("BTN", "BB"),
]

def _rows_for_open_mode(
    *,
    stake: str,
    stacks: Iterable[float],
    pairs: Iterable[Tuple[str, str]],
    ranges: VillainRangeProvider,
    sim_cfg: EVSimCfg,
    rules: PreflopLegality,
    pot_bb: float = 1.5,
) -> List[Dict]:
    out: List[Dict] = []
    ctx = "VS_OPEN"
    action_seq = infer_action_sequence(ctx)
    for stack in stacks:
        for hero_pos, vill_pos in pairs:
            faced_bb = 0.0
            vec = ranges.get(hero_pos, vill_pos, stack, action_seq)
            if vec is None:
                continue
            mask = build_preflop_legal_mask(
                facing_bet=False, faced_bb=faced_bb, stack_bb=stack, rules=rules
            )
            ev = mc_ev_preflop_row(
                hero_hand=None,
                pot_bb=pot_bb,
                stack_bb=stack,
                faced_bb=faced_bb,
                facing_bet=False,
                vocab=PREFLOP_ACTION_VOCAB,
                mask=mask,
                villain_vec_169=vec,
                sim=sim_cfg,
            )
            row = {
                "stakes_id": stake_to_stakes_id_token(stake),
                "hero_pos_raw": hero_pos,
                "villain_pos_raw": vill_pos,
                "street": 0,
                "pot_bb": pot_bb,
                "eff_stack_bb": float(stack),
                "faced_size_bb": 0.0,
                "ctx": ctx,
                "action_seq_1": action_seq[0], "action_seq_2": action_seq[1], "action_seq_3": action_seq[2],
                "facing_bet": False,
                "hero_hand": "",
                "rowsrc": "open_mode",
            }
            for i, t in enumerate(PREFLOP_ACTION_VOCAB):
                row[f"ev_{t}"] = float(ev[i])
                row[f"legal_{t}"] = float(mask[i])
            out.append(row)
    return out

def _rows_for_facing_mode(
    *,
    stake: str,
    stacks: Iterable[float],
    pairs: Iterable[Tuple[str, str]],
    ranges: VillainRangeProvider,
    sim_cfg: EVSimCfg,
    rules: PreflopLegality,
    pot_bb: float = 1.5,
) -> List[Dict]:
    out: List[Dict] = []
    ctx = "VS_OPEN"
    action_seq = infer_action_sequence(ctx)
    for stack in stacks:
        for hero_pos, vill_pos in pairs:
            opened_bb = OPEN_SIZE_BY_POS_NL10.get(vill_pos, 2.5)
            faced_bb = float(opened_bb)
            vec = ranges.get(hero_pos, vill_pos, stack, action_seq)
            if vec is None:
                continue
            mask = build_preflop_legal_mask(
                facing_bet=True, faced_bb=faced_bb, stack_bb=stack, rules=rules
            )
            ev = mc_ev_preflop_row(
                hero_hand=None,
                pot_bb=pot_bb,
                stack_bb=stack,
                faced_bb=faced_bb,
                facing_bet=True,
                vocab=PREFLOP_ACTION_VOCAB,
                mask=mask,
                villain_vec_169=vec,
                sim=sim_cfg,
            )
            row = {
                "stakes_id": stake_to_stakes_id_token(stake),
                "hero_pos_raw": hero_pos,
                "villain_pos_raw": vill_pos,
                "street": 0,
                "pot_bb": pot_bb + faced_bb,  # opener money in pot already
                "eff_stack_bb": float(stack),
                "faced_size_bb": faced_bb,
                "ctx": ctx,
                "action_seq_1": action_seq[0], "action_seq_2": action_seq[1], "action_seq_3": action_seq[2],
                "facing_bet": True,
                "hero_hand": "",
                "rowsrc": "facing_mode",
            }
            for i, t in enumerate(PREFLOP_ACTION_VOCAB):
                row[f"ev_{t}"] = float(ev[i])
                row[f"legal_{t}"] = float(mask[i])
            out.append(row)
    return out

def main():
    ap = argparse.ArgumentParser(description="Build EV labels for Preflop head (parquet).")
    ap.add_argument("--stake", type=str, default="NL10")
    ap.add_argument("--ranges", type=str, required=True, help="path to rangenet_preflop_from_flop_<STAKE>.parquet")
    ap.add_argument("--out", type=str, required=True, help="output parquet path, e.g., data/datasets/ev_preflop.parquet")
    ap.add_argument("--stacks", type=float, nargs="+", default=[25, 60, 100, 150])
    ap.add_argument("--pairs", type=str, nargs="+",
                    default=[f"{a}:{b}" for (a,b) in ALLOWED_VS_OPEN_PAIRS],
                    help="hero:villain pairs like CO:BB BTN:SB ...")
    ap.add_argument("--samples", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no_open", action="store_true")
    ap.add_argument("--no_facing", action="store_true")
    args = ap.parse_args()

    pairs: List[Tuple[str, str]] = []
    for s in args.pairs:
        try:
            a, b = s.split(":")
            pairs.append((a.upper(), b.upper()))
        except Exception:
            continue

    rp = VillainRangeProvider(RangeProviderCfg(ranges_parquet=args.ranges))
    sim = EVSimCfg(num_samples=int(args.samples), seed=int(args.seed))
    rules = PreflopLegality()

    rows: List[Dict] = []
    if not args.no_open:
        rows += _rows_for_open_mode(
            stake=args.stake, stacks=args.stacks, pairs=pairs,
            ranges=rp, sim_cfg=sim, rules=rules
        )
    if not args.no_facing:
        rows += _rows_for_facing_mode(
            stake=args.stake, stacks=args.stacks, pairs=pairs,
            ranges=rp, sim_cfg=sim, rules=rules
        )

    if not rows:
        raise SystemExit("No rows produced. Check --pairs/--stacks and ranges parquet.")

    df = pd.DataFrame(rows, columns=SCHEMA_PREFLOP.all_cols)

    # Basic validations
    assert set(SCHEMA_PREFLOP.label_cols).issubset(df.columns), "label cols missing"
    assert set(SCHEMA_PREFLOP.mask_cols).issubset(df.columns), "mask cols missing"
    assert (df[SCHEMA_PREFLOP.mask_cols].values.max() <= 1.0) and (df[SCHEMA_PREFLOP.mask_cols].values.min() >= 0.0)

    # Write parquet
    df.to_parquet(args.out, index=False)
    print(f"✅ wrote {len(df):,} rows → {args.out}")

if __name__ == "__main__":
    main()