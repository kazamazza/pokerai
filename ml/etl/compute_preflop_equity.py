# ml/etl/compute_preflop_equity.py
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import eval7  # pip install eval7

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.config.types_hands import ALL_HANDS
from ml.features.equity.preflop import equity_preflop_vs_range

# --- Imports you already have in the repo (do not re-declare) ---
# Hand ID mapping (0..168) and parsing helpers
from ml.features.hands import HANDS_169, hand_code_from_id, enumerate_suited_combos
# ^ expected helpers:
# - HANDS_169: List[str] like ["AA","AKs","AQs",...,"72o"]
# - hand_code_from_id(hid:int) -> str like "AKs"
# - enumerate_suited_combos(code:str) -> List[Tuple[eval7.Card, eval7.Card]]
#
# Monker parsing: read a file and return a dict {"AA": 1.0, "AKs": 0.5, ...} for the OPENER
from ml.etl.utils.monker_parser import parse_monker_range_text, load_range_file

# Manifest columns (we rely on your existing manifest builder)
REQ_MANIFEST_COLS = ["stack_bb", "hero_pos", "opener_action", "rel_path", "abs_path"]

def load_manifest(manifest_parquet: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(manifest_parquet)
    missing = [c for c in REQ_MANIFEST_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    return df

def aggregate_villain_range(manifest_rows: pd.DataFrame, root: Path) -> Dict[str, float]:
    """
    Merge all Monker range files in this group into one villain distribution.
    Sum-prob merge then normalize to 1.0 (if any prob mass exists).
    Expects manifest_rows to have 'rel_path' or 'abs_path'.
    """
    agg: Dict[str, float] = {}
    for _, r in manifest_rows.iterrows():
        path = Path(r["abs_path"]) if "abs_path" in r and r["abs_path"] else (root / r["rel_path"])
        d = load_range_file(path)  # <-- correct helper; no actor kwarg
        # sum-merge
        for hand, prob in d.items():
            if prob <= 0:
                continue
            agg[hand] = agg.get(hand, 0.0) + prob

    s = sum(agg.values())
    if s > 0:
        scale = 1.0 / s
        for k in agg:
            agg[k] *= scale
    return agg

def _range_to_weighted_combos(range169: Dict[str, float]) -> List[Tuple[Tuple[eval7.Card, eval7.Card], float]]:
    """
    Expand 169 hand weights to actual two-card combos with per-combo weights,
    distributing the hand weight uniformly over its combos.
    """
    out = []
    for code, w in range169.items():
        combos = enumerate_suited_combos(code)  # list of (Card, Card) with suits expanded
        if not combos:
            continue
        per = w / len(combos)
        for c in combos:
            out.append((c, per))
    return out

def _preflop_equity_vs_range(hero_code: str,
                             villain_combos: List[Tuple[Tuple[eval7.Card, eval7.Card], float]],
                             n_samples: int = 50_000,
                             rng_seed: int | None = 42) -> float:
    """
    Monte Carlo: equity(hero hand) vs weighted villain range.
    """
    rng = np.random.default_rng(rng_seed)
    # expand hero to all combos; we’ll average
    hero_combos = enumerate_suited_combos(hero_code)

    if not hero_combos:
        return 0.5

    # Pre-construct a full deck once
    full_deck = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "cdhs"]

    total_wins = 0.0
    total = 0.0

    # sample hero combos uniformly (each combo same weight)
    for (hc1, hc2) in hero_combos:
        # deck minus hero cards
        deck = [c for c in full_deck if c != hc1 and c != hc2]

        # build villain candidate list excluding collisions per iteration
        # (we’ll resample villain combos each trial)
        for _ in range(n_samples // len(hero_combos)):
            # sample villain combo proportional to weight, with collision check
            # (naive rejection sampling is fine here)
            while True:
                idx = rng.choice(len(villain_combos), p=np.array([w for (_, w) in villain_combos], dtype=float))
                (vc1, vc2), vw = villain_combos[idx]
                if vc1 != hc1 and vc1 != hc2 and vc2 != hc1 and vc2 != hc2:
                    break

            # make a working deck without villain cards
            deck2 = [c for c in deck if c != vc1 and c != vc2]

            # draw 5 board cards
            board = rng.choice(deck2, size=5, replace=False)
            board = list(board)

            hero_hand = [hc1, hc2]
            vill_hand = [vc1, vc2]

            hero_val = eval7.evaluate(hero_hand + board)
            vill_val = eval7.evaluate(vill_hand + board)

            if hero_val > vill_val:
                total_wins += vw
            elif hero_val == vill_val:
                total_wins += 0.5 * vw
            total += vw

    if total <= 1e-12:
        return 0.5
    return float(total_wins / total)

def compute_equity_for_group(df_rows: pd.DataFrame,
                             manifest: pd.DataFrame,
                             root: Path,
                             n_samples: int,
                             seed: int) -> pd.DataFrame:
    """
    df_rows: rows for ONE (stack_bb, hero_pos, opener_action) group from the **hand grid** you want to score
             (e.g., one row per hand_id)
    manifest: the full manifest; we’ll filter it to the same group to build the villain range.
    """
    # Sanity: df_rows must contain grouping keys
    for col in ("stack_bb", "hero_pos", "opener_action"):
        if col not in df_rows.columns:
            raise ValueError(f"df_rows missing '{col}'. Has: {list(df_rows.columns)}")

    # Grab the group context from the first row
    gb = df_rows.iloc[0]
    stack_bb = gb["stack_bb"]
    hero_pos = gb["hero_pos"]
    opener_action = gb["opener_action"]

    # Filter manifest to this exact scenario
    msub = manifest[
        (manifest["stack_bb"] == stack_bb) &
        (manifest["hero_pos"] == hero_pos) &
        (manifest["opener_action"] == opener_action)
    ]
    if msub.empty:
        # No files? return the input with NaNs or zeros, your choice
        out = df_rows.copy()
        out["p_win"] = 0.0
        out["p_tie"] = 0.0
        out["p_lose"] = 1.0
        return out

    # Build villain distribution from the Monker files
    vill_dist = aggregate_villain_range(msub, root)

    # Compute equity for each hero hand_id vs villain range (preflop = exact enumeration)
    out_rows = []
    for _, row in df_rows.iterrows():
        hid = int(row["hand_id"])
        # convert hand_id -> canonical code (e.g., 0->'AA', 1->'KK', ...).
        # If you have HANDS_169 and ID->code helper:
        # hero_code = hand_code_from_id(hid)
        # Or with your current types file:
        hero_code = ALL_HANDS[hid]  # from ml.config.types.hands

        # exact preflop equity vs distribution
        p_win, p_tie, p_lose = equity_preflop_vs_range(hero_code, vill_dist)
        out = dict(row)
        out["p_win"] = p_win
        out["p_tie"] = p_tie
        out["p_lose"] = p_lose
        out_rows.append(out)

    return pd.DataFrame(out_rows)

def run_from_config(cfg: dict) -> None:
    """
    Drive the preflop equity computation entirely from YAML config.
    Expected config keys:

    inputs:
      manifest: data/artifacts/monker_manifest.parquet
      parquet_in: data/processed/equitynet_preflop.parquet

    outputs:
      parquet_out: data/artifacts/equitynet_preflop_evaluated.parquet

    monker:
      root: data/vendor/monker

    equity_mc:
      preflop_samples: 50000
      seed: 42
    """
    def get(path: str, default=None):
        cur = cfg
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    manifest_path = Path(get("inputs.manifest"))
    parquet_in    = Path(get("inputs.parquet_in"))
    parquet_out   = Path(get("outputs.parquet_out"))
    monker_root   = Path(get("monker.root", "data/vendor/monker"))
    samples       = int(get("equity_mc.preflop_samples", 50_000))
    seed          = int(get("equity_mc.seed", 42))

    if not manifest_path or not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not parquet_in or not parquet_in.exists():
        raise FileNotFoundError(f"parquet_in not found: {parquet_in}")

    manifest = load_manifest(str(manifest_path))
    in_df = pd.read_parquet(parquet_in)

    need_cols = {"stack_bb","hero_pos","opener_action","hand_id"}
    miss = need_cols - set(in_df.columns)
    if miss:
        raise ValueError(f"Input parquet missing columns: {sorted(miss)}")

    groups = in_df.groupby(["stack_bb", "hero_pos", "opener_action"],
                           as_index=False, group_keys=False)

    out_df = groups.apply(
        lambda g: compute_equity_for_group(
            df_rows=g,
            manifest=manifest,
            root=monker_root,
            n_samples=samples,
            seed=seed
        )
    )

    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(parquet_out, index=False)
    print(f"✅ wrote {parquet_out} with {len(out_df):,} rows")
    print("   Columns:", ", ".join(out_df.columns))


def main():
    import argparse
    from ml.utils.config import load_model_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="equitynet_preflop",
                    help="YAML with inputs/outputs/monker/equity_mc sections")
    # Optional one-off override for output:
    ap.add_argument("--out", type=str, default=None,
                    help="(Optional) override outputs.parquet_out")
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    if args.out:
        cfg.setdefault("outputs", {})["parquet_out"] = args.out

    run_from_config(cfg)


if __name__ == "__main__":
    main()