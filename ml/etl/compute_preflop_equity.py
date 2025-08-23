from __future__ import annotations
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import eval7
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.features.equity.preflop import equity_preflop_vs_range_cached
from ml.features.hands import hand_code_from_id, enumerate_suited_combos
from ml.etl.utils.monker_parser import load_range_file_cached

REQ_MANIFEST_COLS = ["stack_bb", "hero_pos", "opener_action", "rel_path", "abs_path"]

def load_manifest(manifest_parquet: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(manifest_parquet)
    missing = [c for c in REQ_MANIFEST_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    return df

def aggregate_villain_range(manifest_rows: pd.DataFrame, root: Path) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for _, r in manifest_rows.iterrows():
        path = Path(r["abs_path"]) if "abs_path" in r and r["abs_path"] else (root / r["rel_path"])
        d = load_range_file_cached(path)  # <-- correct helper; no actor kwarg
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



def _sample_discrete(weights: List[float], rng: random.Random) -> int:
    """Return index sampled according to weights (sum ~1)."""
    r = rng.random()
    cum = 0.0
    for i, w in enumerate(weights):
        cum += float(w)
        if r <= cum:
            return i
    return len(weights) - 1


def compute_equity_for_group(
    key: tuple,
    df_rows: pd.DataFrame,
    manifest: pd.DataFrame,
    root: Path,
    n_samples: int,
    seed: int
) -> pd.DataFrame:
    """
    Compute preflop equity for all hero hands in a scenario group.
    key = (stack_bb, hero_pos, opener_action)
    df_rows = the rows for this scenario (different hero hand_ids)
    """
    stack_bb, hero_pos, opener_action = key

    # 2) Aggregate villain range from all manifest rows for this scenario
    mask = (
        (manifest["stack_bb"] == stack_bb) &
        (manifest["hero_pos"] == hero_pos) &
        (manifest["opener_action"] == opener_action)
    )
    mrows = manifest[mask]
    vill_range: Dict[str, float] = {}
    for _, mr in mrows.iterrows():
        path = root / mr["rel_path"]  # or abs_path if you prefer
        r = load_range_file_cached(path)  # dict {hand_code -> prob}
        for k, v in r.items():
            vill_range[k] = vill_range.get(k, 0.0) + v

    # normalize
    s = sum(vill_range.values())
    if s > 0:
        for k in list(vill_range.keys()):
            vill_range[k] /= s

    # 3) For each hero hand_id in df_rows, run MC directly
    rng = random.Random(seed)
    out = []
    for _, row in df_rows.iterrows():
        hand_id = int(row["hand_id"])
        hero_code = hand_code_from_id(hand_id)

        p_win, p_tie, p_lose = equity_preflop_vs_range_cached(
            hero_code=hero_code,
            vill_range=vill_range,
            n_samples=n_samples,
            seed=rng.randint(0, 2**32 - 1),  # keep RNG moving per hand
        )

        out.append({
            "stack_bb": stack_bb,
            "hero_pos": hero_pos,
            "opener_action": opener_action,
            "hand_id": hand_id,
            "p_win": p_win,
            "p_tie": p_tie,
            "p_lose": p_lose,
            "weight": float(row.get("weight", 1.0)),
        })

    return pd.DataFrame(out)

def run_from_config(cfg: dict) -> None:
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
    groups = in_df.groupby(["stack_bb", "hero_pos", "opener_action"], sort=False)

    out_df = groups.apply(
        lambda g: compute_equity_for_group(
            g.name, g, manifest, monker_root, samples, seed
        )
    ).reset_index(drop=True)

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