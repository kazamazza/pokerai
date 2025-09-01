# scripts/equity/compute_equity_from_manifest.py
from __future__ import annotations
import math, random, time
import sys
from pathlib import Path
from typing import List, Tuple, Set

import numpy as np
import pandas as pd

# pip install eval7
import eval7

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.etl.utils.hand import hand_id_to_combo
# ---- constants / helpers ----
ALL_CARDS = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "cdhs"]

def _sample_villain_hand(used: Set[eval7.Card]) -> Tuple[eval7.Card, eval7.Card]:
    """Sample a random distinct villain hand avoiding any 'used' cards."""
    deck = [c for c in ALL_CARDS if c not in used]
    i = random.randrange(len(deck))
    j = random.randrange(len(deck) - 1)
    if j >= i: j += 1
    return deck[i], deck[j]

def _sample_board(street: int, used: Set[eval7.Card]) -> List[eval7.Card]:
    """
    Sample a board consistent with street (0..3).
      0 = preflop (no board)
      1 = flop (3 cards)
      2 = turn (4 cards)
      3 = river (5 cards)
    """
    if street == 0:
        return []
    need = {1: 3, 2: 4, 3: 5}[street]
    deck = [c for c in ALL_CARDS if c not in used]
    random.shuffle(deck)
    return deck[:need]

def _equity_triplet_vs_random_1op(
    hero: Tuple[eval7.Card, eval7.Card],
    street: int,
    board: List[eval7.Card],
    n_sims: int,
) -> Tuple[int, int, int]:
    """
    Monte Carlo outcomes vs one random opponent given partial board.
    Returns counts: (wins, ties, losses) over n_sims.
    """
    assert len(hero) == 2
    assert len(board) in (0, 3, 4, 5)

    used = set(board)
    used.add(hero[0]); used.add(hero[1])

    wins = ties = losses = 0
    for _ in range(n_sims):
        v1, v2 = _sample_villain_hand(used)
        need = 5 - len(board)
        deck = [c for c in ALL_CARDS if c not in used and c not in (v1, v2)]
        random.shuffle(deck)
        full_board = board if need == 0 else (board + deck[:need])

        h_val = eval7.evaluate([hero[0], hero[1], *full_board])
        v_val = eval7.evaluate([v1, v2, *full_board])

        if h_val > v_val: wins += 1
        elif h_val == v_val: ties += 1
        else: losses += 1

    return wins, ties, losses

def compute_equity_from_manifest(
    manifest_parquet: str | Path,
    out_parquet: str | Path,
    *,
    preflop_samples: int = 20000,
    seed: int = 42,
) -> None:
    """
    Reads unified equity manifest and writes:
      street:int, board_cluster_id:float (NaN for preflop), hand_id:int,
      samples:int, p_win:float, p_tie:float, p_lose:float, weight:float
    """
    man = pd.read_parquet(str(manifest_parquet)).copy()
    # Expected manifest cols: street, hand_id, board_cluster_id (NaN ok), samples

    rows = []
    t0 = time.time()

    for i, r in man.iterrows():
        street = int(r["street"])
        hand_id = int(r["hand_id"])
        cluster = r.get("board_cluster_id", np.nan)  # may be NaN for preflop
        row_samples = int(r.get("samples", 0))
        sims = preflop_samples if street == 0 else max(1, row_samples)

        # per-row deterministic seed (so repeated runs are stable but different rows differ)
        base = int(seed)
        cl_key = 0 if (isinstance(cluster, float) and math.isnan(cluster)) else int(cluster)
        row_seed = base * 1_000_003 + street * 10_007 + hand_id * 101 + cl_key
        rng_state = random.getstate()
        random.seed(row_seed)

        # hero hand from your 169 mapping
        hero = hand_id_to_combo(hand_id)  # -> (eval7.Card, eval7.Card)

        # sample a partial board (for postflop); empty [] for preflop
        board = _sample_board(street, set(hero))

        w, t, l = _equity_triplet_vs_random_1op(hero, street, board, sims)
        total = float(w + t + l) if (w + t + l) > 0 else 1.0
        p_win, p_tie, p_lose = w / total, t / total, l / total

        rows.append((
            street,
            (float('nan') if isinstance(cluster, float) and math.isnan(cluster) else int(cluster)),
            hand_id,
            sims,
            p_win, p_tie, p_lose,
            float(sims),   # weight: number of MC sims for this row
        ))

        random.setstate(rng_state)  # restore RNG (optional)

    out_df = pd.DataFrame(
        rows,
        columns=["street","board_cluster_id","hand_id","samples","p_win","p_tie","p_lose","weight"]
    )
    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    dt = time.time() - t0
    print(f"✅ wrote equity parquet → {out_path}  rows={len(out_df):,}  in {dt:.1f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute EquityNet parquet from unified equity manifest.")
    ap.add_argument("--manifest", type=str, default="data/artifacts/equity_manifest.parquet")
    ap.add_argument("--out", type=str, default="data/datasets/equitynet.parquet")
    ap.add_argument("--preflop-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    compute_equity_from_manifest(args.manifest, args.out, preflop_samples=args.preflop_samples, seed=args.seed)