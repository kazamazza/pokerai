#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def _board_mask_52(board: str) -> list[float]:
    """Encode 'Jh8d8c' → 52-bit mask (rank major, suits c,d,h,s per rank)."""
    if not isinstance(board, str) or len(board) < 2:
        return [0.0] * 52
    ranks = "23456789TJQKA"
    suits = "cdhs"
    idx = lambda r, s: ranks.index(r.upper()) * 4 + suits.index(s.lower())
    s = board.replace(" ", "")
    mask = [0.0] * 52
    try:
        for i in range(0, len(s), 2):
            r, u = s[i], s[i+1]
            j = idx(r, u)
            mask[j] = 1.0
    except Exception:
        return [0.0] * 52
    return mask

def main():
    ap = argparse.ArgumentParser("Patch postflop dataset with manifest-derived columns")
    ap.add_argument("--postflop-parquet", required=True,
                    help="e.g. data/datasets/postflop_policy_with_seats.parquet")
    ap.add_argument("--manifest-parquet", required=True,
                    help="e.g. data/artifacts/rangenet_postflop_flop_manifest.parquet")
    ap.add_argument("--out", required=True,
                    help="Output parquet path (won’t overwrite input)")
    args = ap.parse_args()

    post_p = Path(args.postflop_parquet)
    man_p  = Path(args.manifest_parquet)
    out_p  = Path(args.out)
    if not post_p.exists():
        raise FileNotFoundError(post_p)
    if not man_p.exists():
        raise FileNotFoundError(man_p)

    df = pd.read_parquet(post_p)
    mf = pd.read_parquet(man_p)

    # Keep only fields we need from manifest
    mf_keep = mf[[
        "s3_key", "board", "board_cluster_id", "bet_sizes",
        "effective_stack_bb", "pot_bb", "bet_sizing_id"
    ]].copy()

    # Deduplicate manifest rows on s3_key (there should be one per flop-root)
    mf_keep = mf_keep.drop_duplicates(subset=["s3_key"])

    # Join on s3_key
    if "s3_key" not in df.columns:
        raise ValueError("postflop dataset is missing 's3_key' column; cannot join to manifest.")
    merged = df.merge(mf_keep, on="s3_key", how="left", suffixes=("", "_mf"))

    # --- hero_pos (missing in dataset): default to ip_pos at root ---
    if "hero_pos" not in merged.columns:
        if "ip_pos" in merged.columns:
            merged["hero_pos"] = merged["ip_pos"]
        else:
            merged["hero_pos"] = "BTN"  # safe fallback

    # --- effective_stack_bb: prefer manifest, else map from stack_bb ---
    if "effective_stack_bb" not in merged.columns or merged["effective_stack_bb"].isna().all():
        if "stack_bb" in merged.columns:
            merged["effective_stack_bb"] = merged["stack_bb"].astype(float)
        else:
            merged["effective_stack_bb"] = 100.0
    else:
        merged["effective_stack_bb"] = merged["effective_stack_bb"].astype(float).fillna(
            merged.get("stack_bb", 100).astype(float) if "stack_bb" in merged else 100.0
        )

    # --- pot_bb: prefer dataset if present, else manifest pot_bb, else rough fallback ---
    if "pot_bb" in merged.columns:
        merged["pot_bb"] = merged["pot_bb"].astype(float).fillna(merged.get("pot_bb_mf"))
    else:
        merged["pot_bb"] = merged.get("pot_bb_mf", pd.Series(dtype=float)).astype(float)
    merged["pot_bb"] = merged["pot_bb"].fillna(28.0)

    # --- board_cluster_id from manifest (if present) ---
    if "board_cluster_id" not in merged.columns:
        merged["board_cluster_id"] = merged.get("board_cluster_id_mf")
    merged["board_cluster_id"] = merged["board_cluster_id"].astype("Int64")  # allow NaN-like

    # --- board_mask_52 from manifest.board ---
    merged["board_mask_52"] = merged.get("board", "").map(_board_mask_52)

    # --- bet_sizes from manifest (JSON-able list) ---
    if "bet_sizes" not in merged.columns:
        merged["bet_sizes"] = merged.get("bet_sizes_mf")
    # Normalize bet_sizes to plain Python lists of floats
    def _norm_menu(v):
        if v is None:
            return None
        if isinstance(v, str):
            try: v = json.loads(v)
            except Exception: return None
        if isinstance(v, (list, tuple)):
            try: return [float(x) for x in v]
            except Exception: return None
        return None
    merged["bet_sizes"] = merged["bet_sizes"].map(_norm_menu)

    # --- actor & facing_bet (root nodes) ---
    if "actor" not in merged.columns:
        # At flop root the acting player is the one to c-bet (usually IP aggressor).
        # We default to 'ip' which matches most menus; OOP donk roots can be refined later if needed.
        merged["actor"] = "ip"
    if "facing_bet" not in merged.columns:
        merged["facing_bet"] = 0

    # Tidy columns (drop helper *_mf)
    drop_cols = [c for c in merged.columns if c.endswith("_mf")]
    merged = merged.drop(columns=drop_cols)

    # Save
    out_p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_p, index=False)
    print(f"✅ Patched dataset saved → {out_p}")
    # Quick echo of what we added
    added = ["hero_pos","effective_stack_bb","board_cluster_id","board_mask_52","bet_sizes","actor","facing_bet"]
    present = [c for c in added if c in merged.columns]
    print("Added/filled columns:", present)

if __name__ == "__main__":
    main()