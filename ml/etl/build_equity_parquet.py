import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.config.types_hands import ALL_HANDS, HAND_TO_ID
from ml.etl.utils.monker_parser import load_range_file_cached

REQUIRED_MANIFEST_COLS = [
    "stack_bb",       # int, e.g. 12
    "hero_pos",       # str, e.g. 'SB'
    "opener_action",  # str, e.g. 'UTG_Min_CO_Call_SB_AI_...'
    "rel_path",       # str
    "abs_path",       # str
    "n_files",        # int (count for the group)
]

def build_equity_parquet(
    manifest_parquet: str | Path,
    vendor_root: str | Path,
    out_parquet: str | Path,
) -> None:
    """
    Read the manifest parquet (one row per file), parse each file's range,
    aggregate to a soft-label distribution per scenario:
      X = (stack_bb, hero_pos, opener_action)
      Y = 169-d vector of p_freq
      W = weight (sum of files contributing to scenario)
    Output Parquet schema (long form, one row per hand):
      stack_bb:int, hero_pos:str, opener_action:str, hand_id:int, p_freq:float, weight:float
    """
    vendor_root = Path(vendor_root)
    man = pd.read_parquet(manifest_parquet)

    # Validate manifest columns
    missing = [c for c in REQUIRED_MANIFEST_COLS if c not in man.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    # Fill abs_path if missing (use vendor_root / rel_path)
    if "abs_path" not in man.columns or man["abs_path"].isna().any():
        man["abs_path"] = (vendor_root / man["rel_path"].astype(str)).astype(str)

    # Group by scenario keys
    grp_keys = ["stack_bb", "hero_pos", "opener_action"]
    rows: List[Tuple[int, str, str, int, float, float]] = []

    for (stack_bb, hero_pos, opener_action), g in man.groupby(grp_keys):
        # aggregate vector over files
        agg_vec = [0.0] * len(ALL_HANDS)
        total_files = 0

        for _, r in g.iterrows():
            path = Path(str(r["abs_path"]))
            if not path.exists():
                # Skip missing files quietly (or log)
                continue
            d = load_range_file_cached(path)  # dict hand -> prob
            if not d:
                continue
            # convert to vector
            vec = [0.0] * len(ALL_HANDS)
            for hand, p in d.items():
                vec[HAND_TO_ID[hand]] = float(p)

            # renormalize vec just in case
            s = sum(vec)
            if s > 0:
                vec = [v / s for v in vec]

            # simple average (each file weight = 1). You can weight by file quality if needed.
            agg_vec = [a + v for a, v in zip(agg_vec, vec)]
            total_files += 1

        if total_files == 0:
            # no valid files -> skip this scenario
            continue

        # average over files
        agg_vec = [v / total_files for v in agg_vec]

        # final weight: number of files that contributed
        weight = float(total_files)

        # emit long-form rows: one per hand_id
        for hand_id, p in enumerate(agg_vec):
            rows.append((int(stack_bb), str(hero_pos), str(opener_action), int(hand_id), float(p), weight))

    out_df = pd.DataFrame(rows, columns=["stack_bb", "hero_pos", "opener_action", "hand_id", "p_freq", "weight"])

    # Optional safety: ensure each scenario's p sums to ~1
    # (Not strictly required since we normalized, but harmless)
    # You can skip this step if you prefer speed.
    def _fix_group(df: pd.DataFrame) -> pd.DataFrame:
        s = df["p_freq"].sum()
        if s > 0 and abs(s - 1.0) > 1e-6:
            df["p_freq"] = df["p_freq"] / s
        return df

    out_df = out_df.groupby(["stack_bb", "hero_pos", "opener_action"], group_keys=False).apply(_fix_group)

    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"✅ wrote {out_path} with {len(out_df):,} rows across {out_df[['stack_bb','hero_pos','opener_action']].drop_duplicates().shape[0]} scenarios")
    print("   Schema: stack_bb:int, hero_pos:str, opener_action:str, hand_id:int(0..168), p_freq:float, weight:float")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="e.g. artifacts/monker_manifest.parquet")
    ap.add_argument("--vendor_root", default="datasets/vendor/monker", help="root to join rel_path if abs_path missing")
    ap.add_argument("--out", default="datasets/datasets/equitynet_preflop.parquet")
    args = ap.parse_args()
    build_equity_parquet(args.manifest, args.vendor_root, args.out)

if __name__ == "__main__":
    main()