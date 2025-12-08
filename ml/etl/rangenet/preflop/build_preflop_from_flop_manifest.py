import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

from ml.etl.rangenet.utils import to_vec169


# -----------------------------
# 1) Decode positions "UTGvBB"
# -----------------------------
def parse_positions(pos_raw: str):
    if not isinstance(pos_raw, str):
        return None, None

    pos = pos_raw.upper().replace(" ", "")

    # Standard format: "BTNvBB"
    if "V" in pos:
        parts = pos.split("V")
        if len(parts) == 2:
            return parts[0], parts[1]

    # Could not parse
    return None, None


# ---------------------------------
# 2) Convert ctx → synthetic actions
# ---------------------------------
def infer_action_sequence(ctx: str):
    c = (ctx or "").upper()
    if c == "SRP":
        return ["RAISE", "CALL", "NONE"]
    if c == "VS_OPEN":
        return ["RAISE", "CALL", "CALL"]
    if c == "VS_3BET":
        return ["RAISE", "3BET", "CALL"]
    if c == "VS_4BET":
        return ["RAISE", "3BET", "4BET"]
    if c == "LIMPED_SINGLE":
        return ["LIMP", "CHECK", "NONE"]
    if c == "LIMPED_MULTI":
        return ["LIMP", "CALL", "CALL"]
    return ["UNKNOWN", "UNKNOWN", "UNKNOWN"]


# ---------------------------------------
# 3) Parse the range_oop safely (string)
# ---------------------------------------
def parse_range_field(rng_raw):
    if isinstance(rng_raw, str):
        try:
            data = json.loads(rng_raw)
        except Exception:
            # Already a Python repr of list? Try eval safely.
            try:
                data = eval(rng_raw)
            except Exception:
                return np.ones(169, dtype=np.float32) / 169.0
    else:
        data = rng_raw

    return to_vec169(data).astype(np.float32)


# ---------------------------------------
# 4) Main builder
# ---------------------------------------
def build_preflop_dataset(df: pd.DataFrame):
    rows = []

    for _, row in df.iterrows():
        # Extract hero/villain positions
        ip, oop = parse_positions(row.get("positions", ""))
        if not ip or not oop:
            continue

        # We assume hero = OOP seat (BTN is OOP vs CO, etc.)
        hero_pos = oop
        villain_pos = ip

        # Extract ctx → synthetic sequence
        ctx_val = (row.get("ctx") or "SRP").upper()
        seq = infer_action_sequence(ctx_val)

        # Extract range vector from range_oop using to_vec169
        rng_oop = row.get("range_oop")
        if not rng_oop:
            continue

        try:
            y_vec = to_vec169(rng_oop)
            assert len(y_vec) == 169
        except Exception:
            continue

        # Compose dataset row
        out = {
            "stack_bb": float(row.get("effective_stack_bb") or 100.0),
            "hero_pos": hero_pos,
            "villain_pos": villain_pos,
            "action_seq_1": seq[0],
            "action_seq_2": seq[1],
            "action_seq_3": seq[2],
        }

        # Append y_0 … y_168
        for i in range(169):
            out[f"y_{i}"] = float(y_vec[i])

        rows.append(out)

    return pd.DataFrame(rows)


# ---------------------------------------
# 5) CLI
# ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flop-manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("🔍 Loading flop manifest:", args.flop_manifest)
    df = pd.read_parquet(args.flop_manifest)
    print("📦 Total rows:", len(df))

    out_df = build_preflop_dataset(df)

    print(f"✅ Built dataset with {len(out_df):,} rows")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print("💾 Saved to", out_path)


if __name__ == "__main__":
    main()