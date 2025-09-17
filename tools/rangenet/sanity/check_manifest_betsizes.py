import pandas as pd
from pathlib import Path

# path to your flop manifest
path = Path("data/artifacts/rangenet_postflop_flop_manifest.parquet")
df = pd.read_parquet(path)

print(f"Loaded manifest with {len(df)} rows")

# 1. Check if column exists
if "bet_sizing_id" not in df.columns:
    raise SystemExit("❌ manifest is missing bet_sizing_id column")

# 2. Count missing values
missing = df["bet_sizing_id"].isna().sum()
print(f"Missing bet_sizing_id: {missing}")

# 3. Show distribution by context
print("\nCounts by ctx and bet_sizing_id:")
print(df.groupby(["ctx", "bet_sizing_id"]).size().unstack(fill_value=0))

# 4. Preview some rows
print("\nSample rows with bet_sizing_id:")
print(df[["ctx","positions","effective_stack_bb","bet_sizing_id"]].head(10).to_string(index=False))