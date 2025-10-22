#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

MANIFEST = Path("data/artifacts/rangenet_postflop_flop_manifest_NL10.parquet")
OUT_CSV  = Path("data/artifacts/missing_sph_tasks.csv")

print(f"Loading {MANIFEST} ...")
df = pd.read_parquet(MANIFEST)
print(f"Loaded {len(df):,} rows\n")

# Normalize column names for safety
cols = {c.lower(): c for c in df.columns}
def col(name): return cols.get(name.lower(), name)

# Detect fallback / SPH sources
source_col = col("range_source") or col("source")
if source_col not in df.columns:
    print("❌ No 'range_source' or 'source' column in manifest")
    exit(1)

fallback_mask = df[source_col].astype(str).str.contains("fallback|sph", case=False, na=False)
fallbacks = df[fallback_mask].copy()

print(f"Found {len(fallbacks):,} fallback rows "
      f"({len(fallbacks) / len(df) * 100:.2f}% of total)")

if fallbacks.empty:
    print("✅ No fallback ranges detected — all contexts resolved.")
    exit(0)

# Group to identify distinct missing pairs
key_cols = [
    col("ctx"), col("topology"),
    col("ip_actor_flop"), col("oop_actor_flop"),
    col("effective_stack_bb")
]
for k in key_cols:
    if k not in df.columns:
        key_cols.remove(k)

summary = (
    fallbacks
    .groupby(key_cols, dropna=False)
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

# Save and print summary
summary.to_csv(OUT_CSV, index=False)
print("\n=== Fallback summary ===")
print(summary.head(20).to_string(index=False))
print(f"\n📝 Saved detailed list to {OUT_CSV}")