import pandas as pd, numpy as np, pathlib as p

PARQ = "data/datasets/rangenet_preflop_from_flop.parquet"
df = pd.read_parquet(PARQ)

# 1) columns present?
must_have = ["opener_pos","hero_pos","ctx","stack_bb"]
missing = [c for c in must_have if c not in df.columns]
print("missing core cols:", missing)

# 2) label block
y_cols = [c for c in df.columns if c.startswith("y_")]
print("label width:", len(y_cols), "(expect 169)")

# 3) normalization + dtype
ys = df[y_cols].to_numpy(dtype=np.float32)
row_sums = ys.sum(1)
print("row_sums: min", row_sums.min(), "max", row_sums.max(), "mean", row_sums.mean())

# 4) NaNs / negatives
print("has_nans:", np.isnan(ys).any(), "min_label:", ys.min())

# 5) cardinalities (for embeddings)
for cat in ["opener_pos","hero_pos","ctx","stack_bb"]:
    if cat in df.columns:
        print(cat, "nunique:", df[cat].nunique(), "sample:", df[cat].dropna().unique()[:10])