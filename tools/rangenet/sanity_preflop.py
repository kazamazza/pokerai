import pandas as pd, numpy as np

df = pd.read_parquet("data/datasets/rangenet_preflop_from_flop.parquet")

# Check label sums
ysum = df[[f"y_{i}" for i in range(169)]].sum(axis=1).to_numpy()
print("label sum mean:", ysum.mean(), "min:", ysum.min(), "max:", ysum.max())

# Any negatives?
ymin = df[[f"y_{i}" for i in range(169)]].min(axis=1).min()
print("global min label:", ymin)

# Quick scenario counts
print(df.groupby(["ctx","opener_pos","hero_pos","stack_bb"]).size().head(20))