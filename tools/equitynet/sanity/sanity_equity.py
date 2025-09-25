# sanity_equity.py (quick check)
import pandas as pd, numpy as np
df = pd.read_parquet("data/datasets/equitynet.parquet")

print("rows:", len(df))
print("cols:", list(df.columns))

# 1) schema & nulls
print(df.isna().sum())

# 2) street distribution
print(df["street"].value_counts(dropna=False).sort_index())

# 3) preflop is 169 & cluster is NaN only at preflop
print("preflop rows:", (df["street"]==0).sum())
print("preflop cluster NaN:", df.loc[df["street"]==0, "board_cluster_id"].isna().all())

# 4) probability triplet sums to ~1
triplet_sum = (df["p_win"] + df["p_tie"] + df["p_lose"]).values
print("triplet_sum ~1:", float(np.abs(triplet_sum - 1).max()))

# 5) weights positive
print("min weight:", float(df["weight"].min()))
print("min samples:", int(df["samples"].min()))