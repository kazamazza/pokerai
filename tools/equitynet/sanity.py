import pandas as pd

df = pd.read_parquet("data/artifacts/equity_manifest.parquet")
print("Rows:", len(df))
print(df["street"].value_counts(dropna=False).sort_index())

# Preflop expectations
pre = df[df["street"] == 0]
assert len(pre) == 169, f"Expected 169 preflop rows, got {len(pre)}"
assert (pre["board_cluster_id"] == -1).all()

# Postflop expectations
post = df[df["street"].isin([1,2,3])]
assert not post.empty, "No postflop rows found!"

for s in [1,2,3]:
    sub = post[post["street"] == s]
    if sub.empty:
        print(f"⚠️ No rows for street={s}")
        continue
    # cluster ids in range?
    bc_ok = sub["board_cluster_id"].between(0, sub["board_cluster_id"].max()).all()
    print(f"street={s}: rows={len(sub):,}, cluster_id_min={sub['board_cluster_id'].min()}, "
          f"cluster_id_max={sub['board_cluster_id'].max()}, clusters_ok={bc_ok}, "
          f"samples_min={sub['samples'].min()}, samples_max={sub['samples'].max()}")