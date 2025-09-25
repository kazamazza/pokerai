import pandas as pd, hashlib, numpy as np

p = "data/datasets/rangenet_postflop_merged.parquet"
df = pd.read_parquet(p)

# pick a composite “should-be-unique” key per row
KEY_COLS = [
    "bet_sizing_id", "ip_pos", "oop_pos", "hero_pos",
    "ctx", "street", "actor", "board_cluster", "pot_bb", "stack_bb"
]
present = [c for c in KEY_COLS if c in df.columns]
df["__uid"] = (
    df[present]
    .astype(str)
    .agg("|".join, axis=1)
    .apply(lambda s: hashlib.sha1(s.encode()).hexdigest())
)

dups = df.duplicated("__uid", keep=False)
n_dups = int(dups.sum())

print(f"rows={len(df):,}  duplicate rows={n_dups:,}")
if n_dups:
    print("\nSample duplicate groups:")
    for uid, grp in df.loc[dups].groupby("__uid"):
        print(grp[present + ["action"]].head(2).to_string(index=False))
        if grp.shape[0] > 2: print(f"... (+{grp.shape[0]-2} more)")
        break  # just show one group

# If you want to de-dup:
# df = df[~dups].drop(columns="__uid")
# df.to_parquet("data/datasets/rangenet_postflop_merged.dedup.parquet", index=False)