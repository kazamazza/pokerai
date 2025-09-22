# tools/sanity/audit_manifest_dupes.py
import sys, pandas as pd
path = sys.argv[1]
df = pd.read_parquet(path)

# how many unique s3 outputs?
u = df["s3_key"].nunique()
print("rows:", len(df), " unique s3_key:", u, " dupes:", len(df)-u)

# show the worst-offending keys and what columns differ
dupes = (df.groupby("s3_key").size().reset_index(name="n").query("n>1"))
print("\nTop duplicate keys:")
print(dupes.sort_values("n", ascending=False).head(10).to_string(index=False))

# peek at a few duplicate groups to see which fields vary
cols_to_compare = [
  "accuracy","max_iter","allin_threshold","positions","board","bet_sizing_id",
  "pot_bb","effective_stack_bb","range_ip","range_oop"
]
sample_keys = dupes["s3_key"].head(3).tolist()
for k in sample_keys:
    g = df[df["s3_key"]==k][["sha1","s3_key"]+cols_to_compare].reset_index(drop=True)
    print("\n--- s3_key:", k, " (", len(g), "rows ) ---")
    print(g.to_string(index=False))