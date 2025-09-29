import polars as pl

df = pl.read_parquet("data/datasets/populationnet_nl10.parquet")

# How many raw observations per cell?
print(df.select(pl.col("n_rows").sum().alias("total_rows")))

# Cells with very low support (e.g., < 20 decisions)
threshold = 20
low = df.filter(pl.col("n_rows") < threshold).select(
    "stakes_id","street_id","ctx_id","hero_pos_id","villain_pos_id","n_rows"
).sort("n_rows")
print(f"cells with < {threshold} rows: {low.height}")
print(low.head(20))

# Coverage by context x street
cov = (df
    .group_by(["ctx_id","street_id"])
    .len()
    .sort("len", descending=True)
)
print(cov)

# Check probability sanity (should already be normalized)
print(df.select(
    (pl.col("p_fold")+pl.col("p_call")+pl.col("p_raise")).alias("sum_p")
).select(
    pl.col("sum_p").min().alias("min_sum_p"),
    pl.col("sum_p").max().alias("max_sum_p"),
))