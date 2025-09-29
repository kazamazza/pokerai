import polars as pl

# load the final parquet
df = pl.read_parquet("data/datasets/populationnet_nl10.parquet")

print(f"Total cells: {df.height}")

# basic coverage by street
print("\nBy street_id:")
print(df.group_by("street_id").count().sort("count", descending=True))

# coverage by ctx_id (top 10 only)
print("\nBy ctx_id (top 10):")
print(df.group_by("ctx_id").count().sort("count", descending=True).head(10))

# coverage by hero_pos_id
print("\nBy hero_pos_id:")
print(df.group_by("hero_pos_id").count().sort("count", descending=True))

# coverage by villain_pos_id
print("\nBy villain_pos_id:")
print(df.group_by("villain_pos_id").count().sort("count", descending=True))

# optional: distribution of weights
print("\nWeight distribution:")
print(
    df.select([
        pl.col("w").min().alias("min_w"),
        pl.col("w").max().alias("max_w"),
        pl.col("w").mean().alias("mean_w"),
        pl.col("w").median().alias("median_w"),
    ])
)