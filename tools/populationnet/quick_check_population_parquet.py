# quick_check_population_parquet.py
import polars as pl, sys
p = "data/datasets/populationnet_nl10_dev.parquet"
df = pl.read_parquet(p)

assert df.height > 0, "Empty parquet"
for col in ["p_fold","p_call","p_raise","y","w","n_rows","stakes_id","street_id","ctx_id","hero_pos_id","villain_pos_id"]:
    assert col in df.columns, f"Missing column {col}"

# probs sum to ~1
bad = df.filter((pl.col("p_fold")+pl.col("p_call")+pl.col("p_raise")-1.0).abs()>1e-6)
print("prob_sum_bad_rows:", bad.height)

# no NaNs
nan_any = df.select([pl.all().is_nan().any()]).row(0)[0]
print("has_nan:", nan_any)

# label matches argmax
argmax = (pl.concat_list([pl.col("p_fold"),pl.col("p_call"),pl.col("p_raise")])
          .list.arg_max())
mismatch = df.filter(argmax != pl.col("y"))
print("y_mismatch:", mismatch.height)

# basic coverage by ctx
by_ctx = df.group_by(["street_id","ctx_id"]).agg(pl.len().alias("cells"), pl.col("n_rows").sum().alias("rows"))
print(by_ctx.sort(["street_id","ctx_id"]))