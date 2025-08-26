# audit_cell_consistency.py
import polars as pl, json, random

PARQUET = "data/datasets/populationnet_nl10_dev.parquet"
DECISIONS = "data/processed/nl10/decisions.jsonl.gz"  # the validator just downloaded this

df = pl.read_parquet(PARQUET)
dec = pl.read_ndjson(DECISIONS)

# normalize ALL_IN→RAISE like builder
dec = dec.with_columns(
    pl.when(pl.col("act_id")==5).then(2).otherwise(pl.col("act_id")).alias("act_id")
)

GRP = ["stakes_id","street_id","ctx_id","hero_pos_id","villain_pos_id"]
sample_cells = df.sample(n=min(10, df.height), seed=42).select(GRP).to_dicts()

def counts_for_cell(cell):
    mask = pl.all_horizontal([pl.col(k)==cell[k] for k in GRP])
    sub = dec.filter(mask)
    total = sub.height
    if total == 0: return None
    c = sub.group_by("act_id").count()
    n_fold = int(c.filter(pl.col("act_id")==0)["count"].fill_null(0).sum())
    n_call = int(c.filter(pl.col("act_id")==1)["count"].fill_null(0).sum())
    n_raise= int(c.filter(pl.col("act_id")==2)["count"].fill_null(0).sum())
    return total, n_fold, n_call, n_raise

bad = 0
for cell in sample_cells:
    got = counts_for_cell(cell)
    row = df.filter(pl.all_horizontal([pl.col(k)==cell[k] for k in GRP])).row(0, named=True)
    if not got:
        print("⚠️ empty in decisions for", cell); continue
    total, nf, nc, nr = got
    # compare to parquet’s n_rows and probs
    pf, pc, pr = row["p_fold"], row["p_call"], row["p_raise"]
    # reconstruct expected counts (nearest integer) for a loose check
    est_nf, est_nc, est_nr = round(pf*total), round(pc*total), round(pr*total)
    ok = (row["n_rows"]==total) and abs(nf-est_nf)<=1 and abs(nc-est_nc)<=1 and abs(nr-est_nr)<=1
    print("OK" if ok else "BAD", cell, "n_rows(dec/parq)=", total, row["n_rows"],
          "counts(dec)=", (nf,nc,nr), "probs(parq)=", (pf,pc,pr))
    if not ok: bad += 1

print("bad_cells:", bad)