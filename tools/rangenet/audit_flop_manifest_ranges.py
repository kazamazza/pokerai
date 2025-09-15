# tools/rangenet/audit_flop_manifest_ranges.py
import sys, pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "data/artifacts/rangenet_postflop_flop_manifest.parquet"
df = pd.read_parquet(path)

# quick summary
print("Range sources by ctx and stack:")
print(df.groupby(["ctx","effective_stack_bb","range_source"]).size().unstack(fill_value=0).sort_index())

print("\nExamples that should be Monker-backed:")
for ctx in ["VS_OPEN","BLIND_VS_STEAL","VS_3BET","VS_4BET"]:
    for st in [60.0, 100.0, 150.0]:
        sub = df[(df.ctx==ctx) & (df.effective_stack_bb==st)]
        srcs = sub["range_source"].value_counts()
        print(f"{ctx} @{int(st)}bb → {dict(srcs)}")

# show a few real rows
print("\nSample monker rows:")
print(df[df["range_source"].str.startswith("monker")].head(20)[
    ["ctx","topology","effective_stack_bb","pot_bb","positions","bet_sizing_id","range_source"]
])