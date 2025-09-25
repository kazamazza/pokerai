import pandas as pd
df = pd.read_parquet("data/datasets/equitynet.parquet")
print("preflop weight unique:", sorted(df.loc[df.street==0,"weight"].unique()))
print("postflop weight unique:", sorted(df.loc[df.street>0,"weight"].unique())[:5], "...")