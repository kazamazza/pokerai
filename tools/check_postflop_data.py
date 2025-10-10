import pandas as pd
from collections import Counter
from pathlib import Path

# Path to your dataset
DATA_PATH = Path("data/datasets/postflop_policy_with_seats+manifest.parquet")

ACTIONS = [
    "FOLD","CHECK","CALL",
    "BET_25","BET_33","BET_50","BET_66","BET_75","BET_100",
    "DONK_33",
    "RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500","ALLIN"
]

def soft_label_distribution(df: pd.DataFrame):
    """Compute global soft-label mean per action."""
    present = [a for a in ACTIONS if a in df.columns]
    means = df[present].mean().sort_values(ascending=False)
    print("\n--- Global Soft-Label Means ---")
    print(means.head(10))
    print(f"Total actions found: {len(present)}")

    if "facing_bet" in df.columns:
        print("\n--- Soft Means by facing_bet ---")
        for k, g in df.groupby(df["facing_bet"].astype(int)):
            m = g[present].mean().sort_values(ascending=False).head(8)
            print(f"Facing={k}:")
            print(m)

def hard_label_distribution(df: pd.DataFrame):
    """Compute counts if 'action' column exists."""
    print("\n--- Hard Label Distribution ---")
    c = Counter(df["action"].str.upper())
    for k, v in c.most_common(10):
        print(f"{k:>10s} : {v:,}")
    print(f"Total unique actions: {len(c)}")

    if "facing_bet" in df.columns:
        print("\n--- Hard Label Distribution by facing_bet ---")
        for k, g in df.groupby(df["facing_bet"].astype(int)):
            c2 = Counter(g["action"].str.upper())
            print(f"Facing={k}: {dict(sorted(c2.items(), key=lambda x: -x[1])[:6])}")

def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Rows: {len(df):,}, Columns: {len(df.columns)}")
    print(df.head(3))

    if "action" in df.columns:
        hard_label_distribution(df)
    else:
        soft_label_distribution(df)

    # Quick positional/context summary
    for col in ["street", "ctx", "ip_pos", "oop_pos", "facing_bet"]:
        if col in df.columns:
            print(f"\n--- Value counts for {col} ---")
            print(df[col].value_counts().head(10))

if __name__ == "__main__":
    main()