import argparse
from pathlib import Path
import pandas as pd

def canon_pos(p: str) -> str:
    return str(p).strip().upper() if p is not None else None

def load_manifest(p: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(str(p)).copy()

    # Map new schema → legacy column names that the lookup expects
    if "ip_pos" not in df.columns and "ip_actor_flop" in df.columns:
        df["ip_pos"] = df["ip_actor_flop"]
    if "oop_pos" not in df.columns and "oop_actor_flop" in df.columns:
        df["oop_pos"] = df["oop_actor_flop"]

    # Try to derive ctx from topology if missing
    if "ctx" not in df.columns:
        topo2ctx = {
            "srp_hu": "SRP", "srp_multi": "SRP",
            "3bet_hu": "VS_3BET", "3bet_multi": "VS_3BET",
            "4bet_hu": "VS_4BET", "4bet_multi": "VS_4BET",
            "limped_single": "LIMPED_SINGLE",
            "limped_multi": "LIMPED_MULTI",
        }
        if "topology" in df.columns:
            df["ctx"] = df["topology"].map(lambda t: topo2ctx.get(str(t).lower(), None))

    # Canonicalize / types
    for col in ["ip_pos","oop_pos"]:
        if col in df.columns:
            df[col] = df[col].map(canon_pos)
    if "stack_bb" in df.columns:
        df["stack_bb"] = pd.to_numeric(df["stack_bb"], errors="coerce").astype("Int64")
    if "ctx" in df.columns:
        df["ctx"] = df["ctx"].map(lambda s: str(s).upper() if s is not None else None)

    # Keep rows with key fields
    need = ["stack_bb","ip_pos","oop_pos","ctx"]
    df = df[[c for c in need if c in df.columns] + [c for c in df.columns if c not in need]]
    df = df.dropna(subset=["stack_bb","ip_pos","oop_pos","ctx"])
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=str)
    ap.add_argument("--ctx", default="SRP", type=str, help="SRP | VS_3BET | VS_4BET | LIMPED_SINGLE | LIMPED_MULTI")
    ap.add_argument("--pairs", default="BTN-BB,BTN-SB,CO-BB", type=str,
                    help="Comma-separated IP-OOP pairs, e.g. 'UTG-BB,HJ-BB,CO-BB,BTN-BB'")
    ap.add_argument("--stacks", default="25,60,100,150", type=str,
                    help="Comma-separated stacks in bb, e.g. '25,60,100,150'")
    ap.add_argument("--out-prefix", default="data/artifacts/monker_cov", type=str)
    args = ap.parse_args()

    df = load_manifest(args.manifest)
    ctx = args.ctx.strip().upper()
    want_pairs = [tuple(x.strip().upper().split("-")) for x in args.pairs.split(",") if x.strip()]
    want_stacks = [int(s) for s in args.stacks.split(",") if s.strip()]

    # Filter to requested context
    df_ctx = df[df["ctx"] == ctx].copy()

    # Present table (all pairs available for this ctx)
    present = (
        df_ctx.groupby(["stack_bb","ip_pos","oop_pos"], dropna=False)
              .size().reset_index(name="n")
              .sort_values(["stack_bb","ip_pos","oop_pos"])
    )

    # Availability matrix for requested pairs x stacks (1/0)
    rows = []
    present_set = {(int(r.stack_bb), r.ip_pos, r.oop_pos) for _, r in present.iterrows()}
    for ip, oop in want_pairs:
        for st in want_stacks:
            rows.append({
                "ctx": ctx,
                "pair": f"{ip}-{oop}",
                "stack_bb": st,
                "available": 1 if (st, ip, oop) in present_set else 0
            })
    matrix = pd.DataFrame(rows)

    # Print summaries
    print(f"\n=== Coverage for ctx={ctx} ===")
    print(f"Total rows (ctx): {len(df_ctx)}")
    print("\nTop present pairs (first 30):")
    print(present.head(30).to_string(index=False))

    print("\nRequested availability matrix:")
    pivot = matrix.pivot_table(index="pair", columns="stack_bb", values="available", fill_value=0, aggfunc="max")
    print(pivot.to_string())

    # Write CSVs
    out_present = Path(f"{args.out_prefix}_present.csv")
    out_matrix  = Path(f"{args.out_prefix}_matrix.csv")
    out_present.parent.mkdir(parents=True, exist_ok=True)
    present.to_csv(out_present, index=False)
    matrix.to_csv(out_matrix, index=False)
    print(f"\n📄 Wrote: {out_present}")
    print(f"📄 Wrote: {out_matrix}")

if __name__ == "__main__":
    main()