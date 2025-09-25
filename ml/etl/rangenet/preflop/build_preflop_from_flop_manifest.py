import argparse, json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from ml.etl.rangenet.utils import to_vec169


def build_preflop_from_flop_manifest(
    flop_manifest: Path,
    out_parquet: Path,
    root_only: bool = True,
) -> pd.DataFrame:
    df = pd.read_parquet(flop_manifest)

    # --- Select flop root rows only (street==1 + node_key=='root' if present) ---
    if "street" in df.columns:
        df = df[df["street"].astype("Int64") == 1]
    if root_only and "node_key" in df.columns:
        df = df[df["node_key"].astype(str).str.lower() == "root"]
    df = df.reset_index(drop=True)

    # --- Helpers ---
    def _norm_pos_token(p: str) -> str:
        p = str(p).strip().upper()
        # Allow IP/OOP and regular named seats
        if p in {"UTG","HJ","CO","BTN","SB","BB","IP","OOP"}:
            return p
        return p

    def positions_from_manifest_row(row: pd.Series) -> tuple[str, str]:
        pos = str(row.get("positions", "")).strip().upper()
        # Expect "BTNvBB" or "BTN V BB"
        if "V" in pos:
            a, b = [x.strip() for x in pos.split("V", 1)]
            return _norm_pos_token(a), _norm_pos_token(b)
        # Fallback to opener vs defender if separately provided
        a = _norm_pos_token(row.get("opener_pos", "BTN"))
        b = _norm_pos_token(row.get("defender_pos", "BB"))
        return a, b

    def ctx_from_manifest_row(row: pd.Series) -> str:
        if "ctx" in row and pd.notna(row["ctx"]):
            return str(row["ctx"]).strip().upper()
        m = str(row.get("bet_sizing_id", "")).lower()
        if m.startswith("limped_single"): return "LIMPED_SINGLE"
        if m.startswith("limped_multi"):  return "LIMPED_MULTI"
        if m.startswith("3bet"):          return "VS_3BET"
        if m.startswith("4bet"):          return "VS_4BET"
        return "SRP"

    def opener_action_for_ctx(ctx: str) -> str:
        c = str(ctx).upper()
        if c in ("SRP","BLIND_VS_STEAL"): return "RAISE"
        if c == "VS_3BET": return "3BET"
        if c == "VS_4BET": return "4BET"
        if c in ("LIMPED_SINGLE","LIMPED_MULTI"): return "LIMP"
        return "RAISE"

    def to_vec169_safe(rng) -> np.ndarray:
        v = to_vec169(rng)  # your helper
        v = np.asarray(v, dtype="float32").clip(min=0.0)
        s = float(v.sum())
        return (v / s) if s > 0 else (np.ones(169, dtype="float32") / 169.0)

    # --- Derive scenario columns ---
    stacks = np.rint(df.get("effective_stack_bb", 0).astype(float)).astype(int)
    op_pos, hero_pos = zip(*df.apply(positions_from_manifest_row, axis=1))
    ctx_vals = df.apply(ctx_from_manifest_row, axis=1)
    open_act = [opener_action_for_ctx(c) for c in ctx_vals]

    # Require both ranges present
    has_ip  = ("range_ip"  in df.columns) & df["range_ip"].notna()
    has_oop = ("range_oop" in df.columns) & df["range_oop"].notna()
    df = df[has_ip & has_oop].reset_index(drop=True)

    keys_df = pd.DataFrame({
        "stack_bb": stacks,
        "opener_pos": list(op_pos),
        "hero_pos":   list(hero_pos),
        "ctx":        ctx_vals.astype(str).str.upper(),
        "opener_action": [oa.upper() for oa in open_act],
    }).loc[df.index].reset_index(drop=True)

    df = pd.concat([keys_df, df[["range_ip","range_oop","positions"]]], axis=1)

    # --- Group by scenario & average the VILLAIN range (the non-hero side) ---
    rows = []
    for keys, g in df.groupby(["stack_bb","opener_pos","hero_pos","opener_action","ctx"], sort=False):
        vecs = []
        for _, r in g.iterrows():
            opp_open, hero = r["opener_pos"], r["hero_pos"]
            # Compute IP/OOP from positions string (opener v defender)
            ip, oop = positions_from_manifest_row(pd.Series({"positions": r["positions"]}))
            # Villain is "the other seat" vs hero:
            villain_vec = r["range_oop"] if hero == ip else r["range_ip"]
            vecs.append(to_vec169_safe(villain_vec))

        y = np.mean(np.stack(vecs, axis=0), axis=0).astype("float32")
        out = dict(zip(["stack_bb","opener_pos","hero_pos","opener_action","ctx"], keys))
        out["weight"] = float(len(g))
        for i, val in enumerate(y.tolist()):
            out[f"y_{i}"] = float(val)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)
    print(f"✅ wrote {out_parquet} with {len(out_df):,} scenarios (from {len(df):,} flop-root rows))")
    return out_df

def main():
    ap = argparse.ArgumentParser("Build preflop dataset directly from postflop flop manifest")
    ap.add_argument("--flop-manifest", default="data/artifacts/rangenet_postflop_flop_manifest.parquet")
    ap.add_argument("--out", default="data/datasets/rangenet_preflop_from_flop.parquet")
    ap.add_argument("--all-nodes", action="store_true", help="include non-root rows (averaged by scenario)")
    args = ap.parse_args()

    build_preflop_from_flop_manifest(
        Path(args.flop_manifest),
        Path(args.out),
        root_only=not args.all_nodes,
    )

if __name__ == "__main__":
    main()