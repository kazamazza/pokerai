#!/usr/bin/env python
import argparse
import pandas as pd

def norm_pos_str(s: str) -> str:
    s = str(s).upper()
    return s.replace("V", "v") if "V" in s else s

def build_root_key_df_from_manifest(df: pd.DataFrame, use_clusters: bool) -> pd.DataFrame:
    need = ["positions","street","effective_stack_bb","pot_bb","bet_sizing_id","ctx"]
    need += (["board_cluster_id"] if use_clusters else ["board"])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"[manifest] missing columns: {missing}")

    out = pd.DataFrame({
        "positions": df["positions"].astype(str).map(norm_pos_str),
        "street": df["street"].astype(int),
        "effective_stack_bb": df["effective_stack_bb"].astype(int),
        "pot_bb": df["pot_bb"].astype(float),
        "bet_sizing_id": df["bet_sizing_id"].astype(str),
        "ctx": df["ctx"].astype(str),
        "node_key": df.get("node_key", "root").astype(str),
    })
    if use_clusters:
        out["board_key"] = df["board_cluster_id"].astype(int)
    else:
        out["board_key"] = df["board"].astype(str)
    return out

def build_root_key_df_from_merged(df: pd.DataFrame, use_clusters: bool) -> pd.DataFrame:
    need = ["ip_pos","oop_pos","street","stack_bb","pot_bb","bet_sizing_id","ctx","node_key"]
    need += (["board_cluster_id"] if use_clusters else ["board"])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"[merged] missing columns: {missing}")

    pos = (df["ip_pos"].astype(str).str.upper() + "v" + df["oop_pos"].astype(str).str.upper())
    out = pd.DataFrame({
        "positions": pos.map(norm_pos_str),
        "street": df["street"].astype(int),
        "effective_stack_bb": df["stack_bb"].astype(int),
        "pot_bb": df["pot_bb"].astype(float),
        "bet_sizing_id": df["bet_sizing_id"].astype(str),
        "ctx": df["ctx"].astype(str),
        "node_key": df["node_key"].astype(str),
    })
    if use_clusters:
        out["board_key"] = df["board_cluster_id"].astype(int)
    else:
        out["board_key"] = df["board"].astype(str)
    return out.drop_duplicates().reset_index(drop=True)

def vc_align(label, s_left: pd.Series, s_right: pd.Series, top=10):
    l = s_left.copy().rename("manifest")
    r = s_right.copy().rename("merged")
    comp = pd.concat([l, r], axis=1).fillna(0).astype(int)
    comp["diff"] = comp["merged"] - comp["manifest"]
    print(f"\n=== by {label} ===")
    if comp.empty:
        print("(no rows)")
        return
    comp = comp.sort_values("diff", key=lambda s: s.abs(), ascending=False)
    print(comp.head(top).to_string())

def main():
    ap = argparse.ArgumentParser("Final sanity check: merged dataset vs deduped manifest")
    ap.add_argument("--manifest", required=True, help="Path to deduped manifest parquet")
    ap.add_argument("--merged",   required=True, help="Path to merged training parquet")
    ap.add_argument("--use-clusters", action="store_true", help="Use board_cluster_id instead of raw board")
    ap.add_argument("--dump-missing", type=str, default=None, help="CSV for roots in manifest but missing in merged")
    ap.add_argument("--dump-extra",   type=str, default=None, help="CSV for roots in merged but not in manifest")
    args = ap.parse_args()

    m_df = pd.read_parquet(args.manifest)
    t_df = pd.read_parquet(args.merged)

    m_keys = build_root_key_df_from_manifest(m_df, use_clusters=args.use_clusters)
    r_keys = build_root_key_df_from_merged(t_df, use_clusters=args.use_clusters)

    m_keys_uni = m_keys.drop_duplicates().reset_index(drop=True)
    r_keys_uni = r_keys.drop_duplicates().reset_index(drop=True)

    print("=== COUNTS ===")
    print(f"manifest rows (raw): {len(m_df):,}")
    print(f"manifest root_keys (unique): {len(m_keys_uni):,}")
    print(f"merged rows (per-action): {len(t_df):,}")
    print(f"merged root_keys (unique): {len(r_keys_uni):,}")

    key_cols = ["positions","street","effective_stack_bb","pot_bb","bet_sizing_id","ctx","node_key","board_key"]
    m_tuple = set(map(tuple, m_keys_uni[key_cols].itertuples(index=False, name=None)))
    r_tuple = set(map(tuple, r_keys_uni[key_cols].itertuples(index=False, name=None)))

    missing = m_tuple - r_tuple
    extra   = r_tuple - m_tuple

    print(f"\n=== ROOT COVERAGE ===")
    print(f"missing roots (in manifest, not in merged): {len(missing):,}")
    print(f"extra roots    (in merged, not in manifest): {len(extra):,}")

    if args.dump_missing and missing:
        pd.DataFrame(list(missing), columns=key_cols).to_csv(args.dump_missing, index=False)
        print(f"→ wrote missing roots CSV: {args.dump_missing}")
    if args.dump_extra and extra:
        pd.DataFrame(list(extra), columns=key_cols).to_csv(args.dump_extra, index=False)
        print(f"→ wrote extra roots CSV: {args.dump_extra}")

    vc_align("ctx",
             m_keys_uni["ctx"].value_counts(),
             r_keys_uni["ctx"].value_counts())

    vc_align("positions",
             m_keys_uni["positions"].value_counts(),
             r_keys_uni["positions"].value_counts())

    vc_align("bet_sizing_id",
             m_keys_uni["bet_sizing_id"].value_counts(),
             r_keys_uni["bet_sizing_id"].value_counts())

    ycols = [c for c in t_df.columns if c.startswith("y_")]
    if ycols:
        sums = t_df[ycols].sum(axis=1)
        bad = (sums - 1.0).abs() > 1e-6
        print(f"\n=== LABEL NORMALIZATION (merged) ===")
        print(f"non-normalized rows: {int(bad.sum())} / {len(t_df):,}")

if __name__ == "__main__":
    main()