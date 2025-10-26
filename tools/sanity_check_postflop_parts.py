# tools/sanity_check_postflop_parts.py
# Usage:
#   python tools/sanity_check_postflop_parts.py \
#     --root data/datasets/postflop_policy_parts_root/shard-00of01-root-part-00000.parquet \
#     --facing data/datasets/postflop_policy_parts_facing/shard-00of01-facing-part-00000.parquet \
#     --out data/diagnostics/postflop_parts_audit.csv

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ACTION_VOCAB = [
    "FOLD","CHECK","CALL",
    "BET_25","BET_33","BET_50","BET_66","BET_75","BET_100",
    "DONK_25","DONK_33","DONK_50","DONK_66","DONK_75","DONK_100",
    "RAISE_150","RAISE_200","RAISE_250","RAISE_300","RAISE_400","RAISE_500",
    "ALLIN",
]

ROOT_ALLOWED_STATIC = {"CHECK","ALLIN"}
FACING_ALLOWED_STATIC = {"FOLD","CALL","ALLIN"}

def _parse_menu_id(menu_id: str) -> Tuple[str, str]:
    s = str(menu_id or "").strip()
    if "." in s:
        g, r = s.split(".", 1)
        return g.strip(), r.strip()
    return s, ""

def _oop_is_caller(menu_id: str) -> bool:
    g, r = _parse_menu_id(menu_id)
    if g.startswith("limped_single"):
        return True
    if g == "srp_hu":
        return r.endswith("PFR_IP") or r.endswith("Caller_OOP")
    if g == "3bet_hu":
        return r.endswith("Aggressor_IP")
    if g == "4bet_hu":
        return r.endswith("Aggressor_IP")
    return True  # conservative default

def _expected_root_token(menu_id: str, size_pct: int) -> str:
    prefix = "DONK" if _oop_is_caller(menu_id) else "BET"
    s = int(size_pct)
    # snap to the discrete vocab you use
    snap = min([25,33,50,66,75,100], key=lambda b: abs(b - s))
    return f"{prefix}_{snap}"

def _cols_for(prefix: str, cols: List[str]) -> List[str]:
    return [c for c in cols if c.startswith(prefix)]

def _sum_actions(row: pd.Series) -> float:
    return float(sum(float(row.get(a, 0.0) or 0.0) for a in ACTION_VOCAB))

def _has_mass(row: pd.Series, cols: List[str], tol: float) -> bool:
    return any(abs(float(row.get(c, 0.0) or 0.0)) > tol for c in cols)

def _leakage(row: pd.Series, forbidden: List[str], tol: float) -> Dict[str, float]:
    return {c: float(row.get(c, 0.0) or 0.0) for c in forbidden if abs(float(row.get(c, 0.0) or 0.0)) > tol}

def audit_root(df: pd.DataFrame, tol: float) -> List[Dict[str, str]]:
    issues = []
    cols = list(df.columns)
    bet_cols  = [c for c in cols if c.startswith("BET_")]
    donk_cols = [c for c in cols if c.startswith("DONK_")]

    forbidden = (
        ["FOLD","CALL"] +
        [c for c in cols if c.startswith("RAISE_")] +
        [c for c in bet_cols + donk_cols if c not in set(bet_cols + donk_cols)]
    )  # above line just illustrative; bet/donk handled explicitly

    for i, row in df.iterrows():
        row_id = f"root#{i}"
        problems = []

        # 1) flags/actor
        if int(row.get("facing_bet", 0)) != 0:
            problems.append("flag_facing_bet!=0")
        if str(row.get("actor","")).lower() != "oop":
            problems.append("actor!=oop")

        # 2) sum ≈ 1
        s = _sum_actions(row)
        if not (abs(s - 1.0) <= 5e-4):
            problems.append(f"sum!=1 ({s:.6f})")

        # 3) leakage of forbidden tokens
        forb = ["FOLD","CALL"] + [c for c in cols if c.startswith("RAISE_")]
        leak = _leakage(row, forb, tol)
        if leak:
            problems.append(f"forbidden_mass:{leak}")

        # 4) expected size token present & has mass
        size_pct = int(row.get("size_pct") or 0)
        menu_id  = str(row.get("bet_sizing_id") or "")
        expect_tok = _expected_root_token(menu_id, size_pct)
        if expect_tok not in cols:
            problems.append(f"missing_col:{expect_tok}")
        else:
            v = float(row.get(expect_tok, 0.0) or 0.0)
            if v <= tol:
                problems.append(f"no_mass_in_expected:{expect_tok}")

        # 5) only one bet/donk token should have mass (the expected one)
        dyn_tokens = [c for c in bet_cols + donk_cols if c in ACTION_VOCAB]
        mass_tokens = [c for c in dyn_tokens if float(row.get(c, 0.0) or 0.0) > tol]
        if len(mass_tokens) > 1:
            problems.append(f"multi_size_mass:{mass_tokens}")

        if problems:
            issues.append({
                "row": row_id,
                "s3_key": str(row.get("s3_key")),
                "menu": menu_id,
                "size_pct": str(size_pct),
                "problems": "|".join(problems),
            })
    return issues

def audit_facing(df: pd.DataFrame, tol: float) -> List[Dict[str, str]]:
    issues = []
    cols = list(df.columns)
    raise_cols = [c for c in cols if c.startswith("RAISE_")]
    forbidden = ["CHECK"] + [c for c in cols if c.startswith("BET_") or c.startswith("DONK_")]

    for i, row in df.iterrows():
        row_id = f"facing#{i}"
        problems = []

        # 1) flags/actor
        if int(row.get("facing_bet", 0)) != 1:
            problems.append("flag_facing_bet!=1")
        # actor can be "ip" (OOP donk) or "oop" (IP c-bet) — we accept either

        # 2) sum ≈ 1
        s = _sum_actions(row)
        if not (abs(s - 1.0) <= 5e-4):
            problems.append(f"sum!=1 ({s:.6f})")

        # 3) leakage of root-only tokens
        leak = _leakage(row, forbidden, tol)
        if leak:
            problems.append(f"forbidden_mass:{leak}")

        # 4) must have some mass in allowed response space
        allowed_cols = ["FOLD","CALL","ALLIN"] + raise_cols
        if not _has_mass(row, allowed_cols, tol):
            problems.append("no_mass_in_allowed")

        if problems:
            issues.append({
                "row": row_id,
                "s3_key": str(row.get("s3_key")),
                "menu": str(row.get("bet_sizing_id") or ""),
                "faced_size_pct": str(int(row.get("faced_size_pct") or 0)),
                "problems": "|".join(problems),
            })
    return issues

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--facing", required=True)
    ap.add_argument("--out", default="data/diagnostics/postflop_parts_audit.csv")
    ap.add_argument("--tol", type=float, default=1e-7)
    args = ap.parse_args()

    root_path = Path(args.root)
    facing_path = Path(args.facing)
    df_root = pd.read_parquet(root_path)
    df_facing = pd.read_parquet(facing_path)

    root_issues = audit_root(df_root, args.tol)
    facing_issues = audit_facing(df_facing, args.tol)

    print(f"Root rows:   {len(df_root)}")
    print(f"Facing rows: {len(df_facing)}")
    print(f"Root issues:   {len(root_issues)}")
    print(f"Facing issues: {len(facing_issues)}")

    out_rows = []
    out_rows.extend({"split":"root", **r} for r in root_issues)
    out_rows.extend({"split":"facing", **r} for r in facing_issues)
    out_df = pd.DataFrame(out_rows)
    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_file, index=False)
    print(f"🧪 Audit written → {out_file}")

    # Quick summaries
    if root_issues:
        print("Sample root issues:")
        print(pd.DataFrame(root_issues).head(8).to_string(index=False))
    if facing_issues:
        print("Sample facing issues:")
        print(pd.DataFrame(facing_issues).head(8).to_string(index=False))

if __name__ == "__main__":
    main()