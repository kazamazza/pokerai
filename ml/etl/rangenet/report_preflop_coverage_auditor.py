# tools/rangenet/report_preflop_coverage_auditor.py
import argparse, json, math
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

# --- helper tokens (raw, from filename stems) ---
RAISEY_RAW = {"Min", "AI", "3sb"}  # vendor "raise-ish" tokens
CALL_RAW   = {"Call"}
FOLD_RAW   = {"Fold"}

def is_raise_raw(a: Optional[str]) -> bool:
    if not a: return False
    return a in RAISEY_RAW or a.endswith("%")  # e.g. "60%"

def is_call_raw(a: Optional[str]) -> bool:
    return (a or "") in CALL_RAW

def is_fold_raw(a: Optional[str]) -> bool:
    return (a or "") in FOLD_RAW

def first_non_fold_raise(seq: List[dict]) -> Tuple[Optional[str], Optional[str], int]:
    """
    Return (pos, action, index) of first non-fold *raise-ish* action in seq, else (None,None,-1).
    """
    for i, e in enumerate(seq):
        pos = e.get("pos")
        act = e.get("action")
        if is_fold_raw(act):
            continue
        if is_raise_raw(act):
            return pos, act, i
    return None, None, -1

def first_action_of(seq: List[dict], target_pos: str) -> Optional[str]:
    for e in seq:
        if e.get("pos") == target_pos and "action" in e:
            return e["action"]
    return None

def any_raise_before(seq: List[dict], stop_pos: str) -> bool:
    """
    True if any raise-ish appears before stop_pos first acts.
    """
    for e in seq:
        p = e.get("pos")
        a = e.get("action")
        if p == stop_pos:
            return False
        if is_raise_raw(a):
            return True
    return False

def count_limpers_before_first_raise(seq: List[dict]) -> int:
    """
    Count "Limp" tokens before the first raise-ish appears. If no raise-ish at all, count all limps.
    """
    limps = 0
    for e in seq:
        a = e.get("action")
        if a == "Limp":
            limps += 1
        elif is_raise_raw(a):
            break
    return limps

def supports_srp_open_call(seq: List[dict], ip: str, oop: str) -> bool:
    """
    SRP = opener (ip) raises; defender (oop) calls first; and no re-raise before defender acts.
    """
    open_pos, open_act, _ = first_non_fold_raise(seq)
    if open_pos != ip:
        return False
    a_opp = first_action_of(seq, oop)
    if a_opp is None:
        return False
    if any_raise_before(seq, oop):
        return False
    return is_call_raw(a_opp)

def supports_3bet_pot(seq: List[dict], opener: str, defender: str) -> bool:
    """
    Heuristic: opener raises first; defender's *first* action is raise-ish (3bet).
    (We do not require seeing a call back — this is a preflop-range bucket.)
    """
    open_pos, _, _ = first_non_fold_raise(seq)
    if open_pos != opener:
        return False
    a_def = first_action_of(seq, defender)
    return is_raise_raw(a_def)

def supports_4bet_pot(seq: List[dict], opener: str, defender: str) -> bool:
    """
    Heuristic: opener raises, defender raises (3bet), opener raises again (4bet) before defender acts again.
    """
    open_pos, _, open_idx = first_non_fold_raise(seq)
    if open_pos != opener:
        return False

    # find defender first action after open_idx
    def_first_idx = None
    for i in range(open_idx + 1, len(seq)):
        e = seq[i]
        if e.get("pos") == defender:
            def_first_idx = i
            break
    if def_first_idx is None:
        return False
    if not is_raise_raw(seq[def_first_idx].get("action")):
        return False  # not a 3bet

    # opener raises again (4bet) before defender acts second time
    for j in range(def_first_idx + 1, len(seq)):
        e = seq[j]
        if e.get("pos") == opener:
            return is_raise_raw(e.get("action"))
        if e.get("pos") == defender:
            break
    return False

def classify_contexts_for_pair(seq: List[dict], ip: str, oop: str) -> Set[str]:
    """
    Return a set of contexts this sequence supports for (ip, oop):
      - "SRP"
      - "3BET"
      - "4BET"
      - "LIMPED_SINGLE"
      - "LIMPED_MULTI"
      - "STEAL" (BTN/CO open vs SB/BB)
    You can extend this later.
    """
    ctx: Set[str] = set()

    # limped buckets
    limp_count = count_limpers_before_first_raise(seq)
    if limp_count == 1:
        ctx.add("LIMPED_SINGLE")
    elif limp_count >= 2:
        ctx.add("LIMPED_MULTI")

    # SRP
    if supports_srp_open_call(seq, ip, oop):
        ctx.add("SRP")

    # 3bet/4bet (opener=ip, defender=oop)
    if supports_3bet_pot(seq, ip, oop):
        ctx.add("3BET")
        if supports_4bet_pot(seq, ip, oop):
            ctx.add("4BET")

    # Blind-vs-steal flag (only as a label if opener is BTN/CO and defender SB/BB)
    # This is *not* a separate tree type — just a tag you can query for coverage.
    open_pos, _, _ = first_non_fold_raise(seq)
    if open_pos in {"BTN", "CO"} and oop in {"SB", "BB"} and supports_srp_open_call(seq, open_pos, oop):
        ctx.add("STEAL")

    return ctx

def load_cfg(config_path: str):
    # import here to avoid hard deps if you run standalone
    from ml.utils.config import load_model_config
    return load_model_config(config_path)

def nearest(x: float, options: List[int]) -> int:
    return int(min(options, key=lambda s: (abs(s - float(x)), s)))

def build_availability(df: pd.DataFrame) -> Tuple[Set[int], Dict[Tuple[int,str,str,str], Set[str]]]:
    """
    Returns:
      stacks: set of available stacks
      avail: (stack, hero_pos, ip, oop) -> set(contexts)
    """
    stacks = set(int(s) for s in df["stack_bb"].dropna().unique().tolist())
    avail: Dict[Tuple[int,str,str,str], Set[str]] = {}

    for _, r in df.iterrows():
        try:
            seq = json.loads(r["sequence_raw"])
        except Exception:
            continue
        hero = r.get("hero_pos")
        stack = r.get("stack_bb")
        if not hero or pd.isna(stack):
            continue
        stack = int(stack)

        # Collect all seat labels in this sequence
        seen_positions = [e.get("pos") for e in seq if e.get("pos")]
        seen_positions = [p for p in seen_positions if isinstance(p, str)]

        # For every possible (ip, oop) within this filename, accumulate contexts covered *from this hero POV*
        for ip in seen_positions:
            for oop in seen_positions:
                if ip == oop:
                    continue
                ctxs = classify_contexts_for_pair(seq, ip, oop)
                if not ctxs:
                    continue
                key = (stack, hero, ip, oop)
                if key not in avail:
                    avail[key] = set()
                avail[key].update(ctxs)

    return stacks, avail

def main():
    ap = argparse.ArgumentParser(description="Preflop coverage auditor (Monker manifest → missing targets list)")
    ap.add_argument("--config", default="rangenet/postflop", help="model[/variant]/profile")
    ap.add_argument("--manifest", default="data/artifacts/monker_manifest.parquet")
    ap.add_argument("--contexts", nargs="*", default=["SRP","3BET","4BET","LIMPED_SINGLE","LIMPED_MULTI"],
                    help="Context buckets to require")
    ap.add_argument("--out-missing", default="data/artifacts/preflop_missing_targets.csv",
                    help="CSV with rows to solve manually")
    args = ap.parse_args()

    # Load config grid
    cfg = load_cfg(args.config)
    mb = cfg.get("manifest_build", {}) or {}
    req_stacks: List[int] = [int(x) for x in mb.get("stacks_bb", [])]
    req_pairs: List[Tuple[str,str]] = [tuple(p) for p in mb.get("position_pairs", [])]
    req_contexts: List[str] = [c.upper() for c in args.contexts]

    if not req_stacks or not req_pairs:
        raise SystemExit("manifest_build.stacks_bb and position_pairs are required in config")

    # Load richer Monker manifest
    df = pd.read_parquet(args.manifest)
    need = {"stack_bb","hero_pos","sequence_raw","rel_path","abs_path"}
    missing_cols = [c for c in need if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"monker_manifest missing columns: {missing_cols}")

    # Build availability index
    stacks_available, avail = build_availability(df)

    # For coverage, we require: for each target (stack, ip, oop, ctx),
    # both hero POVs exist (hero=ip and hero=oop) at the same stack.
    rows = []
    missing_rows = []
    for stack in req_stacks:
        for ip, oop in req_pairs:
            for ctx in req_contexts:
                key_ip  = (stack, ip, ip, oop)
                key_oop = (stack, oop, ip, oop)
                have_ip_ctx  = ctx in (avail.get(key_ip)  or set())
                have_oop_ctx = ctx in (avail.get(key_oop) or set())
                if have_ip_ctx and have_oop_ctx:
                    cls = "exact"
                    src_ip_stack = src_oop_stack = stack
                    delta_ip = delta_oop = 0
                else:
                    # search nearest stacks to see what's missing
                    ordered = sorted(stacks_available, key=lambda s: (abs(s - stack), s))
                    src_ip_stack = next((s for s in ordered if ctx in (avail.get((s, ip,  ip, oop)) or set())), None)
                    src_oop_stack= next((s for s in ordered if ctx in (avail.get((s, oop, ip, oop)) or set())), None)
                    have_ip_ctx_n  = src_ip_stack is not None
                    have_oop_ctx_n = src_oop_stack is not None

                    if have_ip_ctx_n and have_oop_ctx_n:
                        cls = "nearest_stack"
                        delta_ip  = abs(src_ip_stack  - stack)
                        delta_oop = abs(src_oop_stack - stack)
                    else:
                        cls = "missing"
                        delta_ip  = abs(src_ip_stack  - stack) if src_ip_stack  is not None else None
                        delta_oop = abs(src_oop_stack - stack) if src_oop_stack is not None else None

                        missing_rows.append({
                            "stack_bb": stack,
                            "ip_pos": ip,
                            "oop_pos": oop,
                            "ctx": ctx,
                            "have_ip_any_stack": bool(src_ip_stack is not None),
                            "have_oop_any_stack": bool(src_oop_stack is not None),
                            "nearest_ip_stack": src_ip_stack,
                            "nearest_oop_stack": src_oop_stack,
                        })

                rows.append({
                    "pair": f"{ip}v{oop}",
                    "stack": stack,
                    "ctx": ctx,
                    "class": cls,
                    "range_ip_source_stack": src_ip_stack if cls != "exact" else stack,
                    "range_oop_source_stack": src_oop_stack if cls != "exact" else stack,
                })

    cov = pd.DataFrame(rows)

    # Pivot: for each pair×stack show worst class over contexts (missing > nearest_stack > exact)
    # so you quickly see where you’re thin.
    order = {"missing": 2, "nearest_stack": 1, "exact": 0}
    def worst_class(vals):
        # choose the "worst" over multiple contexts
        if not len(vals): return ""
        best = max(vals, key=lambda v: order.get(v, 3))
        return best

    print("\nCoverage pivot (worst class per pair × stack across requested contexts):")
    if not cov.empty:
        pivot = pd.pivot_table(
            cov, index="pair", columns="stack", values="class",
            aggfunc=worst_class
        )
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        print(pivot.fillna("").to_string())
    else:
        print("(no rows)")

    print("\nTotals (by class across all contexts):")
    if not cov.empty:
        print(cov["class"].value_counts().to_string())
    else:
        print("no data")

    # Write missing targets CSV (to-solve list)
    out_path = Path(args.out_missing)
    if missing_rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(missing_rows).to_csv(out_path, index=False)
        print(f"\n📝 wrote missing-targets CSV → {out_path}  (rows={len(missing_rows)})")
    else:
        print("\n🎉 No missing targets for the requested contexts.")

if __name__ == "__main__":
    main()