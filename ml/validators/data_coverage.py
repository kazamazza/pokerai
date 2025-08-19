# ml/validators/data_coverage.py
from __future__ import annotations
import argparse, gzip, json, math, os, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---------- utils ----------
def open_fn(p: Path):
    return gzip.open if str(p).endswith(".gz") else open

def iter_rows(path: Path) -> Iterable[dict]:
    op = open_fn(path)
    with op(path, "rt", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    yield json.loads(ln)
                except Exception:
                    continue

def bucketize(val: float, edges: List[float]) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)): return "NA"
    try: x = float(val)
    except Exception: return "NA"
    # edges like [0.33, 0.5, 0.75, 1.0, 1.5, 99.0]
    prev = 0.0
    for e in edges:
        if x <= e: return f"{prev:.2f}–{e:.2f}"
        prev = e
    return f">{edges[-1]:.2f}"

def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0

# ---------- adapters (how to read each dataset type) ----------
def adapt_population(row: dict) -> Dict[str, Any]:
    # expects PopNetSample or hand-expanded decision row
    x = row.get("x") or row
    return {
        "street": x.get("street"),
        "stake": x.get("stake_tag") or x.get("stakes"),
        "pos": x.get("actor_pos"),
        "players": x.get("players", 6),
        "amt_to_call": x.get("amount_to_call_bb", 0.0),
        "pot": x.get("pot_bb", 0.0),
        "eff": x.get("effective_stack_bb", 0.0),
        "action": (row.get("y") or {}).get("action"),
    }

def adapt_exploit(row: dict) -> Dict[str, Any]:
    x = row.get("x") or row
    y = row.get("y") or {}
    return {
        "street": x.get("street"),
        "stake": x.get("stake_tag") or x.get("stakes"),
        "pos": x.get("actor_pos"),
        "vill_profile": x.get("vill_profile") or "UNKNOWN",
        "amt_to_call": x.get("amount_to_call_bb", 0.0),
        "pot": x.get("pot_bb", 0.0),
        "eff": x.get("effective_stack_bb", 0.0),
        "action": y.get("action"),
        "size_bucket": y.get("size_bucket", -1),
    }

def adapt_equity(row: dict) -> Dict[str, Any]:
    # keep it light; extend if you store more
    return {
        "street": row.get("street") or (row.get("x") or {}).get("street"),
        "stake": row.get("stake_tag") or (row.get("x") or {}).get("stake_tag"),
        "cluster": row.get("board_cluster_id") or (row.get("x") or {}).get("board_cluster_id"),
    }

def adapt_range(row: dict) -> Dict[str, Any]:
    # expect something like {pos, action_ctx, stack_bb, combo, weight}
    return {
        "pos": row.get("pos") or (row.get("x") or {}).get("pos"),
        "ctx": row.get("action_ctx") or (row.get("x") or {}).get("action_ctx"),
        "stack": row.get("stack_bb") or (row.get("x") or {}).get("stack_bb"),
        "bucket": row.get("bucket") or (row.get("x") or {}).get("bucket"),
    }

ADAPTERS = {
    "population": adapt_population,
    "exploit":    adapt_exploit,
    "equity":     adapt_equity,
    "range":      adapt_range,
}

# ---------- core coverage ----------
def coverage_report(
    path: Path,
    dtype: str,
    size_edges: List[float] | None = None,
    low_threshold: int = 200,     # warn below this
    joint_pairs: List[Tuple[str,str]] | None = None,
) -> Dict[str, Any]:
    adapt = ADAPTERS[dtype]
    N = 0
    H: Dict[str, Counter] = defaultdict(Counter)     # univariate histos
    J: Dict[Tuple[str,str], Counter] = defaultdict(Counter)  # pair histos

    # defaults per dataset
    if size_edges is None:
        size_edges = [0.0, 0.5, 1.0, 1.5, 2.0, 99.0]
    if joint_pairs is None:
        joint_pairs = {
            "population": [("street","pos"), ("street","action")],
            "exploit":    [("street","vill_profile"), ("street","action"), ("pos","vill_profile")],
            "equity":     [("street","cluster")],
            "range":      [("pos","ctx")],
        }.get(dtype, [])

    it = iter_rows(path)
    pbar = tqdm(it, unit="rows", disable=(tqdm is None))
    for r in pbar:
        N += 1
        z = adapt(r)

        # Univariates
        for k in ("street","stake","pos","players","vill_profile","action","size_bucket","cluster","ctx","bucket"):
            v = z.get(k, None)
            if v is not None:
                H[k][v] += 1

        # Numeric buckets (amount_to_call, pot, eff)
        if "amt_to_call" in z:
            H["amt_to_call_bucket"][bucketize(z["amt_to_call"], size_edges)] += 1
        if "pot" in z:
            H["pot_bucket"][bucketize(z["pot"], size_edges)] += 1
        if "eff" in z:
            H["eff_bucket"][bucketize(z["eff"], [10,20,40,80,200,10000])] += 1

        # Joints
        for a,b in joint_pairs:
            va, vb = z.get(a), z.get(b)
            if va is not None and vb is not None:
                J[(a,b)][(va,vb)] += 1

    # Warnings
    warnings = []
    for k, cnt in H.items():
        for v, c in cnt.items():
            if isinstance(v, str) and v == "NA": continue
            if c < low_threshold:
                warnings.append(f"LOW: {k}={v} -> {c} rows")

    # Format report
    def topk(counter: Counter, k=20):
        return sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:k]

    report = {
        "path": str(path),
        "dtype": dtype,
        "rows": N,
        "univariate": {k: {"unique": len(v), "top": topk(v, 20)} for k,v in H.items()},
        "joints": {
            f"{a}×{b}": {
                "unique": len(cnt),
                "sparse_bins_below_threshold": sum(1 for _,c in cnt.items() if c < low_threshold),
                "top": topk(cnt, 25),
            }
            for (a,b), cnt in J.items()
        },
        "warnings": warnings,
    }
    return report

def print_report(rep: Dict[str, Any]):
    print(f"\n=== Coverage: {rep['dtype']} ===")
    print(f"path: {rep['path']}")
    print(f"rows: {rep['rows']:,}")
    # quick univariates
    for k, info in rep["univariate"].items():
        if not info["unique"]: continue
        tops = ", ".join([f"{v}={c}" for (v,c) in info["top"][:6]])
        print(f"- {k}: unique={info['unique']} | top: {tops}")
    # warnings
    if rep["warnings"]:
        print("\nWarnings (low-sample bins):")
        for w in rep["warnings"][:50]:
            print("  •", w)
        if len(rep["warnings"]) > 50:
            print(f"  … +{len(rep['warnings'])-50} more")

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="One-shot data coverage checker for all NN datasets.")
    p.add_argument("--type", choices=list(ADAPTERS.keys()), required=True,
                   help="population | exploit | equity | range")
    p.add_argument("--data", required=True, help="path to JSONL(.gz)")
    p.add_argument("--low", type=int, default=200, help="warn if a bin has < LOW rows")
    p.add_argument("--edges", type=str, default="",
                   help="comma list for pot/amount buckets, e.g. 0.33,0.5,1.0,2.0,99.0")
    p.add_argument("--out", type=str, default="", help="optional path to write JSON report")
    args = p.parse_args()

    path = Path(args.data)
    edges = [float(x) for x in args.edges.split(",") if x] if args.edges else None

    rep = coverage_report(path, args.type, size_edges=edges, low_threshold=args.low)
    print_report(rep)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"\n✅ wrote JSON → {args.out}")

if __name__ == "__main__":
    main()