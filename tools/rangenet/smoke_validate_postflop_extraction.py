from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Mapping
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client
from ml.etl.rangenet.postflop.texas_solver_extractor import TexasSolverExtractor
from ml.utils.config import load_model_config
from ml.etl.rangenet.postflop.build_postflop_policy import (
    _normalize_s3_key,
    _get,
    ACTION_VOCAB, s3_key_for_size
)

def resolve_solver_path(cfg: Mapping[str, Any], s3_key: str, s3_client: Any) -> tuple[Path, bool]:
    """Mirror of your updated _resolve_solver_path; imported here to avoid importing whole builder."""
    import tempfile

    key = _normalize_s3_key(s3_key)

    # optional local override
    local_dir = _get(cfg, "solver.local_solver_dir", None)
    if local_dir:
        p = Path(local_dir) / Path(key)
        if p.is_file():
            return p, False

    # only prefix if not already present
    s3_prefix = (_get(cfg, "solver.s3_prefix", "") or "").strip("/")
    key_norm = key.lstrip("/")
    if s3_prefix and not key_norm.startswith(f"{s3_prefix}/") and key_norm != s3_prefix:
        key = f"{s3_prefix}/{key_norm}"
    else:
        key = key_norm

    cache_dir = Path(_get(cfg, "solver.local_cache_dir", tempfile.gettempdir()))
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(key).name
    tmp_path = Path(tempfile.mkstemp(prefix=".dl_", suffix=f".{filename}", dir=str(cache_dir))[1])
    try:
        s3_client.download_file(s3_key=key, local_path=tmp_path)
        return tmp_path, True
    except Exception as e:
        try: tmp_path.unlink(missing_ok=True)
        except Exception: pass
        raise RuntimeError(f"download failed for {key}: {e}") from e

def parse_bet_sizes_cell(cell) -> list[int]:
    """Robust parse; returns ordered unique integer percents."""
    from decimal import Decimal, ROUND_HALF_UP
    if cell is None:
        return []
    try:
        import pyarrow as pa
        if isinstance(cell, (pa.ListScalar, pa.ListValue, pa.ChunkedArray)):
            cell = cell.as_py()
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(cell, np.ndarray):
            cell = cell.tolist()
    except Exception:
        pass
    seq = cell if isinstance(cell, list) else [cell]
    out, seen = [], set()
    for it in seq:
        if it is None: continue
        v = it.get("element") if isinstance(it, dict) else it
        try:
            f = float(v)
        except Exception:
            continue
        if f <= 3.0:
            pct = int(Decimal(str(f * 100.0)).quantize(0, rounding=ROUND_HALF_UP))
        else:
            pct = int(Decimal(str(f)).quantize(0, rounding=ROUND_HALF_UP))
        if 1 <= pct <= 200 and pct not in seen:
            out.append(pct); seen.add(pct)
    return out

def validate_mix(mix: dict[str, float], vocab: list[str]) -> dict[str, Any]:
    """Return validation metrics for a policy vector. WHY: catch malformed solves early."""
    if not mix:
        return {"ok": False, "sum": 0.0, "nnz": 0, "argmax": None, "issues": ["empty"]}
    vec = np.array([float(mix.get(a, 0.0)) for a in vocab], dtype=np.float64)
    total = float(vec.sum())
    issues = []
    if not np.isfinite(total):
        issues.append("non_finite_sum")
    if total <= 0:
        issues.append("zero_mass")
    # numerical normalization check
    if abs(total - 1.0) > 1e-3:
        issues.append(f"sum_off:{total:.6f}")
    # boundedness
    if np.any(vec < -1e-8):
        issues.append("neg_prob")
    if np.any(vec > 1 + 1e-8):
        issues.append("prob_gt_1")
    nnz = int((vec > 1e-8).sum())
    argmax = vocab[int(vec.argmax())] if total > 0 else None
    return {"ok": len(issues) == 0, "sum": total, "nnz": nnz, "argmax": argmax, "issues": issues}

def probe_parquet(path: Path, expected: str) -> dict[str, Any]:
    """Quick integrity checks on built parquets. expected in {'root','facing'}."""
    df = pd.read_parquet(path)
    problems = []
    if expected == "root":
        problems += [] if (df["actor"] == "ip").all() else ["actor!=ip"]
        problems += [] if (df["facing_bet"] == 0).all() else ["facing_bet!=0"]
        problems += [] if df["size_pct"].notna().all() else ["size_pct nulls"]
        problems += [] if df["faced_size_pct"].isna().all() else ["faced_size_pct not null"]
    else:
        problems += [] if (df["actor"] == "oop").all() else ["actor!=oop"]
        problems += [] if (df["facing_bet"] == 1).all() else ["facing_bet!=1"]
        problems += [] if df["faced_size_pct"].notna().all() else ["faced_size_pct nulls"]
        problems += [] if df["size_pct"].isna().all() else ["size_pct not null"]
    return {"rows": len(df), "problems": problems, "sample": df.head(min(3, len(df))).to_dict(orient="records")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to parquet manifest")
    ap.add_argument("--cfg", type=str, default="rangenet/postflop")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--probe-root", type=str, default=None)
    ap.add_argument("--probe-facing", type=str, default=None)
    ap.add_argument("--out", type=str, default="data/diagnostics/postflop_extraction_check.csv")
    args = ap.parse_args()

    cfg = load_model_config(args.cfg) or {}
    df = pd.read_parquet(args.manifest)
    if args.limit is not None:
        df = df.head(int(args.limit))

    s3c = S3Client()
    x = TexasSolverExtractor()

    out_rows = []
    total_items = 0

    for _, r in df.iterrows():
        base_key = _normalize_s3_key(r.get("s3_key", ""))
        sizes_pct = parse_bet_sizes_cell(r.get("bet_sizes"))
        if not sizes_pct:
            sizes_pct = [33]

        # Extract row context (WHY: pass the exact same context to extractor as training)
        positions = str(r.get("positions") or "")
        ip_pos, oop_pos = positions.split("v") if "v" in positions else ("IP", "OOP")
        ctx = str(r.get("ctx") or "")
        board = str(r.get("board") or "")
        pot_bb = float(r.get("pot_bb") or 0.0)
        stack_bb = float(r.get("effective_stack_bb") or 0.0)
        menu_id = str(r.get("bet_sizing_id") or "")

        for size in sizes_pct:
            s3_key_sz = s3_key_for_size(base_key, int(size))
            path, eph = resolve_solver_path(cfg, s3_key_sz, s3c)
            try:
                ex = x.extract(
                    str(path),
                    ctx=ctx,
                    ip_pos=ip_pos,
                    oop_pos=oop_pos,
                    board=board,
                    pot_bb=pot_bb,
                    stack_bb=stack_bb,
                    bet_sizing_id=menu_id,
                    size_pct=size,
                    root_actor="oop"
                )
                rm = validate_mix(ex.root_mix, ACTION_VOCAB)
                fm = validate_mix(ex.facing_mix, ACTION_VOCAB)

                out_rows.append({
                    "sha1": str(r.get("sha1"))[:10],
                    "size_pct": int(size),
                    "s3_key": s3_key_sz,
                    "root_ok": rm["ok"],
                    "root_sum": rm["sum"],
                    "root_nnz": rm["nnz"],
                    "root_argmax": rm["argmax"],
                    "root_issues": ";".join(rm["issues"]),
                    "facing_ok": fm["ok"],
                    "facing_sum": fm["sum"],
                    "facing_nnz": fm["nnz"],
                    "facing_argmax": fm["argmax"],
                    "facing_issues": ";".join(fm["issues"]),
                })
                total_items += 1
            finally:
                if eph:
                    try: os.unlink(path)
                    except Exception: pass

    # Emit console summary
    ok_root = sum(1 for r in out_rows if r["root_ok"])
    ok_facing = sum(1 for r in out_rows if r["facing_ok"])
    print(f"Checked {total_items} items  |  root_ok={ok_root}  facing_ok={ok_facing}")

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            w.writeheader()
            w.writerows(out_rows)
    print(f"Report → {out_path}")

    # Optional parquet probes
    if args.probe_root:
        res = probe_parquet(Path(args.probe_root), "root")
        print(f"[probe root] rows={res['rows']} problems={res['problems']}")
        if res["problems"]:
            print(" sample:", res["sample"])
    if args.probe_facing:
        res = probe_parquet(Path(args.probe_facing), "facing")
        print(f"[probe facing] rows={res['rows']} problems={res['problems']}")
        if res["problems"]:
            print(" sample:", res["sample"])

if __name__ == "__main__":
    main()