#!/usr/bin/env python3
import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

# canonical writing
def write_canonical_json(out_path: Path, stack_bb: int, ip_pos: str, oop_pos: str, ctx: str,
                         ip_169: np.ndarray, oop_169: np.ndarray):
    out = {
        "meta": {
            "source": "SPH",
            "version": "v2.0.9",
            "stack_bb": int(stack_bb),
            "ctx": str(ctx).upper(),
            "ip_pos": ip_pos,
            "oop_pos": oop_pos,
            "notes": "IP=open prob, OOP=defend prob (1-fold)"
        },
        "ip":  [float(x) for x in ip_169.tolist()],
        "oop": [float(x) for x in oop_169.tolist()],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

def _read_grid_any(path: Path) -> np.ndarray:
    """
    Accepts CSV of 13x13, or JSON with a 13x13 list under keys like 'grid','matrix','weights'.
    Returns a flat (169,) vector in row-major canonical order (assumes SPH already in canonical ordering).
    """
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, header=None)
        arr = df.to_numpy(dtype=float)
    else:
        obj = json.loads(path.read_text())
        # try common keys
        for k in ("grid", "matrix", "weights", "data", "range"):
            if k in obj:
                arr = np.array(obj[k], dtype=float)
                break
        else:
            # maybe it's already a 13x13 list or 169 list
            if isinstance(obj, list):
                arr = np.array(obj, dtype=float)
                if arr.size == 169:
                    return arr.reshape(13, 13).ravel()
                # else expect 13x13
                arr = arr
            else:
                raise ValueError(f"Unrecognized JSON structure in {path}")
    if arr.shape == (13, 13):
        return arr.reshape(13, 13).ravel()
    if arr.size == 169:
        return arr.ravel()
    raise ValueError(f"Expected 13x13 or 169, got {arr.shape} in {path}")

def main():
    ap = argparse.ArgumentParser(description="Pack SPH exports into canonical ip/oop JSON")
    ap.add_argument("--stack", type=int, required=True, help="e.g. 25")
    ap.add_argument("--ip", required=True, help="IP pos, e.g. UTG")
    ap.add_argument("--oop", required=True, help="OOP pos, e.g. BB")
    ap.add_argument("--ctx", default="SRP")
    ap.add_argument("--ip-open", type=Path, required=True, help="CSV/JSON open frequency grid for IP")
    # Choose one of the two OOP options:
    ap.add_argument("--oop-fold", type=Path, help="CSV/JSON fold grid (we do 1-fold)")
    ap.add_argument("--oop-call", type=Path, help="CSV/JSON call grid")
    ap.add_argument("--oop-raise", type=Path, nargs="*", default=[], help="CSV/JSON raise grids")
    ap.add_argument("--out", type=Path, required=True,
                    help="output canonical path, e.g. data/vendor_cache/sph/SRP/25/UTG_v_BB.json")
    args = ap.parse_args()

    ip_open = _read_grid_any(args.ip_open)
    # clamp to [0,1]
    ip_open = np.clip(ip_open, 0.0, 1.0)

    if args.oop_fold:
        fold = _read_grid_any(args.oop_fold)
        fold = np.clip(fold, 0.0, 1.0)
        oop_def = 1.0 - fold
    else:
        parts = []
        if args.oop_call:
            parts.append(np.clip(_read_grid_any(args.oop_call), 0.0, 1.0))
        for p in args.oop_raise:
            parts.append(np.clip(_read_grid_any(p), 0.0, 1.0))
        if not parts:
            raise SystemExit("Provide either --oop-fold OR (--oop-call and optional --oop-raise ...)")
        oop_def = np.sum(parts, axis=0)
        oop_def = np.clip(oop_def, 0.0, 1.0)

    write_canonical_json(
        out_path=args.out,
        stack_bb=args.stack,
        ip_pos=args.ip.upper(),
        oop_pos=args.oop.upper(),
        ctx=args.ctx.upper(),
        ip_169=ip_open,
        oop_169=oop_def,
    )

if __name__ == "__main__":
    main()