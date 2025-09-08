#!/usr/bin/env python3
# tools/rangenet/sanity/abs_preview.py
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.range.solvers.utils.range_utils import hand_to_index, abs_text_to_vec169, vec169_to_monker_string
import numpy as np


def _load_abs_text(in_path: str | Path | None, text_arg: str | None) -> str:
    if text_arg is not None:
        return text_arg.strip()
    if not in_path:
        raise SystemExit("Provide either --in <file> or --text '<ABS strategy text>'")
    p = Path(in_path)
    if not p.exists():
        raise SystemExit(f"Input not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def main():
    ap = argparse.ArgumentParser(
        description="Preview/verify ABS strategy → (169,)+Monker conversion using your shared helpers."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in", dest="in_path", type=Path, help="File with raw ABS strategy text")
    g.add_argument("--text", type=str, help="Raw ABS strategy as a single string")

    ap.add_argument("--peek", nargs="*", default=["AA","AKs","AKo","T8s","22"],
                    help="Hands to inspect (compact codes)")
    ap.add_argument("--save-vec", type=Path, default=None,
                    help="Optional: write the 169-vector JSON here")
    ap.add_argument("--save-monker", type=Path, default=None,
                    help="Optional: write Monker string here (AA:...,AKs:...)")

    args = ap.parse_args()

    abs_text = _load_abs_text(args.in_path, args.text)

    # ✅ Convert using your canonical helper
    vec = abs_text_to_vec169(abs_text)   # expected np.ndarray shape (169,), dtype float, values 0..1
    if not isinstance(vec, np.ndarray) or vec.size != 169:
        raise SystemExit(f"abs_text_to_vec169 returned unexpected shape: {getattr(vec, 'shape', None)}")

    # Quick stats
    nz = int((vec > 0).sum())
    tot = float(vec.sum())
    print(f"[vec169] shape={vec.shape} nnz={nz} sum={tot:.2f} min={float(vec.min()):.6f} max={float(vec.max()):.6f}")

    # Peek specific hands
    if args.peek:
        print("[peek]")
        for h in args.peek:
            try:
                idx = hand_to_index(h)
                print(f"  {h:<4} → {float(vec[idx]):.6f}  (idx={idx})")
            except Exception as e:
                print(f"  {h:<4} → error: {e}")

    # Monker string (using your canonical helper)
    monker = vec169_to_monker_string(vec.tolist())
    # Show a preview (first ~120 chars)
    preview = monker[:120] + ("..." if len(monker) > 120 else "")
    print(f"[monker] {preview}")

    # Optional saves
    if args.save_vec:
        args.save_vec.parent.mkdir(parents=True, exist_ok=True)
        args.save_vec.write_text(json.dumps([float(x) for x in vec.tolist()], indent=2))
        print(f"💾 wrote vec169 JSON → {args.save_vec}")

    if args.save_monker:
        args.save_monker.parent.mkdir(parents=True, exist_ok=True)
        args.save_monker.write_text(monker, encoding="utf-8")
        print(f"💾 wrote Monker CSV → {args.save_monker}")


if __name__ == "__main__":
    main()