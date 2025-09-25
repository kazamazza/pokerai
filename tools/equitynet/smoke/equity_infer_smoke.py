from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.inference.equity import EquityNetInfer

STREET_MAP = {
    "preflop": 0, "0": 0,
    "flop": 1,    "1": 1,
    "turn": 2,    "2": 2,
    "river": 3,   "3": 3,
}

def main():
    ap = argparse.ArgumentParser("EquityNet smoke test")
    ap.add_argument("--ckpt", required=True, help="Path to EquityNet .ckpt")
    ap.add_argument("--sidecar", required=True, help="Path to .ckpt.sidecar.json")
    ap.add_argument("--street", required=True, help="preflop|flop|turn|river or 0|1|2|3")
    ap.add_argument("--hand-id", type=int, required=True, help="0..168")
    ap.add_argument("--cluster", type=int, default=None, help="board_cluster_id (omit for preflop)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    street = STREET_MAP.get(str(args.street).lower())
    if street is None:
        raise SystemExit(f"Unknown street: {args.street}")

    infer = EquityNetInfer.from_checkpoint(args.ckpt, args.sidecar, device=args.device)

    # Build the row the way the trainer expects
    row: Dict[str, Any] = {
        "street": street,
        "hand_id": int(args.hand_id),
    }
    if street > 0 and args.cluster is not None:
        row["board_cluster_id"] = int(args.cluster)

    p = infer.predict_one(row)
    print(f"row={row}")
    print(f"p_win={p[0]:.4f}  p_tie={p[1]:.4f}  p_lose={p[2]:.4f}  sum={sum(p):.6f}")

if __name__ == "__main__":
    main()