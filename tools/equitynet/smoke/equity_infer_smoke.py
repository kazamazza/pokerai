from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.inference.equity import EquityNetInfer

def main():
    ap = argparse.ArgumentParser("EquityNet inference sanity")
    ap.add_argument("--ckpt_dir", type=str, required=True,
                    help="Folder with best.ckpt/last.ckpt and sidecar (best_sidecar.json or sidecar.json)")
    ap.add_argument("--street", type=int, default=1, choices=[0, 1, 2, 3],
                    help="0=preflop, 1=flop, 2=turn, 3=river")
    ap.add_argument("--hand_id", type=int, default=42, help="0..168 (169-grid index)")
    ap.add_argument("--board_cluster_id", type=int, default=None,
                    help="Cluster id for postflop. If omitted and street>0, we try an UNK bucket.")
    ap.add_argument("--batch", type=int, default=3,
                    help="How many identical rows to query (quick batch test)")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")

    # 1) Load wrapper
    infer = EquityNetInfer.from_dir(ckpt_dir)

    # 2) Build rows (model expects categoricals listed in sidecar.feature_order)
    base_row = {"street": int(args.street), "hand_id": int(args.hand_id)}
    # board_cluster_id is optional for street==0; for postflop it helps to pass one
    if args.street > 0 and args.board_cluster_id is not None:
        base_row["board_cluster_id"] = int(args.board_cluster_id)

    rows = [dict(base_row) for _ in range(max(1, int(args.batch)))]

    # 3) Predict
    with torch.no_grad():
        probs = infer.predict(rows)  # -> List[List[3]]

    # 4) Pretty print
    print("✅ EquityNet inference sanity OK")
    print(f"ckpt_dir: {ckpt_dir}")
    print(f"street={args.street} hand_id={args.hand_id} board_cluster_id={args.board_cluster_id}")
    for i, p in enumerate(probs, 1):
        pw, pt, pl = [float(x) for x in p]
        print(f"[{i}] p_win={pw:.4f}  p_tie={pt:.4f}  p_lose={pl:.4f}  (sum={pw+pt+pl:.4f})")


if __name__ == "__main__":
    main()