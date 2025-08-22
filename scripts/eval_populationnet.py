# scripts/eval_populationnet.py
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from ml.utils.checkpoints import pick_best_ckpt
from ml.eval.eval_populationnet import evaluate_populationnet, save_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, required=True, help="Path to population parquet")
    ap.add_argument("--ckpt", type=str, default=None, help="Explicit checkpoint path (.ckpt)")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints/popnet", help="Folder with checkpoints")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None, help="Write JSON report to this path")
    args = ap.parse_args()


    ckpt = args.ckpt or pick_best_ckpt(args.ckpt_dir)
    if not ckpt:
        raise SystemExit(f"No checkpoint found in {args.ckpt_dir} and no --ckpt given.")

    report = evaluate_populationnet(ckpt, args.parquet, batch_size=args.batch_size, seed=args.seed)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_report(report, args.out)
        print(f"✅ wrote report → {args.out}")
    else:
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()