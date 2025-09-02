#!/usr/bin/env python3
import argparse, json, time
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.inference.equity import EquityNetInfer
# These must exist in your repo:
# - EquityNetLit (the LightningModule)
# - EquityNetInfer (the inference wrapper that reads the sidecar encoders and feature_order)
from ml.models.equity_net import EquityNetLit  # noqa: F401 (loaded by Infer)


def _find_sidecar(ckpt: Path) -> Path:
    sc = ckpt.with_suffix(ckpt.suffix + ".sidecar.json")
    if not sc.exists():
        raise FileNotFoundError(f"Sidecar not found next to checkpoint: {sc}")
    return sc


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@torch.no_grad()
def run_contract(ckpt: Path, sidecar: Path, req_path: Path, batch_size: int = 256) -> None:
    infer = EquityNetInfer.from_checkpoint(
        checkpoint_path=str(ckpt),
        sidecar_path=str(sidecar),
        device="auto",
    )

    rows = _load_jsonl(req_path)
    if not rows:
        raise SystemExit(f"No requests in: {req_path}")

    print(f"🔧 Using checkpoint: {ckpt}")
    print(f"🧩 Sidecar:         {sidecar}")
    print(f"📥 Requests:        {req_path}\n")

    t0 = time.time()
    shown = 0
    # simple batching
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        p = infer.predict_proba(batch)  # [B,3] for equity: [p_win,p_tie,p_lose]
        p_np = p.detach().cpu().numpy()
        for j in range(p_np.shape[0]):
            shown += 1
            probs = p_np[j].tolist()
            print("—" * 56)
            print(f"[{shown:03d}]")
            print(f" p_win={probs[0]:.3f}  p_tie={probs[1]:.3f}  p_lose={probs[2]:.3f}")
            print(f" sum={sum(probs):.3f}")
    t1 = time.time()
    print(f"\n✅ Done. Inferred {shown} row(s) in {t1 - t0:.3f}s")


def main():
    ap = argparse.ArgumentParser(description="Contract-style EquityNet inference (no preprocessing).")
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint (expects sidecar next to it)")
    ap.add_argument("--sidecar", type=Path, default=None, help="Override sidecar path (default: <ckpt>.sidecar.json)")
    ap.add_argument("--requests", type=Path, default=Path("data/samples/equity_requests.jsonl"),
                    help="JSONL with request rows; keys must match sidecar feature_order")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    ckpt = args.ckpt
    sc = args.sidecar or _find_sidecar(ckpt)
    run_contract(ckpt=ckpt, sidecar=Path(sc), req_path=args.requests, batch_size=args.batch_size)


if __name__ == "__main__":
    main()