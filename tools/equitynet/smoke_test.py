import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.datasets.equitynet import EquityDatasetParquet, equity_collate_fn
from ml.models.equity_net import EquityNetLit


def _find_ckpt(ckpt_dir: Path) -> Path:
    """Choose best.ckpt > last.ckpt > any *.ckpt (lexicographically last)."""
    best = ckpt_dir / "best.ckpt"
    if best.exists():
        return best
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    cks = sorted([Path(p) for p in glob.glob(str(ckpt_dir / "*.ckpt"))])
    if not cks:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    return cks[-1]


def _sidecar_for(ckpt_path: Path) -> Path:
    """Sidecar sits next to checkpoint with suffix .sidecar.json."""
    sc = ckpt_path.with_suffix(ckpt_path.suffix + ".sidecar.json")
    if not sc.exists():
        raise FileNotFoundError(f"Sidecar not found: {sc}")
    return sc


def _sample_indices(n: int, k: int, seed: int) -> List[int]:
    k = min(k, n)
    rnd = random.Random(seed)
    idxs = list(range(n))
    rnd.shuffle(idxs)
    return idxs[:k]


@torch.no_grad()
def run_smoke(
    parquet_path: Path,
    ckpt_path: Path,
    sidecar_path: Path,
    k: int = 10,
    seed: int = 42,
    batch_size: int = 256,
) -> None:
    # ---- Load sidecar for feature_order (and cards if needed) ----
    sc = json.loads(Path(sidecar_path).read_text())
    feature_order = sc.get("feature_order", [])
    if not feature_order:
        raise RuntimeError("Sidecar missing feature_order; cannot construct dataset.")

    # ---- Dataset (uses sidecar-provided feature_order) ----
    y_cols = ["p_win", "p_tie", "p_lose"]
    ds = EquityDatasetParquet(
        parquet_path=str(parquet_path),
        x_cols=feature_order,
        y_cols=y_cols,
        weight_col="weight",
        device=None,
    )
    if len(ds) == 0:
        raise RuntimeError(f"Parquet has 0 rows: {parquet_path}")

    # pick small random subset
    pick = _sample_indices(len(ds), k, seed)
    subset = Subset(ds, pick)

    dl = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=equity_collate_fn,
        pin_memory=True,
        num_workers=0,
    )

    device = torch.device("cpu")
    model = EquityNetLit.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.to(device).eval()

    print(f"🔧 Using checkpoint: {ckpt_path}")
    print(f"🧩 Sidecar:         {sidecar_path}")
    print(f"📦 Parquet:         {parquet_path}")
    print(f"🧪 Sampling {len(pick)} row(s) …\n")

    shown = 0
    t0 = time.time()
    for x_dict, y, w in dl:
        # EquityNetLit.forward should accept x_dict only and return [B,3] logits
        logits = model(x_dict)
        p = torch.softmax(logits, dim=-1)  # [B,3]

        p_np = p.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        # pull a couple of feature columns to display if present
        # (IDs, because we don’t reverse-map in this smoke test)
        def _maybe(col: str):
            t = x_dict.get(col)
            return t.detach().cpu().numpy() if t is not None else None

        stack_ids = _maybe(feature_order[0]) if feature_order else None

        for i in range(p_np.shape[0]):
            shown += 1
            probs = p_np[i].tolist()
            tgt = y_np[i].tolist()

            print("—" * 56)
            meta_bits = []
            if stack_ids is not None:
                meta_bits.append(f"{feature_order[0]}={int(stack_ids[i])}")
            print(f"[{shown:03d}] " + " ".join(meta_bits))
            print(" pred probs :", ", ".join(f"{k}={probs[j]:.3f}" for j, k in enumerate(["p_win","p_tie","p_lose"])))
            print(" target     :", ", ".join(f"{k}={tgt[j]:.3f}"  for j, k in enumerate(["p_win","p_tie","p_lose"])))
            print(" sum(pred)  :", f"{sum(probs):.3f}")

    dt = time.time() - t0
    print(f"\n✅ Done. Inferred {shown} rows in {dt:.3f}s")


def main():
    ap = argparse.ArgumentParser(description="EquityNet smoke test (quick inference sanity check)")
    ap.add_argument("--parquet", type=str, default="data/datasets/equitynet.parquet",
                    help="path to equity dataset parquet")
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints/equitynet/dev",
                    help="directory containing best.ckpt/last.ckpt")
    ap.add_argument("--ckpt", type=str, default=None, help="override checkpoint path directly")
    ap.add_argument("--k", type=int, default=10, help="number of random rows to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    ckpt_path = Path(args.ckpt) if args.ckpt else _find_ckpt(Path(args.ckpt_dir))
    sidecar_path = _sidecar_for(ckpt_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    run_smoke(
        parquet_path=parquet_path,
        ckpt_path=ckpt_path,
        sidecar_path=sidecar_path,
        k=args.k,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()