import argparse
import glob
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.datasets.population import PopulationDatasetParquet, population_collate_fn
from ml.models.population_net import PopulationNetLit

ACTIONS = ["FOLD", "CALL", "RAISE"]


def _find_ckpt(ckpt_dir: Path) -> Path:
    """
    Choose best.ckpt > last.ckpt > any *.ckpt (lexicographically last).
    """
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
    k: int = 10,
    seed: int = 42,
    batch_size: int = 256,
) -> None:
    # Dataset: use SOFT labels – PopulationNet is trained on soft labels
    ds = PopulationDatasetParquet(str(parquet_path), use_soft_labels=True, device=None)
    if len(ds) == 0:
        raise RuntimeError(f"Parquet has 0 rows: {parquet_path}")

    # Pick a few random rows
    pick = _sample_indices(len(ds), k, seed)
    subset = Subset(ds, pick)

    dl = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=population_collate_fn,
        pin_memory=True,
        num_workers=0,
    )

    device = torch.device("cpu")
    model = PopulationNetLit.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.to(device).eval()

    print(f"🔧 Using checkpoint: {ckpt_path}")
    print(f"📦 Parquet: {parquet_path}")
    print(f"🧪 Sampling {len(pick)} row(s) …\n")

    t0 = time.time()
    shown = 0
    for x_dict, y_soft, w in dl:
        logits = model(x_dict)
        p = torch.softmax(logits, dim=-1)  # [B,3]

        # Move to CPU numpy for pretty printing
        p_np = p.detach().cpu().numpy()
        y_np = y_soft.detach().cpu().numpy()
        w_np = w.detach().cpu().numpy()

        # Pull back some input columns for context
        ctx_np = x_dict["ctx_id"].detach().cpu().numpy()
        street_np = x_dict["street_id"].detach().cpu().numpy()
        hero_np = x_dict["hero_pos_id"].detach().cpu().numpy()
        vill_np = x_dict["villain_pos_id"].detach().cpu().numpy()
        stakes_np = x_dict["stakes_id"].detach().cpu().numpy()

        for i in range(p_np.shape[0]):
            shown += 1
            probs = p_np[i].tolist()
            tgt = y_np[i].tolist()  # soft target
            print(f"—" * 56)
            print(
                f"[{shown:03d}] stake={int(stakes_np[i])} "
                f"street={int(street_np[i])} ctx={int(ctx_np[i])} "
                f"hero_pos={int(hero_np[i])} villain_pos={int(vill_np[i])}"
            )
            print(" pred probs :", ", ".join(f"{a}={probs[j]:.3f}" for j, a in enumerate(ACTIONS)))
            print(" target soft:", ", ".join(f"{a}={tgt[j]:.3f}" for j, a in enumerate(ACTIONS)))
            print(" sum(pred)  :", f"{sum(probs):.3f}")
    t1 = time.time()
    print(f"\n✅ Done. Inferred {shown} rows in {t1 - t0:.3f}s")


def main():
    ap = argparse.ArgumentParser(description="PopulationNet smoke test (quick inference sanity check)")
    ap.add_argument("--stake", type=int, default=10, help="e.g. 10 ⇒ nl10 parquet default path")
    ap.add_argument("--parquet", type=str, default=None,
                    help="override parquet path (default: data/datasets/populationnet_nl{stake}_dev.parquet)")
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints/popnet/dev",
                    help="directory containing best.ckpt/last.ckpt (default: checkpoints/populationnet/dev)")
    ap.add_argument("--ckpt", type=str, default=None, help="override checkpoint path directly")
    ap.add_argument("--k", type=int, default=10, help="number of random rows to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    parquet_path = Path(args.parquet or f"data/datasets/populationnet_nl{args.stake}_dev.parquet")
    ckpt_path = Path(args.ckpt) if args.ckpt else _find_ckpt(Path(args.ckpt_dir))

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    run_smoke(
        parquet_path=parquet_path,
        ckpt_path=ckpt_path,
        k=args.k,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()