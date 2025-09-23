#!/usr/bin/env python
import argparse, json, shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.datasets.preflop_rangenet import PreflopRangeDatasetParquet
from ml.models.preflop_rangenet import RangeNetLit
from ml.utils.rangenet_preflop_sidecar import write_rangenet_preflop_sidecar


def list_candidate_ckpts(ckpts_dir: Path):
    """All .ckpt files except 'last' and 'best'."""
    return sorted(p for p in ckpts_dir.glob("*.ckpt")
                  if p.name not in ("last.ckpt", "best.ckpt"))


def pick_best_by_filename(ckpts):
    """
    Minimalistic: parse KL from filename.
    Format is rangenet-preflop-{epoch}-{val_loss:.4f}.ckpt
    """
    best_p, best_val = None, float("inf")
    for p in ckpts:
        try:
            parts = p.stem.split("-")
            metric = float(parts[-1])  # last part is the val_loss/kl
        except Exception:
            continue
        if metric < best_val:
            best_val = metric
            best_p = p
    return best_p, best_val


def main():
    ap = argparse.ArgumentParser(
        description="Pick best RangeNet-Preflop checkpoint by filename metric and write sidecar"
    )
    ap.add_argument("--ckpts-dir", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, required=True,
                    help="Preflop parquet with 169-dim soft labels")
    args = ap.parse_args()

    ckpts = list_candidate_ckpts(args.ckpts_dir)
    if not ckpts:
        raise SystemExit(f"No candidate ckpts in {args.ckpts_dir}")

    best_ckpt, best_val = pick_best_by_filename(ckpts)
    if not best_ckpt:
        raise SystemExit("Could not parse any metrics from filenames")

    best_dst = args.ckpts_dir / "best.ckpt"
    shutil.copy2(best_ckpt, best_dst)

    # --- build dataset schema for sidecar ---
    ds = PreflopRangeDatasetParquet(str(args.parquet), device=None)
    cards = ds.cards()
    feature_order = list(ds.feature_order)

    # --- load model to capture hparams ---
    model = RangeNetLit.load_from_checkpoint(
        str(best_dst),
        map_location="cpu",
        cards=cards,
        feature_order=feature_order,
    ).eval()

    # --- write sidecar ---
    sidecar = write_rangenet_preflop_sidecar(
        best_ckpt=best_dst,
        ds=ds,
        model=model,
        model_name="RangeNetPreflop",
    )

    # --- meta info ---
    meta = {
        "chosen": best_ckpt.name,
        "filename_val": best_val,
        "parquet": str(args.parquet),
        "sidecar": str(sidecar),
    }
    (args.ckpts_dir / "best.meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n✅ best = {best_ckpt.name}")
    print(f"→ wrote {best_dst.name} and {best_dst.name}.sidecar.json")
    print(f"→ meta: {args.ckpts_dir/'best.meta.json'}")


if __name__ == "__main__":
    main()