# tools/rangenet/postflop/sweep_pick_best.py
import argparse, shutil, json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.models.preflop_rangenet import RangeNetLit
from ml.utils.rangenet_postflop_sidecar import write_rangenet_postflop_sidecar


def list_candidate_ckpts(ckpts_dir: Path) -> List[Path]:
    cks = sorted(ckpts_dir.glob("*.ckpt"))
    return [p for p in cks if p.name not in ("last.ckpt", "best.ckpt")]

def parse_ckpt_metric(p: Path) -> float:
    """
    Expects filename like rangenet_postflop-14-0.0623.ckpt
    Returns metric part as float (here: 0.0623).
    """
    name = p.stem
    parts = name.split("-")
    try:
        return float(parts[-1])
    except Exception:
        raise ValueError(f"Cannot parse metric from {p.name}")

def pick_best(ckpts: List[Path]) -> Tuple[int, Path, float]:
    best_i, best_val = None, float("inf")
    for i, p in enumerate(ckpts):
        val = parse_ckpt_metric(p)
        if val < best_val:
            best_val = val
            best_i = i
    return best_i, ckpts[best_i], best_val

def main():
    ap = argparse.ArgumentParser(description="Pick best RangeNet-Postflop ckpt (by filename metric) and write sidecar")
    ap.add_argument("--ckpts-dir", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, required=True, help="postflop parquet (for schema discovery)")
    args = ap.parse_args()

    ckpts = list_candidate_ckpts(args.ckpts_dir)
    if not ckpts:
        raise SystemExit(f"No candidate ckpts in {args.ckpts_dir}")

    idx, best_src, best_val = pick_best(ckpts)
    best_dst = args.ckpts_dir / "best.ckpt"
    shutil.copy2(best_src, best_dst)

    # Write sidecar (schema discovery from dataset)
    ds = PostflopRangeDatasetParquet(str(args.parquet), device=None)
    model = RangeNetLit.load_from_checkpoint(
        str(best_dst),
        map_location="cpu",
        cards=ds.cards_info.cards,
        feature_order=list(ds.feature_order),
    ).eval()

    sidecar = write_rangenet_postflop_sidecar(
        best_ckpt=best_dst,
        ds=ds,
        model=model,
        model_name="RangeNetPostflop",
    )

    meta = {
        "chosen": best_src.name,
        "val_kl": best_val,
        "parquet": str(args.parquet),
        "sidecar": str(sidecar),
    }
    (args.ckpts_dir / "best.meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n✅ best = {best_src.name}")
    print(f"→ wrote {best_dst.name} and {best_dst.name}.sidecar.json")
    print(f"→ meta: {args.ckpts_dir/'best.meta.json'}")

if __name__ == "__main__":
    main()