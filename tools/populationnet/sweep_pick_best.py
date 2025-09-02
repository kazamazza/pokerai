# tools/populationnet/sweep_pick_best.py
import argparse, json, shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.utils.popnet_sidecar import write_popnet_sidecar
from ml.datasets.population import PopulationDatasetParquet, population_collate_fn
from ml.datasets.utils_dataset import stratified_indices
from ml.models.population_net import PopulationNetLit
from ml.utils.sidecar import save_sidecar_json, load_sidecar


@torch.no_grad()
def evaluate_populationnet(
    ckpt_path: str,
    parquet_path: str,
    batch_size: int = 1024,
    seed: int = 42,
) -> Dict[str, Any]:
    # ---- dataset ----
    ds = PopulationDatasetParquet(parquet_path, use_soft_labels=True, device=None)
    train_idx, val_idx = stratified_indices(ds.df, train_frac=0.8, seed=seed)
    val_dl = DataLoader(
        Subset(ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=population_collate_fn,
        pin_memory=True,
        num_workers=0,
    )

    # ---- model & sidecar ----
    device = torch.device("cpu")
    model = PopulationNetLit.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device).eval()

    sidecar_path = Path(ckpt_path).with_suffix(Path(ckpt_path).suffix + ".sidecar.json")
    sidecar = None
    feature_order = None
    if sidecar_path.exists():
        sidecar = load_sidecar(sidecar_path)
        feature_order = list(sidecar.get("feature_order", []) or None)

    total_w = 0.0
    sum_kl = 0.0
    sum_acc = 0.0
    from collections import defaultdict
    grp = defaultdict(lambda: {"w": 0.0, "kl": 0.0, "acc": 0.0})

    for x_dict, y_soft, w in val_dl:
        # If sidecar declares a feature order, align inputs accordingly
        if feature_order:
            x_dict = {k: x_dict[k] for k in feature_order if k in x_dict}

        logits = model(x_dict)
        p = torch.softmax(logits, dim=-1)

        eps = 1e-12
        y_clamped = (y_soft + eps) / (y_soft + eps).sum(dim=1, keepdim=True)
        log_y = torch.log(y_clamped)
        log_p = torch.log(p + eps)
        kl = (y_clamped * (log_y - log_p)).sum(dim=1)  # [B]

        y_hard = y_soft.argmax(dim=1)
        pred = p.argmax(dim=1)
        acc = (pred == y_hard).float()

        sum_kl += (kl * w).sum().item()
        sum_acc += (acc * w).sum().item()
        total_w += w.sum().item()

        # optional by-group metrics (if these keys exist)
        if "ctx_id" in x_dict and "street_id" in x_dict:
            ctx = x_dict["ctx_id"].cpu().numpy()
            street = x_dict["street_id"].cpu().numpy()
            for i in range(len(ctx)):
                key = (int(ctx[i]), int(street[i]))
                wi = float(w[i])
                grp[key]["w"]   += wi
                grp[key]["kl"]  += float(kl[i]) * wi
                grp[key]["acc"] += float(acc[i]) * wi

    report = {
        "checkpoint": str(ckpt_path),
        "parquet": str(parquet_path),
        "val_weight_sum": total_w,
        "val_kl": sum_kl / max(total_w, 1e-8),
        "val_soft_acc": sum_acc / max(total_w, 1e-8),
    }

    if grp:
        report["by_group"] = {
            f"ctx{c}_street{s}": {
                "kl": g["kl"] / max(g["w"], 1e-8),
                "soft_acc": g["acc"] / max(g["w"], 1e-8),
                "w": g["w"],
            }
            for (c, s), g in sorted(grp.items(), key=lambda kv: -kv[1]["w"])
        }

    return report


def list_candidate_ckpts(ckpts_dir: Path) -> List[Path]:
    cks = sorted(ckpts_dir.glob("*.ckpt"))
    return [p for p in cks if p.name not in ("last.ckpt", "best.ckpt")]

def pick_best(reports: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    """
    Choose best by lowest val_kl; tie-break by highest val_soft_acc.
    Returns (index, report).
    """
    best_i = None
    best_key = None
    for i, r in enumerate(reports):
        key = (round(r["val_kl"], 12), -round(r["val_soft_acc"], 12))  # lower KL, higher acc
        if best_key is None or key < best_key:
            best_key = key
            best_i = i
    return best_i, reports[best_i]

def main():
    ap = argparse.ArgumentParser(description="Evaluate all ckpts in a folder and write best.ckpt (+ sidecar)")
    ap.add_argument("--ckpts-dir", type=Path, required=True, help="e.g. checkpoints/popnet/dev")
    ap.add_argument("--parquet", type=Path, required=True, help="populationnet parquet with soft labels")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-json", type=Path, default=None, help="optional: write full sweep report")
    args = ap.parse_args()

    ckpts = list_candidate_ckpts(args.ckpts_dir)
    if not ckpts:
        raise SystemExit(f"No candidate ckpts in {args.ckpts_dir}")

    reports = []
    for p in ckpts:
        rep = evaluate_populationnet(str(p), str(args.parquet), batch_size=args.batch_size, seed=args.seed)
        reports.append(rep)
        print(f"eval {p.name}: KL={rep['val_kl']:.6f}  softAcc={rep['val_soft_acc']:.4f}")

    if args.report_json:
        args.report_json.write_text(json.dumps({"reports": reports}, indent=2))

    idx, best_rep = pick_best(reports)
    best_src = ckpts[idx]
    best_dst = args.ckpts_dir / "best.ckpt"
    shutil.copy2(best_src, best_dst)
    sidecar = write_popnet_sidecar(
        best_ckpt=best_dst,
        ds=PopulationDatasetParquet(str(args.parquet), use_soft_labels=True, device=None),
        model=PopulationNetLit.load_from_checkpoint(str(best_dst), map_location=torch.device("cpu")).eval(),
    )

    # Small meta for traceability
    meta = {
        "chosen": best_src.name,
        "val_kl": best_rep["val_kl"],
        "val_soft_acc": best_rep["val_soft_acc"],
        "parquet": str(args.parquet),
        "sidecar": str(sidecar),
    }
    (args.ckpts_dir / "best.meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n✅ best = {best_src.name}")
    print(f"→ wrote {best_dst.name} and {best_dst.name}.sidecar.json")
    print(f"→ meta: {args.ckpts_dir/'best.meta.json'}")

if __name__ == "__main__":
    main()