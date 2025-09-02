import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.datasets.utils_dataset import stratified_indices
from ml.models.equity_net import EquityNetLit
from ml.utils.sidecar import save_sidecar_json, load_sidecar
from ml.datasets.equitynet import EquityDatasetParquet, equity_collate_fn



def list_candidate_ckpts(ckpts_dir: Path) -> List[Path]:
    cks = sorted(ckpts_dir.glob("*.ckpt"))
    # exclude reserved names if present
    return [p for p in cks if p.name not in ("last.ckpt", "best.ckpt")]


def _sample_indices(n: int, k: int, seed: int) -> List[int]:
    k = min(k, n)
    rnd = random.Random(seed)
    idxs = list(range(n))
    rnd.shuffle(idxs)
    return idxs[:k]


@torch.no_grad()
def evaluate_equitynet(
    ckpt_path: str,
    parquet_path: str,
    *,
    batch_size: int = 1024,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate a single EquityNet checkpoint on the validation split of a parquet.
    Returns weighted KL and soft-accuracy (argmax vs argmax).
    """
    import pandas as pd
    device = torch.device("cpu")

    # ---- resolve x_cols (prefer sidecar) ----
    sidecar_path = Path(ckpt_path).with_suffix(Path(ckpt_path).suffix + ".sidecar.json")
    feature_order: Optional[List[str]] = None
    if sidecar_path.exists():
        sc = load_sidecar(sidecar_path)
        feature_order = list(sc.get("feature_order", []) or [])
    if not feature_order:
        # fallback: infer from parquet columns
        df_cols = pd.read_parquet(parquet_path, columns=None).columns.tolist()
        expected_x_cols = ["street", "hand_id", "board_cluster_id", "stack_bb", "hero_pos", "opener_action"]
        feature_order = [c for c in expected_x_cols if c in df_cols]
        if not feature_order:
            raise RuntimeError(
                f"Cannot determine x_cols. Sidecar missing or invalid at {sidecar_path} "
                f"and no expected columns found in parquet ({parquet_path})."
            )

    y_cols = ["p_win", "p_tie", "p_lose"]

    # ---- dataset ----
    ds = EquityDatasetParquet(
        parquet_path=parquet_path,
        x_cols=feature_order,
        y_cols=y_cols,
        weight_col="weight",
        device=None,
    )

    # Split (try stratified; else simple)
    try:
        train_idx, val_idx = stratified_indices(ds.df, keys=None, train_frac=0.8, seed=seed)
    except Exception:
        n = len(ds)
        cut = int(0.8 * n)
        idx = list(range(n))
        train_idx, val_idx = idx[:cut], idx[cut:]

    val_dl = DataLoader(
        Subset(ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=equity_collate_fn,
        pin_memory=True,
        num_workers=0,
    )

    # ---- model ----
    model = EquityNetLit.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device).eval()

    total_w = 0.0
    sum_kl = 0.0
    sum_acc = 0.0

    for x_dict, y, w in val_dl:
        B = y.shape[0]
        # forward → logits [B,3]; EquityNetLit.forward expects (x_cat, x_num)
        x_num = torch.empty(B, 0, device=device)  # no numeric features
        logits = model(x_dict, x_num)
        p = torch.softmax(logits, dim=-1)  # [B,3]

        # KL(y || p) with soft labels
        eps = 1e-12
        y_clamped = (y + eps) / (y + eps).sum(dim=1, keepdim=True)
        log_y = torch.log(y_clamped)
        log_p = torch.log(p + eps)
        kl = (y_clamped * (log_y - log_p)).sum(dim=1)  # [B]

        # soft-acc (argmax vs argmax)
        y_hard = y.argmax(dim=1)
        pred = p.argmax(dim=1)
        acc = (pred == y_hard).float()

        sum_kl += (kl * w).sum().item()
        sum_acc += (acc * w).sum().item()
        total_w += w.sum().item()

    report = {
        "checkpoint": str(ckpt_path),
        "parquet": str(parquet_path),
        "val_weight_sum": total_w,
        "val_kl": sum_kl / max(total_w, 1e-8),
        "val_soft_acc": sum_acc / max(total_w, 1e-8),
    }
    return report


def pick_best(reports: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    """
    Choose best by lowest val_kl; tie-break by highest val_soft_acc.
    """
    best_i = None
    best_key = None
    for i, r in enumerate(reports):
        key = (round(r["val_kl"], 12), -round(r["val_soft_acc"], 12))
        if best_key is None or key < best_key:
            best_key = key
            best_i = i
    return best_i, reports[best_i]


def write_equity_sidecar_for_ckpt(best_ckpt: Path, parquet_path: Path) -> Path:
    """
    Ensure a sidecar next to best_ckpt. If one already exists, reuse it.
    Otherwise derive schema from the parquet and write a fresh sidecar.
    """
    # If a sidecar already exists, reuse it
    existing = best_ckpt.with_suffix(best_ckpt.suffix + ".sidecar.json")
    if existing.exists():
        return existing

    import pandas as pd
    # Infer x_cols from parquet (must match what you trained with)
    cols = pd.read_parquet(str(parquet_path)).columns.tolist()
    candidate_x = ["street", "hand_id", "board_cluster_id"]
    x_cols = [c for c in candidate_x if c in cols]
    if not x_cols:
        raise RuntimeError(
            f"Cannot infer x_cols from equity parquet {parquet_path}; "
            f"available columns: {cols}"
        )

    y_cols = ["p_win", "p_tie", "p_lose"]
    ds = EquityDatasetParquet(
        parquet_path=str(parquet_path),
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col="weight",
        device=None,
    )

    feature_order = list(getattr(ds, "feature_order", []))
    if not feature_order:
        raise RuntimeError("EquityDatasetParquet did not expose feature_order; cannot write sidecar.")

    # Pull cards & id_maps from dataset
    cards = ds.cards() if hasattr(ds, "cards") and callable(getattr(ds, "cards", None)) else dict(getattr(ds, "cards", {}))
    id_maps = ds.id_maps() if hasattr(ds, "id_maps") and callable(getattr(ds, "id_maps", None)) else None

    extra = {
        "labels": ["p_win", "p_tie", "p_lose"],
        "soft_labels": True,
        "notes": "EquityNet trained on soft triplet labels (p_win,p_tie,p_lose).",
    }
    return save_sidecar_json(
        best_ckpt,
        model_name="EquityNet",
        feature_order=feature_order,
        cards=cards,
        id_maps=id_maps,  # key name is "id_maps" in our sidecar schema
        extra=extra,
    )


def main():
    ap = argparse.ArgumentParser(description="Evaluate all EquityNet ckpts in a folder and write best.ckpt (+ sidecar)")
    ap.add_argument("--ckpts-dir", type=Path, required=True, help="e.g. checkpoints/equitynet/dev")
    ap.add_argument("--parquet", type=Path, required=True, help="equitynet parquet with triplet soft labels")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-json", type=Path, default=None, help="optional: write full sweep report")
    args = ap.parse_args()

    ckpts = list_candidate_ckpts(args.ckpts_dir)
    if not ckpts:
        raise SystemExit(f"No candidate ckpts in {args.ckpts_dir}")

    reports: List[Dict[str, Any]] = []
    for p in ckpts:
        rep = evaluate_equitynet(str(p), str(args.parquet), batch_size=args.batch_size, seed=args.seed)
        reports.append(rep)
        print(f"eval {p.name}: KL={rep['val_kl']:.6f}  softAcc={rep['val_soft_acc']:.4f}")

    if args.report_json:
        args.report_json.write_text(json.dumps({"reports": reports}, indent=2))

    idx, best_rep = pick_best(reports)
    best_src = ckpts[idx]
    best_dst = args.ckpts_dir / "best.ckpt"
    shutil.copy2(best_src, best_dst)
    sidecar = write_equity_sidecar_for_ckpt(best_dst, args.parquet)

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