# ml/eval/eval_populationnet.py
from __future__ import annotations
import json
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader, Subset
from ml.datasets.population import PopulationDatasetParquet, population_collate_fn
from ml.datasets.utils_dataset import stratified_indices
from ml.models.population_net import PopulationNetLit


@torch.no_grad()
def evaluate_populationnet(ckpt_path: str, parquet_path: str,
                           batch_size: int = 1024, seed: int = 42) -> Dict[str, Any]:
    # dataset with SOFT labels (population = soft by default)
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
    device = torch.device("cpu")
    model = PopulationNetLit.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device).eval()

    total_w = 0.0
    sum_kl = 0.0
    sum_acc = 0.0

    from collections import defaultdict
    grp = defaultdict(lambda: {"w": 0.0, "kl": 0.0, "acc": 0.0})

    for x_dict, y_soft, w in val_dl:
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

        ctx = x_dict["ctx_id"].cpu().numpy()
        street = x_dict["street_id"].cpu().numpy()
        for i in range(len(ctx)):
            key = (int(ctx[i]), int(street[i]))
            wi = float(w[i])
            grp[key]["w"]   += wi
            grp[key]["kl"]  += float(kl[i]) * wi
            grp[key]["acc"] += float(acc[i]) * wi

    report = {
        "checkpoint": ckpt_path,
        "parquet": parquet_path,
        "val_weight_sum": total_w,
        "val_kl": sum_kl / max(total_w, 1e-8),
        "val_soft_acc": sum_acc / max(total_w, 1e-8),
        "by_group": {
            f"ctx{c}_street{s}": {
                "kl": g["kl"] / max(g["w"], 1e-8),
                "soft_acc": g["acc"] / max(g["w"], 1e-8),
                "w": g["w"],
            }
            for (c, s), g in sorted(grp.items(), key=lambda kv: -kv[1]["w"])
        },
    }
    return report

def save_report(report: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)