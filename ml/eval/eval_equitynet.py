from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, Sequence, Tuple, List

import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import numpy as np
import pandas as pd

from ml.datasets.equitynet import EquityDatasetParquet, equity_collate_fn
from ml.models.equity_net import EquityNetLit
from ml.utils.config import load_model_config


def _get(cfg: Dict[str, Any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _kl_row(p: np.ndarray, q: np.ndarray, eps=1e-8) -> float:
    p = np.clip(p, eps, 1.0); p /= p.sum()
    q = np.clip(q, eps, 1.0); q /= q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))

def _soft_acc_row(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.minimum(p, q).sum())

def _scenario_split_indices(df: pd.DataFrame, keys: Sequence[str], train_frac: float, seed: int):
    gb = df.groupby(list(keys), sort=False, as_index=False).indices
    key_list = list(gb.keys())
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(key_list), generator=g).tolist()
    key_list = [key_list[i] for i in perm]
    n_train = int(round(train_frac * len(key_list)))
    train_keys = set(key_list[:n_train])
    train_idx, val_idx = [], []
    for k, idx in gb.items():
        (train_idx if k in train_keys else val_idx).extend(idx.tolist())
    train_idx.sort(); val_idx.sort()
    return train_idx, val_idx

@torch.no_grad()
def evaluate_equity(checkpoint: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    pl.seed_everything(int(_get(cfg, "train.seed", 42)))

    parquet = Path(_get(cfg, "inputs.parquet", _get(cfg, "dataset.parquet")))
    x_cols = _get(cfg, "dataset.x_cols")
    y_cols = _get(cfg, "dataset.y_cols", ["y_win","y_tie","y_lose"])
    w_col  = _get(cfg, "dataset.weight_col", "weight")
    keep   = _get(cfg, "dataset.keep_values", None)
    min_w  = _get(cfg, "dataset.min_weight", None)
    scenario_keys = _get(cfg, "dataset.scenario_keys")  # preflop vs postflop differs here
    assert parquet.exists(), f"Parquet not found: {parquet}"
    assert x_cols and scenario_keys, "Config must define dataset.x_cols and dataset.scenario_keys"

    # Build dataset + split indices deterministically (same as trainers)
    df = pd.read_parquet(parquet)
    if keep:
        for k, vals in keep.items():
            if k in df.columns: df = df[df[k].isin(list(vals))]
    if min_w is not None and w_col in df.columns:
        df = df[df[w_col] >= float(min_w)]
    df = df.reset_index(drop=True)

    train_frac = float(_get(cfg, "train.train_frac", 0.8))
    seed = int(_get(cfg, "train.seed", 42))
    _, val_idx = _scenario_split_indices(df, scenario_keys, train_frac, seed)

    ds = EquityDatasetParquet(
        parquet_path=parquet, x_cols=x_cols, y_cols=y_cols, weight_col=w_col,
        keep_values=keep, min_weight=min_w, device=None
    )
    val_ds = Subset(ds, val_idx)
    val_dl = DataLoader(val_ds, batch_size=4096, shuffle=False,
                        num_workers=int(_get(cfg, "train.num_workers", 0)),
                        collate_fn=equity_collate_fn)

    # Model
    model = EquityNetLit(
        cards=ds.cards(),
        cat_order=ds.feature_order,
        hidden_dims=_get(cfg, "model.hidden_dims", [128,128]),
        dropout=float(_get(cfg, "model.dropout", 0.10)),
        lr=1e-3, weight_decay=1e-4,
    )
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Accumulators
    tot_w = 0.0
    kl_sum = mse_sum = sacc_sum = 0.0

    # For by-group stats (using scenario_keys)
    by_group: Dict[Tuple, Dict[str, float]] = {}

    # Also compute baselines
    uni = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    kl_uni = mse_uni = sacc_uni = 0.0

    for x_dict, y, w in val_dl:
        logits = model(x_dict)
        preds = torch.softmax(logits, dim=-1).cpu().numpy()  # [B,3]
        y_np = y.cpu().numpy()                                # [B,3]
        w_np = w.cpu().numpy().astype(np.float64)            # [B]

        # Extract scenario key tuple for each row
        # Need original df rows for this subset:
        idxs = val_ds.indices if isinstance(val_ds, Subset) else range(len(y_np))
        # Map batch positions to global row indices
        # (DataLoader keeps order for shuffle=False)
        start = getattr(evaluate_equity, "_cursor", 0)
        global_ids = idxs[start:start+len(y_np)]
        evaluate_equity._cursor = start + len(y_np)

        # Metrics per row
        for i, gid in enumerate(global_ids):
            p = preds[i]; t = y_np[i]; ww = float(w_np[i])
            kl = _kl_row(t, p); mse = float(((t - p)**2).sum()); sacc = _soft_acc_row(t, p)
            kl_sum += ww * kl; mse_sum += ww * mse; sacc_sum += ww * sacc
            kl_uni += ww * _kl_row(t, uni); mse_uni += ww * float(((t - uni)**2).sum()); sacc_uni += ww * _soft_acc_row(t, uni)
            tot_w += ww

            # by-group
            key_vals = tuple(df.loc[gid, k] for k in scenario_keys)
            st = by_group.setdefault(key_vals, {"w":0.0, "kl":0.0, "mse":0.0, "sacc":0.0})
            st["w"] += ww; st["kl"] += ww * kl; st["mse"] += ww * mse; st["sacc"] += ww * sacc

    # Finalize
    def _finish(d: Dict[Tuple, Dict[str,float]]):
        out = {}
        for k, v in d.items():
            w = max(v["w"], 1e-9)
            out["|".join(map(str, k))] = {
                "kl": v["kl"]/w, "mse": v["mse"]/w, "soft_acc": v["sacc"]/w, "w": v["w"]
            }
        return out

    report = {
        "checkpoint": str(checkpoint),
        "parquet": str(parquet),
        "val_weight_sum": tot_w,
        "val_kl": kl_sum / max(tot_w, 1e-9),
        "val_mse": mse_sum / max(tot_w, 1e-9),
        "val_soft_acc": sacc_sum / max(tot_w, 1e-9),
        "baseline_uniform": {
            "kl": kl_uni / max(tot_w, 1e-9),
            "mse": mse_uni / max(tot_w, 1e-9),
            "soft_acc": sacc_uni / max(tot_w, 1e-9),
        },
        "by_group": _finish(by_group),
        "scenario_keys": list(scenario_keys),
        "y_cols": list(y_cols),
        "feature_order": list(ds.feature_order),
    }
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="equitynet_preflop or equitynet_postflop (name or path)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt")
    ap.add_argument("--out", type=str, default=None, help="Path to write JSON report")
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    rep = evaluate_equity(Path(args.ckpt), cfg)
    txt = json.dumps(rep, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt)
        print(f"✅ wrote {args.out}")
    else:
        print(txt)

if __name__ == "__main__":
    main()