import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.datasets.equitynet import equity_collate_fn, EquityDatasetParquet
#!/usr/bin/env python3
import argparse, json, math, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# import your project bits

from ml.models.equity_net import EquityNetLit


def _bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)

def _accum_bins_toplabel(p: np.ndarray, y: np.ndarray, w: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Top-label reliability stats.
    p: [N,3] predicted probabilities
    y: [N,3] soft targets
    w: [N]   weights
    Returns (sum_conf, sum_acc, sum_w) per bin
    """
    conf = p.max(axis=1)                     # [N]
    top  = p.argmax(axis=1)                  # [N]
    acc  = y[np.arange(len(y)), top]         # [N]

    edges = _bin_edges(n_bins)
    idx = np.clip(np.digitize(conf, edges, right=False) - 1, 0, n_bins-1)

    sum_conf = np.zeros(n_bins, dtype=np.float64)
    sum_acc  = np.zeros(n_bins, dtype=np.float64)
    sum_w    = np.zeros(n_bins, dtype=np.float64)
    np.add.at(sum_conf, idx, conf * w)
    np.add.at(sum_acc,  idx, acc  * w)
    np.add.at(sum_w,    idx, w)
    return sum_conf, sum_acc, sum_w

def _accum_bins_per_class(p: np.ndarray, y: np.ndarray, w: np.ndarray, n_bins: int) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Classwise reliability stats (per class k, bin by p[:,k], accuracy = y[:,k]).
    Returns dict[k] -> {"sum_conf","sum_acc","sum_w"}
    """
    K = p.shape[1]
    edges = _bin_edges(n_bins)
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for k in range(K):
        conf_k = p[:, k]
        acc_k  = y[:, k]
        idx = np.clip(np.digitize(conf_k, edges, right=False) - 1, 0, n_bins-1)
        sum_conf = np.zeros(n_bins, dtype=np.float64)
        sum_acc  = np.zeros(n_bins, dtype=np.float64)
        sum_w    = np.zeros(n_bins, dtype=np.float64)
        np.add.at(sum_conf, idx, conf_k * w)
        np.add.at(sum_acc,  idx, acc_k  * w)
        np.add.at(sum_w,    idx, w)
        out[k] = {"sum_conf": sum_conf, "sum_acc": sum_acc, "sum_w": sum_w}
    return out

def _ece(sum_conf: np.ndarray, sum_acc: np.ndarray, sum_w: np.ndarray) -> float:
    """ECE over provided bin aggregates."""
    eps = 1e-12
    conf_bar = np.divide(sum_conf, sum_w + eps)
    acc_bar  = np.divide(sum_acc,  sum_w + eps)
    weights  = sum_w / max(sum_w.sum(), eps)
    return float(np.sum(weights * np.abs(acc_bar - conf_bar)))

@torch.no_grad()
def eval_calibration(
    ckpt_path: str,
    parquet_path: str,
    *,
    batch_size: int = 4096,
    n_bins: int = 15,
    sample_frac: float = 1.0,
) -> Dict[str, float]:
    # Dataset
    y_cols = ["p_win", "p_tie", "p_lose"]
    # Try to infer x_cols from parquet columns (robust to schema)
    df_cols = set(__import__("pandas").read_parquet(parquet_path).columns)
    expected_x = ["stack_bb","hero_pos","opener_action","hand_id","street","board_cluster_id"]
    x_cols = [c for c in expected_x if c in df_cols]
    ds = EquityDatasetParquet(
        parquet_path=parquet_path,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col="weight",
        device=None,
    )

    # Optional subsample
    idx = np.arange(len(ds))
    if sample_frac < 1.0:
        n = max(1, int(len(ds) * float(sample_frac)))
        rng = np.random.default_rng(42)
        idx = rng.choice(idx, size=n, replace=False)
    subset = Subset(ds, idx.tolist())

    dl = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        collate_fn=equity_collate_fn, num_workers=0, pin_memory=True
    )

    # Model
    device = torch.device("cpu")
    model = EquityNetLit.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device).eval()

    # Collect predictions/targets
    P, Y, W = [], [], []
    for x_dict, y, w in dl:
        logits = model(x_dict)         # [B,3]
        p = torch.softmax(logits, dim=-1)
        P.append(p.cpu().numpy())
        Y.append(y.cpu().numpy())
        W.append(w.cpu().numpy())
    if not P:
        raise RuntimeError("No data to evaluate.")
    P = np.concatenate(P, axis=0)
    Y = np.concatenate(Y, axis=0)
    W = np.concatenate(W, axis=0)

    # Top-label ECE
    top_conf, top_acc, top_w = _accum_bins_toplabel(P, Y, W, n_bins)
    ece_top = _ece(top_conf, top_acc, top_w)

    # Per-class ECE
    per_cls = _accum_bins_per_class(P, Y, W, n_bins)
    ece_per_class = {int(k): _ece(v["sum_conf"], v["sum_acc"], v["sum_w"]) for k, v in per_cls.items()}

    return {
        "ece_top": ece_top,
        "ece_per_class": ece_per_class,  # 0=win,1=tie,2=lose by your schema
        "n_bins": int(n_bins),
        "n_rows": int(W.shape[0]),
    }


def main():
    ap = argparse.ArgumentParser(description="EquityNet calibration / ECE evaluator")
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
    ap.add_argument("--parquet", type=str, required=True, help="equitynet parquet")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--sample-frac", type=float, default=1.0, help="0<frac<=1.0 to subsample rows")
    ap.add_argument("--out-json", type=str, default=None, help="optional path to write metrics JSON")
    ap.add_argument("--out-csv", type=str, default=None, help="optional reliability CSV (top-label bins)")
    args = ap.parse_args()

    t0 = time.time()
    rep = eval_calibration(args.ckpt, args.parquet, batch_size=args.batch_size, n_bins=args.bins, sample_frac=args.sample_frac)
    dt = time.time() - t0

    print(json.dumps(rep, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rep, indent=2))

    # Optional CSV of top-label bins
    if args.out_csv:
        # rebuild top-label bin table quickly to persist
        # (re-run with no sampling issues)
        y_cols = ["p_win", "p_tie", "p_lose"]
        df_cols = set(__import__("pandas").read_parquet(args.parquet).columns)
        expected_x = ["stack_bb","hero_pos","opener_action","hand_id","street","board_cluster_id"]
        x_cols = [c for c in expected_x if c in df_cols]
        ds = EquityDatasetParquet(args.parquet, x_cols=x_cols, y_cols=y_cols, weight_col="weight", device=None)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=equity_collate_fn)
        device = torch.device("cpu")
        model = EquityNetLit.load_from_checkpoint(args.ckpt, map_location=device).to(device).eval()
        P, Y, W = [], [], []
        for x_dict, y, w in dl:
            logits = model(x_dict)
            p = torch.softmax(logits, dim=-1)
            P.append(p.cpu().numpy()); Y.append(y.cpu().numpy()); W.append(w.cpu().numpy())
        P = np.concatenate(P); Y = np.concatenate(Y); W = np.concatenate(W)
        c, a, ww = _accum_bins_toplabel(P, Y, W, args.bins)
        eps = 1e-12
        conf_bar = c / (ww + eps); acc_bar = a / (ww + eps)
        import pandas as pd
        edges = _bin_edges(args.bins)
        mid = (edges[:-1] + edges[1:]) / 2.0
        df = pd.DataFrame({
            "bin": np.arange(args.bins),
            "bin_left": edges[:-1], "bin_right": edges[1:], "bin_mid": mid,
            "weight": ww, "mean_conf": conf_bar, "mean_acc": acc_bar,
            "gap": np.abs(acc_bar - conf_bar),
        })
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)

    print(f"✅ done in {dt:.2f}s")

if __name__ == "__main__":
    main()