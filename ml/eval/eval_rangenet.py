# ml/eval/eval_rangenet.py
import argparse, json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from ml.inference.rangenet import RangeNetInfer
from ml.utils.config import load_model_config
from ml.utils.sidecar import load_sidecar

EPS = 1e-12

def _topk_overlap(y_true: np.ndarray, y_pred: np.ndarray, k: int = 25) -> np.ndarray:
    """Per-row top-k overlap on 169-vector probabilities."""
    idx_t = np.argpartition(-y_true, k-1, axis=1)[:, :k]
    idx_p = np.argpartition(-y_pred, k-1, axis=1)[:, :k]
    # convert to sets per row (vectorized-ish)
    overlaps = []
    for a, b in zip(idx_t, idx_p):
        overlaps.append(len(set(a.tolist()) & set(b.tolist())) / float(k))
    return np.asarray(overlaps, dtype=np.float32)

def _wavg(x: np.ndarray, w: np.ndarray) -> float:
    s = float(w.sum())
    return float((x * w).sum() / s) if s > 0 else float(np.mean(x))

def _detect_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Return x_cols, y_cols, weight_col, slice_keys based on columns in parquet."""
    # y target columns
    y_cols = [c for c in df.columns if c.startswith("y_")]
    y_cols = sorted(y_cols, key=lambda c: int(c.split("_")[1]))
    if len(y_cols) != 169:
        raise ValueError(f"Expected 169 y_* columns, found {len(y_cols)}")

    # weight
    for wcol in ("weight", "w"):
        if wcol in df.columns:
            weight_col = wcol
            break
    else:
        weight_col = None

    # Heuristic: postflop if these exist
    is_post = ("board_cluster_id" in df.columns) or ("street" in df.columns)

    if is_post:
        # common postflop features you trained on
        candidate_x = ["stack_bb","hero_pos","villain_pos","street","board_cluster_id"]
        x_cols = [c for c in candidate_x if c in df.columns]
        slice_keys = [c for c in candidate_x if c in df.columns]
    else:
        # common preflop features you trained on
        candidate_x = ["stack_bb","hero_pos","opener_pos","opener_action"]
        x_cols = [c for c in candidate_x if c in df.columns]
        slice_keys = [c for c in candidate_x if c in df.columns]

    if not x_cols:
        raise ValueError("Could not detect X columns; please add to config.dataset.x_cols or parquet is missing expected cols.")
    return {
        "is_post": is_post,
        "x_cols": x_cols,
        "y_cols": y_cols,
        "weight_col": weight_col,
        "slice_keys": slice_keys,
    }

def _ensure_prob(mat: np.ndarray) -> np.ndarray:
    s = mat.sum(axis=1, keepdims=True)
    s[s <= EPS] = 1.0
    return mat / s

def eval_rangenet(cfg: Dict[str, Any]) -> None:
    # ---- Resolve IO from cfg ----
    def get(path, default=None):
        cur = cfg
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    parquet_path = Path(
        get("inputs.parquet")
        or get("dataset.parquet")
        or get("rangenet_preflop.outputs.parquet")
        or get("rangenet_postflop.outputs.parquet")
        or "data/datasets/rangenet_preflop.parquet"
    )
    out_dir = Path(get("eval.output_dir", "reports/eval/rangenet"))
    ckpt = get("eval.checkpoint") or get("train.checkpoint") or get("train.best_ckpt")
    sidecar = get("eval.sidecar") or get("train.sidecar")

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    if not ckpt:
        raise ValueError("Missing eval.checkpoint (or train.best_ckpt) in config.")
    if not sidecar:
        # default to sibling files under ckpt dir
        ckpt_dir = Path(ckpt).parent
        sidecar = str(ckpt_dir / "sidecar.json")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    df = pd.read_parquet(parquet_path)
    schema = _detect_schema(df)
    x_cols = schema["x_cols"]
    y_cols = schema["y_cols"]
    w_col = schema["weight_col"]
    slice_keys = schema["slice_keys"]

    y_true = df[y_cols].to_numpy(dtype=np.float32)
    y_true = _ensure_prob(y_true)
    w = df[w_col].to_numpy(dtype=np.float32) if w_col else np.ones((len(df),), dtype=np.float32)

    # ---- Inference wrapper ----
    sc = load_sidecar(sidecar)  # expects {"feature_order":[...], "cards": {...}} at minimum
    infer = RangeNetInfer.from_checkpoint(ckpt, sc, device="cuda")

    # batching
    batch = int(get("eval.batch_size", 2048))
    Xrecs = df[x_cols].to_dict(orient="records")
    preds: List[np.ndarray] = []
    for i in range(0, len(Xrecs), batch):
        preds.append(infer.predict_batch(Xrecs[i:i+batch], as_numpy=True))
    y_pred = np.vstack(preds).astype(np.float32)
    y_pred = _ensure_prob(y_pred)

    # ---- Metrics ----
    kl = np.sum(y_true * (np.log(y_true + EPS) - np.log(y_pred + EPS)), axis=1)
    ce = -np.sum(y_true * np.log(y_pred + EPS), axis=1)
    l2 = np.sqrt(np.sum((y_pred - y_true) ** 2, axis=1))
    cos = (np.sum(y_pred * y_true, axis=1) /
           (np.linalg.norm(y_pred, axis=1) * np.linalg.norm(y_true, axis=1) + EPS))
    ent_true = -np.sum(y_true * np.log(y_true + EPS), axis=1)
    ent_pred = -np.sum(y_pred * np.log(y_pred + EPS), axis=1)
    top10 = _topk_overlap(y_true, y_pred, k=10)
    top25 = _topk_overlap(y_true, y_pred, k=25)

    summary = {
        "rows": int(len(df)),
        "kl_mean": _wavg(kl, w),
        "kl_p95": float(np.quantile(kl, 0.95)),
        "ce_mean": _wavg(ce, w),
        "l2_mean": _wavg(l2, w),
        "cos_mean": _wavg(cos, w),
        "entropy_true_mean": _wavg(ent_true, w),
        "entropy_pred_mean": _wavg(ent_pred, w),
        "top10_overlap_mean": _wavg(top10, w),
        "top25_overlap_mean": _wavg(top25, w),
        "parquet": str(parquet_path),
        "checkpoint": str(ckpt),
        "sidecar": str(sidecar),
        "x_cols": x_cols,
        "y_cols_count": len(y_cols),
        "weight_col": w_col,
        "slice_keys": slice_keys,
    }

    # per-slice table
    eval_df = df[slice_keys].copy()
    eval_df["kl"] = kl
    eval_df["ce"] = ce
    eval_df["l2"] = l2
    eval_df["cos"] = cos
    eval_df["ent_true"] = ent_true
    eval_df["ent_pred"] = ent_pred
    eval_df["top10"] = top10
    eval_df["top25"] = top25
    eval_df["w"] = w

    def agg(g: pd.DataFrame) -> pd.Series:
        ww = g["w"].to_numpy(np.float32)
        def wav(col): return float(np.average(g[col].to_numpy(np.float32), weights=ww)) if ww.sum() > 0 else float(g[col].mean())
        return pd.Series({
            "n_rows": int(len(g)),
            "w_sum": float(ww.sum()),
            "kl_mean": wav("kl"),
            "kl_p95": float(np.quantile(g["kl"], 0.95)),
            "ce_mean": wav("ce"),
            "l2_mean": wav("l2"),
            "cos_mean": wav("cos"),
            "entropy_true_mean": wav("ent_true"),
            "entropy_pred_mean": wav("ent_pred"),
            "top10_overlap_mean": wav("top10"),
            "top25_overlap_mean": wav("top25"),
        })

    per_slice = eval_df.groupby(slice_keys, dropna=False).apply(agg).reset_index()

    # outputs
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    per_slice.to_csv(out_dir / "per_slice.csv", index=False)
    print(f"✅ Wrote eval → {out_dir}")
    print(f"   Overall KL={summary['kl_mean']:.4f}  CE={summary['ce_mean']:.4f}  COS={summary['cos_mean']:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="rangenet_postflop", help="model name or YAML path")
    # optional overrides
    ap.add_argument("--parquet", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--sidecar", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    if args.parquet:
        cfg.setdefault("inputs", {})["parquet"] = args.parquet
    if args.checkpoint:
        cfg.setdefault("eval", {})["checkpoint"] = args.checkpoint
    if args.sidecar:
        cfg.setdefault("eval", {})["sidecar"] = args.sidecar
    if args.outdir:
        cfg.setdefault("eval", {})["output_dir"] = args.outdir

    eval_rangenet(cfg)

if __name__ == "__main__":
    main()