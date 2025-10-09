from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Optional, Dict
import json
import pandas as pd
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.datasets.preflop_rangenet import PreflopRangeDatasetParquet
from ml.models.preflop_rangenet import RangeNetLit, rangenet_preflop_collate_fn
from ml.utils.config import load_model_config
from ml.trainers.helpers import _get
from ml.utils.rangenet_preflop_sidecar import write_preflop_policy_sidecar
from ml.trainers.sweep import run_sweep, parse_scalar_from_ckpt, finalize_best_artifacts


def run_preflop_sweep(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sweep = cfg.get("sweep") or {}
    if not isinstance(sweep, dict) or not sweep:
        raise ValueError("No 'sweep' block in config.")

    base_dir = Path(cfg.get("train", {}).get("checkpoints_dir", "checkpoints/preflop_policy"))
    monitor = cfg.get("train", {}).get("monitor", "val_kl")  # preflop usually monitored on KL
    max_trials = sweep.get("max_trials", None)

    res = run_sweep(
        base_cfg=cfg,
        sweep=sweep,
        run_fn=run_train_preflop,          # runs one trial; returns ckpt path
        score_fn=parse_scalar_from_ckpt,   # parses metric from ckpt/metrics
        base_ckpt_dir=base_dir,
        monitor=monitor,
        max_trials=max_trials,
    )

    if res["best"]["ckpt"]:
        finalize_best_artifacts(Path(res["best"]["ckpt"]), base_dir)

    return res

def run_train_preflop(cfg: Mapping[str, Any]) -> str:
    import random, numpy as np, torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Subset

    # ---------------- Repro ----------------
    seed = int(_get(cfg, "train.seed", 42))
    precision = str(_get(cfg, "train.precision", 4))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)

    # ---------------- Dataset ----------------
    parquet_path = _get(cfg, "inputs.parquet") or _get(cfg, "dataset.parquet")
    if not parquet_path:
        raise ValueError("Missing inputs.parquet (or dataset.parquet) in config")

    x_cols      = _get(cfg, "dataset.x_cols") or ["stack_bb","hero_pos","opener_pos","ctx","opener_action"]
    min_weight  = _get(cfg, "dataset.min_weight", None)
    weight_col  = _get(cfg, "dataset.weight_col", "weight")

    ds = PreflopRangeDatasetParquet(
        parquet_path=parquet_path,
        x_cols=x_cols,
        weight_col=weight_col,
        min_weight=min_weight,
        device=torch.device("cpu"),      # data tensors live on CPU; model can move later
        strict_canon=True,
        debug=True,
    )

    n_total = len(ds)
    if n_total == 0:
        raise RuntimeError(
            "Preflop dataset is empty after canonicalization/filters. "
            "Set strict_canon=False temporarily or inspect debug counts."
        )
    print(f"[trainer] preflop rows after filters: {n_total}")

    cards         = ds.cards_info.cards
    feature_order = list(ds.feature_order)

    # -------- helper: reconstruct a DF for stratify (best-effort) --------
    def _as_dataframe_for_stratify(ds) -> Optional[pd.DataFrame]:
        if hasattr(ds, "df"):
            df = getattr(ds, "df")
            if isinstance(df, pd.DataFrame) and len(df) == len(ds):
                return df.copy()

        try:
            X = ds._X  # (N,F) int64
            feats = list(ds.feature_order)
            id_maps = ds.id_maps()  # raw->id
            # reverse: id->raw
            rev = {k: {int(v): str(k2) for k2, v in mapping.items()} for k, mapping in id_maps.items()}
            N, F = X.shape
            cols = {}
            for j, feat in enumerate(feats):
                card = int(ds.cards_info.cards.get(feat, 0))
                unk_id = max(card - 1, 0)
                ids = X[:, j]
                raw = [rev.get(feat, {}).get(int(i), "__UNK__") if int(i) != unk_id else "__UNK__" for i in ids]
                cols[feat] = raw
            if hasattr(ds, "_W"):
                cols["weight"] = ds._W.tolist()
            return pd.DataFrame(cols)
        except Exception:
            return None

    def _add_stack_bin(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
        if "stack_bb" in df.columns:
            stack_bins = _get(cfg, "dataset.stack_bins", [25, 60, 100, 150])
            # Build dynamic bin edges: below first, between, and above last
            bins = [-1] + stack_bins + [1e9]
            labels = [f"≤{stack_bins[0]}"] + [
                f"{stack_bins[i]}–{stack_bins[i + 1]}"
                for i in range(len(stack_bins) - 1)
            ] + [f">{stack_bins[-1]}"]
            s = pd.to_numeric(df["stack_bb"], errors="coerce").fillna(0).astype(float)
            df["stack_bin"] = pd.cut(s, bins=bins, labels=labels)
        return df

    def _robust_split(ds, *, stratify_keys, train_frac: float, seed: int):
        n = len(ds)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

        if not stratify_keys:
            cut = max(1, int(round(n * train_frac)))
            train_idx = idx[:cut].tolist()
            val_idx   = idx[cut:].tolist()
            if len(val_idx) == 0:
                val_idx = train_idx[-1:]
                train_idx = train_idx[:-1]
            return train_idx, val_idx, "random"

        df = _as_dataframe_for_stratify(ds)
        if df is None:
            print("[split] no DataFrame for stratification; falling back → random")
            return _robust_split(ds, stratify_keys=None, train_frac=train_frac, seed=seed)

        df = _add_stack_bin(df, cfg)

        miss = [k for k in stratify_keys if k not in df.columns]
        if miss:
            print(f"[split] stratify keys missing {miss}; falling back → random")
            return _robust_split(ds, stratify_keys=None, train_frac=train_frac, seed=seed)

        strata = df[list(stratify_keys)].astype(str).agg("||".join, axis=1).values
        uniq, counts = np.unique(strata, return_counts=True)
        if (counts < 2).any() or len(uniq) > n * 0.8:
            print("[split] tiny/too-many strata; falling back → random")
            return _robust_split(ds, stratify_keys=None, train_frac=train_frac, seed=seed)

        train_idx, val_idx = [], []
        for s in uniq:
            s_idx = np.where(strata == s)[0]
            rng.shuffle(s_idx)
            cut = max(1, int(round(len(s_idx) * train_frac)))
            cut = min(cut, len(s_idx) - 1) if len(s_idx) > 1 else 1
            train_idx.extend(s_idx[:cut].tolist())
            val_idx.extend(s_idx[cut:].tolist())

        if len(val_idx) == 0:
            val_idx = train_idx[-1:]
            train_idx = train_idx[:-1]
        return train_idx, val_idx, "stratified"

    # ---------------- Split ----------------
    stratify_keys = _get(cfg, "dataset.stratify_keys", [])
    if isinstance(stratify_keys, (tuple, list)) and len(stratify_keys) == 0:
        stratify_keys = None
    train_frac = float(_get(cfg, "train.train_frac", 0.8))

    train_idx, val_idx, split_mode = _robust_split(
        ds, stratify_keys=stratify_keys, train_frac=train_frac, seed=seed
    )
    print(f"[split] mode={split_mode}  train={len(train_idx)}  val={len(val_idx)}  total={len(ds)}")

    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)

    # ---------------- DataLoaders ----------------
    batch_size = int(_get(cfg, "train.batch_size", 2048))
    if len(train_ds) > 0:
        batch_size = max(1, min(batch_size, len(train_ds)))
    else:
        raise RuntimeError("Training set is empty after split.")

    num_workers = int(_get(cfg, "train.num_workers", 0))
    pin_memory  = bool(_get(cfg, "train.pin_memory", True))

    use_weighted_sampler = bool(_get(cfg, "train.weighted_sampler", False))
    sampler = None
    if use_weighted_sampler and hasattr(ds, "df"):
        train_w = torch.tensor(ds.df.loc[train_idx, weight_col].values, dtype=torch.float32)
        sampler = torch.utils.data.WeightedRandomSampler(train_w, num_samples=len(train_idx), replacement=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=rangenet_preflop_collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=min(batch_size, max(1, len(val_ds))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=rangenet_preflop_collate_fn,
    )

    # ---------------- Model ----------------
    model = RangeNetLit(
        cards=cards,
        feature_order=feature_order,
        hidden_dims=_get(cfg, "model.hidden_dims", [256, 256]),
        dropout=float(_get(cfg, "model.dropout", 0.10)),
        lr=float(_get(cfg, "model.lr", 1e-3)),
        weight_decay=float(_get(cfg, "model.weight_decay", 1e-4)),
        label_smoothing=float(_get(cfg, "model.label_smoothing", 0.02)),
    )

    # ---------------- First-batch sanity ----------------
    xb, yb, wb = next(iter(train_dl))
    assert yb.dim() == 2 and yb.shape[1] == 169, f"y shape {yb.shape} != [B,169]"
    row_sums = yb.sum(dim=1)
    assert torch.isfinite(row_sums).all(), "NaNs/Infs in labels"
    assert (row_sums > 0.99).all() and (row_sums < 1.01).all(), f"Label rows not ~1, mean={row_sums.mean():.6f}"

    with torch.no_grad():
        logits = model(xb)
        eps = 1e-8
        p = torch.softmax(logits, dim=-1)
        y = (yb + eps) / (yb + eps).sum(dim=1, keepdim=True)
        kl = (y * (torch.log(y + eps) - torch.log(p + eps))).sum(dim=1)
        assert (kl >= -1e-5).all(), f"Negative KL detected: min={kl.min().item():.6f}"

    # ---------------- Callbacks / Logger ----------------
    ckpt_dir = Path(_get(cfg, "train.checkpoints_dir", "checkpoints/rangenet_preflop"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    patience = int(_get(cfg, "train.patience", 4))
    logger = None
    if _get(cfg, "logging.logger", "tensorboard") == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(
            save_dir=_get(cfg, "logging.tb_log_dir", "logs/tb"),
            name="rangenet_preflop"
        )

    monitor = _get(cfg, "train.monitor", "val_kl")
    mode    = _get(cfg, "train.mode", "min")

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="rangenet-preflop-{epoch:02d}-{val_kl:.4f}",
        monitor=monitor, mode=mode,
        save_last=True, save_top_k=1,
        auto_insert_metric_name=False,
    )

    # ---------------- Trainer ----------------
    patience = int(_get(cfg, "train.patience", 4))
    min_delta = float(_get(cfg, "train.min_delta", 1e-4))
    grad_clip = float(_get(cfg, "train.gradient_clip_val", 1.0))
    accum = int(_get(cfg, "train.accumulate_grad_batches", 1))

    early_cb = pl.callbacks.EarlyStopping(
        monitor=monitor, mode=mode, patience=patience, min_delta=min_delta
    )

    trainer = pl.Trainer(
        max_epochs=int(_get(cfg, "train.max_epochs", 15)),
        accelerator=_get(cfg, "train.accelerator", "auto"),
        devices=_get(cfg, "train.devices", "auto"),
        precision=precision,
        log_every_n_steps=50,
        callbacks=[ckpt_cb, early_cb],
        deterministic=True,
        enable_progress_bar=True,
        logger=logger,
        gradient_clip_val=grad_clip,
        accumulate_grad_batches=accum,
    )

    # ---------------- Save config & sidecar ----------------
    try:
        from omegaconf import OmegaConf
        cfg_ser = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        cfg_ser = dict(cfg)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg_ser, indent=2))

    # write sidecar with canonical keys used by inference
    write_preflop_policy_sidecar(
        ckpt_dir,
        feature_order=feature_order,
        id_maps=ds.id_maps(),
        cards=cards,
    )

    # ---------------- Train (with auto-resume) ----------------
    resume_from = _get(cfg, "train.resume_from", None)
    if not resume_from:
        last_ckpt = ckpt_dir / "last.ckpt"
        if last_ckpt.exists():
            resume_from = str(last_ckpt)

    if resume_from:
        trainer.fit(model, train_dl, val_dl, ckpt_path=resume_from)
    else:
        trainer.fit(model, train_dl, val_dl)

    print(f"✅ preflop training complete. Best checkpoint: {ckpt_cb.best_model_path}")
    return ckpt_cb.best_model_path or str(ckpt_dir / "last.ckpt")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="rangenet/preflop",
                    help="Model name or YAML path resolved by load_model_config")
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--max_epochs", type=int)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--sweep", action="store_true", help="run the sweep defined in config.sweep")
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    train_cfg = cfg.setdefault("train", {})
    if args.batch_size is not None: train_cfg["batch_size"] = int(args.batch_size)
    if args.max_epochs is not None: train_cfg["max_epochs"] = int(args.max_epochs)
    if args.patience is not None:   train_cfg["patience"]   = int(args.patience)

    if args.sweep:
        run_preflop_sweep(cfg)

    else:
        run_train_preflop(cfg)