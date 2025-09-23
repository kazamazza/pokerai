from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Any, Optional, Dict

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.utils.popnet_sidecar import write_popnet_sidecar
from ml.datasets.population import PopulationDatasetParquet, population_collate_fn
from ml.datasets.utils_dataset import categorical_cardinalities, stratified_indices
from ml.models.population_net import PopulationNetLit

def make_collate_fn(feature_order):
    """
    Batch of (X, Y, W) -> (x_dict, y, w) that PopulationNetLit expects.
    X: [B, F] long; Y: [B, 3] float OR [B] long; W: [B] float
    If Y is [B,3] probs, we convert to hard labels via argmax here.
    """
    def _collate(batch):
        X = torch.stack([b[0] for b in batch], dim=0)       # [B, F]
        Y = torch.stack([b[1] for b in batch], dim=0)       # [B, 3] or [B]
        W = torch.stack([b[2] for b in batch], dim=0)       # [B]

        # Ensure hard labels (long) for CE; if Y is soft, take argmax
        if Y.dim() == 2:
            y = Y.argmax(dim=1).long()
        else:
            y = Y.long()

        x_dict = {name: X[:, i] for i, name in enumerate(feature_order)}
        return x_dict, y, W
    return _collate

def run_train(cfg: Mapping[str, Any]):
    """
    YAML-aware trainer. Expects a nested config with sections:
      dataset: parquet, keep_ctx_ids, keep_street_ids, min_weight, x_cols (optional)
      model:   hidden_dims, dropout, lr, weight_decay
      train:   batch_size, max_epochs, patience, num_workers, seed, accelerator, devices, precision,
               checkpoints_dir, monitor, mode, resume_from, train_frac, stratify_keys
    Flat keys (e.g. --parquet, --batch_size, etc.) are also respected as fallbacks.
    """

    # -------- small nested getter with fallback to flat keys --------
    def get(path: str, default=None):
        """
        Read a dotted key from nested cfg (OmegaConf or dict), fallback to flat key, else default.
        e.g. get("dataset.parquet", "datasets/datasets/populationnet_nl10.parquet")
        """
        # try nested first
        cur = cfg
        try:
            for k in path.split("."):
                if isinstance(cur, dict):
                    cur = cur.get(k, None)
                else:
                    # DictConfig or similar
                    cur = cur[k] if k in cur else None
            if cur is not None:
                return cur
        except Exception:
            pass
        # fallback to a flat key (rightmost token), then default
        flat_key = path.split(".")[-1]
        return cfg.get(flat_key, default)

    # -------- Reproducibility --------
    seed = int(get("train.seed", 42))
    pl.seed_everything(seed)

    # -------- Dataset --------
    parquet_path = get("inputs.parquet", get("parquet", "data/datasets/populationnet_nl10.parquet"))
    keep_ctx_ids = get("dataset.keep_ctx_ids", None)
    keep_street_ids = get("dataset.keep_street_ids", None)
    min_weight = get("dataset.min_weight", get("min_weight", None))
    x_cols = get("dataset.x_cols", None)  # optional override; dataset has sane defaults

    ds = PopulationDatasetParquet(
        parquet_path=parquet_path,
        x_cols=x_cols if x_cols else None,
        keep_ctx_ids=keep_ctx_ids,
        keep_street_ids=keep_street_ids,
        min_weight=min_weight,
        use_soft_labels=True
    )

    # Feature cardinalities for embeddings
    cards = categorical_cardinalities(df=ds.df, x_cols=ds.x_cols)
    feature_order = list(ds.x_cols)

    # Stratified split
    stratify_keys = get("dataset.stratify_keys", ["ctx_id", "street_id"])
    train_frac = float(get("train.train_frac", 0.8))
    train_idx, val_idx = stratified_indices(df=ds.df, group_cols=stratify_keys, train_frac=train_frac, seed=seed)
    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)

    # -------- DataLoaders --------
    batch_size = int(get("train.batch_size", get("batch_size", 2048)))
    num_workers = int(get("train.num_workers", get("num_workers", 0)))
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=population_collate_fn,  # ✅ important
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=population_collate_fn,  # ✅ important
    )

    # -------- Model --------
    model = PopulationNetLit(
        cards=cards,
        hidden_dims=get("model.hidden_dims", [64, 64]),
        dropout=float(get("model.dropout", 0.10)),
        lr=float(get("model.lr", get("lr", 1e-3))),
        weight_decay=float(get("model.weight_decay", 1e-4)),
        feature_order=feature_order,
    )

    # -------- Callbacks (checkpoint + early stopping) --------
    ckpt_dir = Path(get("train.checkpoints_dir", get("ckpt_dir", "checkpoints/popnet")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor = get("train.monitor", "val_loss")
    mode = get("train.mode", "min")
    patience = int(get("train.patience", get("patience", 3)))

    logger = TensorBoardLogger(save_dir="logs", name="popnet")  # or CSVLogger("logs", name="popnet")

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="popnet-{epoch:02d}-{" + monitor + ":.4f}",
        monitor=monitor, mode=mode,
        save_last=True, save_top_k=1,
        auto_insert_metric_name=False,
    )
    early_cb = pl.callbacks.EarlyStopping(
        monitor=monitor, mode=mode, patience=patience
    )

    # -------- Trainer --------
    max_epochs = int(get("train.max_epochs", get("max_epochs", 10)))
    accelerator = get("train.accelerator", "auto")
    devices = get("train.devices", "auto")
    precision = get("train.precision", get("precision", "16-mixed"))
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        log_every_n_steps=50,
        callbacks=[ckpt_cb, early_cb],
        deterministic=True,
        enable_progress_bar=True,
        logger=logger
    )

    # Save run artifacts (config snapshot + feature order + cards)
    try:
        # If cfg is OmegaConf, serialize it; if it's a dict, dump directly
        from omegaconf import OmegaConf
        cfg_json = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    except Exception:
        cfg_json = dict(cfg)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg_json, indent=2))
    (ckpt_dir / "feature_order.json").write_text(json.dumps(feature_order))
    (ckpt_dir / "cards.json").write_text(json.dumps(cards))

    # Optional resume
    resume_from = get("train.resume_from", get("resume_from", None))
    if resume_from:
        trainer.fit(model, train_dl, val_dl, ckpt_path=str(resume_from))
    else:
        trainer.fit(model, train_dl, val_dl)

    # pick best (fallback to last)
    best_ckpt = None
    for cb in trainer.callbacks:
        if hasattr(cb, "best_model_path") and cb.best_model_path:
            best_ckpt = cb.best_model_path
            break
    if best_ckpt is None and 'ckpt_cb' in locals() and hasattr(ckpt_cb, "last_model_path"):
        best_ckpt = ckpt_cb.last_model_path

    print(f"✅ training complete. Best checkpoint: {best_ckpt}")

    # write sidecar next to best checkpoint
    if best_ckpt:
        sidecar_path = write_popnet_sidecar(best_ckpt=best_ckpt, ds=ds, model=model)
        if sidecar_path:
            print(f"💾 wrote sidecar → {sidecar_path}")
        else:
            print("⚠️ Skipped sidecar (missing feature_order/cards)")
    return best_ckpt

if __name__ == "__main__":
    import argparse
    from ml.utils.config import load_model_config  # same helper used elsewhere

    ap = argparse.ArgumentParser()
    # accepts either a short name ("populationnet") or a path ("ml/config/base.yaml")
    ap.add_argument("--config", type=str, default="populationnet",
                    help="Model name or YAML path (resolved by load_model_config)")
    # lightweight overrides (train.* only)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--max_epochs", type=int)
    ap.add_argument("--patience", type=int)
    args = ap.parse_args()

    # Load base config (dict-like)
    cfg = load_model_config(args.config)

    # Merge CLI overrides into train.*
    train_cfg = cfg.setdefault("train", {})
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.max_epochs is not None:
        train_cfg["max_epochs"] = int(args.max_epochs)
    if args.patience is not None:
        train_cfg["patience"] = int(args.patience)

    # Run
    run_train(cfg)