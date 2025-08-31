from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, List

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.loggers import TensorBoardLogger

from ml.datasets.postflop_rangenet import PostflopRangeDatasetParquet
from ml.utils.config import load_model_config

from ml.datasets.rangenet import rangenet_collate_fn


# ----------------- small helpers -----------------
def _get(cfg: Mapping[str, Any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _save_sidecar(ckpt_dir: Path, *, feature_order: Sequence[str],
                  id_maps: Dict[str, Dict[str, int]], cards: Dict[str, int]) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "feature_order.json").write_text(json.dumps(list(feature_order), indent=2))
    (ckpt_dir / "id_maps.json").write_text(json.dumps(id_maps, indent=2))
    (ckpt_dir / "cards.json").write_text(json.dumps(cards, indent=2))


def stratified_indices(df, keys: Sequence[str], train_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    """
    Very small stratified splitter:
      - groups by `keys`
      - per group, takes the first N*f as train (after a stable shuffle by seed)
    Assumes df is the dataframe stored at dataset.df (not huge).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = list(range(len(df)))
    # attach original index
    tmp = df.copy()
    tmp["__idx__"] = idx

    # shuffle within each group (but deterministically)
    train_idx, val_idx = [], []
    for _, g in tmp.groupby(list(keys), dropna=False):
        ids = g["__idx__"].to_numpy()
        rng.shuffle(ids)
        cut = int(round(len(ids) * float(train_frac)))
        train_idx.extend(ids[:cut].tolist())
        val_idx.extend(ids[cut:].tolist())
    return train_idx, val_idx


# ----------------- Model (same as preflop) -----------------
# We reuse the same RangeNet / RangeNetLit you already have
from ml.models.rangenet import RangeNetLit  # adjust import path to where you defined it


# ----------------- Trainer -----------------
def run_train(cfg: Mapping[str, Any]) -> str:
    # -------- Repro --------
    seed = int(_get(cfg, "train.seed", 42))
    pl.seed_everything(seed)

    # -------- Dataset --------
    parquet_path = _get(cfg, "inputs.parquet") or _get(cfg, "dataset.parquet")
    if not parquet_path:
        raise ValueError("Missing inputs.parquet (or dataset.parquet) in config")

    x_cols = _get(cfg, "dataset.x_cols")
    # If user doesn't specify, fall back to dataset locked schema
    if not x_cols:
        x_cols = list(getattr(PostflopRangeDatasetParquet, "DEFAULT_X", [
            "stack_bb","pot_bb","hero_pos","ip_pos","oop_pos","street","ctx","board_cluster"
        ]))

    min_weight = _get(cfg, "dataset.min_weight", None)
    weight_col = _get(cfg, "dataset.weight_col", "weight")

    ds = PostflopRangeDatasetParquet(
        parquet_path=parquet_path,
        x_cols=x_cols,
        weight_col=weight_col,
        min_weight=min_weight,
        device=torch.device("cpu"),
        strict_canon=bool(_get(cfg, "dataset.strict_canon", True)),
    )

    cards = ds.cards_info.cards
    feature_order = list(ds.feature_order)

    # -------- Split (stratify by street by default) --------
    stratify_keys = _get(cfg, "dataset.stratify_keys", ["street"])
    train_frac = float(_get(cfg, "train.train_frac", 0.8))
    if stratify_keys:
        train_idx, val_idx = stratified_indices(ds.df, stratify_keys, train_frac, seed)
    else:
        n = len(ds)
        cut = int(n * train_frac)
        idx = list(range(n))
        train_idx, val_idx = idx[:cut], idx[cut:]

    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)

    # -------- DataLoaders --------
    batch_size = int(_get(cfg, "train.batch_size", 1024))
    num_workers = int(_get(cfg, "train.num_workers", 0))
    pin_memory = bool(_get(cfg, "train.pin_memory", True))

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=rangenet_collate_fn,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=rangenet_collate_fn,
    )

    # -------- Model --------
    model = RangeNetLit(
        cards=cards,
        feature_order=feature_order,
        hidden_dims=_get(cfg, "model.hidden_dims", [256, 256]),
        dropout=float(_get(cfg, "model.dropout", 0.10)),
        lr=float(_get(cfg, "model.lr", 1e-3)),
        weight_decay=float(_get(cfg, "model.weight_decay", 1e-4)),
    )

    # -------- Callbacks / Logger --------
    ckpt_dir = Path(_get(cfg, "train.checkpoints_dir", "checkpoints/rangenet_postflop"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor = _get(cfg, "train.monitor", "val_loss")
    mode = _get(cfg, "train.mode", "min")
    patience = int(_get(cfg, "train.patience", 3))

    logger_name = _get(cfg, "logging.logger", "tensorboard")
    if logger_name == "tensorboard":
        logger = TensorBoardLogger(save_dir=_get(cfg, "logging.tb_log_dir", "logs/tb"), name="rangenet_postflop")
    else:
        logger = False

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="rangenet_postflop-{epoch:02d}-{" + monitor + ":.4f}",
        monitor=monitor, mode=mode,
        save_last=True, save_top_k=1,
        auto_insert_metric_name=False,
    )
    early_cb = pl.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience)

    # -------- Trainer --------
    trainer = pl.Trainer(
        max_epochs=int(_get(cfg, "train.max_epochs", 12)),
        accelerator=_get(cfg, "train.accelerator", "auto"),
        devices=_get(cfg, "train.devices", "auto"),
        precision=_get(cfg, "train.precision", "16-mixed"),
        log_every_n_steps=50,
        callbacks=[ckpt_cb, early_cb],
        deterministic=True,
        enable_progress_bar=True,
        logger=logger,
    )

    # Save config snapshot & sidecar pre-training
    try:
        from omegaconf import OmegaConf
        cfg_ser = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        cfg_ser = dict(cfg)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg_ser, indent=2))
    _save_sidecar(ckpt_dir, feature_order=feature_order, id_maps=ds.id_maps(), cards=cards)

    # Train (resume optional)
    resume_from = _get(cfg, "train.resume_from", None)
    if resume_from:
        trainer.fit(model, train_dl, val_dl, ckpt_path=str(resume_from))
    else:
        trainer.fit(model, train_dl, val_dl)

    print(f"✅ training complete. Best checkpoint: {ckpt_cb.best_model_path}")
    return ckpt_cb.best_model_path or str(ckpt_dir / "last.ckpt")


# ----------------- CLI -----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="rangenet/postflop/dev",
                    help="Model name or YAML path (resolved by load_model_config)")
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--max_epochs", type=int)
    ap.add_argument("--patience", type=int)
    args = ap.parse_args()

    cfg = load_model_config(args.config)

    train_cfg = cfg.setdefault("train", {})
    if args.batch_size is not None: train_cfg["batch_size"] = int(args.batch_size)
    if args.max_epochs is not None: train_cfg["max_epochs"] = int(args.max_epochs)
    if args.patience is not None:   train_cfg["patience"]   = int(args.patience)

    run_train(cfg)