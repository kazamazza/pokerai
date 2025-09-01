# train_equitynet.py
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.datasets.equitynet import EquityDatasetParquet, equity_collate_fn
from ml.datasets.utils_dataset import stratified_indices
from ml.models.equity_net import EquityNetLit
from ml.utils.config import load_model_config


def _get(cfg: Mapping[str, Any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _save_sidecar(ckpt_dir: Path, *, feature_order: Sequence[str],
                  id_maps: dict, cards: dict) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "feature_order.json").write_text(json.dumps(list(feature_order), indent=2))
    (ckpt_dir / "id_maps.json").write_text(json.dumps(id_maps, indent=2))
    (ckpt_dir / "cards.json").write_text(json.dumps(cards, indent=2))


def run_train(cfg: Mapping[str, Any]) -> str:
    # -------- Repro --------
    seed = int(_get(cfg, "train.seed", 42))
    pl.seed_everything(seed)

    # -------- Dataset --------
    parquet_path = _get(cfg, "inputs.parquet")
    if not parquet_path:
        raise ValueError("Missing inputs.parquet in config")

    x_cols = _get(cfg, "dataset.x_cols")
    y_cols = _get(cfg, "dataset.y_cols", ["p_win", "p_tie", "p_lose"])
    weight_col = _get(cfg, "dataset.weight_col", "weight")
    min_weight = _get(cfg, "dataset.min_weight", None)
    print(x_cols, y_cols, weight_col)

    ds = EquityDatasetParquet(
        parquet_path=parquet_path,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        min_weight=min_weight,
        device=torch.device("cpu"),
    )

    cards = ds.cards_info.cards
    feature_order = list(ds.feature_order)

    # Optional stratified split (e.g. by street for postflop)
    stratify_keys = _get(cfg, "dataset.stratify_keys", None)
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
        collate_fn=equity_collate_fn,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=equity_collate_fn,
    )

    # -------- Model --------
    model = EquityNetLit(
        cards=cards,
        cat_order=feature_order,
        hidden_dims=_get(cfg, "model.hidden_dims", [128, 128]),
        dropout=float(_get(cfg, "model.dropout", 0.1)),
        lr=float(_get(cfg, "model.lr", 1e-3)),
        weight_decay=float(_get(cfg, "model.weight_decay", 1e-4)),
        output_mode=_get(cfg, "model.output_mode", "triplet"),  # default triplet
    )

    # -------- Callbacks / Logger --------
    ckpt_dir = Path(_get(cfg, "train.checkpoints_dir", "checkpoints/equitynet"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor = _get(cfg, "train.monitor", "val_loss")
    mode = _get(cfg, "train.mode", "min")
    patience = int(_get(cfg, "train.patience", 5))

    logger = pl.loggers.TensorBoardLogger(
        save_dir=_get(cfg, "logging.tb_log_dir", "logs/tb"),
        name="equitynet",
    )

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="equitynet-{epoch:02d}-{" + monitor + ":.4f}",
        monitor=monitor, mode=mode,
        save_last=True, save_top_k=1,
        auto_insert_metric_name=False,
    )
    early_cb = pl.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience)

    # -------- Trainer --------
    trainer = pl.Trainer(
        max_epochs=int(_get(cfg, "train.max_epochs", 10)),
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

    # Train
    resume_from = _get(cfg, "train.resume_from", None)
    if resume_from:
        trainer.fit(model, train_dl, val_dl, ckpt_path=str(resume_from))
    else:
        trainer.fit(model, train_dl, val_dl)

    print(f"✅ training complete. Best checkpoint: {ckpt_cb.best_model_path}")
    return ckpt_cb.best_model_path or str(ckpt_dir / "last.ckpt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="equitynet",
                    help="Model name or YAML path")
    # quick CLI overrides
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--max_epochs", type=int)
    ap.add_argument("--patience", type=int)
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    if args.batch_size: cfg.setdefault("train", {})["batch_size"] = args.batch_size
    if args.max_epochs: cfg.setdefault("train", {})["max_epochs"] = args.max_epochs
    if args.patience:   cfg.setdefault("train", {})["patience"]   = args.patience

    run_train(cfg)