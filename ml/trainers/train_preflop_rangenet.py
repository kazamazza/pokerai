# ml/train/train_preflop_rangenet.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, Dict
import json
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.datasets.preflop_rangenet import PreflopRangeDatasetParquet
from ml.datasets.utils_dataset import stratified_indices
from ml.models.preflop_rangenet import RangeNetLit                  # logits + KL version
from ml.datasets.rangenet import rangenet_collate_fn
from ml.utils.config import load_model_config               # your helper

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

def run_train_preflop(cfg: Mapping[str, Any]) -> str:
    # -------- Repro --------
    seed = int(_get(cfg, "train.seed", 42))
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)

    # -------- Dataset --------
    parquet_path = _get(cfg, "inputs.parquet") or _get(cfg, "dataset.parquet")
    if not parquet_path:
        raise ValueError("Missing inputs.parquet (or dataset.parquet) in config")

    x_cols = _get(cfg, "dataset.x_cols") or ["stack_bb","hero_pos","opener_pos","ctx","opener_action"]
    min_weight = _get(cfg, "dataset.min_weight", None)
    weight_col = _get(cfg, "dataset.weight_col", "weight")
    device = torch.device(_get(cfg, "train.device", "cpu"))

    ds = PreflopRangeDatasetParquet(
        parquet_path=parquet_path,
        x_cols=x_cols,
        weight_col=weight_col,
        min_weight=min_weight,
        device=device,
        strict_canon=True,
    )

    cards = ds.cards_info.cards
    feature_order = list(ds.feature_order)

    # -------- Split --------
    stratify_keys = _get(cfg, "dataset.stratify_keys", None)
    train_frac = float(_get(cfg, "train.train_frac", 0.8))
    n = len(ds)
    if n < 2:
        raise ValueError(f"Dataset too small: {n} rows")

    if stratify_keys:
        train_idx, val_idx = stratified_indices(ds.df, stratify_keys, train_frac, seed)
    else:
        idx = list(range(n))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        cut = max(1, int(n * train_frac))
        train_idx, val_idx = idx[:cut], idx[cut:]
        if len(val_idx) == 0:  # ensure at least one val sample
            val_idx = train_idx[-1:]
            train_idx = train_idx[:-1]

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
        hidden_dims=_get(cfg, "model.hidden_dims", [128, 128]),
        dropout=float(_get(cfg, "model.dropout", 0.10)),
        lr=float(_get(cfg, "model.lr", 1e-3)),
        weight_decay=float(_get(cfg, "model.weight_decay", 1e-4)),
        label_smoothing=float(_get(cfg, "model.label_smoothing", 0.0)),
    )

    # -------- Callbacks / Logger --------
    ckpt_dir = Path(_get(cfg, "train.checkpoints_dir", "checkpoints/rangenet_preflop"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor = _get(cfg, "train.monitor", "val_loss")
    mode = _get(cfg, "train.mode", "min")
    patience = int(_get(cfg, "train.patience", 3))

    logger_name = _get(cfg, "logging.logger", "tensorboard")
    logger = None
    if logger_name == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(
            save_dir=_get(cfg, "logging.tb_log_dir", "logs/tb"),
            name="rangenet_preflop"
        )

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="rangenet-preflop-{epoch:02d}-{" + monitor + ":.4f}",
        monitor=monitor, mode=mode,
        save_last=True, save_top_k=1,
        auto_insert_metric_name=False,
    )
    early_cb = pl.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience)

    # -------- Trainer --------
    precision = _get(cfg, "train.precision", "16-mixed")
    trainer = pl.Trainer(
        max_epochs=int(_get(cfg, "train.max_epochs", 10)),
        accelerator=_get(cfg, "train.accelerator", "auto"),
        devices=_get(cfg, "train.devices", "auto"),
        precision=precision,
        log_every_n_steps=50,
        callbacks=[ckpt_cb, early_cb],
        deterministic=True,
        enable_progress_bar=True,
        logger=logger,
    )

    # Save config snapshot & sidecar
    try:
        from omegaconf import OmegaConf
        cfg_ser = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        cfg_ser = dict(cfg)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg_ser, indent=2))
    _save_sidecar(ckpt_dir, feature_order=feature_order, id_maps=ds.id_maps(), cards=cards)

    # -------- Train (with auto-resume) --------
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
    ap.add_argument("--config", type=str, default="rangenet/preflop/dev",
                    help="Model name or YAML path resolved by load_model_config")
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--max_epochs", type=int)
    ap.add_argument("--patience", type=int)
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    train_cfg = cfg.setdefault("train", {})
    if args.batch_size is not None: train_cfg["batch_size"] = int(args.batch_size)
    if args.max_epochs is not None: train_cfg["max_epochs"] = int(args.max_epochs)
    if args.patience is not None:   train_cfg["patience"]   = int(args.patience)

    run_train_preflop(cfg)