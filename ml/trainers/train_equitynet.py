import sys
from pathlib import Path
from typing import Any, Mapping
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.models.equity_net import EquityNetLit
from ml.utils.equity_sidecar import write_equity_sidecar
from ml.datasets.equitynet import EquityDatasetParquet, equity_collate_fn
from ml.datasets.utils_dataset import stratified_indices
from ml.utils.config import load_model_config
from ml.trainers.sweep import finalize_best_artifacts, parse_scalar_from_ckpt, run_sweep
from ml.trainers.helpers import _get


def run_equity_sweep(cfg: dict):
    base_dir = Path(cfg.get("train", {}).get("checkpoints_dir", "checkpoints/equitynet"))
    res = run_sweep(
        base_cfg=cfg,
        sweep=cfg["sweep"],
        run_fn=run_train,
        score_fn=parse_scalar_from_ckpt,   # or a custom parser
        base_ckpt_dir=base_dir,
        monitor=cfg.get("train", {}).get("monitor", "val_loss"),
        max_trials=cfg.get("sweep", {}).get("max_trials"),
    )
    if res["best"]["ckpt"]:
        finalize_best_artifacts(Path(res["best"]["ckpt"]), base_dir)
    return res


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

    # Train
    resume_from = _get(cfg, "train.resume_from", None)
    if resume_from:
        trainer.fit(model, train_dl, val_dl, ckpt_path=str(resume_from))
    else:
        trainer.fit(model, train_dl, val_dl)
    # ----- after training -----
    best_ckpt = None
    for cb in trainer.callbacks:
        if hasattr(cb, "best_model_path") and cb.best_model_path:
            best_ckpt = cb.best_model_path
            break
    if best_ckpt is None and 'ckpt_cb' in locals() and getattr(ckpt_cb, "last_model_path", None):
        best_ckpt = ckpt_cb.last_model_path

    print(f"✅ training complete. Best checkpoint: {best_ckpt}")

    # Write sidecar next to the *checkpoint file* (NOT the directory)
    sidecar_path = write_equity_sidecar(best_ckpt=best_ckpt, ds=ds, model=model, model_name="EquityNet")
    if sidecar_path:
        print(f"💾 wrote sidecar → {sidecar_path}")
    else:
        print("⚠️ Skipped sidecar (missing feature_order/id_maps/cards)")

    return best_ckpt or str(ckpt_dir / "last.ckpt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="equitynet",
                    help="Model name or YAML path")
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--max_epochs", type=int)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--sweep", action="store_true",
                    help="Run hyperparameter sweep using the 'sweep' block in the config")
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    if args.batch_size: cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.max_epochs: cfg.setdefault("train", {})["max_epochs"] = int(args.max_epochs)
    if args.patience:   cfg.setdefault("train", {})["patience"]   = int(args.patience)

    # If --sweep passed OR config contains a 'sweep' block → run sweep
    if args.sweep or ("sweep" in cfg and isinstance(cfg["sweep"], dict) and cfg["sweep"]):
        run_equity_sweep(cfg)
    else:
        run_train(cfg)