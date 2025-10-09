from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Mapping, Any
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.datasets.population import PopulationDatasetParquet, population_collate_fn
from ml.datasets.utils_dataset import categorical_cardinalities, stratified_indices
from ml.models.population_net import PopulationNetLit
from ml.trainers.helpers import _get
from ml.trainers.sweep import parse_scalar_from_ckpt, run_sweep, finalize_best_artifacts
from ml.utils.popnet_sidecar import write_popnet_sidecar


def run_train_population_sweep(cfg: Mapping[str, Any]) -> dict:
    """
    Launch a grid sweep for PopulationNet using the shared sweep harness.
    Expects cfg['sweep'] to be a dict of dotted keys -> lists.
    Returns the sweep results dict (with 'best' entry, etc.).
    """
    sweep_cfg = _get(cfg, "sweep", None)
    if not isinstance(sweep_cfg, dict) or not sweep_cfg:
        raise ValueError("No 'sweep' block in config.")

    base_ckpt_dir = Path(_get(cfg, "train.checkpoints_dir", "checkpoints/popnet"))
    base_ckpt_dir.mkdir(parents=True, exist_ok=True)

    results = run_sweep(
        base_cfg=dict(cfg),                     # pass full base config
        sweep=sweep_cfg,                        # dotted keys grid
        run_fn=run_train,                       # single-trial trainer -> ckpt path
        score_fn=parse_scalar_from_ckpt,        # ckpt -> scalar (lower better)
        base_ckpt_dir=base_ckpt_dir,            # where trial dirs live
        monitor=_get(cfg, "train.monitor", "val_loss"),
        max_trials=_get(cfg, "sweep.max_trials", None),
    )

    # Copy best.ckpt / best_sidecar.json / best_config.json into base_ckpt_dir
    finalize_best_artifacts(Path(results["best"]["ckpt"]), base_ckpt_dir)
    return results

def run_train(cfg: Mapping[str, Any]) -> str:
    """Train one PopulationNet trial; returns best checkpoint path."""
    # -------- Repro --------
    seed = int(_get(cfg, "train.seed", 42))
    pl.seed_everything(seed)

    # -------- Dataset --------
    parquet_path     = _get(cfg, "inputs.parquet", "data/datasets/populationnet_nl10.parquet")
    keep_ctx_ids     = _get(cfg, "dataset.keep_ctx_ids", None)
    keep_street_ids  = _get(cfg, "dataset.keep_street_ids", None)
    min_weight       = _get(cfg, "dataset.min_weight", None)
    x_cols_override  = _get(cfg, "dataset.x_cols", None)

    ds = PopulationDatasetParquet(
        parquet_path=parquet_path,
        x_cols=x_cols_override,            # None -> dataset default order
        keep_ctx_ids=keep_ctx_ids,
        keep_street_ids=keep_street_ids,
        min_weight=min_weight,
        use_soft_labels=True,
    )

    # Feature space
    cards         = categorical_cardinalities(df=ds.df, x_cols=ds.x_cols)
    feature_order = list(ds.x_cols)

    # -------- Split --------
    stratify_keys = _get(cfg, "dataset.stratify_keys", ["ctx_id", "street_id"])
    train_frac    = float(_get(cfg, "train.train_frac", 0.8))
    train_idx, val_idx = stratified_indices(ds.df, stratify_keys, train_frac, seed)
    train_ds, val_ds   = Subset(ds, train_idx), Subset(ds, val_idx)

    # -------- DataLoaders --------
    batch_size   = int(_get(cfg, "train.batch_size", 2048))
    num_workers  = int(_get(cfg, "train.num_workers", 0))
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=population_collate_fn
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=population_collate_fn
    )

    # -------- Model --------
    model = PopulationNetLit(
        cards=cards,
        feature_order=feature_order,
        hidden_dims=_get(cfg, "model.hidden_dims", [64, 64]),
        dropout=float(_get(cfg, "model.dropout", 0.10)),
        lr=float(_get(cfg, "model.lr", 1e-3)),
        weight_decay=float(_get(cfg, "model.weight_decay", 1e-4)),
        # optional warmup/cosine knobs (present = enabled)
        warmup_steps=int(_get(cfg, "model.warmup_steps", 0)),
        max_steps=int(_get(cfg, "model.max_steps", 0)),
        min_lr_scale=float(_get(cfg, "model.min_lr_scale", 0.05)),
        use_soft_labels=True,
    )

    # -------- Logging / Ckpt --------
    ckpt_dir = Path(_get(cfg, "train.checkpoints_dir", "checkpoints/popnet"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor  = _get(cfg, "train.monitor", "val_loss")
    mode     = _get(cfg, "train.mode", "min")
    patience = int(_get(cfg, "train.patience", 3))

    logger  = TensorBoardLogger(save_dir=_get(cfg, "logging.tb_log_dir", "logs/tb"), name="popnet")
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="popnet-{epoch:02d}-{" + monitor + ":.4f}",
        monitor=monitor, mode=mode, save_last=True, save_top_k=1, auto_insert_metric_name=False
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

    # Snapshot helpful run artifacts
    try:
        from omegaconf import OmegaConf
        cfg_json = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        cfg_json = dict(cfg)
    (ckpt_dir / "feature_order.json").write_text(json.dumps(feature_order))
    (ckpt_dir / "cards.json").write_text(json.dumps(cards))
    (ckpt_dir / "config.json").write_text(json.dumps(cfg_json, indent=2))

    # Train (resume if asked)
    resume_from = _get(cfg, "train.resume_from", None)
    trainer.fit(model, train_dl, val_dl, ckpt_path=str(resume_from) if resume_from else None)

    # Pick best (fallback to last)
    best_ckpt = None
    for cb in trainer.callbacks:
        if getattr(cb, "best_model_path", ""):
            best_ckpt = cb.best_model_path
            break
    if not best_ckpt and getattr(ckpt_cb, "last_model_path", ""):
        best_ckpt = ckpt_cb.last_model_path

    print(f"✅ training complete. Best checkpoint: {best_ckpt}")

    # Sidecar

    if best_ckpt:
        sidecar_path = write_popnet_sidecar(best_ckpt=best_ckpt, ds=ds, model=model)
        if sidecar_path:
            print(f"💾 wrote sidecar → {sidecar_path}")
        else:
            print("⚠️ skipped sidecar (missing schema)")

    return best_ckpt or ""

if __name__ == "__main__":
    import argparse
    from ml.utils.config import load_model_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="populationnet", help="Model name or YAML path")
    ap.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep defined in config.sweep")
    # light overrides
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--max_epochs", type=int)
    ap.add_argument("--patience", type=int)
    args = ap.parse_args()

    cfg = load_model_config(args.config)

    # Merge CLI overrides into cfg.train.*
    tr = cfg.setdefault("train", {})
    if args.batch_size is not None: tr["batch_size"] = int(args.batch_size)
    if args.max_epochs is not None: tr["max_epochs"] = int(args.max_epochs)
    if args.patience   is not None: tr["patience"]   = int(args.patience)

    if args.sweep:
        out = run_train_population_sweep(cfg)
        print(f"🏁 sweep done. Best: {out.get('best')}")
    else:
        ckpt = run_train(cfg)
        print(f"✅ done. Best ckpt: {ckpt}")