from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.datasets.evnet import EVParquetDataset, ev_collate_fn, _import_symbol
from ml.datasets.utils_dataset import stratified_indices
from ml.models.evnet import EVLit, EVNetConfig, EVNet
from ml.trainers.sweep import run_sweep, parse_scalar_from_ckpt, finalize_best_artifacts
from ml.utils.config import load_model_config
from ml.utils.ev_sidecar import write_ev_sidecar

# ───────────────────────────────────────────────────────────────────────────────
# Utils
# ───────────────────────────────────────────────────────────────────────────────
def _get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    obj = cfg
    for part in key.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return default
        obj = obj[part]
    return obj


def _resolve_paths_from_cfg(cfg: Dict[str, Any]) -> Tuple[str, str, str]:
    """Returns (parquet_path, ckpt_dir, tb_dir)."""
    parquet = _get(cfg, "inputs.parquet") or _get(cfg, "paths.parquet_path")
    if not parquet:
        raise ValueError("Missing inputs.parquet or paths.parquet_path in config.")
    ckpt_dir = _get(cfg, "train.checkpoints_dir", "checkpoints/ev")
    tb_dir = _get(cfg, "logging.tb_log_dir", "logs/tb")
    return str(parquet), str(ckpt_dir), str(tb_dir)


def _resolve_action_vocab(cfg: Dict[str, Any], parquet_path: str) -> List[str]:
    # 1) Explicit in cfg
    av = _get(cfg, "model.action_vocab")
    if isinstance(av, (list, tuple)) and av:
        return [str(x) for x in av]

    # 2) Try canonical module
    try:
        from ml.models.vocab_actions import ROOT_ACTION_VOCAB, FACING_ACTION_VOCAB, PREFLOP_ACTION_VOCAB  # type: ignore
        cols = set(pd.read_parquet(parquet_path, columns=None).columns)
        if "street" in cols:
            # Facing EV parquet includes size_frac
            if "size_frac" in cols:
                return list(FACING_ACTION_VOCAB)
            # Root EV parquet has CHECK/BET tokens
            return list(ROOT_ACTION_VOCAB)
        # Preflop
        return list(PREFLOP_ACTION_VOCAB)
    except Exception:
        pass

    # 3) Infer from ev_* columns
    df_cols = list(pd.read_parquet(parquet_path, columns=None).columns)
    tok_cols = [c for c in df_cols if c.startswith("ev_")]
    if not tok_cols:
        raise RuntimeError("Cannot infer action_vocab (no model.action_vocab and no ev_* columns).")
    return [c[3:] for c in tok_cols]


def _make_datasets(
    parquet_path: str,
    cfg: Dict[str, Any],
    action_vocab: List[str],
) -> Tuple[EVParquetDataset, EVParquetDataset]:
    d_cfg = cfg.get("dataset", {}) or {}
    x_cols: List[str] = list(d_cfg.get("x_cols", []))
    cont_cols: List[str] = list(d_cfg.get("cont_cols", []))
    y_cols = d_cfg.get("y_cols")  # usually None → infer from action_vocab
    weight_col = d_cfg.get("weight_col", "weight")
    stratify_keys: List[str] = list(d_cfg.get("stratify_keys", []))
    seed = int(d_cfg.get("seed", 42))
    train_frac = float(_get(cfg, "train.train_frac", 0.8))

    full_ds = EVParquetDataset(
        parquet_path=parquet_path,
        action_vocab=action_vocab,
        x_cols=x_cols,
        cont_cols=cont_cols,
        y_cols=y_cols,
        weight_col=weight_col,
    )

    n = len(full_ds.df)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)

    if stratify_keys:
        train_idx, val_idx = stratified_indices(full_ds.df, stratify_keys, train_frac, seed)
    else:
        rng.shuffle(idx)
        cut = int(n * train_frac)
        train_idx, val_idx = idx[:cut], idx[cut:]

    train_df = full_ds.df.iloc[train_idx].reset_index(drop=True)
    val_df   = full_ds.df.iloc[val_idx].reset_index(drop=True)

    # Reuse id_maps so encodings are identical across splits
    id_maps = getattr(full_ds, "id_maps", None)

    train_ds = EVParquetDataset(
        dataframe=train_df,
        action_vocab=action_vocab,
        x_cols=x_cols,
        cont_cols=cont_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        id_maps=id_maps,
    )
    val_ds = EVParquetDataset(
        dataframe=val_df,
        action_vocab=action_vocab,
        x_cols=x_cols,
        cont_cols=cont_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        id_maps=id_maps,
    )
    return train_ds, val_ds


def _make_trainer(cfg: Dict[str, Any], ckpt_dir: str, tb_dir: str, resume_from: Optional[str]) -> pl.Trainer:
    t = cfg.get("train", {}) or {}
    max_epochs = int(t.get("max_epochs", 20))
    patience = int(t.get("patience", 5))
    monitor = str(t.get("monitor", "val_loss"))
    mode = str(t.get("mode", "min"))
    accelerator = t.get("accelerator", "auto")
    devices = t.get("devices", "auto")
    precision = t.get("precision", "16-mixed")
    num_sanity_val_steps = int(t.get("num_sanity_val_steps", 2))
    grad_clip = float(t.get("grad_clip", 0.0))
    log_every_n = int(t.get("log_every_n_steps", 50))

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    logger = TensorBoardLogger(save_dir=tb_dir, name="ev")

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="ev-{epoch:02d}-{%s:.4f}" % monitor,
        monitor=monitor,
        mode=mode,
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    es_cb = EarlyStopping(monitor=monitor, mode=mode, patience=patience, verbose=True)
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        default_root_dir=ckpt_dir,
        logger=logger,
        callbacks=[ckpt_cb, es_cb, lr_cb],
        gradient_clip_val=grad_clip,
        num_sanity_val_steps=num_sanity_val_steps,
        enable_progress_bar=True,
        log_every_n_steps=log_every_n,
        deterministic=False,
        resume_from_checkpoint=resume_from,  # accepted by older PL; for newer, pass via fit(ckpt_path=…)
    )
    return trainer


def run_train(cfg: Mapping[str, Any]) -> str:
    # seed
    seed = int(cfg.get("train", {}).get("seed", 42))
    pl.seed_everything(seed)

    # paths
    parquet_path = (cfg.get("inputs", {}) or {}).get("parquet") or (cfg.get("paths", {}) or {}).get("parquet_path")
    if not parquet_path:
        raise ValueError("Missing parquet path (inputs.parquet or paths.parquet_path)")

    # vocab resolve
    model_cfg = cfg.get("model", {}) or {}
    action_vocab = model_cfg.get("action_vocab")
    if action_vocab is None and "action_vocab_import" in model_cfg:
        action_vocab = _import_symbol(model_cfg["action_vocab_import"])
    if not action_vocab:
        raise ValueError("Provide model.action_vocab or model.action_vocab_import")

    ds_cfg = cfg.get("dataset", {}) or {}
    x_cols    = ds_cfg.get("x_cols") or []
    cont_cols = ds_cfg.get("cont_cols") or []
    y_cols    = ds_cfg.get("y_cols") or None
    weight_col = ds_cfg.get("weight_col", "weight")
    strat_keys = ds_cfg.get("stratify_keys", None)

    # dataset
    ds = EVParquetDataset(
        parquet_path=parquet_path,
        action_vocab=action_vocab,
        x_cols=x_cols,
        cont_cols=cont_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        cache_arrays=True,
    )

    # split
    train_frac = float(cfg.get("train", {}).get("train_frac", 0.8))
    if strat_keys:
        train_idx, val_idx = stratified_indices(ds.df, strat_keys, train_frac, seed)
    else:
        n = len(ds); perm = np.random.RandomState(seed).permutation(n); cut = int(n * train_frac)
        train_idx, val_idx = perm[:cut].tolist(), perm[cut:].tolist()

    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)

    # loaders
    batch_size  = int(cfg.get("train", {}).get("batch_size", 2048))
    num_workers = int(cfg.get("train", {}).get("num_workers", 0))
    pin_memory  = bool(cfg.get("train", {}).get("pin_memory", True))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory, collate_fn=ev_collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=ev_collate_fn)

    # model config (infer cont_dim from one sample)
    sample = ds[0]
    ev_config = {
        "cat_cardinalities": [len(ds.id_maps[c]) for c in ds.x_cols],
        "cont_dim": int(sample["x_cont"].shape[-1]),
        "action_vocab": list(ds.action_vocab),
        "hidden_dims": cfg.get("model", {}).get("hidden_dims", [256, 256]),
        "dropout": float(cfg.get("model", {}).get("dropout", 0.10)),
        "max_emb_dim": int(cfg.get("model", {}).get("max_emb_dim", 32)),
    }
    lit = EVLit(
        config=ev_config,
        lr=float(cfg.get("model", {}).get("lr", 1e-3)),
        weight_decay=float(cfg.get("model", {}).get("weight_decay", 1e-4)),
    )

    # trainer
    ckpt_dir = Path(cfg.get("train", {}).get("checkpoints_dir", "checkpoints/evnet"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor  = cfg.get("train", {}).get("monitor", "val_loss")
    mode     = cfg.get("train", {}).get("mode", "min")
    patience = int(cfg.get("train", {}).get("patience", 5))

    logger = pl.loggers.TensorBoardLogger(save_dir=cfg.get("logging", {}).get("tb_log_dir", "logs/tb"), name="evnet")
    ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath=str(ckpt_dir), filename="evnet-{epoch:02d}-{" + monitor + ":.4f}",
                                           monitor=monitor, mode=mode, save_last=True, save_top_k=1, auto_insert_metric_name=False)
    early_cb = pl.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience)

    trainer = pl.Trainer(
        max_epochs=int(cfg.get("train", {}).get("max_epochs", 20)),
        accelerator=cfg.get("train", {}).get("accelerator", "auto"),
        devices=cfg.get("train", {}).get("devices", "auto"),
        precision=cfg.get("train", {}).get("precision", "16-mixed"),
        log_every_n_steps=50,
        callbacks=[ckpt_cb, early_cb],
        deterministic=True,
        enable_progress_bar=True,
        logger=logger,
    )

    resume = cfg.get("train", {}).get("resume_from", None)
    trainer.fit(lit, train_dl, val_dl, ckpt_path=str(resume) if resume else None)  # single fit

    # find best
    best_ckpt = None
    for cb in trainer.callbacks:
        p = getattr(cb, "best_model_path", "")
        if p:
            best_ckpt = p; break
    best_ckpt = best_ckpt or getattr(ckpt_cb, "last_model_path", None) or str(ckpt_dir / "last.ckpt")

    # sidecar (use your existing writer)
    meta = {
        "model_name": "EVNet",
        "action_vocab": list(ds.action_vocab),
        "x_cols": list(ds.x_cols),
        "cont_cols": list(ds.cont_cols_raw),
        "id_maps": ds.id_maps,
        "notes": "Auto-written (minimal trainer)",
        "units": "bb",                              # WHY: remove EV unit ambiguity
        "split": cfg.get("model", {}).get("split"), # optional "preflop"|"root"|"facing"
    }

    write_ev_sidecar(best_ckpt, meta, filename="best_sidecar.json")
    return best_ckpt


# ───────────────────────────────────────────────────────────────────────────────
# Sweep (same style as equity)
# ───────────────────────────────────────────────────────────────────────────────
def run_ev_sweep(cfg: Dict[str, Any]):
    base_dir = Path(_get(cfg, "train.checkpoints_dir", "checkpoints/ev"))
    res = run_sweep(
        base_cfg=cfg,
        sweep=cfg["sweep"],
        run_fn=run_train,
        score_fn=parse_scalar_from_ckpt,
        base_ckpt_dir=base_dir,
        monitor=_get(cfg, "train.monitor", "val_loss"),
        max_trials=_get(cfg, "sweep.max_trials"),
    )
    if res.get("best", {}).get("ckpt"):
        finalize_best_artifacts(Path(res["best"]["ckpt"]), base_dir)
    return res

# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="ev/preflop/base",
        help='Either "model/variant/profile" (e.g., ev/postflop_root/base) or a direct YAML path.',
    )
    ap.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep using cfg.sweep.")
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--max_epochs", type=int)
    ap.add_argument("--patience", type=int)
    args = ap.parse_args()

    # Load cfg (same resolution semantics you use elsewhere)
    cfg_arg = args.config.strip()
    if cfg_arg.endswith(".yaml"):
        cfg = load_model_config(path=cfg_arg)
    else:
        parts = cfg_arg.split("/")
        if len(parts) == 3:
            model, variant, profile = parts
        elif len(parts) == 2:
            model, variant = parts
            profile = "base"
        else:
            # fallback to repo path
            cfg = load_model_config(path=f"ml/config/{cfg_arg}.yaml")
            model = variant = profile = ""
        if "cfg" not in locals():
            cfg = load_model_config(model=model, variant=variant, profile=profile)

    # Lightweight CLI overrides
    if args.batch_size:
        cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.max_epochs:
        cfg.setdefault("train", {})["max_epochs"] = int(args.max_epochs)
    if args.patience:
        cfg.setdefault("train", {})["patience"] = int(args.patience)

    # Sweep or single run
    if args.sweep and isinstance(cfg.get("sweep"), dict) and cfg["sweep"]:
        run_ev_sweep(cfg)
    else:
        print("No sweep.")


if __name__ == "__main__":
    main()