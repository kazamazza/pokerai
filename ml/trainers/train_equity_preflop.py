# ml/trainers/train_equity_preflop.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.datasets.equitynet import EquityDatasetParquet, equity_collate_fn
from ml.models.equity_net import EquityNetLit
from ml.utils.config import load_model_config


# Project imports – adjust paths if your package layout differs


def _get(cfg: Dict[str, Any], path: str, default=None):
    cur: Any = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _load_filtered_df_for_split(
    parquet_path: Path,
    keep_values: Dict[str, Sequence[Any]] | None,
    min_weight: float | None,
) -> pd.DataFrame:
    """
    Read the same parquet the dataset will use and apply the exact same high-level filters
    so row order matches dataset indexing (we rely on reset_index(drop=True)).
    """
    df = pd.read_parquet(parquet_path)

    if keep_values:
        for k, vals in keep_values.items():
            if k in df.columns:
                df = df[df[k].isin(list(vals))]

    if min_weight is not None and "weight" in df.columns:
        df = df[df["weight"] >= float(min_weight)]

    df = df.reset_index(drop=True)
    return df


def _scenario_split_indices(
    df: pd.DataFrame,
    scenario_keys: Sequence[str],
    train_frac: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    Split by unique (stack_bb, hero_pos, opener_action) scenarios so that entire scenarios
    land either in train or val. Returns row indices into df.
    """
    if not scenario_keys:
        raise ValueError("scenario_keys must be a non-empty list")

    # Group rows by scenario key
    gb = df.groupby(list(scenario_keys), sort=False, as_index=False).indices  # dict[key_tuple] -> index array
    keys = list(gb.keys())

    # Shuffle keys deterministically
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(keys), generator=rng).tolist()
    keys_shuffled = [keys[i] for i in perm]

    n_train_keys = int(round(train_frac * len(keys_shuffled)))
    train_keys = set(keys_shuffled[:n_train_keys])

    train_idx: List[int] = []
    val_idx: List[int] = []

    for k, idx_arr in gb.items():
        if k in train_keys:
            train_idx.extend(idx_arr.tolist())
        else:
            val_idx.extend(idx_arr.tolist())

    # Keep original order within each split
    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


def run_train(config_name_or_path: str, **cli_overrides):
    # -------- Load config --------
    cfg = load_model_config(config_name_or_path)

    # Apply optional CLI overrides (kept minimal)
    if "batch_size" in cli_overrides and cli_overrides["batch_size"] is not None:
        cfg.setdefault("train", {})["batch_size"] = int(cli_overrides["batch_size"])
    if "max_epochs" in cli_overrides and cli_overrides["max_epochs"] is not None:
        cfg.setdefault("train", {})["max_epochs"] = int(cli_overrides["max_epochs"])
    if "patience" in cli_overrides and cli_overrides["patience"] is not None:
        cfg.setdefault("train", {})["patience"] = int(cli_overrides["patience"])

    # -------- Dataset config --------
    parquet_path = Path(_get(cfg, "inputs.parquet", _get(cfg, "dataset.parquet")))
    if not parquet_path or not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    x_cols = _get(cfg, "dataset.x_cols", ["stack_bb", "hero_pos", "opener_action", "hand_id"])
    y_cols = _get(cfg, "dataset.y_cols", ["y_win", "y_tie", "y_lose"])
    weight_col = _get(cfg, "dataset.weight_col", "weight")
    keep_values = _get(cfg, "dataset.keep_values", None)  # optional dict of filters
    min_weight = _get(cfg, "dataset.min_weight", None)

    # Scenario split keys (preflop: no board_cluster_id)
    scenario_keys = _get(cfg, "dataset.scenario_keys", ["stack_bb", "hero_pos", "opener_action"])

    # -------- Reproducibility --------
    seed = int(_get(cfg, "train.seed", 42))
    pl.seed_everything(seed)

    # -------- Build split indices (by scenario) --------
    df_for_split = _load_filtered_df_for_split(parquet_path, keep_values, min_weight)
    train_frac = float(_get(cfg, "train.train_frac", 0.8))
    train_idx, val_idx = _scenario_split_indices(
        df_for_split, scenario_keys=scenario_keys, train_frac=train_frac, seed=seed
    )

    # -------- Build Dataset objects (must match filtering) --------
    ds = EquityDatasetParquet(
        parquet_path=parquet_path,
        x_cols=x_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        keep_values=keep_values,
        min_weight=min_weight,
        device=None,  # let DataLoader move to device later
    )

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    # -------- DataLoaders --------
    batch_size = int(_get(cfg, "train.batch_size", 2048))
    num_workers = int(_get(cfg, "train.num_workers", 0))
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=equity_collate_fn
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=equity_collate_fn
    )

    # -------- Model --------
    cards = ds.cards()              # dict: {feature_name -> cardinality}
    feature_order = ds.feature_order  # order of x_cols used by dataset

    model = EquityNetLit(
        cards=cards,
        cat_order=feature_order,
        hidden_dims=_get(cfg, "model.hidden_dims", [128, 128]),
        dropout=float(_get(cfg, "model.dropout", 0.10)),
        lr=float(_get(cfg, "model.lr", 1e-3)),
        weight_decay=float(_get(cfg, "model.weight_decay", 1e-4)),
    )

    # -------- Callbacks & Logger --------
    ckpt_dir = Path(_get(cfg, "train.checkpoints_dir", "checkpoints/equity_pre"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor = _get(cfg, "train.monitor", "val_loss")
    mode = _get(cfg, "train.mode", "min")
    patience = int(_get(cfg, "train.patience", 3))

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="equity_pre-{epoch:02d}-{" + monitor + ":.4f}",
        monitor=monitor, mode=mode,
        save_last=True, save_top_k=1,
        auto_insert_metric_name=False,
    )
    early_cb = pl.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience)

    logger = TensorBoardLogger(
        save_dir="logs",
        name="equity_preflop",
        default_hp_metric=False,
    )

    # -------- Trainer --------
    trainer = pl.Trainer(
        max_epochs=int(_get(cfg, "train.max_epochs", 10)),
        accelerator=_get(cfg, "train.accelerator", "auto"),
        devices=_get(cfg, "train.devices", "auto"),
        precision=_get(cfg, "train.precision", "16-mixed"),
        log_every_n_steps=50,
        callbacks=[ckpt_cb, early_cb],
        logger=logger,
        deterministic=True,
        enable_progress_bar=True,
    )

    # -------- Optional resume --------
    resume_from = _get(cfg, "train.resume_from", None)
    if resume_from:
        trainer.fit(model, train_dl, val_dl, ckpt_path=str(resume_from))
    else:
        trainer.fit(model, train_dl, val_dl)

    print("✅ training complete")
    print(f" Best ckpt: {ckpt_cb.best_model_path or '(n/a)'}  ({monitor}={ckpt_cb.best_model_score})")
    print(f" Last ckpt: {Path(ckpt_dir, 'last.ckpt')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="equitynet_preflop",
                    help="Config name or YAML path (expects dataset/model/train sections)")
    # small optional overrides
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--max_epochs", type=int)
    ap.add_argument("--patience", type=int)
    args = ap.parse_args()

    run_train(
        args.config,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()