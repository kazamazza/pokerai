from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, Dict, Optional
import json
from warnings import warn

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.datasets.preflop_rangenet import PreflopRangeDatasetParquet
from ml.datasets.utils_dataset import stratified_indices
from ml.models.preflop_rangenet import RangeNetLit, rangenet_preflop_collate_fn  # logits + KL version
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
        strict_canon=True,  # keep strict, the dataset will tell you if it needs relaxing
        debug=True,
    )

    n_total = len(ds)
    if n_total == 0:
        raise RuntimeError(
            "Preflop dataset is empty after canonicalization/filters. "
            "Try strict_canon=False once, or inspect the debug counts to align ctx/positions/actions."
        )
    print(f"[trainer] preflop rows after filters: {n_total}")

    cards = ds.cards_info.cards
    feature_order = list(ds.feature_order)

    def _as_dataframe_for_stratify(ds) -> Optional[pd.DataFrame]:
        """
        Best-effort: get a pandas frame we can use for stratification.
        Priority:
          1) ds.dataframe() or ds.df if present (some of your datasets expose it)
          2) rebuild a minimal frame from encoded X + encoders + feature_order
        """
        # 1) direct access
        if hasattr(ds, "df"):
            try:
                df = getattr(ds, "df")
                if isinstance(df, pd.DataFrame) and len(df) == len(ds):
                    return df.copy()
            except Exception:
                pass
        if hasattr(ds, "dataframe") and callable(getattr(ds, "dataframe")):
            try:
                df = ds.dataframe()
                if isinstance(df, pd.DataFrame) and len(df) == len(ds):
                    return df.copy()
            except Exception:
                pass

        # 2) reconstruct minimally from encoders if available
        try:
            X = ds._X  # (N, F) int64
            feats = list(ds.feature_order)  # names
            id_maps = ds.id_maps()  # {feat: {raw->id}}
            # Build reverse maps id->raw (string)
            rev = {k: {int(v): str(k2) for k2, v in mapping.items()} for k, mapping in id_maps.items()}
            N, F = X.shape
            cols = {}
            for j, feat in enumerate(feats):
                card = int(ds.cards_info.cards.get(feat, 0))
                # unknown bucket is last id; map to "__UNK__"
                unk_id = max(card - 1, 0)
                ids = X[:, j]
                raw = [rev.get(feat, {}).get(int(i), "__UNK__") if int(i) != unk_id else "__UNK__" for i in ids]
                cols[feat] = raw
            # weight if available
            w = getattr(ds, "_W", None)
            if w is not None and len(w) == len(X):
                cols["weight"] = w.tolist()
            return pd.DataFrame(cols)
        except Exception:
            return None

    def _robust_split(ds, *, stratify_keys, train_frac: float, seed: int):
        n = len(ds)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

        # If no stratify keys, do simple random split
        if not stratify_keys:
            cut = max(1, int(round(n * train_frac)))
            train_idx = idx[:cut].tolist()
            val_idx = idx[cut:].tolist()
            if len(val_idx) == 0:  # guarantee at least 1 for val
                val_idx = train_idx[-1:]
                train_idx = train_idx[:-1]
            return train_idx, val_idx, "random"

        # Try stratification
        df = _as_dataframe_for_stratify(ds)
        if df is None:
            warn("Stratification requested but no dataframe available; falling back to random split.")
            return _robust_split(ds, stratify_keys=None, train_frac=train_frac, seed=seed)

        # sanity: all keys present
        miss = [k for k in stratify_keys if k not in df.columns]
        if miss:
            warn(f"Stratify keys {miss} not found in dataset; falling back to random split.")
            return _robust_split(ds, stratify_keys=None, train_frac=train_frac, seed=seed)

        # build strata labels
        key = tuple(str(k) for k in stratify_keys)
        strata = df[list(key)].astype(str).agg("||".join, axis=1).values

        # Check bucket sizes
        _, counts = np.unique(strata, return_counts=True)
        if (counts < 2).any() or len(np.unique(strata)) > n * 0.8:
            # Too many tiny buckets → unreliable. Fall back.
            warn("Stratification would create tiny buckets; falling back to random split.")
            return _robust_split(ds, stratify_keys=None, train_frac=train_frac, seed=seed)

        # Per-stratum split
        train_idx, val_idx = [], []
        for s in np.unique(strata):
            s_idx = np.where(strata == s)[0]
            rng.shuffle(s_idx)
            cut = max(1, int(round(len(s_idx) * train_frac)))
            # also ensure we don't produce an empty val for this stratum
            cut = min(cut, len(s_idx) - 1) if len(s_idx) > 1 else 1
            train_idx.extend(s_idx[:cut].tolist())
            val_idx.extend(s_idx[cut:].tolist())

        if len(val_idx) == 0:  # global guard
            val_idx = train_idx[-1:]
            train_idx = train_idx[:-1]

        return train_idx, val_idx, "stratified"

    # -------- Split --------
    stratify_keys = _get(cfg, "dataset.stratify_keys", None)
    # Accept both 'stratify_by' and 'stratify_keys'
    if not stratify_keys:
        stratify_keys = _get(cfg, "dataset.stratify_by", None)
    if isinstance(stratify_keys, (tuple, list)) and len(stratify_keys) == 0:
        stratify_keys = None  # empty list → treat as no stratification

    train_frac = float(_get(cfg, "train.train_frac", 0.8))
    train_idx, val_idx, split_mode = _robust_split(ds, stratify_keys=stratify_keys, train_frac=train_frac, seed=seed)
    print(f"[split] mode={split_mode}  train={len(train_idx)}  val={len(val_idx)}  total={len(ds)}")

    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)

    # -------- DataLoaders --------
    batch_size = int(_get(cfg, "train.batch_size", 1024))
    # never let batch_size exceed the training set
    if len(train_ds) > 0:
        batch_size = max(1, min(batch_size, len(train_ds)))
    else:
        raise RuntimeError("Training set is empty after split; cannot proceed.")

    num_workers = int(_get(cfg, "train.num_workers", 0))
    pin_memory = bool(_get(cfg, "train.pin_memory", True))

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=rangenet_preflop_collate_fn,
    )
    val_dl = DataLoader(
        val_ds, batch_size=min(batch_size, max(1, len(val_ds))), shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=rangenet_preflop_collate_fn,
    )

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
    patience = int(_get(cfg, "train.patience", 3))

    logger_name = _get(cfg, "logging.logger", "tensorboard")
    logger = None
    if logger_name == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(
            save_dir=_get(cfg, "logging.tb_log_dir", "logs/tb"),
            name="rangenet_preflop"
        )

    monitor = "val_kl"
    mode = "min"

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="rangenet-preflop-{epoch:02d}-{val_kl:.4f}",  # will now match sweep
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
    ap.add_argument("--config", type=str, default="rangenet/preflop",
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