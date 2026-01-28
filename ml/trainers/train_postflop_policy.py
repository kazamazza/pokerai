from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, cast

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from ml.datasets.postflop_policy_dataset import collate_postflop_policy, PostflopPolicyIterableDataset, \
    PostflopDatasetSpec
from ml.models.postflop_policy_model import PostflopPolicyModel
from ml.utils.config import load_model_config
from ml.utils.sidecar import ModelSidecarBuilder
from ml.models.vocab_actions import ROOT_ACTION_VOCAB, FACING_ACTION_VOCAB
from ml.models.postflop_policy_lit import PostflopPolicyLit


Target = Literal["root", "facing"]


def _get(cfg: Dict[str, Any], path: str, default=None):
    cur: Any = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _stratified_split_indices(df, keys: Sequence[str], train_frac: float, seed: int):
    """
    Minimal stratified split without bringing in a helper zoo.
    Deterministic and good enough.
    """
    rng = np.random.default_rng(seed)
    if not keys:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        cut = int(len(idx) * train_frac)
        return idx[:cut].tolist(), idx[cut:].tolist()

    # groupby keys -> split within each group
    g = df.groupby(list(keys), dropna=False).indices
    train_idx, val_idx = [], []
    for _, rows in g.items():
        rows = np.array(list(rows), dtype=int)
        rng.shuffle(rows)
        cut = int(len(rows) * train_frac)
        train_idx.extend(rows[:cut].tolist())
        val_idx.extend(rows[cut:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def train_postflop(cfg: Dict[str, Any], *, target: Target) -> str:
    pl.seed_everything(int(_get(cfg, "train.seed", 42)), workers=True)

    # ----- dataset block name -----
    dataset_key = "dataset_postflop_root" if target == "root" else "dataset_postflop_facing"
    ds_cfg: Dict[str, Any] = cast(Dict[str, Any], (cfg.get(dataset_key) or {}))
    if not ds_cfg:
        raise ValueError(f"Missing dataset config block: {dataset_key}")

    # parts dir (NOT a single parquet file) — the builder writes *parts*
    parts_dir = _get(cfg, f"inputs.{target}_parts_dir", None) or _get(cfg, "inputs.parts_dir", None)
    if not parts_dir:
        raise ValueError(f"Missing inputs.{target}_parts_dir (or inputs.parts_dir)")

    # action vocab / labels
    action_vocab = ROOT_ACTION_VOCAB if target == "root" else FACING_ACTION_VOCAB
    y_cols = list(ds_cfg.get("y_cols") or action_vocab)

    # feature columns
    x_cols = list(ds_cfg.get("x_cols") or [])
    cont_cols = list(ds_cfg.get("cont_cols") or [])

    # dataset controls
    weight_col = str(ds_cfg.get("weight_col") or "weight")
    valid_col = str(ds_cfg.get("valid_col") or "valid")
    drop_invalid = bool(ds_cfg.get("drop_invalid", True))

    # split controls (typed)
    split_fracs_raw = ds_cfg.get("split_fracs", (0.96, 0.02, 0.02))
    try:
        a, b, c = split_fracs_raw  # type: ignore[misc]
        split_fracs: tuple[float, float, float] = (float(a), float(b), float(c))
    except Exception:
        split_fracs = (0.96, 0.02, 0.02)

    shuffle_files = bool(ds_cfg.get("shuffle_files", True))

    # vocab artifact
    cat_vocabs_json = str(_get(cfg, "artifacts.postflop_cat_vocabs_json", ""))  # required
    if not cat_vocabs_json:
        raise ValueError("Missing artifacts.postflop_cat_vocabs_json")

    spec = PostflopDatasetSpec(
        kind=target,  # "root" | "facing"
        parts_dir=str(parts_dir),
        cat_vocabs_json=cat_vocabs_json,
        x_cols=x_cols,
        cont_cols=cont_cols,
        y_cols=y_cols,
        weight_col=weight_col,
        valid_col=valid_col,
        seed=int(ds_cfg.get("seed", 42)),
        shard_index=ds_cfg.get("shard_index", None),
        shard_count=ds_cfg.get("shard_count", None),
    )

    ds = PostflopPolicyIterableDataset(
        spec,
        split="train",
        split_fracs=split_fracs,
        shuffle_files=shuffle_files,
        drop_invalid=drop_invalid,
    )

    # ----- split -----
    train_frac = float(_get(cfg, "train.train_frac", 0.80))
    stratify_keys = list(ds_cfg.get("stratify_keys") or [])
    train_idx, val_idx = _stratified_split_indices(ds.df, stratify_keys, train_frac, int(_get(cfg, "train.seed", 42)))

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    # ----- loaders -----
    batch_size = int(_get(cfg, "train.batch_size", 1024))
    num_workers = int(_get(cfg, "train.num_workers", 0))
    pin_memory = bool(_get(cfg, "train.pin_memory", True))

    train_dl = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)) if len(train_ds) else 1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_postflop_policy,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=min(batch_size, len(val_ds)) if len(val_ds) else 1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_postflop_policy,
    )

    # ----- model net -----
    input_dim = ds.input_dim  # easiest if dataset exposes this
    # otherwise compute: len(x_num) + sum(cont vector sizes) (+ optionally cat features if you append)
    net = PostflopPolicyModel(
        input_dim=input_dim,
        output_dim=len(y_cols),
        hidden_dim=int(_get(cfg, "model.hidden_dim", 256)),
        num_layers=int(_get(cfg, "model.num_layers", 3)),
        dropout=float(_get(cfg, "model.dropout", 0.10)),
    )

    lit = PostflopPolicyLit(
        net=net,
        lr=float(_get(cfg, "model.lr", 1e-3)),
        weight_decay=float(_get(cfg, "model.weight_decay", 1e-4)),
        label_smoothing=float(_get(cfg, "model.label_smoothing", 0.0)),
        action_vocab=y_cols,
    )

    # ----- checkpoints dir -----
    ckpt_dir = Path(_get(cfg, f"train.checkpoints_dir_{target}", f"checkpoints/postflop_policy_{target}"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    monitor = str(_get(cfg, "train.monitor", "val_loss"))
    mode = str(_get(cfg, "train.mode", "min"))
    patience = int(_get(cfg, "train.patience", 3))

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"postflop_{target}-" + "{epoch:02d}-{" + monitor + ":.4f}",
        monitor=monitor,
        mode=mode,
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    early_cb = pl.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience)

    trainer = pl.Trainer(
        max_epochs=int(_get(cfg, "train.max_epochs", 10)),
        accelerator=_get(cfg, "train.accelerator", "auto"),
        devices=_get(cfg, "train.devices", "auto"),
        precision=_get(cfg, "train.precision", "16-mixed"),
        deterministic=True,
        callbacks=[ckpt_cb, early_cb],
        gradient_clip_val=float(_get(cfg, "train.grad_clip", 1.0)),
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    # ----- write config snapshot -----
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # ----- write sidecar (ONE unified builder) -----
    ModelSidecarBuilder(
        model_name=f"PostflopPolicy_{target}",
        feature_order=ds.cat_cols,
        cont_features=ds.cont_cols,
        id_maps=ds.id_maps,
        action_vocab=y_cols,
        cards=getattr(ds, "cards", None),
        extras={"target": target},
    ).write(ckpt_dir)

    # ----- train -----
    resume_from = _get(cfg, "train.resume_from", None)
    trainer.fit(lit, train_dl, val_dl, ckpt_path=str(resume_from) if resume_from else None)

    best = ckpt_cb.best_model_path or str(ckpt_dir / "last.ckpt")
    print(f"✅ postflop {target} training complete. best={best}")
    return best


def main():
    import argparse

    ap = argparse.ArgumentParser("Train Postflop Policy (root/facing)")
    ap.add_argument("--config", type=str, default="ml/config/rangenet/postflop/base.yaml")
    ap.add_argument("--target", type=str, choices=["root", "facing"], required=True)
    ap.add_argument("--parquet", type=str, default=None, help="Override inputs.<target>_parquet")
    args = ap.parse_args()

    cfg = load_model_config(args.config)

    if args.parquet:
        cfg.setdefault("inputs", {})
        cfg["inputs"][f"{args.target}_parquet"] = args.parquet

    train_postflop(cfg, target=args.target)


if __name__ == "__main__":
    main()