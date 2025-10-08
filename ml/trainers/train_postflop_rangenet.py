from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, List
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.loggers import TensorBoardLogger

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.datasets.utils_dataset import stratified_indices
from ml.datasets.postflop_rangenet import postflop_policy_collate_fn, PostflopPolicyDatasetParquet
from ml.utils.config import load_model_config
from ml.utils.rangenet_postflop_sidecar import write_postflop_policy_sidecar
from ml.models.postflop_policy_net import PostflopPolicyLit
from ml.trainers.helpers import _trial_dir, _expand_grid, _parse_val_kl_from_ckpt

# ----------------- small helpers -----------------
def _get(cfg: Mapping[str, Any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur

def run_train_postflop_sweep(cfg: dict) -> dict:
    import copy, json, shutil
    import pandas as pd

    sweep = cfg.get("sweep", None)
    if not sweep or not isinstance(sweep, dict):
        raise ValueError("No 'sweep' block in config.")

    trials = _expand_grid(sweep)
    base_ckpt_dir = Path(_get(cfg, "train.checkpoints_dir", "checkpoints/postflop_policy"))
    base_ckpt_dir.mkdir(parents=True, exist_ok=True)

    monitor = _get(cfg, "train.monitor", "val_loss")
    results = []
    best = {"score": float("inf"), "ckpt": None, "params": None}

    for i, params in enumerate(trials, 1):
        trial_cfg = copy.deepcopy(cfg)
        # apply trial params (supports dotted keys like "model.lr")
        for k, v in params.items():
            cur = trial_cfg
            parts = k.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = v

        tdir = _trial_dir(base_ckpt_dir, params)
        trial_cfg.setdefault("train", {})["checkpoints_dir"] = str(tdir)
        Path(tdir).mkdir(parents=True, exist_ok=True)

        print(f"\n🧪 Trial {i}/{len(trials)} → {json.dumps(params)}")
        ckpt_path = run_train_postflop(trial_cfg)
        score = _parse_val_kl_from_ckpt(ckpt_path) or float("inf")

        results.append({"trial": i, "score": score, "ckpt": ckpt_path, **params})
        pd.DataFrame(results).to_csv(base_ckpt_dir / "sweep_results.csv", index=False)

        if score < best["score"]:
            best = {"score": score, "ckpt": ckpt_path, "params": params}

        print(f"   ⮑ {monitor}={score:.6f}  best_so_far={best['score']:.6f}")

    print("\n🏁 Sweep complete.")
    print(f"Best {monitor}: {best['score']:.6f}")
    print(f"Checkpoint: {best['ckpt']}")
    print(f"Params: {json.dumps(best['params'])}")

    # copy best artifacts to root
    if best["ckpt"]:
        best_ckpt = Path(best["ckpt"])
        trial_dir = best_ckpt.parent
        # ckpt
        shutil.copy2(best_ckpt, base_ckpt_dir / "best.ckpt")
        # sidecar
        for name in ("postflop_policy_sidecar.json", "sidecar.json"):
            src = trial_dir / name
            if src.exists():
                shutil.copy2(src, base_ckpt_dir / "best_sidecar.json")
                break
        # config
        cfg_src = trial_dir / "config.json"
        if cfg_src.exists():
            shutil.copy2(cfg_src, base_ckpt_dir / "best_config.json")

    return {"best": best, "results": results}

def run_train_postflop(cfg: Mapping[str, Any]) -> str:
    # -------- Repro --------
    seed = int(_get(cfg, "train.seed", 42))
    pl.seed_everything(seed)

    # -------- Dataset --------
    parquet_path = _get(cfg, "inputs.parquet") or _get(cfg, "dataset.parquet")
    if not parquet_path:
        raise ValueError("Missing inputs.parquet (or dataset.parquet) in config")

    use_cluster = bool(_get(cfg, "model.use_board_cluster", True))
    if use_cluster:
        try:
            # Only append if not already present
            if "board_cluster_id" not in PostflopPolicyDatasetParquet.CAT_FEATURES:
                PostflopPolicyDatasetParquet.CAT_FEATURES = (
                        list(PostflopPolicyDatasetParquet.CAT_FEATURES) + ["board_cluster"]
                )
        except Exception:
            pass  # harmless if code layout differs

    ds = PostflopPolicyDatasetParquet(
        parquet_path=parquet_path,
        weight_col=_get(cfg, "dataset.weight_col", "weight"),
        device=torch.device("cpu"),
        strict_canon=bool(_get(cfg, "dataset.strict_canon", True)),
    )

    # categorical vocab sizes (used for embeddings)
    id_maps = ds.id_maps()  # may be partial; fine for sidecar
    cards = ds.cards()  # definite {feat: vocab_size}
    feature_order = list(ds.cat_features)

    # -------- Split --------
    stratify_keys = _get(cfg, "dataset.stratify_keys", ["street", "ip_pos", "oop_pos"])
    train_frac = float(_get(cfg, "train.train_frac", 0.8))
    if stratify_keys:
        train_idx, val_idx = stratified_indices(ds.df, stratify_keys, train_frac, seed)
    else:
        n = len(ds); cut = int(n * train_frac)
        idx = list(range(n))
        train_idx, val_idx = idx[:cut], idx[cut:]

    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)

    batch = int(_get(cfg, "train.batch_size", 1024))
    batch_train = max(1, min(batch, len(train_ds)))
    batch_val = max(1, min(batch, len(val_ds))) if len(val_ds) > 0 else 1

    # -------- DataLoaders --------
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_train,
        shuffle=True,
        num_workers=int(_get(cfg, "train.num_workers", 0)),
        pin_memory=bool(_get(cfg, "train.pin_memory", True)),
        collate_fn=postflop_policy_collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_val,
        shuffle=False,
        num_workers=int(_get(cfg, "train.num_workers", 0)),
        pin_memory=bool(_get(cfg, "train.pin_memory", True)),
        collate_fn=postflop_policy_collate_fn,
    )

    # -------- Model --------
    model = PostflopPolicyLit(
        card_sizes=cards,
        cat_feature_order=feature_order,
        lr=float(_get(cfg, "model.lr", 1e-3)),
        weight_decay=float(_get(cfg, "model.weight_decay", 1e-4)),
        label_smoothing=float(_get(cfg, "model.label_smoothing", 0.0)),
        board_hidden=int(_get(cfg, "model.board_hidden", 64)),
        mlp_hidden=_get(cfg, "model.hidden_dims", [128, 128]),
        dropout=float(_get(cfg, "model.dropout", 0.10)),
    )

    # -------- Callbacks / Logger --------
    ckpt_dir = Path(_get(cfg, "train.checkpoints_dir", "checkpoints/postflop_policy"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor  = _get(cfg, "train.monitor", "val_loss")
    mode     = _get(cfg, "train.mode", "min")
    patience = int(_get(cfg, "train.patience", 3))

    logger_name = _get(cfg, "logging.logger", "tensorboard")
    logger = TensorBoardLogger(
        save_dir=_get(cfg, "logging.tb_log_dir", "logs/tb"),
        name="postflop_policy"
    ) if logger_name == "tensorboard" else False

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="postflop_policy-{epoch:02d}-{" + monitor + ":.4f}",
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
        gradient_clip_val=float(_get(cfg, "train.grad_clip", 1.0)),
    )

    # Save config snapshot & sidecar
    try:
        from omegaconf import OmegaConf
        cfg_ser = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        cfg_ser = dict(cfg)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg_ser, indent=2))
    write_postflop_policy_sidecar(
        ckpt_dir=ckpt_dir,
        feature_order=feature_order,  # list(ds.cat_features)
        cards=cards,  # ds.cards()
        id_maps=id_maps,  # ds.id_maps() (may be partial; still useful)
    )

    # Train
    resume_from = _get(cfg, "train.resume_from", None)
    if resume_from:
        trainer.fit(model, train_dl, val_dl, ckpt_path=str(resume_from))
    else:
        trainer.fit(model, train_dl, val_dl)

    print(f"✅ postflop policy training complete. Best checkpoint: {ckpt_cb.best_model_path}")
    return ckpt_cb.best_model_path or str(ckpt_dir / "last.ckpt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", type=str, default="rangenet/postflop",
        help="Model name or YAML path (resolved by load_model_config)"
    )
    # quick overrides for single-run training
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--max_epochs", type=int)
    ap.add_argument("--patience", type=int)

    # sweep controls
    ap.add_argument("--sweep", action="store_true",
                    help="Run a hyperparameter sweep if a sweep block exists (or force empty to error).")
    ap.add_argument("--max_trials", type=int,
                    help="Optional cap on number of trials to run from the sweep grid.")
    args = ap.parse_args()

    cfg = load_model_config(args.config)

    # ---- apply single-run overrides (always safe; for sweep they become defaults) ----
    train_cfg = cfg.setdefault("train", {})
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.max_epochs is not None:
        train_cfg["max_epochs"] = int(args.max_epochs)
    if args.patience is not None:
        train_cfg["patience"] = int(args.patience)

    # ---- decide: sweep or single run ----
    has_sweep_block = isinstance(cfg.get("sweep"), dict) and len(cfg["sweep"]) > 0

    # allow CLI to cap trials even if YAML has sweep
    if args.max_trials is not None:
        cfg.setdefault("sweep", {})
        cfg["sweep"]["max_trials"] = int(args.max_trials)

    if args.sweep or has_sweep_block:
        # You should have run_train_postflop_sweep(cfg) implemented
        summary = run_train_postflop_sweep(cfg)
        # optional: print best
        best = summary.get("best", {})
        print("\n🏁 Sweep best:")
        print("  val_metric:", best.get("val", best.get("val_loss", best.get("val_kl"))))
        print("  ckpt:", best.get("ckpt"))
    else:
        run_train_postflop(cfg)