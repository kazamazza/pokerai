from __future__ import annotations
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.loggers import TensorBoardLogger

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.datasets.utils_dataset import stratified_indices
from ml.utils.config import load_model_config
from ml.utils.rangenet_postflop_sidecar import write_postflop_policy_sidecar
from ml.trainers.helpers import _get
from ml.trainers.sweep import run_sweep, parse_scalar_from_ckpt, finalize_best_artifacts
from ml.datasets.postflop_policy_root import PostflopPolicyDatasetRoot, postflop_policy_root_collate_fn
from ml.models.postflop_policy_side_net import PostflopPolicySideLit
from ml.models.vocab_actions import ROOT_ACTION_VOCAB


def run_postflop_sweep(cfg: dict):
    # rebase dirs & parquet for ROOT
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("inputs", {})
    cfg["inputs"]["parquet"] = cfg["inputs"].get("root_parquet")  # point sweep to root set
    base_dir = Path(cfg.get("train", {}).get("checkpoints_dir_root", "checkpoints/postflop_policy_root"))

    res = run_sweep(
        base_cfg=cfg,
        sweep=cfg["sweep"],
        run_fn=run_train_postflop_root,     # ✅ root runner
        score_fn=parse_scalar_from_ckpt,
        base_ckpt_dir=base_dir,
        monitor=cfg.get("train", {}).get("monitor", "val_loss"),
        max_trials=cfg.get("sweep", {}).get("max_trials"),
    )
    if res["best"]["ckpt"]:
        finalize_best_artifacts(Path(res["best"]["ckpt"]), base_dir)
    return res

def run_train_postflop_root(cfg: Mapping[str, Any]) -> str:
    # -------- Repro --------
    seed = int(_get(cfg, "train.seed", 42))
    pl.seed_everything(seed)

    # -------- Dataset --------
    parquet_path = (
        _get(cfg, "inputs.root_parquet")  # ✅ root-specific parquet
        or _get(cfg, "inputs.parquet")    # fallback for ad-hoc runs
    )
    if not parquet_path:
        raise ValueError("Missing inputs.root_parquet for ROOT training")

    # Schema from YAML
    cat_features = list(_get(cfg, "dataset.cat_features", []))
    cont_features = list(_get(cfg, "dataset.cont_features", []))
    weight_col = _get(cfg, "dataset.weight_col", "weight")
    strict_canon = bool(_get(cfg, "dataset.strict_canon", True))

    # Optionally add board cluster categorical
    use_cluster = bool(_get(cfg, "model.use_board_cluster", True))
    if use_cluster and "board_cluster_id" not in cat_features:
        cat_features.append("board_cluster_id")

    ds = PostflopPolicyDatasetRoot(
        parquet_path=parquet_path,
        cat_features=cat_features,
        cont_features=cont_features,
        weight_col=weight_col,
        strict_canon=strict_canon,
    )

    df_cols = set(ds._df.columns)

    ROOT_PREFIXES = ("CHECK", "BET_", "DONK_")
    FACING_PREFIXES = ("FOLD", "CALL", "RAISE_", "ALLIN")

    missing = [a for a in ROOT_ACTION_VOCAB if a not in df_cols]
    extra_facingish = [c for c in df_cols if any(c.startswith(p) for p in FACING_PREFIXES)]

    assert not missing, f"Missing root targets: {missing}"
    assert not extra_facingish, f"Unexpected facing tokens in ROOT dataset: {extra_facingish}"

    id_maps = ds.id_maps
    cards   = ds.cards
    feature_order = list(ds.cat_features)

    # -------- Split --------
    stratify_keys = _get(cfg, "dataset.stratify_keys", ["street", "ip_pos", "oop_pos"])
    train_frac    = float(_get(cfg, "train.train_frac", 0.8))
    if stratify_keys:
        train_idx, val_idx = stratified_indices(ds._df, stratify_keys, train_frac, seed)
    else:
        n = len(ds); cut = int(n * train_frac)
        idx = list(range(n))
        train_idx, val_idx = idx[:cut], idx[cut:]

    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)

    batch = int(_get(cfg, "train.batch_size", 1024))
    batch_train = max(1, min(batch, len(train_ds)))
    batch_val   = max(1, min(batch, len(val_ds))) if len(val_ds) > 0 else 1

    # -------- Loaders --------
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_train,
        shuffle=True,
        num_workers=int(_get(cfg, "train.num_workers", 0)),
        pin_memory=bool(_get(cfg, "train.pin_memory", True)),
        collate_fn=postflop_policy_root_collate_fn,     # ✅ root-only
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_val,
        shuffle=False,
        num_workers=int(_get(cfg, "train.num_workers", 0)),
        pin_memory=bool(_get(cfg, "train.pin_memory", True)),
        collate_fn=postflop_policy_root_collate_fn,     # ✅ root-only
    )

    # -------- Model (single-side IP) --------
    model = PostflopPolicySideLit(
        side="ip",  # ✅ root side
        card_sizes=cards,
        cat_feature_order=feature_order,
        lr=float(_get(cfg, "model.lr", 1e-3)),
        weight_decay=float(_get(cfg, "model.weight_decay", 1e-4)),
        label_smoothing=float(_get(cfg, "model.label_smoothing", 0.0)),
        board_hidden=int(_get(cfg, "model.board_hidden", 64)),
        mlp_hidden=_get(cfg, "model.hidden_dims", [128, 128]),
        dropout=float(_get(cfg, "model.dropout", 0.10)),
        action_vocab=ROOT_ACTION_VOCAB,  # ✅ restrict logits to root-legal tokens inside the module
    )

    # -------- Callbacks / Logger --------
    ckpt_dir = Path(_get(cfg, "train.checkpoints_dir_root", "checkpoints/postflop_policy_root"))  # ✅ new dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor  = _get(cfg, "train.monitor", "val_loss")
    mode     = _get(cfg, "train.mode", "min")
    patience = int(_get(cfg, "train.patience", 3))

    logger_name = _get(cfg, "logging.logger", "tensorboard")
    logger = TensorBoardLogger(
        save_dir=_get(cfg, "logging.tb_log_dir", "logs/tb"),
        name="postflop_policy_root"
    ) if logger_name == "tensorboard" else False

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="postflop_policy_root-{epoch:02d}-{" + monitor + ":.4f}",
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

    # ---- write sidecar BEFORE training ----
    write_postflop_policy_sidecar(
        ckpt_dir=ckpt_dir,
        feature_order=feature_order,
        cards=cards,
        id_maps=ds.id_maps,
        cont_features=["board_mask_52", "pot_bb", "stack_bb", "size_frac"],  # matches your dataset/model usage
        action_vocab=ROOT_ACTION_VOCAB,  # or ROOT_ACTION_VOCAB in the root trainer
        extras={"side": "oop"},  # {"side": "ip"} for root
    )

    # Read the thing we actually wrote:
    sc = json.loads((ckpt_dir / "sidecar.json").read_text())
    assert ("board_cluster_id" in sc["feature_order"]) == ("board_cluster_id" in ds.cat_features)
    assert "board_cluster_id" not in sc["cont_features"]

    # --- Train ---
    resume_from = _get(cfg, "train.resume_from", None)
    trainer.fit(model, train_dl, val_dl, ckpt_path=str(resume_from) if resume_from else None)

    # After training, freeze a 'best_sidecar.json' next to best ckpt (if any)
    best_path = ckpt_cb.best_model_path
    if best_path:
        (Path(best_path).parent / "best_sidecar.json").write_text(json.dumps(sc, indent=2))

    print(f"✅ facing training complete. Best checkpoint: {ckpt_cb.best_model_path}")
    return ckpt_cb.best_model_path or str(ckpt_dir / "last.ckpt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", type=str, default="rangenet/postflop",
        help="Model name or YAML path (resolved by load_model_config)"
    )
    ap.add_argument("--parquet", type=str, help="Override input parquet path for training/sweep")
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

    # Inject CLI parquet override (works for both run_train and sweep)
    if args.parquet:
        cfg.setdefault("inputs", {})["parquet"] = args.parquet

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
        summary = run_postflop_sweep(cfg)
        # optional: print best
        best = summary.get("best", {})
        print("\n🏁 Sweep best:")
        print("  val_metric:", best.get("val", best.get("val_loss", best.get("val_kl"))))
        print("  ckpt:", best.get("ckpt"))
    else:
        run_train_postflop_root(cfg)