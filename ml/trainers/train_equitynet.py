import json, time, random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.utils.config import load_section, load_cfg
from ml.datasets.equity_net_dataset import EquityNetDataset
from ml.models.equity_net import EquityNet

# ---------- small helpers ----------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def infer_vocabs(ds: EquityNetDataset):
    """
    Infer dynamic dims from a small scan:
      - board_vocab:  max(board_cluster_id)+1
      - bucket_vocab: max(bucket_id)>=0 ? +1 : 1
      - opp_emb_dim:  len(opp_range_emb) or 0
      - board_feats_dim: len(board_feats) or 0  (for logging only; model pads internally)
    """
    max_board, max_bucket = 0, -1
    opp_dim, bf_dim = 0, 0
    n = min(len(ds), 200)
    for i in range(n):
        X, _ = ds[i]
        max_board = max(max_board, int(X["board_cluster_id"]))
        b = int(X["bucket_id"])
        if b >= 0:
            max_bucket = max(max_bucket, b)
        if opp_dim == 0:
            opp_dim = int(len(X["opp_emb"]))
        if bf_dim == 0:
            bf_dim = int(len(X["board_feats"]))
    board_vocab  = max_board + 1
    bucket_vocab = (max_bucket + 1) if max_bucket >= 0 else 1
    return board_vocab, bucket_vocab, opp_dim, bf_dim

def to_device(batch_dict, device):
    return {k: v.to(device) for k, v in batch_dict.items()}

# ---------- main trainer ----------
def train(yaml_path="ml/config/settings.yaml"):
    cfg_all, _, seed = load_cfg(yaml_path)
    set_seed(seed)

    # --- resolve training profile ---
    import os, yaml
    train_root = cfg_all.get("training", {}) or {}
    profile = os.getenv("ML_PROFILE", train_root.get("default_profile", "dev"))

    # pick profile -> section -> equitynet
    profiles = train_root.get("profiles", {}) or {}
    prof_cfg = profiles.get(profile)
    if not prof_cfg:
        raise ValueError(f"training profile '{profile}' not found. available={list(profiles.keys())}")

    t_eq = (prof_cfg.get("equitynet", {}) or {})
    if not t_eq:
        raise ValueError(f"'equitynet' section missing in training profile '{profile}'")

    # --- paths / io ---
    DATA_PATH = Path(t_eq.get("data_path", "data/equity/equity_dataset.v1.jsonl.gz"))
    MODEL_DIR = Path(t_eq.get("model_dir", "models"))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = MODEL_DIR / t_eq.get("model_path", "equitynet_best.pt")
    CURVE_PATH = MODEL_DIR / t_eq.get("curve_path", "equitynet_training.json")

    # --- hyperparams (all from t_eq) ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = int(t_eq.get("batch_size", 512))
    EPOCHS = int(t_eq.get("epochs", 15))
    LR = float(t_eq.get("lr", 1e-3))
    VAL_SPLIT = float(t_eq.get("val_split", 0.1))
    NUM_WORKERS = int(t_eq.get("num_workers", 0))
    EMB_DIM = int(t_eq.get("emb_dim", 32))
    HIDDEN = int(t_eq.get("hidden", 128))
    PATIENCE = int(t_eq.get("early_stop_patience", 4))  # 0 disables early stop
    LOSS_NAME = str(t_eq.get("loss", "L1")).upper()  # "L1" or "MSE"

    print(f"[TRAIN] profile='{profile}' | data='{DATA_PATH}' | batch={BATCH_SIZE} | epochs={EPOCHS} | lr={LR}")

    # Dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    ds = EquityNetDataset(DATA_PATH)
    val_size = max(1, int(len(ds) * VAL_SPLIT))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- read K directly from your clustering config (authoritative) ---
    cfg_all, tcfg, seed = load_cfg(yaml_path)
    flop_cfg = (cfg_all.get("board_clustering", {}) or {}).get("flop", {}) or {}
    BOARD_VOCAB = int(flop_cfg.get("k", 256))  # e.g., 256

    # you can still infer the others from data
    _, bucket_vocab, opp_dim, bf_dim = infer_vocabs(ds)

    print(f"[vocab] board={BOARD_VOCAB} (from clusters) | "
          f"bucket={bucket_vocab} | opp_emb_dim={opp_dim} | board_feats_dim={bf_dim}")

    model = EquityNet(
        hand_vocab=169,
        board_vocab=BOARD_VOCAB,  # <-- fixed, authoritative
        bucket_vocab=bucket_vocab,
        emb_dim=EMB_DIM,
        hidden=HIDDEN
    ).to(DEVICE)

    loss_fn = nn.L1Loss() if LOSS_NAME == "L1" else nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # Train
    best_val = float("inf")
    best_epoch = -1
    history = {"epoch": [], "train": [], "val": []}
    no_improve = 0
    t0 = time.time()

    for ep in range(1, EPOCHS+1):
        # --- train ---
        model.train()
        train_sum, n_train = 0.0, 0
        for X, y in train_loader:
            X = to_device(X, DEVICE); y = y.to(DEVICE)
            opt.zero_grad()
            pred = model(X)               # [B,1]
            loss = loss_fn(pred, y)       # y is [B,1]
            loss.backward(); opt.step()
            bs = y.size(0)
            train_sum += loss.item() * bs
            n_train += bs
        train_metric = train_sum / max(1, n_train)

        # --- validate ---
        model.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X = to_device(X, DEVICE); y = y.to(DEVICE)
                pred = model(X)
                loss = loss_fn(pred, y)
                bs = y.size(0)
                val_sum += loss.item() * bs
                n_val += bs
        val_metric = val_sum / max(1, n_val)

        history["epoch"].append(ep)
        history["train"].append(train_metric)
        history["val"].append(val_metric)
        print(f"Epoch {ep:02d} | train_{LOSS_NAME}={train_metric:.4f} | val_{LOSS_NAME}={val_metric:.4f}")

        # save best + early stop
        if val_metric < best_val - 1e-4:
            best_val = val_metric
            best_epoch = ep
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✅ saved best → {MODEL_PATH} (val_{LOSS_NAME}={best_val:.4f})")
        else:
            no_improve += 1
            if PATIENCE > 0 and no_improve >= PATIENCE:
                print(f"  ⏹ early stop at epoch {ep} (no improvement for {PATIENCE} epochs)")
                break

    elapsed = time.time() - t0
    # Persist curve for quick charting later
    CURVE_PATH.write_text(json.dumps(history, indent=2))

    # Final summary (paste this back to me for a quick check)
    summary = {
        "data_path": str(DATA_PATH),
        "model_path": str(MODEL_PATH),
        "device": str(DEVICE),
        "epochs_trained": len(history["epoch"]),
        "best_epoch": best_epoch,
        f"best_val_{LOSS_NAME}": round(best_val, 6),
        "train_size": train_size,
        "val_size": val_size,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "emb_dim": EMB_DIM,
        "hidden": HIDDEN,
        "board_vocab": BOARD_VOCAB,
        "bucket_vocab": bucket_vocab,
        "opp_emb_dim": opp_dim,
        "board_feats_dim": bf_dim,
        "elapsed_sec": round(elapsed, 2)
    }
    print("\n=== EquityNet Training Summary ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    train()