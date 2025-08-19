# ml/train/train_rangenet.py
from __future__ import annotations
import os, sys, json, math, time, argparse, random
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



ROOT_DIR = Path(__file__).resolve().parents[2]
import sys; sys.path.append(str(ROOT_DIR))
import pandas as pd
from ml.datasets.rangenet_parquet_dataset import RangeNetParquetDataset
from ml.models.range_net import RangeNet

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# -------------------- small utils (kept local for drop-in) --------------------
def load_cfg(yaml_path="ml/config/settings.yaml"):
    import yaml
    cfg_all = yaml.safe_load(Path(yaml_path).read_text())
    seed = int(cfg_all.get("seed", 42))
    return cfg_all, seed

def load_section(cfg: dict, key: str, profile_env: str = "ML_PROFILE", default: str = "dev") -> dict:
    sect = cfg.get(key, {}) or {}
    prof = os.getenv(profile_env, sect.get("default_profile", default))
    profiles = (sect.get("profiles") or {})
    if isinstance(profiles, dict) and prof in profiles:
        return profiles[prof]
    return sect

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    return [b.to(device) for b in batch]

# 169 grid order (AA..22, suited above diag, off below)
_RANKS = "AKQJT98765432"
def _pair(i): return _RANKS[i] + _RANKS[i]
def _suited(i, j): return _RANKS[i] + _RANKS[j] + "s"
def _offsuit(i, j): return _RANKS[i] + _RANKS[j] + "o"
def hand169_order() -> List[str]:
    names = []
    for i in range(13):
        for j in range(13):
            if i == j: names.append(_pair(i))
            elif i < j: names.append(_suited(i, j))
            else: names.append(_offsuit(i, j))
    return names

HAND169 = hand169_order()
HAND2IDX = {h:i for i,h in enumerate(HAND169)}

# -------------------- Train --------------------
def train(yaml_path="ml/config/settings.yaml"):
    cfg_all, seed = load_cfg(yaml_path)
    set_seed(seed)

    tcfg = load_section(cfg_all.get("training", {}), "rangenet")
    # sensible dev defaults
    dataset_dir = Path(tcfg.get("dataset_dir", "data/datasets/rangenet.v1"))
    model_dir   = Path(tcfg.get("model_dir", "models")); model_dir.mkdir(parents=True, exist_ok=True)
    model_path  = model_dir / tcfg.get("model_path", "rangenet_dev.pt")
    curve_path  = model_dir / tcfg.get("curve_path", "rangenet_dev_training.json")

    batch_size  = int(tcfg.get("batch_size", 1024))
    epochs      = int(tcfg.get("epochs", 10))
    lr          = float(tcfg.get("lr", 1e-3))
    hidden      = int(tcfg.get("hidden", 256))
    dropout     = float(tcfg.get("dropout", 0.10))
    num_workers = int(tcfg.get("num_workers", 0))
    early_stop  = int(tcfg.get("early_stop_patience", 4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = (device.type == "cuda")

    # load splits
    tr_path = dataset_dir / "train.parquet"
    va_path = dataset_dir / "val.parquet"
    te_path = dataset_dir / "test.parquet"
    if not tr_path.exists() or not va_path.exists():
        raise FileNotFoundError(f"Expected parquet splits in {dataset_dir} (train/val).")

    # establish ctx vocab from train (and freeze)
    df_train = pd.read_parquet(tr_path)
    ctx_vocab = sorted(df_train["ctx"].astype(str).unique().tolist())
    pos_vocab = ["UTG","HJ","CO","BTN","SB","BB"]

    ds_tr = RangeNetParquetDataset(tr_path, ctx_vocab=ctx_vocab, pos_vocab=pos_vocab)
    ds_va = RangeNetParquetDataset(va_path, ctx_vocab=ctx_vocab, pos_vocab=pos_vocab)

    input_dim = len(pos_vocab) + len(ctx_vocab) + 1 + 169
    out_dim   = ds_tr.out_dim

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_mem, persistent_workers=(num_workers>0))
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_mem, persistent_workers=(num_workers>0))

    model = RangeNet(input_dim=input_dim, out_dim=out_dim, hidden=hidden, dropout=dropout).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    kldiv = nn.KLDivLoss(reduction="batchmean")  # expects log-probs vs probs

    def _to_dev(obj): return to_device(obj, device)

    print(f"[rangenet] D={input_dim} | K={out_dim} | N={len(ds_tr):,}/{len(ds_va):,} (train/val) "
          f"| ctx={ctx_vocab} | device={device}")

    best_val = float("inf"); best_ep = 0; patience = 0
    history = {"epoch": [], "train": [], "val": [], "val_acc": []}

    for ep in range(1, epochs+1):
        # ---- train ----
        model.train()
        tr_loss = 0.0; tr_steps = 0
        pbar = tqdm(total=len(dl_tr), desc=f"ep {ep:02d}/{epochs} [train]", leave=False) if tqdm else None
        t0 = time.time()

        for step, (X, Y) in enumerate(dl_tr, 1):
            X = _to_dev(X); Y = _to_dev(Y)
            out = model(X)
            logp = F.log_softmax(out["logits"], dim=-1)
            loss = kldiv(logp, Y["y_dist"])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_loss += float(loss.item()); tr_steps += 1
            if pbar:
                elapsed = time.time()-t0; ips = step/max(elapsed,1e-6)
                pbar.set_postfix({"loss": f"{tr_loss/tr_steps:.3f}", "it/s": f"{ips:.1f}"})
                pbar.update(1)
        if pbar: pbar.close()

        avg_tr = tr_loss / max(1, tr_steps)

        # ---- validate ----
        model.eval()
        va_loss = 0.0; va_steps = 0; correct = 0; total = 0
        pbar = tqdm(total=len(dl_va), desc=f"ep {ep:02d}/{epochs} [val]  ", leave=False) if tqdm else None
        with torch.no_grad():
            for step, (X, Y) in enumerate(dl_va, 1):
                X = _to_dev(X); Y = _to_dev(Y)
                out = model(X)
                logp = F.log_softmax(out["logits"], dim=-1)
                loss = kldiv(logp, Y["y_dist"])
                va_loss += float(loss.item()); va_steps += 1

                # top-1 vs teacher's mode (argmax of target dist)
                pred = out["logits"].argmax(dim=-1)
                tgt  = Y["y_dist"].argmax(dim=-1)
                correct += (pred == tgt).sum().item()
                total   += pred.numel()

                if pbar: pbar.update(1)
        if pbar: pbar.close()

        avg_va = va_loss / max(1, va_steps)
        acc    = correct / max(1, total)

        history["epoch"].append(ep)
        history["train"].append(avg_tr)
        history["val"].append(avg_va)
        history["val_acc"].append(acc)

        print(f"Epoch {ep:02d} | train={avg_tr:.4f} | val={avg_va:.4f} | top1={acc:.3f}")

        # early stop + ckpt
        if avg_va < best_val - 1e-5:
            best_val = avg_va; best_ep = ep; patience = 0
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ saved best → {model_path} (val={best_val:.4f})")
        else:
            patience += 1
            if patience >= early_stop:
                print(f"  ⏹ early stop at epoch {ep} (no val improvement for {early_stop} epochs)")
                break

    # save curve
    curve_path.write_text(json.dumps(history, indent=2))
    print(f"✅ saved curve → {curve_path}")
    print(json.dumps({
        "dataset_dir": str(dataset_dir),
        "model_path": str(model_path),
        "epochs_trained": len(history["epoch"]),
        "best_epoch": best_ep,
        "best_val_loss": round(best_val, 6),
        "batch_size": batch_size,
        "lr": lr,
        "hidden": hidden,
        "dropout": dropout
    }, indent=2))

if __name__ == "__main__":
    # CLI is optional; YAML drives by default
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", default="ml/config/settings.yaml")
    args = p.parse_args()
    train(args.yaml)