# ml/validators/validate_rangenet.py
import argparse, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.append(str(ROOT))

from ml.datasets.rangenet_parquet_dataset import RangeNetParquetDataset
from ml.models.range_net import RangeNet
from ml.trainers.train_rangenet import load_cfg, set_seed, load_section

def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    return [b.to(device) for b in batch]

def soft_ce(logits, target_dist):
    """
    Cross-entropy for soft labels: E_q[-log p]
    logits: [B, K], target_dist: [B, K] (must sum ~1)
    """
    logp = F.log_softmax(logits, dim=-1)
    return -(target_dist * logp).sum(dim=-1).mean()

def get_logits(out):
    """Accept tensor or dict from model.forward and return a [B,K] logits tensor."""
    if isinstance(out, dict):
        if "action_logits" in out:
            return out["action_logits"]
        # fallback: first tensor-ish value
        for v in out.values():
            if torch.is_tensor(v):
                return v
        raise RuntimeError("Model output dict has no tensor logits.")
    return out  # assume it's already a tensor

def validate(
    yaml_path: str = "ml/config/settings.yaml",
    split: str = "val",            # "val" or "test"
    batch_size: int = 1024,
    max_rows: int | None = None,   # for a quick gate; None = all
):
    cfg_all, seed = load_cfg(yaml_path)
    set_seed(seed)

    tcfg = load_section(cfg_all.get("training", {}), "rangenet")
    dataset_dir = Path(tcfg.get("dataset_dir", "data/datasets/rangenet.v1"))
    model_dir   = Path(tcfg.get("model_dir", "models"))
    model_path  = model_dir / tcfg.get("model_path", "rangenet_dev.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ctx vocab from train split (freeze vocab)
    tr_path = dataset_dir / "train.parquet"
    val_path = dataset_dir / "val.parquet"
    te_path  = dataset_dir / "test.parquet"
    if not tr_path.exists():
        raise FileNotFoundError(f"Missing train split parquet at {tr_path}")

    df_train = pd.read_parquet(tr_path)
    ctx_vocab = sorted(df_train["ctx"].astype(str).unique().tolist())
    pos_vocab = ["UTG","HJ","CO","BTN","SB","BB"]

    # Choose eval split
    split_path = val_path if split == "val" else te_path
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split} split parquet at {split_path}")

    ds = RangeNetParquetDataset(split_path, ctx_vocab=ctx_vocab, pos_vocab=pos_vocab)
    if max_rows is not None and max_rows > 0 and len(ds) > max_rows:
        # cheap subsample by slicing (deterministic with seed)
        indices = list(range(max_rows))
        # simple wrapper
        from torch.utils.data import Subset
        ds = Subset(ds, indices)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- figure out dims from the (possibly Subset-wrapped) dataset ----
    base_ds = ds.dataset if hasattr(ds, "dataset") else ds  # unwrap Subset if needed

    # preferred properties from RangeNetParquetDataset
    input_dim = getattr(base_ds, "input_len", None)
    if input_dim is None:
        # ultra-safe fallback: derive from vocab lengths if needed
        pos_vocab = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
        input_dim = len(pos_vocab) + len(ctx_vocab) + 1 + 169  # pos one-hot + ctx one-hot + stack + 169-hand

    num_actions = getattr(base_ds, "out_dim", None)
    if num_actions is None:
        # last-resort: peek one row
        first = next(iter(dl))[1]["y_dist"].shape[-1]
        num_actions = int(first)

    # ---- Model ----
    model = RangeNet(
        input_dim=input_dim,
        num_actions=num_actions,  # <-- not 'output_dim'
        hidden=int(tcfg.get("hidden", 256)),
        dropout=float(tcfg.get("dropout", 0.10)),
    ).to(device)

    # load weights
    state = torch.load(model_path, map_location=device)

    # remap old -> new
    remapped = {}
    for k, v in state.items():
        k2 = k
        k2 = k2.replace("f.", "net.")  # old trunk -> new trunk
        k2 = k2.replace("head.", "head_actions.")  # old head  -> new head
        remapped[k2] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"[warn] missing keys after remap: {sorted(missing)[:6]}{' ...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys after remap: {sorted(unexpected)[:6]}{' ...' if len(unexpected) > 6 else ''}")
    model.eval()

    # Metrics
    n = 0
    sum_soft_ce = 0.0
    correct = 0

    t0 = time.time()
    with torch.no_grad():
        for X, Y in dl:
            X = to_device(X, device)
            Y = to_device(Y, device)

            # forward (support both dict or tensor return)
            out = model(X)
            logits = get_logits(out)        # <-- always a tensor now

            y_dist = Y["y_dist"]  # [B,K] target distribution

            # CE against soft targets
            ce = soft_ce(logits, y_dist)
            sum_soft_ce += float(ce.item()) * y_dist.size(0)

            # top-1 vs target argmax (mode of target dist)
            pred = logits.argmax(dim=-1)                       # [B]
            gold = y_dist.argmax(dim=-1)                       # [B]
            correct += (pred == gold).sum().item()

            n += y_dist.size(0)

    avg_ce = sum_soft_ce / max(1, n)
    top1   = correct / max(1, n)
    elapsed = time.time() - t0

    print(f"✅ RangeNet Validation | split={split} | avg_soft_CE={avg_ce:.4f} "
          f"| top1_acc_vs_target={top1:.4f} | N={n:,} | ctx={len(ctx_vocab)} "
          f"| time={elapsed:.2f}s")
    # optional structured line for logs
    print(json.dumps({
        "split": split, "N": n, "avg_soft_ce": round(avg_ce, 6), "top1": round(top1, 6),
        "model_path": str(model_path), "dataset_dir": str(dataset_dir),
        "device": str(device),
    }, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", default="ml/config/settings.yaml")
    p.add_argument("--split", choices=["val","test"], default="val")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--N", type=int, default=0, help="quick gate: cap number of rows (0=all)")
    args = p.parse_args()
    validate(args.yaml, args.split, args.batch_size, None if args.N == 0 else args.N)