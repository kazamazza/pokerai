import sys
from pathlib import Path
import json, random, os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
# assumes you'll implement this Dataset next:
# it should emit: X={"x_vec": FloatTensor[B,D]}, y=FloatTensor[B,M]
from ml.datasets.population_net_dataset import PopulationNetDataset
from ml.models.population_net import PopulationNet

def load_cfg(yaml_path="ml/config/settings.yaml"):
    import yaml
    cfg_all = yaml.safe_load(Path(yaml_path).read_text())
    troot = (cfg_all.get("training", {}) or {}).get("populationnet", {}) or {}
    return cfg_all, troot, int(cfg_all.get("seed", 42))

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def train(yaml_path="ml/config/settings.yaml"):
    cfg_all, tcfg, seed = load_cfg(yaml_path)
    set_seed(seed)

    # dev/prod profile support (mirrors your EquityNet setup)
    profile = os.getenv("ML_PROFILE", (tcfg.get("default_profile") or "dev"))
    prof = (tcfg.get("profiles") or {}).get(profile, {})
    if not prof:
        raise ValueError(f"populationnet training profile '{profile}' not found")

    data_path  = Path(prof.get("data_path", "data/population/population.v1.jsonl.gz"))
    model_dir  = Path(prof.get("model_dir", "models")); model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / prof.get("model_path", f"populationnet_{profile}.pt")
    curve_path = model_dir / prof.get("curve_path", f"populationnet_{profile}_training.json")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(prof.get("batch_size", 1024))
    epochs     = int(prof.get("epochs", 10))
    lr         = float(prof.get("lr", 1e-3))
    val_split  = float(prof.get("val_split", 0.1))
    hidden     = int(prof.get("hidden", 256))
    dropout    = float(prof.get("dropout", 0.15))
    num_workers= int(prof.get("num_workers", 0))
    loss_name  = str(prof.get("loss", "L1")).upper()  # "L1" or "MSE"

    if not data_path.exists():
        raise FileNotFoundError(f"PopulationNet dataset not found: {data_path}")

    ds = PopulationNetDataset(data_path)     # must expose .input_dim and .output_dim
    val_size  = max(1, int(len(ds) * val_split))
    train_size= len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    dl_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    dl_va = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[population] D={ds.input_dim} | M={ds.output_dim} | N={len(ds):,} "
          f"(train={train_size:,}, val={val_size:,})")

    model = PopulationNet(input_dim=ds.input_dim, output_dim=ds.output_dim,
                          hidden=hidden, dropout=dropout).to(device)
    loss_fn = nn.L1Loss() if loss_name == "L1" else nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best = float("inf"); history={"epoch":[], "train":[], "val":[]}
    for ep in range(1, epochs+1):
        model.train(); tr_sum=0.0; n_tr=0
        for X, y in dl_tr:
            X = {k:v.to(device) for k,v in X.items()}
            y = y.to(device)
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward(); opt.step()
            bs=y.size(0); tr_sum += loss.item()*bs; n_tr += bs
        tr = tr_sum/max(1,n_tr)

        model.eval(); va_sum=0.0; n_va=0
        with torch.no_grad():
            for X, y in dl_va:
                X = {k:v.to(device) for k,v in X.items()}
                y = y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                bs=y.size(0); va_sum += loss.item()*bs; n_va += bs
        va = va_sum/max(1,n_va)

        history["epoch"].append(ep); history["train"].append(tr); history["val"].append(va)
        print(f"Epoch {ep:02d} | train_{loss_name}={tr:.4f} | val_{loss_name}={va:.4f}")
        if va < best - 1e-4:
            best = va
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ saved best → {model_path} (val_{loss_name}={best:.4f})")

    curve_path.write_text(json.dumps(history, indent=2))
    print("\n=== PopulationNet Summary ===")
    print(json.dumps({
        "profile": profile, "data_path": str(data_path), "model_path": str(model_path),
        "input_dim": ds.input_dim, "output_dim": ds.output_dim,
        "best_val_"+loss_name: round(best,6),
        "epochs": len(history["epoch"]), "batch_size": batch_size, "lr": lr
    }, indent=2))

if __name__ == "__main__":
    train()