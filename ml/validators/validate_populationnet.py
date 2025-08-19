# ml/validators/validate_populationnet.py
from __future__ import annotations
import os, json, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

# local
ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.append(str(ROOT))
from ml.datasets.population_net_dataset import PopulationNetDataset
from ml.models.population_net import PopulationNet

def _load_cfg(yaml_path="ml/config/settings.yaml"):
    import yaml
    cfg_all = yaml.safe_load(Path(yaml_path).read_text())
    t_all   = (cfg_all.get("training") or {})
    t_pop   = (t_all.get("populationnet") or {})
    profile = os.getenv("ML_PROFILE", t_pop.get("default_profile", "dev"))
    prof    = (t_pop.get("profiles") or {}).get(profile, {})
    if not prof:
        raise ValueError(f"populationnet training profile '{profile}' not found in settings.yaml")
    return profile, prof

@torch.no_grad()
def validate(yaml_path="ml/config/settings.yaml", n_eval=4096, batch_size=2048):
    profile, prof = _load_cfg(yaml_path)

    data_path  = Path(prof.get("data_path", "data/population/popnet.v1.jsonl.gz"))
    model_dir  = Path(prof.get("model_dir", "models"))
    model_path = model_dir / prof.get("model_path", f"populationnet_{profile}.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset → small random slice for speed
    ds  = PopulationNetDataset(data_path)
    n   = min(len(ds), n_eval)
    # deterministic-ish sample
    generator = torch.Generator().manual_seed(42)
    idxs = torch.randperm(len(ds), generator=generator)[:n].tolist()
    subset = torch.utils.data.Subset(ds, idxs)
    dl = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = PopulationNet(input_dim=ds.input_dim,
                          output_dim=ds.output_dim,
                          hidden=prof.get("hidden", 256),
                          dropout=prof.get("dropout", 0.15)).to(device)
    # Load weights
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Metrics
    mae_sum, count = 0.0, 0
    correct, total = 0, 0

    for batch in dl:
        # dataset returns {"x_vec": ..., ...}, y_vec
        X = {k: v.to(device) for k, v in batch[0].items()}
        y = batch[1].to(device)       # [B, M]
        pred = model(X)               # [B, M]

        # MAE over full action vector
        mae = torch.nn.functional.l1_loss(pred, y, reduction="sum")
        mae_sum += mae.item()
        count   += y.numel()

        # Argmax accuracy
        correct += (pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
        total   += y.size(0)

    avg_mae = mae_sum / max(1, count)
    acc     = correct / max(1, total)

    summary = {
        "profile": profile,
        "data_path": str(data_path),
        "model_path": str(model_path),
        "N_eval": n,
        "D": ds.input_dim,
        "M": ds.output_dim,
        "avg_L1": round(avg_mae, 6),
        "top1_acc": round(acc, 4)
    }
    print(f"✅ PopNet Validation | avg_L1={summary['avg_L1']:.4f} | top1_acc={summary['top1_acc']:.4f} "
          f"(N={n}, M={ds.output_dim})")
    # also dump a JSON for quick charting if you want
    out = Path(prof.get("curve_path", f"populationnet_{profile}_validation.json"))
    out.write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    validate()