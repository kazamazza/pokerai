import sys
from pathlib import Path
import torch

from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.datasets.equity_net_dataset import EquityNetDataset
from ml.models.equity_net import EquityNet
import json
import yaml


def validate(model_path="models/equitynet_best.pt",
             data_path="data/equity/equity_dataset.v1.jsonl.gz",
             settings_path="ml/config/settings.yaml",
             curve_path="models/equitynet_training.json",
             batch_size=256):

    # Load dataset
    ds = EquityNetDataset(data_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # --- Determine board_vocab (do NOT infer from samples) ---
    board_vocab = None

    # 1) Try training curve (if present)
    cp = Path(curve_path)
    if cp.exists():
        try:
            hist = json.loads(cp.read_text())
            # summary was printed, but we also persisted the history dict; fall back to settings if not found
            # some setups persist summary separately; skip if missing
        except Exception:
            pass

    # 2) Read K from settings -> board_clustering.flop.k
    if board_vocab is None:
        cfg = yaml.safe_load(Path(settings_path).read_text())
        board_vocab = int(cfg["board_clustering"]["flop"]["k"])

    # 3) Bucket vocab can still be inferred (or set to 1 for opp embeddings)
    bucket_vocab = 1

    # Init model with the SAME dims as training
    model = EquityNet(
        hand_vocab=169,
        board_vocab=board_vocab,   # <= force 256 (or whatever K you trained)
        bucket_vocab=bucket_vocab,
        emb_dim=32,
        hidden=128
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Evaluate quick MAE
    loss_fn = torch.nn.L1Loss()
    val_sum, n_val = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            pred = model(X)
            loss = loss_fn(pred, y)
            bs = y.size(0)
            val_sum += loss.item() * bs
            n_val += bs
            if n_val >= 2000:
                break

    print(f"✅ Validation Gate | avg_L1={val_sum/max(1,n_val):.4f} over {n_val} samples "
          f"(board_vocab={board_vocab})")

if __name__ == "__main__":
    validate()