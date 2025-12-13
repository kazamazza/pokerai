# tools/smoke_train_ev.py
import sys
from pathlib import Path

import torch, pandas as pd, numpy as np
from torch.utils.data import Subset, DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from ml.datasets.evnet import EVParquetDataset, ev_collate_fn
from ml.models.evnet import EVLit

def overfit_smoke(parquet, action_vocab, x_cols, cont_cols, y_cols=None, weight_col="weight"):
    ds = EVParquetDataset(parquet_path=parquet, action_vocab=action_vocab,
                          x_cols=x_cols, cont_cols=cont_cols, y_cols=y_cols, weight_col=weight_col)
    idx = np.random.RandomState(0).choice(len(ds), size=2048, replace=False)
    sds = Subset(ds, idx)
    dl  = DataLoader(sds, batch_size=256, shuffle=True, collate_fn=ev_collate_fn)

    sample = sds[0]
    ev_cfg = {
        "cat_cardinalities": [len(ds.id_maps[c]) for c in ds.x_cols],
        "cont_dim": int(sample["x_cont"].shape[-1]),
        "action_vocab": list(ds.action_vocab),
        "hidden_dims": [256,256], "dropout": 0.10, "max_emb_dim": 48,
    }
    model = EVLit(config=ev_cfg, lr=7e-4, weight_decay=1e-4)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)

    for step, batch in enumerate(dl, 1):
        opt.zero_grad()
        yhat = model(batch["x_cat"], batch["x_cont"])
        loss = model.criterion(yhat, batch["y"], batch.get("w"), batch.get("y_mask"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0:
            print(f"step {step:03d} | loss {loss.item():.4f}")
        if step == 50:
            break

if __name__ == "__main__":
    # Point these at the parquet/sidecar you’re about to train
    overfit_smoke(
      parquet="data/datasets/evnet_postflop_root.parquet",
      action_vocab=["CHECK","BET_25","BET_33","BET_50","BET_66","BET_75","BET_100","DONK_33"],
      x_cols=["hero_pos","ip_pos","oop_pos","ctx","street","board_cluster_id","stakes_id","hand_id"],
      cont_cols=["board_mask_52","pot_bb","stack_bb","size_frac"],
    )