#!/usr/bin/env python3
import argparse, json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.datasets.postflop_rangenet import PostflopPolicyDatasetParquet, postflop_policy_collate_fn
from ml.models.postflop_policy_net import PostflopPolicyLit
# Import your own dataset/model/collate and sidecar loader
from ml.utils.sidecar import load_sidecar  # expects feature_order/cards/id_maps/etc.

@torch.no_grad()
def masked_log_softmax(logits, mask):
    big_neg = torch.finfo(logits.dtype).min / 4
    masked = torch.where(mask > 0.5, logits, big_neg)
    return torch.log_softmax(masked, dim=-1)

@torch.no_grad()
def masked_ce(logits, target, mask):
    # normalize targets within mask
    t = (target * mask)
    t = t / (t.sum(dim=-1, keepdim=True) + 1e-8)
    logp = masked_log_softmax(logits, mask)
    ce = -(t * logp).sum(dim=-1)  # [B]
    return ce

def main():
    ap = argparse.ArgumentParser("Quick eval masked CE on a checkpoint")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--max_rows", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=1024)
    args = ap.parse_args()

    # Load dataset (CPU is fine)
    ds = PostflopPolicyDatasetParquet(args.parquet, device=torch.device("cpu"))
    if len(ds) > args.max_rows:
        # deterministic sample
        df = ds.df.sample(args.max_rows, random_state=123).reset_index(drop=True)
        # shallow clone dataset with sliced df
        ds.df = df

    # Dataloader
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=postflop_policy_collate_fn
    )

    # Load model
    sc = load_sidecar(Path(args.sidecar))
    model = PostflopPolicyLit.load_from_checkpoint(
        args.ckpt,
        card_sizes=sc["cards"],
        cat_feature_order=sc["feature_order"],
        lr=1e-3, weight_decay=1e-4, label_smoothing=0.0,
        board_hidden=64, mlp_hidden=(128,128), dropout=0.1,
        strict=False
    )
    model.eval(); model.freeze()

    # Eval
    tot_w = 0.0
    sum_loss = 0.0
    sum_ip = 0.0
    sum_oop = 0.0

    with torch.no_grad():
        for x_cat, x_cont, y_ip, y_oop, m_ip, m_oop, w in dl:
            li, lo = model(x_cat, x_cont)
            l_ip = masked_ce(li, y_ip, m_ip)   # [B]
            l_oop = masked_ce(lo, y_oop, m_oop)
            l = l_ip + l_oop
            bw = w.float()
            sum_loss += (l * bw).sum().item()
            sum_ip   += (l_ip * bw).sum().item()
            sum_oop  += (l_oop * bw).sum().item()
            tot_w    += bw.sum().item()

    print(f"Rows eval'd: {int(tot_w)} (weight-sum)")
    print(f"Val(masked CE): {sum_loss / max(tot_w,1e-8):.6f}")
    print(f"  IP:  {sum_ip  / max(tot_w,1e-8):.6f}")
    print(f"  OOP: {sum_oop / max(tot_w,1e-8):.6f}")

if __name__ == "__main__":
    main()