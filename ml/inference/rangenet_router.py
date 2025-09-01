import numpy as np
import torch

from ml.inference.rangenet import RangeNetInfer


class RangeNetRouter:
    """
    Street-aware wrapper that holds two RangeNetInfer instances:
      - preflop (street == 0)
      - postflop (street in {1,2,3})
    """
    def __init__(self, pre_infer: RangeNetInfer, post_infer: RangeNetInfer):
        self.pre = pre_infer
        self.post = post_infer

    @classmethod
    def from_checkpoints(cls,
                         pre_ckpt: str, pre_sidecar: str,
                         post_ckpt: str, post_sidecar: str,
                         device: str | torch.device = "auto"):
        pre = RangeNetInfer.from_checkpoint(pre_ckpt, pre_sidecar, device=device)
        post = RangeNetInfer.from_checkpoint(post_ckpt, post_sidecar, device=device)
        return cls(pre, post)

    def predict_one(self, features: dict) -> np.ndarray:
        st = int(features.get("street", 0))  # 0=preflop, 1/2/3=postflop
        if st == 0:
            # ensure only preflop keys are present (optional: strip extras)
            return self.pre.predict_one(features)
        else:
            return self.post.predict_one(features)

    def predict_batch(self, rows: list[dict], as_numpy: bool = True):
        if not rows:
            return np.zeros((0,169), dtype=np.float32) if as_numpy else torch.zeros((0,169))
        # Split by street, call each model, then reassemble in original order
        idx_pre = [i for i,r in enumerate(rows) if int(r.get("street",0)) == 0]
        idx_post= [i for i,r in enumerate(rows) if int(r.get("street",0)) != 0]
        outs = [None] * len(rows)
        if idx_pre:
            batch = [rows[i] for i in idx_pre]
            y = self.pre.predict_batch(batch, as_numpy=as_numpy)
            for j,i in enumerate(idx_pre): outs[i] = y[j]
        if idx_post:
            batch = [rows[i] for i in idx_post]
            y = self.post.predict_batch(batch, as_numpy=as_numpy)
            for j,i in enumerate(idx_post): outs[i] = y[j]
        if as_numpy:
            return np.stack(outs, axis=0)
        return torch.stack(outs, dim=0)