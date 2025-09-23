#!/usr/bin/env python
import argparse, json, random
from pathlib import Path

import torch

from ml.inference.postflop import PostflopPolicyInfer
from ml.inference.preflop import RangeNetPreflopInfer


# ---------------- Helpers ----------------
def _find_ckpt(ckpt_dir: Path) -> Path:
    best = ckpt_dir / "best.ckpt"
    if best.exists():
        return best
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    cks = sorted(ckpt_dir.glob("*.ckpt"))
    if not cks:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return cks[-1]


def _sidecar_for(ckpt: Path) -> Path:
    sc = ckpt.with_suffix(ckpt.suffix + ".sidecar.json")
    if not sc.exists():
        raise FileNotFoundError(f"Sidecar missing: {sc}")
    return sc


# ---------------- Smoke Tests ----------------
@torch.no_grad()
def smoke_preflop(ckpt_dir: Path, k: int = 3):
    ckpt = _find_ckpt(ckpt_dir)
    sidecar = _sidecar_for(ckpt)
    infer = RangeNetPreflopInfer.from_checkpoint(ckpt, sidecar)

    # tiny fake rows
    rows = [
        {
            "stack_bb": 100,
            "hero_pos": "UTG",
            "opener_pos": "HJ",
            "ctx": "SRP",
            "opener_action": "RAISE",
        }
        for _ in range(k)
    ]

    probs = infer.predict(rows)
    print("=== Preflop Smoke ===")
    for i, p in enumerate(probs, 1):
        print(f"[{i}] sum={sum(p):.3f} first5={p[:5]}")


@torch.no_grad()
def smoke_postflop(ckpt_dir: Path, k: int = 3):
    ckpt = _find_ckpt(ckpt_dir)
    sidecar = _sidecar_for(ckpt)
    infer = PostflopPolicyInfer.from_checkpoint(ckpt, sidecar)

    rows = [
        {
            "stack_bb": 80,
            "pot_bb": 10,
            "hero_pos": "BTN",
            "ip_pos": "BTN",
            "oop_pos": "BB",
            "ctx": "VS_OPEN",
            "street": "FLOP",
            "board_mask_52": [0.0]*52,   # all-zero board for smoke
            "actor": "ip",
        }
        for _ in range(k)
    ]

    probs = infer.predict(rows)
    print("=== Postflop Smoke ===")
    for i, p in enumerate(probs, 1):
        print(f"[{i}] sum={sum(p):.3f} first5={p[:5]}")


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflop-ckpt", type=str, default="checkpoints/rangenet_preflop/dev")
    ap.add_argument("--postflop-ckpt", type=str, default="checkpoints/rangenet_postflop/dev")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    smoke_preflop(Path(args.preflop_ckpt), args.k)
    smoke_postflop(Path(args.postflop_ckpt), args.k)


if __name__ == "__main__":
    main()