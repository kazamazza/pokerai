# tools/rangenet/smoke/preflop_infer_smoke.py
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.etl.rangenet.utils import ALL_HANDS
from ml.inference.preflop import RangeNetPreflopInfer


def topk_pretty(p: torch.Tensor, k: int = 10) -> List[str]:
    # p: [169]
    vals, idx = torch.topk(p, k)
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        name = ALL_HANDS[i] if ALL_HANDS and i < len(ALL_HANDS) else f"id_{i}"
        out.append(f"{name}:{v:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser("Smoke test for RangeNet Preflop inference")
    ap.add_argument("--ckpt", required=True, help="Path to .ckpt")
    ap.add_argument("--sidecar-dir", default=None, help="Dir containing feature_order.json/id_maps.json/cards.json (defaults to ckpt parent)")

    # Minimal inputs (must match your feature_order exactly)
    ap.add_argument("--stack", type=float, default=100)
    ap.add_argument("--hero-pos", type=str, default="BB")
    ap.add_argument("--opener-pos", type=str, default="BTN")
    ap.add_argument("--opener-action", type=str, default="RAISE")
    ap.add_argument("--ctx", type=str, default="SRP")
    args = ap.parse_args()

    infer = RangeNetPreflopInfer.from_checkpoint_dir(
        ckpt_path=args.ckpt,
        ckpt_dir=args.sidecar_dir,
        device="auto",
    )

    row: Dict[str, Any] = {
        # Make sure these names match the saved feature_order in your sidecar!
        "stack_bb": float(args.stack),
        "hero_pos": args.hero_pos.upper(),
        "opener_pos": args.opener_pos.upper(),
        "opener_action": args.opener_action.upper(),
        "ctx": args.ctx.upper(),
    }

    probs = infer.predict_proba([row])  # [1,169]
    p = probs[0]
    print(f"row={row}")
    print(f"sum={p.sum().item():.6f}")
    print("top10:", ", ".join(topk_pretty(p, 10)))

    # sanity: show a couple of strategically-relevant buckets if you like
    if ALL_HANDS:
        def idx(hand: str) -> int:
            return ALL_HANDS.index(hand) if hand in ALL_HANDS else -1
        for key in ["AA", "AKs", "AQs", "A5s", "KQs", "QJs", "JTs", "98s"]:
            i = idx(key)
            if i >= 0:
                print(f"{key}: {p[i].item():.4f}")


if __name__ == "__main__":
    main()