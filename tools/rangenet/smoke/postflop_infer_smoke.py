#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.features.boards.board_clusterers.kmeans import KMeansBoardClusterer
from ml.inference.policy.types import PolicyRequest
from ml.inference.postflop import PostflopPolicyInfer
from ml.utils.board_mask import make_board_mask_52
import argparse
from pathlib import Path
import torch


def _resolve_ckpt_and_sidecar(ckpt_dir: Path):
    """Prefer best.ckpt + best_sidecar.json; fall back to last.ckpt + sidecar.json."""
    best_ckpt = ckpt_dir / "best.ckpt"
    last_ckpt = ckpt_dir / "last.ckpt"

    # sidecar names
    best_sc  = ckpt_dir / "best_sidecar.json"
    sidecar  = ckpt_dir / "sidecar.json"

    ckpt_path = best_ckpt if best_ckpt.exists() else last_ckpt
    if not ckpt_path.exists():
        # also allow pattern-matched PL checkpoints if present
        cands = sorted(ckpt_dir.glob("postflop_policy-*-*.ckpt"))
        if cands:
            ckpt_path = cands[-1]
        else:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    sidecar_path = best_sc if best_sc.exists() else sidecar
    if not sidecar_path.exists():
        raise FileNotFoundError(f"No sidecar JSON found in {ckpt_dir} (looked for best_sidecar.json / sidecar.json)")

    return ckpt_path, sidecar_path


def _pretty_topk(names, probs, k=10):
    import numpy as np
    p = torch.tensor(probs).float().numpy()
    idx = p.argsort()[::-1][:k]
    return [(names[i], float(p[i])) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True,
                    help="Folder with best.ckpt/last.ckpt and best_sidecar.json/sidecar.json")
    ap.add_argument("--clusterer_artifact", type=str, required=True,
                    help="Path to kmeans artifact (e.g. data/artifacts/board_clusters_kmeans_128.json)")
    ap.add_argument("--board", type=str, default="AsKh2d", help="Flop board, e.g. AsKh2d")
    ap.add_argument("--ip_pos", type=str, default="BTN")
    ap.add_argument("--oop_pos", type=str, default="BB")
    ap.add_argument("--ctx", type=str, default="VS_OPEN")
    ap.add_argument("--pot_bb", type=float, default=18.0)
    ap.add_argument("--eff_stack_bb", type=float, default=100.0)
    ap.add_argument("--actor", type=str, default="ip", choices=["ip","oop"])
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")

    ckpt_path, sidecar_path = _resolve_ckpt_and_sidecar(ckpt_dir)

    # 1) Load inference wrapper (model + sidecar)
    infer = PostflopPolicyInfer.from_checkpoint(ckpt_path, sidecar_path, device="auto")

    # 2) Load board clusterer if your model was trained with board_cluster_id
    #    (the infer wrapper will ignore it if that feature wasn't in feature_order)
    infer.clusterer = KMeansBoardClusterer.load(args.clusterer_artifact)

    # 3) Build request (preferred API). If your wrapper doesn't have predict(req),
    #    we fall back to predict_one(row) below.
    req = PolicyRequest(
        street=1,
        ip_pos=args.ip_pos,
        oop_pos=args.oop_pos,
        ctx=args.ctx,
        pot_bb=float(args.pot_bb),
        eff_stack_bb=float(args.eff_stack_bb),
        board=args.board,
        # actor is provided to the call, not stored in request
    )

    try:
        # Preferred/modern path: high-level predict using PolicyRequest
        out = infer.predict(req, actor=args.actor, temperature=float(args.temperature))
        actions = out.actions
        probs_ip = out.debug.get("probs_ip") or out.probs  # if your predict returns single-side
        probs_oop = out.debug.get("probs_oop") or out.probs  # graceful fallback
    except AttributeError:
        # Fallback: older wrappers that expect a row dict via predict_one(...)
        # Prepare row dict the model expects. The infer wrapper handles:
        #  - board → 52-mask internally
        #  - (optional) board → cluster_id if feature present
        row = {
            "hero_pos": args.ip_pos,  # include if your cat features include hero_pos
            "ip_pos": args.ip_pos,
            "oop_pos": args.oop_pos,
            "ctx": args.ctx,
            "street": 1,
            "board_mask_52": make_board_mask_52(args.board),
            "pot_bb": float(args.pot_bb),
            "eff_stack_bb": float(args.eff_stack_bb),
        }
        # Add cluster id if the model expects it
        if "board_cluster_id" in infer.feature_order:
            row["board_cluster_id"] = int(infer.clusterer.predict_one(args.board))  # type: ignore[attr-defined]

        res = infer.predict_one(
            row,
            actor=args.actor,
            temperature=float(args.temperature),
        )
        actions = res["actions"]
        probs_ip = res["probs_ip"]
        probs_oop = res["probs_oop"]

    print("✅ PostflopPolicy inference sanity OK")
    print(f"Board: {args.board}  ctx={args.ctx}  ip={args.ip_pos}  oop={args.oop_pos}  actor={args.actor}")
    if "board_cluster_id" in infer.feature_order:
        cid = int(infer.clusterer.predict_one(args.board))  # type: ignore[attr-defined]
        print(f"Cluster ID: {cid}")

    print(f"Top-10 IP:  {_pretty_topk(actions, probs_ip, 10)}")
    print(f"Top-10 OOP: {_pretty_topk(actions, probs_oop, 10)}")


if __name__ == "__main__":
    main()