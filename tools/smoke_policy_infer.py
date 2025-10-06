#!/usr/bin/env python3
import argparse, json
import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.policy import PolicyInfer
from ml.inference.policy.types import PolicyRequest
# --- your imports ---
from ml.inference.postflop import PostflopPolicyInfer


# ---- tiny stubs so PolicyInfer will initialize (we won’t use them on flop) ----
class _StubExploit:
    def predict(self, *a, **k):  # never called in this smoke
        return {"exploit": 0.0}

class _StubEquity:
    def hand_equity(self, *a, **k):  # never called here
        return 0.5

class _StubRangePre:
    def predict(self, *a, **k):  # never called here
        return {"range": [0.0] * 169}

def _board_mask_52(board_str: str) -> list[float]:
    """Optional: encode 'Jh8d8c' into a 52-bit mask; safe to pass zeros if you prefer."""
    if not board_str:
        return [0.0] * 52
    ranks = "23456789TJQKA"
    suits = "cdhs"  # columns within rank
    idx = lambda r, s: ranks.index(r.upper()) * 4 + suits.index(s.lower())
    mask = [0.0] * 52
    s = board_str.strip()
    # accept formats like "Jh8d8c" or "J h 8 d 8 c"
    s = s.replace(" ", "")
    # read 2-chunks
    for i in range(0, len(s), 2):
        r, u = s[i], s[i+1]
        j = idx(r, u)
        mask[j] = 1.0
    return mask

def main():
    ap = argparse.ArgumentParser("Smoke test PolicyInfer (postflop)")
    ap.add_argument("--ckpt",    required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--ip",      default="IP")
    ap.add_argument("--oop",     default="OOP")
    ap.add_argument("--ctx",     default="VS_OPEN")    # VS_OPEN | VS_3BET | VS_4BET | LIMPED_SINGLE | LIMPED_MULTI
    ap.add_argument("--street",  type=int, default=1)  # 1 = FLOP
    ap.add_argument("--pot",     type=float, default=28.0)
    ap.add_argument("--stack",   type=float, default=100.0)
    ap.add_argument("--board",   type=str, default="Jh8d8c")  # optional
    ap.add_argument("--hand", type=str, default=None, help="Hero hand like 'AsKd' (optional)")
    args = ap.parse_args()

    # 1) load postflop infer
    post = PostflopPolicyInfer.from_checkpoint(args.ckpt, args.sidecar, device="cpu")

    def _hand_mask_52(hand: str) -> list[float]:
        if not hand or len(hand) != 4:  # e.g., "AsKd"
            return [0.0] * 52
        ranks = "23456789TJQKA"
        suits = "cdhs"

        def idx(r, s): return ranks.index(r.upper()) * 4 + suits.index(s.lower())

        m = [0.0] * 52
        m[idx(hand[0], hand[1])] = 1.0
        m[idx(hand[2], hand[3])] = 1.0
        return m

    hand_mask = _hand_mask_52(args.hand) if args.hand else [0.0] * 52

    # 2) wire PolicyInfer with stubs + postflop
    deps = PolicyInferDeps(
        policy_post=post,
        exploit=_StubExploit(),
        equity=_StubEquity(),
        range_pre=_StubRangePre(),
        clusterer=None,
        params={"postflop.cont_features": ["board_mask_52", "pot_bb", "eff_stack_bb"]},
    )
    policy = PolicyInfer(deps)

    # 3) make a realistic flop request
    req = PolicyRequest(
        street=args.street,     # flop
        hero_pos="BTN",         # optional at this layer
        villain_pos=None,
        ctx=args.ctx,           # matches sidecar ctx ids (VS_OPEN / VS_3BET / VS_4BET / LIMPED_*)
        pot_bb=args.pot,
        eff_stack_bb=args.stack,
        board=args.board,
        raw={
            "ip_pos": args.ip,
            "oop_pos": args.oop,
            "board_mask_52": _board_mask_52(args.board),
            "hero_hand": args.hand,  # human readable
            "hero_mask_52": hand_mask,  # machine form (optional for later)
        },
    )

    # 4) ask policy (postflop path)
    out = policy.predict(dict(vars(req)))
    actions = out.get("actions", [])
    probs = out.get("probs", [])

    print("✅ Policy response")
    print(json.dumps(out, indent=2))

    def _topk_unique(pairs, k=5):
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        seen, uniq = set(), []
        for a, p in pairs:
            if a in seen:
                continue
            uniq.append((a, p))
            seen.add(a)
            if len(uniq) >= k:
                break
        return uniq

    # Prefer dense dict if available (full vocab) to avoid any pre-filter/sequencing artifacts
    if isinstance(out.get("dense"), dict) and out["dense"]:
        dense_pairs = list(out["dense"].items())
        top = _topk_unique(dense_pairs, k=5)
        pretty = ", ".join(f"{a}={p:.3f}" for a, p in top)
        print("Top-5 (dense):", pretty)

    # Fallback to (actions, probs) vector
    elif isinstance(actions, list) and isinstance(probs, list) and len(actions) == len(probs) and probs:
        top = _topk_unique(list(zip(actions, probs)), k=5)
        pretty = ", ".join(f"{a}={p:.3f}" for a, p in top)
        print("Top-5:", pretty)
    else:
        print("Top-5: (unavailable — missing or mismatched actions/probs)")

if __name__ == "__main__":
    main()