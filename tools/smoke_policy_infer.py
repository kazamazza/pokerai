# scripts/policy_infer_smoke.py
#!/usr/bin/env python3
"""
Simple smoke test for your real PolicyInfer.
- No CLI args, no dynamic import tricks, no mocks.
- Edit the TODOs (import paths & checkpoint paths) to match your repo.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# ============================
# TODO: NORMAL IMPORTS — adjust module paths if file names differ
# ============================

# Orchestrator & types
from ml.inference.policy.policy import PolicyInfer, PolicyInferDeps, PolicyBlendConfig  # TODO: confirm module path
from ml.inference.postflop_infer_single import PostflopPolicyInferSingle

# Dependencies
from ml.inference.postflop_router import PostflopPolicyRouter                     # TODO: confirm module path (file likely postflop_policy_router.py)
from ml.inference.preflop import PreflopPolicy                                    # TODO
from ml.inference.equity import EquityNetInfer                                    # TODO
from ml.inference.population import PopulationNetInference                         # TODO
from ml.inference.player_exploit_store import PlayerExploitStore                   # TODO
from ml.models.equity_net import EquityNetLit
from ml.models.preflop_rangenet import RangeNetLit

# Clusterer is optional; import if you have a concrete class
# from ml.features.boards.board_clusterers import BoardClusterer                   # optional

# Model classes (Lightning) used by the inference wrappers
# Adjust paths if different in your repo:



# ============================
# Paths to assets (edit as needed)
# ============================
CKPTS = {
    "equity":    {"ckpt": "checkpoints/equitynet/best.ckpt",            "sidecar": "checkpoints/equitynet/best_sidecar.json"},
    "popnet":    {"ckpt": "checkpoints/popnet/best.ckpt",               "sidecar": "checkpoints/popnet/best_sidecar.json"},
    "preflop":   {"ckpt": "checkpoints/range_pre/best.ckpt",            "sidecar": "checkpoints/range_pre/best_sidecar.json"},
    "post_root": {"ckpt": "checkpoints/postflop_policy_root/best.ckpt", "sidecar": "checkpoints/postflop_policy_root/best_sidecar.json"},
    "post_face": {"ckpt": "checkpoints/postflop_policy_facing/best.ckpt","sidecar":"checkpoints/postflop_policy_facing/best_sidecar.json"},
    # Optional persisted clusterer, if you have one:
    "clusterer": {"ckpt": None, "sidecar": None},
}


# ============================
# Helpers
# ============================
def _load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSON sidecar: {p}")
    with p.open("r") as f:
        return json.load(f)


def _pp_response(title: str, resp: Any, debug_bytes: int = 900) -> None:
    print(f"\n=== {title} ===")
    # PolicyResponse API
    try:
        data = resp.as_dict()
        actions = data.get("actions", [])
        probs   = data.get("probs", [])
        top     = data.get("top_action", "NONE")
        dbg     = data.get("debug", {})
    except Exception:
        # Fallback to attrs
        actions = getattr(resp, "actions", [])
        probs   = getattr(resp, "probs", [])
        top     = actions[probs.index(max(probs))] if actions and probs else "NONE"
        dbg     = getattr(resp, "debug", {})
    print(f"top_action: {top}")
    for a, p in zip(actions, probs):
        print(f"{a:>12}: {p:.3f}")
    try:
        s = json.dumps(dbg, indent=2, default=str)
        print("debug:", (s if len(s) <= debug_bytes else s[:debug_bytes] + "…"))
    except Exception:
        pass


# ============================
# Build dependencies (real objects)
# ============================
def build_equity_infer(device: Optional[torch.device] = None) -> EquityNetInfer:
    sc = _load_json(CKPTS["equity"]["sidecar"])
    model = EquityNetLit.load_from_checkpoint(CKPTS["equity"]["ckpt"], map_location=device or "cpu")
    return EquityNetInfer(
        model=model,
        feature_order=sc["feature_order"],
        id_maps=sc.get("id_maps", {}),
        cards=sc.get("cards", {}),
        device=device,
    )



def _load_json(path: str) -> Dict[str, Any]:
    import json, pathlib
    p = pathlib.Path(path)
    with p.open("r") as f:
        return json.load(f)

# --- updated build_preflop_policy using sidecar-derived init args ---
def build_preflop_policy(device: Optional[torch.device] = None) -> PreflopPolicy:
    sc = _load_json(CKPTS["preflop"]["sidecar"])
    cards = sc.get("cards") or {}
    feature_order = sc.get("feature_order") or []
    map_loc = device or torch.device("cpu")

    try:
        # Preferred: let Lightning reconstruct with sidecar-provided init args
        model = RangeNetLit.load_from_checkpoint(
            checkpoint_path=CKPTS["preflop"]["ckpt"],
            map_location=map_loc,
            cards=cards,
            feature_order=feature_order,
        )
    except TypeError:
        # Fallback: manual init + strict state load
        model = RangeNetLit(cards=cards, feature_order=feature_order)
        state = torch.load(CKPTS["preflop"]["ckpt"], map_location=map_loc)
        # Lightning ckpts often store weights under "state_dict"
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        model.load_state_dict(state_dict, strict=True)

    # Construct your inference wrapper with the same sidecar metadata
    return PreflopPolicy(
        model=model,
        feature_order=feature_order,
        cards=cards,
        id_maps=sc.get("id_maps", {}),
        device=map_loc if isinstance(map_loc, torch.device) else torch.device(str(map_loc)),
    )


def build_postflop_router(device: Optional[torch.device] = None) -> PostflopPolicyRouter:
    """Construct router from two single-side inferencers."""
    global torch
    sc_root = _load_json(CKPTS["post_root"]["sidecar"])
    sc_face = _load_json(CKPTS["post_face"]["sidecar"])

    dev_str = "auto"
    try:
        import torch
        if device is not None and isinstance(device, torch.device):
            dev_str = device.type  # "cpu" | "cuda"
        elif isinstance(device, str) and device:
            dev_str = device
    except Exception:
        pass

    m_root = PostflopPolicyInferSingle.from_checkpoint(
        checkpoint_path=CKPTS["post_root"]["ckpt"],
        sidecar_path=CKPTS["post_root"]["sidecar"],
        device=dev_str,
    )
    m_face = PostflopPolicyInferSingle.from_checkpoint(
        checkpoint_path=CKPTS["post_face"]["ckpt"],
        sidecar_path=CKPTS["post_face"]["sidecar"],
        device=dev_str,
    )

    m_root = PostflopPolicyInferSingle.from_checkpoint(CKPTS["post_root"]["ckpt"], CKPTS["post_root"]["sidecar"],
                                                       device="cpu")
    m_face = PostflopPolicyInferSingle.from_checkpoint(CKPTS["post_face"]["ckpt"], CKPTS["post_face"]["sidecar"],
                                                       device="cpu")
    return PostflopPolicyRouter(root=m_root, facing=m_face, device=torch.device("cpu"))


def build_population_infer(device: Optional[torch.device] = None) -> PopulationNetInference:
    return PopulationNetInference(CKPTS["popnet"]["ckpt"], device=device)


def build_exploit_store() -> PlayerExploitStore:
    return PlayerExploitStore(cfg=None)


def build_clusterer():
    # If you have a persisted clusterer, load it here. Otherwise return None.
    return None


def build_deps(device: Optional[torch.device] = None) -> PolicyInferDeps:
    return PolicyInferDeps(
        pop=build_population_infer(device),
        exploit=build_exploit_store(),
        equity=build_equity_infer(device),
        range_pre=build_preflop_policy(device),
        policy_post=build_postflop_router(device),
        clusterer=build_clusterer(),
        params={},
    )


# ============================
# Requests to test
# ============================
def payloads() -> List[tuple[str, Dict[str, Any]]]:
    preflop = (
        "Preflop (BTN vs BB)",
        {
            "stakes": "NL10",
            "street": 0,
            "ctx": "VS_OPEN",
            "hero_pos": "BTN",
            "villain_pos": "BB",
            "hero_hand": "AhKh",
            "board": None,
            "pot_bb": 1.5,
            "eff_stack_bb": 100.0,
            "facing_bet": False,
            "allow_allin": False,
            "actions_hist": ["pre: BTN open 2.5", "BB call"],
            "raw": {},
        },
    )
    postflop_root_ip = (
        "Postflop ROOT (IP, flop Ad7d4c)",
        {
            "stakes": "NL50",
            "street": 1,
            "ctx": "VS_OPEN",
            "hero_pos": "BTN",
            "villain_pos": "BB",
            "hero_hand": "AsKs",
            "board": "Ad7d4c",
            "pot_bb": 7.0,
            "eff_stack_bb": 95.0,
            "facing_bet": False,
            "bet_sizes": [0.33, 0.66],
            "allow_allin": True,
            "villain_id": "villain-001",
            "actions_hist": ["pre: BTN raise 2.5", "BB call"],
            "raw": {},
        },
    )
    postflop_facing_oop = (
        "Postflop FACING (OOP BB vs 33%)",
        {
            "stakes": "NL50",
            "street": 1,
            "ctx": "VS_OPEN",
            "hero_pos": "BB",
            "villain_pos": "BTN",
            "hero_hand": "9d9c",
            "board": "Ad7d4c",
            "pot_bb": 7.0,
            "eff_stack_bb": 95.0,
            "facing_bet": True,
            "faced_size_pct": 33.0,
            "faced_size_frac": 0.33,
            "raise_buckets": [150, 200, 300],
            "allow_allin": True,
            "villain_id": "villain-001",
            "actions_hist": ["pre: BTN raise 2.5", "BB call", "flop: BTN bet 33"],
            "raw": {},
        },
    )
    return [preflop, postflop_root_ip, postflop_facing_oop]


# ============================
# Main
# ============================
def main() -> int:
    device = torch.device("cpu")  # keep cpu for smoke

    # Fail fast if files are missing
    for k, v in CKPTS.items():
        for key in ("ckpt", "sidecar"):
            p = v.get(key)
            if p and not Path(p).exists():
                print(f"[warn] Missing {k}.{key}: {p} (edit CKPTS at top if different)")

    deps = build_deps(device)
    infer = PolicyInfer(deps, blend_cfg=PolicyBlendConfig.default())

    for title, payload in payloads():
        try:
            resp = infer.predict(payload)
            _pp_response(title, resp)
        except Exception as e:
            print(f"\n=== {title} FAILED ===")
            print("payload:", json.dumps(payload, indent=2))
            print("error  :", repr(e))
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())