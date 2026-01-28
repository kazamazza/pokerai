import json
from pathlib import Path

import torch
import torch.nn as nn

from ml.infer.postflop.router import route_postflop_state
from ml.infer.postflop.encoder import load_postflop_infer_artifacts, encode_postflop_state_dense
from ml.models.vocab_actions import ROOT_ACTION_VOCAB, FACING_ACTION_VOCAB


class DummyNet(nn.Module):
    def __init__(self, out_dim: int, bias: float = 0.0):
        super().__init__()
        self.out_dim = out_dim
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # deterministic logits independent of x
        # shape: [B, out_dim]
        b = x.shape[0]
        base = torch.arange(self.out_dim, dtype=torch.float32).unsqueeze(0).repeat(b, 1)
        return base + self.bias


def test_router_picks_correct_target():
    s_root = {"size_pct": 33}
    r1 = route_postflop_state(s_root)
    assert r1.target == "root"
    assert r1.size_pct == 33

    s_face = {"faced_size_pct": 50}
    r2 = route_postflop_state(s_face)
    assert r2.target == "facing"
    assert r2.faced_size_pct == 50


def test_encoder_and_probs_sum_to_one(tmp_path: Path):
    # ---- write minimal cat vocabs ----
    cat_vocabs = {
        "vocabs": {
            "ctx": {"SRP": 0, "VS_3BET": 1},
            "ip_pos": {"BTN": 0, "SB": 1},
            "oop_pos": {"BB": 0},
            "bet_sizing_id": {"srp_hu.PFR_IP": 0},
            "solver_version": {"v1": 0},
            "board_cluster_id": {"0": 0, "1": 1},
            "street": {"1": 0, "2": 1, "3": 2},
        }
    }
    voc_path = tmp_path / "cat_vocabs.json"
    voc_path.write_text(json.dumps(cat_vocabs))

    # ---- sidecars ----
    # feature_order mixes cat + scalars; cont_features is empty in your newest plan
    root_sidecar = {
        "feature_order": ["solver_version", "street", "ctx", "ip_pos", "oop_pos", "board_cluster_id", "pot_bb", "effective_stack_bb", "bet_sizing_id", "size_pct"],
        "cont_features": [],
        "action_vocab": list(ROOT_ACTION_VOCAB),
    }
    facing_sidecar = {
        "feature_order": ["solver_version", "street", "ctx", "ip_pos", "oop_pos", "board_cluster_id", "pot_bb", "effective_stack_bb", "bet_sizing_id", "faced_size_pct"],
        "cont_features": [],
        "action_vocab": list(FACING_ACTION_VOCAB),
    }

    root_sc = tmp_path / "root_sidecar.json"
    face_sc = tmp_path / "facing_sidecar.json"
    root_sc.write_text(json.dumps(root_sidecar))
    face_sc.write_text(json.dumps(facing_sidecar))

    root_art = load_postflop_infer_artifacts(sidecar_path=root_sc, cat_vocabs_path=voc_path)
    face_art = load_postflop_infer_artifacts(sidecar_path=face_sc, cat_vocabs_path=voc_path)

    # ---- encode a root state ----
    state_root = {
        "solver_version": "v1",
        "street": 1,
        "ctx": "SRP",
        "ip_pos": "BTN",
        "oop_pos": "BB",
        "board_cluster_id": 0,
        "pot_bb": 10.0,
        "effective_stack_bb": 100.0,
        "bet_sizing_id": "srp_hu.PFR_IP",
        "size_pct": 33,
    }
    x = encode_postflop_state_dense(state=state_root, artifacts=root_art)
    assert x.ndim == 1
    assert x.numel() == len(root_sidecar["feature_order"])

    # ---- check deterministic softmax sums to 1 ----
    net = DummyNet(out_dim=len(ROOT_ACTION_VOCAB))
    logits = net(x.unsqueeze(0)).squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    s = float(probs.sum().item())
    assert abs(s - 1.0) < 1e-6