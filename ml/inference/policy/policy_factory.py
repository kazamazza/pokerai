import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from ml.features.boards import load_board_clusterer
from ml.features.boards.board_clusterers.rule_based import RuleBasedBoardClusterer
from ml.inference.equity import EquityNetInfer
from ml.inference.player_exploit_store import PlayerExploitStore
from ml.inference.policy.deps import PolicyInferDeps
from ml.inference.policy.policy import PolicyInfer
from ml.inference.population import PopulationNetInference
from ml.inference.postflop_infer_single import PostflopPolicyInferSingle
from ml.inference.postflop_router import PostflopPolicyRouter
from ml.inference.preflop import PreflopPolicy
from ml.models.equity_net import EquityNetLit
from ml.models.preflop_rangenet import RangeNetLit


class PolicyInferFactory:
    CKPTS = {
        "equity":    {"ckpt": "checkpoints/equitynet/best.ckpt",            "sidecar": "checkpoints/equitynet/best_sidecar.json"},
        "popnet":    {"ckpt": "checkpoints/popnet/best.ckpt",               "sidecar": "checkpoints/popnet/best_sidecar.json"},
        "preflop":   {"ckpt": "checkpoints/range_pre/best.ckpt",            "sidecar": "checkpoints/range_pre/best_sidecar.json"},
        "post_root": {"ckpt": "checkpoints/postflop_policy_root/best.ckpt", "sidecar": "checkpoints/postflop_policy_root/best_sidecar.json"},
        "post_face": {"ckpt": "checkpoints/postflop_policy_facing/best.ckpt","sidecar":"checkpoints/postflop_policy_facing/best_sidecar.json"},
        "clusterer": {"ckpt": None, "sidecar": None},
    }

    ARTIFACT = "data/artifacts/board_clusters_kmeans_64.json"

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")

    def _load_json(self, path: str | Path) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing JSON sidecar: {p}")
        with p.open("r") as f:
            return json.load(f)

    def _build_equity_infer(self) -> EquityNetInfer:
        sc = self._load_json(self.CKPTS["equity"]["sidecar"])
        model = EquityNetLit.load_from_checkpoint(self.CKPTS["equity"]["ckpt"], map_location=self.device)
        return EquityNetInfer(
            model=model,
            feature_order=sc["feature_order"],
            id_maps=sc.get("id_maps", {}),
            cards=sc.get("cards", {}),
            device=self.device,
        )

    def _build_preflop_policy(self) -> PreflopPolicy:
        sc = self._load_json(self.CKPTS["preflop"]["sidecar"])
        feature_order = sc.get("feature_order") or []
        cards = sc.get("cards") or {}

        try:
            model = RangeNetLit.load_from_checkpoint(
                checkpoint_path=self.CKPTS["preflop"]["ckpt"],
                map_location=self.device,
                cards=cards,
                feature_order=feature_order,
            )
        except TypeError:
            model = RangeNetLit(cards=cards, feature_order=feature_order)
            state = torch.load(self.CKPTS["preflop"]["ckpt"], map_location=self.device)
            model.load_state_dict(state["state_dict"], strict=True)

        return PreflopPolicy(
            model=model,
            feature_order=feature_order,
            cards=cards,
            id_maps=sc.get("id_maps", {}),
            device=self.device,
        )

    def _build_postflop_router(self) -> PostflopPolicyRouter:
        dev_str = str(self.device) if self.device else "auto"
        m_root = PostflopPolicyInferSingle.from_checkpoint(
            self.CKPTS["post_root"]["ckpt"], self.CKPTS["post_root"]["sidecar"], device=dev_str
        )
        m_face = PostflopPolicyInferSingle.from_checkpoint(
            self.CKPTS["post_face"]["ckpt"], self.CKPTS["post_face"]["sidecar"], device=dev_str
        )
        return PostflopPolicyRouter(root=m_root, facing=m_face, device=self.device)

    def _build_population_infer(self) -> PopulationNetInference:
        return PopulationNetInference(self.CKPTS["popnet"]["ckpt"], device=self.device)

    def _build_exploit_store(self) -> PlayerExploitStore:
        return PlayerExploitStore(cfg=None)

    def _build_clusterer(self):
        try:
            if not Path(self.ARTIFACT).exists():
                print(f"[warn] board cluster artifact not found: {self.ARTIFACT} -> falling back to rule-based(128)")
                return RuleBasedBoardClusterer(n_clusters=128)
            return load_board_clusterer({
                "board_clustering": {
                    "type": "kmeans",
                    "artifact": self.ARTIFACT,
                    "n_clusters": 64,
                }
            })
        except Exception as e:
            print(f"[warn] failed to load board clusterer ({e}); using rule-based(128)")
            return RuleBasedBoardClusterer(n_clusters=128)

    def create(self) -> PolicyInfer:
        deps = PolicyInferDeps(
            pop=self._build_population_infer(),
            exploit=self._build_exploit_store(),
            equity=self._build_equity_infer(),
            range_pre=self._build_preflop_policy(),
            policy_post=self._build_postflop_router(),
            clusterer=self._build_clusterer(),
            params={},
        )
        return PolicyInfer(deps)