from dataclasses import dataclass
from typing import Dict, Any
from ml.features.boards import BoardClusterer
from ml.inference.equity import EquityNetInfer
from ml.inference.player_exploit_store import PlayerExploitStore
from ml.inference.population import PopulationNetInference
from ml.inference.postflop import PostflopPolicyInfer
from ml.inference.preflop import PreflopPolicy


@dataclass
class PolicyInferDeps:
    pop: PopulationNetInference | None = None
    exploit: PlayerExploitStore | None = None
    equity: EquityNetInfer | None = None
    range_pre: PreflopPolicy | None = None
    policy_post: PostflopPolicyInfer | None = None
    clusterer: BoardClusterer | None = None
    params: Dict[str, Any] | None = None