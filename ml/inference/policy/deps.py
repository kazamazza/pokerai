from dataclasses import dataclass
from typing import Dict, Any
from ml.features.boards import BoardClusterer
from ml.inference.equity import EquityNetInfer
from ml.inference.ev.router import EVRouter
from ml.inference.player_exploit_store import PlayerExploitStore
from ml.inference.population import PopulationNetInference
from ml.inference.postflop_router import PostflopPolicyRouter
from ml.inference.preflop import PreflopPolicy


@dataclass
class PolicyInferDeps:
    pop: PopulationNetInference | None = None
    exploit: PlayerExploitStore | None = None
    equity: EquityNetInfer | None = None
    range_pre: PreflopPolicy | None = None
    policy_post: PostflopPolicyRouter | None = None
    clusterer: BoardClusterer | None = None
    ev: EVRouter | None = None
    params: Dict[str, Any] | None = None