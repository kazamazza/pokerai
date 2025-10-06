from dataclasses import dataclass
from typing import Dict, Any
from ml.features.boards import BoardClusterer
from ml.inference.equity import EquityNetInfer
from ml.inference.exploit import ExploitNetInfer
from ml.inference.population import PopulationNetInference
from ml.inference.postflop import PostflopPolicyInfer
from ml.inference.preflop import PreflopPolicy


@dataclass
class PolicyInferDeps:
    # optional
    pop: PopulationNetInference | None = None
    # required
    exploit: ExploitNetInfer | None = None
    equity: EquityNetInfer | None = None
    # rangenets
    range_pre: PreflopPolicy | None = None
    # postflop policy
    policy_post: PostflopPolicyInfer | None = None
    # utils
    clusterer: BoardClusterer | None = None
    params: Dict[str, Any] | None = None