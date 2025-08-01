from typing import List, Literal, Optional, Union
from pydantic import BaseModel


class ActionBranch(BaseModel):
    action: str
    size: Optional[float] = None
    frequency: float
    next: Optional[dict] = None

class StrategyNode(BaseModel):
    combos: List[str]             # ["AhKs", "QdTs", ...]
    actions: List[ActionBranch]   # branching options for this combo

class ClusterStrategy(BaseModel):
    cluster_id: int               # 0 → 63 (if 64 clusters)
    board: str                    # canonical flop board (e.g. "JdTs9h")
    ip_range: List[str]           # combos from preflop chart
    oop_range: List[str]
    ip_strategy: StrategyNode     # root node for in-position
    oop_strategy: StrategyNode    # root node for out-of-position