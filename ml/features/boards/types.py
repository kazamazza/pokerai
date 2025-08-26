# ml/features/boards/types.py
from __future__ import annotations
from typing import Protocol, List, Optional

class BoardClusterer(Protocol):
    n_clusters: Optional[int]  # may be None for rule-based

    def predict(self, boards: List[str]) -> List[int]: ...
    def predict_one(self, board: str) -> int: ...