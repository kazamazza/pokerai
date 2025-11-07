import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch


def _is_raise_like(tok: str) -> bool:
    T = tok.upper()
    return T.startswith("BET_") or T.startswith("RAISE_") or T.startswith("DONK_") or T == "ALLIN"

@dataclass
class ActionVocab:
    actions: List[str]
    index: Dict[str, int]
    _fcr_proj: Optional["torch.Tensor"] = None  # lazy

    @classmethod
    def from_actions(cls, actions: List[str]) -> "ActionVocab":
        return cls(actions=list(actions), index={a: i for i, a in enumerate(actions)})

    def update(self, actions: List[str]) -> None:
        if actions == self.actions:
            return
        self.actions = list(actions)
        self.index = {a: i for i, a in enumerate(actions)}
        self._fcr_proj = None  # invalidate

    def fcr_projection(self) -> "torch.Tensor":
        import torch
        if self._fcr_proj is not None:
            return self._fcr_proj
        V = len(self.actions)
        P = torch.zeros(3, V, dtype=torch.float32)
        if "FOLD" in self.index: P[0, self.index["FOLD"]] = 1.0
        if "CALL" in self.index: P[1, self.index["CALL"]] = 1.0
        raise_like = [i for a, i in self.index.items() if _is_raise_like(a)]
        if raise_like:
            w = 1.0 / float(len(raise_like))
            for i in raise_like: P[2, i] = w
        self._fcr_proj = P
        return self._fcr_proj

    def sha256(self) -> str:
        m = hashlib.sha256()
        for a in self.actions:
            m.update(a.encode("utf-8"))
        return m.hexdigest()[:12]

    def allin_idx(self) -> Optional[int]:
        return self.index.get("ALLIN")