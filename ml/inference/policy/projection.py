# policy/projection.py
import torch
from typing import List

class FCRProjector:
    def __init__(self):
        self._cache = None
        self._for_vocab = None

    def build(self, actions: List[str]) -> torch.Tensor:
        if self._for_vocab == actions and self._cache is not None:
            return self._cache
        V = len(actions)
        P = torch.zeros(3, V, dtype=torch.float32)
        vix = {a:i for i,a in enumerate(actions)}
        if "FOLD" in vix: P[0, vix["FOLD"]] = 1.0
        if "CALL" in vix: P[1, vix["CALL"]] = 1.0
        raise_like = [i for i,t in enumerate(actions)
                      if t.startswith(("BET_","RAISE_")) or t in ("ALLIN","DONK_33","DONK_50")]
        if raise_like:
            w = 1.0/len(raise_like)
            for i in raise_like: P[2,i] = w
        self._cache, self._for_vocab = P, list(actions)
        return P

    def lift(self, sig3, actions: List[str], dtype, device) -> torch.Tensor:
        if not isinstance(sig3, torch.Tensor):
            sig3 = torch.tensor(sig3, dtype=dtype, device=device).view(3)
        P = self.build(actions).to(device=device, dtype=dtype)  # [3,V]
        return torch.matmul(sig3, P).view(1,-1)