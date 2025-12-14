import torch

class FCRProjector:
    def __init__(self):
        self._cached_actions: list[str] | None = None
        self._P: torch.Tensor | None = None

    def _build(self, actions: list[str], dtype, device) -> torch.Tensor:
        V = len(actions)
        P = torch.zeros(3, V, dtype=dtype, device=device)
        raise_mask = torch.zeros(V, dtype=dtype, device=device)
        for i, tok in enumerate(actions):
            T = tok.upper()
            if T == "FOLD": P[0, i] = 1.0
            if T == "CALL": P[1, i] = 1.0
            if T.startswith("BET_") or T.startswith("RAISE_") or T.startswith("DONK_") or T == "ALLIN":
                raise_mask[i] = 1.0
        s = raise_mask.sum()
        if s.item() > 0:
            P[2] = raise_mask / s
        self._cached_actions = list(actions)
        self._P = P
        return P

    def lift(self, sig3, actions: list[str], dtype, device) -> torch.Tensor:
        if not isinstance(sig3, torch.Tensor):
            sig3 = torch.tensor(sig3, dtype=dtype, device=device)
        sig3 = sig3.view(3)
        if self._P is None or self._cached_actions != actions:
            P = self._build(actions, dtype, device)
        else:
            P = self._P.to(dtype=dtype, device=device)
        delta = torch.matmul(sig3, P)  # [V]
        return delta.view(1, -1)