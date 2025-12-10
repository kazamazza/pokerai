from __future__ import annotations
from typing import Optional, Sequence, Set
import torch

from ml.models.vocab_actions import ROOT_ACTION_VOCAB, FACING_ACTION_VOCAB


# ---------- helpers ----------
def _has_size(menu: Optional[Sequence[float]], target: float, tol: float = 1e-3) -> bool:
    if not menu:
        return False
    for s in menu:
        try:
            if abs(float(s) - float(target)) < tol:
                return True
        except Exception:
            continue
    return False

def _bet_tokens_from_menu(menu: Optional[Sequence[float]]) -> Set[str]:
    if not menu:
        return set()
    want: Set[str] = set()
    if _has_size(menu, 0.25): want.add("BET_25")
    if _has_size(menu, 0.33): want.add("BET_33")
    if _has_size(menu, 0.50): want.add("BET_50")
    if _has_size(menu, 0.66): want.add("BET_66")
    if _has_size(menu, 0.75): want.add("BET_75")
    if _has_size(menu, 1.00): want.add("BET_100")
    return want

def _raise_token(mult: float, tol: float = 1e-3) -> Optional[str]:
    table = {1.5: "RAISE_150", 2.0: "RAISE_200", 3.0: "RAISE_300", 4.0: "RAISE_400", 5.0: "RAISE_500"}
    for k, tok in table.items():
        if abs(float(mult) - k) < tol:
            return tok
    return None

def _raise_tokens_from_buckets(buckets: Optional[Sequence[float]]) -> Set[str]:
    if not buckets:
        return set()
    out: Set[str] = set()
    for b in buckets:
        tok = _raise_token(b)
        if tok:
            out.add(tok)
    return out

# ---------- masks ----------
def mask_root(
    actions: Sequence[str],
    *,
    actor: str,
    ctx: Optional[str],
    bet_menu: Optional[Sequence[float]],
    allow_allin: Optional[bool] = None,  # ignored for root unless root vocab includes ALLIN
) -> torch.Tensor:
    A = [str(a).upper() for a in actions]
    actor_norm = (actor or "").strip().lower()
    ctx_norm = (ctx or "").strip().upper()

    # Legal set seeded from canonical root vocab
    legal: Set[str] = set()
    if "CHECK" in ROOT_ACTION_VOCAB:
        legal.add("CHECK")
    legal |= (_bet_tokens_from_menu(bet_menu) & set(ROOT_ACTION_VOCAB))

    # DONK_33 only when OOP and ctx in {VS_OPEN, LIMPED_SINGLE} and 0.33 is on the menu
    if (
        "DONK_33" in ROOT_ACTION_VOCAB
        and actor_norm == "oop"
        and ctx_norm in {"VS_OPEN", "LIMPED_SINGLE"}
        and _has_size(bet_menu, 0.33)
    ):
        legal.add("DONK_33")

    # Root doesn’t normally expose ALLIN; keep disabled even if present in vocab unless you opt-in later.

    m = torch.zeros(len(A), dtype=torch.float32)
    for i, tok in enumerate(A):
        if tok in legal:
            m[i] = 1.0
    if m.sum().item() == 0:
        m.fill_(1.0)
    return m

def mask_facing(
    actions: Sequence[str],
    *,
    raise_buckets: Optional[Sequence[float]],
    allow_allin: Optional[bool] = None,
) -> torch.Tensor:
    A = [str(a).upper() for a in actions]
    allow_ai = True if allow_allin is None else bool(allow_allin)

    legal: Set[str] = set()
    if "FOLD" in FACING_ACTION_VOCAB:
        legal.add("FOLD")
    if "CALL" in FACING_ACTION_VOCAB:
        legal.add("CALL")
    legal |= (_raise_tokens_from_buckets(raise_buckets) & set(FACING_ACTION_VOCAB))
    if allow_ai and "ALLIN" in FACING_ACTION_VOCAB:
        legal.add("ALLIN")

    m = torch.zeros(len(A), dtype=torch.float32)
    for i, tok in enumerate(A):
        if tok in legal:
            m[i] = 1.0
    if m.sum().item() == 0:
        m.fill_(1.0)
    return m