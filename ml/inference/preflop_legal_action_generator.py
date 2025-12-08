# file: ml/inference/preflop/generator.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ml.models.vocab_actions import PREFLOP_ACTION_VOCAB


@dataclass
class PreflopLegalActionGenerator:
    """
    Deterministic preflop action generator (centi-bb tokens).
    - OPEN_250 => 2.50bb total.
    - RAISE_750 => 7.50bb final total vs faced.
    - Emits CHECK only when BB has a free option (caller sets free_check=True).
    - Filters by stack and faced size; preserves stable ordering.
    """
    open_sizes_cbb: Sequence[int]  = field(default_factory=lambda: (200, 250, 300))
    raise_totals_cbb: Sequence[int] = field(default_factory=lambda: (600, 750, 900, 1200))
    allow_allin: bool = False
    max_open_cbb: Optional[int] = None  # None -> cap at stack
    _tol: float = 1e-9

    def _order_key(self, tok: str) -> Tuple[int, float]:
        if tok == "FOLD":  return (0, 0.0)
        if tok == "CHECK": return (1, 0.0)
        if tok == "CALL":  return (2, 0.0)
        if tok.startswith("OPEN_"):
            try: v = float(tok.split("_", 1)[1])
            except Exception: v = 0.0
            return (3, v)
        if tok.startswith("RAISE_"):
            try: v = float(tok.split("_", 1)[1])
            except Exception: v = 0.0
            return (4, v)
        if tok == "ALLIN": return (5, float("inf"))
        return (9, 0.0)

    @staticmethod
    def _bb_to_cbb(x: float) -> int:
        return int(round(max(0.0, float(x)) * 100))

    @staticmethod
    def _cbb_to_bb(x: int) -> float:
        return max(0, int(x)) / 100.0

    def _cap_open(self, open_cbb: int, stack_cbb: int) -> bool:
        if open_cbb <= 0: return False
        if self.max_open_cbb is not None and open_cbb > self.max_open_cbb: return False
        return open_cbb <= stack_cbb

    @staticmethod
    def _cap_raise(total_cbb: int, stack_cbb: int, faced_cbb: int) -> bool:
        if total_cbb <= faced_cbb: return False
        return total_cbb <= stack_cbb

    def generate(
        self,
        *,
        stack_bb: float,
        facing_bet: bool,
        faced_size_bb: Optional[float] = None,
        faced_frac: Optional[float] = None,
        free_check: bool = False,  # True when BB can check for free (no unopened action faced)
    ) -> List[str]:
        stack_cbb = self._bb_to_cbb(stack_bb)
        faced_cbb = 0
        if faced_size_bb is not None:
            faced_cbb = self._bb_to_cbb(faced_size_bb)
        elif faced_frac is not None and stack_bb > 0:
            faced_cbb = self._bb_to_cbb(float(faced_frac) * float(stack_bb))

        toks: List[str] = ["FOLD"]

        if not facing_bet:
            if free_check:
                toks.append("CHECK")
            for oc in self.open_sizes_cbb:
                oc = int(oc)
                if self._cap_open(oc, stack_cbb):
                    toks.append(f"OPEN_{oc}")
            if self.allow_allin and stack_cbb > 0:
                toks.append("ALLIN")
        else:
            if faced_cbb > 0:
                toks.append("CALL")
            for rc in self.raise_totals_cbb:
                rc = int(rc)
                if self._cap_raise(rc, stack_cbb, faced_cbb):
                    toks.append(f"RAISE_{rc}")
            if self.allow_allin and stack_cbb > 0:
                toks.append("ALLIN")

        # Dedup + stable order
        toks = sorted(set(toks), key=self._order_key)

        # Optional: clamp to known vocab (keeps training/inference aligned)
        vocab = set(PREFLOP_ACTION_VOCAB)
        toks = [t for t in toks if t in vocab]
        if not toks:
            # conservative safety: if everything filtered, expose minimal safe set
            toks = ["FOLD"] + (["CHECK"] if free_check else []) + (["CALL"] if facing_bet and faced_cbb > 0 else [])
        return toks