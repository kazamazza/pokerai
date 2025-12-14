from dataclasses import dataclass, field
from typing import List, Sequence, Optional, Tuple

@dataclass
class PostflopLegalActionGenerator:
    """
    Deterministic postflop action generator (percent-based menus).
    - Root (no bet faced): CHECK + BET_{pct} [+ ALLIN?]
    - Facing a bet: FOLD + CALL + RAISE_{pct} [+ ALLIN?]
    - Never emits DONK_*; ordering is stable.
    """
    bet_menu_pcts: Sequence[int] = field(default_factory=lambda: (33, 50, 66, 100))
    raise_pcts: Sequence[int]    = field(default_factory=lambda: (150, 200, 250, 300, 400, 500))
    allow_allin: bool            = False

    def _order_key(self, tok: str) -> Tuple[int, float]:
        if tok == "FOLD":   return (0, 0.0)
        if tok == "CHECK":  return (1, 0.0)
        if tok == "CALL":   return (2, 0.0)
        if tok.startswith("BET_"):
            try: v = float(tok.split("_", 1)[1])
            except Exception: v = 0.0
            return (3, v)
        if tok.startswith("RAISE_"):
            try: v = float(tok.split("_", 1)[1])
            except Exception: v = 0.0
            return (4, v)
        if tok == "ALLIN":  return (5, float("inf"))
        return (9, 0.0)

    @staticmethod
    def _norm_pcts(xs: Optional[Sequence[float | int]]) -> List[int]:
        out: List[int] = []
        for x in (xs or []):
            try:
                xi = int(round(float(x)))
                if 1 <= xi <= 5000:  # sane guard; 5000% pot is… not our menu :)
                    out.append(xi)
            except Exception:
                pass
        # dedup + sorted
        return sorted(set(out))

    def generate(
        self,
        *,
        side: str,                # "root" or "facing"
        bet_sizes: Optional[Sequence[float | int]] = None,   # e.g. [0.33, 0.66] or [33, 66]
        raise_pcts: Optional[Sequence[float | int]] = None,  # e.g. [150, 200, 300]
        allow_allin: Optional[bool] = None,
    ) -> List[str]:
        allow_ai = self.allow_allin if allow_allin is None else bool(allow_allin)

        bs = self._norm_pcts([x*100 for x in bet_sizes] if bet_sizes and any(isinstance(x, float) and x<=1.0 for x in bet_sizes)
                             else bet_sizes)
        rs = self._norm_pcts(raise_pcts)

        if not bs:
            bs = list(self.bet_menu_pcts)
        if not rs:
            rs = list(self.raise_pcts)

        toks: List[str] = []
        if side == "root":
            toks.append("CHECK")
            toks.extend([f"BET_{p}" for p in bs])
            if allow_ai:
                toks.append("ALLIN")
        else:  # facing
            toks.extend(["FOLD", "CALL"])
            toks.extend([f"RAISE_{p}" for p in rs])
            if allow_ai:
                toks.append("ALLIN")

        # stable order
        toks = sorted(set(toks), key=self._order_key)
        return toks