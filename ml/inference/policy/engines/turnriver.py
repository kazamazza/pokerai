from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

@dataclass
class TRConfig:
    # equity margins added on top of pot-odds to avoid thin punts
    call_margin_turn: float = 0.03
    call_margin_river: float = 0.00
    raise_gate_turn: float = 0.68   # need >= this to raise turn (or low SPR)
    raise_gate_river: float = 0.82  # near-nutty for river raises
    low_spr_raise: float = 2.2      # allow more raises when SPR is small
    ip_probe_eq: float = 0.56       # IP turn probe/bet when >= this
    ip_probe_light_eq: float = 0.50 # light 33% probe when SPR big and pool overfolds
    oop_donk_eq: float = 0.60       # optional OOP donk gate (off by default if not desired)

class TurnRiverHeuristics:
    def __init__(self, cfg: Optional[TRConfig] = None):
        self.cfg = cfg or TRConfig()

    @staticmethod
    def _pot_odds_to_call(pot_bb: float, bet_bb: float) -> float:
        # need equity >= bet / (pot + bet)
        return bet_bb / max(pot_bb + bet_bb, 1e-9)

    @staticmethod
    def _first_available(tokens: List[str], wanted: List[str]) -> Optional[str]:
        avail = set(tokens)
        for w in wanted:
            if w in avail: return w
        return None

    def decide_root(self, tokens: List[str], p_win: Optional[float], spr: Optional[float], *,
                    ip: bool, ctx: str, street: int, bet_menu_pcts: Optional[List[int]]) -> str:
        # default: check
        best = "CHECK"
        if p_win is None:
            # super conservative fallback
            return best if "CHECK" in tokens else (self._first_available(tokens, [f"BET_33","BET_66"]) or tokens[0])

        # Select size preference
        menu = bet_menu_pcts or [33, 66]
        big = f"BET_66" if 66 in menu else (f"BET_{menu[-1]}" if menu else None)
        small = f"BET_33" if 33 in menu else (f"BET_{menu[0]}" if menu else None)

        # Turn
        if street == 2:
            if p_win >= self.cfg.ip_probe_eq:
                best = big if big in tokens else (small or best)
            elif ip and p_win >= self.cfg.ip_probe_light_eq and (spr is None or spr > 6.0):
                # light pressure when IP + deep; pick small
                if small in tokens: best = small
        # River: bet thin for value when clearly ahead
        elif street == 3:
            if p_win >= 0.62:                # thin value
                best = small if small in tokens else (big or best)
            if p_win >= 0.72:                # clear value → bigger
                best = big if big in tokens else (small or best)

        return best if best in tokens else (self._first_available(tokens, ["CHECK"]) or tokens[0])

    def decide_facing(self, tokens: List[str], p_win: Optional[float], spr: Optional[float],
                      *, pot_bb: float, faced_frac: Optional[float], street: int) -> str:
        # defaults
        if "CALL" not in tokens: return tokens[0]

        if p_win is None or faced_frac is None or pot_bb <= 0:
            # no equity or size → take safe path (fold small equity only if CLEARLY dominated)
            return "CALL"

        bet_bb = faced_frac * pot_bb
        need = self._pot_odds_to_call(pot_bb, bet_bb)

        # River: pure pot-odds calling station bias is okay; turn: add margin
        margin = self.cfg.call_margin_river if street == 3 else self.cfg.call_margin_turn

        # Raise gates
        can_raise = any(t.startswith("RAISE_") for t in tokens)
        high_eq = p_win >= (self.cfg.raise_gate_river if street == 3 else self.cfg.raise_gate_turn)
        low_spr = (spr is not None and spr <= self.cfg.low_spr_raise)

        if can_raise and (high_eq or low_spr and p_win >= need + margin + 0.05):
            # pick the smallest available raise bucket by default
            for t in ("RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500"):
                if t in tokens: return t

        # Call / fold by pot odds
        return "CALL" if p_win >= need + margin else ("FOLD" if "FOLD" in tokens else "CALL")