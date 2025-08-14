# utils/pots.py

BLINDS_SB = 0.5
BLINDS_BB = 1.0

def _call_top_up(open_size_bb: float, caller_pos: str) -> float:
    p = caller_pos.upper()
    if p == "BB":
        return max(open_size_bb - BLINDS_BB, 0.0)
    if p == "SB":
        return max(open_size_bb - BLINDS_SB, 0.0)
    return open_size_bb  # field positions hadn’t posted a blind

def compute_srp_flop_pot_bb(
    opener_pos: str,
    caller_pos: str,
    *,
    open_size_bb: float = 2.5,
    num_players: int = 6,
    ante_bb: float = 0.0
) -> float:
    antes_total = num_players * ante_bb
    caller_add = _call_top_up(open_size_bb, caller_pos)
    return BLINDS_SB + BLINDS_BB + antes_total + open_size_bb + caller_add

def compute_3bet_flop_pot_bb(
    threebet_size_bb: float,
    *,
    num_players: int = 6,
    ante_bb: float = 0.0
) -> float:
    antes_total = num_players * ante_bb
    return BLINDS_SB + BLINDS_BB + antes_total + 2.0 * threebet_size_bb

def compute_4bet_flop_pot_bb(
    fourbet_size_bb: float,
    *,
    num_players: int = 6,
    ante_bb: float = 0.0
) -> float:
    antes_total = num_players * ante_bb
    return BLINDS_SB + BLINDS_BB + antes_total + 2.0 * fourbet_size_bb

def get_flop_pot_from_context(
    *,
    action_context: str,
    opener_pos: str,
    caller_pos: str,
    open_size_bb: float = 2.5,
    threebet_size_bb: float = 8.5,
    fourbet_size_bb: float = 22.0,
    num_players: int = 6,
    ante_bb: float = 0.0
) -> float:
    ctx = action_context.upper()
    if ctx == "OPEN":
        return compute_srp_flop_pot_bb(
            opener_pos, caller_pos,
            open_size_bb=open_size_bb, num_players=num_players, ante_bb=ante_bb
        )
    if ctx == "VS_3BET":
        return compute_3bet_flop_pot_bb(
            threebet_size_bb, num_players=num_players, ante_bb=ante_bb
        )
    if ctx == "VS_4BET":
        return compute_4bet_flop_pot_bb(
            fourbet_size_bb, num_players=num_players, ante_bb=ante_bb
        )
    # Fallback: SRP assumption
    return compute_srp_flop_pot_bb(
        opener_pos, caller_pos,
        open_size_bb=open_size_bb, num_players=num_players, ante_bb=ante_bb
    )