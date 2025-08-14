BLINDS_SB = 0.5
BLINDS_BB = 1.0

def compute_srp_flop_pot_bb(open_size_bb: float, ante_bb: float = 0.0) -> float:
    # blinds + antes + open (uncalled) + call (simplified)
    # SB=0.5, BB=1.0 ⇒ blinds 1.5; one caller adds open_size to pot
    return 0.5 + 1.0 + ante_bb + open_size_bb + open_size_bb

def compute_3bet_flop_pot_bb(open_size_bb: float, threebet_size_bb: float, ante_bb: float = 0.0) -> float:
    # SB + BB + antes + open + 3bet + call of 3bet
    return 0.5 + 1.0 + ante_bb + open_size_bb + threebet_size_bb + threebet_size_bb

def compute_4bet_flop_pot_bb(open_size_bb: float, threebet_size_bb: float, fourbet_size_bb: float, ante_bb: float = 0.0) -> float:
    # SB + BB + antes + open + 3bet + 4bet + call of 4bet
    return 0.5 + 1.0 + ante_bb + open_size_bb + threebet_size_bb + fourbet_size_bb + fourbet_size_bb

def get_flop_pot_from_context(action_context: str) -> float:
    """
    Quick, deterministic pot estimate in BBs. Replace with your exact pot-tracker if you have one.
    """
    ctx = action_context.upper()
    OPEN_SIZE = 2.5
    THREEBET_SIZE = 9.0     # tune to your preflop sizing
    FOURBET_SIZE  = 22.0    # tune to your preflop sizing
    ANTE_BB = 0.0

    if ctx == "OPEN" or ctx == "VS_LIMP" or ctx == "VS_ISO":
        return compute_srp_flop_pot_bb(OPEN_SIZE, ANTE_BB)
    elif ctx == "VS_3BET":
        return compute_3bet_flop_pot_bb(OPEN_SIZE, THREEBET_SIZE, ANTE_BB)
    elif ctx == "VS_4BET":
        return compute_4bet_flop_pot_bb(OPEN_SIZE, THREEBET_SIZE, FOURBET_SIZE, ANTE_BB)
    else:
        # Default to SRP if unknown
        return compute_srp_flop_pot_bb(OPEN_SIZE, ANTE_BB)