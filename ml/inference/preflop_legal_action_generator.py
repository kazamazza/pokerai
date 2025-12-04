class PreflopLegalActionGenerator:
    """
    Generates legal preflop action tokens like FOLD, CALL, OPEN_200, RAISE_300 etc
    based on stack, facing, and faced size.
    """

    def __init__(self, *, min_open_bb=2.0, raise_multipliers=[2.5, 3.0, 4.0], max_bb_factor=2.0):
        self.min_open_bb = min_open_bb
        self.raise_multipliers = raise_multipliers
        self.max_bb_factor = max_bb_factor

    def generate(self, stack_bb: float, facing_bet: bool, faced_frac: float | None = None) -> list[str]:
        faced_frac = faced_frac or 0.0
        max_raise = stack_bb * self.max_bb_factor
        tokens = []

        if facing_bet:
            tokens.append("FOLD")
            if faced_frac > 0.0:
                tokens.append("CALL")
            for m in self.raise_multipliers:
                amt = round(faced_frac * stack_bb * m)
                if self.min_open_bb <= amt <= max_raise:
                    tokens.append(f"RAISE_{int(amt)}")
        else:
            tokens.append("FOLD")
            for m in self.raise_multipliers:
                amt = round(self.min_open_bb * m)
                if amt <= max_raise:
                    tokens.append(f"OPEN_{int(amt)}")

        return sorted(set(tokens), key=lambda x: ["FOLD", "CALL", "CHECK"].index(x) if x in ["FOLD", "CALL", "CHECK"] else 99)