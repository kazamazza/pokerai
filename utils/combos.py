def get_169_combo_list() -> list[str]:
    ranks = ['A','K','Q','J','T','9','8','7','6','5','4','3','2']
    combos = []
    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            if i == j:
                combos.append(f"{r1}{r2}")   # pairs
            elif i < j:
                combos.append(f"{r1}{r2}s") # suited
            else:
                combos.append(f"{r2}{r1}o") # offsuit (already high-first)
    return combos