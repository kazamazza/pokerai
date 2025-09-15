ORDER = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
IDX = {p: i for i, p in enumerate(ORDER)}
LATE = {"CO", "BTN"}
BLINDS = {"SB", "BB"}

def canon_pos(p: str) -> str:
    return str(p).upper()

def dist(opener: str, defender: str) -> int:
    """# of seats strictly between opener and defender (mod 6)."""
    return (IDX[defender] - IDX[opener] - 1) % len(ORDER)

# Opener fallback ladders (walk to nearby seats in both directions)
OPEN_SUBS = {
    "UTG": ["HJ", "CO", "BTN"],
    "HJ":  ["UTG", "CO", "BTN"],
    "CO":  ["HJ", "BTN", "UTG"],
    "BTN": ["CO", "HJ", "UTG"],
    # SB “opener” is rare in SRP_IP; include safe options
    "SB":  ["BB", "CO", "HJ", "UTG"],
    "BB":  [],  # BB shouldn't be the opener in SRP
}

# Minimal defender fallback (mostly blind-vs-steal): swap blinds
DEF_SUBS = {
    "BB": ["SB"],
    "SB": ["BB"],
    "BTN": [], "CO": [], "HJ": [], "UTG": [],
}

def _guess_opener(ip: str, oop: str, ctx: str) -> str:
    """
    Heuristic opener guess from (ip, oop, ctx).
    - SRP-like: late-pos vs blind → late-pos opened; SB vs BB → SB opened.
    - Else: earlier seat in table order.
    """
    ip = canon_pos(ip); oop = canon_pos(oop); ctx = str(ctx).upper()
    if ctx in {"SRP", "VS_OPEN", "OPEN", "BLIND_VS_STEAL"}:
        if ip in LATE and oop in BLINDS:
            return ip
        if oop in LATE and ip in BLINDS:
            return oop
        if {ip, oop} == BLINDS:
            return "SB"
    return ip if IDX[ip] < IDX[oop] else oop

def candidate_pairs(ip: str, oop: str, *, ctx: str, allow_pair_subs: bool):
    """
    Yield (cand_ip, cand_oop, level_name, substituted: bool) in priority order.

    - Preserves original IP/OOP roles.
    - Substitutes opener first, then defender, then both.
    - Final SRP safety: early-pos vs blind → BTN vs same blind.
    """
    ip = canon_pos(ip); oop = canon_pos(oop); ctx = str(ctx).upper()

    def ok_for_ctx(opener: str, defender: str) -> bool:
        if ctx == "LIMP_MULTI":
            # require at least one seat between opener and defender
            return dist(opener, defender) >= 1
        return True

    opener = _guess_opener(ip, oop, ctx)
    defender = oop if opener == ip else ip  # whichever isn't the opener

    def to_ip_oop(new_opener: str, new_defender: str) -> tuple[str, str]:
        # Remap back to original IP/OOP roles
        if opener == ip:
            # original IP was the opener → keep IP as opener
            return new_opener, new_defender
        else:
            # original IP was the defender → keep IP as defender
            return new_defender, new_opener

    # --- 1) Exact pair (no substitution) ---
    if ok_for_ctx(opener, defender):
        yield (ip, oop, "exact_pair", False)

    if not allow_pair_subs:
        return

    yielded: set[tuple[str, str]] = set()

    # --- 2) Substitute opener (defender fixed) ---
    for op_sub in OPEN_SUBS.get(opener, []):
        if not ok_for_ctx(op_sub, defender):
            continue
        cand_ip, cand_oop = to_ip_oop(op_sub, defender)
        key = (cand_ip, cand_oop)
        if key not in yielded:
            yielded.add(key)
            yield (cand_ip, cand_oop, "sub_opener", True)

    # --- 3) Substitute defender (opener fixed) ---
    for df_sub in DEF_SUBS.get(defender, []):
        if not ok_for_ctx(opener, df_sub):
            continue
        cand_ip, cand_oop = to_ip_oop(opener, df_sub)
        key = (cand_ip, cand_oop)
        if key not in yielded:
            yielded.add(key)
            yield (cand_ip, cand_oop, "sub_defender", True)

    # --- 4) Cross-sub: try ALL opener×defender rungs ---
    for op_sub in OPEN_SUBS.get(opener, []):
        for df_sub in DEF_SUBS.get(defender, []):
            if not ok_for_ctx(op_sub, df_sub):
                continue
            cand_ip, cand_oop = to_ip_oop(op_sub, df_sub)
            key = (cand_ip, cand_oop)
            if key not in yielded:
                yielded.add(key)
                yield (cand_ip, cand_oop, "sub_both", True)

    # --- 5) SRP late-default (last resort) ---
    if ctx in {"SRP", "VS_OPEN", "OPEN", "BLIND_VS_STEAL"}:
        if opener not in {"BTN", "CO"} and defender in BLINDS:
            op_sub = "BTN"
            df_sub = defender
            if ok_for_ctx(op_sub, df_sub):
                cand_ip, cand_oop = to_ip_oop(op_sub, df_sub)
                key = (cand_ip, cand_oop)
                if key not in yielded:
                    yielded.add(key)
                    yield (cand_ip, cand_oop, "srp_late_default", True)