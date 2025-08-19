from ml.utils.constants import R2I, SUITS


def _is_card(s: str) -> bool:
    return len(s) == 2 and s[0] in RANKS and s[1] in SUITS

def _expand_xy_plus(top: str, low: str, tail: str) -> list[str]:
    itop, ilow = R2I[top], R2I[low]
    out = []
    for i in range(ilow, itop):
        k = RANKS[i]
        if R2I[top] > R2I[k]:
            out.append(f"{top}{k}{tail}")
    return out or [f"{top}{low}{tail}"]

def console_safe_range(range_str: str, *, expand_all_plus: bool = False) -> str:
    if not range_str:
        raise ValueError("range is empty before sanitize")

    toks_in = [t.strip() for t in range_str.split(",") if t.strip()]
    out: list[str] = []
    seen: set[str] = set()

    # NEW: collect any concrete combos we encounter
    concrete: list[str] = []

    for raw in toks_in:
        core, weight = (raw.split(":", 1) + [""])[:2]
        weight = (":" + weight.strip()) if weight else ""
        core = core.strip()

        # Already 'XY ZW'? → collect as 4-char for compression later
        if " " in core:
            parts = [p.strip() for p in core.split() if p.strip()]
            if len(parts) >= 2 and _is_card(parts[0]) and _is_card(parts[1]) and parts[0] != parts[1]:
                concrete.append(parts[0] + parts[1])   # e.g. "AcAd"
            continue

        # 4-char concrete 'AcAd'? → collect for compression later
        if len(core) == 4 and _is_card(core[:2]) and _is_card(core[2:]) and core[:2] != core[2:]:
            concrete.append(core)                       # e.g. "AcAd"
            continue

        # ---- shorthand handling as before ----
        coreU = core.upper().replace(" ", "")

        # Pairs: '22' or '22+'
        if len(coreU) in (2, 3) and coreU[0] == coreU[1] and coreU[0] in RANKS:
            if coreU.endswith("+") and expand_all_plus:
                start = R2I[coreU[0]]
                for i in range(start, len(RANKS)):
                    p = RANKS[i] * 2 + weight
                    if p not in seen:
                        seen.add(p); out.append(p)
            else:
                tok = coreU + weight
                if tok not in seen:
                    seen.add(tok); out.append(tok)
            continue

        # Two-rank shorthand: AKs / AKo / QTo+
        if len(coreU) >= 3 and coreU[0] in RANKS and coreU[1] in RANKS and R2I[coreU[0]] > R2I[coreU[1]]:
            top, low, tail = coreU[0], coreU[1], coreU[2:]
            if tail in ("S", "O"):
                tok = f"{top}{low}{tail.lower()}{weight}"
                if tok not in seen:
                    seen.add(tok); out.append(tok)
                continue
            if tail in ("S+", "O+"):
                sflag = tail[0].lower()
                if top == "A" and not expand_all_plus:
                    tok = f"{top}{low}{sflag}+{weight}"
                    if tok not in seen:
                        seen.add(tok); out.append(tok)
                else:
                    for h in _expand_xy_plus(top, low, sflag):
                        tok = h + weight
                        if tok not in seen:
                            seen.add(tok); out.append(tok)
                continue

        # drop unknown
        continue

    # NEW: if we collected concrete, compress to shorthand and merge
    if concrete:
        # uses your existing compressor (pairs need 6, suited 4, offsuit 12 to emit)
        shorthand = _combos_to_shorthand(concrete)   # returns like "66,AKs,AKo,..." (canonicalized)
        if shorthand:
            out.extend([t for t in shorthand.split(",") if t and t not in seen])

    s = ",".join(out)
    if not s:
        raise ValueError(f"range sanitized to empty from: {range_str!r}")
    return s

def assert_console_range_ok(s: str) -> None:
    """Validate final console range string; allow shorthand or 'XY ZW' combos."""
    if not s:
        raise ValueError("range string is empty")
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            raise ValueError("empty token in range")

        # weight suffix
        hand, *_ = tok.split(":", 1)

        # Concrete 'XY ZW'
        if " " in hand:
            parts = [p for p in hand.split() if p]
            if len(parts) != 2 or not _is_card(parts[0]) or not _is_card(parts[1]) or parts[0] == parts[1]:
                raise ValueError(f"bad concrete token '{tok}'")
            continue

        # Shorthand: '22', '22+', 'AKs', 'AKo', 'QTo+' etc.
        handU = hand.upper()
        # pairs
        if (len(handU) == 2 and handU[0] == handU[1] and handU[0] in RANKS) or \
           (len(handU) == 3 and handU[0] == handU[1] and handU[2] == "+" and handU[0] in RANKS):
            continue
        # two-rank with suitedness (+ optional '+')
        if len(handU) in (3, 4) and handU[0] in RANKS and handU[1] in RANKS and R2I[handU[0]] > R2I[handU[1]]:
            tail = handU[2:]
            if tail in ("S", "O", "S+", "O+"):
                continue
        raise ValueError(f"bad shorthand token '{tok}'")


RANKS = "23456789TJQKA"


def _combos_to_shorthand(combos: list[str]) -> str:
    """
    Compress concrete combos (e.g., 'Ac Ad', 'As Ah', 'Kd Qd', 'Kh Qh', 'Kd Qh'...)
    into solver shorthand tokens like 'AA', 'AKs', 'AKo', possibly mixed.
    Strategy:
      - Pairs: if all 6 combos present -> 'RR'
      - Suited XY: if all 4 combos present -> 'XYs'
      - Offsuit XY: if all 12 combos present -> 'XYo'
      - Otherwise: drop partials (console doesn’t like raw concrete lists).
    This matches how your villain maps are generated (mostly full sets).
    """
    # normalize 'AcAd' or 'Ac Ad' into ('A','c','A','d')
    def split(h: str) -> tuple[str,str,str,str]:
        h = h.replace(" ", "")
        return h[0], h[1], h[2], h[3]

    # buckets
    pairs: dict[str, set[str]] = {}          # '6' -> set of 'cd','ch','cs','dh','ds','hs' (6)
    suited: dict[tuple[str,str], set[str]] = {}   # ('A','K') -> set of suits 'c','d','h','s' where both share suit (4)
    offs: dict[tuple[str,str], set[tuple[str,str]]] = {}  # ('A','K') -> set of (s1,s2) with s1!=s2 (12)

    for c in combos:
        r1,s1,r2,s2 = split(c)
        # canonical rank order (top > low)
        if R2I[r1] < R2I[r2]:
            r1,s1, r2,s2 = r2,s2, r1,s1

        if r1 == r2:
            key = r1
            pair = "".join(sorted([s1+s2, s2+s1]))  # just mark presence; we only care about count
            pairs.setdefault(key, set()).add(s1+s2 if s1 < s2 else s2+s1)
        else:
            key = (r1, r2)
            if s1 == s2:
                suited.setdefault(key, set()).add(s1)
            else:
                offs.setdefault(key, set()).add((s1, s2))

    out_tokens: list[str] = []

    # pairs: need all 6
    for r, seen in pairs.items():
        if len(seen) >= 6:
            out_tokens.append(f"{r}{r}")

    # suited: need all 4
    for (hi, lo), seen in suited.items():
        if len(seen) >= 4:
            out_tokens.append(f"{hi}{lo}s")

    # offsuit: need all 12
    for (hi, lo), seen in offs.items():
        if len(seen) >= 12:
            out_tokens.append(f"{hi}{lo}o")

    # collapse and canonicalize with your existing function
    return canonicalize_range_string_cached(",".join(sorted(out_tokens)))

def _sanitize_shorthand_only(range_str: str) -> str:
    """
    Accepts ONLY solver shorthand (pairs, pairs+, XYs, XYs+, XYo, XYo+).
    Normalizes to upper-case ranks with LOWER-case suitedness and optional '+'.
    Drops spaces, rejects malformed tokens, orders ranks hi>lo.
    """
    if not range_str:
        raise ValueError("empty range")

    out = []
    seen = set()
    for raw in (t.strip() for t in range_str.split(",") if t.strip()):
        core, weight = (raw.split(":", 1) + [""])[:2]
        weight = (":" + weight.strip()) if weight else ""
        tok = core.replace(" ", "").upper()

        # pairs '66' or '66+'
        if len(tok) in (2, 3) and tok[0] == tok[1] and tok[0] in RANKS:
            if len(tok) == 3 and tok[2] != "+":
                continue
            final = tok + weight
            if final not in seen:
                seen.add(final); out.append(final)
            continue

        # two ranks with tail (S/O) and optional '+'
        if len(tok) >= 3 and tok[0] in RANKS and tok[1] in RANKS:
            hi, lo = tok[0], tok[1]
            if R2I[hi] <= R2I[lo]:
                # enforce canonical order hi>lo
                hi, lo = lo, hi
                tok = hi + lo + tok[2:]

            tail = tok[2:]
            if tail not in ("S","O","S+","O+"):
                continue
            suited = tail[0].lower()              # 's' or 'o'
            plus   = "+" if tail.endswith("+") else ""
            final = f"{hi}{lo}{suited}{plus}{weight}"
            if final not in seen:
                seen.add(final); out.append(final)
            continue

        # otherwise ignore
        continue

    if not out:
        raise ValueError(f"range sanitized to empty from: {range_str!r}")
    return ",".join(out)