# ---- hand_id -> single canonical combo (eval7.Card, eval7.Card) ----
# deps: hand_id_to_label(hand_id) -> str in {"AA","AKs","AKo",...}


RANKS = "AKQJT98765432"  # 13 ranks in order
HAND_LABELS = []

# --- build canonical 169 labels ---
for i, r1 in enumerate(RANKS):
    for j, r2 in enumerate(RANKS):
        if i < j:
            HAND_LABELS.append(r1 + r2 + "s")   # suited
            HAND_LABELS.append(r1 + r2 + "o")   # offsuit
        elif i == j:
            HAND_LABELS.append(r1 + r2)         # pair
# sort them in standard order: pairs first, then suited, then offsuit
# This is the convention we used for HAND_COUNT=169
def _canonical_order(label: str) -> tuple[int, int, str]:
    rmap = {r: k for k, r in enumerate(RANKS)}
    if len(label) == 2:  # pair
        return (0, rmap[label[0]], "")
    r1, r2, suf = label[0], label[1], label[2]
    if suf == "s":
        return (1, rmap[r1], rmap[r2])
    elif suf == "o":
        return (2, rmap[r1], rmap[r2])
    else:
        raise ValueError(f"Bad label: {label}")

HAND_LABELS = sorted(HAND_LABELS, key=_canonical_order)

# ---- helpers ----
def hand_id_to_label(hand_id: int) -> str:
    """Map 0..168 → canonical label (e.g. 0:'AA', 1:'AKs', ..., 168:'32o')."""
    if hand_id < 0 or hand_id >= len(HAND_LABELS):
        raise ValueError(f"hand_id {hand_id} out of range (0..{len(HAND_LABELS)-1})")
    return HAND_LABELS[hand_id]

def label_to_hand_id(label: str) -> int:
    """Reverse lookup: 'AKs' → int ID."""
    if label not in HAND_LABELS:
        raise ValueError(f"Unknown label {label}")
    return HAND_LABELS.index(label)

HAND_COUNT = len(HAND_LABELS)  # should be 169

try:
    import eval7
except ImportError as e:
    raise ImportError(
        "eval7 is required for equity computations. pip install eval7"
    ) from e

def hand_id_to_combo(hand_id: int) -> tuple[eval7.Card, eval7.Card]:
    """
    Deterministically pick ONE representative combo for a 169-class hand_id:
      - Pairs:      Ah Ad
      - Suited:     (first rank)h (second rank)h  e.g. AKs -> Ah Kh
      - Offsuit:    (first rank)s (second rank)d e.g. AKo -> As Kd
    """
    lab = hand_id_to_label(hand_id)  # e.g. "AKs"
    if len(lab) == 2:  # pair
        r = lab[0]
        return eval7.Card(r + 'h'), eval7.Card(r + 'd')
    r1, r2, suf = lab[0], lab[1], lab[2]
    if suf == 's':
        return eval7.Card(r1 + 'h'), eval7.Card(r2 + 'h')
    elif suf == 'o':
        return eval7.Card(r1 + 's'), eval7.Card(r2 + 'd')
    else:
        raise ValueError(f"Bad normalized hand label: {lab}")


def hand_id_to_combos(hand_id: int) -> list[tuple[eval7.Card, eval7.Card]]:
    lab = hand_id_to_label(hand_id)
    ranks = 'AKQJT98765432'
    suits = 'cdhs'
    out: list[tuple[eval7.Card, eval7.Card]] = []

    if len(lab) == 2:  # pair
        r = lab[0]
        for i in range(4):
            for j in range(i+1, 4):
                out.append((eval7.Card(r + suits[i]), eval7.Card(r + suits[j])))
        return out

    r1, r2, suf = lab[0], lab[1], lab[2]
    if suf == 's':
        for s in suits:
            out.append((eval7.Card(r1 + s), eval7.Card(r2 + s)))
    elif suf == 'o':
        for s1 in suits:
            for s2 in suits:
                if s1 == s2:
                    continue
                out.append((eval7.Card(r1 + s1), eval7.Card(r2 + s2)))
    else:
        raise ValueError(f"Bad normalized hand label: {lab}")
    return out