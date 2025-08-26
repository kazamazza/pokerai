from ml.config.types_hands import HAND_TO_ID, RANK_TO_I


def summarize_169(y169: list[float]) -> dict:
    """
    Example summarization of a 169-dim range vector.
    You can adjust buckets as you like.
    """
    # Suppose you already have a HAND_GROUPS dict mapping { "AA": "premium_pair", "AKs": "broadway", ... }
    summaries = {
        "pairs": sum(y169[i] for h,i in HAND_TO_ID.items() if h[0]==h[1]),
        "suited": sum(y169[i] for h,i in HAND_TO_ID.items() if h.endswith("s")),
        "offsuit": sum(y169[i] for h,i in HAND_TO_ID.items() if h.endswith("o")),
    }
    return summaries

def hand_to_169(combo: str) -> str:
    """
    Convert concrete 2-card combo like 'AsKh' to canonical 169 code.
    """
    if not combo or len(combo) != 4:
        return ""
    r1, s1, r2, s2 = combo[0], combo[1], combo[2], combo[3]
    if RANK_TO_I[r1] > RANK_TO_I[r2]:
        hi, lo, sh, sl = r1, r2, s1, s2
    else:
        hi, lo, sh, sl = r2, r1, s2, s1
    if hi == lo:
        return hi+lo
    return f"{hi}{lo}{'s' if sh == sl else 'o'}"


def hand_to_id(combo: str) -> int:
    code = hand_to_169(combo)
    return HAND_TO_ID.get(code, -1)


def tilt_toward_raise(actions, probs, amount=0.15):
    if "CALL" not in actions or "RAISE" not in actions:
        return actions, probs
    i_call = actions.index("CALL")
    i_raise = actions.index("RAISE")
    delta = min(probs[i_call], amount)
    probs[i_call] -= delta
    probs[i_raise] += delta
    return actions, probs


def renormalize_and_mask(actions, probs, mask=set()):
    new_actions, new_probs = [], []
    for a, p in zip(actions, probs):
        if a not in mask:
            new_actions.append(a)
            new_probs.append(p)
    s = sum(new_probs) or 1.0
    new_probs = [p/s for p in new_probs]
    return new_actions, new_probs