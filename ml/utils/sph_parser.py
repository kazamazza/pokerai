import re
from collections import defaultdict

# Ranks high->low for output formatting
RANKS = "AKQJT98765432"

def hand_class_from_combo(combo: str) -> str:
    # combo like "Jh7h", "AdKh", "TcTd"
    r1, s1, r2, s2 = combo[0], combo[1], combo[2], combo[3]
    if r1 == r2:
        return r1 + r2               # e.g., "JJ"
    suited = (s1 == s2)
    if suited:
        return r1 + r2 + "s"         # e.g., "AKs"
    else:
        return r1 + r2 + "o"         # e.g., "AKo"

def normalize_class(name: str) -> str:
    # Ensure canonical order: higher rank first
    if len(name) == 2:  # pair
        return name
    r1, r2, t = name[0], name[1], name[2]
    if RANKS.index(r1) > RANKS.index(r2):
        return r1 + r2 + t
    else:
        return r2 + r1 + t

def parse_spfh_copy(text: str) -> dict:
    # Pull out weighted groups: [weight]combo,combo[/weight]
    weights = []
    used = [False] * len(text)

    class_sum = defaultdict(float)
    class_cnt  = defaultdict(int)

    # 1) Handle [w]...[/w] groups
    for m in re.finditer(r'\[(\d+(?:\.\d+)?)\](.*?)\[/\1\]', text, flags=re.DOTALL):
        w = float(m.group(1))
        w = w/100.0 if w > 1.0 else w
        block = m.group(2)
        for tok in re.split(r'[,\s]+', block.strip()):
            if len(tok) == 4 and tok[0] in RANKS and tok[2] in RANKS:
                key = normalize_class(hand_class_from_combo(tok))
                class_sum[key] += w
                class_cnt[key]  += 1
        # mark range to skip in unweighted pass
        start, end = m.span()
        for i in range(start, end):
            used[i] = True

    # 2) Remaining plain combos (implicitly 100%)
    plain = ''.join(ch if not used[i] else ' ' for i, ch in enumerate(text))
    for tok in re.split(r'[,\s]+', plain.strip()):
        if len(tok) == 4 and tok[0] in RANKS and tok[2] in RANKS:
            key = normalize_class(hand_class_from_combo(tok))
            class_sum[key] += 1.0
            class_cnt[key]  += 1

    # 3) Average per hand class (pairs have 6 combos, suited 4, offsuit 12—but we average over actual listed combos)
    class_avg = {}
    # Build full 169-key order
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i == j:
                key = r1+r2
            elif i < j:
                key = r1+r2+"s"
            else:
                key = r1+r2+"o"
            if class_cnt.get(key, 0) > 0:
                class_avg[key] = round(class_sum[key] / class_cnt[key], 6)
            else:
                class_avg[key] = 0.0
    return class_avg

if __name__ == "__main__":
    with open("raw.txt","r",encoding="utf-8") as f:
        raw = f.read()
    weights = parse_spfh_copy(raw)

    # Output as a single CSV-style line like your example
    ordered = []
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i == j:
                k = r1+r2
            elif i < j:
                k = r1+r2+"s"
            else:
                k = r1+r2+"o"
            ordered.append(f"{k}:{weights[k]}")
    line = ",".join(ordered)
    with open("range_169.txt","w",encoding="utf-8") as f:
        f.write(line+"\n")
    print("Saved to range_169.txt")