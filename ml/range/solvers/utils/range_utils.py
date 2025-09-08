from typing import Dict
import numpy as np
import re, json
from pathlib import Path
from ml.config.types_hands import RANKS, SUITS


def hand_to_index(code: str) -> int:
    # "AA", "AKs", "AKo"
    if len(code) == 2:  # pair (AA..22)
        r = RANKS.index(code[0])
        return r * 13 + r
    if len(code) == 3:  # e.g., AKs / AKo
        hi, lo, s = code[0], code[1], code[2]
        i = RANKS.index(hi); j = RANKS.index(lo)
        if s == "s":
            return i * 13 + j   # upper triangle (i < j)
        elif s == "o":
            return j * 13 + i   # lower triangle (j < i)
    raise ValueError(f"bad abstract hand code: {code}")

def to_compact_index(cards: str) -> int:
    # AhKh -> AKs, AdKc -> AKo, etc.
    if len(cards) != 4:
        raise ValueError(f"Bad card spec: {cards}")
    r1, s1, r2, s2 = cards[0], cards[1], cards[2], cards[3]
    if r1 == r2:
        return hand_to_index(r1 + r2)
    suited = (s1 == s2)
    hi, lo = (r1, r2) if RANKS.index(r1) < RANKS.index(r2) else (r2, r1)
    return hand_to_index(hi + lo + ("s" if suited else "o"))
def zeros_169():
    return np.zeros(169, dtype=np.float32)

def parse_range_text_to_grid(path: Path) -> np.ndarray:
    txt = Path(path).read_text(encoding="utf-8").strip()

    # --- JSON ---
    try:
        obj = json.loads(txt)
        if isinstance(obj, list) and len(obj) == 169:
            return np.array(obj, dtype=np.float32)
        if isinstance(obj, dict):
            if "range" in obj and len(obj["range"]) == 169:
                return np.array(obj["range"], dtype=np.float32)
            # dict of CARD:VALUE, e.g. {"AA":1.0, "AKs":0.25, ...}
            vals = [0.0] * 169
            ok = False
            for k, v in obj.items():
                try:
                    vals[hand_to_index(k)] = float(v)
                    ok = True
                except Exception:
                    pass
            if ok:
                return np.array(vals, dtype=np.float32)
    except Exception:
        pass

    # --- CARD:VALUE plain text ---
    if ":" in txt and any(tag in txt for tag in ("AA", "AKs", "KQo", "72o")):
        vals = [0.0] * 169
        for tok in re.split(r"[,\s]+", txt):
            if not tok or ":" not in tok:
                continue
            hand, val = tok.split(":")
            vals[hand_to_index(hand)] = float(val)
        return np.array(vals, dtype=np.float32)

    # --- ABS blocks + bare combos ---
    if "[" in txt and "]" in txt and "/" in txt:
        # Collect combo-level weights
        combo_weights: Dict[str, float] = {}

        # 1) bracketed groups: [value] combos... [/value]
        for m in re.finditer(r"\[(.*?)\](.*?)\[/\1\]", txt, flags=re.S):
            raw = m.group(1).strip()
            try:
                w = float(raw)
                if w > 1.0:
                    w = w / 100.0
            except Exception:
                continue
            hands_blob = m.group(2)
            for h in (t for t in re.split(r"[,\s]+", hands_blob) if t):
                # Expect 4-char combo like Th8h, AdKc, TdTh
                if len(h) == 4:
                    combo_weights[h] = w

        # 2) bare combos outside blocks default to 1.0
        txt_no_groups = re.sub(r"\[(.*?)\](.*?)\[/\1\]", " ", txt, flags=re.S)
        for h in (t for t in re.split(r"[,\s]+", txt_no_groups) if t):
            if len(h) == 4 and h[0] in RANKS and h[2] in RANKS:
                combo_weights.setdefault(h, 1.0)

        # Average per abstract key (TT, AKs, AKo, T8s, …)
        sums: Dict[str, float] = {}
        cnts: Dict[str, int] = {}
        for combo, w in combo_weights.items():
            try:
                key = compact_key_for_combo(combo)  # e.g. Th8h → T8s, AdKc → AKo, TdTh → TT
                sums[key] = sums.get(key, 0.0) + float(w)
                cnts[key] = cnts.get(key, 0) + 1
            except Exception:
                continue

        vec = np.zeros(169, dtype=np.float32)
        for key, s in sums.items():
            try:
                idx = hand_to_index(key)
                vec[idx] = s / float(cnts[key])
            except Exception:
                continue
        return vec

    # --- Flat list of 169 numbers (csv/whitespace; supports %) ---
    toks = re.split(r"[,\s]+", txt)
    nums: list[float] = []
    for t in toks:
        if not t:
            continue
        if t.endswith("%"):
            try:
                nums.append(float(t[:-1]) / 100.0)
            except Exception:
                pass
        else:
            try:
                nums.append(float(t))
            except Exception:
                pass
    if len(nums) == 169:
        return np.array(nums, dtype=np.float32)

    raise ValueError(f"Unrecognized range format in {path}")

def _hand_to_index_compact(cards: str) -> int:
    """
    Collapse e.g. AhKh → AKs, AdKc → AKo, etc.
    """
    if len(cards) != 4:
        raise ValueError(f"Bad card spec: {cards}")
    r1, s1, r2, s2 = cards[0], cards[1], cards[2], cards[3]
    if r1 == r2:
        return hand_to_index(r1 + r2)
    suited = (s1 == s2)
    high, low = (r1, r2) if RANKS.index(r1) < RANKS.index(r2) else (r2, r1)
    code = high + low + ("s" if suited else "o")
    return hand_to_index(code)


IDX = {r:i for i,r in enumerate(RANKS)}

def _class_from_combo(cc: str) -> str:
    # cc like 'AhKh','TdTh','AdKc'
    r1,s1,r2,s2 = cc[0], cc[1], cc[2], cc[3]
    if r1 not in RANKS or r2 not in RANKS or s1 not in SUITS or s2 not in SUITS:
        raise ValueError("bad combo")
    if r1 == r2:
        return r1 + r2  # pair e.g. 'TT'
    hi, lo = (r1, r2) if IDX[r1] < IDX[r2] else (r2, r1)
    suited = (s1 == s2)
    return hi + lo + ("s" if suited else "o")  # e.g. 'AKs' / 'AKo'


def _combos_per_class(code: str) -> int:
    if len(code) == 2:  return 6   # pairs
    if code.endswith("s"): return 4
    if code.endswith("o"): return 12
    raise ValueError("bad class code")

def compact_key_for_combo(combo: str) -> str:
    """
    Map a raw 4-char combo (e.g. 'Th8h','AdKc','TdTh')
    into its abstract hand key ('T8s','AKo','TT').
    """
    if len(combo) != 4:
        raise ValueError(f"Bad combo string: {combo}")
    r1, s1, r2, s2 = combo[0], combo[1], combo[2], combo[3]

    # Pair
    if r1 == r2:
        return r1 + r2

    # Normalize rank order: high rank first
    if RANKS.index(r1) > RANKS.index(r2):
        r1, s1, r2, s2 = r2, s2, r1, s1

    if s1 == s2:
        return r1 + r2 + "s"
    else:
        return r1 + r2 + "o"

_COMBO_RE = re.compile(r"\b([AKQJT98765432][cdhs][AKQJT98765432][cdhs])\b")

def abs_text_to_vec169(abs_text: str) -> np.ndarray:
    # 1) setup
    totals = {}       # "AA"/"AKs"/"AKo" -> total combos (6/4/12)
    combo_seen = set()  # to avoid double-counting same 4-card combo
    sums = {}         # "AA"/"AKs"/"AKo" -> sum of combo weights

    def _ensure_key(k):
        if k not in totals:
            if len(k) == 2:      # pair
                totals[k] = 6
            elif k.endswith("s"):
                totals[k] = 4
            else:
                totals[k] = 12
            sums[k] = 0.0

    def _add_combo(combo4, weight):
        # combo4 like "Th8h", "AdKc", "TdTh"
        key = compact_key_for_combo(combo4)  # "T8s"/"AKo"/"TT"
        if (key, combo4) in combo_seen:
            return
        combo_seen.add((key, combo4))
        _ensure_key(key)
        sums[key] += float(weight)

    # 2) bracketed groups: [p] ... [/p]
    import re
    for m in re.finditer(r"\[(.*?)\](.*?)\[/\1\]", abs_text, flags=re.S):
        raw = m.group(1).strip()
        try:
            p = float(raw)
            if p > 1.0:
                p = p / 100.0
        except:
            continue
        blob = m.group(2)
        for tok in re.split(r"[,\s]+", blob):
            if len(tok) == 4 and tok[0] in RANKS and tok[2] in RANKS:
                _add_combo(tok, p)

    # 3) bare combos outside groups => 100%
    txt_no_groups = re.sub(r"\[(.*?)\](.*?)\[/\1\]", " ", abs_text, flags=re.S)
    for tok in re.split(r"[,\s]+", txt_no_groups):
        if len(tok) == 4 and tok[0] in RANKS and tok[2] in RANKS:
            _add_combo(tok, 1.0)

    # 4) build vec169 by averaging over combos per abstract hand
    vec = np.zeros(169, dtype=np.float32)
    for key, s in sums.items():
        idx = hand_to_index(key)
        denom = float(totals[key])
        vec[idx] = max(0.0, min(1.0, s / denom))
    return vec

def _tag_for_cell(i: int, j: int) -> str:
    """Build Monker hand tag for 13x13 cell (i=row rank, j=col rank) with RANKS='AKQJT...2'."""
    ri, rj = RANKS[i], RANKS[j]
    if i == j:
        return ri + rj                 # pair: AA, KK, ...
    if i < j:
        return ri + rj + "s"           # upper triangle: suited
    else:
        return rj + ri + "o"           # lower triangle: offsuit (hi+lo+'o')

def vec169_to_monker_string(arr, *, drop_zeros=True, precision=6) -> str:
    """
    Convert a 169 vector (or 13x13) into Monker CSV "CARD:VALUE,..."
    using the SAME 13x13 indexing convention as packing:
      index = i*13 + j with i,j over RANKS='AKQJT98765432'
      pairs on diagonal, suited upper, offsuit lower.
    """
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2 and a.shape == (13, 13):
        a = a.reshape(169)
    elif a.ndim == 1 and a.size == 169:
        pass
    else:
        raise ValueError(f"vec169_to_monker_string: expected 169 or 13x13, got {a.shape}")

    a = np.clip(a, 0.0, 1.0)

    parts = []
    for i in range(13):
        for j in range(13):
            v = float(a[i*13 + j])
            if drop_zeros and v <= 0.0:
                continue
            parts.append(f"{_tag_for_cell(i,j)}:{v:.{precision}f}")

    # If everything was zero and we dropped zeros, emit explicit zeros for a few top hands
    if not parts:
        parts = [f"{_tag_for_cell(0,0)}:0.000000"]  # AA:0.0 at minimum

    return ",".join(parts)


def _combo_to_abstract_index(cards: str) -> tuple[int, int]:
    # e.g. "Th8h","AdKc","TdTh" -> (abstract_index, combos_in_cell)
    if len(cards) != 4:
        raise ValueError(f"bad combo: {cards}")
    r1, s1, r2, s2 = cards[0], cards[1], cards[2], cards[3]
    i1, i2 = RANKS.index(r1), RANKS.index(r2)
    if r1 == r2:
        idx = hand_to_index(r1 + r2)
        return idx, 6   # 6 pair combos
    suited = (s1 == s2)
    hi, lo = (r1, r2) if i1 < i2 else (r2, r1)
    idx = hand_to_index(hi + lo + ("s" if suited else "o"))
    return idx, (4 if suited else 12)

def parse_abs_text_to_vec169(path: Path) -> np.ndarray:
    """
    Parse SPH ABS text:
      - [xx.xx] ... [/xx.xx] blocks ⇒ those combos get xx.xx% (xx/100)
      - bare combos outside blocks ⇒ those combos get 100%
    Aggregate to abstract 169 by averaging across combos per cell.
    """
    import re, json
    txt = path.read_text(encoding="utf-8")

    # Fast path for already-structured input
    try:
        obj = json.loads(txt)
        if isinstance(obj, list) and len(obj) == 169:
            return np.array(obj, dtype=np.float32)
        if isinstance(obj, dict):
            for k in ("range","grid","matrix","weights","data","ip","oop"):
                if k in obj and np.size(obj[k]) == 169:
                    arr = np.array(obj[k], dtype=np.float32).reshape(169)
                    return np.clip(arr, 0.0, 1.0)
    except Exception:
        pass

    # 1) Collect bracketed groups
    taken = [0.0] * 169
    count = [0]   * 169

    def add_combo(cards: str, v: float):
        try:
            idx, denom = _combo_to_abstract_index(cards)
            # accumulate sum over combos; we'll divide by denom at the end
            taken[idx] += v
            count[idx] += 1
        except Exception:
            pass

    # mark all spans to strip later
    spans = []
    for m in re.finditer(r"\[(.*?)\](.*?)\[/\1\]", txt, flags=re.S):
        raw = m.group(1).strip()
        try:
            v = float(raw)
            if v > 1.0: v = v / 100.0
        except Exception:
            continue
        hands_blob = m.group(2)
        # split by comma/space
        for tok in re.split(r"[,\s]+", hands_blob):
            tok = tok.strip()
            if len(tok) == 4 and tok[0] in RANKS and tok[2] in RANKS:
                add_combo(tok, v)
        spans.append((m.start(), m.end()))

    # 2) Bare combos outside groups ⇒ 100%
    # Remove bracketed parts, then scan remaining for 4-char combos
    tmp = []
    last = 0
    for a,b in spans:
        tmp.append(txt[last:a])
        last = b
    tmp.append(txt[last:])
    outside = "".join(tmp)

    for tok in re.split(r"[,\s]+", outside):
        tok = tok.strip()
        if len(tok) == 4 and tok[0] in RANKS and tok[2] in RANKS:
            add_combo(tok, 1.0)

    # 3) Finalize: average by combos per abstract cell
    out = np.zeros(169, dtype=np.float32)
    for idx in range(169):
        if count[idx] > 0:
            # average across the combos seen for that abstract cell
            # NOTE: if some combos for that cell are missing, we treat them as 0 (consistent with ABS export)
            # You can choose to normalize by full denom instead; this matches ABS semantics better.
            denom = count[idx]  # combos we actually saw
            out[idx] = float(taken[idx]) / float(denom)
    return np.clip(out, 0.0, 1.0)