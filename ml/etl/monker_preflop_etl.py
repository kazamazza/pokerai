from __future__ import annotations
import re, json, gzip, hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from ml.types import RANKS

# ------------------------------
# Canonical orders / seating
# ------------------------------

POSITION_ORDER = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]  # normalize MP->HJ
POS_SET = set(POSITION_ORDER)

def canonical_169_order() -> List[str]:
    order = []
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i == j:   order.append(f"{r1}{r2}")
            elif i < j: order.append(f"{r1}{r2}s")
            else:       order.append(f"{r2}{r1}o")
    return order

ORDER169 = canonical_169_order()
IDX169: Dict[str,int] = {h:i for i,h in enumerate(ORDER169)}

# ------------------------------
# Action normalization
# ------------------------------
# Split tokens on underscores OR hyphens
TOKEN_SPLIT = re.compile(r"[_\-]+")

# Rich alias table (uppercased before lookup)
ACT_CANON = {
    # all-in
    "AI":"AI", "ALLIN":"AI", "ALL_IN":"AI", "SHOVE":"AI", "PUSH":"AI",

    # open / raise / limp
    "OPEN":"OPEN", "RAISE":"RAISE", "LIMP":"LIMP", "BET":"RAISE", "MINRAISE":"RAISE",
    "RFI":"OPEN", "OPENRAISE":"OPEN", "OPEN_RAISE":"OPEN", "OR":"OPEN",
    "ISO":"RAISE", "ISORAISE":"RAISE", "ISOLATE":"RAISE",

    # calls
    "CALL":"CALL", "COLD":"CALL", "COLD_CALL":"CALL", "OVERCALL":"CALL",
    "FLAT":"CALL", "FLATCALL":"CALL", "FLAT_CALL":"CALL", "CC":"CALL",

    # 3/4-bets
    "3BET":"3BET", "3_BET":"3BET", "THREEBET":"3BET", "3X":"3BET",
    "4BET":"4BET", "4_BET":"4BET", "FOURBET":"4BET",

    # folds (ignored in participants)
    "FOLD":"FOLD"
}

ACTIVE_ACTS = {"OPEN","RAISE","LIMP","AI","CALL","3BET","4BET"}
AGGR_ACTS   = {"OPEN","RAISE","LIMP","AI","3BET","4BET"}
RESP_ACTS   = {"CALL","3BET","4BET","AI"}

# ------------------------------
# Helpers
# ------------------------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def parse_filename_actions(name: str) -> List[Dict[str,str]]:
    """
    Tolerant parser for patterns like:
      UTG_AI_HJ_Call ...  or  UTG-AI-HJ-Call ...  mixed case / aliases allowed
    -> [{"pos":"UTG","act":"AI"}, ...]
    """
    stem = Path(name).stem
    parts = [p for p in TOKEN_SPLIT.split(stem) if p]
    out: List[Dict[str,str]] = []
    i = 0
    while i < len(parts):
        pos = parts[i].upper()
        if pos == "MP": pos = "HJ"  # normalize MP->HJ for 6-max
        if pos not in POS_SET:
            i += 1
            continue
        if i + 1 >= len(parts):
            break
        act_raw = parts[i+1].upper()
        act = ACT_CANON.get(act_raw, ACT_CANON.get(act_raw.replace(" ", "_"), act_raw))
        out.append({"pos": pos, "act": act})
        i += 2
    return out

def participants(actions: List[Dict[str,str]]) -> List[str]:
    """Distinct seats with non-fold preflop actions (order preserved)."""
    seen = {}
    for a in actions:
        if a["act"] in ACTIVE_ACTS:
            seen.setdefault(a["pos"], True)
    return list(seen.keys())

def first_aggressor(actions: List[Dict[str,str]]) -> Dict[str,str] | None:
    for a in actions:
        if a["act"] in AGGR_ACTS:
            return a
    return None

def responders_after(actions: List[Dict[str,str]], opener_pos: str) -> List[str]:
    """Unique responders (not opener) who took RESP_ACTS after opener."""
    seen = {}
    for a in actions:
        if a["pos"] != opener_pos and a["act"] in RESP_ACTS:
            seen.setdefault(a["pos"], True)
    return list(seen.keys())

def classify_hu_multiway(actions: List[Dict[str,str]]) -> Tuple[bool,str,str|None,str|None]:
    """
    Returns (is_multiway, reason, opener_pos, first_responder_pos)
    - no aggressor -> MW ('no_aggressor')  [we can't place the spot]
    - 0 responders -> HU ('opener_only')
    - 1 responder  -> HU ('single_responder')
    - 2+ responders-> MW ('multiple_responders')
    """
    aggr = first_aggressor(actions)
    if not aggr:
        return True, "no_aggressor", None, None
    opener = aggr["pos"]
    rs = responders_after(actions, opener)
    if len(rs) == 0:
        return False, "opener_only", opener, None
    if len(rs) == 1:
        return False, "single_responder", opener, rs[0]
    return True, "multiple_responders", opener, rs[0]

def derive_context(opener_pos: str|None,
                   first_resp_pos: str|None,
                   actions: List[Dict[str,str]]) -> Tuple[str, str|None, str|None]:
    """
    Returns (action_context, ip_position, oop_position).
    If no responder -> 'OPEN' (opener-only).
    With responder: VS_OPEN / VS_3BET / VS_4BET depending on depth.
    """
    if opener_pos is None:
        return ("UNKNOWN", None, None)

    acts = [a["act"] for a in actions]
    if "4BET" in acts:
        ctx = "VS_4BET"
    elif "3BET" in acts:
        ctx = "VS_3BET"
    elif first_resp_pos is not None:
        ctx = "VS_OPEN"
    else:
        ctx = "OPEN"

    ip_pos = oop_pos = None
    if first_resp_pos:
        idx = {p:i for i,p in enumerate(POSITION_ORDER)}
        oop_pos = opener_pos if idx[opener_pos] < idx[first_resp_pos] else first_resp_pos
        ip_pos  = first_resp_pos if oop_pos == opener_pos else opener_pos

    return (ctx, ip_pos, oop_pos)

def parse_range_line(text: str) -> Dict[str, float]:
    """'AA:1.0,A2s:0.0,...' -> dict; clips to [0,1]; ignores unknown keys."""
    out: Dict[str,float] = {}
    for tok in text.strip().split(","):
        if not tok or ":" not in tok:
            continue
        k, v = tok.split(":", 1)
        k = k.strip().upper()
        if k.endswith("S"): k = k[:-1] + "s"
        if k.endswith("O"): k = k[:-1] + "o"
        try:
            w = float(v)
        except ValueError:
            continue
        w = max(0.0, min(1.0, w))
        if k in IDX169:
            out[k] = w
    return out

def range_map_to_vec169(rmap: Dict[str,float]) -> List[float]:
    vec = [0.0]*169
    for h, w in rmap.items():
        vec[IDX169[h]] = float(w)
    return vec

# ------------------------------
# File -> JSON row
# ------------------------------
def parse_monkerviewer_file(path: Path) -> Dict:
    """
    Expect: .../<stack>bb/<HERO_POS>/<FILENAME>.txt
    Output: {"meta": {...}, "range_map": {...}, "vector169": [...]}
    """
    stack_dir = path.parent.parent.name  # e.g., '12bb'
    hero_pos  = path.parent.name.upper()
    if hero_pos == "MP": hero_pos = "HJ"

    m_stack = re.search(r"(\d+)", stack_dir)
    stack_bb = int(m_stack.group(1)) if m_stack else None

    actions = parse_filename_actions(path.name)
    is_multi, reason, opener_pos, first_resp_pos = classify_hu_multiway(actions)
    ctx, ip_pos, oop_pos = derive_context(opener_pos, first_resp_pos, actions)

    text = path.read_text(encoding="utf-8")
    first_line = next((ln.strip() for ln in text.splitlines() if ":" in ln), "")

    rmap = parse_range_line(first_line)
    vec169 = range_map_to_vec169(rmap)

    meta = {
        "format_version": "monker_text_v1",
        "vendor": "monkerguy",
        "source_path": str(path),
        "stack_bb": stack_bb,
        "stack_key": f"{stack_bb}bb" if stack_bb is not None else None,
        "hero_position": hero_pos,
        "hero_index": POSITION_ORDER.index(hero_pos) if hero_pos in POSITION_ORDER else None,
        "action_sequence": actions,
        "participants": participants(actions),
        "is_multiway": bool(is_multi),
        "mw_reason": reason,  # opener_only / single_responder / multiple_responders / no_aggressor
        "derived": {
            "action_context": ctx,          # OPEN / VS_OPEN / VS_3BET / VS_4BET / UNKNOWN
            "ip_position": ip_pos,
            "oop_position": oop_pos,
            "opener": opener_pos,
            "first_responder": first_resp_pos
        },
        "hand_order": RANKS,
        "line_sha1": sha1(first_line),
    }
    return {"meta": meta, "range_map": rmap, "vector169": vec169}

# ------------------------------
# Export driver
# ------------------------------
def export_jsonl(root_dir: Path,
                 out_path: Path,
                 skip_multiway: bool = True,
                 include_contexts: List[str] | None = None) -> Dict[str,object]:
    """
    Recursively parse all *.txt and write JSONL(.gz).
    include_contexts limits to derived.action_context set.
    """
    total = hu = mw = 0
    stacks, seats = {}, {}
    mw_reasons, hu_reasons = {}, {}

    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        for path in root_dir.rglob("*.txt"):
            rec = parse_monkerviewer_file(path)
            total += 1

            if rec["meta"]["is_multiway"]:
                mw += 1
                r = rec["meta"].get("mw_reason","unknown")
                mw_reasons[r] = mw_reasons.get(r, 0) + 1
                if skip_multiway:
                    continue
            else:
                hu += 1
                r = rec["meta"].get("mw_reason","")
                hu_reasons[r] = hu_reasons.get(r, 0) + 1

            if include_contexts:
                if rec["meta"]["derived"]["action_context"] not in include_contexts:
                    continue

            sk = rec["meta"]["stack_key"]; st = rec["meta"]["hero_position"]
            stacks[sk] = stacks.get(sk, 0) + 1
            seats[st]  = seats.get(st, 0) + 1

            out.write(json.dumps(rec) + "\n")

    return {
        "total_files_seen": total,
        "written_hu_rows": hu if skip_multiway else (hu + mw),
        "skipped_multiway": mw if skip_multiway else 0,
        "by_stack": stacks,
        "by_seat": seats,
        "hu_reasons": hu_reasons,
        "mw_reasons": mw_reasons
    }

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    root = Path("data/vendor/monker")
    out  = Path("data/preflop/preflop.hu.v1.jsonl.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = export_jsonl(
        root_dir=root,
        out_path=out,
        skip_multiway=False,
        include_contexts=["OPEN","VS_OPEN","VS_3BET","VS_4BET"]
    )
    print("✅ Export summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")