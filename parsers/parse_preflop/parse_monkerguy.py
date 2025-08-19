import re, json
from pathlib import Path
from typing import Dict, List, Tuple

# ----- 169 ordering helpers (AA..22, suited above diag, off below)
RANKS = "AKQJT98765432"
def canonical_169_order() -> List[str]:
    order = []
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i == j:
                order.append(f"{r1}{r2}")          # pairs
            elif i < j:
                order.append(f"{r1}{r2}s")         # suited (high first)
            else:
                order.append(f"{r2}{r1}o")         # offsuit (high first)
    return order

ORDER169 = canonical_169_order()
IDX169: Dict[str,int] = {h:i for i,h in enumerate(ORDER169)}

POS_SET = {"UTG","HJ","MP","CO","BTN","SB","BB"}  # accept both HJ/MP if your pack uses them
ACT_CANON = {
    "AI":"AI","ALLIN":"AI","SHOVE":"AI",
    "RAISE":"RAISE","OPEN":"OPEN","LIMP":"LIMP",
    "CALL":"CALL","3BET":"3BET","4BET":"4BET","FOLD":"FOLD"
}

def parse_filename_actions(name: str) -> List[Dict[str,str]]:
    """
    Parse patterns like 'UTG_AI_HJ_Call_CO_Call_BTN_Call_SB_Call_BB_Call.txt'
    → [{"pos":"UTG","act":"AI"}, {"pos":"HJ","act":"CALL"}, ...]
    """
    stem = Path(name).stem
    parts = stem.split("_")
    out = []
    i = 0
    while i < len(parts)-0:
        pos = parts[i].upper()
        if pos not in POS_SET: i += 1; continue
        if i+1 >= len(parts): break
        act_raw = parts[i+1].upper()
        act = ACT_CANON.get(act_raw, act_raw)
        out.append({"pos": pos, "act": act})
        i += 2
    return out

def parse_range_line(text: str) -> Dict[str, float]:
    """
    Parse 'AA:1.0,A2s:0.0,...' to dict; clip to [0,1]; ignore unknown tokens.
    """
    m: Dict[str,float] = {}
    for tok in text.strip().split(","):
        if not tok: continue
        if ":" not in tok: continue
        k, v = tok.split(":", 1)
        k = k.strip().upper()          # normalize, e.g. 'A2s'->'A2S'
        # restore suited/off suffix lowercase to match our ORDER169 keys
        if k.endswith("S"): k = k[:-1] + "s"
        if k.endswith("O"): k = k[:-1] + "o"
        try:
            w = float(v)
        except ValueError:
            continue
        w = max(0.0, min(1.0, w))      # clip
        if k in IDX169:
            m[k] = w
        # if k is a pair range like '22-66' etc., expand here if needed
    return m

def range_map_to_vec169(rmap: Dict[str,float]) -> List[float]:
    """
    Produce a dense 169 vector. Missing hands default to 0.0.
    """
    vec = [0.0]*169
    for h, w in rmap.items():
        vec[IDX169[h]] = float(w)
    return vec

def parse_monkerviewer_file(path: Path) -> Dict:
    """
    path: .../<stack>bb/<HERO_POS>/<FILENAME>.txt
    """
    # infer meta from path
    stack_dir = path.parent.parent.name  # e.g., '12bb'
    hero_pos  = path.parent.name.upper()
    stack_match = re.search(r"(\d+)", stack_dir)
    stack_bb = int(stack_match.group(1)) if stack_match else None

    actions = parse_filename_actions(path.name)
    is_multiway = sum(1 for a in actions if a["act"] in {"OPEN","RAISE","AI","3BET","4BET","LIMP"}) > 1 \
                  or any(a["act"]=="CALL" for a in actions if actions and actions[0]["pos"]!="BB")

    # load range line (first non-empty line)
    txt = path.read_text(encoding="utf-8").strip()
    # Some exports put header lines; grab the first line that has ':' pairs
    line = next((ln for ln in txt.splitlines() if ":" in ln), "")

    rmap = parse_range_line(line)
    vec169 = range_map_to_vec169(rmap)

    return {
        "meta": {
            "stack_bb": stack_bb,
            "hero_position": hero_pos,
            "action_sequence": actions,
            "is_multiway": bool(is_multiway),
            "source_path": str(path)
        },
        "range_map": rmap,
        "vector169": vec169
    }