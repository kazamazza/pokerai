# ml/etl/range/export_rangenet_rows.py
from __future__ import annotations
import re, json, gzip, csv, sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

# ---- repo paths (so imports work if you run this directly) ----
ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from ml.schema.range_net_schema import (
    RangeNetSample, RangeNetFeatures, RangeNetLabel
)

# ---------- CONFIG ----------
MONKER_DIR = Path("data/vendor/monker")            # e.g. data/vendor/monker/12bb/BTN/*.txt
OUT_PATH   = Path("data/ranges/rangenet.v1.jsonl.gz")
DEV_MAX_FILES: Optional[int] = None                # e.g. 2_000 to throttle in dev

POSITIONS6 = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]

# ---------- 169 helpers ----------
_RANKS = "AKQJT98765432"
def _pair(i): return _RANKS[i] + _RANKS[i]
def _suited(i, j):  return _RANKS[i] + _RANKS[j] + "s"
def _offsuit(i, j): return _RANKS[i] + _RANKS[j] + "o"

HAND169: List[str] = []
for i in range(13):
    for j in range(13):
        if i == j: HAND169.append(_pair(i))
        elif i < j: HAND169.append(_suited(i, j))
        else: HAND169.append(_offsuit(i, j))

# ---------- filename → actions parsing ----------
ACT_CANON = {
    "AI":"AI","ALLIN":"AI","SHOVE":"AI",
    "RAISE":"RAISE","OPEN":"OPEN","LIMP":"LIMP",
    "CALL":"CALL","3BET":"3BET","4BET":"4BET","FOLD":"FOLD"
}

def parse_filename_actions(name: str) -> List[Dict[str, str]]:
    """'UTG_AI_HJ_Call_CO_Call_BTN_Call_SB_Call_BB_Call.txt' → [{'pos':'UTG','act':'AI'}, ...]"""
    stem = Path(name).stem
    parts = stem.split("_")
    out = []
    i = 0
    while i + 1 < len(parts):
        pos = parts[i].upper()
        act_raw = parts[i+1].upper()
        if pos in POSITIONS6:
            out.append({"pos": pos, "act": ACT_CANON.get(act_raw, act_raw)})
            i += 2
        else:
            i += 1
    return out

# ---------- load a simple Monker line: 'AA:1.0,A2s:0.0,...' ----------
def parse_range_line(text: str) -> Dict[str, float]:
    m: Dict[str, float] = {}
    for tok in text.strip().split(","):
        if not tok or ":" not in tok: continue
        k, v = tok.split(":", 1)
        k = k.strip().upper()
        # normalize suffix back to lower to match HAND169
        if k.endswith("S"): k = k[:-1] + "s"
        if k.endswith("O"): k = k[:-1] + "o"
        try:
            w = float(v)
        except ValueError:
            continue
        w = max(0.0, min(1.0, w))
        if k in HAND169:
            m[k] = w
    return m

def load_monker_txt(path: Path) -> Dict[str, float]:
    """Grab the first non-empty, colon-containing line and parse it."""
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" in ln:
            return parse_range_line(ln)
    return {}

# ---------- meta heuristics for your layout ----------
def parse_meta_from_path(p: Path) -> Dict:
    """
    Expect: data/vendor/monker/<stack>bb/<HERO_POS>/<FILENAME>.txt
    """
    # stack from parent dir of hero_pos
    stack_dir = p.parent.parent.name.lower()   # '12bb'
    m = re.search(r"(\d+)\s*bb", stack_dir)
    stack_bb = int(m.group(1)) if m else 15

    hero_pos = p.parent.name.upper()

    actions = parse_filename_actions(p.name)
    # try to find hero's declared act in the filename (last token often includes hero)
    hero_act: Optional[str] = None
    for a in actions:
        if a["pos"] == hero_pos:
            hero_act = a["act"]

    # rough context
    ctx = "OPEN"
    prior_acts = [a["act"] for a in actions if a["pos"] != hero_pos]
    if any(a in ("RAISE","OPEN","AI") for a in prior_acts):
        ctx = "VS_OPEN"
    if any(a == "3BET" for a in prior_acts):
        ctx = "VS_3BET"
    if any(a == "4BET" for a in prior_acts):
        ctx = "VS_4BET"

    return {
        "stake_tag": "NLx",
        "rake_tier": "MICRO/HIGH",
        "ante_bb": 0.0,
        "stack_bb": stack_bb,
        "hero_pos": hero_pos,
        "btn_pos": "BTN",
        "positions": {pos: pos for pos in POSITIONS6},
        "ctx": ctx,
        "hero_act": hero_act,          # for label mapping
        "action_seq": actions,         # kept for debugging if needed
    }

# ---------- dev mapping: range weight → [fold, call, raise] ----------
def action_probs_for(weight: float, hero_act: Optional[str]) -> List[float]:
    """
    Very simple dev mapping:
      - if filename says hero 'CALL' → probs = [1-w, w, 0]
      - else treat as raise/shove/open → probs = [1-w, 0, w]
    """
    w = max(0.0, min(1.0, float(weight)))
    if hero_act == "CALL":
        return [1.0 - w, w, 0.0]
    else:
        return [1.0 - w, 0.0, w]

# ---------- per-file → per-hand rows ----------
def file_to_rows(path: Path) -> Iterable[Dict]:
    meta = parse_meta_from_path(path)
    rmap = load_monker_txt(path)  # {'AA':1.0,'AKs':0.75,...}

    # default pot/amounts are 0 preflop for OPEN; adjust later for VS_* when you wire sizes
    for hand in HAND169:
        w = rmap.get(hand, 0.0)
        x = RangeNetFeatures(
            version="rangenet.v1",
            stake_tag=meta["stake_tag"],
            players=6,
            rake_tier=meta["rake_tier"],
            ante_bb=float(meta["ante_bb"]),
            open_size_policy="STD_2_5",
            hero_pos=meta["hero_pos"],
            btn_pos=meta["btn_pos"],
            positions=meta["positions"],
            ctx=meta["ctx"],
            multiway="HU",
            effective_stack_bb=float(meta["stack_bb"]),
            pot_bb=0.0,
            amount_to_call_bb=0.0,
            last_raise_to_bb=None,
            min_raise_to_bb=None,
            opener_pos=None,
            three_bettor_pos=None,
            four_bettor_pos=None,
            cold_callers=0,
            squeezers=0,
            hand_bucket=hand,
            hand_combo=None,
        )
        y = RangeNetLabel(
            action_probs=action_probs_for(w, meta["hero_act"]),
            exp_raise_bb=None,
            ev_bb=None,
        )
        sample = RangeNetSample(x=x, y=y)
        yield json.loads(sample.model_dump_json())

# ---------- main ----------
def run(monker_dir: Path = MONKER_DIR, out_path: Path = OUT_PATH, max_files: Optional[int] = DEV_MAX_FILES):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # enumerate: <monker>/<stack>bb/<POS>/*.txt
    files: List[Path] = []
    for stack_dir in sorted(monker_dir.glob("*bb")):
        for pos_dir in sorted(stack_dir.iterdir()):
            if not pos_dir.is_dir(): continue
            for fp in sorted(pos_dir.glob("*.txt")):
                files.append(fp)

    if max_files is not None:
        files = files[:max_files]

    n_files = len(files)
    n_rows  = 0
    bad     = 0

    open_gz = gzip.open if str(out_path).endswith(".gz") else open
    with open_gz(out_path, "wt", encoding="utf-8") as fout:
        for fp in files:
            try:
                for row in file_to_rows(fp):
                    fout.write(json.dumps(row, separators=(",", ":")) + "\n")
                    n_rows += 1
            except Exception as e:
                bad += 1

    # quick coverage snapshot
    cov: Dict[Tuple[int,str,str], int] = {}
    for fp in files:
        m = parse_meta_from_path(fp)
        key = (int(m["stack_bb"]), m["hero_pos"], m["ctx"])
        cov[key] = cov.get(key, 0) + 1

    print(f"✅ Range export → {out_path} | files={n_files:,} | rows={n_rows:,} | bad={bad:,}")
    print("   coverage (stack, hero_pos, ctx) top-10:")
    for (k, v) in sorted(cov.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"     {k}: {v}")

if __name__ == "__main__":
    run()