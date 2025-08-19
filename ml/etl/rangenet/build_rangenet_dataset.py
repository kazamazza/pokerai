from __future__ import annotations
import os, re, json, gzip, random, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

# Optional progress
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# -------- config (edit paths if needed) --------
CHART_DIR = Path("data/vendor/monker")          # root of your vendor charts
OUT_PATH  = Path("data/rangenet/rangenet.v1.jsonl.gz")  # consolidated dataset output
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Try to import your existing helpers; if not present, we inline light versions
# You showed these earlier (ORDER169, IDX169, parse_filename_actions, parse_range_line, parse_monkerviewer_file)
try:
    from ml.etl.monker_preflop_etl import (
        parse_monkerviewer_file, IDX169
    )
    HAVE_HELPERS = True
except Exception:
    HAVE_HELPERS = False
    RANKS = "AKQJT98765432"
    def canonical_169_order() -> List[str]:
        order = []
        for i, r1 in enumerate(RANKS):
            for j, r2 in enumerate(RANKS):
                if i == j:   order.append(f"{r1}{r2}")
                elif i < j: order.append(f"{r1}{r2}s")
                else:       order.append(f"{r2}{r1}o")
        return order
    ORDER169 = canonical_169_order()
    IDX169   = {h:i for i,h in enumerate(ORDER169)}
    POS_SET  = {"UTG","HJ","MP","CO","BTN","SB","BB"}
    ACT_CANON = {
        "AI":"AI","ALLIN":"AI","SHOVE":"AI",
        "RAISE":"RAISE","OPEN":"OPEN","LIMP":"LIMP",
        "CALL":"CALL","3BET":"3BET","4BET":"4BET","FOLD":"FOLD"
    }
    def parse_filename_actions(name: str) -> List[Dict[str,str]]:
        stem = Path(name).stem
        parts = stem.split("_")
        out = []
        i = 0
        while i < len(parts):
            pos = parts[i].upper()
            if pos not in POS_SET:
                i += 1; continue
            if i+1 >= len(parts): break
            act = ACT_CANON.get(parts[i+1].upper(), parts[i+1].upper())
            out.append({"pos": pos, "act": act})
            i += 2
        return out
    def parse_range_line(text: str) -> Dict[str, float]:
        m: Dict[str,float] = {}
        for tok in text.strip().split(","):
            if not tok or ":" not in tok: continue
            k, v = tok.split(":", 1)
            k = k.strip().upper()
            if k.endswith("S"): k = k[:-1] + "s"
            if k.endswith("O"): k = k[:-1] + "o"
            try: w = float(v)
            except ValueError: continue
            w = max(0.0, min(1.0, w))
            if k in IDX169:
                m[k] = w
        return m
    def range_map_to_vec169(rmap: Dict[str,float]) -> List[float]:
        vec = [0.0]*169
        for h, w in rmap.items():
            vec[IDX169[h]] = float(w)
        return vec
    def parse_monkerviewer_file(path: Path) -> Dict:
        stack_dir = path.parent.parent.name  # e.g. '12bb'
        hero_pos  = path.parent.name.upper()
        m = re.search(r"(\d+)", stack_dir)
        stack_bb = int(m.group(1)) if m else None

        actions = parse_filename_actions(path.name)
        is_multiway = sum(1 for a in actions if a["act"] in {"OPEN","RAISE","AI","3BET","4BET","LIMP"}) > 1 \
                      or any(a["act"]=="CALL" for a in actions if actions and actions[0]["pos"]!="BB")

        txt = path.read_text(encoding="utf-8", errors="ignore").strip()
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

# --------- light context normalization ---------
ACTION_CTX_MAP = {
    "OPEN":   "OPEN",
    "RAISE":  "VS_OPEN",   # generic catch-all; if 3BET/4BET present we’ll refine below
    "3BET":   "VS_OPEN",
    "4BET":   "VS_3BET",
    "LIMP":   "VS_LIMP",
    "CALL":   "VS_OPEN",
    "AI":     "OPEN"       # many vendors label shoves as AI instead of RAISE/OPEN
}

def infer_action_context(actions: List[Dict[str,str]], hero_pos: str) -> Tuple[str, str]:
    """
    Returns (action_context, multiway_context) using your “types” enums.
    """
    opp_aggr = any(a["pos"] != hero_pos and a["act"] in {"OPEN","RAISE","3BET","4BET","AI"} for a in actions)
    levels   = [a["act"] for a in actions if a["pos"] != hero_pos]
    # default
    ctx = "OPEN" if not opp_aggr else "VS_OPEN"
    if "4BET" in levels:
        ctx = "VS_4BET"
    elif "3BET" in levels:
        ctx = "VS_3BET"
    elif "LIMP" in levels and not opp_aggr:
        ctx = "VS_LIMP"

    # multiway: count distinct aggressors/callers before hero
    involved = set(a["pos"] for a in actions if a["pos"] != hero_pos)
    mw = "HU"
    if len(involved) == 2: mw = "3WAY"
    elif len(involved) >= 3: mw = "4WAY_PLUS"

    return ctx, mw

def row_from_parsed(parsed: Dict) -> Dict:
    meta = parsed["meta"]
    vec  = parsed["vector169"]
    hero = meta.get("hero_position", "BTN")
    ctx, mw = infer_action_context(meta.get("action_sequence", []), hero)

    # Minimal, RangeNet-friendly context (expand later if you like)
    x = {
        "version": "rangenet.v1",
        "stack_bb": meta.get("stack_bb"),
        "rake_tier": None,          # fill if your folder structure encodes it
        "ante_bb": 0.0,             # cash default
        "hero_pos": hero,
        "action_context": ctx,      # OPEN, VS_OPEN, VS_3BET, VS_4BET, VS_LIMP
        "multiway_context": mw,     # HU, 3WAY, 4WAY_PLUS
        "source_path": meta.get("source_path"),
    }
    y = {
        "range169": vec            # length 169, floats in [0,1]
    }
    return {"x": x, "y": y}

# --------- main builder ---------
def build(chart_dir: Path = CHART_DIR, out_path: Path = OUT_PATH, max_files: int | None = None, shuffle: bool = True):
    if not chart_dir.exists():
        raise FileNotFoundError(f"Chart directory not found: {chart_dir}")

    # Collect files
    files = [p for p in chart_dir.rglob("*.txt") if p.is_file()]
    if shuffle:
        random.shuffle(files)
    if max_files:
        files = files[:max_files]

    total = len(files)
    if total == 0:
        print(f"⚠️ No .txt chart files found under {chart_dir}")
        return

    # Coverage counters
    by_stack: Dict[int,int] = {}
    by_pos:   Dict[str,int] = {}
    by_ctx:   Dict[str,int] = {}

    # Write
    n_ok = n_bad = 0
    t0 = time.time()
    with gzip.open(out_path, "wt", encoding="utf-8") as fout:
        pbar = tqdm(total=total, unit="files") if tqdm else None
        for i, path in enumerate(files, 1):
            try:
                parsed = parse_monkerviewer_file(path)
                vec = parsed.get("vector169", [])
                if not isinstance(vec, list) or len(vec) != 169:
                    n_bad += 1
                    if pbar: pbar.update(1)
                    continue
                sample = row_from_parsed(parsed)
                fout.write(json.dumps(sample, separators=(",", ":")) + "\n")
                n_ok += 1

                # coverage
                st = parsed["meta"].get("stack_bb")
                if isinstance(st, int): by_stack[st] = by_stack.get(st, 0) + 1
                hp = parsed["meta"].get("hero_position", "UNK")
                by_pos[hp] = by_pos.get(hp, 0) + 1
                ctx = sample["x"]["action_context"]
                by_ctx[ctx] = by_ctx.get(ctx, 0) + 1

            except Exception:
                n_bad += 1
            finally:
                if pbar:
                    pbar.update(1)
                    if i % 200 == 0:
                        pbar.set_postfix({"ok": n_ok, "bad": n_bad})
        if pbar: pbar.close()

    elapsed = time.time() - t0
    print(f"✅ RangeNet build → {out_path} | files={total:,} | rows_ok={n_ok:,} | bad={n_bad:,} | time={elapsed:.1f}s")

    # Pretty coverage summary
    def _sorted(d: Dict, key_cast=lambda k: k):
        return dict(sorted(d.items(), key=lambda kv: key_cast(kv[0])))

    print("\n— Coverage (by stack_bb) —")
    for k,v in _sorted(by_stack, int).items():
        print(f"  {k:>3}bb : {v:,}")
    print("\n— Coverage (by hero_pos) —")
    for k,v in _sorted(by_pos).items():
        print(f"  {k:>3} : {v:,}")
    print("\n— Coverage (by context) —")
    for k,v in _sorted(by_ctx).items():
        print(f"  {k:>7} : {v:,}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--charts", type=str, default=str(CHART_DIR), help="root dir of vendor charts")
    p.add_argument("--out",    type=str, default=str(OUT_PATH),  help="output JSONL.GZ")
    p.add_argument("--max",    type=int, default=0,              help="limit files (0=all)")
    p.add_argument("--no-shuffle", action="store_true",          help="disable random ordering")
    args = p.parse_args()

    build(
        chart_dir=Path(args.charts),
        out_path=Path(args.out),
        max_files=(args.max or None),
        shuffle=(not args.no_shuffle),
    )