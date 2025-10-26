# tools/peek_solved_json.py
# Usage:
#   python tools/peek_solved_json.py --manifest data/artifacts/rangenet_postflop_flop_manifest_NL10_SMOKE.parquet --limit 8
# Prints raw + mapped views for root and first-bet node to help debug "no raises in facing".

import argparse, json, gzip, re, tempfile, os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np



ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from infra.storage.s3_client import S3Client
# tools/peek_dump_solved.py
# Usage:
#   python tools/peek_dump_solved.py \
#     --manifest data/artifacts/rangenet_postflop_flop_manifest_NL10_SMOKE.parquet \
#     --limit 40 \
#     --outdir data/peek

import argparse, json, gzip, os, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- simple regexes ---
BET_RE   = re.compile(r"\bbet\b", re.IGNORECASE)
RAISE_RE = re.compile(r"\braise\b", re.IGNORECASE)
CALL_RE  = re.compile(r"\bcall\b", re.IGNORECASE)
CHECK_RE = re.compile(r"\bcheck\b", re.IGNORECASE)
FOLD_RE  = re.compile(r"\bfold\b", re.IGNORECASE)
ALLIN_RE = re.compile(r"\ball[-\s]*in\b|\bjam\b", re.IGNORECASE)

# ---------- helpers to handle pyarrow/numpy ----------
def _to_py(obj: Any) -> Any:
    try:
        if hasattr(obj, "as_py"):
            return obj.as_py()
        if hasattr(obj, "to_pylist"):
            return obj.to_pylist()
    except Exception:
        pass
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj.item()
        return obj.tolist()
    return obj

def _norm_size_pct(x: Any) -> int:
    x = _to_py(x)
    f = float(x)
    if f <= 3.0:
        f *= 100.0
    f = max(1.0, min(200.0, f))
    return int(round(f))

def parse_menu_sizes(cell: Any) -> List[int]:
    cell = _to_py(cell)
    if cell is None:
        return []
    out: List[int] = []
    if isinstance(cell, list):
        for it in cell:
            it = _to_py(it)
            v = it.get("element") if isinstance(it, dict) else it
            if v is None:
                continue
            try:
                out.append(_norm_size_pct(v))
            except Exception:
                pass
    else:
        try:
            out.append(_norm_size_pct(cell))
        except Exception:
            pass
    return sorted(set(out))

def s3_key_for_size(base_key: str, size_pct: int) -> str:
    base = str(base_key).rstrip("/")
    sp = _norm_size_pct(size_pct)
    return f"{base}/size={sp}p/output_result.json.gz"

# ---------- solver JSON shape helpers ----------
def _open_json_any(path: str) -> Dict[str, Any]:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _normalize_children(node: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    for k in ("childrens", "children"):
        ch = node.get(k)
        if isinstance(ch, dict):
            return ch
        if isinstance(ch, list):
            out: Dict[str, Dict[str, Any]] = {}
            for c in ch:
                if isinstance(c, dict):
                    label = c.get("label") or c.get("action") or str(len(out))
                    out[str(label)] = c
            return out
    return {}

def _read_child_weight(node: Dict[str, Any]) -> Optional[float]:
    for k in ("prob", "p", "weight", "frequency", "freq", "w"):
        v = node.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    data = node.get("data")
    if isinstance(data, dict):
        for k in ("prob", "p", "weight", "frequency", "freq", "w"):
            v = data.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
    return None

def _renorm(acts: List[str], mix: List[float]) -> Tuple[List[str], List[float]]:
    mix = [max(0.0, float(x)) for x in mix]
    s = sum(mix)
    if s <= 0:
        return acts, ([1.0 / len(acts)] * len(acts) if acts else [])
    return acts, [x / s for x in mix]

def _find_any_strategy(node: Dict[str, Any]) -> Tuple[List[str], List[float], str]:
    if isinstance(node.get("actions"), list) and isinstance(node.get("strategy"), list):
        a = [str(x) for x in node["actions"]]
        m = [float(v) for v in node["strategy"]]
        a, m = _renorm(a, m)
        return a, m, "node.actions+strategy(list)"
    s = node.get("strategy")
    if isinstance(s, dict) and "actions" in s and "strategy" in s:
        actions = [str(x) for x in s["actions"]]
        k = len(actions)
        strat = s["strategy"]
        if isinstance(strat, dict) and strat:
            acc = [0.0] * k
            n = 0
            for row in strat.values():
                if isinstance(row, list) and len(row) == k:
                    for i, x in enumerate(row):
                        acc[i] += float(x)
                    n += 1
            if n > 0:
                mix = [v / max(n, 1) for v in acc]
                actions, mix = _renorm(actions, mix)
                return actions, mix, "node.strategy{combo->list}"
        if isinstance(strat, list) and len(strat) == k:
            mix = [float(x) for x in strat]
            actions, mix = _renorm(actions, mix)
            return actions, mix, "node.strategy(list)"
    if isinstance(s, dict) and s and all(isinstance(v, (int, float)) for v in s.values()):
        actions = list(s.keys())
        mix = [float(s[k]) for k in actions]
        actions, mix = _renorm([str(x) for x in actions], mix)
        return actions, mix, "node.strategy(map)"
    kids = _normalize_children(node)
    acts, mix = [], []
    if kids:
        for lbl, ch in kids.items():
            w = _read_child_weight(ch)
            if w is not None:
                acts.append(str(lbl))
                mix.append(float(w))
    if acts and sum(mix) > 0:
        acts, mix = _renorm(acts, mix)
        return acts, mix, "children.weighted"
    if kids:
        acts = list(kids.keys())
        mix = [1.0 / len(acts)] * len(acts)
        return acts, mix, "children.uniform"
    return [], [], "none"

def _find_first_bet_node(root: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    kids = _normalize_children(root)
    for lbl, ch in kids.items():
        if "bet" in str(lbl).lower():
            return ch, [str(lbl)]
    for lbl, ch in kids.items():
        if "check" in str(lbl).lower():
            kids2 = _normalize_children(ch)
            for lbl2, ch2 in kids2.items():
                if "bet" in str(lbl2).lower():
                    return ch2, [str(lbl), str(lbl2)]
    stack = [([], root)]
    while stack:
        path, node = stack.pop()
        for lbl, ch in _normalize_children(node).items():
            np = path + [str(lbl)]
            if "bet" in str(lbl).lower():
                return ch, np
            stack.append((np, ch))
    return None, []

# ---------- S3 mirror+ungzip ----------
def mirror_and_unzip(s3_key: str, outdir: Path) -> Path:
    """
    Save uncompressed JSON under outdir with a readable name; returns final path.
    """
    # ensure parent
    outdir.mkdir(parents=True, exist_ok=True)

    # download gzip to temp
    tmp_gz = outdir / (".tmp_" + Path(s3_key).name)
    S3Client().download_file(s3_key, tmp_gz)

    # read gz bytes → json text (don’t pretty-print to avoid huge files)
    with gzip.open(tmp_gz, "rt", encoding="utf-8") as f:
        text = f.read()

    # craft readable filename from s3_key
    # .../pos=COvBB/stack=25/pot=19.5/board=KsJd8s/.../sizes=<menu>/.../<sha>/size=66p/output_result.json.gz
    parts = s3_key.split("/")
    pos    = next((p.split("=")[1] for p in parts if p.startswith("pos=")), "pos")
    stack  = next((p.split("=")[1] for p in parts if p.startswith("stack=")), "stk")
    pot    = next((p.split("=")[1] for p in parts if p.startswith("pot=")), "pot")
    board  = next((p.split("=")[1] for p in parts if p.startswith("board=")), "board")
    sizes  = next((p.split("=")[1] for p in parts if p.startswith("sizes=")), "menu")
    sizep  = next((p.split("=")[1] for p in parts if p.startswith("size=")), "sz")
    sha    = next((parts[i+1] for i,p in enumerate(parts) if len(p)==2 and i+1 < len(parts) and len(parts[i+1])>=10), "sha")[:10]

    safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s))
    fname = f"{safe(sha)}__{safe(pos)}__stk{safe(stack)}__pot{safe(pot)}__{safe(board)}__{safe(sizes)}__sz{safe(sizep)}.json"
    out_path = outdir / fname

    if out_path.exists():
        out_path.unlink()
    with open(out_path, "w", encoding="utf-8") as w:
        w.write(text)

    try:
        tmp_gz.unlink()
    except Exception:
        pass

    return out_path

def summarize_nodes(payload: Dict[str, Any], pot_bb: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    root = payload.get("root", payload)
    acts, mix, where = _find_any_strategy(root)
    root_sum = float(sum(mix))
    root_view = {
        "where": where,
        "sum": root_sum,
        "labels": ";".join([f"{a}:{p:.3f}" for a,p in zip(acts, mix) if p>1e-6])[:2000],
    }

    node, path = _find_first_bet_node(root)
    if node is None:
        facing_view = {"path": "(none)", "where": "-", "sum": 0.0, "labels": ""}
        return root_view, facing_view

    a2, m2, w2 = _find_any_strategy(node)
    facing_view = {
        "path": " → ".join(path),
        "where": w2,
        "sum": float(sum(m2)),
        "labels": ";".join([f"{a}:{p:.3f}" for a,p in zip(a2, m2) if p>1e-6])[:2000],
        "has_bet_token": any(BET_RE.search(a) for a in a2),
        "has_raise_token": any(RAISE_RE.search(a) for a in a2),
        "has_call": any(CALL_RE.search(a) for a in a2),
        "has_fold": any(FOLD_RE.search(a) for a in a2),
    }
    return root_view, facing_view

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--outdir", type=str, default="data/peek")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.manifest)
    if args.limit is not None:
        df = df.head(int(args.limit))

    rows: List[Dict[str, Any]] = []
    written = 0
    for _, r in df.iterrows():
        base_key = str(r.get("s3_key"))
        sizes = parse_menu_sizes(r.get("bet_sizes")) or [33]
        pot_bb = float(r.get("pot_bb") or 0.0)
        sha = str(r.get("sha1") or "")[:10]

        for sp in sizes:
            s3k = s3_key_for_size(base_key, sp)
            try:
                path = mirror_and_unzip(s3k, outdir)
            except Exception as e:
                rows.append({
                    "sha": sha, "size_pct": sp, "s3_key": s3k,
                    "file": "", "ok": False, "reason": f"download/unzip: {e}"
                })
                continue

            try:
                payload = _open_json_any(str(path))  # now uncompressed .json
                root_view, facing_view = summarize_nodes(payload, pot_bb)
                rows.append({
                    "sha": sha, "size_pct": sp, "s3_key": s3k,
                    "file": str(path.relative_to(outdir)),
                    "ok": True,
                    "root_where": root_view["where"], "root_sum": root_view["sum"], "root_labels": root_view["labels"],
                    "facing_path": facing_view["path"], "facing_where": facing_view.get("where",""),
                    "facing_sum": facing_view.get("sum",0.0), "facing_labels": facing_view.get("labels",""),
                    "facing_has_bet_token": facing_view.get("has_bet_token", False),
                    "facing_has_raise_token": facing_view.get("has_raise_token", False),
                    "facing_has_call": facing_view.get("has_call", False),
                    "facing_has_fold": facing_view.get("has_fold", False),
                })
                written += 1
            except Exception as e:
                rows.append({
                    "sha": sha, "size_pct": sp, "s3_key": s3k,
                    "file": str(path.relative_to(outdir)),
                    "ok": False, "reason": f"open/summarize: {e}"
                })

    rep = pd.DataFrame(rows)
    rep_path = outdir / "peek_report.csv"
    rep.to_csv(rep_path, index=False)
    print(f"📄 Report: {rep_path}")
    print(f"📂 JSONs:  {outdir.resolve()}  (one .json per s3 key / size)")
    print(f"✅ Done. Files: {written}, Rows: {len(rows)}")

if __name__ == "__main__":
    main()