#!/usr/bin/env python3
"""
Scan coverage + eval JSONs across all models and produce a single summary:
- coverage health per model
- eval quality per model
- worst-by-weight groups to target next

Assumes files are stored in:
  reports/coverage/<model>/*.json
  reports/eval/<model>/*.json
Outputs:
  reports/summary/overall_report.json
  reports/summary/overall_report.md
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

MODELS = [
    "populationnet",
    "equity_preflop",
    "equity_postflop",
    "exploitnet",
]

ROOT = Path("reports")
COV_DIR = ROOT / "coverage"
EVAL_DIR = ROOT / "eval"
OUT_DIR = ROOT / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds (tweak as you like)
COVERAGE_OK_PCT_WARN = 0.85      # warn if < 85% ok cells
EVAL_VAL_KL_WARN = 0.60          # warn if val_kl above this
TOP_K_GROUPS = 5                 # show top-k worst groups per model


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _find_latest_json(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("*.json"))


# ---------------- Coverage parsers ----------------

def parse_population_coverage(d: Dict[str, Any]) -> Dict[str, Any]:
    # Expected structure from our coverage script: {"populationnet": {...}}
    blob = d.get("populationnet", {})
    inc = blob.get("include_contexts", [])
    # Fallbacks
    ok_cells = len(blob.get("ok_cells", []))
    freq_cells = blob.get("freq_cells", [])
    total_cells = len(freq_cells) if freq_cells else ok_cells

    # If we have ctx_summaries, sum cells_ok as proxy
    if blob.get("ctx_summaries"):
        total_cells = sum(int(x.get("cells_ok", 0)) for x in blob["ctx_summaries"])
        ok_cells = total_cells  # those are already filtered OK contexts

    ok_pct = (ok_cells / total_cells) if total_cells > 0 else 0.0
    return {
        "model": "populationnet",
        "ok_cells": ok_cells,
        "total_cells": total_cells,
        "ok_pct": ok_pct,
        "include_contexts": inc,
    }


def parse_generic_coverage(d: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Generic reader for coverage JSONs you might add for equitynet/exploit:
    Prefer keys: {"summary": {"total_cells": X, "ok_cells": Y, ...}}
    Otherwise, try to infer from simple lists.
    """
    summ = d.get("summary", {})
    total = summ.get("total_cells")
    ok = summ.get("ok_cells")
    if total is None or ok is None:
        # Try common fallbacks
        ok_cells = d.get("ok_cells", [])
        freq_cells = d.get("freq_cells", [])
        ok = len(ok_cells) if isinstance(ok_cells, list) else summ.get("ok_cells", 0)
        total = len(freq_cells) if isinstance(freq_cells, list) else summ.get("total_cells", ok)
    ok_pct = (ok / total) if total else 0.0
    return {
        "model": model,
        "ok_cells": ok,
        "total_cells": total,
        "ok_pct": ok_pct,
    }


def summarize_coverage() -> Dict[str, Any]:
    out = {"by_model": {}, "warnings": []}
    for m in MODELS:
        files = _find_latest_json(COV_DIR / m)
        if not files:
            out["by_model"][m] = {"missing": True}
            out["warnings"].append(f"[Coverage] No coverage files for {m}")
            continue
        # Use most recent JSON (sorted by name)
        cov = _load_json(files[-1])
        if not cov:
            out["by_model"][m] = {"error": "unreadable"}
            out["warnings"].append(f"[Coverage] Unreadable JSON for {m}: {files[-1]}")
            continue

        if m == "populationnet":
            cm = parse_population_coverage(cov)
        else:
            cm = parse_generic_coverage(cov, m)

        out["by_model"][m] = cm
        if cm.get("ok_pct", 1.0) < COVERAGE_OK_PCT_WARN:
            out["warnings"].append(f"[Coverage] {m} ok_pct={cm['ok_pct']:.2f} below threshold {COVERAGE_OK_PCT_WARN}")
    return out


# ---------------- Eval parsers ----------------

def parse_eval_file(path: Path, model: str) -> Dict[str, Any] | None:
    d = _load_json(path)
    if not d:
        return None
    # Expected fields we emitted earlier:
    # { "checkpoint": ..., "val_weight_sum": ..., "val_kl": ..., "val_soft_acc": ...,
    #   "by_group": { "ctx1_street0": {"kl":..., "soft_acc":..., "w":...}, ... } }
    val_kl = d.get("val_kl")
    val_soft = d.get("val_soft_acc")
    by_group = d.get("by_group", {})
    # rank worst groups by (w * kl)
    worst = []
    for gid, g in by_group.items():
        kl = float(g.get("kl", 0.0))
        w = float(g.get("w", 0.0))
        score = w * kl
        worst.append((gid, score, kl, w, float(g.get("soft_acc", 0.0))))
    worst.sort(key=lambda t: t[1], reverse=True)
    topk = [{
        "group": g[0],
        "score_w_kl": g[1],
        "kl": g[2],
        "w": g[3],
        "soft_acc": g[4],
    } for g in worst[:TOP_K_GROUPS]]

    return {
        "model": model,
        "file": str(path),
        "val_kl": val_kl,
        "val_soft_acc": val_soft,
        "val_weight_sum": d.get("val_weight_sum"),
        "worst_groups": topk,
    }


def summarize_evals() -> Dict[str, Any]:
    out = {"by_model": {}, "warnings": []}
    for m in MODELS:
        files = _find_latest_json(EVAL_DIR / m)
        if not files:
            out["by_model"][m] = {"missing": True}
            out["warnings"].append(f"[Eval] No eval files for {m}")
            continue
        latest = files[-1]
        summ = parse_eval_file(latest, m)
        if not summ:
            out["by_model"][m] = {"error": "unreadable"}
            out["warnings"].append(f"[Eval] Unreadable eval for {m}: {latest}")
            continue
        out["by_model"][m] = summ
        if summ.get("val_kl") is not None and float(summ["val_kl"]) > EVAL_VAL_KL_WARN:
            out["warnings"].append(f"[Eval] {m} val_kl={summ['val_kl']:.3f} above {EVAL_VAL_KL_WARN}")
    return out


# ---------------- Combine + write ----------------

def to_markdown(coverage: Dict[str, Any], evals: Dict[str, Any]) -> str:
    lines = []
    lines.append("# ML Report Summary\n")
    lines.append("## Coverage\n")
    for m, cm in coverage["by_model"].items():
        if cm.get("missing"):
            lines.append(f"- **{m}**: _no coverage files_")
            continue
        if cm.get("error"):
            lines.append(f"- **{m}**: _coverage unreadable_")
            continue
        ok_pct = cm.get("ok_pct", 0.0)
        ok = cm.get("ok_cells", "—")
        total = cm.get("total_cells", "—")
        lines.append(f"- **{m}**: ok_cells={ok}/{total} (ok_pct={ok_pct:.2f})")
    if coverage.get("warnings"):
        lines.append("\n**Coverage warnings:**")
        for w in coverage["warnings"]:
            lines.append(f"- {w}")

    lines.append("\n## Eval\n")
    for m, em in evals["by_model"].items():
        if em.get("missing"):
            lines.append(f"- **{m}**: _no eval files_")
            continue
        if em.get("error"):
            lines.append(f"- **{m}**: _eval unreadable_")
            continue
        lines.append(f"- **{m}**: val_kl={em.get('val_kl', '—')}, soft_acc={em.get('val_soft_acc', '—')}")
        worst = em.get("worst_groups", [])
        if worst:
            lines.append(f"  - worst groups (by w×kl):")
            for g in worst:
                lines.append(f"    - {g['group']}: score={g['score_w_kl']:.1f}, kl={g['kl']:.3f}, w={g['w']:.0f}, soft_acc={g['soft_acc']:.3f}")
    if evals.get("warnings"):
        lines.append("\n**Eval warnings:**")
        for w in evals["warnings"]:
            lines.append(f"- {w}")

    # Quick actionable pointers
    lines.append("\n## Next Actions\n")
    lines.append("- For models with **low coverage ok_pct**: generate more data for those cells (check coverage JSON for which).")
    lines.append("- For models with **high val_kl**: inspect top worst groups (above) and target those slices in data generation.")
    lines.append("- Re-train with updated parquets; re-run eval; repeat until top-weighted KLs shrink.")
    return "\n".join(lines)


def main():
    cov = summarize_coverage()
    evl = summarize_evals()

    # Write JSON
    report = {"coverage": cov, "eval": evl}
    (OUT_DIR / "overall_report.json").write_text(json.dumps(report, indent=2))

    # Write Markdown
    md = to_markdown(cov, evl)
    (OUT_DIR / "overall_report.md").write_text(md)

    print(f"✅ wrote {OUT_DIR/'overall_report.json'}")
    print(f"✅ wrote {OUT_DIR/'overall_report.md'}")


if __name__ == "__main__":
    main()