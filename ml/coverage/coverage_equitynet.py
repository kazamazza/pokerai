from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable, List

# ------------------------
# Helpers (reusable, no external deps)
# ------------------------

_POS_NAMES = {"UTG", "HJ", "CO", "BTN", "SB", "BB"}

# Normalize common action tokens from filenames to a compact set we’ll use as opener_action
_ACTION_ALIASES = {
    "OPEN": "OPEN",
    "MIN": "MIN", "MINRAISE": "MIN",
    "RAISE": "RAISE",
    "AI": "ALL_IN", "ALLIN": "ALL_IN", "ALL_IN": "ALL_IN",
    "LIMP": "LIMP",
    "CALL": "CALL",
    "FOLD": "FOLD",
    "3BET": "RAISE", "4BET": "RAISE",  # you can refine later if needed
}

def _to_alias(tok: str) -> Optional[str]:
    t = tok.upper()
    return _ACTION_ALIASES.get(t)

def _parse_stack_bb(dir_name: str) -> Optional[int]:
    """
    Accepts names like '12bb' or '25BB' → 12, 25
    """
    m = re.match(r"^\s*(\d+)\s*bb\s*$", dir_name.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))

def _infer_opener_action_from_filename(stem: str) -> Optional[str]:
    """
    Heuristic: filenames look like 'UTG_Min_HJ_Fold_CO_Call_...'
    We prefer the earliest token pair that looks like '<POS>_<ACTION>' and treat that as the opener.
    Returns normalized action (e.g., 'MIN', 'ALL_IN', 'OPEN', ...), or None if not found.
    """
    # Split by underscores and tidy
    parts = [p for p in stem.replace("__", "_").split("_") if p]
    # Scan for POS + ACTION pairs
    for i in range(len(parts) - 1):
        pos, act = parts[i].upper(), parts[i+1].upper()
        if pos in _POS_NAMES:
            alias = _to_alias(act)
            if alias:
                return alias
    # Fallback: single action present somewhere
    for p in parts:
        alias = _to_alias(p)
        if alias:
            return alias
    return None

def _iter_chart_files(root: Path) -> Iterable[Tuple[int, str, Path, str]]:
    """
    Yields tuples: (stack_bb, hero_pos, file_path, opener_action)
    Only includes *.txt files under <root>/<stackbb>/<hero_pos>/...
    """
    for bb_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        stack_bb = _parse_stack_bb(bb_dir.name)
        if stack_bb is None:
            continue
        for pos_dir in sorted(p for p in bb_dir.iterdir() if p.is_dir()):
            hero_pos = pos_dir.name.upper()
            if hero_pos not in _POS_NAMES:
                continue
            for txt in pos_dir.rglob("*.txt"):
                opener_action = _infer_opener_action_from_filename(txt.stem)
                yield (stack_bb, hero_pos, txt, opener_action or "UNKNOWN")

# ------------------------
# Main coverage builder
# ------------------------

def build_equitynet_manifest(
    root: str | Path,
    out_json: str | Path,
    min_files_per_group: int = 3,
) -> Dict:
    root = Path(root)
    out_json = Path(out_json)
    counts: Dict[Tuple[int, str, str], int] = defaultdict(int)

    for stack_bb, hero_pos, txt_path, opener_action in _iter_chart_files(root):
        key = (stack_bb, hero_pos, opener_action)
        counts[key] += 1

    by_group: List[Dict] = []
    ok_groups: List[Dict] = []

    for (stack_bb, hero_pos, opener_action), n in sorted(counts.items()):
        row = {
            "stack_bb": stack_bb,
            "hero_pos": hero_pos,
            "opener_action": opener_action,
            "n_files": n,
        }
        by_group.append(row)
        if n >= min_files_per_group:
            ok_groups.append({
                "stack_bb": stack_bb,
                "hero_pos": hero_pos,
                "opener_action": opener_action,
            })

    payload = {
        "model": "equitynet",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "group_keys": ["stack_bb", "hero_pos", "opener_action"],
        "thresholds": {"min_files_per_group": int(min_files_per_group)},
        "by_group": by_group,
        "ok_groups": ok_groups,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"✅ wrote EquityNet coverage → {out_json} ({len(by_group)} groups, {len(ok_groups)} OK)")

    return payload

# ------------------------
# CLI
# ------------------------

def main():
    ap = argparse.ArgumentParser(description="Build EquityNet coverage/manifest from Monker charts")
    ap.add_argument("--root", type=str, default="datasets/vendor/monker", help="Monker charts root folder")
    ap.add_argument("--out", type=str, default="ml/config/coverage/equitynet_manifest.json", help="Output JSON path")
    ap.add_argument("--min_files_per_group", type=int, default=3, help="Min txt files to mark a group as OK")
    args = ap.parse_args()

    build_equitynet_manifest(
        root=args.root,
        out_json=args.out,
        min_files_per_group=args.min_files_per_group,
    )

if __name__ == "__main__":
    main()