# tools/rangenet/_preflop_range_lookup.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import pandas as pd
import json

POS_ALIASES = {
    "BU": "BTN", "EP": "UTG", "MP": "HJ",
}
def norm_pos(p: str) -> str:
    p = (p or "").upper()
    return POS_ALIASES.get(p, p)

def nearest(v: float, options: List[float]) -> float:
    return min(options, key=lambda x: (abs(x - v), x))

def _load_vendor_range_compact(path: Path) -> str:
    s = path.read_text(encoding="utf-8").strip()
    if not s or "..." in s:
        raise RuntimeError(f"Empty/placeholder vendor range: {path}")
    if "," not in s or ":" not in s:
        raise RuntimeError(f"Unrecognized vendor range format: {path}")
    return s

def _first_non_fold(seq: list[dict]) -> Optional[Tuple[str, str]]:
    for e in seq:
        a = (e.get("action") or "").upper()
        if a and a != "FOLD":
            return norm_pos(e.get("pos")), a
    return None

def _first_action_of(seq: list[dict], pos: str) -> Optional[str]:
    target = norm_pos(pos)
    for e in seq:
        if norm_pos(e.get("pos")) == target:
            return (e.get("action") or "").upper()
    return None

def _re_raised_before(seq: list[dict], before_pos: str) -> bool:
    target = norm_pos(before_pos)
    for e in seq:
        if norm_pos(e.get("pos")) == target:
            return False
        a = (e.get("action") or "").upper()
        if a in ("RAISE", "ALL_IN", "3BET", "4BET", "5BET"):
            return True
    return False

class PreflopRangeLookup:
    """
    Strict resolver using *actual manifest rows* (no fabricated stems).
    Index: (stack, hero_pos, ip, oop) -> list of rows with real filename/abs_path.
    """
    def __init__(self, manifest_parquet: str | Path):
        df = pd.read_parquet(str(manifest_parquet)).copy()
        need = {"stack_bb","hero_pos","sequence","filename_stem","abs_path"}
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise RuntimeError(f"monker_manifest missing columns: {missing}")

        # normalize
        df["hero_pos"] = df["hero_pos"].map(norm_pos)
        # coerce stack ints
        df["stack_bb"] = df["stack_bb"].astype("Int64")

        self.stacks: List[int] = sorted(int(x) for x in df["stack_bb"].dropna().unique().tolist())
        if not self.stacks:
            raise RuntimeError("No stacks found in monker_manifest")

        # Build index
        idx: Dict[Tuple[int, str, str, str], List[dict]] = {}

        for _, r in df.iterrows():
            stack = int(r["stack_bb"])
            hero  = norm_pos(str(r["hero_pos"]))
            try:
                seq = json.loads(r["sequence"])
            except Exception:
                continue
            if not isinstance(seq, list) or len(seq) < 2:
                continue

            opener = _first_non_fold(seq)
            if not opener:
                continue
            ip_pos, ip_act = opener
            if ip_act not in ("OPEN","RAISE","ALL_IN","LIMP","CALL"):
                continue

            # for every possible OOP seat present in seq, check SRP (first action CALL, no re-raise before)
            seen_positions = [norm_pos(e.get("pos")) for e in seq if e.get("pos")]
            for oop_pos in seen_positions:
                if oop_pos == ip_pos:
                    continue
                a_opp = _first_action_of(seq, oop_pos)
                if a_opp != "CALL":
                    continue
                if _re_raised_before(seq, oop_pos):
                    continue

                key = (stack, hero, ip_pos, oop_pos)
                idx.setdefault(key, []).append({
                    "filename_stem": r["filename_stem"],
                    "abs_path": r["abs_path"],
                })

        self.idx = idx  # (stack, hero_pos, ip, oop) -> list of candidates

    def _pick_row(self, stack: int, hero: str, ip: str, oop: str) -> Optional[dict]:
        key = (int(stack), norm_pos(hero), norm_pos(ip), norm_pos(oop))
        rows = self.idx.get(key)
        if rows:
            # simple choice: first; you can add heuristics (prefer all folds except oop, etc.)
            return rows[0]
        return None

    def ranges_for_pair(
            self,
            *,
            stack_bb: float,
            ip: str,
            oop: str,
            strict: bool = True,  # NEW: when False, return (None, None) instead of raising
    ) -> Tuple[Optional[str], Optional[str]]:
        ip, oop = norm_pos(ip), norm_pos(oop)
        if not self.stacks:
            raise RuntimeError("No stacks indexed")
        nearest_stack = int(nearest(float(stack_bb), [float(s) for s in self.stacks]))

        # IP range from IP folder (hero_pos = IP)
        row_ip = self._pick_row(nearest_stack, hero=ip, ip=ip, oop=oop)
        # OOP range from OOP folder (hero_pos = OOP)
        row_oop = self._pick_row(nearest_stack, hero=oop, ip=ip, oop=oop)

        # Try other stacks by increasing distance if one side is missing
        if (row_ip is None or row_oop is None) and len(self.stacks) > 1:
            ordered = sorted(self.stacks, key=lambda s: (abs(s - nearest_stack), s))
            for s in ordered:
                if row_ip is None:
                    row_ip = self._pick_row(s, hero=ip, ip=ip, oop=oop)
                if row_oop is None:
                    row_oop = self._pick_row(s, hero=oop, ip=ip, oop=oop)
                if row_ip and row_oop:
                    break

        if row_ip is None or row_oop is None:
            # -------- improved diagnostics --------
            lines = [
                f"Missing vendor rows for {ip}v{oop} (requested {stack_bb}bb, nearest {nearest_stack}bb)."
            ]
            # Show what *does* exist for this pair across stacks & hero_sides
            have = [(k[0], k[1]) for k in self.idx.keys() if k[2] == ip and k[3] == oop]
            if have:
                lines.append("Available for this pair (stack_bb, hero_side):")
                # unique + sorted
                seen = sorted(set(have), key=lambda t: (t[0], t[1]))
                lines += [f"  - {s}bb, hero={h}" for (s, h) in seen]
            else:
                lines.append("No entries for this pair were found in the manifest index.")

            msg = "\n".join(lines)
            if strict:
                raise RuntimeError(msg)
            # allow caller to skip gracefully
            return None, None

        rng_ip = _load_vendor_range_compact(Path(row_ip["abs_path"]))
        rng_oop = _load_vendor_range_compact(Path(row_oop["abs_path"]))
        return rng_ip, rng_oop