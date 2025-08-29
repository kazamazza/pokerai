from __future__ import annotations
import os
from typing import Union
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import pandas as pd

from ml.etl.rangenet.preflop.monker_helpers import canon_pos, nearest_stack, is_srp_open_call, first_non_fold_opener

Number = Union[int, float]


def _load_vendor_range_compact(path: Path) -> str:
    """
    Load a vendor preflop range file in compact form (single-line CSV like 'AA:1.0,AKs:0.5,...').

    Validates that:
      - The file decodes successfully (tries UTF-8, then falls back to Latin-1 with explanation).
      - The content is non-empty and not a placeholder (no '...').
      - It looks like compact ranges (must contain at least one ',' and one ':')

    Returns the raw string as-is (caller passes directly to the solver).
    """
    if not path.exists():
        raise RuntimeError(f"Vendor range file not found: {path}")

    text: str
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Some packs come with odd encodings; try a permissive fallback.
        try:
            text = path.read_text(encoding="latin-1")
        except Exception as e:
            raise RuntimeError(f"Failed to decode vendor range file (utf-8, latin-1): {path}") from e

    s = text.strip()
    if not s:
        raise RuntimeError(f"Empty vendor range: {path}")

    # Guard against placeholder/dummy files
    if "..." in s:
        raise RuntimeError(f"Placeholder vendor range (contains '...'): {path}")

    # Very light sanity check: expect tokens like "XX:prob,YY:prob"
    if ("," not in s) or (":" not in s):
        raise RuntimeError(f"Unrecognized vendor range format (expected 'hand:weight' CSV): {path}")

    return s



POS_ORDER = ["UTG","HJ","CO","BTN","SB","BB"]
POS_IDX = {p:i for i,p in enumerate(POS_ORDER)}

def _one_step_neighbors(ip: str) -> list[str]:
    i = POS_IDX[ip]
    nbrs = []
    if i-1 >= 0: nbrs.append(POS_ORDER[i-1])  # one seat earlier
    if i+1 < len(POS_ORDER): nbrs.append(POS_ORDER[i+1])  # one seat later
    return [p for p in nbrs if p in ("UTG","HJ","CO","BTN")]  # keep opens in RFI seats

def _candidate_pairs(ip: str, oop: str, allow_pair_subs: bool) -> list[tuple[str,str,int,bool]]:
    # (cand_ip, cand_oop, fallback_level, substituted?)
    out = [(ip, oop, 0, False)]  # exact
    if allow_pair_subs:
        for ip2 in _one_step_neighbors(ip):
            out.append((ip2, oop, 2, True))  # 1-step opener sub
    return out

class PreflopRangeLookup:
    """
    Use monker_manifest to resolve preflop ranges.
    Index built from manifest using the same raw semantics as scan_monker
    (sequence_raw + SRP open/call detection).

    Index key: (stack_bb, hero_pos, ip, oop) -> [row dicts]

    Fallbacks:
      - nearest stack (always)
      - optional pair substitution (same OOP, nearest IP, 1-step)
      - optional max_stack_delta guard
    """

    def __init__(
        self,
        manifest_parquet: str | Path,
        *,
        s3_client: Optional["S3Client"] = None,
        s3_prefix: Optional[str] = None,
        cache_dir: str = "data/vendor_cache",
        allow_pair_subs: bool = False,
        max_stack_delta: Optional[int] = None,
    ):
        df = pd.read_parquet(str(manifest_parquet)).copy()
        need = {"stack_bb","hero_pos","sequence_raw","rel_path","abs_path"}
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise RuntimeError(f"monker_manifest missing columns: {missing}")

        # normalize hero_pos, coerce stacks
        df["hero_pos"] = df["hero_pos"].map(canon_pos)
        df["stack_bb"] = df["stack_bb"].astype("Int64")

        self.df = df
        self.stacks: list[int] = sorted(int(x) for x in df["stack_bb"].dropna().unique().tolist())
        if not self.stacks:
            raise RuntimeError("No stacks found in monker_manifest")

        self.s3 = s3_client
        self.s3_prefix = (s3_prefix or "").strip("/") if s3_prefix else None
        self.cache_dir = Path(cache_dir)
        self.allow_pair_subs = bool(allow_pair_subs)
        self.max_stack_delta = max_stack_delta if (max_stack_delta is None or max_stack_delta >= 0) else None

        # -------- build index from RAW vendor sequence --------
        # We derive opener from sequence_raw (raw tokens incl. percents) and detect SRP OPEN/CALL
        self.idx: Dict[Tuple[int, str, str, str], List[dict]] = {}
        for _, r in df.iterrows():
            try:
                seq_raw = json.loads(r["sequence_raw"])
            except Exception:
                continue
            if not isinstance(seq_raw, list) or len(seq_raw) < 2:
                continue

            opener_pos_raw, _ = first_non_fold_opener(seq_raw)   # raw tokens (incl. "Min","AI","60%")
            if not opener_pos_raw:
                continue

            # loop possible defenders present in this line
            seen_positions = {e.get("pos") for e in seq_raw if e.get("pos")}
            hero = canon_pos(str(r["hero_pos"]))
            stack = int(r["stack_bb"])

            for oop_pos_raw in seen_positions:
                ip_pos = canon_pos(opener_pos_raw)
                oop_pos = canon_pos(oop_pos_raw)
                if not ip_pos or not oop_pos or ip_pos == oop_pos:
                    continue

                # must qualify as SRP open/call using RAW-aware helper
                if not is_srp_open_call(seq_raw, ip_pos, oop_pos):
                    continue

                # only index from relevant hero POVs (IP and OOP)
                if hero not in (ip_pos, oop_pos):
                    continue

                key = (stack, hero, ip_pos, oop_pos)
                self.idx.setdefault(key, []).append({
                    "rel_path": r["rel_path"],
                    "abs_path": r["abs_path"],
                })

        # For substitution fallback inspection
        self._seen_pairs = {(k[2], k[3]) for k in self.idx.keys()}

    # ---------- path resolution ----------
    def _resolve_local_path(self, row: dict) -> Path:
        abs_path = Path(row["abs_path"])
        if abs_path.exists():
            return abs_path

        rel_path = Path(row["rel_path"])
        cache_path = self.cache_dir / rel_path
        if cache_path.exists():
            return cache_path

        if not (self.s3 and self.s3_prefix):
            raise RuntimeError(f"Vendor file not found: {rel_path}")

        # Normalize to S3-style keys (always '/')
        s3_key = f"{self.s3_prefix}/{str(rel_path).replace(os.sep, '/')}"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Use your helper that creates parent dirs and skips if exists
        self.s3.download_file_if_missing(s3_key, cache_path)
        if not cache_path.exists():
            raise RuntimeError(f"Failed to materialize vendor file: s3://{getattr(self.s3,'bucket','?')}/{s3_key}")
        return cache_path

    # ---------- index access ----------
    def _pick_row(self, stack: int, hero: str, ip: str, oop: str) -> Optional[dict]:
        key = (int(stack), canon_pos(hero), canon_pos(ip), canon_pos(oop))
        rows = self.idx.get(key)
        return rows[0] if rows else None

    def _ordered_stacks(self, target: int) -> List[int]:
        return sorted(self.stacks, key=lambda s: (abs(s - target), s))

    # ---------- main API ----------
    def ranges_for_pair(
        self,
        *,
        stack_bb: float,
        ip: str,
        oop: str,
        strict: bool = True,
    ) -> Tuple[Optional[str], Optional[str], Dict[str, object]]:
        ip, oop = canon_pos(ip), canon_pos(oop)
        near_stack = nearest_stack(stack_bb, self.stacks)

        last_meta: Dict[str, object] = {}
        for cand_ip, cand_oop, level, substituted in _candidate_pairs(ip, oop, self.allow_pair_subs):
            # Walk nearest stack first, then outward
            row_ip = row_oop = None
            meta = {
                "range_ip_source_stack": None,
                "range_oop_source_stack": None,
                "range_ip_stack_delta": None,
                "range_oop_stack_delta": None,
                "range_ip_fallback_level": None,
                "range_oop_fallback_level": None,
                "range_pair_substituted": substituted,
                "range_ip_source_pair": f"{cand_ip}v{cand_oop}",
                "range_oop_source_pair": f"{cand_ip}v{cand_oop}",
            }

            for s in self._ordered_stacks(int(near_stack)):
                delta = abs(s - int(near_stack))
                if self.max_stack_delta is not None and delta > self.max_stack_delta:
                    continue

                if row_ip is None:
                    row_ip = self._pick_row(s, hero=cand_ip, ip=cand_ip, oop=cand_oop)
                    if row_ip:
                        meta["range_ip_source_stack"] = s
                        meta["range_ip_stack_delta"] = delta
                        meta["range_ip_fallback_level"] = (0 if (level==0 and delta==0) else (2 if substituted else 1))

                if row_oop is None:
                    row_oop = self._pick_row(s, hero=cand_oop, ip=cand_ip, oop=cand_oop)
                    if row_oop:
                        meta["range_oop_source_stack"] = s
                        meta["range_oop_stack_delta"] = delta
                        meta["range_oop_fallback_level"] = (0 if (level==0 and delta==0) else (2 if substituted else 1))

                if row_ip and row_oop:
                    path_ip  = self._resolve_local_path(row_ip)
                    path_oop = self._resolve_local_path(row_oop)
                    rng_ip   = _load_vendor_range_compact(path_ip)
                    rng_oop  = _load_vendor_range_compact(path_oop)
                    return rng_ip, rng_oop, meta

            last_meta = meta  # keep best diagnostics

        if strict:
            raise RuntimeError(f"No ranges found for {ip}v{oop}@{stack_bb}")
        return None, None, last_meta