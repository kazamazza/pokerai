from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json, os

import numpy as np
import pandas as pd

from infra.storage.s3_client import S3Client
from ml.config.types_hands import RANKS
from ml.etl.rangenet.candidate_pairs import candidate_pairs
from ml.etl.rangenet.preflop.monker_helpers import canon_pos, first_non_fold_opener
from ml.etl.utils.range_lookup import is_srp_open_call, _load_vendor_range_compact, _candidate_pairs, nearest_stack


# in ml/etl/rangenet/preflop/range_lookup.py (or wherever SphIndex lives)

def _join_s3(*parts: str) -> str:
    return "/".join(str(p).strip("/").replace("\\", "/") for p in parts if p is not None and str(p) != "")

def _ensure_subdir(rel_path: str, subdir: str) -> str:
    """Ensure rel_path is rooted under subdir (monker|sph) without duplicating it."""
    rel = str(rel_path).lstrip("/").replace("\\", "/")
    sub = subdir.strip("/")
    return rel if rel.startswith(sub + "/") else f"{sub}/{rel}"

class SphIndex:
    """
    Minimal index over your Simple Preflop Hold'em exports.
    Expects sph_manifest.parquet with:
      stack_bb, ip_pos, oop_pos, ctx, hero_pos, rel_path, abs_path
    """


    def __init__(
        self,
        manifest_parquet: str | Path,
        cache_dir: str = "data/vendor_cache",
        *,
        s3_vendor: str | None = None,         # e.g. "s3://bucket/data/vendor_cache"
        s3_client: "S3Client | None" = None,
    ):
        df = pd.read_parquet(str(manifest_parquet)).copy()
        need = {"stack_bb", "ip_pos", "oop_pos", "ctx", "hero_pos", "rel_path", "abs_path"}
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise RuntimeError(f"sph_manifest missing columns: {missing}")

        # normalize
        df["ip_pos"]   = df["ip_pos"].map(canon_pos)
        df["oop_pos"]  = df["oop_pos"].map(canon_pos)
        df["hero_pos"] = df["hero_pos"].map(lambda p: canon_pos(p) or str(p).upper())
        df["ctx"]      = df["ctx"].map(lambda s: str(s).upper())
        df["stack_bb"] = df["stack_bb"].astype("Int64")

        self.df = df
        self.cache_dir = Path(cache_dir)
        self.s3_vendor = (s3_vendor or "").rstrip("/") if s3_vendor else None
        self.s3 = s3_client

        self.stacks: list[int] = sorted(int(x) for x in df["stack_bb"].dropna().unique().tolist())

        # key: (stack, ctx, hero, ip, oop)
        self.idx: Dict[Tuple[int, str, str, str, str], List[dict]] = {}
        for _, r in df.iterrows():
            key = (int(r["stack_bb"]), str(r["ctx"]), str(r["hero_pos"]), str(r["ip_pos"]), str(r["oop_pos"]))
            self.idx.setdefault(key, []).append({
                "rel_path": str(r["rel_path"]),
                "abs_path": str(r["abs_path"]),
            })

    def _pick(self, stack: int, ctx: str, hero: str, ip: str, oop: str) -> Optional[dict]:
        key = (int(stack), str(ctx).upper(), str(hero), canon_pos(ip), canon_pos(oop))
        rows = self.idx.get(key)
        return rows[0] if rows else None

    def _resolve_path(self, row: dict) -> Path:
        # 1) Absolute
        p_abs = Path(row.get("abs_path") or "")
        if p_abs and p_abs.exists():
            return p_abs

        # 2) Cache dir
        rel = str(row["rel_path"]).lstrip("/").replace("\\", "/")
        p_local = self.cache_dir / rel
        if p_local.exists():
            return p_local

        # 3) S3 → cache
        if self.s3_vendor and self.s3:
            # Ensure rel_path under 'sph/...'
            rel_sph = _ensure_subdir(rel, "sph")
            s3_key = _join_s3(self.s3_vendor, rel_sph)
            p_local.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file_if_missing(s3_key, p_local)
            if p_local.exists():
                return p_local

        raise RuntimeError(f"SPH file not found locally or on S3: {row}")

def _hand_to_index(code: str) -> int:
    code = code.strip()
    if len(code) == 2:  # pair: "AA"
        r = RANKS.index(code[0])
        return r * 13 + r
    if len(code) == 3:  # "AKs" / "AKo"
        i = RANKS.index(code[0]); j = RANKS.index(code[1])
        if code[2] == "s":   # suited = upper triangle (row-major)
            return i * 13 + j
        elif code[2] == "o": # offsuit = lower triangle
            return j * 13 + i
    raise ValueError(f"Bad hand code: {code}")

def _vec_from_monker_string(txt: str) -> np.ndarray:
    vals = np.zeros(169, dtype=np.float32)
    for tok in re.split(r"[,\s]+", txt.strip()):
        if not tok or ":" not in tok:
            continue
        hand, v = tok.split(":", 1)
        vals[_hand_to_index(hand)] = float(v)
    return np.clip(vals, 0.0, 1.0)

def _vec_from_csv(path: Path) -> np.ndarray:
    """
    Read either:
      - numeric CSV (13x13 or 169 numbers), or
      - Monker 'CARD:VALUE' string accidentally saved with .csv extension.
    """
    txt = path.read_text(encoding="utf-8").strip()
    # If it looks like Monker string, parse that directly
    if ":" in txt and any(h in txt for h in ("AA", "AKs", "72o")):
        return _vec_from_monker_string(txt)

    # Otherwise treat as numeric CSV (allow commas/whitespace/%; 13x13 or 169)
    rows = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[,;\s]+", line) if p != ""]
        rows.extend(parts)

    nums = []
    for x in rows:
        if x.endswith("%"):
            nums.append(float(x[:-1]) / 100.0)
        else:
            nums.append(float(x))
    arr = np.array(nums, dtype=np.float32)

    if arr.size == 169:
        return np.clip(arr.reshape(169), 0.0, 1.0)
    if arr.size == 13 * 13:
        return np.clip(arr.reshape(13, 13).reshape(169), 0.0, 1.0)

    raise ValueError(f"CSV at {path} is not 169/13x13 (got {arr.size})")

def _to_169_list(arr: np.ndarray) -> str:
    arr = np.clip(arr.astype(np.float32).reshape(169), 0.0, 1.0)
    return json.dumps([float(v) for v in arr.tolist()])

def load_sph_range_compact(path: Path, *, pick: str | None = None) -> str:
    """
    Returns JSON string of 169 floats in [0,1].
    Supports:
      - Monker CSV (13x13 or 169 values): ip.csv / oop.csv
      - Monker string: "AA:1.0,AKs:0.5,..."
      - JSON list (169 or 13x13)
      - JSON dict with keys: ip/oop/range/matrix/grid/weights/data
    If JSON has both 'ip' and 'oop', pass pick='ip' or pick='oop'.
    """
    p = Path(path)

    # 1) CSV path (our packed SPH files)
    if p.suffix.lower() == ".csv":
        return _to_169_list(_vec_from_csv(p))

    # 2) Try JSON payloads
    txt = p.read_text(encoding="utf-8").strip()
    try:
        obj = json.loads(txt)
        # bare list
        if isinstance(obj, list):
            arr = np.array(obj, dtype=np.float32)
            if arr.size == 169:
                return _to_169_list(arr)
            if arr.size == 13*13:
                return _to_169_list(arr.reshape(13,13))
            raise ValueError(f"SPH JSON list wrong length at {path}: {arr.size}")
        # dict
        if isinstance(obj, dict):
            if pick in {"ip","oop"} and pick in obj:
                return _to_169_list(np.array(obj[pick], dtype=np.float32))
            for k in ("range","grid","matrix","weights","data"):
                if k in obj:
                    return _to_169_list(np.array(obj[k], dtype=np.float32))
            if "ip" in obj and "oop" in obj:
                raise ValueError(f"SPH file has both 'ip' and 'oop' at {path}; pass pick='ip' or 'oop'.")
    except Exception:
        pass  # not JSON

    # 3) Monker string (CARD:VALUE,...)
    if ":" in txt and any(h in txt for h in ("AA","AKs","72o")):
        return _to_169_list(_vec_from_monker_string(txt))

    # 4) Raw numbers 169 fallback
    nums = [w for w in re.split(r"[,\s]+", txt) if w]
    try:
        arr = np.array([float(x[:-1])/100.0 if x.endswith("%") else float(x) for x in nums], dtype=np.float32)
        if arr.size == 169:
            return _to_169_list(arr)
        if arr.size == 13*13:
            return _to_169_list(arr.reshape(13,13))
    except Exception:
        pass

    raise ValueError(f"Unrecognized SPH format at {path}")


# ---------- Monker + SPH unified lookup ----------
class PreflopRangeLookup:
    """
    Resolve preflop ranges with dual-source fallback.

    Sources:
      - Monker manifest (rich vendor pack; currently SRP-only in index)
      - SPH manifest (your 72 solves; allows ctx filter)

    Fallback order:
      1) Monker exact      (stack, ip, oop; ctx ignored for now)
      2) Monker nearest    (nearest stack)
      3) SPH exact         (stack, ctx, ip, oop)
      4) SPH nearest       (nearest stack within ctx)
      5) (optional) opener substitution within same OOP

    Returns (rng_ip, rng_oop, meta) or raises if strict.
    """

    def __init__(
        self,
        monker_manifest_parquet: str | Path,
        *,
        sph_manifest_parquet: str | Path | None = None,
        s3_client: Optional[S3Client] = None,
        s3_vendor: Optional[str] = None,
        cache_dir: str = "data/vendor_cache",
        allow_pair_subs: bool = False,
        max_stack_delta: Optional[int] = None,
    ):
        # -------- Monker --------
        df = pd.read_parquet(str(monker_manifest_parquet)).copy()
        need = {"stack_bb","hero_pos","sequence_raw","rel_path","abs_path"}
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise RuntimeError(f"monker_manifest missing columns: {missing}")

        df["hero_pos"] = df["hero_pos"].map(canon_pos)
        df["stack_bb"] = df["stack_bb"].astype("Int64")
        self.monker_df = df
        self.monker_stacks: list[int] = sorted(int(x) for x in df["stack_bb"].dropna().unique().tolist())

        self.s3 = s3_client
        self.s3_vendor = (s3_vendor or "").strip("/") if s3_vendor else None
        self.cache_dir = Path(cache_dir)
        self.allow_pair_subs = bool(allow_pair_subs)
        self.max_stack_delta = max_stack_delta if (max_stack_delta is None or max_stack_delta >= 0) else None

        # Build monker SRP-only index (as before)
        self._monker_idx: Dict[Tuple[int, str, str, str], List[dict]] = {}
        print("vendor key is", self.s3_vendor)
        for _, r in df.iterrows():
            try:
                seq_raw = json.loads(r["sequence_raw"])
            except Exception:
                continue
            if not isinstance(seq_raw, list) or len(seq_raw) < 2:
                continue

            opener_pos_raw, _ = first_non_fold_opener(seq_raw)
            if not opener_pos_raw:
                continue

            seen_positions = {e.get("pos") for e in seq_raw if e.get("pos")}
            hero = canon_pos(str(r["hero_pos"]))
            stack = int(r["stack_bb"])
            for oop_pos_raw in seen_positions:
                ip_pos = canon_pos(opener_pos_raw)
                oop_pos = canon_pos(oop_pos_raw)
                if not ip_pos or not oop_pos or ip_pos == oop_pos:
                    continue
                if not is_srp_open_call(seq_raw, ip_pos, oop_pos):
                    continue
                if hero not in (ip_pos, oop_pos):
                    continue
                key = (stack, hero, ip_pos, oop_pos)
                self._monker_idx.setdefault(key, []).append({
                    "rel_path": r["rel_path"],
                    "abs_path": r["abs_path"],
                })
        self._seen_pairs = {(k[2], k[3]) for k in self._monker_idx.keys()}

        # -------- SPH (optional) --------
        self.sph: Optional[SphIndex] = None
        if sph_manifest_parquet:
            self.sph = SphIndex(sph_manifest_parquet, cache_dir=cache_dir) if sph_manifest_parquet and Path(sph_manifest_parquet).exists() else None

        # Union of available stacks for nearest-fallback ordering
        self.stacks: list[int] = sorted({
            *self.monker_stacks,
            *(self.sph.stacks if self.sph else []),
        })

    def _resolve_monker_path(self, row: dict) -> Path:
        abs_path = Path(row["abs_path"])
        if abs_path.exists():
            return abs_path

        rel_path = Path(row["rel_path"])
        cache_path = self.cache_dir / rel_path
        if cache_path.exists():
            return cache_path

        if not (self.s3 and self.s3_vendor):
            raise RuntimeError(f"Vendor file not found locally: {rel_path}")

        # Build S3 key from base vendor folder + monker subdir + rel_path (dedup if needed)
        rel_monker = _ensure_subdir(str(rel_path), "monker")
        s3_key = _join_s3(self.s3_vendor, rel_monker)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file_if_missing(s3_key, cache_path)
        if not cache_path.exists():
            raise RuntimeError(f"Failed to materialize vendor file: {s3_key}")
        return cache_path

    # ---------- Monker index access ----------
    def _monker_pick(self, stack: int, hero: str, ip: str, oop: str) -> Optional[dict]:
        key = (int(stack), canon_pos(hero), canon_pos(ip), canon_pos(oop))
        rows = self._monker_idx.get(key)
        return rows[0] if rows else None

    def _ordered_stacks(self, target: int) -> List[int]:
        return sorted(self.stacks, key=lambda s: (abs(s - int(target)), s))

    def ranges_for_pair(
            self,
            *,
            stack_bb: float,
            ip: str,
            oop: str,
            ctx: str = "SRP",
            strict: bool = True,
    ) -> Tuple[Optional[str], Optional[str], Dict[str, object]]:
        """
        Unified vendor lookup with smart fallbacks:
          - SRP: try Monker first (exact/nearest), then SPH (exact/nearest).
          - LIMP_SINGLE/LIMP_MULTI: try SPH first (exact/nearest); Monker as a last resort.
        Pair substitution: controlled by self.allow_pair_subs via candidate_pairs(...).
        Returns (rng_ip_json, rng_oop_json, meta).
        """
        ip = canon_pos(ip)
        oop = canon_pos(oop)
        ctx = str(ctx).upper()
        near_stack = nearest_stack(stack_bb, self.stacks)

        # Source priority by context
        # - SRP: Monker first
        # - Limped: SPH first (Monker typically has no limped coverage)
        if ctx == "SRP":
            source_order = ("monker", "sph")
        else:
            source_order = ("sph", "monker")

        last_meta: Dict[str, Any] = {}
        for cand_ip, cand_oop, level, substituted in candidate_pairs(ip, oop, ctx=ctx,
                                                                     allow_pair_subs=self.allow_pair_subs):
            # Prepare base meta for this candidate
            base_meta = {
                "source": None,  # "monker" or "sph"
                "ctx": ctx,
                "range_pair_substituted": bool(substituted),
                "range_ip_source_pair": f"{cand_ip}v{cand_oop}",
                "range_oop_source_pair": f"{cand_ip}v{cand_oop}",
                "range_ip_source_stack": None,
                "range_oop_source_stack": None,
                "range_ip_stack_delta": None,
                "range_oop_stack_delta": None,
                "range_ip_fallback_level": level,
                "range_oop_fallback_level": level,
            }

            # Within a candidate pair, scan exact→nearest stacks
            for s in self._ordered_stacks(int(near_stack)):
                delta = abs(s - int(near_stack))
                if self.max_stack_delta is not None and delta > self.max_stack_delta:
                    continue

                # Try sources in chosen priority order
                for source in source_order:
                    meta = dict(base_meta)
                    if source == "sph" and self.sph is not None:
                        row_ip = self.sph._pick(s, ctx, hero=cand_ip, ip=cand_ip, oop=cand_oop)
                        row_oop = self.sph._pick(s, ctx, hero=cand_oop, ip=cand_ip, oop=cand_oop)
                        if row_ip and row_oop:
                            p_ip = self.sph._resolve_path(row_ip)
                            p_oop = self.sph._resolve_path(row_oop)
                            # SPH canonical compact: explicitly select side
                            rng_ip = load_sph_range_compact(p_ip, pick="ip")
                            rng_oop = load_sph_range_compact(p_oop, pick="oop")
                            meta.update({
                                "source": "sph",
                                "range_ip_source_stack": s, "range_oop_source_stack": s,
                                "range_ip_stack_delta": delta, "range_oop_stack_delta": delta,
                            })
                            return rng_ip, rng_oop, meta

                    if source == "monker":
                        row_ip = self._monker_pick(s, hero=cand_ip, ip=cand_ip, oop=cand_oop)
                        row_oop = self._monker_pick(s, hero=cand_oop, ip=cand_ip, oop=cand_oop)
                        if row_ip and row_oop:
                            p_ip = self._resolve_monker_path(row_ip)
                            p_oop = self._resolve_monker_path(row_oop)
                            rng_ip = _load_vendor_range_compact(p_ip)  # already canonical 169-float JSON
                            rng_oop = _load_vendor_range_compact(p_oop)
                            meta.update({
                                "source": "monker",
                                "range_ip_source_stack": s, "range_oop_source_stack": s,
                                "range_ip_stack_delta": delta, "range_oop_stack_delta": delta,
                            })
                            return rng_ip, rng_oop, meta

                # keep best-so-far diagnostics (nearest attempt) if we fail
                last_meta = {**base_meta, "range_ip_source_stack": s, "range_oop_source_stack": s,
                             "range_ip_stack_delta": delta, "range_oop_stack_delta": delta}

        # Nothing found
        if strict:
            raise RuntimeError(f"No ranges found for {ip}v{oop}@{stack_bb} ctx={ctx}")
        return None, None, last_meta