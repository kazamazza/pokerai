from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json, os

import numpy as np
import pandas as pd

from infra.storage.s3_client import S3Client
from ml.etl.rangenet.candidate_pairs import candidate_pairs
from ml.etl.rangenet.preflop.monker_helpers import canon_pos, first_non_fold_opener
from ml.etl.utils.range_lookup import is_srp_open_call, _load_vendor_range_compact, nearest_stack
from ml.range.solvers.utils.range_utils import hand_to_index

def _join_s3(*parts: str) -> str:
    return "/".join(str(p).strip("/").replace("\\", "/") for p in parts if p is not None and str(p) != "")

def _ensure_subdir(rel_path: str, subdir: str) -> str:
    """Ensure rel_path is rooted under subdir (monker|sph) without duplicating it."""
    rel = str(rel_path).lstrip("/").replace("\\", "/")
    sub = subdir.strip("/")
    return rel if rel.startswith(sub + "/") else f"{sub}/{rel}"

class SphIndex:
    def __init__(
        self,
        manifest_parquet: str | Path,
        cache_dir: str = "data/vendor_cache",
        *,
        s3_vendor: str | None = None,
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
        # 1) Absolute: only if non-empty and is a file
        abs_str = (row.get("abs_path") or "").strip()
        if abs_str:
            p_abs = Path(abs_str)
            if p_abs.exists() and p_abs.is_file():
                return p_abs

        # 2) Cache dir + rel_path: only if is a file
        rel = str(row["rel_path"]).lstrip("/").replace("\\", "/")
        p_local = self.cache_dir / rel
        if p_local.exists() and p_local.is_file():
            return p_local
        print("vendor folder is", self.s3_vendor)
        # 3) S3 → cache
        if self.s3_vendor and self.s3:
            rel_sph = _ensure_subdir(rel, "sph")  # keep your helper
            s3_key = _join_s3(self.s3_vendor, rel_sph)
            p_local.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file_if_missing(s3_key, p_local)
            if p_local.exists() and p_local.is_file():
                return p_local

        raise RuntimeError(f"SPH file not found locally or on S3: {row}")

def _monker_to_vec169(monker_str: str) -> list[float]:
    vals = [0.0]*169
    for tok in re.split(r"[,\s]+", monker_str.strip()):
        if not tok or ":" not in tok: continue
        hand, v = tok.split(":", 1)
        vals[hand_to_index(hand.strip())] = float(v)
    return vals

def load_sph_range_compact(path: Path, *, pick: str|None=None) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    # 1) JSON list/dict?
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            arr = np.array(obj, dtype=float)
            if arr.size == 169:
                return json.dumps(arr.reshape(169).tolist())
            if arr.shape == (13,13):
                return json.dumps(arr.reshape(169).tolist())
        if isinstance(obj, dict):
            for k in ("ip","oop","range","grid","matrix","weights","data"):
                if k in obj:
                    arr = np.array(obj[k], dtype=float)
                    if arr.size == 169:
                        return json.dumps(arr.reshape(169).tolist())
                    if arr.shape == (13,13):
                        return json.dumps(arr.reshape(169).tolist())
            raise ValueError(f"SPH JSON needs one of the known keys at {path}")
    except Exception:
        pass
    # 2) Monker string (your ip.csv/oop.csv)
    if ":" in text and any(tag in text for tag in ("AA","AKs","KQo","22","72o")):
        vec = _monker_to_vec169(text)
        return json.dumps(vec)
    # 3) 13x13 CSV of numbers
    try:
        df = pd.read_csv(path, header=None)
        arr = df.to_numpy(dtype=float)
        if arr.size == 169:
            return json.dumps(arr.reshape(169).tolist())
    except Exception:
        pass
    raise ValueError(f"Unrecognized SPH format at {path}")


class PreflopRangeLookup:
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
        if sph_manifest_parquet and Path(sph_manifest_parquet).exists():
            self.sph = SphIndex(
                manifest_parquet=sph_manifest_parquet,
                cache_dir=cache_dir,
                s3_vendor="data/vendor",  # e.g. "s3://mybucket/data/vendor_cache"
                s3_client=S3Client(),  # your existing S3 client class
            )
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
        ip = canon_pos(ip)
        oop = canon_pos(oop)
        ctx = str(ctx).upper()
        near_stack = nearest_stack(stack_bb, self.stacks)

        # SRP: prefer Monker, others: prefer SPH
        source_order = ("monker", "sph") if ctx == "SRP" else ("sph", "monker")

        last_meta: Dict[str, Any] = {}

        for cand_ip, cand_oop, level, substituted in candidate_pairs(
                ip, oop, ctx=ctx, allow_pair_subs=self.allow_pair_subs
        ):
            base_meta = {
                "source": None,
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

            for s in self._ordered_stacks(int(near_stack)):
                delta = abs(s - int(near_stack))
                if self.max_stack_delta is not None and delta > self.max_stack_delta:
                    continue

                for source in source_order:
                    meta = dict(base_meta)

                    # ---------- SPH ----------
                    if source == "sph" and self.sph is not None:
                        # IMPORTANT: SPH manifest uses hero="IP"/"OOP" (not seat names)
                        row_ip = self.sph._pick(s, ctx, hero="IP", ip=cand_ip, oop=cand_oop)
                        row_oop = self.sph._pick(s, ctx, hero="OOP", ip=cand_ip, oop=cand_oop)
                        if row_ip and row_oop:
                            p_ip = self.sph._resolve_path(row_ip)
                            p_oop = self.sph._resolve_path(row_oop)
                            rng_ip = load_sph_range_compact(p_ip, pick="ip")
                            rng_oop = load_sph_range_compact(p_oop, pick="oop")
                            meta.update({
                                "source": "sph",
                                "range_ip_source_stack": s, "range_oop_source_stack": s,
                                "range_ip_stack_delta": delta, "range_oop_stack_delta": delta,
                                "sph_ip_path": str(p_ip), "sph_oop_path": str(p_oop),
                            })
                            return rng_ip, rng_oop, meta

                    # ---------- Monker ----------
                    if source == "monker":
                        row_ip = self._monker_pick(s, hero=cand_ip, ip=cand_ip, oop=cand_oop)
                        row_oop = self._monker_pick(s, hero=cand_oop, ip=cand_ip, oop=cand_oop)
                        if row_ip and row_oop:
                            p_ip = self._resolve_monker_path(row_ip)
                            p_oop = self._resolve_monker_path(row_oop)
                            rng_ip = _load_vendor_range_compact(p_ip)
                            rng_oop = _load_vendor_range_compact(p_oop)
                            meta.update({
                                "source": "monker",
                                "range_ip_source_stack": s, "range_oop_source_stack": s,
                                "range_ip_stack_delta": delta, "range_oop_stack_delta": delta,
                                "monker_ip_path": str(p_ip), "monker_oop_path": str(p_oop),
                            })
                            return rng_ip, rng_oop, meta

                # keep diagnostics if this stack attempt failed
                last_meta = {
                    **base_meta,
                    "range_ip_source_stack": s,
                    "range_oop_source_stack": s,
                    "range_ip_stack_delta": delta,
                    "range_oop_stack_delta": delta,
                }

        if strict:
            raise RuntimeError(f"No ranges found for {ip}v{oop}@{stack_bb} ctx={ctx}")
        return None, None, last_meta