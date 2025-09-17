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
    def __init__(self, manifest_parquet: str | Path, cache_dir: str | Path = "data/vendor_cache",
                 *, s3_vendor: str | None = None, s3_client: "S3Client | None" = None):
        df = pd.read_parquet(str(manifest_parquet)).copy()
        need = {"stack_bb","ip_pos","oop_pos","ctx","hero_pos","rel_path","abs_path"}
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise RuntimeError(f"sph_manifest missing columns: {missing}")

        # normalize
        df["ip_pos"]   = df["ip_pos"].map(canon_pos)
        df["oop_pos"]  = df["oop_pos"].map(canon_pos)
        df["hero_pos"] = df["hero_pos"].map(lambda p: canon_pos(p) or str(p).upper())
        df["ctx"]      = df["ctx"].map(lambda s: str(s).upper())
        df["stack_bb"] = pd.to_numeric(df["stack_bb"], errors="coerce").astype("Int64")

        self.df = df

        # expose stacks for union in PreflopRangeLookup
        self.stacks: list[int] = sorted(int(x) for x in df["stack_bb"].dropna().unique().tolist())

        # (optional) quick diags
        self.ctxs  = sorted(df["ctx"].dropna().unique().tolist())
        self.pairs = sorted({(str(r.ip_pos), str(r.oop_pos)) for _, r in df[["ip_pos","oop_pos"]].dropna().iterrows()})

        # primary cache root + fallback
        self.cache_dir = Path(cache_dir)
        self.cache_fallback = Path("data/vendor_cache")
        self.s3_vendor = (s3_vendor or "").rstrip("/") if s3_vendor else None
        self.s3 = s3_client

        # build index
        self.idx: Dict[Tuple[int,str,str,str,str], List[dict]] = {}
        for _, r in df.iterrows():
            self.idx.setdefault(
                (int(r["stack_bb"]), str(r["ctx"]), str(r["hero_pos"]), str(r["ip_pos"]), str(r["oop_pos"])), []
            ).append({
                "rel_path": str(r["rel_path"]),
                "abs_path": str(r["abs_path"]) if pd.notna(r["abs_path"]) else "",
            })

    def _pick(self, stack: int, ctx: str, hero: str, ip: str, oop: str) -> Optional[dict]:
        return (self.idx.get((int(stack), str(ctx).upper(), str(hero), canon_pos(ip), canon_pos(oop))) or [None])[0]

    def _resolve_path(self, row: dict) -> Path:
        # 1) absolute path if it’s a real string path
        abs_val = row.get("abs_path")
        abs_str = abs_val if isinstance(abs_val, str) else ""
        if abs_str:
            p_abs = Path(abs_str)
            if p_abs.is_file():
                return p_abs

        # 2) try both cache roots with rel_path
        rel = str(row["rel_path"]).lstrip("/").replace("\\", "/")
        for root in (self.cache_dir, self.cache_fallback):
            p_local = root / rel
            if p_local.is_file():
                return p_local

        # 3) S3 → cache (under the primary cache_dir)
        if self.s3_vendor and self.s3:
            rel_sph = _ensure_subdir(rel, "sph")
            s3_key = _join_s3(self.s3_vendor, rel_sph)
            p_local = self.cache_dir / rel
            p_local.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file_if_missing(s3_key, p_local)
            if p_local.is_file():
                return p_local

        raise RuntimeError(f"SPH file not found locally or on S3: rel={row.get('rel_path')} abs={row.get('abs_path')}")

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
    # -------- Context normalization (public labels -> canonical) --------
    CTX_ALIAS = {
        # SRP-like contexts all normalize to SRP
        "OPEN": "SRP",
        "VS_OPEN": "SRP",
        "VS_OPEN_RFI": "SRP",
        "BLIND_VS_STEAL": "SRP",
        "SRP": "SRP",
        # higher-raise trees
        "3BET": "VS_3BET",
        "VS_3BET": "VS_3BET",
        "4BET": "VS_4BET",
        "VS_4BET": "VS_4BET",
        # limp variants (kept distinct)
        "LIMPED_SINGLE": "LIMPED_SINGLE",
        "LIMP_SINGLE": "LIMPED_SINGLE",
        "LIMPED_MULTI": "LIMPED_MULTI",
        "LIMP_MULTI": "LIMPED_MULTI",
    }

    def __init__(
        self,
        monker_manifest_parquet: str | Path,
        *,
        sph_manifest_parquet: str | Path | None = None,   # accepted for backward compat; ignored now
        s3_client: Optional["S3Client"] = None,
        s3_vendor: Optional[str] = None,
        cache_dir: str = "data/vendor_cache",
        allow_pair_subs: bool = False,
        max_stack_delta: Optional[int] = None,
    ):
        # ---- Load & normalize manifest (new/old schema tolerant) ----
        df = pd.read_parquet(str(monker_manifest_parquet)).copy()

        # 1) Map new-schema column names to legacy names this class uses
        if "ip_pos" not in df.columns and "ip_actor_flop" in df.columns:
            df["ip_pos"] = df["ip_actor_flop"]
        if "oop_pos" not in df.columns and "oop_actor_flop" in df.columns:
            df["oop_pos"] = df["oop_actor_flop"]

        # 2) hero_pos: explicit -> hint -> derive from rel_path parent
        if "hero_pos" not in df.columns:
            if "hero_pos_hint" in df.columns:
                df["hero_pos"] = df["hero_pos_hint"]
            elif "rel_path" in df.columns:
                def _hero_from_rel(p):
                    try:
                        return Path(str(p)).parent.name  # .../<HERO>/<file>.txt
                    except Exception:
                        return None
                df["hero_pos"] = df["rel_path"].map(_hero_from_rel)
            else:
                df["hero_pos"] = None

        # 3) ctx: derive from topology if missing
        if "ctx" not in df.columns:
            topo2ctx = {
                "srp_hu": "SRP", "srp_multi": "SRP",
                "3bet_hu": "VS_3BET", "3bet_multi": "VS_3BET",
                "4bet_hu": "VS_4BET", "4bet_multi": "VS_4BET",
                "limped_single": "LIMPED_SINGLE",
                "limped_multi":  "LIMPED_MULTI",
            }
            if "topology" in df.columns:
                df["ctx"] = df["topology"].map(lambda t: topo2ctx.get(str(t).lower(), None))
            else:
                df["ctx"] = None

        # 4) ensure path columns exist
        if "rel_path" not in df.columns: df["rel_path"] = None
        if "abs_path" not in df.columns: df["abs_path"] = None

        # 5) canon + types
        for c in ("hero_pos", "ip_pos", "oop_pos"):
            if c in df.columns:
                df[c] = df[c].map(canon_pos)
        if "ctx" in df.columns:
            df["ctx"] = df["ctx"].map(lambda s: str(s).upper() if s is not None else None)
        if "stack_bb" in df.columns:
            df["stack_bb"] = pd.to_numeric(df["stack_bb"], errors="coerce").astype("Int64")

        # Keep only usable HU rows with all key fields present
        df = df[
            df["ctx"].notna()
            & df["stack_bb"].notna()
            & df["hero_pos"].notna()
            & df["ip_pos"].notna()
            & df["oop_pos"].notna()
        ].copy()

        # Save & props
        self.monker_df = df
        self.monker_stacks: List[int] = sorted(int(s) for s in df["stack_bb"].dropna().unique())

        # IO knobs
        self.s3: Optional["S3Client"] = s3_client
        self.s3_vendor = (s3_vendor or "").strip("/") if s3_vendor else None
        self.cache_dir = Path(cache_dir)
        self.allow_pair_subs = bool(allow_pair_subs)
        self.max_stack_delta = max_stack_delta if (max_stack_delta is None or max_stack_delta >= 0) else None

        # ---- SPH (optional; used for limped ctx) ----
        self.sph: Optional[SphIndex] = None
        if sph_manifest_parquet and Path(sph_manifest_parquet).exists():
            self.sph = SphIndex(
                manifest_parquet=sph_manifest_parquet,
                cache_dir=self.cache_dir,  # use same cache root
                s3_vendor=(self.s3_vendor or "data/vendor"),
                s3_client=S3Client(),
            )

        # union stacks so nearest-stack search can land on SPH-only stacks
        if self.sph is not None:
            self.stacks = sorted({*self.monker_stacks, *self.sph.stacks})
        else:
            self.stacks = list(self.monker_stacks)

        # ---- Build indices ----
        # Context-aware index: (stack, ctx, hero, ip, oop)
        self._monker_idx_ctx: Dict[Tuple[int, str, str, str, str], List[dict]] = {}
        # SRP alias index: (stack, hero, ip, oop)
        self._monker_idx_srp: Dict[Tuple[int, str, str, str], List[dict]] = {}
        # NEW: pair-level (ignores hero)
        # (stack, ctx, ip, oop)
        self._monker_idx_ctx_pair: Dict[Tuple[int, str, str, str], List[dict]] = {}
        # (stack, ip, oop) for SRP alias
        self._monker_idx_srp_pair: Dict[Tuple[int, str, str], List[dict]] = {}

        for _, r in df.iterrows():
            st   = int(r["stack_bb"])
            ctx  = str(r["ctx"])
            hero = str(r["hero_pos"])
            ip   = str(r["ip_pos"])
            oop  = str(r["oop_pos"])

            # full (includes hero)
            k_ctx_full = (st, ctx, hero, ip, oop)
            self._monker_idx_ctx.setdefault(k_ctx_full, []).append({
                "rel_path": str(r["rel_path"]) if pd.notna(r["rel_path"]) else "",
                "abs_path": str(r["abs_path"])  if pd.notna(r["abs_path"])  else "",
                "hero": hero, "ip": ip, "oop": oop, "ctx": ctx, "stack": st
            })

            # pair-level (ignores hero)
            k_ctx_pair = (st, ctx, ip, oop)
            self._monker_idx_ctx_pair.setdefault(k_ctx_pair, []).append({
                "rel_path": str(r["rel_path"]) if pd.notna(r["rel_path"]) else "",
                "abs_path": str(r["abs_path"])  if pd.notna(r["abs_path"])  else "",
                "hero": hero, "ip": ip, "oop": oop, "ctx": ctx, "stack": st
            })

            if ctx == "SRP":
                k_srp_full = (st, hero, ip, oop)
                self._monker_idx_srp.setdefault(k_srp_full, []).append({
                    "rel_path": str(r["rel_path"]) if pd.notna(r["rel_path"]) else "",
                    "abs_path": str(r["abs_path"])  if pd.notna(r["abs_path"])  else "",
                    "hero": hero, "ip": ip, "oop": oop, "ctx": ctx, "stack": st
                })
                k_srp_pair = (st, ip, oop)
                self._monker_idx_srp_pair.setdefault(k_srp_pair, []).append({
                    "rel_path": str(r["rel_path"]) if pd.notna(r["rel_path"]) else "",
                    "abs_path": str(r["abs_path"])  if pd.notna(r["abs_path"])  else "",
                    "hero": hero, "ip": ip, "oop": oop, "ctx": ctx, "stack": st
                })


        # quick diag set
        self._seen_pairs_ctx = {(k[1], k[3], k[4]) for k in self._monker_idx_ctx.keys()}
    # ---------- path resolution ----------
    def _resolve_monker_path(self, row: dict) -> Path:
        abs_path = Path(row.get("abs_path") or "")
        if str(abs_path) and abs_path.exists():
            return abs_path

        rel_path_str = row.get("rel_path") or ""
        if not rel_path_str:
            raise RuntimeError("Missing rel_path/abs_path for vendor file")

        rel_path = Path(rel_path_str)
        cache_path = self.cache_dir / rel_path
        if cache_path.exists():
            return cache_path

        if not (self.s3 and self.s3_vendor):
            raise RuntimeError(f"Vendor file not found locally: {rel_path}")

        rel_monker = _ensure_subdir(str(rel_path), "monker")
        s3_key = _join_s3(self.s3_vendor, rel_monker)
        cache_path = self.cache_dir / rel_monker
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file_if_missing(s3_key, cache_path)
        if not cache_path.exists():
            raise RuntimeError(f"Failed to materialize vendor file: {s3_key}")
        return cache_path

    # ---------- small helpers ----------
    def _canon_ctx(self, ctx: str) -> str:
        c = str(ctx).upper()
        return type(self).CTX_ALIAS.get(c, c)

    def _canon_ctx_sph(self, ctx: str) -> str:
        c = str(ctx).upper()
        if c == "LIMPED_SINGLE": return "LIMP_SINGLE"
        if c == "LIMPED_MULTI":  return "LIMP_MULTI"
        return c

    def _monker_pick_ctx(self, stack: int, ctx: str, hero: str, ip: str, oop: str) -> Optional[dict]:
        key = (int(stack), self._canon_ctx(ctx), canon_pos(hero), canon_pos(ip), canon_pos(oop))
        rows = self._monker_idx_ctx.get(key)
        return rows[0] if rows else None

    def _monker_pick_srp(self, stack: int, hero: str, ip: str, oop: str) -> Optional[dict]:
        key = (int(stack), canon_pos(hero), canon_pos(ip), canon_pos(oop))
        rows = self._monker_idx_srp.get(key)
        return rows[0] if rows else None

    # NEW: choose best two rows (prefer hero==ip and hero==oop)
    def _pick_best_two(self, rows: List[dict], ip: str, oop: str) -> Tuple[Optional[dict], Optional[dict]]:
        if not rows:
            return None, None
        ip_row  = next((r for r in rows if r["hero"] == ip), None)
        oop_row = next((r for r in rows if r["hero"] == oop), None)
        if ip_row is None:
            ip_row = rows[0]
        if oop_row is None:
            oop_row = rows[1] if len(rows) > 1 else rows[0]
        return ip_row, oop_row

    def _monker_pick_ctx_pair(self, stack: int, ctx: str, ip: str, oop: str) -> Tuple[Optional[dict], Optional[dict]]:
        key = (int(stack), self._canon_ctx(ctx), canon_pos(ip), canon_pos(oop))
        rows = self._monker_idx_ctx_pair.get(key, [])
        return self._pick_best_two(rows, canon_pos(ip), canon_pos(oop))

    def _monker_pick_srp_pair(self, stack: int, ip: str, oop: str) -> Tuple[Optional[dict], Optional[dict]]:
        key = (int(stack), canon_pos(ip), canon_pos(oop))
        rows = self._monker_idx_srp_pair.get(key, [])
        return self._pick_best_two(rows, canon_pos(ip), canon_pos(oop))

    def _ordered_stacks(self, target: int) -> List[int]:
        return sorted(self.stacks, key=lambda s: (abs(s - int(target)), s))

    # ---------- public API ----------
    def ranges_for_pair(
        self,
        *,
        stack_bb: float,
        ip: str,
        oop: str,
        ctx: str = "SRP",
        strict: bool = True,
    ) -> Tuple[str, str, Dict[str, object]]:
        """
        Returns (range_ip_json169, range_oop_json169, meta).
        range_* are JSON-serialized 169-length lists (vendor-compact converted).
        meta['source'] ∈ {'monker:<ctx>','monker:SRP-fallback','fallback:default'}.
        """
        ip = canon_pos(ip)
        oop = canon_pos(oop)
        if not ip or not oop or ip == oop:
            raise RuntimeError(f"Bad positions ip={ip} oop={oop}")

        ctx = self._canon_ctx(ctx)
        near_stack = nearest_stack(stack_bb, self.stacks)

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
                if self.max_stack_delta is not None and abs(s - int(near_stack)) > self.max_stack_delta:
                    continue
                delta = abs(s - int(near_stack))

                # 1) exact-context (requires both heroes present)
                row_ip  = self._monker_pick_ctx(s, ctx, hero=cand_ip,  ip=cand_ip,  oop=cand_oop)
                row_oop = self._monker_pick_ctx(s, ctx, hero=cand_oop, ip=cand_ip,  oop=cand_oop)
                if row_ip and row_oop:
                    p_ip, p_oop = self._resolve_monker_path(row_ip), self._resolve_monker_path(row_oop)
                    rng_ip, rng_oop = _load_vendor_range_compact(p_ip), _load_vendor_range_compact(p_oop)
                    meta = {**base_meta, "source": f"monker:{ctx}",
                            "range_ip_source_stack": s, "range_oop_source_stack": s,
                            "range_ip_stack_delta": delta, "range_oop_stack_delta": delta,
                            "monker_ip_path": str(p_ip), "monker_oop_path": str(p_oop)}
                    return rng_ip, rng_oop, meta

                # 1b) NEW: pair-level for this ctx (ignore hero; pick best available heroes)
                row_ip2, row_oop2 = self._monker_pick_ctx_pair(s, ctx, ip=cand_ip, oop=cand_oop)
                if row_ip2 and row_oop2:
                    p_ip, p_oop = self._resolve_monker_path(row_ip2), self._resolve_monker_path(row_oop2)
                    rng_ip, rng_oop = _load_vendor_range_compact(p_ip), _load_vendor_range_compact(p_oop)
                    meta = {**base_meta, "source": f"monker:{ctx}",
                            "range_ip_source_stack": s, "range_oop_source_stack": s,
                            "range_ip_stack_delta": delta, "range_oop_stack_delta": delta,
                            "monker_ip_path": str(p_ip), "monker_oop_path": str(p_oop)}
                    return rng_ip, rng_oop, meta

                # 2) SRP fallback for SRP-like contexts (strict hero)
                if ctx == "SRP":
                    row_ip  = self._monker_pick_srp(s, hero=cand_ip,  ip=cand_ip,  oop=cand_oop)
                    row_oop = self._monker_pick_srp(s, hero=cand_oop, ip=cand_ip,  oop=cand_oop)
                    if row_ip and row_oop:
                        p_ip, p_oop = self._resolve_monker_path(row_ip), self._resolve_monker_path(row_oop)
                        rng_ip, rng_oop = _load_vendor_range_compact(p_ip), _load_vendor_range_compact(p_oop)
                        meta = {**base_meta, "source": "monker:SRP-fallback",
                                "range_ip_source_stack": s, "range_oop_source_stack": s,
                                "range_ip_stack_delta": delta, "range_oop_stack_delta": delta,
                                "monker_ip_path": str(p_ip), "monker_oop_path": str(p_oop)}
                        return rng_ip, rng_oop, meta

                    # 2b) NEW: SRP pair-level alias
                    row_ip2, row_oop2 = self._monker_pick_srp_pair(s, ip=cand_ip, oop=cand_oop)
                    if row_ip2 and row_oop2:
                        p_ip, p_oop = self._resolve_monker_path(row_ip2), self._resolve_monker_path(row_oop2)
                        rng_ip, rng_oop = _load_vendor_range_compact(p_ip), _load_vendor_range_compact(p_oop)
                        meta = {**base_meta, "source": "monker:SRP-fallback",
                                "range_ip_source_stack": s, "range_oop_source_stack": s,
                                "range_ip_stack_delta": delta, "range_oop_stack_delta": delta,
                                "monker_ip_path": str(p_ip), "monker_oop_path": str(p_oop)}
                        return rng_ip, rng_oop, meta

                # SPH fallback for limped contexts
                if self.sph is not None and ctx in {"LIMPED_SINGLE", "LIMPED_MULTI"}:
                    ctx_sph = self._canon_ctx_sph(ctx)
                    row_ip = self.sph._pick(s, ctx_sph, hero="IP", ip=cand_ip, oop=cand_oop)
                    row_oop = self.sph._pick(s, ctx_sph, hero="OOP", ip=cand_ip, oop=cand_oop)
                    if row_ip and row_oop:
                        p_ip = self.sph._resolve_path(row_ip)
                        p_oop = self.sph._resolve_path(row_oop)
                        rng_ip = load_sph_range_compact(p_ip, pick="ip")
                        rng_oop = load_sph_range_compact(p_oop, pick="oop")
                        meta = {**base_meta, "source": f"sph:{ctx_sph}",
                                "range_ip_source_stack": s, "range_oop_source_stack": s,
                                "range_ip_stack_delta": delta, "range_oop_stack_delta": delta,
                                "sph_ip_path": str(p_ip), "sph_oop_path": str(p_oop)}
                        return rng_ip, rng_oop, meta

        # 3) last-resort flat 169 distribution (keeps pipeline moving)
        flat_range = json.dumps([1.0] * 169)
        meta = {
            "source": "fallback:default",
            "ctx": ctx,
            "range_pair_substituted": False,
            "range_ip_source_pair": f"{ip}v{oop}",
            "range_oop_source_pair": f"{ip}v{oop}",
            "range_ip_source_stack": near_stack,
            "range_oop_source_stack": near_stack,
            "range_ip_stack_delta": 0,
            "range_oop_stack_delta": 0,
        }
        if strict and not flat_range:
            raise RuntimeError(f"No ranges found for {ip}v{oop}@{stack_bb} ctx={ctx}")
        return flat_range, flat_range, meta