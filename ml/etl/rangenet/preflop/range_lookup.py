from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json, os
import pandas as pd

from ml.etl.rangenet.preflop.monker_helpers import canon_pos, first_non_fold_opener
from ml.etl.utils.range_lookup import is_srp_open_call, _load_vendor_range_compact, _candidate_pairs, nearest_stack


class SphIndex:
    """
    Minimal index over your Simple Preflop Hold'em exports.

    Expected columns in sph_manifest.parquet (you'll produce this with scan_sph.py):
      - stack_bb (int)
      - ip_pos (str)
      - oop_pos (str)
      - ctx (str)              # e.g. "SRP", "LIMPED_SINGLE", "LIMPED_MULTI"
      - hero_pos (str)         # "IP" or "OOP" (or the concrete pos; we accept both)
      - rel_path (str)         # relative path inside your cache dir
      - abs_path (str)         # absolute path if you keep local files

    Loader `_load_sph_range_compact(path)` must return a 169-length
    compact representation (same shape you return for Monker).
    """
    def __init__(self, manifest_parquet: str | Path, cache_dir: str = "data/vendor_cache"):
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
        self.stacks: list[int] = sorted(int(x) for x in df["stack_bb"].dropna().unique().tolist())

        # key: (stack, ctx, hero, ip, oop)
        self.idx: Dict[Tuple[int, str, str, str, str], List[dict]] = {}
        for _, r in df.iterrows():
            key = (int(r["stack_bb"]), str(r["ctx"]), str(r["hero_pos"]), str(r["ip_pos"]), str(r["oop_pos"]))
            self.idx.setdefault(key, []).append({"rel_path": r["rel_path"], "abs_path": r["abs_path"]})

    def _pick(self, stack: int, ctx: str, hero: str, ip: str, oop: str) -> Optional[dict]:
        key = (int(stack), str(ctx).upper(), str(hero), canon_pos(ip), canon_pos(oop))
        rows = self.idx.get(key)
        return rows[0] if rows else None

    def _resolve_path(self, row: dict) -> Path:
        p = Path(row["abs_path"])
        if p.exists():
            return p
        p2 = self.cache_dir / row["rel_path"]
        if p2.exists():
            return p2
        raise RuntimeError(f"SPH file not found: {row}")

def load_sph_range_compact(path: Path, *, pick: str | None = None) -> str:
    """
    Return a JSON string with 169 floats (0..1).
    If the file has {"ip":[...],"oop":[...]}, pass pick="ip" or "oop".
    If the file is a bare 169-length list, pick is ignored.
    """
    import json
    obj = json.loads(Path(path).read_text(encoding="utf-8"))

    def _as_payload(arr):
        if not isinstance(arr, list) or len(arr) != 169:
            raise ValueError(f"SPH range wrong length at {path}")
        # ensure floats
        arr = [float(x) for x in arr]
        return json.dumps(arr)

    if isinstance(obj, list):               # bare 169
        return _as_payload(obj)
    if isinstance(obj, dict):
        if pick in {"ip","oop"} and pick in obj:
            return _as_payload(obj[pick])
        # If payload_kind hinted ip/oop via filename, you can ignore 'pick'
        # or infer from your manifest row.
        # Fallbacks:
        if "range" in obj:
            return _as_payload(obj["range"])
    raise ValueError(f"Unrecognized SPH range format at {path}")


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
        s3_client: Optional["S3Client"] = None,
        s3_prefix: Optional[str] = None,
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
        self.s3_prefix = (s3_prefix or "").strip("/") if s3_prefix else None
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
        if sph_manifest_parquet:
            self.sph = SphIndex(sph_manifest_parquet, cache_dir=cache_dir)

        # Union of available stacks for nearest-fallback ordering
        self.stacks: list[int] = sorted({
            *self.monker_stacks,
            *(self.sph.stacks if self.sph else []),
        })

    # ---------- Monker path resolution ----------
    def _resolve_monker_path(self, row: dict) -> Path:
        abs_path = Path(row["abs_path"])
        if abs_path.exists():
            return abs_path
        rel_path = Path(row["rel_path"])
        cache_path = self.cache_dir / rel_path
        if cache_path.exists():
            return cache_path
        if not (self.s3 and self.s3_prefix):
            raise RuntimeError(f"Vendor file not found: {rel_path}")
        s3_key = f"{self.s3_prefix}/{str(rel_path).replace(os.sep, '/')}"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file_if_missing(s3_key, cache_path)
        if not cache_path.exists():
            raise RuntimeError(f"Failed to materialize vendor file: s3://{getattr(self.s3,'bucket','?')}/{s3_key}")
        return cache_path

    # ---------- Monker index access ----------
    def _monker_pick(self, stack: int, hero: str, ip: str, oop: str) -> Optional[dict]:
        key = (int(stack), canon_pos(hero), canon_pos(ip), canon_pos(oop))
        rows = self._monker_idx.get(key)
        return rows[0] if rows else None

    def _ordered_stacks(self, target: int) -> List[int]:
        return sorted(self.stacks, key=lambda s: (abs(s - int(target)), s))

    # ---------- API ----------
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
        ctx is currently used only for SPH (Monker index here is SRP-only).
        """
        ip, oop = canon_pos(ip), canon_pos(oop)
        ctx = str(ctx).upper()
        near_stack = nearest_stack(stack_bb, self.stacks)

        last_meta: Dict[str, Any] = {}
        # iterate exact pair and (optionally) opener-substitution candidates
        for cand_ip, cand_oop, level, substituted in _candidate_pairs(ip, oop, self.allow_pair_subs):
            # 1) Monker (exact/nearest)
            row_ip = row_oop = None
            meta = {
                "source": None,  # "monker" or "sph"
                "range_pair_substituted": substituted,
                "range_ip_source_pair": f"{cand_ip}v{cand_oop}",
                "range_oop_source_pair": f"{cand_ip}v{cand_oop}",
                "range_ip_source_stack": None,
                "range_oop_source_stack": None,
                "range_ip_stack_delta": None,
                "range_oop_stack_delta": None,
                "ctx": ctx,
            }

            for s in self._ordered_stacks(int(near_stack)):
                delta = abs(s - int(near_stack))
                if self.max_stack_delta is not None and delta > self.max_stack_delta:
                    continue

                if row_ip is None:
                    row_ip = self._monker_pick(s, hero=cand_ip, ip=cand_ip, oop=cand_oop)
                    if row_ip:
                        meta.update({"source": "monker", "range_ip_source_stack": s, "range_ip_stack_delta": delta})
                if row_oop is None:
                    row_oop = self._monker_pick(s, hero=cand_oop, ip=cand_ip, oop=cand_oop)
                    if row_oop:
                        meta.update({"source": "monker", "range_oop_source_stack": s, "range_oop_stack_delta": delta})

                if row_ip and row_oop:
                    p_ip  = self._resolve_monker_path(row_ip)
                    p_oop = self._resolve_monker_path(row_oop)
                    rng_ip  = _load_vendor_range_compact(p_ip)
                    rng_oop = _load_vendor_range_compact(p_oop)
                    return rng_ip, rng_oop, meta

            # 2) SPH (exact/nearest within ctx)
            if self.sph is not None:
                row_ip = row_oop = None
                for s in self._ordered_stacks(int(near_stack)):
                    delta = abs(s - int(near_stack))
                    if self.max_stack_delta is not None and delta > self.max_stack_delta:
                        continue

                    if row_ip is None:
                        row_ip = self.sph._pick(s, ctx, hero=cand_ip, ip=cand_ip, oop=cand_oop)
                        if row_ip:
                            meta.update({"source": "sph", "range_ip_source_stack": s, "range_ip_stack_delta": delta})
                    if row_oop is None:
                        row_oop = self.sph._pick(s, ctx, hero=cand_oop, ip=cand_ip, oop=cand_oop)
                        if row_oop:
                            meta.update({"source": "sph", "range_oop_source_stack": s, "range_oop_stack_delta": delta})

                    if row_ip and row_oop:
                        p_ip  = self.sph._resolve_path(row_ip)
                        p_oop = self.sph._resolve_path(row_oop)
                        rng_ip  = load_sph_range_compact(p_ip)   # implement after you choose export
                        rng_oop = load_sph_range_compact(p_oop)
                        return rng_ip, rng_oop, meta

            last_meta = meta  # keep best diagnostics

        if strict:
            raise RuntimeError(f"No ranges found for {ip}v{oop}@{stack_bb} ctx={ctx}")
        return None, None, last_meta