from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# --- position normalization ---
POS_ALIASES = {"BU": "BTN", "EP": "UTG", "MP": "HJ", "LJ": "HJ"}
def norm_pos(p: str | None) -> str:
    p = (p or "").upper()
    return POS_ALIASES.get(p, p)

# --- stack helper ---
def nearest(v: float, options: List[float]) -> float:
    return min(options, key=lambda x: (abs(x - v), x))

# --- vendor range loader (compact line) ---
def _load_vendor_range_compact(path: Path) -> str:
    s = path.read_text(encoding="utf-8", errors="strict").strip()
    if not s or "..." in s:
        raise RuntimeError(f"Empty/placeholder vendor range: {path}")
    # sanity: expect "XX:prob,..." patterns
    if "," not in s or ":" not in s:
        raise RuntimeError(f"Unrecognized vendor range format: {path}")
    return s

# --- action token handling (supports normalized or raw) ---
RAISEY = {"RAISE","ALL_IN","3BET","4BET","5BET","MIN","AI","3SB"}   # treat vendor raw as raise-y
CALL_TOKENS = {"CALL"}                                              # add "Call" if your manifest not normalized
FOLD_TOKENS = {"FOLD"}

def _tok(e: dict) -> Tuple[str,str]:
    """Return (POS, ACTION_UPPER) for a sequence entry; missing action->''."""
    pos = norm_pos(e.get("pos"))
    act = (e.get("action") or "").upper()
    return pos, act

def _first_non_fold(seq: List[dict]) -> Optional[Tuple[str,str]]:
    """First actor whose action is not a pure fold."""
    for e in seq:
        pos, act = _tok(e)
        if act and act not in FOLD_TOKENS:
            return pos, act
    return None

def _first_action_of(seq: List[dict], pos_target: str) -> Optional[str]:
    tgt = norm_pos(pos_target)
    for e in seq:
        pos, act = _tok(e)
        if pos == tgt:
            return act or None
    return None

def _re_raised_before(seq: List[dict], before_pos: str) -> bool:
    tgt = norm_pos(before_pos)
    for e in seq:
        pos, act = _tok(e)
        if pos == tgt:
            return False
        if act in RAISEY:
            return True
    return False

class PreflopRangeLookup:
    """
    Uses actual monker_manifest rows to fetch IP/OOP compact ranges for a (stack, ip, oop) SRP OPEN/CALL.
    - Index key: (stack_bb, hero_pos, ip, oop) -> list[{abs_path, rel_path, ...}]
    - Fallback: nearest stack; optional pair substitution (same OOP, nearest IP seat).
    - S3 lazy fetch: if abs_path missing, downloads rel_path under s3_prefix into cache_dir.
    Returns (rng_ip, rng_oop, meta).
    """

    def __init__(
        self,
        manifest_parquet: str | Path,
        *,
        s3_client: Optional["S3Client"] = None,
        s3_prefix: Optional[str] = None,      # e.g. "data/vendor/monker"
        cache_dir: str = "data/vendor_cache", # local lazy cache root
        allow_pair_subs: bool = False,        # allow BTN→CO (same OOP) fallback
    ):
        df = pd.read_parquet(str(manifest_parquet)).copy()
        need = {"stack_bb","hero_pos","sequence","abs_path","rel_path"}
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise RuntimeError(f"monker_manifest missing columns: {missing}")

        df["hero_pos"] = df["hero_pos"].map(norm_pos)
        df["stack_bb"] = df["stack_bb"].astype("Int64")

        self.df = df
        self.stacks: List[int] = sorted(int(x) for x in df["stack_bb"].dropna().unique().tolist())
        if not self.stacks:
            raise RuntimeError("No stacks found in monker_manifest")

        self.s3 = s3_client
        self.s3_prefix = (s3_prefix or "").strip("/") if s3_prefix else None
        self.cache_dir = Path(cache_dir)
        self.allow_pair_subs = allow_pair_subs

        # Build index: (stack, hero, ip, oop) -> [row]
        self.idx: Dict[Tuple[int, str, str, str], List[dict]] = {}
        for _, r in df.iterrows():
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
            # opener must be an open-ish act
            if ip_act not in RAISEY and ip_act not in CALL_TOKENS:
                continue

            # collect any OOP seat whose first action is CALL and no re-raise before
            seen_positions = [norm_pos(e.get("pos")) for e in seq if e.get("pos")]
            for oop_pos in seen_positions:
                if oop_pos == ip_pos:
                    continue
                a_opp = _first_action_of(seq, oop_pos)
                if a_opp not in CALL_TOKENS:
                    continue
                if _re_raised_before(seq, oop_pos):
                    continue

                key = (int(r["stack_bb"]), norm_pos(str(r["hero_pos"])), ip_pos, oop_pos)
                self.idx.setdefault(key, []).append({
                    "abs_path": r["abs_path"],
                    "rel_path": r["rel_path"],
                })

        # Precompute available pairs for optional substitution
        self._seen_pairs = {(k[2], k[3]) for k in self.idx.keys()}

        # IP seat proximity order (used only if allow_pair_subs=True)
        self._pos_order = ["UTG","HJ","CO","BTN","SB","BB"]
        self._pos_index = {p:i for i,p in enumerate(self._pos_order)}

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
            raise RuntimeError(
                f"Vendor range file not found locally: {abs_path} "
                f"and no S3 configured to fetch {rel_path}"
            )

        s3_key = f"{self.s3_prefix}/{str(rel_path).replace(os.sep, '/')}"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file_if_missing(s3_key, cache_path)
        if not cache_path.exists():
            raise RuntimeError(f"Failed to materialize vendor file: s3://{self.s3.bucket}/{s3_key}")
        return cache_path

    # ---------- candidate generation & stack ordering ----------
    def _candidate_pairs(self, ip: str, oop: str) -> List[Tuple[str,str,int,bool]]:
        # (ip, oop, fallback_level, substituted?)
        out = [(ip, oop, 0, False)]  # exact
        if self.allow_pair_subs:
            alts = [(ip2, oop2) for (ip2, oop2) in self._seen_pairs if oop2 == oop and ip2 != ip]
            def ip_dist(ip2: str) -> int:
                return abs(self._pos_index.get(ip2, 99) - self._pos_index.get(ip, 99))
            alts_sorted = sorted([p for p in alts if p[1] == oop], key=lambda t: ip_dist(t[0]))
            out += [(ip2, oop, 2, True) for (ip2, _) in alts_sorted]
        return out

    def _ordered_stacks(self, target: int) -> List[int]:
        return sorted(self.stacks, key=lambda s: (abs(s - target), s))

    def _pick_row(self, stack: int, hero: str, ip: str, oop: str) -> Optional[dict]:
        return (self.idx.get((int(stack), norm_pos(hero), norm_pos(ip), norm_pos(oop))) or [None])[0]

    # ---------- main API ----------
    def ranges_for_pair(
        self,
        *,
        stack_bb: float,
        ip: str,
        oop: str,
        strict: bool = True,
    ) -> Tuple[Optional[str], Optional[str], Dict[str, object]]:
        """
        Returns (rng_ip, rng_oop, meta).
        meta includes:
          - range_ip_source_stack / range_oop_source_stack
          - range_ip_stack_delta  / range_oop_stack_delta
          - range_ip_fallback_level / range_oop_fallback_level (0=exact, 1=nearest stack, 2=pair-sub)
          - range_pair_substituted (bool)
          - range_ip_source_pair / range_oop_source_pair (e.g., 'BTNvBB')
        """
        ip, oop = norm_pos(ip), norm_pos(oop)
        if not self.stacks:
            raise RuntimeError("No stacks indexed")
        near_stack = int(nearest(float(stack_bb), [float(s) for s in self.stacks]))

        last_meta: Dict[str, object] = {}
        for cand_ip, cand_oop, level_base, substituted in self._candidate_pairs(ip, oop):
            # Try nearest stack then outward
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

            for s in self._ordered_stacks(near_stack):
                if row_ip is None:
                    row_ip = self._pick_row(s, hero=cand_ip, ip=cand_ip, oop=cand_oop)
                    if row_ip:
                        meta["range_ip_source_stack"] = s
                        meta["range_ip_stack_delta"] = abs(s - near_stack)
                        meta["range_ip_fallback_level"] = level_base if s == near_stack else max(level_base, 1)
                if row_oop is None:
                    row_oop = self._pick_row(s, hero=cand_oop, ip=cand_ip, oop=cand_oop)
                    if row_oop:
                        meta["range_oop_source_stack"] = s
                        meta["range_oop_stack_delta"] = abs(s - near_stack)
                        meta["range_oop_fallback_level"] = level_base if s == near_stack else max(level_base, 1)
                if row_ip and row_oop:
                    # resolve & load
                    path_ip  = self._resolve_local_path(row_ip)
                    path_oop = self._resolve_local_path(row_oop)
                    rng_ip  = _load_vendor_range_compact(path_ip)
                    rng_oop = _load_vendor_range_compact(path_oop)
                    return rng_ip, rng_oop, meta

            last_meta = meta  # keep best diagnostics

        # No match
        diag = [
            f"Missing vendor rows for {ip}v{oop} (requested {stack_bb}bb, nearest {near_stack}bb)."
        ]
        have = [(k[0], k[1]) for k in self.idx.keys() if k[2] == ip and k[3] == oop]
        if have:
            diag.append("Available for this pair (stack_bb, hero_side):")
            for (s, h) in sorted(set(have), key=lambda t: (t[0], t[1])):
                diag.append(f"  - {s}bb, hero={h}")
        else:
            diag.append("No entries for this pair were found in the manifest index.")

        if strict:
            raise RuntimeError("\n".join(diag))
        return None, None, last_meta