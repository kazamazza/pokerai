import json, gzip, re
from typing import Any, Dict, Tuple, Optional, List, Sequence, Literal
from ml.models.policy_consts import ACTION_VOCAB
from .solver_schema import SolverExtraction

FOLD_RE  = re.compile(r"\bfold\b", re.IGNORECASE)
CALL_RE  = re.compile(r"\bcall\b", re.IGNORECASE)
CHECK_RE = re.compile(r"\bcheck\b", re.IGNORECASE)
BET_RE   = re.compile(r"\bbet\b", re.IGNORECASE)
RAISE_RE = re.compile(r"\braise\b", re.IGNORECASE)
ALLIN_RE = re.compile(r"\ball[-\s]*in\b|\bjam\b", re.IGNORECASE)

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")  # keep \t \n \r


class TexasSolverExtractor:
    def __init__(self) -> None:
        pass

    # --- UPDATE helper to honor root_bet_kind ---
    def _override_root_mix(
            self,
            mix: Dict[str, float],
            *,
            root_actor: str,
            root_bet_kind: Literal["donk", "bet"],  # NEW
            size_pct: Optional[int],
    ) -> Dict[str, float]:
        """Force root to legal tokens; tie bet/donk to requested size; preserve ALLIN if present."""
        allin_mass = float(mix.get("ALLIN", 0.0))

        out = {a: 0.0 for a in ACTION_VOCAB}
        out["CHECK"] = float(mix.get("CHECK", 0.0))

        if size_pct is not None:
            prefix = "DONK" if root_bet_kind == "donk" else "BET"
            size_tok = f"{prefix}_{int(size_pct)}"
            if size_tok in out:
                betlike_mass = sum(v for k, v in mix.items() if k.startswith("BET_") or k.startswith("DONK_"))
                out[size_tok] = betlike_mass

        if allin_mass > 0.0:
            out["ALLIN"] = allin_mass  # keep shove if stack-capped

        # forbid call/raise at root
        for k in list(out.keys()):
            if k.startswith("RAISE_") or k in ("CALL", "FOLD"):
                out[k] = 0.0
        return out

    def _filter_facing_only(self, mix: Dict[str, float]) -> Dict[str, float]:
        """
        Keep only FOLD/CALL/RAISE_* (and ALLIN). Zero everything else.
        WHY: facing model learns responses vs a bet.
        """
        out = {a: 0.0 for a in ACTION_VOCAB}
        for k, v in mix.items():
            if k in ("FOLD", "CALL", "ALLIN") or k.startswith("RAISE_"):
                out[k] = float(v)
        return out

    def _bucket_raise_to_percent_token(
            self,
            to_bb: float,
            *,
            pot_bb: float,
            faced_bet_bb: float,
            allowed_percents: list[int]  # e.g. [150, 200, 300]
    ) -> Optional[str]:
        """
        Map an absolute raise-to (BB) into RAISE_% token using the solver's 'pot %' semantics.
        Try both common solver pot bases at facing node:
          base1 = pot + faced_bet
          base2 = pot + 2 * faced_bet   (accounts for call-included variants)
        Pick the (base, percent) whose theoretical target is closest to to_bb.
        """
        if to_bb is None or pot_bb is None or faced_bet_bb is None or faced_bet_bb <= 0:
            return None

        bases = [pot_bb + faced_bet_bb, pot_bb + 2.0 * faced_bet_bb]
        best = None  # (abs_err, token)
        for base in bases:
            if base <= 0:
                continue
            for pct in allowed_percents:
                target = (pct / 100.0) * base
                err = abs(to_bb - target)
                tok = f"RAISE_{pct}"
                cand = (err, tok)
                if (best is None) or (cand < best):
                    best = cand

        return best[1] if best else None

    def extract(
            self,
            path: str,
            *,
            ctx: str,
            ip_pos: str,
            oop_pos: str,
            board: str,
            pot_bb: float,
            stack_bb: float,
            bet_sizing_id: str,
            bet_sizes: Optional[List[float]] = None,
            raise_mults: Optional[List[float]] = None,
            size_pct: Optional[int] = None,  # requested size (33/66/100)
            root_actor: str = "oop",  # "ip" or "oop"
            root_bet_kind: Literal["donk", "bet"] = "donk"  # NEW: OOP root menu kind
    ) -> SolverExtraction:
        """Single entry point. Returns a filled SolverExtraction (ok=True) or reason."""
        ex = SolverExtraction(
            ctx=ctx,
            ip_pos=ip_pos,
            oop_pos=oop_pos,
            board=board,
            pot_bb=float(pot_bb),
            stack_bb=float(stack_bb),
            bet_sizing_id=str(bet_sizing_id),
        )

        try:
            payload = self._open_json_any(path)
            root = payload.get("root", payload)
            if not isinstance(root, dict):
                return self._fail(ex, "malformed_root")

            # keep for diagnostics
            ex.meta["root_actor"] = root_actor
            ex.meta["root_bet_kind"] = root_bet_kind
            ex.meta["requested_size_pct"] = int(size_pct) if size_pct is not None else None

            # --- 1) Root mix ---
            ex.root_mix, root_meta = self._read_node_action_mix(
                root,
                mode="root",
                pot_bb=ex.pot_bb,
                stack_bb=ex.stack_bb,
                bet_sizes=bet_sizes,
                bet_sizing_id=ex.bet_sizing_id,
            )
            ex.meta["root_meta"] = root_meta

            # deterministic override to legal tokens + correct size bucket
            if ex.root_mix:
                ex.root_mix = self._override_root_mix(
                    ex.root_mix,
                    root_actor=root_actor,
                    root_bet_kind=root_bet_kind,  # NEW
                    size_pct=size_pct,
                )

            # --- 2) Facing (response to first bet) ---
            # --- 2) Facing (response to first bet) ---
            bet_node, via_path = self._find_first_bet_node(root)
            ex.meta["facing_path"] = via_path
            if bet_node is not None:
                # 1) Prefer explicit faced size from filename/menu (sz33p etc.)
                faced_from_menu = None
                if size_pct is not None:
                    try:
                        faced_from_menu = (float(size_pct) / 100.0) * ex.pot_bb
                    except Exception:
                        faced_from_menu = None

                # 2) Fallback to parsing the bet label if needed
                facing_pct, facing_to_bb = self._extract_bet_size(via_path[-1], ex.pot_bb)

                # 3) Decide faced bet bb with strong preference to the menu hint
                if faced_from_menu is not None and faced_from_menu > 0:
                    ex.facing_bet_bb = faced_from_menu
                else:
                    ex.facing_bet_bb = (
                        facing_to_bb if facing_to_bb is not None
                        else (ex.pot_bb * (facing_pct or 0) / 100.0)
                    )

                ex.facing_mix, face_meta = self._read_node_action_mix(
                    bet_node,
                    mode="facing",
                    pot_bb=ex.pot_bb,
                    stack_bb=ex.stack_bb,
                    facing_bet_bb=ex.facing_bet_bb,
                    raise_mults=raise_mults,
                    bet_sizing_id=ex.bet_sizing_id,
                )
                ex.meta["facing_meta"] = face_meta

                if ex.facing_mix:
                    ex.facing_mix = self._filter_facing_only(ex.facing_mix)

            # --- 3) validate ---
            if (ex.root_mix and self._sum(ex.root_mix) > 0) or (ex.facing_mix and self._sum(ex.facing_mix) > 0):
                if ex.root_mix:
                    ex.root_mix = self._renorm_map(ex.root_mix)
                if ex.facing_mix:
                    ex.facing_mix = self._renorm_map(ex.facing_mix)
                ex.ok = True
                return ex

            return self._fail(ex, "zero_mass")

        except Exception as e:
            return self._fail(ex, f"exception: {e}")

    def _read_node_action_mix(
            self,
            node: Dict[str, Any],
            *,
            mode: str,  # "root" or "facing"
            pot_bb: float,
            stack_bb: Optional[float] = None,
            facing_bet_bb: Optional[float] = None,
            bet_sizes: Optional[List[float]] = None,  # e.g. [0.33, 0.66, 1.0]
            raise_mults: Optional[List[float]] = None,  # accepts [1.5,2.0,3.0] or [150,200,300]
            bet_sizing_id: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:

        acts, mix, where, raw_labels = self._find_any_strategy(node)

        # Root BET buckets (unchanged)
        if bet_sizes:
            bet_bucket_pct = sorted({
                int(round(100.0 * float(s)))
                for s in bet_sizes
                if s is not None and 0.0 < float(s) <= 2.0
            }) or [25, 33, 50, 66, 75, 100]
        else:
            bet_bucket_pct = [25, 33, 50, 66, 75, 100]

        # Facing RAISE buckets: normalize either multipliers or raw percents → percents
        raise_buckets_pct: list[int] = []
        if raise_mults:
            for x in raise_mults:
                try:
                    xv = float(x)
                    if xv <= 10.0:
                        raise_buckets_pct.append(int(round(xv * 100)))
                    else:
                        raise_buckets_pct.append(int(round(xv)))
                except Exception:
                    pass
        if not raise_buckets_pct:
            raise_buckets_pct = [150, 200, 300]

        meta = {
            "where": where,
            "raw_actions": raw_labels,
            "bet_buckets_pct": bet_bucket_pct,
            "raise_buckets_pct": sorted(raise_buckets_pct),
            "bet_sizing_id": bet_sizing_id,
            "mode": mode,
            "pot_bb": pot_bb,
            "stack_bb": stack_bb,
            "facing_bet_bb": facing_bet_bb,
        }

        mapped: Dict[str, float] = {a: 0.0 for a in ACTION_VOCAB}

        for lbl, p in zip(acts, mix):
            lab = (lbl or "").lower().strip()
            if not lab or p <= 0.0:
                continue

            # Simple verbs
            if FOLD_RE.search(lab):
                mapped["FOLD"] += p;
                continue
            if CHECK_RE.search(lab):
                mapped["CHECK"] += p;
                continue
            if CALL_RE.search(lab):
                mapped["CALL"] += p;
                continue
            if ALLIN_RE.search(lab):
                mapped["ALLIN"] += p;
                continue

            # ---- BET label ----
            if BET_RE.search(lab):
                pct, to_bb = self._extract_bet_size(lbl, pot_bb)

                # If no absolute, derive from % of pot
                if to_bb is None and pct is not None:
                    pv = float(pct)
                    pct_val_tmp = pv * 100.0 if pv <= 3.0 else pv
                    to_bb = (pct_val_tmp / 100.0) * pot_bb

                if mode == "facing":
                    # BET at facing == raise-to target → bucket by percent of (pot [+ bet])
                    tok = self._bucket_raise_to_percent_token(
                        to_bb,
                        pot_bb=pot_bb,
                        faced_bet_bb=facing_bet_bb,
                        allowed_percents=raise_buckets_pct,
                    )
                    if tok and tok in ACTION_VOCAB:
                        mapped[tok] += p
                    # else: drop unbucketable oddities
                    continue

                # Root mode: bucket to BET_%
                pct_val = None
                if pct is not None:
                    pv = float(pct)
                    pct_val = pv * 100.0 if pv <= 3.0 else pv
                    pct_val = max(0.0, min(pct_val, 200.0))

                tok = "BET_50"
                if pct_val is not None and bet_bucket_pct:
                    nearest = min(bet_bucket_pct, key=lambda b: abs(b - pct_val))
                    cand = f"BET_{int(nearest)}"
                    if cand in ACTION_VOCAB:
                        tok = cand
                mapped[tok] += p
                continue

            # ---- RAISE label ----
            if RAISE_RE.search(lab):
                pct, to_bb = self._extract_bet_size(lbl, pot_bb)

                # Treat 'raise to stack' as jam
                if (to_bb is not None) and (stack_bb is not None):
                    try:
                        if float(to_bb) >= 0.98 * float(stack_bb):
                            mapped["ALLIN"] += p
                            continue
                    except Exception:
                        pass

                if mode == "facing":
                    tok = self._bucket_raise_to_percent_token(
                        to_bb,
                        pot_bb=pot_bb,
                        faced_bet_bb=facing_bet_bb,
                        allowed_percents=raise_buckets_pct,
                    )
                    if tok and tok in ACTION_VOCAB:
                        mapped[tok] += p
                    # else: drop if we can’t place it confidently
                    continue

            # Unknown verbs are ignored

        return mapped, meta


    def _find_any_strategy(self, node: Dict[str, Any]) -> Tuple[List[str], List[float], str, List[str]]:
        """
        Try multiple layouts:
         A) node = {"actions":[...], "strategy":[...]}
         B) node = {"strategy":{"actions":[...], "strategy":{combo:[...]}}} → average across combos
         C) node = {"strategy":{"CALL":0.4,"FOLD":0.6}}
         D) per-child weights (prob/freq/weight) with labels in children
        """
        # A: flat arrays
        if isinstance(node.get("actions"), list) and isinstance(node.get("strategy"), list):
            a = [str(x) for x in node["actions"]]
            m = [float(v) for v in node["strategy"]]
            a, m = self._renorm(a, m)
            return a, m, "node.actions+strategy(list)", a

        s = node.get("strategy")
        # B: dict with actions + per-combo rows
        if isinstance(s, dict) and "actions" in s and "strategy" in s:
            actions = [str(x) for x in s["actions"]]
            k = len(actions)
            strat = s["strategy"]
            if isinstance(strat, dict) and strat:
                acc = [0.0]*k; n = 0
                for row in strat.values():
                    if isinstance(row, list) and len(row) == k:
                        for i, x in enumerate(row): acc[i] += float(x)
                        n += 1
                if n > 0:
                    mix = [v/max(n,1) for v in acc]
                    actions, mix = self._renorm(actions, mix)
                    return actions, mix, "node.strategy{combo->list}", actions
            # or plain list in strategy
            if isinstance(strat, list) and len(strat) == k:
                mix = [float(x) for x in strat]
                actions, mix = self._renorm(actions, mix)
                return actions, mix, "node.strategy(list)", actions

        # C: simple map
        if isinstance(s, dict) and s and all(isinstance(v,(int,float)) for v in s.values()):
            actions = list(s.keys())
            mix = [float(s[k]) for k in actions]
            actions, mix = self._renorm([str(x) for x in actions], mix)
            return actions, mix, "node.strategy(map)", actions

        # D: child weights
        kids = self._normalize_children(node)
        acts, mix = [], []
        if kids:
            for lbl, ch in kids.items():
                w = self._read_child_weight(ch)
                if w is not None:
                    acts.append(str(lbl))
                    mix.append(float(w))
        if acts and sum(mix) > 0:
            acts, mix = self._renorm(acts, mix)
            return acts, mix, "children.weighted", acts

        # fallback: no strategy → equal over children labels (rare, keep for completeness)
        if kids:
            acts = list(kids.keys())
            mix = [1.0/len(acts)]*len(acts)
            return acts, mix, "children.uniform", acts

        return [], [], "none", []

    def _extract_bet_size(self, label: str, pot_bb: float) -> tuple[Optional[float], Optional[float]]:
        lab = (label or "").strip()

        # 0) VERB + NUMBER → absolute bb (e.g., "BET 2.000000", "RAISE 23.000000")
        m = re.match(r'^\s*(?:BET|RAISE)\s+(\d+(?:\.\d+)?)\s*$', lab, flags=re.IGNORECASE)
        if m:
            try:
                to_bb = float(m.group(1))
                return (None, to_bb)  # absolute target bb; no percentage in label
            except:
                pass

        # 1) explicit percent (e.g., "33%")
        m = re.search(r'\b(\d+(?:\.\d+)?)\s*%', lab)
        pct = float(m.group(1)) if m else None

        # 2) explicit "to <bb>" (or "… to 18")
        m = re.search(r'\bto\s+(\d+(?:\.\d+)?)\b', lab, flags=re.IGNORECASE)
        to_bb = float(m.group(1)) if m else None

        # 3) if only pct known, derive absolute
        if to_bb is None and pct is not None:
            to_bb = (pct / 100.0) * float(pot_bb)

        return pct, to_bb

    def _pct_from_to_bb(self, to_bb: Optional[float], pot_bb: float) -> Optional[float]:
        if to_bb is None or pot_bb <= 1e-9: return None
        return (to_bb / pot_bb) * 100.0

    def _bucket_bet_pct(self, val: Optional[float]) -> str:
        """
        Map a bet size to vocab buckets.
        Accepts either fraction-of-pot (e.g. 0.33, 0.66) or percent (33, 66).
        """
        default = "BET_50"
        if val is None:
            return default

        # if it's a fraction (<= 3.0), convert to percent
        pct = float(val) * 100.0 if val <= 3.0 else float(val)
        # clamp to sane range
        if pct <= 0:
            return default
        if pct > 120:
            pct = 100.0  # we don't have >100% buckets in ACTION_VOCAB

        buckets = [25, 33, 50, 66, 75, 100]
        nearest = min(buckets, key=lambda b: abs(b - pct))
        tok = f"BET_{int(nearest)}"
        return tok if tok in ACTION_VOCAB else default

    def _bucket_raise_to(self, to_val: Optional[float], facing_val: Optional[float]) -> str:
        """
        Map a raise-to target to vocab buckets.
        Handles either:
          - absolute raise-to in 'bb' with current bet 'facing_val' in 'bb'  → mult = to_val / facing_val
          - direct multiplier in 'to_val' when facing_val is missing/small
        """
        default = "RAISE_300"
        if to_val is None:
            return default

        to_val = float(to_val)
        mult: Optional[float] = None

        if facing_val is not None and float(facing_val) > 1e-9:
            mult = to_val / float(facing_val)
        else:
            # Heuristic: if to_val is within plausible multiplier range, treat it as multiplier directly
            if 1.1 <= to_val <= 6.0:
                mult = to_val

        if mult is None or mult <= 1.0:  # invalid / min-raise edge cases → default
            return default

        candidates = [1.5, 2.0, 3.0, 4.0, 5.0]
        nearest = min(candidates, key=lambda x: abs(x - mult))
        tok = f"RAISE_{int(round(nearest * 100))}"
        return tok if tok in ACTION_VOCAB else default

    def _normalize_children(self, node: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        for k in ("childrens", "children"):
            ch = node.get(k)
            if isinstance(ch, dict): return ch
            if isinstance(ch, list):
                out = {}
                for c in ch:
                    if isinstance(c, dict):
                        label = c.get("label") or c.get("action") or str(len(out))
                        out[str(label)] = c
                return out
        return {}

    def _read_child_weight(self, node: Dict[str, Any]) -> Optional[float]:
        for k in ("prob","p","weight","frequency","freq","w"):
            v = node.get(k)
            if v is not None:
                try: return float(v)
                except: pass
        data = node.get("data")
        if isinstance(data, dict):
            for k in ("prob","p","weight","frequency","freq","w"):
                v = data.get(k)
                if v is not None:
                    try: return float(v)
                    except: pass
        return None

    def _find_first_bet_node(self, root: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        kids = self._normalize_children(root)
        # root->BET
        for lbl, ch in kids.items():
            if "bet" in str(lbl).lower():
                return ch, [str(lbl)]
        # root->CHECK->BET
        for lbl, ch in kids.items():
            if "check" in str(lbl).lower():
                kids2 = self._normalize_children(ch)
                for lbl2, ch2 in kids2.items():
                    if "bet" in str(lbl2).lower():
                        return ch2, [str(lbl), str(lbl2)]
        # shallow DFS fallback
        stack = [([], root)]
        while stack:
            path, node = stack.pop()
            for lbl, ch in self._normalize_children(node).items():
                np = path + [str(lbl)]
                if "bet" in str(lbl).lower():
                    return ch, np
                stack.append((np, ch))
        return None, []

    # ----- IO / math helpers -----


    def _first_json_object(self, txt: str) -> str | None:
        i = 0
        n = len(txt)
        while i < n and txt[i] not in "{[":
            i += 1
        if i >= n:
            return None
        opener = txt[i]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_str = False
        esc = False
        for j in range(i, n):
            ch = txt[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        return txt[i:j + 1]
        return None

    def _open_json_any(self, path: str) -> Dict[str, Any]:
        """
        Robust loader:
          - reads whole file (or s3 object) into bytes first
          - validates gzip stream
          - retries once if JSON decode fails (handles occasional short reads)
          - returns {} on hard failure so caller can emit a sentinel
        """
        import io, json, gzip
        from json import JSONDecodeError

        def _load_bytes(b: bytes) -> Dict[str, Any]:
            # decompress if gz
            if path.endswith(".gz"):
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(b)) as gz:
                        txt = gz.read().decode("utf-8", errors="strict")
                except Exception:
                    # fall back with tolerant decode to detect partial content
                    with gzip.GzipFile(fileobj=io.BytesIO(b)) as gz:
                        txt = gz.read().decode("utf-8", errors="replace")
            else:
                txt = b.decode("utf-8", errors="replace")

            # tolerant JSON decode first; if it fails, we retry once later
            try:
                return json.loads(txt)
            except JSONDecodeError:
                raise

        # --- read all bytes (local fs) ---
        if path.startswith("s3://"):
            # Let the caller use S3Client to fetch to a local tmp and pass the path.
            # If you must read direct, do so via your S3 client here and set `blob`.
            raise RuntimeError("s3 paths should be downloaded to a local file before calling _open_json_any")
        else:
            with open(path, "rb") as fh:
                blob = fh.read()

        # try decode, retry once
        try:
            return _load_bytes(blob)
        except Exception:
            # One retry in case of transient short read
            try:
                return _load_bytes(blob)
            except Exception:
                return {}  # caller will treat as malformed and emit sentinel

    def _renorm(self, acts: List[str], mix: List[float]) -> Tuple[List[str], List[float]]:
        mix = [max(0.0, float(x)) for x in mix]
        s = sum(mix)
        if s <= 0:
            return acts, [1.0/len(acts)]*len(acts) if acts else ([],[])
        return acts, [x/s for x in mix]

    def _renorm_map(self, d: Dict[str, float]) -> Dict[str, float]:
        s = sum(d.values()) or 1.0
        return {k: (v/s) for k,v in d.items()}

    def _sum(self, d: Dict[str, float]) -> float:
        return float(sum(d.values()))

    def _fail(self, ex: SolverExtraction, reason: str) -> SolverExtraction:
        ex.ok = False
        ex.reason = reason
        return ex