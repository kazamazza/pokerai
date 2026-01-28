from __future__ import annotations

import gzip
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal


ActionMix = Dict[str, float]


@dataclass
class SolverExtraction:
    # Identifiers/context (copied from manifest or inferred)
    ctx: str
    ip_pos: str
    oop_pos: str
    board: str
    pot_bb: float
    stack_bb: float
    bet_sizing_id: str

    # What we require from the JSON
    root_mix: ActionMix = field(default_factory=dict)     # solver-native labels
    facing_mix: ActionMix = field(default_factory=dict)   # solver-native labels
    facing_bet_bb: Optional[float] = None

    # Diagnostics for auditability
    meta: Dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    reason: Optional[str] = None


FOLD_RE  = re.compile(r"\bfold\b", re.IGNORECASE)
CALL_RE  = re.compile(r"\bcall\b", re.IGNORECASE)
CHECK_RE = re.compile(r"\bcheck\b", re.IGNORECASE)
BET_RE   = re.compile(r"\bbet\b|\bdonk\b|\bprobe\b", re.IGNORECASE)
RAISE_RE = re.compile(r"\braise\b|\breraise\b|\bre-raise\b|\b3bet\b|\b4bet\b|\bx\b", re.IGNORECASE)
ALLIN_RE = re.compile(r"\ball[-\s]*in\b|\bjam\b|\bshove\b", re.IGNORECASE)

_NUM = re.compile(r"([-+]?\d+(?:\.\d+)?)")


RootBetKind = Literal["donk", "bet"]


def _norm_ws(s: str) -> str:
    return " ".join(str(s or "").strip().split())


def _last_number(s: str) -> Optional[float]:
    m = _NUM.findall(str(s))
    if not m:
        return None
    try:
        return float(m[-1])
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


class TexasSolverExtractor:
    """
    Parser-only extractor.

    - Reads a TexasSolver JSON (optionally gz).
    - Extracts a root action mix (CHECK + bet-like mass) using solver-native labels.
    - Finds the first bet node and extracts the facing response mix using solver-native labels.
    - Does NOT map to ML vocabs. That is handled by ml/policy/solver_action_mapping.py
    """

    def __init__(self) -> None:
        pass

    # ============================================================
    # Public API
    # ============================================================
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
        size_pct: Optional[int] = None,               # requested concrete size (33/50/66/etc)
        root_actor: str = "oop",                      # "ip" or "oop" (kept for meta)
        root_bet_kind: RootBetKind = "donk",          # influences label normalization
    ) -> SolverExtraction:
        ex = SolverExtraction(
            ctx=str(ctx),
            ip_pos=str(ip_pos),
            oop_pos=str(oop_pos),
            board=str(board),
            pot_bb=float(pot_bb),
            stack_bb=float(stack_bb),
            bet_sizing_id=str(bet_sizing_id),
        )

        ex.meta["root_actor"] = root_actor
        ex.meta["root_bet_kind"] = root_bet_kind
        ex.meta["requested_size_pct"] = int(size_pct) if size_pct is not None else None
        ex.meta["raise_mults_in"] = list(raise_mults) if raise_mults is not None else None

        try:
            payload = self._open_json_any(path)
            if not isinstance(payload, dict) or not payload:
                return self._fail(ex, "malformed_or_empty_json")

            # Some dumps wrap under "root"
            root = payload.get("root", payload)
            if not isinstance(root, dict) or not root:
                return self._fail(ex, "malformed_root")

            # 1) Root mix
            root_mix, root_meta = self._read_node_action_mix(root)
            ex.meta["root_meta"] = root_meta

            # Reduce root mix to CHECK + one bet-like label tied to requested size (if provided)
            ex.root_mix = self._canonicalize_root_mix(
                root_mix,
                size_pct=size_pct,
                root_bet_kind=root_bet_kind,
            )

            # 2) Facing
            bet_node, via_path = self._find_first_bet_node(root)
            ex.meta["facing_path"] = via_path

            if bet_node is None:
                # It's possible solver tree has no bet node (rare). Root-only still ok if has mass.
                if self._sum(ex.root_mix) > 1e-12:
                    ex.ok = True
                    return ex
                return self._fail(ex, "no_bet_node")

            # resolve facing bet size in BB
            ex.facing_bet_bb, fb_meta = self._resolve_facing_bet_bb(
                pot_bb=ex.pot_bb,
                size_pct=size_pct,
                via_path=via_path,
            )
            ex.meta["facing_bet_meta"] = fb_meta
            if ex.facing_bet_bb is None or ex.facing_bet_bb <= 0:
                return self._fail(ex, "facing_size_unresolved")

            facing_mix, facing_meta = self._read_node_action_mix(bet_node)
            ex.meta["facing_meta"] = facing_meta

            # Filter facing to FOLD/CALL/RAISE-like/ALLIN, keeping labels solver-native
            ex.facing_mix = self._canonicalize_facing_mix(
                facing_mix,
                pot_bb=ex.pot_bb,
                stack_bb=ex.stack_bb,
                facing_bet_bb=ex.facing_bet_bb,
                raise_mults=raise_mults,
            )

            # 3) Validate minimal contract
            if self._sum(ex.root_mix) > 1e-12:
                ex.root_mix = self._renorm_map(ex.root_mix)
            if self._sum(ex.facing_mix) > 1e-12:
                ex.facing_mix = self._renorm_map(ex.facing_mix)

            if self._sum(ex.root_mix) > 1e-12 or self._sum(ex.facing_mix) > 1e-12:
                ex.ok = True
                return ex

            return self._fail(ex, "zero_mass")

        except Exception as e:
            return self._fail(ex, f"exception:{type(e).__name__}:{e}")

    # ============================================================
    # Root canonicalization (no hard vocab)
    # ============================================================
    def _canonicalize_root_mix(
        self,
        mix: ActionMix,
        *,
        size_pct: Optional[int],
        root_bet_kind: RootBetKind,
    ) -> ActionMix:
        """
        Root should be "no bet faced". Keep:
          - CHECK mass
          - a single bet-like label representing the requested size (if provided),
            else keep all bet-like labels as-is (rare; mostly you solve size-specific).
          - ALLIN if present (rare edge)
        """
        if not mix:
            return {}

        check_mass = 0.0
        allin_mass = 0.0
        betlike_mass = 0.0
        other_mass = 0.0

        for k, v in mix.items():
            p = _safe_float(v)
            if p is None or p <= 0:
                continue
            lab = str(k or "")
            if CHECK_RE.search(lab):
                check_mass += p
            elif ALLIN_RE.search(lab):
                allin_mass += p
            elif BET_RE.search(lab):
                betlike_mass += p
            else:
                # root shouldn't contain call/fold/raise, but tolerate solver oddities
                other_mass += p

        out: ActionMix = {}
        if check_mass > 0:
            out["CHECK"] = check_mass

        # Tie bet label to requested size if available
        if betlike_mass > 0:
            if size_pct is not None:
                kind = "DONK" if root_bet_kind == "donk" else "BET"
                out[f"{kind} {int(size_pct)}%"] = betlike_mass
            else:
                out["BET"] = betlike_mass

        if allin_mass > 0:
            out["ALLIN"] = allin_mass

        # If everything got shoved into "other_mass", keep it under a single label for visibility
        if (check_mass <= 0 and betlike_mass <= 0 and allin_mass <= 0) and other_mass > 0:
            out["OTHER"] = other_mass

        return out

    # ============================================================
    # Facing canonicalization (no hard vocab)
    # ============================================================
    def _canonicalize_facing_mix(
        self,
        mix: ActionMix,
        *,
        pot_bb: float,
        stack_bb: float,
        facing_bet_bb: float,
        raise_mults: Optional[List[float]],
    ) -> ActionMix:
        """
        Facing node: keep solver-native labels but normalize common ones.

        We keep:
          - FOLD
          - CALL
          - ALLIN
          - RAISE labels converted to a consistent "RAISE {mult}x" when we can infer a mult.

        We do NOT bucket to your final vocab here. Mapping does that later.
        """
        if not mix:
            return {}

        allowed_mults = self._normalize_raise_mults(raise_mults)

        out: ActionMix = {}
        debug_rows: List[Dict[str, Any]] = []

        for k, v in mix.items():
            p = _safe_float(v)
            if p is None or p <= 0:
                continue
            lab = _norm_ws(k)

            if FOLD_RE.search(lab):
                out["FOLD"] = out.get("FOLD", 0.0) + p
                continue
            if CALL_RE.search(lab):
                out["CALL"] = out.get("CALL", 0.0) + p
                continue
            if ALLIN_RE.search(lab):
                out["ALLIN"] = out.get("ALLIN", 0.0) + p
                continue

            if RAISE_RE.search(lab):
                # infer raise-to bb if possible, then infer mult against solver bases
                pct, to_bb = self._extract_bet_size(lab, pot_bb)
                if to_bb is None and pct is not None:
                    # numeric-only fallback: some labels return pct as a number
                    try:
                        to_bb = float(pct)
                    except Exception:
                        to_bb = None

                # all-in by size vs stack?
                if to_bb is not None and stack_bb > 0 and to_bb >= 0.98 * stack_bb:
                    out["ALLIN"] = out.get("ALLIN", 0.0) + p
                    debug_rows.append({"label": lab, "p": p, "to_bb": to_bb, "chosen": "ALLIN"})
                    continue

                multA, multB, chosen = self._infer_raise_mult(
                    to_bb=to_bb,
                    pot_bb=pot_bb,
                    facing_bet_bb=facing_bet_bb,
                    allowed_mults=allowed_mults,
                )

                debug_rows.append({
                    "label": lab,
                    "p": float(p),
                    "to_bb": float(to_bb) if to_bb is not None else None,
                    "multA": multA,
                    "multB": multB,
                    "chosen": chosen,
                })

                if chosen is None:
                    # keep raw label if we couldn't infer; better for audit + later mapping may parse "3x"
                    out[lab] = out.get(lab, 0.0) + p
                else:
                    out[f"RAISE {chosen:.3g}x"] = out.get(f"RAISE {chosen:.3g}x", 0.0) + p
                continue

            # If it's a BET-like label at facing node (rare but happens in some trees),
            # keep it for visibility. Mapping layer can decide what to do.
            if BET_RE.search(lab):
                out[lab] = out.get(lab, 0.0) + p
                continue

            # unknown label: keep (auditability > guessing)
            out[lab] = out.get(lab, 0.0) + p

        # store debug
        if debug_rows:
            # merged into meta by caller; we return as-is here
            pass

        return out

    def _normalize_raise_mults(self, raise_mults: Optional[List[float]]) -> List[float]:
        """
        Accepts either [2.0, 3.0, 4.5] or [200, 300, 450] or mixed.
        Returns sorted unique multipliers >1.0.
        """
        mults: List[float] = []
        if raise_mults:
            for x in raise_mults:
                v = _safe_float(x)
                if v is None:
                    continue
                if v > 10.0:
                    v = v / 100.0
                if 1.0 < v <= 15.0:
                    mults.append(v)

        if not mults:
            mults = [2.0, 3.0, 4.5]

        return sorted(set(mults))

    def _infer_raise_mult(
        self,
        *,
        to_bb: Optional[float],
        pot_bb: float,
        facing_bet_bb: float,
        allowed_mults: List[float],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        TexasSolver can interpret raise sizes relative to different bases at facing node.
        We try both:
          baseA = pot + faced_bet
          baseB = pot + 2*faced_bet
        We compute mult = to_bb/base and snap to nearest allowed bucket.
        Return (multA, multB, chosen_bucket_mult).
        """
        if to_bb is None or to_bb <= 0 or facing_bet_bb <= 0:
            return None, None, None

        baseA = pot_bb + facing_bet_bb
        baseB = pot_bb + 2.0 * facing_bet_bb

        multA = (to_bb / baseA) if baseA > 1e-9 else None
        multB = (to_bb / baseB) if baseB > 1e-9 else None

        # pick best by absolute raise-to error after snapping
        best: Optional[Tuple[float, float]] = None  # (abs_err, snapped_mult)
        for base, mult in ((baseA, multA), (baseB, multB)):
            if mult is None or not (1.01 <= mult <= 15.0):
                continue
            snapped = min(allowed_mults, key=lambda m: (abs(m - mult), m))
            target = snapped * base
            err = abs(to_bb - target)
            cand = (err, snapped)
            if best is None or cand < best:
                best = cand

        return multA, multB, (best[1] if best else None)

    # ============================================================
    # Node parsing (robust)
    # ============================================================
    def _read_node_action_mix(self, node: Dict[str, Any]) -> Tuple[ActionMix, Dict[str, Any]]:
        acts, mix, where, raw_labels = self._find_any_strategy(node)
        meta = {"where": where, "raw_actions": raw_labels}

        out: ActionMix = {}
        for a, p in zip(acts, mix):
            prob = _safe_float(p)
            if prob is None or prob <= 0:
                continue
            lab = _norm_ws(a)
            out[lab] = out.get(lab, 0.0) + float(prob)

        # normalize but keep sparse
        out = self._renorm_map(out) if self._sum(out) > 1e-12 else out
        return out, meta

    def _find_any_strategy(self, node: Dict[str, Any]) -> Tuple[List[str], List[float], str, List[str]]:
        """
        Try multiple layouts:
         A) node = {"actions":[...], "strategy":[...]}
         B) node = {"strategy":{"actions":[...], "strategy":{combo:[...]}}} → avg combos
         C) node = {"strategy":{"CALL":0.4,"FOLD":0.6}}
         D) per-child weights (prob/freq/weight) with labels in children
        """
        # A
        if isinstance(node.get("actions"), list) and isinstance(node.get("strategy"), list):
            a = [str(x) for x in node["actions"]]
            m = [float(v) for v in node["strategy"]]
            a, m = self._renorm(a, m)
            return a, m, "node.actions+strategy(list)", a

        s = node.get("strategy")

        # B
        if isinstance(s, dict) and "actions" in s and "strategy" in s:
            actions = [str(x) for x in s["actions"]]
            k = len(actions)
            strat = s["strategy"]

            if isinstance(strat, dict) and strat:
                acc = [0.0] * k
                n = 0
                for row in strat.values():
                    if isinstance(row, list) and len(row) == k:
                        for i, x in enumerate(row):
                            acc[i] += float(x)
                        n += 1
                if n > 0:
                    mix = [v / max(n, 1) for v in acc]
                    actions, mix = self._renorm(actions, mix)
                    return actions, mix, "node.strategy{combo->list}", actions

            if isinstance(strat, list) and len(strat) == k:
                mix = [float(x) for x in strat]
                actions, mix = self._renorm(actions, mix)
                return actions, mix, "node.strategy(list)", actions

        # C
        if isinstance(s, dict) and s and all(isinstance(v, (int, float)) for v in s.values()):
            actions = [str(x) for x in s.keys()]
            mix = [float(s[k]) for k in s.keys()]
            actions, mix = self._renorm(actions, mix)
            return actions, mix, "node.strategy(map)", actions

        # D
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

        if kids:
            acts = list(kids.keys())
            mix = [1.0 / len(acts)] * len(acts)
            return acts, mix, "children.uniform", acts

        return [], [], "none", []

    def _normalize_children(self, node: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        for k in ("childrens", "children"):
            ch = node.get(k)
            if isinstance(ch, dict):
                return ch
            if isinstance(ch, list):
                out = {}
                for c in ch:
                    if isinstance(c, dict):
                        label = c.get("label") or c.get("action") or str(len(out))
                        out[str(label)] = c
                return out
        return {}

    def _read_child_weight(self, node: Dict[str, Any]) -> Optional[float]:
        for k in ("prob", "p", "weight", "frequency", "freq", "w"):
            v = node.get(k)
            if v is not None:
                f = _safe_float(v)
                if f is not None:
                    return f
        data = node.get("data")
        if isinstance(data, dict):
            for k in ("prob", "p", "weight", "frequency", "freq", "w"):
                v = data.get(k)
                if v is not None:
                    f = _safe_float(v)
                    if f is not None:
                        return f
        return None

    # ============================================================
    # Find facing node + size resolution
    # ============================================================
    def _find_first_bet_node(self, root: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        kids = self._normalize_children(root)

        # root -> BET
        for lbl, ch in kids.items():
            if BET_RE.search(str(lbl)):
                return ch, [str(lbl)]

        # root -> CHECK -> BET
        for lbl, ch in kids.items():
            if CHECK_RE.search(str(lbl)):
                kids2 = self._normalize_children(ch)
                for lbl2, ch2 in kids2.items():
                    if BET_RE.search(str(lbl2)):
                        return ch2, [str(lbl), str(lbl2)]

        # shallow DFS fallback
        stack: List[Tuple[List[str], Dict[str, Any]]] = [([], root)]
        while stack:
            path, node = stack.pop()
            for lbl, ch in self._normalize_children(node).items():
                np = path + [str(lbl)]
                if BET_RE.search(str(lbl)):
                    return ch, np
                if isinstance(ch, dict):
                    stack.append((np, ch))

        return None, []

    def _resolve_facing_bet_bb(
        self,
        *,
        pot_bb: float,
        size_pct: Optional[int],
        via_path: List[str],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Choose faced bet size:
          1) MENU hint (size_pct * pot)
          2) parse from label on via_path[-1] (percent or absolute)
        """
        meta: Dict[str, Any] = {"method": None, "via_label": None, "size_pct": size_pct}

        # 1) menu hint wins
        if size_pct is not None:
            v = (float(size_pct) / 100.0) * float(pot_bb)
            if v > 1e-9:
                meta["method"] = "menu"
                return v, meta

        # 2) parse from label
        if via_path:
            meta["via_label"] = via_path[-1]
            pct, to_bb = self._extract_bet_size(via_path[-1], pot_bb)
            if to_bb is not None and to_bb > 0:
                meta["method"] = "label_abs_or_derived"
                return float(to_bb), meta

        return None, meta

    def _extract_bet_size(self, label: str, pot_bb: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Returns (pct, to_bb) where pct is percent-of-pot if explicitly present.
        Handles:
          - 'BET 33%' → pct=33, to_bb=0.33*pot
          - 'BET to 9.0' → to_bb=9.0
          - 'BET 2.000000' → to_bb=2.0 (absolute)
          - 'DONK 50%' etc
        """
        lab = str(label or "").strip()

        # absolute "BET 2.000000" or "RAISE 23.000000"
        m = re.match(r'^\s*(?:BET|RAISE)\s+(\d+(?:\.\d+)?)\s*$', lab, flags=re.IGNORECASE)
        if m:
            try:
                to_bb = float(m.group(1))
                return None, to_bb
            except Exception:
                pass

        # explicit percent
        m = re.search(r'\b(\d+(?:\.\d+)?)\s*%', lab)
        pct = float(m.group(1)) if m else None

        # explicit "to <bb>"
        m = re.search(r'\bto\s+(\d+(?:\.\d+)?)\b', lab, flags=re.IGNORECASE)
        to_bb = float(m.group(1)) if m else None

        # derive absolute from pct if needed
        if to_bb is None and pct is not None:
            to_bb = (pct / 100.0) * float(pot_bb)

        return pct, to_bb

    # ============================================================
    # IO / normalization helpers
    # ============================================================
    def _open_json_any(self, path: str) -> Dict[str, Any]:
        p = str(path)
        blob: bytes
        with open(p, "rb") as fh:
            blob = fh.read()

        if p.endswith(".gz"):
            with gzip.GzipFile(fileobj=io.BytesIO(blob)) as gz:
                txt = gz.read().decode("utf-8", errors="replace")
        else:
            txt = blob.decode("utf-8", errors="replace")

        # Sometimes solver dumps extra logs; best-effort parse first JSON object
        try:
            return json.loads(txt)
        except Exception:
            obj = self._first_json_object(txt)
            if obj:
                try:
                    return json.loads(obj)
                except Exception:
                    return {}
        return {}

    def _first_json_object(self, txt: str) -> Optional[str]:
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

    def _renorm(self, acts: List[str], mix: List[float]) -> Tuple[List[str], List[float]]:
        mix = [max(0.0, float(x)) for x in mix]
        s = sum(mix)
        if s <= 0:
            return (acts, [1.0 / len(acts)] * len(acts)) if acts else ([], [])
        return acts, [x / s for x in mix]

    def _renorm_map(self, d: ActionMix) -> ActionMix:
        s = sum(float(v) for v in d.values())
        if s <= 1e-12:
            return d
        return {k: float(v) / s for k, v in d.items()}

    def _sum(self, d: ActionMix) -> float:
        return float(sum(float(v) for v in d.values()))

    def _fail(self, ex: SolverExtraction, reason: str) -> SolverExtraction:
        ex.ok = False
        ex.reason = reason
        return ex