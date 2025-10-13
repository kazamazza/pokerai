import json, gzip, re
from typing import Any, Dict, Tuple, Optional, List, Sequence
from .solver_schema import SolverExtraction

ACTION_VOCAB = [
    "FOLD","CHECK","CALL",
    "BET_25","BET_33","BET_50","BET_66","BET_75","BET_100",
    "DONK_33",
    "RAISE_150","RAISE_200","RAISE_300","RAISE_400","RAISE_500",
    "ALLIN",
]

FOLD_RE  = re.compile(r"\bfold\b", re.IGNORECASE)
CALL_RE  = re.compile(r"\bcall\b", re.IGNORECASE)
CHECK_RE = re.compile(r"\bcheck\b", re.IGNORECASE)
BET_RE   = re.compile(r"\bbet\b", re.IGNORECASE)
RAISE_RE = re.compile(r"\braise\b", re.IGNORECASE)
ALLIN_RE = re.compile(r"\ball[-\s]*in\b|\bjam\b", re.IGNORECASE)

class TexasSolverExtractor:
    def __init__(self) -> None:
        pass

    # ---------- public API ----------

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
    ) -> SolverExtraction:
        """Single entry point. Returns a filled SolverExtraction (ok=True) or reason."""
        ex = SolverExtraction(
            ctx=ctx, ip_pos=ip_pos, oop_pos=oop_pos, board=board,
            pot_bb=float(pot_bb), stack_bb=float(stack_bb), bet_sizing_id=str(bet_sizing_id)
        )
        try:
            payload = self._open_json_any(path)
            root = payload.get("root", payload)
            if not isinstance(root, dict):
                return self._fail(ex, "malformed_root")

            # 1) IP root mix (not facing): CHECK / BET_xx
            ex.root_mix, root_meta = self._read_node_action_mix(root, mode="root", pot_bb=ex.pot_bb)
            ex.meta["root_meta"] = root_meta

            # 2) OOP facing mix (find first bet edge IP->BET; or CHECK->BET)
            bet_node, via_path = self._find_first_bet_node(root)
            ex.meta["facing_path"] = via_path
            if bet_node is not None:
                facing_pct, facing_to_bb = self._extract_bet_size(via_path[-1], ex.pot_bb)
                ex.facing_bet_bb = facing_to_bb if facing_to_bb is not None else (ex.pot_bb * (facing_pct or 0)/100.0)
                ex.facing_mix, face_meta = self._read_node_action_mix(
                    bet_node, mode="facing", pot_bb=ex.pot_bb, facing_bet_bb=ex.facing_bet_bb
                )
                ex.meta["facing_meta"] = face_meta

            # 3) Validate minimum contract
            if (ex.root_mix and self._sum(ex.root_mix) > 0) or (ex.facing_mix and self._sum(ex.facing_mix) > 0):
                # normalize just in case
                if ex.root_mix:  ex.root_mix  = self._renorm_map(ex.root_mix)
                if ex.facing_mix: ex.facing_mix = self._renorm_map(ex.facing_mix)
                ex.ok = True
                return ex

            return self._fail(ex, "zero_mass")

        except Exception as e:
            return self._fail(ex, f"exception: {e}")

    # ---------- core helpers ----------

    def _read_node_action_mix(
        self, node: Dict[str, Any], *, mode: str, pot_bb: float, facing_bet_bb: Optional[float]=None
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Read actions/mix at a node using (1) strategy tables (several layouts),
        (2) per-child weights, and map to ACTION_VOCAB. Includes robust size bucketing.
        """
        acts, mix, where, raw_labels = self._find_any_strategy(node)
        meta = {"where": where, "raw_actions": raw_labels}

        mapped: Dict[str, float] = {a: 0.0 for a in ACTION_VOCAB}
        for lbl, p in zip(acts, mix):
            lab = (lbl or "").lower().strip()
            if not lab or p <= 0: continue

            if FOLD_RE.search(lab):  mapped["FOLD"]  += p; continue
            if CHECK_RE.search(lab): mapped["CHECK"] += p; continue
            if CALL_RE.search(lab):  mapped["CALL"]  += p; continue
            if ALLIN_RE.search(lab): mapped["ALLIN"] += p; continue

            if BET_RE.search(lab):
                pct, to_bb = self._extract_bet_size(lbl, pot_bb)
                # root mode → bucket bet % of pot
                tok = self._bucket_bet_pct(pct if pct is not None else self._pct_from_to_bb(to_bb, pot_bb))
                mapped[tok] += p; continue

            if RAISE_RE.search(lab):
                pct, to_bb = self._extract_bet_size(lbl, pot_bb)
                # facing mode → bucket by raise-to / facing-bet
                tok = self._bucket_raise_to(to_bb, facing_bet_bb)
                mapped[tok] += p; continue

            # Fallback: unknown verbs → ignore

        return mapped, meta

    # ----- strategy discovery (schema tolerant) -----

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

    # ----- label parsing & bucketing -----

    def _extract_bet_size(self, label: str, pot_bb: float) -> Tuple[Optional[float], Optional[float]]:
        t = label.lower()
        # 33%
        m = re.search(r'(\d+(?:\.\d+)?)\s*%', t)
        if m:
            pct = float(m.group(1)); return pct, (pct/100.0)*pot_bb
        # 0.66x pot | 0.66 pot
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:x\s*)?pot', t)
        if m:
            pct = float(m.group(1))*100.0; return pct, (pct/100.0)*pot_bb
        # to 6.0bb
        m = re.search(r'to\s*(\d+(?:\.\d+)?)\s*bb', t)
        if m:
            to_bb = float(m.group(1)); return None, to_bb
        # bare number → assume bb
        m = re.search(r'\b(\d+(?:\.\d+)?)\b', t)
        if m:
            val = float(m.group(1)); return None, val
        return None, None

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

    # ----- structure utilities -----

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

    def _open_json_any(self, path: str) -> Dict[str, Any]:
        if str(path).endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

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