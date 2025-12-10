# ml/inference/context_infer.py
from typing import Tuple, Optional, Dict, Any, List
import re

def action_seq_for_ctx(ctx: str) -> list[str]:
    c = (ctx or "").upper()
    if c == "VS_OPEN":       return ["RAISE", "CALL", ""]
    if c == "VS_3BET":       return ["RAISE", "3BET", "CALL"]
    if c == "VS_4BET":       return ["RAISE", "3BET", "4BET"]
    if c == "LIMPED_SINGLE": return ["LIMP", "CHECK", ""]
    return ["", "", ""]

class ContextInferer:
    VALID = {"VS_OPEN", "VS_3BET", "VS_4BET", "LIMPED_SINGLE"}

    @staticmethod
    def _norm_ctx(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        s = str(raw).strip().upper().replace("-", "_").replace(" ", "_")
        # tolerant aliases
        if s in {"BLIND_VS_STEAL", "STEAL", "SRP", "SINGLE_RAISED"}:
            return "VS_OPEN"
        if s in {"LIMPED", "LIMP"}:
            return "LIMPED_SINGLE"
        return s if s in ContextInferer.VALID else None

    @staticmethod
    def _from_action_seq(seq: Optional[List[str]]) -> Optional[str]:
        if not seq:
            return None
        S = [str(t).upper() for t in seq]
        if any("4BET" in t for t in S): return "VS_4BET"
        if any("3BET" in t for t in S): return "VS_3BET"
        if any("RAISE" in t for t in S): return "VS_OPEN"
        if any("LIMP" in t or "CHECK" in t for t in S): return "LIMPED_SINGLE"
        return None

    @staticmethod
    def _from_history(lines: Optional[List[str]]) -> Tuple[Optional[str], Optional[str]]:
        if not lines:
            return None, None
        pre = [l for l in lines if isinstance(l, str) and re.match(r"^\s*(pre|pf)\s*:", l, re.I)]
        text = "\n".join(pre).lower()

        # 4bet first so it wins ties
        m4 = re.search(r"\b4\s*bet\b|\bfour\s*bet\b", text)
        if m4: return "VS_4BET", m4.group(0)
        m3 = re.search(r"\b3\s*bet\b|\bthree\s*bet\b|\bre-?\s*raise\b", text)
        if m3: return "VS_3BET", m3.group(0)
        mR = re.search(r"\braise\b|\bopens?\b|\biso(?:-|\s*)raise\b|\biso\b", text)
        if mR: return "VS_OPEN", mR.group(0)

        # Only limps/calls/checks/completes → limped
        if re.search(r"\blimp\b|\bcomplet(es|e)d?\b|\bcall(s|ed)?\b|\bchecks?\b", text):
            return "LIMPED_SINGLE", "limp_only"
        return None, None

    @classmethod
    def infer_with_reason(cls, req: Any) -> Tuple[Optional[str], Dict[str, Any]]:
        # 1) explicit ctx
        ctx_raw = getattr(req, "ctx", None) or getattr(getattr(req, "raw", {}) or {}, "get", lambda *_: None)("ctx")
        ctx_norm = cls._norm_ctx(ctx_raw)
        if ctx_norm:
            return ctx_norm, {"source": "explicit", "matched": ctx_norm}

        # 2) structured action_seq
        seq = getattr(req, "action_seq", None) or []
        c2 = cls._from_action_seq(seq)
        if c2:
            return c2, {"source": "action_seq", "matched": c2, "seq": list(seq)}

        # 3) history parse
        c3, snippet = cls._from_history(getattr(req, "actions_hist", None) or [])
        if c3:
            return c3, {"source": "history", "matched": c3, "snippet": snippet}

        # 4) default
        return "VS_OPEN", {"source": "default", "matched": "VS_OPEN"}