# file: inference/postflop_ctx_and_row.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Mapping
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- Context inference ---------------------------

CTX_CANON = {"LIMPED_SINGLE", "VS_OPEN", "VS_3BET", "VS_4BET", "BLIND_VS_STEAL"}

# Allowed (IP, OOP) pairs per your scenarios (IP→OOP).
ALLOWED_PAIRS: Dict[str, set[Tuple[str, str]]] = {
    "VS_OPEN": {
        ("UTG", "BB"), ("UTG", "SB"),
        ("HJ", "BB"),  ("HJ", "SB"),
        ("CO", "BB"),  ("CO", "SB"),
        ("BTN","BB"),  ("BTN","SB"),
    },
    "BLIND_VS_STEAL": {  # if distinct ctx is enabled
        ("BTN","BB"), ("BTN","SB"),
        ("CO","BB"),  ("CO","SB"),
    },
    "VS_3BET": {
        ("BTN","BB"), ("BTN","SB"),
        ("CO","BB"),  ("CO","SB"),
    },
    "VS_4BET": {
        ("BTN","BB"), ("BTN","SB"),
        ("CO","BB"),  ("CO","SB"),
    },
    "LIMPED_SINGLE": {("BB", "SB")},
}

# Pot bands used only when preflop history is unavailable.
POT_BANDS = {
    "LIMPED_SINGLE": (1.0, 3.0),
    "VS_OPEN": (4.0, 7.5),
    "VS_3BET": (10.0, 25.0),
    "VS_4BET": (26.0, 1000.0),
}


def _flop_ip(ip: str, oop: str) -> Tuple[str, str]:
    """Return (IP, OOP) at flop from two positions. BB is IP vs SB; otherwise later index is IP."""
    ip = ip.upper()
    oop = oop.upper()
    if {ip, oop} == {"SB", "BB"}:
        return ("BB", "SB")
    order = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
    try:
        return (ip, oop) if order.index(ip) > order.index(oop) else (oop, ip)
    except ValueError:
        # Unknown positions ⇒ assume the first is IP to avoid crashing; validation will catch it.
        return (ip, oop)


def _coerce_actions_hist(req: Any) -> List[Dict[str, Any]]:
    """
    Accepts:
      - dict-of-lists by street: {"preflop":[{type,...}, ...], "flop":[...], ...}
      - list[dict] with "street" and "action/type"
      - list[str] logs like "pre: BTN raises 2.5"
    Returns a normalized list of dicts with keys: street(int), type(str).
    """
    ah = getattr(req, "actions_hist", None) or []
    out: List[Dict[str, Any]] = []

    if isinstance(ah, dict):
        for st_name, events in ah.items():
            st = 0 if st_name.lower().startswith("pre") else (1 if st_name.lower().startswith("flop") else (2 if st_name.lower().startswith("turn") else 3))
            for e in (events or []):
                if isinstance(e, dict):
                    typ = (e.get("type") or e.get("action") or "").upper()
                else:
                    typ = ""
                out.append({"street": st, "type": typ})
        return out

    if isinstance(ah, list):
        for e in ah:
            if isinstance(e, dict):
                st = int((e.get("street") or 0))
                typ = (e.get("type") or e.get("action") or "").upper()
                out.append({"street": st, "type": typ})
            else:
                s = str(e or "")
                st = 0 if re.search(r"\bpre\b|\bpreflop\b", s, re.I) else (1 if "flop" in s.lower() else (2 if "turn" in s.lower() else 3))
                if re.search(r"\braise\b", s, re.I):
                    typ = "RAISE"
                elif re.search(r"\bcall\b", s, re.I):
                    typ = "CALL"
                elif re.search(r"\blimp\b", s, re.I):
                    typ = "LIMP"
                else:
                    typ = ""
                out.append({"street": st, "type": typ})
        return out

    return out


class ContextInferer:
    """
    Infers canonical ctx for postflop: one of
      {LIMPED_SINGLE, VS_OPEN, VS_3BET, VS_4BET, BLIND_VS_STEAL?}
    """

    @staticmethod
    def infer_with_reason(req: Any) -> Tuple[Optional[str], str]:
        try:
            street = int(getattr(req, "street", 0) or 0)
        except Exception:
            street = 0

        if street == 0:
            return None, "preflop_no_ctx"

        h = (getattr(req, "hero_pos", "") or "").upper()
        v = (getattr(req, "villain_pos", "") or "").upper()
        (ip, oop) = _flop_ip(h, v)

        hist = _coerce_actions_hist(req)
        pre = [a for a in hist if int(a.get("street", 0)) == 0]
        n_raise = sum(1 for a in pre if a.get("type") == "RAISE")
        n_limp = sum(1 for a in pre if a.get("type") == "LIMP")

        # 1) History-driven
        if pre:
            if n_raise == 0:
                ctx = "LIMPED_SINGLE"
            elif n_raise == 1:
                # Optional BvS specialization (kept canonical unless sidecar supports it)
                ctx = "VS_OPEN"
            elif n_raise == 2:
                ctx = "VS_3BET"
            else:
                ctx = "VS_4BET"
            ok = (ip, oop) in ALLOWED_PAIRS.get(ctx, set())
            if not ok and ctx == "VS_OPEN":
                # If BvS pairs but you want distinct ctx, you'd switch here.
                if (ip, oop) in ALLOWED_PAIRS.get("BLIND_VS_STEAL", set()):
                    ctx = "BLIND_VS_STEAL"
                    ok = True
            if not ok:
                return None, f"pair_not_allowed: ctx={ctx} pair={(ip,oop)}"
            if ctx == "LIMPED_SINGLE" and {h, v} != {"SB", "BB"}:
                return None, "limp_requires_sb_bb"
            return ctx, "ok_history"

        # 2) Pot-band fallback
        pot = float(getattr(req, "pot_bb", 0.0) or 0.0)
        for ctx_name, (lo, hi) in POT_BANDS.items():
            if not (lo <= pot <= hi):
                continue

            # Hard guard: limped pots require SB vs BB seats
            if ctx_name == "LIMPED_SINGLE" and {h, v} != {"SB", "BB"}:
                continue  # skip impossible limp for these seats

            ctx = ctx_name
            ok = (ip, oop) in ALLOWED_PAIRS.get(ctx, set())

            # Optional: recognize BvS as a specialization of VS_OPEN if pairs match
            if not ok and ctx == "VS_OPEN" and (ip, oop) in ALLOWED_PAIRS.get("BLIND_VS_STEAL", set()):
                ctx = "BLIND_VS_STEAL"
                ok = True

            if not ok:
                continue  # try next band instead of failing

            return ctx, "ok_pot_band"

        # Tolerant fallback: tiny pot but seats not SB/BB → treat as SRP if pair fits
        try:
            limp_hi = POT_BANDS["LIMPED_SINGLE"][1]
        except KeyError:
            limp_hi = 3.0

        if pot <= limp_hi and (ip, oop) in ALLOWED_PAIRS.get("VS_OPEN", set()):
            return "VS_OPEN", "tolerate_small_pot_as_srp"

        return None, "insufficient_info"

    @staticmethod
    def infer_from_request(req: Any) -> Optional[str]:
        ctx, _ = ContextInferer.infer_with_reason(req)
        return ctx