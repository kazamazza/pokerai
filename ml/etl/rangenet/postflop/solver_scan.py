from typing import Any, Dict, Mapping
from collections import deque

def _u(x: Any) -> str: return str(x).upper()

def presence_scan(payload: Mapping[str, Any]) -> Dict[str, bool]:
    has_raise = has_call = has_fold = has_allin = False
    q = deque([payload])
    while q:
        n = q.popleft()
        acts = n.get("actions") or []
        U = [_u(a) for a in acts]
        if any(u.startswith("RAISE") or "RE-RAISE" in u or "RERAISE" in u or "MIN-RAISE" in u or "MINRAISE" in u for u in U):
            has_raise = True
        if any(u.startswith("CALL") for u in U): has_call = True
        if any(u.startswith("FOLD") for u in U): has_fold = True
        if any(("ALLIN" in u or "ALL-IN" in u or "JAM" in u) for u in U): has_allin = True
        ch = n.get("childrens") or n.get("children") or {}
        for v in getattr(ch, "values", lambda: [])():
            if isinstance(v, dict):
                q.append(v)
    return {"has_raise": has_raise, "has_call": has_call, "has_fold": has_fold, "has_allin": has_allin}