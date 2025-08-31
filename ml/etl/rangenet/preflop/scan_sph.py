from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Any, Optional

def _ctx_from_dir(d: str) -> str:
    d = d.upper()
    # Map your directory names to your Ctx enum labels
    if d in {"SRP","OPEN","SINGLE_RAISED"}: return "SRP"
    if d in {"LIMPED_SINGLE","LIMP_SINGLE"}: return "LIMPED_SINGLE"
    if d in {"LIMPED_MULTI","LIMP_MULTI"}:   return "LIMPED_MULTI"
    return d

def _parse_pair(name: str) -> Optional[tuple[str,str]]:
    # expecting e.g. "BTN_vs_BB"
    toks = name.replace("-", "_").split("_vs_")
    if len(toks) != 2: return None
    ip, oop = toks[0].upper(), toks[1].upper()
    return ip, oop

def _is_json_169(p: Path) -> bool:
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            if "ip" in obj and isinstance(obj["ip"], list) and len(obj["ip"]) == 169: return True
            if "oop" in obj and isinstance(obj["oop"], list) and len(obj["oop"]) == 169: return True
        if isinstance(obj, list) and len(obj) == 169: return True
        return False
    except Exception:
        return False

def scan_sph(root: Path) -> pd.DataFrame:
    """
    Expect layout: root/CTX/IP_vs_OOP/<stack>bb/(ip.json|oop.json)  OR one file ranges.json with {"ip":[...],"oop":[...]}
    Produces one row per hero POV.
    """
    rows: List[Dict[str, Any]] = []
    for ctx_dir in root.iterdir():
        if not ctx_dir.is_dir(): continue
        ctx = _ctx_from_dir(ctx_dir.name)
        for pair_dir in ctx_dir.iterdir():
            if not pair_dir.is_dir(): continue
            pair = _parse_pair(pair_dir.name)
            if not pair: continue
            ip_pos, oop_pos = pair
            for stack_dir in pair_dir.iterdir():
                if not stack_dir.is_dir(): continue
                name = stack_dir.name.lower()
                if not name.endswith("bb"): continue
                try:
                    stack = int(name[:-2])
                except ValueError:
                    continue

                # Case A: ip.json / oop.json
                ip_file  = stack_dir / "ip.json"
                oop_file = stack_dir / "oop.json"
                if ip_file.exists() and _is_json_169(ip_file):
                    rows.append({
                        "stack_bb": stack,
                        "ip_pos": ip_pos,
                        "oop_pos": oop_pos,
                        "ctx": ctx,
                        "hero_pos": ip_pos,
                        "rel_path": str(ip_file.relative_to(root)),
                        "abs_path": str(ip_file.resolve()),
                        "payload_kind": "json169_ip",
                    })
                if oop_file.exists() and _is_json_169(oop_file):
                    rows.append({
                        "stack_bb": stack,
                        "ip_pos": ip_pos,
                        "oop_pos": oop_pos,
                        "ctx": ctx,
                        "hero_pos": oop_pos,
                        "rel_path": str(oop_file.relative_to(root)),
                        "abs_path": str(oop_file.resolve()),
                        "payload_kind": "json169_oop",
                    })

                # Case B: single file ranges.json
                both = stack_dir / "ranges.json"
                if both.exists() and _is_json_169(both):
                    # we’ll load & pick ip/oop at read time; index twice
                    rows.append({
                        "stack_bb": stack,
                        "ip_pos": ip_pos,
                        "oop_pos": oop_pos,
                        "ctx": ctx,
                        "hero_pos": ip_pos,
                        "rel_path": str(both.relative_to(root)),
                        "abs_path": str(both.resolve()),
                        "payload_kind": "json169_both_ip",
                    })
                    rows.append({
                        "stack_bb": stack,
                        "ip_pos": ip_pos,
                        "oop_pos": oop_pos,
                        "ctx": ctx,
                        "hero_pos": oop_pos,
                        "rel_path": str(both.relative_to(root)),
                        "abs_path": str(both.resolve()),
                        "payload_kind": "json169_both_oop",
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        # normalize types
        df["stack_bb"] = df["stack_bb"].astype("Int64")
        # keep columns consistent with Monker manifest where possible
    return df