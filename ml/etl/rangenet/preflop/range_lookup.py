import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd

POS_OPEN_TOKENS = {"OPEN", "RAISE"}
RERAISE_TOKENS  = {"RAISE", "ALL_IN", "3BET", "4BET", "5BET"}

def _parse_seq(seq_json: str):
    try:
        seq = json.loads(seq_json)
        return seq if isinstance(seq, list) else []
    except Exception:
        return []

def _is_srp_open_call(seq, ip_pos: str, oop_pos: str) -> bool:
    if not seq:
        return False
    # opener must be ip_pos and first action must be OPEN/RAISE
    if not (seq[0].get("pos") == ip_pos and seq[0].get("action") in POS_OPEN_TOKENS):
        return False
    re_raised = False
    for step in seq[1:]:
        pos = step.get("pos")
        act = step.get("action")
        if act in RERAISE_TOKENS:
            re_raised = True
        if pos == oop_pos:
            return (act == "CALL") and (not re_raised)
    return False

def _nearest_stack(target: float, available: list[float]) -> Optional[float]:
    if not available:
        return None
    return min(available, key=lambda s: abs(float(s) - float(target)))

def _load_range_file_to_compact(path: Path, min_keep: float = 0.0) -> str:
    """
    Vendor file format: 'AA:1.0,A2s:0.0,A2o:0.174,...'
    Convert to compact string 'AA,A2o:0.174,...' (omit entries with weight<=min_keep).
    """
    txt = Path(path).read_text().strip()
    if not txt:
        return ""
    pairs = []
    for tok in txt.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            hand, val = tok.split(":", 1)
            try:
                p = float(val)
            except ValueError:
                continue
            if p <= min_keep:
                continue
            if abs(p - 1.0) < 1e-9:
                pairs.append(hand)
            else:
                pairs.append(f"{hand}:{p:g}")
        else:
            # some files may include pure 'TT' without ':1.0'
            pairs.append(tok)
    return ",".join(pairs)

class PreflopRangeLookup:
    def __init__(self, manifest_parquet: str | Path):
        self.df = pd.read_parquet(manifest_parquet)
        # ensure required columns exist
        need = {"stack_bb","hero_pos","opener_pos","opener_action","sequence","abs_path"}
        missing = [c for c in need if c not in self.df.columns]
        if missing:
            raise ValueError(f"monker_manifest missing columns: {missing}")
        # coerce stack to float for nearest selection
        self.df["stack_bb"] = self.df["stack_bb"].astype(float)

    def ranges_for_pair(self, *, stack_bb: float, ip: str, oop: str) -> Tuple[str, str]:
        df = self.df
        # candidates by opener/hero
        cand = df[(df["opener_pos"] == ip) & (df["hero_pos"] == oop)]
        if cand.empty:
            # fallback: any rows that include both positions; last resort
            cand = df[(df["opener_pos"] == ip)]
            if cand.empty:
                return "", ""

        # exact stack first
        exact = cand[cand["stack_bb"] == float(stack_bb)]
        if exact.empty:
            # pick nearest stack available for this pair
            stacks = sorted(cand["stack_bb"].unique().tolist())
            ns = _nearest_stack(stack_bb, stacks)
            exact = cand[cand["stack_bb"] == ns]

        # filter to SRP open->call sequences
        exact = exact[exact["opener_action"].isin(POS_OPEN_TOKENS)]
        exact = exact[exact["sequence"].apply(lambda s: _is_srp_open_call(_parse_seq(s), ip, oop))]
        if exact.empty:
            return "", ""

        # choose the most common sequence (stability)
        grp = exact.groupby(["sequence"]).size().sort_values(ascending=False)
        seq_json = grp.index[0]
        row = exact[exact["sequence"] == seq_json].iloc[0]
        path = Path(row["abs_path"])

        # IP is opener; OOP is hero
        ip_rng  = _load_range_file_to_compact(path, min_keep=0.0)
        oop_rng = _load_range_file_to_compact(path, min_keep=0.0)  # same file for both sides in vendor packs

        return ip_rng, oop_rng