# tools/debug_extractor_on_file.py
import re, sys, json, gzip
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, cast, Literal

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from ml.models.policy_consts import ACTION_VOCAB
from ml.etl.rangenet.postflop.texas_solver_extractor import TexasSolverExtractor

S3_RE = re.compile(
    r"street=(?P<street>\d+)/pos=(?P<pos>[^/]+)/stack=(?P<stack>[\d.]+)/pot=(?P<pot>[\d.]+)/board=(?P<board>[0-9A-Za-z]{3,6})/acc=[^/]+/sizes=(?P<sizing>[^/]+)/[^/]+/[^/]+/size=(?P<size>\d{2,3})p"
)

def parse_ctx_from_path(p: str) -> Tuple[str, str, str, float, float, int, str]:
    """
    Returns (ip_pos, oop_pos, board, pot_bb, stack_bb, size_pct, menu_id)
    Works for:
      - S3-style keys
      - peek filenames like: <hash>__HJvBB__stk60__pot6.5__3s6h8s__srp_hu.Caller_OOP__sz33p.json
    """
    # Try S3-style path first
    m = S3_RE.search(p)
    if m:
        ip, oop = m.group("pos").split("v")
        return (
            ip, oop,
            m.group("board"),
            float(m.group("pot")),
            float(m.group("stack")),
            int(m.group("size")),
            m.group("sizing"),
        )

    # Fallback: parse peek filename by splitting on "__"
    name = Path(p).name
    parts = name.split("__")
    # Expected: [<hash>, <pos>, stk..., pot..., <board>, <menu>, sz...json]
    # We’ll be lenient and scan:
    ip, oop, board, pot, stack, menu, size = None, None, None, None, None, None, None
    for part in parts:
        if "v" in part and len(part) <= 6 and part.upper().count("V") == 1:
            # e.g., HJvBB / BTNvBB
            try:
                ip, oop = part.split("v")
            except Exception:
                pass
        elif part.startswith("stk"):
            stack = float(part.replace("stk", ""))
        elif part.startswith("pot"):
            pot = float(part.replace("pot", ""))
        elif re.fullmatch(r"[0-9A-Za-z]{6}", part) or re.fullmatch(r"[0-9A-Za-z]{3}", part):
            # board like 3s6h8s or AhKcQd (6 chars). Accept 3 as fallback.
            board = part
        elif part.startswith("sz") and part.endswith("p.json"):
            try:
                size = int(part[2:-6])  # between 'sz' and 'p.json'
            except Exception:
                pass
        elif "." in part and not part.endswith(".json"):
            # menu id e.g. srp_hu.Caller_OOP
            menu = part

    # Fill sensible defaults if missing
    if not (ip and oop):
        ip, oop = "BTN", "BB"
    if board is None:
        raise ValueError(f"Could not parse board from filename: {name}")
    if pot is None or stack is None:
        raise ValueError(f"Could not parse pot/stack from filename: {name}")
    if size is None:
        size = 33
    if menu is None:
        menu = "srp_hu.PFR_IP"

    return ip, oop, board, float(pot), float(stack), int(size), menu

def infer_ctx_from_menu(menu_id: str) -> str:
    u = menu_id.lower()
    if "4bet" in u: return "VS_4BET"
    if "3bet" in u: return "VS_3BET"
    if "limped" in u: return "LIMPED_SINGLE" if "single" in u else "LIMPED_MULTI"
    return "VS_OPEN"

def oop_is_caller(menu_id: str) -> bool:
    return "caller_oop" in menu_id.lower()

def root_bet_kind_for(menu_id: str) -> str:
    return "donk" if oop_is_caller(menu_id) else "bet"

def show_mix(title: str, mix: Dict[str, float]):
    nz = {k: round(v, 6) for k, v in mix.items() if v > 0}
    order = [a for a in ACTION_VOCAB if a in nz]
    print(f"\n[{title}] nonzero:")
    for k in order:
        print(f"  {k:>9s}: {nz[k]}")

def main():
    if len(sys.argv) < 2:
        print("usage: python tools/debug_extractor_on_file.py <path/to/output_result.json[.gz] or peek.json>")
        sys.exit(1)

    path = sys.argv[1]
    ip_pos, oop_pos, board, pot_bb, stack_bb, size_pct, menu_id = parse_ctx_from_path(path)
    ctx = infer_ctx_from_menu(menu_id)
    root_actor = "oop"
    root_bet_kind = root_bet_kind_for(menu_id)

    # Use your configured buckets; fine to hardcode for debug
    raise_mults = [1.5, 2.0, 3.0]

    x = TexasSolverExtractor()
    ex = x.extract(
        path=path,
        ctx=ctx,
        ip_pos=ip_pos,
        oop_pos=oop_pos,
        board=board,
        pot_bb=pot_bb,
        stack_bb=stack_bb,
        bet_sizing_id=menu_id,
        size_pct=size_pct,
        root_actor=root_actor,
        root_bet_kind=root_bet_kind,
        raise_mults=raise_mults,
    )

    print("\n=== Extractor result ===")
    print("ok:", ex.ok, "reason:", ex.reason)
    print("meta:", json.dumps(ex.meta, indent=2, default=str))

    if ex.root_mix:   show_mix("ROOT", ex.root_mix)
    if ex.facing_mix: show_mix("FACING", ex.facing_mix)

    if not ex.ok:
        sys.exit(2)

if __name__ == "__main__":
    main()