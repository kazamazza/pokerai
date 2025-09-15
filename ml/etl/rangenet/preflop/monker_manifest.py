from __future__ import annotations
import argparse, hashlib, json, os, re, tarfile, tempfile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client
from ml.etl.rangenet.preflop.monker_helpers import canon_action, canon_pos, parse_seq_from_stem, classify_context, \
    find_hu_pair_in_text, detect_raise_depth_from_text

# ---------- constants ----------
POS_ORDER = ["UTG","HJ","CO","BTN","SB","BB"]
POS_SET = set(POS_ORDER)
SIZE_TOKEN_RE = re.compile(r'(?P<num>\d+(\.\d+)?)x', re.IGNORECASE)

# ---------- sha helpers ----------
def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ---------- minimal fallbacks ----------
SEP_SPLIT   = re.compile(r"[ _\-\.\+]+")
CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])")

def _fallback_parse_seq_from_stem(stem: str) -> List[Dict[str,str]]:
    """
    Very forgiving tokenizer if vendor stems don't match your original parser.
    Splits on _, -, space, dot, plus and CamelCase. Then reuses your canon_*.
    """
    try:
        from ml.etl.rangenet.preflop.monker_helpers import canon_pos as _canon_pos, canon_action as _canon_action
    except Exception:
        # assume they are imported globally if helpers module isn't available
        _canon_pos, _canon_action = canon_pos, canon_action

    toks = []
    for tok in SEP_SPLIT.split(stem):
        if not tok: continue
        toks.extend(CAMEL_SPLIT.split(tok))
    toks = [t for t in (t.strip() for t in toks) if t]

    seq: List[Dict[str,str]] = []
    i = 0
    while i < len(toks):
        p = _canon_pos(toks[i])
        if not p:
            i += 1
            continue
        action = None
        if i + 1 < len(toks) and not _canon_pos(toks[i + 1]):
            action = toks[i + 1]
            i += 2
        else:
            i += 1
        rec = {"pos": p}
        if action is not None:
            rec["action"] = action
        seq.append(rec)
    return seq

# ---------- parse helpers ----------
def parse_stack_from_parts(parts: List[str]) -> Optional[int]:
    for p in parts:
        q = p.lower()
        if q.endswith("bb"):
            try:
                return int(q[:-2])
            except ValueError:
                pass
    return None

def _is_preflop_allin(seq_raw):
    preflop_only = []
    for e in seq_raw:
        a = canon_action(e.get("action"))
        if a in {"CBET","DONK","BET","CHECK"}:  # heuristically first postflop markers
            break
        preflop_only.append(a)
    return "ALL_IN" in preflop_only

def _continuers(seq: List[Dict[str,str]]) -> List[str]:
    """Players who reach flop (any non-fold preflop)."""
    keep: List[str] = []
    seen: set[str] = set()
    for e in seq:
        pos = canon_pos(e.get("pos"))
        if not pos: continue
        act = str(e.get("action","")).upper()
        if act in {"CALL","RAISE","BET","3BET","4BET","AI","ALL_IN","CHECK"}:
            seen.add(pos)
        elif act == "FOLD" and pos in seen:
            seen.remove(pos)
    return sorted(list(seen), key=lambda p: POS_ORDER.index(p))

def parse_sizes_from_stem(stem: str) -> Dict[str,str]:
    """Monker stems show actions but usually not sizes; still keep a best-effort hook."""
    s = stem.lower().replace('-', '_')
    sizes: Dict[str,str] = {}
    for tok in ["open","3bet","4bet"]:
        m = re.search(rf'{tok}\s*{SIZE_TOKEN_RE.pattern}', s)
        if m:
            sizes[tok] = f"{m.group('num')}x"; continue
        m = re.search(rf'{SIZE_TOKEN_RE.pattern}\s*_{tok}', s)
        if m and tok not in sizes:
            sizes[tok] = f"{m.group('num')}x"
    return sizes

# ---------- topology + route ----------
@dataclass(frozen=True)
class TopologyResult:
    topology: str                      # 'srp_hu' | '3bet_hu' | '4bet_hu' | 'limped_single' | 'limped_multi' | 'unknown'
    opener: Optional[str]
    three_bettor: Optional[str]
    callers_final: List[str]
    route: List[str]                   # e.g., ['open:2.5x','call'] or ['open','3bet','call'] if no sizes
    sizes_ref: Dict[str,str]

def infer_topology_and_route(seq_raw: List[Dict[str,str]], stem: str) -> TopologyResult:
    """
    Infer preflop topology by *raise order* (not literal tokens).
    - 1st raise  => opener
    - 2nd distinct raiser => three_bettor (3-bet)
    - 3rd+ raise => 4-bet+
    ALL_IN counts as a raise at its order; do NOT force it to 4-bet.
    """
    # Canonicalize (use your canon_action)
    acts: List[Tuple[str, str]] = []
    for e in seq_raw:
        pos = canon_pos(e.get("pos"))
        if not pos:
            continue
        act = canon_action(e.get("action"))  # e.g., "MIN","55%" → "RAISE"; "AI" → "ALL_IN"
        if act:
            acts.append((pos, act))

    sizes_ref = parse_sizes_from_stem(stem)

    opener: Optional[str] = None
    three_bettor: Optional[str] = None
    limp_count = 0
    raise_count = 0

    # Anything that is a *preflop* aggressive action should count as a raise-like event.
    # Keep "OPEN" if your canon_action can emit it; otherwise "RAISE"/"BET" + "ALL_IN" suffice.
    RAISELIKE = {"OPEN", "RAISE", "BET", "ALL_IN", "3BET", "4BET", "5BET"}

    for pos, act in acts:
        if act == "LIMP":
            limp_count += 1
            continue

        if act in RAISELIKE:
            if raise_count == 0:
                # First raise → opener
                opener = pos
                raise_count = 1
            else:
                # Subsequent raises
                if three_bettor is None and pos != opener:
                    # First *different* raiser after opener → 3-bettor
                    three_bettor = pos
                    raise_count = max(raise_count, 2)
                else:
                    # Any further raise (by anyone) puts us in 4-bet+ territory
                    raise_count = max(raise_count, 3)
            continue

    # Who reaches the flop (any non-fold preflop)
    cont = _continuers(seq_raw)

    # Build route tokens (include sizes if we parsed any)
    def _tok(name: str) -> str:
        sz = sizes_ref.get(name.lower())
        return f"{name.lower()}:{sz}" if sz else name.lower()

    # ----- Topology selection -----

    # Limped (no raises before the flop)
    if raise_count == 0 and limp_count >= 1:
        if limp_count == 1 and set(cont) >= {"SB", "BB"}:
            return TopologyResult("limped_single", "LIMP", None, cont, ["limp", "check"], sizes_ref)
        return TopologyResult("limped_multi", "LIMP", None, cont, ["limp", "overcalls", "check"], sizes_ref)

    # Single-raised pot
    if raise_count == 1 and opener:
        route = [_tok("open"), "call"]
        topo = "srp_hu" if len(cont) == 2 else "srp_multi"
        callers_final = [p for p in cont if p != opener]
        return TopologyResult(topo, opener, None, callers_final, route, sizes_ref)

    # 3-bet pot
    if raise_count == 2 and opener and three_bettor:
        route = [_tok("open"), _tok("3bet"), "call"]
        topo = "3bet_hu" if len(cont) == 2 else "3bet_multi"
        # everyone except opener & 3-bettor are callers; include opener if he flats
        callers_final = [p for p in cont if p not in {opener, three_bettor}] + ([opener] if opener in cont else [])
        return TopologyResult(topo, opener, three_bettor, callers_final, route, sizes_ref)

    # 4-bet+ pot (includes jams that were the 3rd raise overall)
    if raise_count >= 3:
        route = [_tok("open"), _tok("3bet"), _tok("4bet"), "call"]
        topo = "4bet_hu" if len(cont) == 2 else "4bet_multi"
        return TopologyResult(topo, opener, three_bettor, cont, route, sizes_ref)

    # Fallback
    return TopologyResult("unknown", opener, three_bettor, cont, [], sizes_ref)

# ---------- IP/OOP rules ----------
def derive_ip_oop(topo: str, opener: Optional[str], three_bettor: Optional[str], cont: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if len(cont) != 2: return None, None
    a, b = cont[0], cont[1]

    if topo == "limped_single":
        if set(cont) == {"SB","BB"}: return "SB","BB"
        return (max(cont, key=lambda p: POS_ORDER.index(p)),
                min(cont, key=lambda p: POS_ORDER.index(p)))

    if topo.startswith("srp"):
        if opener:
            other = a if a != opener else b
            return opener, other

    if topo.startswith("3bet"):
        # BTN open → BB 3bet → BTN call ⇒ BTN IP
        if opener == "BTN" and three_bettor in {"SB","BB"} and "BTN" in cont:
            other = three_bettor if three_bettor in cont else (a if a != "BTN" else b)
            return "BTN", other
        # 3bettor on BTN ⇒ BTN IP
        if three_bettor == "BTN" and "BTN" in cont:
            other = a if a != "BTN" else b
            return "BTN", other
        # fallback: later seat IP
        return (max(cont, key=lambda p: POS_ORDER.index(p)),
                min(cont, key=lambda p: POS_ORDER.index(p)))

    if topo.startswith("4bet"):
        if "BTN" in cont:
            other = a if a != "BTN" else b
            return "BTN", other
        return (max(cont, key=lambda p: POS_ORDER.index(p)),
                min(cont, key=lambda p: POS_ORDER.index(p)))

    return None, None

# ---------- expected menu label (audit only) ----------
def expected_menu_id(
    topo: str,
    ip: Optional[str],
    oop: Optional[str],
    opener: Optional[str],
    three_bettor: Optional[str],
) -> Optional[str]:
    """
    Assign a canonical menu ID based on topology and role.
    These IDs must match exactly with keys in BET_SIZE_MENUS.
    """
    if topo == "srp_hu":
        if opener and ip:
            return "srp_hu.PFR_IP" if opener == ip else "srp_hu.PFR_OOP"
        return "srp_hu.unknown"

    if topo == "3bet_hu":
        if three_bettor and oop and three_bettor == oop:
            return "3bet_hu.Aggressor_OOP"
        if three_bettor and ip and three_bettor == ip:
            return "3bet_hu.Aggressor_IP"
        return "3bet_hu.unknown"

    if topo == "4bet_hu":
        if three_bettor and oop and three_bettor == oop:
            return "4bet_hu.Aggressor_OOP"
        if three_bettor and ip and three_bettor == ip:
            return "4bet_hu.Aggressor_IP"
        return "4bet_hu.unknown"

    if topo == "limped_single":
        return "limped_single.SB_IP"

    return None

# ---------- scanner ----------
def scan_monker(root: Path, rake_tier: str = "nl10_5pct_1bbcap") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # stage counters for debugging
    c_total = c_txt = c_allin = c_topo = c_hu = c_ip = 0

    for path in root.rglob("*.txt"):
        c_total += 1
        if path.name.startswith("._"):
            continue
        c_txt += 1

        parts = list(path.parts)
        stack_bb = parse_stack_from_parts(parts)

        # hint only
        hero_pos = canon_pos(path.parent.name)
        if hero_pos not in POS_SET:
            hero_pos = None

        stem = path.stem

        # prefer your parser; fallback if it returns too little
        seq_raw = parse_seq_from_stem(stem)
        if len(seq_raw) < 2:
            seq_raw = _fallback_parse_seq_from_stem(stem)

        # keep a count (info) but DO NOT exclude on AI/ALL_IN — Monker stems can contain AI postflop
        if _is_preflop_allin(seq_raw):
            c_allin += 1  # informational only

        ctx_info = classify_context(seq_raw)

        # 1) Try normal topology inference
        topo = infer_topology_and_route(seq_raw, stem)

        # 2) Fallback: infer HU pair + topology from path/stem text if unknown
        if topo.topology in {"unknown"}:
            text = str(path)  # include directories; more signal than stem alone
            a, b = find_hu_pair_in_text(text)
            if a and b:
                rd = detect_raise_depth_from_text(text)
                sizes_ref = parse_sizes_from_stem(stem)
                # Build a minimal TopologyResult consistent with poker heuristics
                if rd == 0:
                    topo = TopologyResult(
                        topology="limped_single" if {a,b} == {"SB","BB"} else "limped_multi",
                        opener="LIMP",
                        three_bettor=None,
                        callers_final=[a,b],
                        route=["limp","check"],
                        sizes_ref=sizes_ref
                    )
                elif rd == 1:
                    # SRP: if late-pos vs blind, late-pos is opener; else first seat token
                    if (a in {"BTN","CO"} and b in {"SB","BB"}) or (b in {"BTN","CO"} and a in {"SB","BB"}):
                        opener_fb = "BTN" if "BTN" in {a,b} else ("CO" if "CO" in {a,b} else a)
                    else:
                        opener_fb = a
                    topo = TopologyResult(
                        topology="srp_hu",
                        opener=opener_fb,
                        three_bettor=None,
                        callers_final=[a,b],
                        route=["open","call"],
                        sizes_ref=sizes_ref
                    )
                elif rd == 2:
                    # 3bet: blind is usually 3-bettor vs late-position opener
                    if (a in {"BTN","CO"} and b in {"SB","BB"}):
                        opener_fb, three_fb = a, b
                    elif (b in {"BTN","CO"} and a in {"SB","BB"}):
                        opener_fb, three_fb = b, a
                    else:
                        opener_fb, three_fb = a, b
                    topo = TopologyResult(
                        topology="3bet_hu",
                        opener=opener_fb,
                        three_bettor=three_fb,
                        callers_final=[a,b],
                        route=["open","3bet","call"],
                        sizes_ref=sizes_ref
                    )
                else:
                    topo = TopologyResult(
                        topology="4bet_hu",
                        opener=a,
                        three_bettor=b,
                        callers_final=[a,b],
                        route=["open","3bet","4bet","call"],
                        sizes_ref=sizes_ref
                    )
            # if we still can't infer anything, we will drop this file later

        topo = infer_topology_and_route(seq_raw, stem)

        # Fallback: infer HU pair + topology from path/stem text if unknown
        if topo.topology in {"unknown"}:
            text = str(path)  # include directories for more signal than stem alone
            a, b = find_hu_pair_in_text(text)
            if a and b:
                rd = detect_raise_depth_from_text(text)
                sizes_ref = parse_sizes_from_stem(stem)
                if rd == 0:
                    topo = TopologyResult(
                        topology="limped_single" if {a, b} == {"SB", "BB"} else "limped_multi",
                        opener="LIMP",
                        three_bettor=None,
                        callers_final=[a, b],
                        route=["limp", "check"],
                        sizes_ref=sizes_ref
                    )
                elif rd == 1:
                    # SRP: if late-pos vs blind, late-pos is opener; else first seat token
                    if (a in {"BTN", "CO"} and b in {"SB", "BB"}) or (b in {"BTN", "CO"} and a in {"SB", "BB"}):
                        opener_fb = "BTN" if "BTN" in {a, b} else ("CO" if "CO" in {a, b} else a)
                    else:
                        opener_fb = a
                    topo = TopologyResult(
                        topology="srp_hu",
                        opener=opener_fb,
                        three_bettor=None,
                        callers_final=[a, b],
                        route=["open", "call"],
                        sizes_ref=sizes_ref
                    )
                elif rd == 2:
                    # 3bet: blind usually 3-bets vs late-position opener
                    if (a in {"BTN", "CO"} and b in {"SB", "BB"}):
                        opener_fb, three_fb = a, b
                    elif (b in {"BTN", "CO"} and a in {"SB", "BB"}):
                        opener_fb, three_fb = b, a
                    else:
                        opener_fb, three_fb = a, b
                    topo = TopologyResult(
                        topology="3bet_hu",
                        opener=opener_fb,
                        three_bettor=three_fb,
                        callers_final=[a, b],
                        route=["open", "3bet", "call"],
                        sizes_ref=sizes_ref
                    )
                else:
                    topo = TopologyResult(
                        topology="4bet_hu",
                        opener=a,
                        three_bettor=b,
                        callers_final=[a, b],
                        route=["open", "3bet", "4bet", "call"],
                        sizes_ref=sizes_ref
                    )

        if topo.topology in {"unknown"}:
            continue
        c_topo += 1

        # 3) Determine HU continuers
        cont = _continuers(seq_raw)
        if len(cont) != 2:
            # Fallback: force HU continuers from path/stem if available
            aa, bb = find_hu_pair_in_text(str(path))
            if aa and bb:
                cont = [aa, bb]
            else:
                continue
        c_hu += 1

        # 4) IP/OOP from deterministic rules
        ip_pos, oop_pos = derive_ip_oop(topo.topology, topo.opener, topo.three_bettor, cont)
        if not ip_pos or not oop_pos:
            continue
        c_ip += 1

        # defender seat
        if topo.topology.startswith("srp"):
            defender = cont[0] if cont[0] != topo.opener else cont[1]
        elif topo.topology.startswith("3bet"):
            defender = cont[0] if cont[0] != topo.three_bettor else cont[1]
        else:
            defender = cont[0] if cont[0] != ip_pos else cont[1]

        route_key = ">".join(topo.route) if topo.route else ""
        scenario_key = "|".join([
            topo.topology,
            topo.opener or "NA",
            defender or "NA",
            ip_pos,
            str(stack_bb or "NA"),
            rake_tier,
            route_key
        ])

        file_sha1 = sha1_file(path)
        seq_canon = [{"pos": e["pos"], **({"action": canon_action(e.get("action"))} if "action" in e else {})}
                     for e in seq_raw]
        seq_json = json.dumps(seq_canon, sort_keys=True)
        sig = sha1_str(f"{stack_bb}|{seq_json}|{scenario_key}")

        rows.append({
            "topology": topo.topology,
            "stack_bb": stack_bb,
            "rake_tier": rake_tier,
            "opener": topo.opener,
            "three_bettor": topo.three_bettor,
            "defender": defender,
            "ip_actor_flop": ip_pos,
            "oop_actor_flop": oop_pos,
            "route": json.dumps(topo.route),
            "sizes_ref": json.dumps(topo.sizes_ref),
            "expected_menu_id": expected_menu_id(topo.topology, ip_pos, oop_pos, topo.opener, topo.three_bettor),
            "hero_pos_hint": hero_pos,
            "sequence": seq_json,
            "sequence_raw": json.dumps(seq_raw, sort_keys=True),
            "raise_depth": ctx_info.get("raise_depth"),
            "limp_count": ctx_info.get("limp_count"),
            "multiway": ctx_info.get("multiway"),
            "seen_positions": json.dumps(cont),
            "filename_stem": stem,
            "rel_path": str(path.relative_to(root)),
            "abs_path": str(path.resolve()),
            "file_sha1": file_sha1,
            "sig": sig,
            "source": "MonkerGuy",
            "scenario_key": scenario_key,
        })

    print(f"\n[scan] total files seen: {c_total}")
    print(f"[scan] .txt considered  : {c_txt}")
    print(f"[scan] stems flagged AI : {c_allin} (info only; not excluded)")
    print(f"[scan] topo inferred    : {c_topo}")
    print(f"[scan] HU continuers    : {c_hu}")
    print(f"[scan] IP/OOP derived   : {c_ip}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df[
        df["topology"].isin(["srp_hu","3bet_hu","limped_single","4bet_hu"]) &
        df["stack_bb"].notna() &
        df["ip_actor_flop"].notna() &
        df["oop_actor_flop"].notna() &
        df["scenario_key"].notna()
    ].copy()

    df.sort_values(["rel_path"], inplace=True)
    df = df.drop_duplicates(subset=["scenario_key"], keep="first").reset_index(drop=True)
    print(f"[scan] final manifest rows: {len(df)}")
    return df

# ---------- coverage ----------
def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["topology","opener","defender","stack_bb","rake_tier"]
    piv = (df.groupby(keys, dropna=False)
             .agg(n=("scenario_key","count"),
                  menus=("expected_menu_id","nunique"))
             .reset_index()
             .sort_values(keys))
    return piv

# ---------- IO ----------
def write_manifest(df: pd.DataFrame, out_parquet: Path, out_jsonl: Path | None = None):
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_parquet, index=False)
        print(f"✅ wrote manifest → {out_parquet}")
    except Exception as e:
        print(f"⚠️ Parquet write failed ({e}); writing JSONL fallback.")
        if out_jsonl is None:
            out_jsonl = out_parquet.with_suffix(".jsonl")
        with out_jsonl.open("w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec) + "\n")
        print(f"✅ wrote manifest (JSONL) → {out_jsonl}")

def write_coverage(df_cov: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_cov.to_csv(out_path, index=False)
    print(f"📊 wrote coverage report → {out_path}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", type=str, default=None, help="Local monker .tar.gz (preferred)")
    ap.add_argument("--s3-bucket", type=str, default="pokeraistore")
    ap.add_argument("--s3-key", type=str, default="data/vendor/monker.tar.gz")
    ap.add_argument("--out", type=str, default="data/artifacts/monker_manifest.parquet")
    ap.add_argument("--coverage", type=str, default="data/artifacts/monker_coverage.csv")
    ap.add_argument("--rake-tier", type=str, default="nl10_5pct_1bbcap")
    args = ap.parse_args()

    out = Path(args.out)
    cov = Path(args.coverage)

    # 1) obtain tarball
    if args.tar:
        local_gz = Path(args.tar)
        if not local_gz.exists():
            raise FileNotFoundError(f"--tar not found: {local_gz}")
        tmpdir_ctx = tempfile.TemporaryDirectory()
        tmpdir = Path(tmpdir_ctx.name)
        use_cleanup = True
    else:
        # S3 fallback
        try:
            s3c = S3Client()
        except Exception as e:
            raise RuntimeError("S3Client not available and --tar not provided") from e
        tmpdir_ctx = tempfile.TemporaryDirectory()
        tmpdir = Path(tmpdir_ctx.name)
        local_gz = tmpdir / "monker.tar.gz"
        print(args.s3_key)
        print(f"✅ Downloaded: s3://{args.s3_bucket}/{args.s3_key} → {local_gz}")
        s3c.download_file(args.s3_key, local_gz)
        use_cleanup = True

    # 2) extract
    extract_root = tmpdir / "monker"
    print(f"▶️ Extracting {local_gz} → {extract_root}")
    with tarfile.open(local_gz, "r:gz") as tar:
        tar.extractall(path=extract_root)
    print(f"✅ Extracted into {extract_root}")

    # 3) scan
    df = scan_monker(extract_root, rake_tier=args.rake_tier)
    if df.empty:
        print(f"⚠️ No usable HU postflop rows found under {extract_root}")
        if use_cleanup:
            tmpdir_ctx.cleanup()
        return

    # 4) write outputs
    write_manifest(df, out)
    df_cov = coverage_report(df)
    write_coverage(df_cov, cov)

    if use_cleanup:
        tmpdir_ctx.cleanup()
    print("🧹 Temp dir cleaned up")

if __name__ == "__main__":
    main()