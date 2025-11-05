#!/usr/bin/env python3
import argparse, json, os, random, re, sys, tempfile, subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# ✅ Import the SAME extractor class you use in the builder
# Adjust this import only if your project path differs.
from ml.etl.rangenet.postflop.build_postflop_policy import TexasSolverExtractor

S3_DIR_RE = re.compile(
    r"street=(?P<street>\d+)/"
    r"pos=(?P<pos>[^/]+)/"
    r"stack=(?P<stack>[\d.]+)/"
    r"pot=(?P<pot>[\d.]+)/"
    r"board=(?P<board>[0-9A-Za-z]{3,6})/"
    r"acc=[^/]+/"
    r"sizes=(?P<sizing>[^/]+)"
    r"(?:/[^/]{2}/[^/]+)?"
    r"(?:/size=(?P<size>\d{2,3})p)?"
    r"(?:/(?:output_result\.json\.gz|result\.json))?$"
)

def parse_menu_id_from_key(key: str) -> str | None:
    m = S3_DIR_RE.search(key)
    return m.group("sizing") if m else None

def parse_sizes_hint(s: str) -> dict[str, list[int]]:
    """
    Format: 'srp_hu.PFR_IP=33,66;srp_hu.Caller_OOP=33;3bet_hu.Aggressor_IP=33,66;...'
    """
    out: dict[str, list[int]] = {}
    if not s:
        return out
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        sizes = []
        for t in v.split(","):
            t = t.strip()
            if not t:
                continue
            try:
                sizes.append(int(t))
            except:
                pass
        if sizes:
            out[k.strip()] = sorted(set(sizes))
    return out

# Built-in fallback (matches your NL10 bet_menus)
DEFAULT_MENU_SIZES = {
    "srp_hu.PFR_IP":         [33, 66],
    "srp_hu.Caller_OOP":     [33],
    "bvs.Any":               [33, 66],
    "3bet_hu.Aggressor_IP":  [33, 66],
    "3bet_hu.Aggressor_OOP": [33, 66],
    "4bet_hu.Aggressor_IP":  [33],
    "4bet_hu.Aggressor_OOP": [33],
    "limped_single.SB_IP":   [33],
    "limped_multi.Any":      [33],
}

def sizes_for_menu(menu_id: str, sizes_map: dict[str, list[int]]) -> list[int]:
    if menu_id in sizes_map:
        return sizes_map[menu_id]
    # try loose match by suffix (e.g., menu ids that embed position role)
    for k, v in sizes_map.items():
        if k.lower() in menu_id.lower():
            return v
    return [33]  # last-resort

def s3_list_prefix(bucket: str, prefix: str) -> list[str]:
    # returns child "folder" names (no leading/trailing slash)
    cmd = ["aws", "s3", "ls", f"s3://{bucket}/{prefix.rstrip('/')}/"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError:
        return []
    dirs = []
    for line in out.splitlines():
        # aws s3 ls prints "                           PRE ab/" for prefixes
        m = re.search(r"PRE\s+([^/]+)/", line)
        if m:
            dirs.append(m.group(1))
    return dirs

def expand_to_concrete_paths(p: str, *, bucket: str, prefix: str, sizes_map: dict[str, list[int]]) -> list[str]:
    """
    Expand a manifest entry to concrete S3 object candidates by:
      1) Identifying the menu id (sizes=<menu>)
      2) If /size=XXp missing, enumerate the shard fanout (<shard2>/<hash40>)
      3) Emit .../<shard2>/<hash40>/size=XXp/output_result.json.gz (and result.json fallback)
    """
    # Already a concrete file?
    if p.startswith("s3://"):
        s3_key = p[len("s3://"+bucket+"/"):] if p.startswith(f"s3://{bucket}/") else p
    else:
        s3_key = p

    # If it already ends with a file, just return it
    if re.search(r"/(output_result\.json\.gz|result\.json)$", s3_key):
        return [f"s3://{bucket}/{s3_key}"]

    # Require sizes=<menu> to exist
    m = re.search(r"(.*?/sizes=)([^/]+)(?:/(.*))?$", s3_key)
    if not m:
        # Not a solver key; return as-is (lets caller skip)
        return [f"s3://{bucket}/{s3_key}"]

    base_before_menu, menu_id, tail = m.group(1), m.group(2), (m.group(3) or "")
    base_dir = f"{base_before_menu}{menu_id}"

    # If size=XXp already present in tail, just finish it
    if re.search(r"/size=\d{2,3}p(?:/.*)?$", tail):
        base = f"{base_dir}/{tail.strip('/')}"
        return [
            f"s3://{bucket}/{base}/output_result.json.gz",
            f"s3://{bucket}/{base}/result.json",
        ]

    # Otherwise we need to walk shard fanout: <shard2>/<hash40>
    # First-level dirs (2-hex)
    lvl1 = s3_list_prefix(bucket, base_dir)
    # Heuristic: if none found, maybe there is no fanout; try directly with size
    if not lvl1:
        sizes = sizes_for_menu(menu_id, sizes_map)
        cands = []
        for sp in sizes:
            cands.append(f"s3://{bucket}/{base_dir}/size={sp}p/output_result.json.gz")
            cands.append(f"s3://{bucket}/{base_dir}/size={sp}p/result.json")
        return cands

    # Limit fanout breadth to keep it fast in a smoke test
    lvl1 = lvl1[:16]  # sample first 16 shards

    cands = []
    sizes = sizes_for_menu(menu_id, sizes_map)
    for d1 in lvl1:
        # second level under each shard
        lvl2 = s3_list_prefix(bucket, f"{base_dir}/{d1}")
        lvl2 = lvl2[:16] if lvl2 else []
        for d2 in lvl2:
            for sp in sizes:
                base = f"{base_dir}/{d1}/{d2}/size={sp}p"
                cands.append(f"s3://{bucket}/{base}/output_result.json.gz")
                cands.append(f"s3://{bucket}/{base}/result.json")
    return cands

def download_first_ok(candidates: list[str]) -> str | None:
    """Try each S3 candidate and return the first successfully downloaded local path."""
    for s3_uri in candidates:
        # download to temp; if fails, try next
        try:
            import tempfile
            suffix = ".json.gz" if s3_uri.endswith(".gz") else ".json"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.close()
            subprocess.run(["aws", "s3", "cp", s3_uri, tmp.name],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return tmp.name
        except subprocess.CalledProcessError:
            try: os.remove(tmp.name)
            except: pass
            continue
    return None

def concrete_keys_for_manifest_row(bucket: str, prefix: str, row_key: str) -> list[str]:
    """
    Turn a manifest 'directory-ish' key into concrete s3://... object paths to try
    (in priority order).
    """
    # row_key may or may not already start with prefix
    base = f"s3://{bucket}/{row_key.lstrip('/')}"
    m = S3_DIR_RE.search(base)
    if not m:
        # If the manifest already stored a full concrete key, just try it
        return [base]

    # Build candidates
    cand = []
    size = m.group("size")
    # Make sure we end at the directory that contains files (strip any trailing file)
    dir_base = re.sub(r"/(?:output_result\.json\.gz|result\.json)$", "", base).rstrip("/")

    if size:
        cand.append(f"{dir_base}/output_result.json.gz")
        cand.append(f"{dir_base}/result.json")
        # Also try without /size=XXp in case the object lives one level up
        dir_wo_size = re.sub(r"/size=\d{2,3}p$", "", dir_base)
        cand.append(f"{dir_wo_size}/output_result.json.gz")
        cand.append(f"{dir_wo_size}/result.json")
    else:
        cand.append(f"{dir_base}/output_result.json.gz")
        cand.append(f"{dir_base}/result.json")

    # de-dup
    out, seen = [], set()
    for c in cand:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def aws_cp(s3_uri: str) -> str | None:
    """Download s3://... to a temp file. Returns local path or None."""
    import tempfile, subprocess, os
    local = tempfile.NamedTemporaryFile(delete=False,
                                        suffix=".json.gz" if s3_uri.endswith(".gz") else ".json")
    local.close()
    try:
        subprocess.run(["aws", "s3", "cp", s3_uri, local.name],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return local.name
    except subprocess.CalledProcessError:
        try: os.remove(local.name)
        except: pass
        return None

def is_s3(p: str) -> bool:
    return p.startswith("s3://")

def normalize_s3_object_key(k: str) -> List[str]:
    """
    Given a manifest 'directory-ish' key, produce concrete object keys to try.
    Priority:
      1) .../size=XXp/output_result.json.gz (if size present)
      2) .../size=XXp/result.json
      3) .../output_result.json.gz
      4) .../result.json
    """
    cand = []
    m = S3_DIR_RE.search(k)
    if not m:
        return [k]  # best effort, maybe it's already concrete

    base = k.rstrip("/")
    size = m.group("size")

    if size:
        cand.append(f"{base}/output_result.json.gz")
        cand.append(f"{base}/result.json")
        # also try without the size segment just in case
        k_wo_size = re.sub(r"/size=\d{2,3}p$", "", base)
        cand.append(f"{k_wo_size}/output_result.json.gz")
        cand.append(f"{k_wo_size}/result.json")
    else:
        cand.append(f"{base}/output_result.json.gz")
        cand.append(f"{base}/result.json")

    # dedupe, keep order
    out = []
    seen = set()
    for c in cand:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def aws_cp_to_tmp(s3_path: str) -> Optional[str]:
    # try multiple concrete object keys until one copies
    for key in normalize_s3_object_key(s3_path):
        tmp = tempfile.NamedTemporaryFile(delete=False,
                                          suffix=".json.gz" if key.endswith(".gz") else ".json")
        tmp.close()
        try:
            subprocess.run(["aws", "s3", "cp", key, tmp.name],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return tmp.name
        except subprocess.CalledProcessError:
            try: os.remove(tmp.name)
            except: pass
            continue
    return None

def parse_ctx_from_path(p: str) -> Tuple[str, str, str, float, float, Optional[int], str]:
    """
    Returns (ip_pos, oop_pos, board, pot_bb, stack_bb, size_pct_or_None, menu_id).
    Works on S3 keys whether or not they include /size=XXp and a filename,
    and on peek filenames.
    """
    m = S3_DIR_RE.search(p)
    if m:
        ip, oop = m.group("pos").split("v")
        size = m.group("size")
        return (
            ip, oop,
            m.group("board"),
            float(m.group("pot")),
            float(m.group("stack")),
            int(size) if size else None,
            m.group("sizing"),
        )

    # Fallback: peek filenames
    name = Path(p).name
    parts = name.split("__")
    ip, oop, board, pot, stack, menu, size = None, None, None, None, None, None, None
    for part in parts:
        if "v" in part and len(part) <= 6 and part.upper().count("V") == 1:
            try: ip, oop = part.split("v")
            except: pass
        elif part.startswith("stk"):
            stack = float(part.replace("stk", ""))
        elif part.startswith("pot"):
            pot = float(part.replace("pot", ""))
        elif re.fullmatch(r"[0-9A-Za-z]{6}", part) or re.fullmatch(r"[0-9A-Za-z]{3}", part):
            board = part
        elif part.startswith("sz") and part.endswith("p.json"):
            try: size = int(part[2:-6])
            except: pass
        elif "." in part and not part.endswith(".json"):
            menu = part

    if not (ip and oop): ip, oop = "BTN", "BB"
    if board is None:      raise ValueError(f"Could not parse board from filename: {name}")
    if pot is None or stack is None:
        raise ValueError(f"Could not parse pot/stack from filename: {name}")
    if menu is None:       menu = "srp_hu.PFR_IP"
    return ip, oop, board, float(pot), float(stack), (int(size) if size else None), menu

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

def load_paths(args) -> List[str]:
    paths = []
    if args.manifest:
        df = pd.read_parquet(args.manifest)
        col = None
        for c in ["s3_key", "path", "key", "solver_key"]:
            if c in df.columns:
                col = c; break
        if not col:
            raise RuntimeError("Manifest missing a path column (s3_key/path/key/solver_key).")
        paths.extend(df[col].astype(str).tolist())

    if args.paths_file:
        with open(args.paths_file, "r") as f:
            for line in f:
                line=line.strip()
                if line: paths.append(line)

    if args.peek_dir:
        for p in Path(args.peek_dir).glob("*.json*"):
            paths.append(str(p))

    if args.limit and len(paths) > args.limit:
        random.seed(42)
        paths = random.sample(paths, args.limit)
    return paths

def normalize_raise_buckets(arg: str) -> List[float]:
    # "150,200,300" or "1.5,2.0,3.0" → always multipliers
    out = []
    for tok in arg.split(","):
        v = float(tok)
        out.append(v/100.0 if v > 10.0 else v)
    return sorted({x for x in out if x > 1.0})

def parse_raises_from_raw(raw_actions: List[str]) -> List[float]:
    sizes = []
    for a in raw_actions or []:
        a = str(a).upper()
        if a.startswith("RAISE"):
            m = re.search(r"RAISE\s+(\d+(?:\.\d+)?)", a)
            if m:
                try: sizes.append(float(m.group(1)))
                except: pass
    return sizes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, help="Parquet with s3_key/path column")
    ap.add_argument("--paths-file", type=str, help="Text file with paths (local or s3://)")
    ap.add_argument("--peek-dir", type=str, help="Directory of peek JSON/JSON.GZ")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--raise-buckets", type=str, default="150,200,300",
                    help="Allowed buckets (percent or multipliers)")
    ap.add_argument("--show", type=int, default=12, help="Show N example files")
    ap.add_argument("--s3-bucket", type=str, required=True,
                    help="S3 bucket where solver outputs live (e.g. poker-nn-store)")
    ap.add_argument("--s3-prefix", type=str, default="solver/outputs/v1",
                    help="Prefix inside the bucket (default: solver/outputs/v1)")
    ap.add_argument("--sizes-hint", type=str, default="",
                    help="menu->sizes mapping like 'srp_hu.PFR_IP=33,66;srp_hu.Caller_OOP=33;...'")
    args = ap.parse_args()

    paths = load_paths(args)
    if not paths:
        print("No inputs. Provide --manifest / --paths-file / --peek-dir.")
        sys.exit(1)

    # Build sizes map (hint overrides defaults)
    sizes_map = DEFAULT_MENU_SIZES.copy()
    sizes_map.update(parse_sizes_hint(args.sizes_hint))

    allowed_mults = normalize_raise_buckets(args.raise_buckets)
    allowed_tokens = [f"RAISE_{int(round(m*100))}" for m in allowed_mults]

    # Tallies
    counts: Dict[str,int] = {t:0 for t in allowed_tokens}
    counts.update({"CALL":0,"ALLIN":0,"FOLD":0})
    processed = 0
    had_raise = 0
    examples = []

    x = TexasSolverExtractor()

    for p in paths:
        # Expand to concrete file candidates (respecting size=XXp)
        if is_s3(p) or p.startswith(args.s3_prefix):
            cands = expand_to_concrete_paths(p, bucket=args.s3_bucket,
                                             prefix=args.s3_prefix, sizes_map=sizes_map)
        else:
            # local peek file
            cands = [p]

        local = None
        chosen_key_for_ctx = None
        try:
            # Download/choose first valid cand
            if len(cands) == 1 and os.path.exists(cands[0]):
                local = cands[0]
                chosen_key_for_ctx = cands[0]
            else:
                local = download_first_ok(cands)
                chosen_key_for_ctx = next((c for c in cands if c.startswith("s3://")),
                                          cands[0] if cands else None)

            if not local:
                print(f"[warn] could not fetch any object for {p}")
                continue

            key_for_ctx = chosen_key_for_ctx or p
            ip_pos, oop_pos, board, pot_bb, stack_bb, size_pct, menu_id = parse_ctx_from_path(key_for_ctx)
            ctx = infer_ctx_from_menu(menu_id)
            root_actor = "oop"
            root_bet_kind = root_bet_kind_for(menu_id)

            ex = x.extract(
                path=local,
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
                raise_mults=[float(m) for m in allowed_mults],
            )

            processed += 1
            if ex.facing_mix:
                for k, v in ex.facing_mix.items():
                    if v > 0:
                        if k in counts:
                            counts[k] += 1
                        elif k.startswith("RAISE_"):
                            counts.setdefault(k, 0)
                            counts[k] += 1
                if any(k.startswith("RAISE_") and v > 0 for k, v in ex.facing_mix.items()):
                    had_raise += 1

                if len(examples) < args.show:
                    raw = (ex.meta.get("facing_meta") or {}).get("raw_actions") or []
                    chosen = sorted([k for k, v in ex.facing_mix.items() if k.startswith("RAISE_") and v > 0])

                    facing_meta = ex.meta.get("facing_meta") or {}
                    dbg = facing_meta.get("facing_debug") or []

                    examples.append({
                        "path": key_for_ctx,
                        "pot": ex.pot_bb,
                        "stack": ex.stack_bb,
                        "faced_bet_bb": ex.facing_bet_bb,
                        "raw_raises_to": parse_raises_from_raw(raw),
                        "chosen_tokens": chosen,
                        "raw_actions_head": raw[:8],
                        # NEW: include the first few debug rows so you can inspect p/to_bb/mults/bucket
                        "debug_first_rows": dbg[:8],
                    })

        except Exception as e:
            print(f"[warn] failed on {p}: {e}")
        finally:
            if local and is_s3(p) and os.path.exists(local):
                try: os.remove(local)
                except: pass

        if processed >= args.limit:
            break

    print("\n=== RAISE bucket tallies ===")
    for k in sorted(counts.keys()):
        print(f"{k:>10s}: {counts[k]}")
    print(f"\nprocessed: {processed} | had_raise_nodes: {had_raise}")

    print("\n=== Examples ===")
    for exi in examples:
        print(json.dumps(exi, indent=2, default=str))

if __name__ == "__main__":
    main()