# ml/validation/validate_population_ctx.py
from __future__ import annotations
import argparse, gzip, json, sys, math, random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterable

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from infra.storage.s3_client import S3Client
from ml.core.types import Street, Act, Ctx

POS_ID_TO_NAME = {0:"UTG",1:"HJ",2:"CO",3:"BTN",4:"SB",5:"BB"}
LATE_POS_IDS = {2,3,4}  # CO/BTN/SB → treat CO/BTN as “steal”, SB optional

# --- IO helpers ---
def read_jsonl_gz(path: Path) -> Iterable[Dict[str, Any]]:
    openf = gzip.open if str(path).endswith(".gz") else open
    with openf(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def index_hands_by_id(hands_path: Path, needed_ids: set[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for obj in read_jsonl_gz(hands_path):
        hid = obj.get("hand_id")
        if hid in needed_ids:
            out[hid] = obj
    return out

def get_actor_name_by_pos_id(hand: Dict[str, Any], pos_id: int) -> Optional[str]:
    pos_map = hand.get("position_by_player") or {}
    for player, info in pos_map.items():
        if int(info["id"]) == int(pos_id):
            return player
    return None

# --- Preflop derivation ---
def derive_preflop_ctx(hand: Dict[str, Any], hero_name: str) -> Tuple[int, int, Optional[int]]:
    """
    Returns (ctx_id, hero_first_act_id, n_limpers_before_raise)
    """
    actions = [a for a in hand.get("actions", []) if int(a["street"]) == Street.PREFLOP]
    if not actions:
        return Ctx.OPEN, None, None  # degenerate

    # Count limps before first raise
    n_limpers = 0
    first_raise_actor = None
    raises_before_hero = 0

    # find hero first action index
    hero_idx = None
    for i, a in enumerate(actions):
        if a.get("actor") == hero_name:
            hero_idx = i
            break

    # If hero never acts preflop (weird), just safest default
    if hero_idx is None:
        return Ctx.OPEN, None, None

    # scan up to hero first action
    for i in range(hero_idx):
        a = actions[i]
        act = int(a.get("act"))
        if act == Act.CALL and first_raise_actor is None:
            n_limpers += 1
        if act in (Act.RAISE, Act.ALL_IN):
            if first_raise_actor is None:
                first_raise_actor = a.get("actor")
            raises_before_hero += 1

    hero_first_act_id = int(actions[hero_idx].get("act"))

    # Limped buckets (no raise yet)
    if raises_before_hero == 0:
        if n_limpers >= 2:
            return (Ctx.LIMPED_MULTI, hero_first_act_id, n_limpers)
        if n_limpers == 1:
            return (Ctx.LIMPED_SINGLE, hero_first_act_id, n_limpers)
        # no limps yet; if hero is SB/BB facing late position open? Not yet (no raise).
        # If hero raises here with no limps → OPEN
        return (Ctx.OPEN, hero_first_act_id, n_limpers)

    # There is at least 1 raise before hero acts → VS_OPEN/VS_3BET/VS_4BET
    if raises_before_hero == 1:
        # Extra: blind vs steal (hero in blind, opener in late positions)
        opener = first_raise_actor
        opener_pos_id = None
        pos_map = hand.get("position_by_player") or {}
        if opener in pos_map:
            opener_pos_id = int(pos_map[opener]["id"])
        hero_pos_map = hand.get("position_by_player") or {}
        hero_pos_id = None
        if hero_name in hero_pos_map:
            hero_pos_id = int(hero_pos_map[hero_name]["id"])
        if hero_pos_id in (4,5) and opener_pos_id in LATE_POS_IDS:
            return (Ctx.BLIND_VS_STEAL, hero_first_act_id, n_limpers)
        return (Ctx.VS_OPEN, hero_first_act_id, n_limpers)

    if raises_before_hero == 2:
        return (Ctx.VS_3BET, hero_first_act_id, n_limpers)
    # 3 or more raises before hero acts → VS_4BET (or higher compressed)
    return (Ctx.VS_4BET, hero_first_act_id, n_limpers)

# --- Postflop derivation (basic) ---
def derive_postflop_ctx(hand: Dict[str, Any], street_id: int, hero_name: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (ctx_id, hero_first_act_id) for FLOP/TURN/RIVER.
    Heuristics:
      - VS_CBET: hero’s first decision occurs *facing a bet* from preflop aggressor on the FLOP.
      - VS_DONK: hero’s first decision occurs *facing a bet* from the out-of-position player who was NOT aggressor previous street.
    """
    actions = [a for a in hand.get("actions", []) if int(a["street"]) == int(street_id)]
    if not actions:
        return None, None

    # find hero first action index
    hero_idx = None
    for i, a in enumerate(actions):
        if a.get("actor") == hero_name:
            hero_idx = i
            break
    if hero_idx is None:
        return None, None

    hero_first_act_id = int(actions[hero_idx].get("act"))

    # If hero faces a BET/ALL_IN before acting
    facing_bet = any(int(a.get("act")) in (Act.BET, Act.ALL_IN, Act.RAISE) for a in actions[:hero_idx])

    if facing_bet:
        # rough aggressor check: who last raised on previous street?
        preflop_agg = find_preflop_aggressor(hand)
        first_bettor = next((a.get("actor") for a in actions[:hero_idx] if int(a.get("act")) in (Act.BET, Act.ALL_IN)), None)
        if street_id == Street.FLOP and first_bettor == preflop_agg:
            return (Ctx.VS_CBET, hero_first_act_id)
        # else treat as donk for now (bet by non-aggressor)
        return (Ctx.VS_DONK, hero_first_act_id)

    # no bet before hero → no vs_* context; leave as None
    return None, hero_first_act_id

def find_preflop_aggressor(hand: Dict[str, Any]) -> Optional[str]:
    # last raiser preflop
    agg = None
    for a in hand.get("actions", []):
        if int(a.get("street")) != Street.PREFLOP: break
        if int(a.get("act")) in (Act.RAISE, Act.ALL_IN):
            agg = a.get("actor")
    return agg

# --- Validator core ---
def validate(decisions_path: Path,
             hands_path: Path,
             max_rows: Optional[int] = 200,
             sample_seed: int = 42) -> None:
    # 1) load all decision rows (optionally sample)
    rows = list(read_jsonl_gz(decisions_path))
    if max_rows and len(rows) > max_rows:
        random.Random(sample_seed).shuffle(rows)
        rows = rows[:max_rows]

    # 2) build hand_id index
    hand_ids = {r.get("hand_id") for r in rows if r.get("hand_id")}
    hands = index_hands_by_id(hands_path, hand_ids)

    ok = 0
    bad = 0
    by_ctx_ok: Dict[int,int] = {}
    by_ctx_bad: Dict[int,int] = {}

    for r in rows:
        hid = r.get("hand_id")
        if not hid or hid not in hands:
            print(f"⚠️  hand_id {hid} missing from hands file; skip")
            continue
        hand = hands[hid]

        street_id = int(r["street_id"])
        ctx_id    = int(r["ctx_id"])
        act_id    = int(r["act_id"])
        hero_pos  = int(r["hero_pos_id"])

        hero_name = get_actor_name_by_pos_id(hand, hero_pos)
        if not hero_name:
            print(f"\n⚠️  hand_id={hid} could not resolve hero pos_id={hero_pos}")
            continue

        # Summaries to assist your eyeballing
        if street_id == Street.PREFLOP:
            derived_ctx, hero_first_act_id, n_limpers = derive_preflop_ctx(hand, hero_name)
            raises_before_hero = None
            # recompute raises_before_hero quickly
            pre = [a for a in hand["actions"] if int(a["street"]) == Street.PREFLOP]
            hero_idx = next((i for i, a in enumerate(pre) if a.get("actor") == hero_name), None)
            if hero_idx is not None:
                raises_before_hero = sum(1 for a in pre[:hero_idx] if int(a.get("act")) in (Act.RAISE, Act.ALL_IN))
        else:
            dctx, hero_first_act_id = derive_postflop_ctx(hand, street_id, hero_name)
            derived_ctx, n_limpers, raises_before_hero = dctx, None, None

        # Print compact explanation
        print("\n—" * 24)
        print(f"hand_id: {hid}")
        print(f"street : {street_id} ({Street(street_id).name})")
        print(f"hero   : pos_id={hero_pos} name={hero_name}")
        print(
            f"ETL    : ctx_id={ctx_id} ({Ctx(ctx_id).name if ctx_id in Ctx._value2member_map_ else ctx_id}), act_id={act_id} ({Act(act_id).name if act_id in Act._value2member_map_ else act_id})")
        if street_id == Street.PREFLOP:
            print(f"SUM    : limpers_before_raise={n_limpers}  raises_before_hero={raises_before_hero}")
        if derived_ctx is not None:
            print(
                f"HINT   : derived_ctx={derived_ctx} ({Ctx(derived_ctx).name if derived_ctx in Ctx._value2member_map_ else derived_ctx})  hero_first_act={hero_first_act_id} ({Act(hero_first_act_id).name if isinstance(hero_first_act_id, int) and hero_first_act_id in Act._value2member_map_ else hero_first_act_id})")

        # Show raw action timeline for that street
        acts = [a for a in hand["actions"] if int(a["street"]) == street_id]

        def act_name(aid: int) -> str:
            try:
                return Act(aid).name
            except:
                return str(aid)

        for i, a in enumerate(acts):
            who = a.get("actor")
            aid = int(a.get("act"))
            amt = a.get("amount_bb")
            mark = "👉" if who == hero_name else "  "
            print(f"   {mark} [{i:02d}] {who:>12s}  {act_name(aid):>5s}  {amt if amt is not None else ''}")

def _s3_join(prefix: str, key: str) -> str:
    return prefix.rstrip("/") + "/" + key.lstrip("/")

def download_parsed_pair_for_stake(
    stake: int,
    s3_prefix: str,
    local_dir: Path,
    s3: "S3Client",   # your project S3 client
) -> Tuple[Path, Path]:
    """
    Downloads decisions/hands for a stake to local_dir if missing.
    Expected S3 keys (customize if your naming differs):
      {s3_prefix}/nl{stake}/decisions.jsonl.gz
      {s3_prefix}/nl{stake}/hands.jsonl.gz
    Returns local paths (decisions_path, hands_path).
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    dec_local = local_dir / f"decisions_nl{stake}.jsonl.gz"
    hands_local = local_dir / f"hands_nl{stake}.jsonl.gz"

    dec_key = _s3_join(s3_prefix, f"nl{stake}/decisions.jsonl.gz")
    hands_key = _s3_join(s3_prefix, f"nl{stake}/hands.jsonl.gz")

    if not dec_local.exists():
        print(f"⬇️  downloading decisions: s3://{s3.bucket}/{dec_key} → {dec_local}")
        s3.download_file(dec_key, dec_local)
    else:
        print(f"✅ decisions cached: {dec_local}")

    if not hands_local.exists():
        print(f"⬇️  downloading hands: s3://{s3.bucket}/{hands_key} → {hands_local}")
        s3.download_file(hands_key, hands_local)
    else:
        print(f"✅ hands cached: {hands_local}")

    return dec_local, hands_local

# --- filter & pick helpers ---
def filter_decisions_by_ctx(decisions_path: Path, ctx_id: int) -> List[Dict[str, Any]]:
    return [r for r in read_jsonl_gz(decisions_path) if int(r.get("ctx_id", -1)) == int(ctx_id)]

def pick_random_rows(rows: List[Dict[str, Any]], k: int = 1, seed: int = 42) -> List[Dict[str, Any]]:
    if not rows:
        return []
    rng = random.Random(seed)
    if k <= 1:
        return [rng.choice(rows)]
    # sample without replacement if possible
    k = min(k, len(rows))
    return rng.sample(rows, k)

# --- small shim to reuse your printing logic for a single row ---
def _print_validation_one(r: Dict[str, Any], hand: Dict[str, Any]) -> None:
    street_id = int(r["street_id"])
    ctx_id    = int(r["ctx_id"])
    act_id    = int(r["act_id"])
    hero_pos  = int(r["hero_pos_id"])
    hid       = r.get("hand_id")

    hero_name = get_actor_name_by_pos_id(hand, hero_pos)
    print("\n" + "—"*24)
    print(f"hand_id: {hid}")
    print(f"street : {street_id} ({Street(street_id).name})")
    print(f"hero   : pos_id={hero_pos} name={hero_name}")
    print(f"ETL    : ctx_id={ctx_id} ({Ctx(ctx_id).name if ctx_id in Ctx._value2member_map_ else ctx_id}), "
          f"act_id={act_id} ({Act(act_id).name if act_id in Act._value2member_map_ else act_id})")

    if street_id == Street.PREFLOP:
        derived_ctx, hero_first_act_id, n_limpers = derive_preflop_ctx(hand, hero_name)
        pre = [a for a in hand["actions"] if int(a["street"]) == Street.PREFLOP]
        hero_idx = next((i for i, a in enumerate(pre) if a.get("actor") == hero_name), None)
        raises_before_hero = (sum(1 for a in pre[:hero_idx] if int(a.get("act")) in (Act.RAISE, Act.ALL_IN))
                              if hero_idx is not None else None)
        print(f"SUM    : limpers_before_raise={n_limpers}  raises_before_hero={raises_before_hero}")
        if derived_ctx is not None:
            print(f"HINT   : derived_ctx={derived_ctx} ({Ctx(derived_ctx).name if derived_ctx in Ctx._value2member_map_ else derived_ctx})  "
                  f"hero_first_act={hero_first_act_id} ({Act(hero_first_act_id).name if isinstance(hero_first_act_id, int) and hero_first_act_id in Act._value2member_map_ else hero_first_act_id})")
    else:
        dctx, hero_first_act_id = derive_postflop_ctx(hand, street_id, hero_name)
        if dctx is not None:
            print(f"HINT   : derived_ctx={dctx} ({Ctx(dctx).name if dctx in Ctx._value2member_map_ else dctx})  "
                  f"hero_first_act={hero_first_act_id} ({Act(hero_first_act_id).name if isinstance(hero_first_act_id, int) and hero_first_act_id in Act._value2member_map_ else hero_first_act_id})")

    acts = [a for a in hand["actions"] if int(a["street"]) == street_id]
    def act_name(aid: int) -> str:
        try: return Act(aid).name
        except: return str(aid)
    for i, a in enumerate(acts):
        who = a.get("actor"); aid = int(a.get("act")); amt = a.get("amount_bb")
        mark = "👉" if who == hero_name else "  "
        print(f"   {mark} [{i:02d}] {who:>12s}  {act_name(aid):>5s}  {amt if amt is not None else ''}")

# --- revised main() using stake + ctx ---
def main():
    ap = argparse.ArgumentParser(description="Validate PopulationNet contexts against raw hands")
    ap.add_argument("--stake", type=int, default=10, help="e.g., 10 for NL10")
    ap.add_argument("--ctx", type=int, default=None, help="Context ID to inspect (e.g., 1=VS_OPEN, 2=VS_3BET …)")
    ap.add_argument("--k", type=int, default=1, help="Number of random samples to inspect for --ctx")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s3-prefix", type=str, default="parsed", help="S3 prefix under the bucket, e.g. 'parsed'")
    ap.add_argument("--bucket", type=str, default=None, help="Override S3 bucket name if not in S3Client config")
    ap.add_argument("--local-cache", type=Path, default=Path("data/processed"), help="Local cache dir for downloads")
    # optional direct file override (skips S3 download)
    ap.add_argument("--decisions", type=Path, default=None)
    ap.add_argument("--hands", type=Path, default=None)
    args = ap.parse_args()

    # Resolve inputs (either provided files or S3)
    if args.decisions and args.hands:
        decisions_path = args.decisions
        hands_path = args.hands
    else:
        s3 = S3Client(bucket_name=args.bucket) if args.bucket else S3Client()
        local_dir = args.local_cache / f"nl{args.stake}"
        decisions_path, hands_path = download_parsed_pair_for_stake(
            stake=args.stake,
            s3_prefix=args.s3_prefix,
            local_dir=local_dir,
            s3=s3,
        )

    if not decisions_path.exists():
        print(f"decisions not found: {decisions_path}", file=sys.stderr); sys.exit(1)
    if not hands_path.exists():
        print(f"hands not found: {hands_path}", file=sys.stderr); sys.exit(1)
    print(f"validating {args} stake")
    ALL_CTX = [0, 1, 2, 3, 4, 5, 6, 10, 13, 14]

    if args.ctx is not None:
        # 1) filter by context and pick random rows
        all_ctx_rows = filter_decisions_by_ctx(decisions_path, args.ctx)
        picks = pick_random_rows(all_ctx_rows, k=args.k, seed=args.seed)
        print(f"picks: {picks}", file=sys.stderr)
        if not picks:
            print(f"⚠️ No rows found for ctx_id={args.ctx} in {decisions_path}")
            sys.exit(0)
        # 2) index hands for only the picked hand_ids
        needed = {r.get("hand_id") for r in picks if r.get("hand_id")}
        hands_idx = index_hands_by_id(hands_path, needed)
        # 3) run validation flow for the picked rows only
        print(f"ℹ️ Validating {len(picks)} row(s) for ctx_id={args.ctx} …")
        for r in picks:
            hid = r.get("hand_id")
            if not hid or hid not in hands_idx:
                print(f"\n⚠️ hand_id {hid} missing in hands; skip")
                continue
            _print_validation_one(r, hands_idx[hid])
    else:
        # Sweep all contexts: sample k rows per ctx and print
        for ctx in ALL_CTX:
            all_ctx_rows = filter_decisions_by_ctx(decisions_path, ctx)
            picks = pick_random_rows(all_ctx_rows, k=args.k, seed=args.seed)
            ctx_name = Ctx(ctx).name if ctx in Ctx._value2member_map_ else str(ctx)
            if not picks:
                print(f"\n=== CTX {ctx} ({ctx_name}) ===")
                print(f"⚠️ No rows found for ctx_id={ctx} in {decisions_path}")
                continue

            needed = {r.get("hand_id") for r in picks if r.get("hand_id")}
            hands_idx = index_hands_by_id(hands_path, needed)

            print(f"\n=== CTX {ctx} ({ctx_name}) — {len(picks)} sample(s) ===")
            for r in picks:
                hid = r.get("hand_id")
                if not hid or hid not in hands_idx:
                    print(f"\n⚠️ hand_id {hid} missing in hands; skip")
                    continue
                _print_validation_one(r, hands_idx[hid])


if __name__ == "__main__":
    main()