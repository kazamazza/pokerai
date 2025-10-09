# tools/populationnet/preprocess_decisions.py
from pathlib import Path
import json, gzip
import polars as pl

def _read_jsonl(path: Path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def _index_hands_by_id(hands_path: Path) -> dict:
    idx = {}
    for h in _read_jsonl(hands_path):
        idx[str(h["hand_id"])] = h
    return idx

def _derive_action_context(hand, d):
    """Return dict with pot_before_bb, to_call_bb, eff_stack_bb, facing_id for this decision row d."""
    # Inputs available in samples you showed:
    # hand["board"] (list of cards), hand["actions"] (seq), positions map, seats stack_size, etc.
    # d: {"hand_id","street_id","hero_pos_id","villain_pos_id","act_id","amount_bb",...}

    street = int(d["street_id"])
    hero_pos_id = int(d["hero_pos_id"])

    # Walk through hand["actions"] up to (and excluding) hero’s action on this street,
    # accumulating pot and current bet to call.
    pot = 0.0
    street_open = False
    current_bet = 0.0
    committed = {}  # player_id -> committed bb on this street

    # Build seat → stack bb, and a pos_id → player_id map (from hand["position_by_player"])
    pos2pid = {v["id"]: k for k, v in hand.get("position_by_player", {}).items()}
    hero_pid = pos2pid.get(hero_pos_id)

    # Very light, robust pass (handles missing amounts by treating as 0)
    for a in hand.get("actions", []):
        if int(a["street"]) != street:
            # reset street state when we cross into/away from this street
            if int(a["street"]) < street:
                pot += float(a.get("amount_bb") or 0.0)
            continue
        act = int(a["act"])
        amt = float(a.get("amount_bb") or 0.0)
        pid = a.get("actor")

        # If this is hero’s action, STOP before applying it
        if pid == hero_pid:
            break

        # Update pot/current street state
        if act in (1, 2, 5):  # call/raise/all-in → money into pot
            pot += amt
            street_open = True
            if act in (2, 5):   # raise/all-in sets the bar
                current_bet = amt
        # (checks/bets with null amounts just leave pot as-is)

    # to_call for hero is current_bet minus hero’s own committed (approx 0 in this minimal pass)
    to_call_bb = max(current_bet, 0.0)
    facing_id = 1 if street_open and to_call_bb > 0 else 0

    # eff stack at this moment: use seats stacks; if villain_pos_id exists use min(hero, villain)
    seats = hand.get("seats", [])
    pid2stack = {s["player_id"]: float(s["stack_size"]) for s in seats if s.get("status") == "active"}
    vil_pid = pos2pid.get(int(d.get("villain_pos_id", -1)))
    hero_stack = pid2stack.get(hero_pid, 100.0)
    vil_stack  = pid2stack.get(vil_pid, 100.0)
    eff_stack_bb = min(hero_stack, vil_stack)

    return {
        "pot_before_bb": float(max(pot, 0.0)),
        "to_call_bb": float(max(to_call_bb, 0.0)),
        "eff_stack_bb": float(max(eff_stack_bb, 0.0)),
        "facing_id": int(facing_id),
    }

def preprocess(hands_path: Path, decisions_path: Path,
               spr_bins=(0,2,6,100), bet_thresholds=(0.33,0.5,0.66,1.0,1.5,2.5)) -> pl.DataFrame:
    hands = _index_hands_by_id(hands_path)

    # derive context row-by-row
    rows = []
    for d in _read_jsonl(decisions_path):
        h = hands.get(str(d["hand_id"]))
        if not h:
            continue
        ctx = _derive_action_context(h, d)
        rows.append({**d, **ctx})

    df = pl.from_dicts(rows)

    # spr + bins
    df = df.with_columns([
        (pl.col("eff_stack_bb") / pl.col("pot_before_bb").clip(lower_bound=1e-6))
          .clip(lower_bound=0.0, upper_bound=100.0)
          .alias("spr"),
        pl.when(pl.col("facing_id")==1)
          .then((pl.col("to_call_bb") / pl.col("pot_before_bb").clip(lower_bound=1e-6))
                .clip(lower_bound=0.0, upper_bound=5.0))
          .otherwise(pl.lit(-1.0))
          .alias("bet_pct_of_pot"),
    ])

    # bin helpers
    def cut(vals):  # inclusive-right bins
        return pl.when(vals <= spr_bins[1]).then(0) \
                 .when(vals <= spr_bins[2]).then(1) \
                 .otherwise(2)

    # general bucket via thresholds
    def bucket(expr, thr):
        out = pl.lit(-1)
        for i, t in enumerate(thr):
            out = pl.when(expr <= t).then(i).otherwise(out)
        # if > max threshold, put into last+1
        out = pl.when((expr > thr[-1]) & (expr != -1)).then(len(thr)).otherwise(out)
        return out

    df = df.with_columns([
        bucket(pl.col("spr"), spr_bins[1:]).alias("spr_bin"),  # simple: -1 not possible here
        bucket(pl.col("bet_pct_of_pot"), bet_thresholds).alias("bet_size_bucket"),
    ])

    return df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=Path, required=True)
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spr_bins", type=str, default="0,2,6,100")
    ap.add_argument("--bet_thresholds", type=str, default="0.33,0.5,0.66,1.0,1.5,2.5")
    args = ap.parse_args()

    spr_bins = [float(x) for x in args.spr_bins.split(",")]
    bet_thr  = [float(x) for x in args.bet_thresholds.split(",")]

    df = preprocess(args.hands, args.decisions, spr_bins=spr_bins, bet_thresholds=bet_thr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(args.out))
    print(f"✅ wrote enriched decisions → {args.out} rows={df.height:,}")