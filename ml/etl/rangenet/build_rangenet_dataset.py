#!/usr/bin/env python3
import argparse, gzip, json, random, os, math
from pathlib import Path

import pandas as pd

ACTION_KEYS_DEFAULT = [
    "FOLD", "CHECK", "CALL",
    "BET_0.33", "BET_0.5", "BET_0.66", "BET_1.0", "BET_1.5",
    "RAISE_2.0", "RAISE_2.5", "RAISE_3.0", "ALLIN"
]

def to_rank(c):  # "As" -> ("A","s")
    return c[0], c[1]

def canon_card(c):
    # keep as two-char string; model can embed chars
    return c

def canon_hand(hole):
    # hole may be "AsKd" or ["As","Kd"]
    if isinstance(hole, str) and len(hole) == 4:
        return hole[:2] + hole[2:]
    if isinstance(hole, list) and len(hole) == 2:
        return hole[0] + hole[1]
    return str(hole)

def canon_board(board):
    if board in ("", None): return ""
    if isinstance(board, list):
        return "".join(board)
    return board

def action_vector(actions, action_keys):
    vec = [float(actions.get(k, 0.0)) for k in action_keys]
    s = sum(vec)
    if s <= 0:
        # fallback: put all mass on CHECK or CALL if present, else FOLD
        if "CHECK" in action_keys:
            vec[action_keys.index("CHECK")] = 1.0
        elif "CALL" in action_keys:
            vec[action_keys.index("CALL")] = 1.0
        else:
            vec[action_keys.index("FOLD")] = 1.0
        s = 1.0
    return [v / s for v in vec]

def argmax_idx(xs):
    m, i = -1.0, 0
    for j, v in enumerate(xs):
        if v > m: m, i = v, j
    return i

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile",   default="data/ranges/rangenet.v1.jsonl.gz")
    ap.add_argument("--outdir",   default="data/datasets/rangenet.v1")
    ap.add_argument("--actions",  nargs="*", default=["fold","call","raise"])  # only used for argmax sanity
    ap.add_argument("--target",   choices=["dist","argmax"], default="dist")
    ap.add_argument("--seed",     type=int, default=17)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--test_frac",type=float, default=0.05)
    ap.add_argument("--no_parquet", action="store_true", help="also write JSONL if you prefer")
    args = ap.parse_args()

    random.seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Stream & normalize ----------
    rows = []
    opener = gzip.open if args.infile.endswith(".gz") else open
    with opener(args.infile, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rx, ry = rec.get("x", {}), rec.get("y", {})

            # stacks: prefer effective_stack_bb (float), else stack_bb (int)
            eff_stack = rx.get("effective_stack_bb", None)
            stack_bb  = rx.get("stack_bb", None)
            stack     = eff_stack if eff_stack is not None else stack_bb
            if stack is None:
                continue
            stack = float(stack)

            hero_pos = rx.get("hero_pos", None)
            ctx      = rx.get("ctx", None)  # "OPEN" / "VS_OPEN" / "VS_3BET" / ...
            hole = rx.get("hand_bucket") or rx.get("hand_combo")
            if not hole:
                continue
            hole = str(hole)

            action_probs = ry.get("action_probs", None)
            if not action_probs:
                continue

            rows.append({
                # minimal features we’ll need downstream
                "stake_tag":  rx.get("stake_tag", "NLx"),
                "rake_tier":  rx.get("rake_tier"),
                "hero_pos":   hero_pos,
                "btn_pos":    rx.get("btn_pos"),
                "ctx":        ctx,
                "stack_bb":   stack,
                "hand_bucket": hole,
                # labels
                "action_probs": action_probs,
                "exp_raise_bb": ry.get("exp_raise_bb"),
                "ev_bb":        ry.get("ev_bb"),
                # optional opponent-role hints (won’t always exist)
                "opener_pos":       rx.get("opener_pos"),
                "three_bettor_pos": rx.get("three_bettor_pos"),
                "four_bettor_pos":  rx.get("four_bettor_pos"),
            })

    if not rows:
        raise RuntimeError(f"No usable rows in {args.infile}")

    df = pd.DataFrame(rows)

    # sanity: required columns should exist
    required = ["stack_bb", "hero_pos", "ctx", "hand_bucket", "action_probs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in assembled rows: {missing}")

    # drop obviously bad rows
    df = df.dropna(subset=["stack_bb", "hero_pos", "ctx", "hand_bucket"])
    df = df[df["action_probs"].map(lambda v: isinstance(v, (list, tuple)) and len(v) >= 2)]

    # normalize dtypes
    df["stack_bb"] = df["stack_bb"].astype(float)
    df["hero_pos"] = df["hero_pos"].astype(str)
    df["ctx"] = df["ctx"].astype(str)
    df["hand_bucket"] = df["hand_bucket"].astype(str).str.upper()

    # ---------- 2) Basic cleaning ----------
    def _stack_bucket(v):
        try:
            return int(round(float(v)))
        except Exception:
            return 0

    # scenario_key for stratified splitting later
    df["scenario_key"] = df.apply(
        lambda r: f"{_stack_bucket(r['stack_bb'])}|{r['hero_pos']}|{r['ctx']}", axis=1
    )

    # ---------- 2) Basic cleaning ----------
    # (use the *actual* columns we created above)

    # Make the scenario key we will split on: (stack bucketed a bit) + hero_pos + ctx
    # You can make this richer later (include opener/3bettor/fourbettor when present).
    def _stack_bucket(v):
        # bucket stacks so a spot doesn’t split across fine-grained floats (e.g., 17.99 vs 18.0)
        try:
            b = int(round(float(v)))
        except Exception:
            b = 0
        return b

    df["scenario_key"] = df.apply(
        lambda r: f"{_stack_bucket(r['stack_bb'])}|{r['hero_pos']}|{r['ctx']}", axis=1
    )

    # ---------- 3) Make splits by scenario_key ----------
    keys = df["scenario_key"].dropna().unique().tolist()
    random.shuffle(keys)
    n = len(keys)
    n_test = math.floor(n * args.test_frac)
    n_val  = math.floor(n * args.val_frac)
    test_keys = set(keys[:n_test])
    val_keys  = set(keys[n_test:n_test+n_val])
    def split_of(k):
        if k in test_keys: return "test"
        if k in val_keys:  return "val"
        return "train"
    df["split"] = df["scenario_key"].map(split_of)

    # ---------- 4) Optionally convert target to argmax ----------
    if args.target == "argmax":
        # keep also the distribution if you like (comment out if not wanted)
        df["y_class"] = df["action_probs"].apply(lambda v: int(np.argmax(np.asarray(v, dtype=np.float32))))
    # for dist: we keep action_probs as-is

    # ---------- 5) Write shards ----------
    def _write_part(dsub: pd.DataFrame, name: str):
        # Minimal, compact columns
        cols = [
            "stake_tag", "rake_tier", "hero_pos", "btn_pos", "ctx",
            "stack_bb", "hand_bucket", "opener_pos", "three_bettor_pos", "four_bettor_pos",
            "action_probs", "exp_raise_bb", "ev_bb"
        ]
        if args.target == "argmax":
            cols = [c for c in cols if c != "action_probs"] + ["y_class"]

        part = dsub[cols].reset_index(drop=True)

        # Parquet (fast) + optional JSONL
        pq_path = outdir / f"{name}.parquet"
        if not args.no_parquet:
            try:
                part.to_parquet(pq_path, index=False)
            except Exception as e:
                print(f"Parquet write failed ({e}); falling back to JSONL for {name}")

        jl_path = outdir / f"{name}.jsonl.gz"
        with gzip.open(jl_path, "wt", encoding="utf-8") as fout:
            for rec in part.to_dict(orient="records"):
                fout.write(json.dumps(rec, separators=(",", ":")) + "\n")

        return pq_path, jl_path, len(part)

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    t_paths = _write_part(train_df, "train")
    v_paths = _write_part(val_df,   "val")
    s_paths = _write_part(test_df,  "test")

    print("✅ RangeNet dataset built")
    print(f"   train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")
    print(f"   scenario_keys={len(keys):,} | target={args.target}")

if __name__ == "__main__":
    main()