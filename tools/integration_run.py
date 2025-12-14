#!/usr/bin/env python3
"""
Dynamic policy integration sweep for Poker Neural Network.

What this does:
- Generates exactly N requests across preflop/postflop, root/facing, and contexts.
- Sends them concurrently to the /policy API.
- Applies *soft* assertions (never crash your run) and counts hard failures (HTTP/shapes/non-finite EVs).
- Summarizes results + promotion usage and shows a few sample problem cases.

Usage:
  python tools/integration_sweep.py --api http://localhost:8000/policy --n 2000 --threads 16 --timeout 8
"""

import argparse, json, math, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ----------------------------- Config knobs -----------------------------------

RAND_HANDS = [
    "AhKh","AdKd","AsKs","QhQs","JhTh","9c9d","7s7h","AcJc",
    "KQo","A5s","T9s","98s","76s","KJs","AQo","TT","22"
]
POSTFLOP_BET_SIZES = [0.33, 0.66]      # expected by your router
PREFLOP_RAISE_BUCKETS = [1.5, 2.0, 3.0]
CONTEXTS = ["VS_OPEN", "VS_3BET", "VS_4BET", "LIMPED_SINGLE"]

# ----------------------------- Small helpers ----------------------------------

def _sum_ok(xs, eps=1e-5):
    try:
        s = float(sum(xs))
        return math.isfinite(s) and abs(s - 1.0) < 0.05
    except Exception:
        return False

def _find(tokens, name):
    return tokens.index(name) if name in tokens else -1

def _get(tokens, vals, name):
    i = _find(tokens, name)
    return (vals[i] if i >= 0 else None)

def _prefix_in(tokens, pfx):
    return any(t.startswith(pfx) for t in tokens)

# ----------------------------- Generators -------------------------------------

def preflop_unopened_bb():
    """BB unopened → free check present; OPEN_* allowed; no CALL/RAISE_*"""
    return {
        "stakes": "NL10",
        "street": 0,
        "ctx": "VS_OPEN",
        "hero_pos": "BB",
        "villain_pos": "BTN",
        "hero_hand": random.choice(RAND_HANDS),

        "pot_bb": 1.5,
        "eff_stack_bb": 100,

        "facing_bet": False,
        "bet_sizes": POSTFLOP_BET_SIZES,
        "raise_buckets": PREFLOP_RAISE_BUCKETS,
        "allow_allin": False,
        "debug": True,

        "actions_hist": []
    }

def preflop_facing(opener="BTN", hero="BB", open_bb=2.5):
    """Hero facing an open → expect CALL + RAISE_*, forbid OPEN_ and CHECK."""
    posted = 1.0 if hero == "BB" else (0.5 if hero == "SB" else 0.0)
    faced_frac = max(open_bb - posted, 0.0) / 100.0  # cost-to-call / stack (100bb)
    return {
        "stakes": "NL10",
        "street": 0,
        "ctx": "VS_OPEN",
        "hero_pos": hero,
        "villain_pos": opener,
        "hero_hand": random.choice(RAND_HANDS),

        "pot_bb": 1.5,
        "eff_stack_bb": 100,

        "facing_bet": True,
        "faced_size_frac": round(float(faced_frac), 6),

        "raise_buckets": PREFLOP_RAISE_BUCKETS,
        "allow_allin": False,
        "debug": True,

        # minimal structured preflop hint for context infer (optional)
        "actions_hist": [{"player_id": opener, "action": "RAISE", "street": 0}]
    }

def postflop_root(ctx="VS_OPEN", hero="SB", vill="BTN", pot=5.5):
    """OOP acts first → root branch: expect CHECK + BET_*, forbid CALL/FOLD/DONK_."""
    return {
        "stakes": "NL10",
        "street": 1,
        "ctx": ctx,
        "hero_pos": hero,
        "villain_pos": vill,
        "hero_hand": random.choice(RAND_HANDS),

        "board": "Ts5cKd",
        "pot_bb": pot,
        "eff_stack_bb": 100,

        "facing_bet": False,
        "bet_sizes": POSTFLOP_BET_SIZES,
        "allow_allin": False,
        "debug": True,

        "actions_hist": []
    }

def postflop_facing(ctx="VS_OPEN", hero="BTN", vill="BB", pot=5.5, faced=0.33):
    """IP facing bet → facing branch: expect FOLD/CALL/RAISE_*, forbid BET_/DONK_/CHECK."""
    return {
        "stakes": "NL10",
        "street": 1,
        "ctx": ctx,
        "hero_pos": hero,
        "villain_pos": vill,
        "hero_hand": random.choice(RAND_HANDS),

        "board": "Ts5cKd",
        "pot_bb": pot,
        "eff_stack_bb": 100,

        "facing_bet": True,
        "faced_size_frac": float(faced),

        "bet_sizes": POSTFLOP_BET_SIZES,
        "allow_allin": False,
        "debug": True,

        "actions_hist": [{"player_id": vill, "action": "BET", "street": 1}]
    }

def make_batch(seed=42, n=200):
    """
    Build exactly n jobs, balanced across 6 families:
      - preflop unopened BB
      - preflop facing (BTN/CO/SB mix)
      - postflop root (random ctx)
      - postflop facing (random ctx, faced in {0.33, 0.66})
    """
    random.seed(seed)
    fs = [0.33, 0.66]

    fams = [
        ("preflop_unopened_bb", lambda: preflop_unopened_bb()),
        ("preflop_facing_btn",  lambda: preflop_facing("BTN", "BB", 2.5)),
        ("preflop_facing_co",   lambda: preflop_facing("CO",  "BB", 2.2)),
        ("preflop_facing_sb",   lambda: preflop_facing("SB",  "BB", 3.0)),
        ("postflop_root",       lambda: postflop_root(ctx=random.choice(CONTEXTS), hero="SB",  vill="BTN")),
        ("postflop_facing",     lambda: postflop_facing(ctx=random.choice(CONTEXTS), hero="BTN", vill="BB", faced=random.choice(fs))),
    ]

    k = n // len(fams)
    r = n - k * len(fams)

    jobs = []
    for i, (name, fn) in enumerate(fams):
        for j in range(k):
            jobs.append((f"{name}_{j}", fn()))
        if i < r:
            jobs.append((f"{name}_extra", fn()))

    random.shuffle(jobs)
    assert len(jobs) == n
    return jobs

# ----------------------------- Soft assertions --------------------------------

def _assert_soft(req, resp):
    """
    Return (passed, notes). Only shape/non-finite EVs mark hard failure.
    Everything else is a soft note to guide tuning.
    """
    notes = []
    ok = True

    actions = resp.get("actions", [])
    probs   = resp.get("probs", [])
    evs     = resp.get("evs", [])
    debug   = resp.get("debug", {}) or {}

    # shape checks
    if not actions or len(actions) != len(probs) or len(actions) != len(evs):
        return False, ["shape_mismatch(actions/probs/evs)"]

    # numeric checks
    if not _sum_ok(probs):
        notes.append("probs_sum≈1 check loose-fail")
    if any((not math.isfinite(float(x))) for x in evs):
        return False, ["nonfinite_evs"]

    # context echo (soft)
    ctx_req = (req.get("ctx") or "VS_OPEN").upper()
    if "ctx" in debug and isinstance(debug["ctx"], str):
        if debug["ctx"].upper() != ctx_req:
            notes.append(f"ctx_echo_mismatch resp={debug['ctx']} req={ctx_req}")

    # anchors
    street  = int(req.get("street", 0))
    facing  = bool(req.get("facing_bet", False))
    hero    = (req.get("hero_pos") or "").upper()

    ev_fold  = _get(actions, evs, "FOLD")
    ev_check = _get(actions, evs, "CHECK")
    free_check = (street == 0 and hero == "BB" and not facing)

    if ev_fold is not None and abs(ev_fold) > 1e-3:
        notes.append(f"ev_fold_anchor_not_zero={ev_fold:.3f}")
    if free_check and ev_check is not None and abs(ev_check) > 1e-3:
        notes.append(f"ev_check_anchor_not_zero={ev_check:.3f}")

    # legality/menu sanity (soft)
    toks = set(actions)
    if street == 0:
        if facing:
            if "CALL" not in toks: notes.append("preflop_facing_missing_CALL")
            if any(t.startswith("OPEN_") for t in toks): notes.append("preflop_facing_has_OPEN")
            if "CHECK" in toks: notes.append("preflop_facing_has_CHECK")
        else:
            if hero == "BB" and "CHECK" not in toks: notes.append("preflop_unopened_bb_missing_CHECK")
            if not any(t.startswith("OPEN_") for t in toks): notes.append("preflop_unopened_missing_OPEN")
            if "CALL" in toks or any(t.startswith("RAISE_") for t in toks):
                notes.append("preflop_unopened_has_CALL_or_RAISE")
    else:
        if facing:
            if "CALL" not in toks: notes.append("postflop_facing_missing_CALL")
            if any(t.startswith("BET_") for t in toks): notes.append("postflop_facing_has_BET")
            if any(t.startswith("DONK_") for t in toks): notes.append("postflop_facing_has_DONK")
            if "CHECK" in toks: notes.append("postflop_facing_has_CHECK")
        else:
            if "CHECK" not in toks: notes.append("postflop_root_missing_CHECK")
            if not any(t.startswith("BET_") for t in toks): notes.append("postflop_root_missing_BET")
            if "CALL" in toks: notes.append("postflop_root_has_CALL")
            if "FOLD" in toks: notes.append("postflop_root_has_FOLD")
            if any(t.startswith("DONK_") for t in toks): notes.append("postflop_root_has_DONK")

    # EV–prob alignment (soft)
    try:
        i_evbest   = int(max(range(len(evs)),   key=lambda i: float(evs[i])))
        i_probbest = int(max(range(len(probs)), key=lambda i: float(probs[i])))
        if i_evbest != i_probbest:
            notes.append(f"ev_prob_mismatch ev_best={actions[i_evbest]} prob_best={actions[i_probbest]}")
    except Exception:
        pass

    # Optional: monotonic EV hint (soft)
    def monotone_subset(names):
        idx = [actions.index(t) for t in names if t in actions]
        vals = [evs[i] for i in idx]
        if len(vals) >= 3:
            nondec = all(float(vals[i]) <= float(vals[i+1]) + 1e-6 for i in range(len(vals)-1))
            if not nondec:
                notes.append(f"ev_not_monotone_{names}")

    if street == 0 and facing:
        monotone_subset(["CALL","RAISE_600","RAISE_750","RAISE_900","RAISE_1200"])
    if street > 0 and facing:
        names = [t for t in actions if t.startswith("RAISE_")]
        def _k(x):
            try:   return float(x.split("_")[1])
            except: return 0.0
        names = ["CALL"] + sorted(names, key=_k)
        monotone_subset(names)

    return ok, notes

# ----------------------------- Runner -----------------------------------------

def call_api(url, req, timeout):
    r = requests.post(url, json=req, timeout=timeout)
    r.raise_for_status()
    return r.json()

def run_suite(api, total=200, threads=8, seed=42, timeout=8.0):
    cases = make_batch(seed=seed, n=total)

    metrics = {
        "total": len(cases),
        "hard_fail": 0,
        "soft_notes": 0,
        "notes_top": {},
        "promo_seen": 0,
        "promo_applied": 0,
        "family_pass": {},       # name prefix -> pass count
        "family_total": {},      # name prefix -> total
        "examples": []
    }

    def fam_of(name):
        return name.split("_")[0] if "_" in name else name

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = {ex.submit(call_api, api, req, timeout): (name, req) for name, req in cases}
        for fut in as_completed(futs):
            name, req = futs[fut]
            fam = fam_of(name)
            metrics["family_total"][fam] = metrics["family_total"].get(fam, 0) + 1
            try:
                resp = fut.result()
            except Exception as e:
                metrics["hard_fail"] += 1
                if len(metrics["examples"]) < 5:
                    metrics["examples"].append({"name": name, "error": str(e), "req": req})
                continue

            ok, notes = _assert_soft(req, resp)
            if ok:
                metrics["family_pass"][fam] = metrics["family_pass"].get(fam, 0) + 1
            else:
                metrics["hard_fail"] += 1
                if len(metrics["examples"]) < 5:
                    metrics["examples"].append({"name": name, "notes": notes, "req": req, "resp": resp})

            metrics["soft_notes"] += len(notes)
            for n in notes:
                metrics["notes_top"][n] = metrics["notes_top"].get(n, 0) + 1

            # promotion stats
            dbg = resp.get("debug") or {}
            promo = (dbg.get("promotion") or {})
            if isinstance(promo, dict) and ("applied" in promo):
                metrics["promo_seen"] += 1
                if promo.get("applied"):
                    metrics["promo_applied"] += 1

    dt = time.time() - t0
    passed = metrics["total"] - metrics["hard_fail"]

    # ---------------------------- Summary -------------------------------------
    print("\n=== INTEGRATION SUMMARY ===")
    print(f"Requests: {metrics['total']}  |  Hard-fail: {metrics['hard_fail']}  |  Pass: {passed}  |  Time: {dt:.2f}s")

    # Per-family pass rates
    fams = sorted(metrics["family_total"].keys())
    if fams:
        print("\nPer-family pass rates:")
        for f in fams:
            tot = metrics["family_total"].get(f, 0)
            okc = metrics["family_pass"].get(f, 0)
            rate = (100.0 * okc / max(1, tot))
            print(f"  - {f:18s}: {okc}/{tot} ({rate:.1f}%)")

    # Promotion stats
    if metrics["promo_seen"]:
        rate = 100.0 * metrics["promo_applied"] / max(1, metrics["promo_seen"])
        print(f"\nPromotion: seen={metrics['promo_seen']}  applied={metrics['promo_applied']} ({rate:.1f}%)")

    # Top soft notes
    if metrics["notes_top"]:
        print("\nTop notes (soft warnings):")
        for k, v in sorted(metrics["notes_top"].items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"  - {k}: {v}")

    # Examples
    if metrics["examples"]:
        print("\nSample failures/warnings:")
        for exm in metrics["examples"]:
            print(f"--- {exm.get('name')} ---")
            print(json.dumps(exm, ensure_ascii=False, indent=2)[:1400])

    print("\nDone.")

# ----------------------------- Main -------------------------------------------

def main():
    ap = argparse.ArgumentParser("Dynamic policy integration sweep")
    ap.add_argument("--api", default="http://localhost:8000/policy", help="Policy endpoint")
    ap.add_argument("--n", type=int, default=200, help="Total requests to send")
    ap.add_argument("--threads", type=int, default=8, help="Concurrent workers")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--timeout", type=float, default=8.0, help="Per-request timeout (sec)")
    args = ap.parse_args()

    run_suite(args.api, total=args.n, threads=args.threads, seed=args.seed, timeout=args.timeout)

if __name__ == "__main__":
    main()