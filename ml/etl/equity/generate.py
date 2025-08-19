from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.utils.config import load_section, load_cfg
from ml.utils.constants import RANKS
import json, gzip, time, random
from pathlib import Path
from typing import Dict, List
import yaml
import eval7

from ml.utils.boards import load_cluster_map, flop_cluster_id_safe, canon_card_key
from ml.utils.ranges import opp_range_vec169  # your existing canonical 169-vector helper

# 169 hands (AA..22, suited above diag, offsuit below)
ID2HAND = (
    [r+r for r in RANKS] +
    [RANKS[i]+RANKS[j]+'s' for i in range(len(RANKS)) for j in range(i+1, len(RANKS))] +
    [RANKS[i]+RANKS[j]+'o' for i in range(len(RANKS)) for j in range(i+1, len(RANKS))]
)
assert len(ID2HAND) == 169

# ---------------- helpers ----------------

def hand_string_to_cards(hand: str) -> List[eval7.Card]:
    """'AKs','AQo','TT' -> two concrete cards consistent with suitedness."""
    if hand.endswith("s"):
        r1, r2 = hand[0], hand[1]
        suit = random.choice("shdc")
        return [eval7.Card(r1 + suit), eval7.Card(r2 + suit)]
    elif hand.endswith("o"):
        r1, r2 = hand[0], hand[1]
        s1, s2 = random.sample("shdc", 2)
        return [eval7.Card(r1 + s1), eval7.Card(r2 + s2)]
    else:
        r = hand[0]
        s1, s2 = random.sample("shdc", 2)
        return [eval7.Card(r + s1), eval7.Card(r + s2)]

def sample_villain_cards(opp_map: Dict[str,float]) -> List[eval7.Card]:
    """Weighted shorthand sampling; light, fast approximation."""
    bag: List[str] = []
    for h, w in opp_map.items():
        if w <= 0: continue
        bag.extend([h] * max(1, int(w * 50)))
    return hand_string_to_cards(random.choice(bag) if bag else random.choice(ID2HAND))

def deal_flop(deck: eval7.Deck) -> List[eval7.Card]:
    return deck.sample(3)

def deal_turn_river(deck: eval7.Deck) -> List[eval7.Card]:
    return deck.sample(2)

# -------------- config --------------

CFG = yaml.safe_load(Path("ml/config/settings.yaml").read_text())
paths = CFG.get("paths", {})
PRELOP_FILE = Path(paths.get("preflop_jsonl", "data/preflop/preflop.hu.v1.jsonl.gz"))
OUT_FILE    = Path(paths.get("equity_jsonl",   "data/equity/equity_dataset.v1.jsonl.gz"))

# Load flop clusters
flop_cfg = CFG.get("board_clustering", {}).get("flop", {})
FLOP_PATH = flop_cfg.get("out_path", "data/boards/flop_clusters.k256.lite_v1.json")
FLOP_CLUSTERS, FLOP_META = load_cluster_map(FLOP_PATH)

# ------------------------------
# Profile-aware equity_generation
# ------------------------------
eg_root = CFG.get("equity_generation", {}) or {}
# Read profile from env (ML_PROFILE) or fall back to default_profile in config
eg_profile = os.getenv("ML_PROFILE", eg_root.get("default_profile", "dev"))

profiles = eg_root.get("profiles", {}) or {}
EG = profiles.get(eg_profile)
if not EG:
    raise ValueError(
        f"Equity generation profile '{eg_profile}' not found. "
        f"Available: {list(profiles.keys())}"
    )

# Assign params
SEED        = int(EG.get("seed", 42))
N_SAMPLES   = int(EG.get("samples_per_hand", 64))
KEEP_FRAC   = float(EG.get("keep_fraction", 1.0))
STACKS_WH   = set(int(x) for x in (EG.get("stacks_whitelist") or []))
HANDS_SUB   = [
    h.strip().upper().replace("S", "s").replace("O", "o")
    for h in (EG.get("hands_subset") or [])
]
LOG_EVERY   = int(EG.get("log_every_rows", 5000))
random.seed(SEED)

print(f"[CFG] Using equity_generation profile='{eg_profile}' | samples={N_SAMPLES}, keep_frac={KEEP_FRAC}")

# -------------- core MC (aggregated) --------------

def simulate_hand_agg(hid: int,
                      opp_map: Dict[str,float],
                      opp_vec169: List[float],
                      n_trials: int) -> Dict:
    """
    Aggregate n_trials of hero hand vs opp_map → mean equity & variance.
    Also assigns a reliable flop cluster id (never -1) via safe lookup.
    """
    hero_hand = ID2HAND[hid]

    # one representative flop for metadata (equity itself aggregates over many different flops)
    # this representative id is used only to populate board_cluster_id field so schema passes
    rep_deck = eval7.Deck()
    rep_flop = rep_deck.sample(3)
    rep_key  = [canon_card_key(str(c)) for c in rep_flop]
    rep_cid  = flop_cluster_id_safe(rep_key, FLOP_CLUSTERS, FLOP_META)  # ✅ guaranteed >= 0

    s = 0.0
    s2 = 0.0
    for _ in range(n_trials):
        h_cards = hand_string_to_cards(hero_hand)
        v_cards = sample_villain_cards(opp_map)
        while set(v_cards) & set(h_cards):
            v_cards = sample_villain_cards(opp_map)

        deck = eval7.Deck()
        for c in h_cards + v_cards: deck.cards.remove(c)
        flop = deal_flop(deck)
        turn_river = deal_turn_river(deck)
        board5 = flop + turn_river

        hs = eval7.evaluate(h_cards + board5)
        vs = eval7.evaluate(v_cards + board5)
        eq = 1.0 if hs > vs else (0.5 if hs == vs else 0.0)
        s  += eq
        s2 += eq * eq

    mean = s / n_trials
    var  = max(0.0, (s2 / n_trials) - (mean * mean))

    return {
        "x": {
            "hand_id":          int(hid),
            "board_cluster_id": int(rep_cid),    # ✅ always >= 0
            "bucket_id":        -1,              # using opp_range_emb path
            "opp_range_emb":    opp_vec169,      # 169-d; your Dataset will compress if desired
            "board_feats":      []               # optional future
        },
        "y": {"equity": float(mean), "n": int(n_trials), "var": float(var)}
    }

# -------------- driver --------------

def generate_dataset():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # prescan to estimate target (after whitelist/thinning)
    with gzip.open(PRELOP_FILE, "rt", encoding="utf-8") as f_scan:
        raw_lines = [ln for ln in f_scan if ln.strip()]

    def keep(meta: Dict) -> bool:
        sb = int(meta.get("stack_bb", 0) or 0)
        if STACKS_WH and sb not in STACKS_WH: return False
        if KEEP_FRAC < 1.0 and random.random() > KEEP_FRAC: return False
        return True

    est_spots = 0
    for ln in raw_lines:
        try:
            if keep(json.loads(ln)["meta"]): est_spots += 1
        except Exception:
            pass

    if HANDS_SUB:
        # allow subset for smoke tests
        idx = {h:i for i,h in enumerate(ID2HAND)}
        hero_ids = [idx[h] for h in HANDS_SUB if h in idx]
        if not hero_ids:
            hero_ids = list(range(169))
    else:
        hero_ids = list(range(169))

    rows_per_spot = len(hero_ids)
    target = est_spots * rows_per_spot

    print("➡️  Generating EquityNet samples (aggregated)")
    print(f"    preflop spots (raw): {len(raw_lines):,}")
    print(f"    est kept spots:      {est_spots:,}  (keep_fraction={KEEP_FRAC}, stacks={sorted(STACKS_WH) or 'ALL'})")
    print(f"    rows/spot:           {rows_per_spot:,}  (aggregated over {N_SAMPLES} trials)")
    print(f"    total target (est):  {target:,}")
    print(f"    flop clusters:       K={FLOP_META.get('k')} ({FLOP_PATH})")

    t0 = time.time()
    last_log = t0
    total = 0

    with gzip.open(OUT_FILE, "wt", encoding="utf-8") as out, \
         gzip.open(PRELOP_FILE, "rt", encoding="utf-8") as f:

        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            meta = rec.get("meta", {})
            if not keep(meta): continue

            opp_map = rec["range_map"]
            opp_vec = opp_range_vec169(opp_map)  # compute once per spot

            for hid in hero_ids:
                row = simulate_hand_agg(hid, opp_map, opp_vec, N_SAMPLES)
                out.write(json.dumps(row) + "\n")
                total += 1

                now = time.time()
                if (total % max(1000, LOG_EVERY) == 0) or (now - last_log >= 5.0):
                    rate = total / max(1.0, now - t0)
                    pct  = (100.0 * total / max(1, target)) if target else 0.0
                    eta  = ((target - total) / rate / 60.0) if target and rate > 0 else 0.0
                    print(f"  • rows={total:,}/{target:,} ({pct:4.1f}%) | rate={rate:,.0f}/s | ETA ~{eta:,.1f} min", flush=True)
                    last_log = now

    dt = time.time() - t0
    print(f"✅ Equity dataset written: {OUT_FILE} | rows={total:,} | time={dt/60:.1f} min | avg_rate={total/max(1,dt):,.0f}/s", flush=True)

if __name__ == "__main__":
    cfg_all, tcfg, seed, profile = load_cfg()
    print(
        f"[training.equitynet] profile={profile} | batch_size={tcfg['batch_size']} | epochs={tcfg['epochs']} | lr={tcfg['lr']}")
    random.seed(SEED)
    generate_dataset()