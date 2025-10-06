#!/usr/bin/env python3
import re, argparse
from pathlib import Path
import pandas as pd

POS_RE = re.compile(r"pos=([A-Z]+)v([A-Z]+)")

# Flop IP precedence (higher index = acts later = IP)
# BB acts first vs everyone; BTN last; SB is IP vs BB.
FLOP_IP_ORDER = {"BB":0, "SB":1, "UTG":2, "HJ":3, "CO":4, "BTN":5}

def parse_pos_from_key(s: str):
    """Return (a, b) from '.../pos=AvB/...'; else (None,None)."""
    if not isinstance(s, str):
        return (None, None)
    m = POS_RE.search(s)
    if not m:
        return (None, None)
    return (m.group(1), m.group(2))

def ip_oop_from_pair(a: str, b: str):
    """Decide IP/OOP on flop using precedence."""
    ra = FLOP_IP_ORDER.get(a, -1)
    rb = FLOP_IP_ORDER.get(b, -1)
    if ra == -1 or rb == -1:
        # fallback: assume first is IP (keeps dataset usable)
        return a, b
    return (a, b) if ra > rb else (b, a)

def main():
    ap = argparse.ArgumentParser("Augment postflop parquet with real seats derived from s3_key")
    ap.add_argument("--in", dest="inp", required=True, help="Input parquet (merged parts)")
    ap.add_argument("--out", dest="out", required=True, help="Output parquet with ip_pos/oop_pos/hero_pos")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    df = pd.read_parquet(inp)
    if "s3_key" not in df.columns:
        raise ValueError("Input parquet missing 's3_key' column")

    # Extract seats
    a_b = df["s3_key"].map(parse_pos_from_key)
    df["seat_a"] = a_b.map(lambda t: t[0])
    df["seat_b"] = a_b.map(lambda t: t[1])

    # Compute IP/OOP (flop)
    io = a_b.map(lambda t: ip_oop_from_pair(t[0], t[1]) if t[0] and t[1] else (None, None))
    df["ip_pos_real"]  = io.map(lambda t: t[0])
    df["oop_pos_real"] = io.map(lambda t: t[1])

    # Fill training columns: prefer newly derived seats, fallback to old tokens
    df["ip_pos"]  = df["ip_pos_real"].fillna(df.get("ip_pos", "IP"))
    df["oop_pos"] = df["oop_pos_real"].fillna(df.get("oop_pos", "OOP"))

    # Hero seat from actor (if present), else leave None
    actor_col = "actor" if "actor" in df.columns else None
    if actor_col:
        df["hero_pos"] = df.apply(
            lambda r: r["ip_pos"] if str(r["actor"]).lower() == "ip"
            else (r["oop_pos"] if str(r["actor"]).lower() == "oop" else None),
            axis=1
        )

    # Cleanup helper columns
    df = df.drop(columns=[c for c in ["seat_a","seat_b","ip_pos_real","oop_pos_real"] if c in df.columns])

    # Basic sanity: how many rows got seats?
    got = df["ip_pos"].notna().sum()
    print(f"✅ rows with derived ip_pos/oop_pos: {got}/{len(df)}")

    # Write
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"💾 wrote {out}")

if __name__ == "__main__":
    main()