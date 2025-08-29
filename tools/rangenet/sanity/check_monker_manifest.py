import pandas as pd, json
df = pd.read_parquet("data/artifacts/monker_manifest.parquet")

print("rows:", len(df))
print("stacks:", sorted(df["stack_bb"].dropna().unique().tolist()))
print("hero_pos:", sorted(df["hero_pos"].dropna().unique().tolist()))

# opener raw tokens – should include things like '60%', 'Min', 'AI'
print("top opener_action_raw:", df["opener_action_raw"].value_counts().head(10).to_string())

# sanity: try to find at least one BTNvSB SRP open/call at any stack:
def raw(seq_json):
    try:
        return json.loads(seq_json)
    except Exception:
        return []
def is_call(a): return a in ("Call","CALL")

hits = 0
for _, r in df.iterrows():
    seq = raw(r.get("sequence_raw") or r.get("sequence"))
    # opener is first non-fold raise-ish:
    # (reuse the same first_non_fold_opener from helpers if you can import it here)
    # naive inline check:
    found_open = None
    for e in seq:
        a = e.get("action") or ""
        if a not in ("Fold","FOLD") and (a == "Min" or a == "AI" or a == "3sb" or a.endswith("%")):
            found_open = e.get("pos")
            break
    if found_open == "BTN":
        # look for SB's first action == Call before any re-raise
        reraised = False
        first_sb = None
        for e2 in seq:
            p2 = e2.get("pos")
            a2 = e2.get("action") or ""
            if p2 == "SB":
                first_sb = a2
                break
            if a2 in ("Min","AI","3sb") or a2.endswith("%"):
                reraised = True; break
        if first_sb and is_call(first_sb) and not reraised:
            hits += 1
            break

print("quick BTNvSB SRP open/call hits:", hits)