# tools/rangenet/debug/peek_pairs_in_manifest.py
import json, pandas as pd

df = pd.read_parquet("data/artifacts/monker_manifest.parquet")

def has_btn_open_sb_call(seq_json):
    try: seq = json.loads(seq_json)
    except: return False
    # first non-fold opener must be BTN with canonical RAISE/ALL_IN
    opener_pos = opener_act = None
    for e in seq:
        a = e.get("action")
        if a and a not in ("FOLD",):
            opener_pos, opener_act = e.get("pos"), a
            break
    if opener_pos != "BTN" or opener_act not in ("RAISE","ALL_IN"): return False

    # SB first action must be CALL; and no raise-y before SB acts
    raisey = {"RAISE","ALL_IN","3BET","4BET","5BET","OPEN","LIMP"}
    reraised = False
    for e in seq:
        p, a = e.get("pos"), e.get("action")
        if p == "SB":
            return a == "CALL" and not reraised
        if a in raisey: reraised = True
    return False

# How many rows at all?
print("rows:", len(df))

# Count any BTN_BB SRP open/call rows by stack and hero folder present
sub = df[df["sequence"].apply(has_btn_open_sb_call)]
print("BTN open / SB call rows:", len(sub))
print(sub[["stack_bb","hero_pos","rel_path"]].head(20).to_string(index=False))
print("Stacks covered:", sorted(sub["stack_bb"].dropna().unique().tolist()))