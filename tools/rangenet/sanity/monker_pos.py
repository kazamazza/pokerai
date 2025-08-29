import pandas as pd, json, re
df = pd.read_parquet("data/artifacts/monker_manifest.parquet")

def seq(j):
    try: return json.loads(j)
    except: return []

def is_open(tok):  # treat % as raise too
    return tok in ("Min","AI","3sb") or bool(re.match(r"^\d+%$", tok or ""))

def first_open_pos(s):
    for e in s:
        a = (e.get("action") or "")
        if a not in ("Fold","FOLD") and is_open(a):
            return e.get("pos")
    return None

def first_action_of(s, pos):
    for e in s:
        if e.get("pos") == pos and "action" in e:
            return e["action"]
    return None

hits = 0
for _, r in df.iterrows():
    s = seq(r["sequence_raw"] if "sequence_raw" in r else r["sequence"])
    if first_open_pos(s) == "BTN":
        # ensure no re-raise before SB acts
        reraised = False
        for e in s:
            if e.get("pos") == "SB": break
            if is_open(e.get("action") or ""): reraised = True; break
        if not reraised and first_action_of(s,"SB") in ("Call","CALL"):
            hits += 1
            break
print("BTN vs SB SRP open/call hits:", hits)