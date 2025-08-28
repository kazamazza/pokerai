# tools/rangenet/check_manifest_actions.py
import json, sys, pandas as pd
df = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "data/artifacts/monker_manifest.parquet")
bad = 0
for s in df["sequence"]:
    for e in json.loads(s):
        a = e.get("action")
        if a and a not in {"ALL_IN","RAISE","OPEN","LIMP","CALL","BET","CHECK","CBET","DONK","3BET","4BET","5BET","FOLD"}:
            bad += 1; break
print("✅ actions normalized" if bad == 0 else f"❌ found {bad} rows with non-canonical actions")