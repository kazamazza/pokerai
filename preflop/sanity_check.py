import os, io, gzip, json, boto3
from collections import Counter

REGION = os.getenv("AWS_REGION", "eu-central-1")
BUCKET = os.getenv("AWS_BUCKET_NAME", "pokeraistore")
s3 = boto3.client("s3", region_name=REGION)

OPEN_OPENER_ACTIONS = {"open"}
VS_OPEN_DEFENDER_ACTIONS = {"call","flat","overcall","cold_call","defend","3bet","4bet","jam"}

def s3_json_gz(key: str) -> dict:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    by = obj["Body"].read()
    with gzip.GzipFile(fileobj=io.BytesIO(by)) as gz:
        return json.loads(gz.read().decode("utf-8"))

def key_open(ip, oop, stack, prof, expl, mw, pop):
    fn = f"{ip}_vs_{oop}_{stack}bb.json.gz"
    return f"preflop/ranges/profile={prof}/exploit={expl}/multiway={mw}/pop={pop}/action=OPEN/{fn}"

def key_vs_open(ip, oop, stack, prof, expl, mw, pop):
    # defender file has roles as given (hero facing open)
    fn = f"{ip}_vs_{oop}_{stack}bb.json.gz"
    return f"preflop/ranges/profile={prof}/exploit={expl}/multiway={mw}/pop={pop}/action=VS_OPEN/{fn}"

def collect_actions(doc: dict, wanted: set[str]) -> list[str]:
    actions = doc.get("actions") or {}
    out = []
    seen = set()
    for k, v in actions.items():
        if k.lower() in wanted:
            for c in v:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
    return out

def summarize(doc: dict, title: str):
    acts = doc.get("actions") or {}
    print(f"\n== {title} ==")
    print("action keys:", list(acts.keys()))
    counts = {k: len(v) for k, v in acts.items()}
    print("counts:", counts)
    # show a couple of examples from each bucket
    for k, v in acts.items():
        print(f"  {k}: {v[:5]}")

def triage_btn_vs_co_100bb():
    ip, oop, stack = "BTN", "CO", 100
    prof, expl, mw, pop = "GTO","GTO","HU","REGULAR"

    k_open    = key_open(ip, oop, stack, prof, expl, mw, pop)
    k_vs_open = key_vs_open(oop, ip, stack, prof, expl, mw, pop)  # defender: CO_vs_BTN

    print("OPEN key    :", k_open)
    print("VS_OPEN key :", k_vs_open)

    try:
        open_doc = s3_json_gz(k_open)
    except Exception as e:
        print("!! failed to load OPEN:", e); return
    try:
        vs_open_doc = s3_json_gz(k_vs_open)
    except Exception as e:
        print("!! failed to load VS_OPEN:", e); return

    summarize(open_doc, "OPEN (BTN_vs_CO)")
    summarize(vs_open_doc, "VS_OPEN (CO_vs_BTN)")

    ip_range  = collect_actions(open_doc, OPEN_OPENER_ACTIONS)
    oop_range = collect_actions(vs_open_doc, VS_OPEN_DEFENDER_ACTIONS)

    print(f"\nIP range size : {len(ip_range)} (from OPEN/open)")
    print(f"OOP range size: {len(oop_range)} (from VS_OPEN/{VS_OPEN_DEFENDER_ACTIONS})")

    same = set(ip_range) == set(oop_range)
    inter = len(set(ip_range) & set(oop_range))
    print(f"equal_sets? {same} ; intersection size = {inter}")

if __name__ == "__main__":
    triage_btn_vs_co_100bb()