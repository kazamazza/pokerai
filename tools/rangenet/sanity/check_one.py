import json, gzip, io
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.etl.rangenet.postflop.solver_policy_parser import root_node, get_children, actions_and_mix, resolve_child

p = Path("data/debug_samples/3bet_hu.Aggressor_OOP.json.gz")
b = p.read_bytes()
payload = json.loads(gzip.GzipFile(fileobj=io.BytesIO(b)).read().decode("utf-8"))
root = root_node(payload)
acts, mix = actions_and_mix(root)
children = get_children(root)

for a,p in zip(acts, mix):
    if str(a).upper().startswith("BET"):
        bet_node = resolve_child(children, a)
        a2,m2 = actions_and_mix(bet_node)
        print("OOP responses vs root bet:", a2)  # you’ll see CALL/FOLD(/ALL-IN), no “RAISE …”
        break