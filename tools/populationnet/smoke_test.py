import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from ml.inference.population import PopulationNetInference

ckpt = "checkpoints/popnet/popnet-11-0.9560.ckpt"
pop = PopulationNetInference(ckpt)

row = {"stakes_id": 2, "street_id": 1, "ctx_id": 13, "hero_pos_id": 4, "villain_pos_id": 0}
print(pop.predict_proba(row))
print(pop.predict_class(row))