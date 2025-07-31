import os
import json
import sys
from pathlib import Path

import eval7
from itertools import combinations
from collections import Counter
from typing import List
from sklearn.cluster import KMeans

sys.path.append(str(Path(__file__).resolve().parent.parent))

from flop.clustering.cluster_config import FlopClusterGranularity

# === Step 1: Generate all 3-card flops ===
deck = [str(card) for card in eval7.Deck()]
raw_flops = list(combinations(deck, 3))  # 22,100 unique 3-card combinations

# === Step 2: Normalize to canonical string ===
def normalize_flop(flop: List[str]) -> str:
    return ''.join(sorted(flop))

# === Step 3: Feature extractor ===
def extract_flop_features(flop: List[str]) -> List[float]:
    ranks = [card[0] for card in flop]
    suits = [card[1] for card in flop]

    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    is_paired = 1 if 2 in rank_counts.values() else 0
    is_triplet = 1 if 3 in rank_counts.values() else 0
    num_suits = len(suit_counts)
    is_monotone = 1 if num_suits == 1 else 0
    is_two_tone = 1 if num_suits == 2 else 0

    rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
                '7': 7, '8': 8, '9': 9, 'T': 10,
                'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    rank_nums = sorted(rank_map[r] for r in ranks)
    high = max(rank_nums)
    low = min(rank_nums)
    gap1 = rank_nums[1] - rank_nums[0]
    gap2 = rank_nums[2] - rank_nums[1]

    return [
        is_paired,
        is_triplet,
        is_monotone,
        is_two_tone,
        high / 14,
        low / 14,
        gap1 / 12,
        gap2 / 12,
    ]

# === Step 4: Build feature matrix ===
flop_keys = []
flop_features = []

for flop in raw_flops:
    try:
        flop_list = list(flop)  # Convert tuple to list
        norm = normalize_flop(flop_list)
        features = extract_flop_features(flop_list)
        flop_keys.append(norm)
        flop_features.append(features)
    except Exception as e:
        print(f"[SKIP] Invalid flop: {flop} → {e}")
        continue

# === Step 5: Cluster using KMeans ===
n_clusters = FlopClusterGranularity.DEFAULT.value
model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = model.fit_predict(flop_features)

# === Step 6: Save JSON mapping ===
cluster_map = {key: int(label) for key, label in zip(flop_keys, labels)}

out_path = "data/flop/flop_cluster_map.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w") as f:
    json.dump(cluster_map, f, indent=2)

print(f"✅ Flop cluster map saved: {out_path}")