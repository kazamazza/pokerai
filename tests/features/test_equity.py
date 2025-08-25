# tests/features/test_equity.py

import math
import pytest
from typing import Dict, List

from ml.features.equity.postflop import equity_postflop_vs_range
from ml.features.equity.preflop import equity_preflop_vs_range

TOL = 1e-6

def _sum_to_one(*vals):
    return abs(sum(vals) - 1.0) < 1e-3

def test_preflop_equity_aa_vs_72o():
    vill_range: Dict[str, float] = {"72o": 1.0}
    win, tie, lose = equity_preflop_vs_range(
        hero_code="AA",
        vill_range=vill_range,
        n_samples=2000,
        seed=42,
    )
    assert _sum_to_one(win, tie, lose)
    assert win > 0.75  # AA should crush 72o preflop

def test_preflop_equity_symmetry():
    # AKs vs AQs: P(AKs wins) ≈ P(AQs loses), when we swap hero/villain
    vill1 = {"AQs": 1.0}
    w1, t1, l1 = equity_preflop_vs_range("AKs", vill1, n_samples=3000, seed=1)

    vill2 = {"AKs": 1.0}
    w2, t2, l2 = equity_preflop_vs_range("AQs", vill2, n_samples=3000, seed=1)

    assert _sum_to_one(w1, t1, l1)
    assert _sum_to_one(w2, t2, l2)
    # Symmetry check: win of one ≈ lose of the other (allowing small MC noise)
    assert abs(w1 - l2) < 0.05
    assert abs(w2 - l1) < 0.05

def test_postflop_equity_strong_hand_on_board():
    # Board: As Kd 2c, Hero: AhAd (top set)
    board = ["As", "Kd", "2c"]
    vill = {"72o": 1.0}
    win, tie, lose = equity_postflop_vs_range(
        board_cards=board,
        hero_code="AhAd",
        vill_range=vill,
        n_samples=1000,
        seed=42,
    )
    assert _sum_to_one(win, tie, lose)
    assert win > 0.90

def test_postflop_equity_draw_vs_made_hand():
    # Board: 9h 6h 2c; Hero: 7h8h (OESD + backdoor hearts), Villain: AA (overpair)
    board = ["9h", "6h", "2c"]
    vill = {"AsAc": 1.0}
    win, tie, lose = equity_postflop_vs_range(
        board_cards=board,
        hero_code="7h8h",
        vill_range=vill,
        n_samples=3000,
        seed=42,
    )
    assert _sum_to_one(win, tie, lose)
    # Some equitynet but < 50%
    assert 0.50 < win < 0.62