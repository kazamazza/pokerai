import math
from ml.policy.solver_action_mapping import (
    map_root_mix_to_root_vocab,
    map_facing_mix_to_facing_vocab,
    oop_root_kind_for_bet_sizing_id,
)
from ml.models.vocab_actions import ROOT_ACTION_VOCAB, FACING_ACTION_VOCAB


def _sum(d: dict[str, float]) -> float:
    return float(sum(float(v) for v in d.values()))


def test_oop_root_kind_for_bet_sizing_id():
    assert oop_root_kind_for_bet_sizing_id("srp_hu.PFR_IP") == "donk"
    assert oop_root_kind_for_bet_sizing_id("3bet_hu.Aggressor_OOP") == "bet"
    assert oop_root_kind_for_bet_sizing_id("3bet_hu.Aggressor_IP") == "donk"
    assert oop_root_kind_for_bet_sizing_id("limped_single.BB_IP") == "donk"


def test_map_root_mix_buckets_and_normalizes():
    mix = {"CHECK": 0.25, "BET 67%": 0.75}
    out = map_root_mix_to_root_vocab(mix, root_kind="donk", size_pct=67)

    # only ROOT_ACTION_VOCAB keys
    assert set(out.keys()) == set(ROOT_ACTION_VOCAB)

    s = _sum(out)
    assert math.isclose(s, 1.0, rel_tol=0, abs_tol=1e-6)

    # 67 -> nearest is 66 (given your vocab)
    assert out["BET_66"] > 0
    assert out["CHECK"] > 0


def test_map_root_mix_ignores_other_bet_sizes():
    mix = {"CHECK": 0.2, "BET 25%": 0.4, "BET 75%": 0.4}
    out = map_root_mix_to_root_vocab(mix, root_kind="donk", size_pct=25)
    # should only count the 25% bet into the chosen token
    assert out["BET_25"] > 0
    assert out["BET_75"] == 0.0


def test_map_facing_mix_raises_bucketed_and_normalized():
    mix = {"FOLD": 0.1, "CALL": 0.4, "RAISE 3X": 0.3, "ALL-IN": 0.2}
    out = map_facing_mix_to_facing_vocab(mix, raise_mults=[2.0, 3.0, 4.5])

    assert set(out.keys()) == set(FACING_ACTION_VOCAB)
    s = _sum(out)
    assert math.isclose(s, 1.0, rel_tol=0, abs_tol=1e-6)

    assert out["FOLD"] > 0
    assert out["CALL"] > 0
    assert out["RAISE_TO_300"] > 0
    assert out["ALLIN"] > 0