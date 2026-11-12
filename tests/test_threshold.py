"""Threshold from data: held-out good scores, plus damaged parts when there are some.

Changelog
  0.1.0  2026-11-12  first version.
"""
import numpy as np
import pytest

from qc import threshold as th


def test_good_only_threshold_sits_above_every_good_score():
    rng = np.random.default_rng(0)
    good = rng.gumbel(5, 0.3, 20)
    t = th.from_scores(good)
    assert t["threshold"] >= good.max()
    assert 0.005 < t["est_false_reject"] <= 0.0101
    assert "No damaged parts yet" in t["text"]


def test_threshold_follows_the_data_not_a_fixed_number():
    rng = np.random.default_rng(1)
    a = th.from_scores(rng.gumbel(5, 0.3, 20))["threshold"]
    b = th.from_scores(rng.gumbel(50, 3, 20))["threshold"]
    assert b > 5 * a


def test_lower_target_moves_the_line_up():
    good = np.random.default_rng(2).gumbel(5, 0.3, 30)
    assert th.from_scores(good, target_false_reject=0.001)["threshold"] > th.from_scores(good, target_false_reject=0.05)["threshold"]


def test_separable_damaged_parts_keep_the_good_only_line():
    rng = np.random.default_rng(3)
    good = rng.gumbel(5, 0.3, 20)
    bad = rng.uniform(9, 20, 12)
    t = th.from_scores(good, bad)
    assert t["damaged_caught"] == 12
    assert t["threshold"] == pytest.approx(th.from_scores(good)["threshold"])
    assert "all 12 damaged parts score above it" in t["method"]


def test_overlap_uses_the_cost_balance():
    rng = np.random.default_rng(4)
    good = rng.gumbel(5, 0.3, 20)
    bad = np.concatenate([rng.uniform(9, 20, 10), [5.6, 5.8]])       # two mild ones among the good tail
    t = th.from_scores(good, bad, cost_bad_accepted=10, cost_good_rejected=1)
    assert "cost balance" in t["method"]
    assert t["threshold"] < th.from_scores(good)["threshold"]
    # cheap misses -> the line stays higher
    t2 = th.from_scores(good, bad, cost_bad_accepted=0.1, cost_good_rejected=1)
    assert t2["threshold"] >= t["threshold"]


def test_needs_three_good_scores():
    with pytest.raises(ValueError):
        th.from_scores([1.0, 2.0])


def test_wilson_interval():
    lo, hi = th.wilson(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.75
