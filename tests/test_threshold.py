"""Threshold from data: held-out good scores, plus damaged parts when there are some.

Changelog
  0.1.0  2026-11-12  first version.
  0.2.0  2026-11-12  audit H2 / M8: the cost rule with the expected defect rate and the false-reject cap; a
                     damaged picture that scores like a good part never moves the line; the line never rises
                     above the good-only line; a held-out good outlier is set aside and named.
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


def test_overlap_uses_the_cost_balance_only_within_the_cap():
    rng = np.random.default_rng(4)
    good = rng.gumbel(5, 0.3, 20)
    t_good = th.from_scores(good)["threshold"]
    mild = good.max() + 0.5 * (t_good - good.max())                     # above every good part, below the line
    bad = np.concatenate([rng.uniform(9, 20, 10), [mild]])
    # many damaged parts in production and a cap that allows it: the line moves down to catch the mild one
    t = th.from_scores(good, bad, cost_bad_accepted=10, cost_good_rejected=1, expected_defect_rate=0.3,
                       max_false_reject=0.2)
    assert "cost balance" in t["method"] and t["threshold"] < mild < t_good
    assert t["est_false_reject"] <= 0.2
    # rare damage: catching it is not worth the good parts it would scrap
    t2 = th.from_scores(good, bad, expected_defect_rate=0.0005, max_false_reject=0.2)
    assert t2["threshold"] == pytest.approx(t_good)
    # no room under the cap: the line stays
    t3 = th.from_scores(good, bad, expected_defect_rate=0.3, max_false_reject=0.01)
    assert t3["threshold"] == pytest.approx(t_good)


AUDIT_GOOD = np.random.default_rng(7).gumbel(5.0, 0.5, 20)              # 20 held-out good scores (audit H2)
AUDIT_CLEAR = np.random.default_rng(8).uniform(9, 20, 9)                 # nine clearly damaged parts


@pytest.mark.parametrize("tenth", [5.4, 5.9, 5.25])
def test_one_damaged_picture_that_scores_like_a_good_one_never_moves_the_line(tenth):
    """The audit's cases: the 10th damaged picture shows its undamaged face (5.4), is faint (5.9), or scores
    below the good median (5.25). The first build moved the line to 49 % / 16 % false rejects, or lifted it
    by ~45 %. Now the line is the good-only line and the picture is named."""
    t_good = th.from_scores(AUDIT_GOOD)["threshold"]
    t = th.from_scores(AUDIT_GOOD, np.append(AUDIT_CLEAR, tenth), bad_names=["d%d" % k for k in range(10)])
    assert t["threshold"] <= t_good + 1e-9                               # never raised (audit fix c)
    true_fr = float(__import__("scipy").stats.gumbel_r.sf(t["threshold"], 5.0, 0.5))
    assert true_fr < 0.02                                                 # long-run false rejects stay ~1 %
    if tenth <= AUDIT_GOOD.max():
        assert [d["name"] for d in t["look_like_good"]] == ["d9"] and "d9" in t["text"]


def test_line_never_rises_above_the_good_only_line():
    good = AUDIT_GOOD
    bad = np.append(AUDIT_CLEAR, np.median(good) - 0.1)
    assert th.from_scores(good, bad)["threshold"] <= th.from_scores(good)["threshold"] + 1e-9


def test_a_held_out_good_outlier_is_set_aside_and_named():
    good = np.append(np.random.default_rng(9).gumbel(5.2, 0.15, 20), 9.0)
    names = ["g%02d.png" % k for k in range(21)]
    t = th.from_scores(good, good_names=names)
    assert t["good_set_aside"] and t["good_set_aside"][0]["name"] == "g20.png"
    assert t["threshold"] < 9.0 and "g20.png" in t["text"]
    plain = np.random.default_rng(10).gumbel(5.0, 0.3, 30)
    assert not th.from_scores(plain)["good_set_aside"]


def test_needs_three_good_scores():
    with pytest.raises(ValueError):
        th.from_scores([1.0, 2.0])


def test_wilson_interval():
    lo, hi = th.wilson(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.75
