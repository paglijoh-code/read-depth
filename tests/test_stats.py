import math

import numpy as np
import pytest

from surgery.stats import (
    accuracy, ece, grouped_bootstrap, macro_f1, mcnemar,
    paired_grouped_bootstrap, softmax, summarize,
)


def test_metrics_by_hand():
    y, p = [0, 0, 1, 2], [0, 1, 1, 1]
    # 2/4 correct; F1s: class 0 = 2/3, class 1 = 2/4,
    # class 2 (never predicted) = 0. Classes 3..5 are excluded.
    assert accuracy(y, p) == 0.5
    assert macro_f1(y, p) == pytest.approx((2 / 3 + 1 / 2) / 3)
    # Predicted-only class 1 counts: F1s = 2/3 and 0.
    assert macro_f1([0, 0], [0, 1]) == pytest.approx(1 / 3)
    probs = [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3],
             [0.1, 0.8, 0.1], [1, 0, 0]]
    # Two bins: low acc=.5, conf=.4; high acc=1, conf=.9.
    # ECE = .5*.1 + .5*.1 = .1; confidence 1 is included.
    assert ece([0, 0, 1, 0], probs, n_bins=2) == pytest.approx(0.1)
    # Confidence exactly .5 belongs to the upper bin: signed errors partially
    # cancel (.5 - 1)/2; putting .5 in the lower bin would instead give .75.
    assert ece([0, 0], [[0.5, 0.5], [0, 1]], 2) == pytest.approx(.25)


def test_softmax():
    probs = softmax([[1000, 1001, 999], [-1000, -1000, -1000]])
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    np.testing.assert_allclose(probs[1], [1 / 3] * 3)
    np.testing.assert_allclose(probs[0], softmax([[0, 1, -1]])[0])


def test_bootstrap_normal_approximation_and_determinism():
    y = np.zeros(2000, dtype=int)
    p = np.r_[np.zeros(1400, dtype=int), np.ones(600, dtype=int)]
    result = grouped_bootstrap(accuracy, np.arange(2000), y, p, n_boot=3000)
    half = 1.96 * math.sqrt(.7 * .3 / 2000)
    assert result["lo"] == pytest.approx(.7 - half, abs=.01)
    assert result["hi"] == pytest.approx(.7 + half, abs=.01)
    assert result == grouped_bootstrap(accuracy, np.arange(2000), y, p,
                                       n_boot=3000)


def test_perfect_cluster_correlation():
    y = np.zeros(40, dtype=int)
    p = np.r_[np.zeros(28, dtype=int), np.ones(12, dtype=int)]
    base = grouped_bootstrap(accuracy, np.arange(40), y, p, n_boot=3000)
    grouped = grouped_bootstrap(accuracy, np.repeat(np.arange(40), 10),
                                np.repeat(y, 10), np.repeat(p, 10), n_boot=3000)
    naive = grouped_bootstrap(accuracy, np.arange(400),
                              np.repeat(y, 10), np.repeat(p, 10), n_boot=3000)
    assert grouped == base
    assert grouped["half_width"] > 2 * naive["half_width"]


def test_constant_metric_and_unequal_groups():
    groups = ["a", "a", "a", "b"]
    result = grouped_bootstrap(lambda x: 7, groups, [0] * 4, n_boot=100)
    assert result["point"] == result["lo"] == result["hi"] == 7
    assert result["half_width"] == 0
    # Cluster draws aa, ab/ba, bb have row means 1, 3/4, 0.
    result = grouped_bootstrap(accuracy, groups, [0] * 4, [0, 0, 0, 1],
                               n_boot=1000)
    assert result["point"] == .75
    assert (result["lo"], result["hi"]) == (0, 1)
    # The general path must also retain every row in unequal-size groups.
    generic = grouped_bootstrap(lambda y, p: accuracy(y, p), groups,
                                [0] * 4, [0, 0, 0, 1], n_boot=1000)
    assert generic == result


@pytest.mark.parametrize("metric", [accuracy, macro_f1])
def test_paired(metric):
    y = np.tile(np.arange(6), 20)
    a = (y + 1) % 6
    identical = paired_grouped_bootstrap(metric, np.arange(120), y, a, a,
                                         n_boot=300)
    assert identical["point"] == identical["half_width"] == 0
    assert identical["p_two_sided"] == 1
    better = paired_grouped_bootstrap(metric, np.arange(120), y, a, y,
                                      n_boot=300)
    assert better["point"] == 1
    assert better["lo"] > 0
    assert better["p_two_sided"] == 0
    assert better == paired_grouped_bootstrap(metric, np.arange(120), y, a, y,
                                             n_boot=300)


def test_mcnemar():
    assert mcnemar([1] * 10, [0] * 10) == dict(b=10, c=0, p_exact=2 * .5**10)
    assert mcnemar([1, 0], [0, 1])["p_exact"] == 1
    assert mcnemar([1, 0], [1, 0])["p_exact"] == 1
    # b=7,c=2: twice the Binomial(9,.5) lower tail through 2.
    exact = 2 * sum(math.comb(9, k) for k in range(3)) / 2**9
    assert mcnemar([1] * 7 + [0] * 2, [0] * 7 + [1] * 2)["p_exact"] == exact


def test_summary():
    logits = [[5, 0, 0, 0, 0, 0], [0, 5, 0, 0, 0, 0]]
    result = summarize([0, 1], logits, ["a", "b"], n_boot=30)
    assert result["n"] == result["n_groups"] == 2
    assert result["accuracy"]["point"] == result["macro_f1"]["point"] == 1
    assert result["ece"]["point"] == pytest.approx(1 - math.exp(5)/(math.exp(5)+5))
    assert result == summarize([0, 1], logits, ["a", "b"], n_boot=30)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        accuracy([], [])
    with pytest.raises(ValueError):
        macro_f1([6], [0])
    with pytest.raises(ValueError):
        ece([0], [[.2, .2]])
    with pytest.raises(ValueError):
        grouped_bootstrap(accuracy, [0], [0, 1], [0, 1])
    with pytest.raises(ValueError):
        grouped_bootstrap(accuracy, [0], [0], [0], n_boot=0)
