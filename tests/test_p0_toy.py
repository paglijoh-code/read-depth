import numpy as np
import pytest

from surgery.atlas import p0_measures as M
from surgery.atlas.p0_harness import analyze_task
from surgery.atlas.p0_toy import ToyAdapter, make_toy_items


def _run(shuffle=False, k=4, B=8, n_perm=20):
    ad = ToyAdapter(B=B, k=k)
    ids, y, an, y_an, sf = make_toy_items(600, seed=1)
    groups = np.arange(600) % 10
    y = np.array(y)
    ytr = y[:400].copy()
    if shuffle:
        ytr = np.random.default_rng(5).permutation(ytr)
    r = analyze_task(ad, ids[:400], ytr, groups[:400], ids[400:], y[400:], an[400:], y_an[400:], sf[400:],
                     list(range(4)), n_perm=n_perm, n_flip=2000, n_draw=500)
    return r


@pytest.fixture(scope="module")
def res():
    return _run()


def test_toy_use_valid_and_onset_at_planted_block(res):
    k = 4
    assert res["valid"]["use"], res["use"]["peak"]
    on = res["onset"]["use"]
    assert k - 1 < on["layers"] <= k + 1e-9
    assert res["n_pairs_qualifying"] >= 40


def test_toy_arrival_not_earlier(res):
    assert res["onset"]["arrival"] is not None
    # tolerance 0.05 layers, not 1e-9: the use curve is probe minus a per-block random baseline, which bends its
    # plateau slightly and can move its half-peak crossing by a few hundredths of a layer
    assert res["onset"]["arrival"]["layers"] >= res["onset"]["use"]["layers"] - 0.05


def test_toy_random_and_surface_controls(res):
    k = 4
    rr = np.array(res["use"]["random_recovery"])
    assert rr[k:].mean() < 0.3
    assert np.abs(res["use"]["surface_recovery"]).max() < 1e-6


def test_toy_probe_and_twin_signal_and_onset(res):
    assert res["valid"]["probe"] and res["valid"]["twin"]
    assert res["onset"]["probe"]["layers"] > 3
    assert res["probe"]["acc"][0] < 0.45 and res["probe"]["acc"][3] < 0.45  # no key info before block k
    assert res["probe"]["acc"][4] > 0.95


def test_toy_shuffled_labels_fail_probe_filter():
    r = _run(shuffle=True)
    assert not r["probe"]["valid"]
    assert r["onset"]["probe"] is None


def test_toy_other_planted_block():
    r = _run(k=6, B=10)
    assert 5 < r["onset"]["use"]["layers"] <= 6 + 1e-9
