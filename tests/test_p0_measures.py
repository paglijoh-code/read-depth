import numpy as np

from surgery.atlas import p0_measures as M


def test_onset_interpolation_and_width():
    c = [0, 0, 1, 2, 2]  # peak 2, half = 1 reached exactly at index 2; 10% at 1+0.2, 90% at 2+0.8
    o = M.onset(c, B=4)
    assert abs(o["layers"] - 2.0) < 1e-12 and abs(o["rel"] - 0.5) < 1e-12
    assert abs(o["width"] - (2.8 - 1.2) / 4) < 1e-12
    o = M.onset([0, 0.2, 0.8, 1.0], B=3)  # half 0.5 between 0.2 and 0.8
    assert abs(o["layers"] - 1.5) < 1e-12
    assert M.onset([0, -0.1, 0.0], 2) is None


def test_onset_block_one():
    assert M.onset([0, 1, 1, 1], 3)["layers"] == 0.5


def test_probe_subspace_orthonormal_and_centred():
    rng = np.random.default_rng(0)
    coef, scale = rng.standard_normal((4, 20)), rng.uniform(0.5, 2, 20)
    Q = M.probe_subspace(scale, coef)
    assert Q.shape == (20, 3) and np.allclose(Q.T @ Q, np.eye(3), atol=1e-6)
    raw = coef / scale
    raw = raw - raw.mean(0)
    assert np.allclose(Q @ (Q.T @ raw.T), raw.T, atol=1e-6)  # span holds all centred class directions


def test_random_subspaces_seeded_orthonormal():
    a, b = M.random_subspaces(16, 4, seed=3), M.random_subspaces(16, 4, seed=3)
    assert np.array_equal(a, b) and a.shape == (5, 10, 16, 3)
    assert np.allclose(a[2, 1].T @ a[2, 1], np.eye(3), atol=1e-5)
    assert not np.allclose(a[1], a[2])


def _blocks(n, nb, signal_from, rng):
    y = np.arange(n) % 4
    S = rng.standard_normal((n, nb, 12)).astype(np.float32)
    for b in range(signal_from, nb):
        S[:, b, :3] += 3.0 * np.eye(4)[y][:, :3] - 0.75
    return S, y


def test_probe_curve_detects_signal_and_rejects_noise():
    rng = np.random.default_rng(0)
    S, y = _blocks(600, 5, 2, rng)
    g = np.arange(600) % 10
    r = M.probe_curve(S[:400], y[:400], g[:400], S[400:], y[400:], n_perm=30, seed=0)
    assert r["acc"][0] < 0.4 and r["acc"][4] > 0.8
    assert r["valid"] is True and r["test"] == "binomial_bonferroni"  # v11: the binomial test needs no 100 permutations
    r = M.probe_curve(S[:400], y[:400], g[:400], S[400:], y[400:], n_perm=100, seed=0)
    assert r["valid"] and r["p_perm"] < 0.01 and r["peak_selectivity"] >= 0.2
    assert r["p_binom_bonf"] < 0.01 and len(r["fits"]) == 5 and len(r["cv_acc"]) == 5
    yp = np.random.default_rng(1).permutation(y[:400])
    r = M.probe_curve(S[:400], yp, g[:400], S[400:], y[400:], n_perm=100, seed=0)
    assert not r["valid"]


def test_twin_divergence_signal_and_null():
    rng = np.random.default_rng(0)
    n, nb, d = 200, 4, 16
    I = rng.standard_normal((n, nb, d)).astype(np.float32)
    A, F = I.copy(), I.copy()
    A[:, 2:] = rng.standard_normal((n, 2, d))  # answer twin differs from block 2 on
    F[:, 1:] += 0.05 * rng.standard_normal((n, 3, d))
    r = M.twin_divergence(I, A, F, n_draw=500)
    assert r["valid"] and r["peak_block"] >= 2 and r["t"][0] == 0.0
    r = M.twin_divergence(I, F, F.copy() + 0.0, n_draw=500)
    assert not r["valid"]


def test_pair_mask_and_delta():
    li = np.array([[2, 0, 0, 0.], [2, 0, 0, 0], [0, 3, 0, 0], [0.5, 0.3, 0, 0]])
    lt = np.array([[0, 3, 0, 0.], [0, 2.5, 0, 0], [0, 0, 5, 0], [0, 0.5, 0, 0]])
    ai, at = np.array([0, 0, 1, 0]), np.array([1, 1, 2, 1])
    m = M.pair_mask(li, lt, ai, at)
    assert m.tolist() == [True, True, True, False]  # last pair: gap 0.7 < 1 logit
    d = M.delta(li, ai, at)
    assert np.allclose(d, [-2, -2, -3, -0.2])
    wrong = M.pair_mask(li, np.array([[3, 0, 0, 0.]] * 4), ai, at)  # twin answered wrongly
    assert not wrong.any()


def test_use_scores_recovers_known_ratio_and_signflip():
    rng = np.random.default_rng(0)
    P, K, R = 80, 6, 10
    d_item = -2 + 0.1 * rng.standard_normal(P)
    d_twin = 2 + 0.1 * rng.standard_normal(P)
    full = d_twin - d_item
    frac = np.array([0, 0, 0.1, 0.9, 0.9, 0.9])
    d_probe = d_item[:, None] + frac[None] * full[:, None] + 0.05 * rng.standard_normal((P, K))
    d_rand = d_item[:, None, None] + 0.05 * full[:, None, None] * np.ones((P, R, K)) + 0.05 * rng.standard_normal((P, R, K))
    d_surf = d_item[:, None] + 0.01 * rng.standard_normal((P, K))
    r = M.use_scores(d_item, d_twin, d_probe, d_rand, d_surf, n_flip=2000)
    assert len(r["use"]) == K + 1 and r["use"][0] == 0
    assert abs(r["use"][4] - 0.85) < 0.05 and abs(r["probe_recovery"][4] - 0.9) < 0.05
    assert r["valid"] and r["p_signflip"] < 0.01 and r["peak_block"] in (4, 5, 6)
    # null: probe no better than random -> invalid
    d_probe0 = d_item[:, None] + 0.05 * full[:, None] + 0.05 * rng.standard_normal((P, K))
    r0 = M.use_scores(d_item, d_twin, d_probe0, d_rand, d_surf, n_flip=2000)
    assert not r0["valid"]
    # too few pairs
    r1 = M.use_scores(d_item[:30], d_twin[:30], d_probe[:30], d_rand[:30], d_surf[:30], n_flip=500)
    assert not r1["valid"]
    # surface recovery >= 0.05 invalidates
    r2 = M.use_scores(d_item, d_twin, d_probe, d_rand, d_surf + 0.1 * full[:, None], n_flip=500)
    assert not r2["valid"]


def test_arrival_scores_validity():
    P, K = 50, 5
    di, dt = np.full(P, -2.0), np.full(P, 2.0)
    whole = di[:, None] + np.array([0, 0.6, 1, 1, 1.0])[None] * 4
    ws = np.tile(di[:, None], (1, K))
    r = M.arrival_scores(di, dt, whole, ws, B=6)
    assert r["valid"] and r["arrival"][0] == 0
    late = di[:, None] + np.array([0, 0, 0, 0, 1.0])[None] * 4  # onset after block B-2
    assert not M.arrival_scores(di, dt, late, ws, B=6)["valid"]
    assert not M.arrival_scores(di, dt, whole, ws + 1.0, B=6)["valid"]


def test_logit_lens_acc():
    lg = np.zeros((4, 3, 4))
    lg[:, 1, :] = np.eye(4)
    assert M.logit_lens_acc(lg, np.arange(4)) == [0.25, 1.0, 0.25]


def test_twin_valid_sym_rejects_swapped_twins():
    rng = np.random.default_rng(3)
    n, nb, d = 60, 5, 8
    I = rng.standard_normal((n, nb, d))
    A = I + 0.0
    F = I + 0.0
    A[:, 3:] += 2.0 * rng.standard_normal((n, nb - 3, d))  # answer twin moves late blocks a lot
    F[:, 1:3] += 0.5 * rng.standard_normal((n, 2, d))  # surface twin moves early blocks a little
    A[:, 1:3] += 0.1 * rng.standard_normal((n, 2, d))
    r = M.twin_divergence(I, A, F, n_draw=200)
    assert r["valid"] and r["valid_sym"] and r["trough"] < 0
    s = M.twin_divergence(I, F, A, n_draw=200)  # swapped
    assert not s["valid_sym"]
