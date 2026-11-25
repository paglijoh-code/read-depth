"""Tests of surgery/atlas/tree.py on planted data (fast: tiny null/bootstrap counts)."""
import json

import numpy as np
import pytest

from surgery.atlas import tree as T

MODELS = T.FITTED
META = T.model_meta("/nonexistent/models.json")
FAMS = [f"fam{i:02d}" for i in range(30)]
TYPES = ["surface form", "fact lookup", "binding/reference", "arithmetic", "code tracing", "rule application"]


def make_rows(onset_fn, kind="unmodified", prompt="plain", seed=0, models=MODELS, noise=0.0, valid_fn=None,
              half_noise=0.01, rng_seed=0, use_fn=None, acc_fn=None, twin_fn=None):
    """onset_fn(model_idx, family_idx, task_in_family) -> rel onset (probe). use_fn / twin_fn likewise (default none)."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for mi, m in enumerate(models):
        for fi, f in enumerate(FAMS):
            for ti in range(2):
                rel = float(np.clip(onset_fn(mi, fi, ti) + rng.normal(0, noise), 0.02, 0.98))
                B = META[m]["n_blocks"]
                ok = True if valid_fn is None else valid_fn(mi, fi, ti)
                on = {"probe": dict(rel=rel, layers=rel * B, width=0.1) if ok else None, "twin": None, "use": None}
                valid = {"probe": ok, "twin": False, "use": False}
                half = {"probe": [rel + rng.normal(0, half_noise), rel + rng.normal(0, half_noise)] if ok else None,
                        "twin": None, "use": None}
                for name, fn in (("use", use_fn), ("twin", twin_fn)):
                    if fn is not None:
                        v = float(np.clip(fn(mi, fi, ti, rel) + rng.normal(0, noise), 0.02, 0.98))
                        on[name] = dict(rel=v, layers=v * B, width=0.1)
                        valid[name] = True
                        half[name] = [v + rng.normal(0, half_noise), v + rng.normal(0, half_noise)]
                rows.append(dict(model=m, lineage=META[m]["lineage"], n_params=META[m]["n_params"], n_blocks=B,
                                 release_date=None, task=f"{f}_{'ab'[ti]}", family=f, type_label=TYPES[fi % 6],
                                 kind=kind, prompt=prompt, seed=seed, valid=valid, onset=on, onset_half=half,
                                 onset_dropped={}, early_read=bool(ok and rel <= 0.10),
                                 trained_acc=None if acc_fn is None else acc_fn(mi, fi, ti)))
    return rows


# ------------------------------------------------------------------------------------------------ primitives
def test_icc_2_1():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    assert T.icc_2_1(np.stack([a, a], 1)) == pytest.approx(1.0)
    assert T.icc_2_1(np.stack([a, rng.normal(0, 1, 200)], 1)) < 0.2
    assert 0.4 < T.icc_2_1(np.stack([a + rng.normal(0, 1, 200), a + rng.normal(0, 1, 200)], 1)) < 0.6  # true 0.5


def test_ari():
    assert T.ari([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert T.ari([0, 0, 1, 1], [0, 1, 0, 1]) < 0.1
    from sklearn.metrics import adjusted_rand_score
    a, b = np.random.default_rng(1).integers(0, 3, (2, 50))
    assert T.ari(a, b) == pytest.approx(adjusted_rand_score(a, b))


def test_signflip_exact_toy():
    # 3 models, all improved by 1: only the all-plus flip reaches mean 1 -> p = 1/8
    assert T.signflip_p([1.0, 1.0, 1.0]) == pytest.approx(1 / 8)
    # d = (2, -1, 0.5): mean 0.5; flips with mean >= 0.5: (+,+,+)=1.5/3 .. enumerate by hand
    d = np.array([2.0, -1.0, 0.5])
    n_ge = sum(1 for s in __import__("itertools").product([-1, 1], repeat=3) if (np.array(s) @ d) / 3 >= 0.5 - 1e-12)
    assert T.signflip_p(d) == pytest.approx(n_ge / 8)
    assert T.signflip_p(np.ones(14)) == pytest.approx(1 / 2 ** 14)  # exact 2^14 enumeration


def test_em_matches_sklearn():
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(-2, 0.5, 60), rng.normal(2, 0.7, 40)])
    ours = T.gmm_fit_batch(x, 2, n_starts=10, floor=1e-6, rng=np.random.default_rng(0))
    gm = GaussianMixture(2, n_init=5, random_state=0, reg_covar=1e-9).fit(x[:, None])
    assert ours["ll"][0] >= gm.score(x[:, None]) * len(x) - 1e-6  # sklearn stops at tol 1e-3
    assert ours["ll"][0] == pytest.approx(gm.score(x[:, None]) * len(x), abs=0.01)
    assert np.allclose(np.sort(ours["mu"][0]), np.sort(gm.means_[:, 0]), atol=1e-2)
    assert np.allclose(np.sort(ours["w"][0]), np.sort(gm.weights_), atol=1e-2)
    # k = 1 closed form
    one = T.gmm_fit_batch(x, 1, floor=1e-6)
    assert one["mu"][0, 0] == pytest.approx(x.mean())


def test_em_floor_and_batch():
    x = np.concatenate([np.zeros(5), np.ones(5) * 3.0])  # two point masses: variance would collapse
    f = T.gmm_fit_batch(x, 2, floor=0.05, rng=np.random.default_rng(0))
    assert (f["var"] >= 0.05 - 1e-12).all()
    X = np.stack([x, x + 1])
    assert T.gmm_fit_batch(X, 2, rng=np.random.default_rng(0))["mu"].shape == (2, 2)
    b = T.bic_select(np.random.default_rng(0).normal(size=(3, 40)), n_starts=5)
    assert b["bic"].shape == (3, 3) and (b["dbic"] >= 0).all()


@pytest.mark.parametrize("dist", ["normal", "laplace"])
def test_log_concave_density(dist):
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 400) if dist == "normal" else rng.laplace(0, 1, 400)
    lc = T.LogConcave(x, smoothed=False)
    g = np.linspace(-1.5, 1.5, 13)
    true = np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi) if dist == "normal" else 0.5 * np.exp(-np.abs(g))
    est = np.exp(lc.logpdf_hat(g))
    assert np.abs(est - true).max() < 0.08
    # concave log-density, integrates to 1
    d2 = np.diff(lc.slope)
    assert (d2 <= 1e-7).all()
    assert lc.seg_mass.sum() == pytest.approx(1.0)
    grid = np.linspace(lc.xs[0], lc.xs[-1], 20001)
    assert np.trapezoid(np.exp(lc.logpdf_hat(grid)), grid) == pytest.approx(1.0, abs=1e-3)


def test_log_concave_sampler_variance_preserved():
    rng = np.random.default_rng(6)
    x = rng.normal(1, 2, 30)
    lc = T.LogConcave(x, smoothed=True)
    assert lc.var_hat <= lc.sample_var + 1e-9
    s = lc.sample(300000, rng)
    assert s.var() == pytest.approx(x.var(), rel=0.02)
    assert s.mean() == pytest.approx(lc.mean_hat, abs=0.02)
    raw = T.LogConcave(x, smoothed=False).sample(300000, rng)
    assert raw.var() == pytest.approx(lc.var_hat, rel=0.02)
    assert raw.min() >= x.min() - 1e-9 and raw.max() <= x.max() + 1e-9


def test_onset_curve_matches_p0():
    from surgery.atlas import p0_measures as M
    c = [0.0, 0.05, 0.2, 0.6, 0.9, 1.0, 0.95]
    a, b = T.onset_curve(c, 7), M.onset(c, 7)
    assert a["rel"] == pytest.approx(b["rel"]) and a["width"] == pytest.approx(b["width"])
    assert T.onset_curve([0, -1], 2) is None


# ------------------------------------------------------------------------------------------------ table pieces
def test_family_values_counts_early_and_censoring():
    stage = lambda mi, fi, ti: 0.05 if fi < 3 else 0.3 + 0.01 * fi  # families 0-2 early
    valid = lambda mi, fi, ti: not (fi == 29 and mi < 10)  # family 29 valid in only 4 models
    tab = T.Tab(make_rows(stage, valid_fn=valid), "unmodified")
    assert tab.M == 14 and tab.T == 60
    cnt = T.family_counts(tab, "probe")
    assert cnt["n_counting"] == 29 and "fam29" not in cnt["counting"]
    assert all(tab.early_task[:6]) and not tab.early_task[6]
    F, c, keep = T.family_values(tab, "probe", drop_early=True)
    assert np.isnan(F[:, :3]).all() and c[0] == 0
    # censoring: dropped curves on the 3 largest models onset later by >= 0.15 -> under-sampled
    rows = make_rows(stage, valid_fn=lambda mi, fi, ti: fi != 5)
    for r in rows:
        if r["family"] == "fam05":
            r["onset_dropped"] = {"probe": 0.9}
    tab = T.Tab(rows, "unmodified")
    cc = T.censoring_check(tab, "probe")
    assert cc["undersampled"] and cc["gap"] > 0.15 and cc["models"][0] == "Qwen2.5-3B"
    assert not T.censoring_check(T.Tab(make_rows(stage), "unmodified"), "probe")["undersampled"]
    # more than 3 families invalid on all 3 largest models
    tab = T.Tab(make_rows(stage, valid_fn=lambda mi, fi, ti: not (fi < 5 and MODELS[mi] in
                                                                   ("Qwen2.5-3B", "pythia-2.8b", "Qwen3-1.7B-Base"))),
                "unmodified")
    assert T.censoring_check(tab)["undersampled"]


def test_icc_halves_and_seeds():
    f = lambda mi, fi, ti: 0.1 + 0.8 * (fi / 30)
    tab = T.Tab(make_rows(f, half_noise=0.005, noise=0.01), "unmodified")
    assert T.icc_halves(tab) > 0.95
    tab2 = T.Tab(make_rows(f, half_noise=0.4), "unmodified")
    assert T.icc_halves(tab2) < 0.5
    s0 = T.Tab(make_rows(f, kind="trained", noise=0.01, rng_seed=1), "trained")
    s1 = T.Tab(make_rows(f, kind="trained", seed=1, noise=0.01, rng_seed=2), "trained", seed=1)
    assert T.icc_seeds(s0, s1) > 0.9
    assert np.isnan(T.icc_halves(T.Tab(make_rows(f, half_noise=0.01), "unmodified"), "use"))


# ------------------------------------------------------------------------------------------------ N1 / N2 / N3
def stages3(mi, fi, ti):
    return [0.25, 0.5, 0.75][fi % 3] + 0.004 * (ti - 0.5)


def test_n1_holds_fails_and_l0():
    tab = T.Tab(make_rows(stages3, noise=0.02), "unmodified")
    r = T.read_n1(tab, n_boot=50)
    assert r["verdict"] == "HOLDS" and r["median"] > 0.5 and r["n_pairs"] == 91 and r["leaf"] is None
    # independent orders per model
    rng = np.random.default_rng(0)
    perm = {m: rng.uniform(0.15, 0.85, 30) for m in range(14)}
    tab = T.Tab(make_rows(lambda mi, fi, ti: perm[mi][fi], noise=0.0), "unmodified")
    r = T.read_n1(tab, n_boot=50)
    assert r["verdict"] == "FAILS" and r["leaf"] == "L1" and "not shared" in r["sentence"]
    # too few pairs: only 6 models carry 15+ families
    few = make_rows(stages3, valid_fn=lambda mi, fi, ti: mi < 6 or fi < 10)
    r = T.read_n1(T.Tab(few, "unmodified"), n_boot=10)
    assert r["leaf"] == "L0" and r["n_pairs"] < 45


def test_n1_between_goes_to_l1b_without_reserves():
    rng = np.random.default_rng(2)
    shared = rng.uniform(0.15, 0.85, 30)
    own = {m: rng.uniform(0.15, 0.85, 30) for m in range(14)}
    tab = T.Tab(make_rows(lambda mi, fi, ti: 0.45 * shared[fi] + 0.55 * own[mi][fi]), "unmodified")
    r = T.read_n1(tab, n_boot=100)
    assert r["verdict"] == "between", r["median"]
    assert r["leaf"] == "L1b"
    r2 = T.read_n1(tab, n_boot=100, reserves_available=True)
    assert r2["leaf"] is None and r2["reserves_pending"]


def test_n2_three_stages_discrete():
    tab = T.Tab(make_rows(stages3, noise=0.02, half_noise=0.003), "unmodified")
    r = T.read_n2(tab, n_null=60, n_boot=40, n_starts=6)
    assert r["verdict"] == "discrete" and r["k"] == 3 and r["leaf"] is None
    assert r["null_beaten"] and r["stability"] >= 0.7 and r["separation_ok"]
    assert r["icc"] > 0.7


def test_n2_unimodal_continuous_and_lmeas():
    rng = np.random.default_rng(4)
    fam = rng.normal(0.5, 0.12, 30)
    f = lambda mi, fi, ti: fam[fi] + 0.01 * (ti - 0.5)
    tab = T.Tab(make_rows(f, noise=0.02), "unmodified")
    r = T.read_n2(tab, n_null=80, n_boot=20, n_starts=6)
    assert r["verdict"] == "continuous" and r["leaf"] == "L2c"
    # unreliable halves -> L-meas, and the ladder value rescues it
    bad = T.Tab(make_rows(stages3, noise=0.02, half_noise=0.5), "unmodified")
    r = T.read_n2(bad, n_null=20, n_boot=10, n_starts=4)
    assert r["leaf"] == "L-meas" and r["ladder_run"] is False and not r["passed"]
    r = T.read_n2(bad, n_null=40, n_boot=10, n_starts=4, icc_fresh=0.8)
    assert r["verdict"] == "discrete"


def test_n2_trivial_close_and_contrast():
    # two groups, the low one made only of early-read tasks -> L2s
    f = lambda mi, fi, ti: 0.06 if fi < 10 else 0.6 + 0.004 * (ti - 0.5)
    tab = T.Tab(make_rows(f, noise=0.0), "unmodified")
    r = T.read_n2(tab, n_null=60, n_boot=30, n_starts=6, icc=0.9)
    assert r["verdict"] == "trivial" and r["leaf"] == "L2s"
    # contrast
    d = dict(verdict="discrete", leaf=None)
    c = T.contrast({"unmodified": dict(verdict="continuous", leaf="L2c"), "trained": d})
    assert c["leaf"] == "L2t"
    c = T.contrast({"unmodified": d, "trained": dict(verdict="close", leaf="L2x")})
    assert c["leaf"] == "L2u" and "too close" in c["sentence"]
    assert T.contrast({"unmodified": d, "trained": d}) is None


def test_n3_stages_hold_out_of_sample():
    tab = T.Tab(make_rows(stages3, noise=0.02, half_noise=0.003), "unmodified")
    n2 = T.read_n2(tab, n_null=40, n_boot=10, n_starts=6)
    assert n2["verdict"] == "discrete"
    tiny = T.Tab(make_rows(stages3, noise=0.02, models=[T.HELDOUT]), "unmodified", models=[T.HELDOUT])
    n3 = T.read_n3(tab, n2, n_null=20, n_starts=6, n_perm=50, tiny_tab=tiny)
    assert n3["N3a"]["verdict"] == "HOLDS" and n3["N3a"]["median_ari"] >= 0.5
    assert len(n3["N3a"]["folds"]) == 5
    assert n3["N3b"]["verdict"] == "HOLDS"
    assert n3["leaf"] == "L3" and "five lineages" in n3["sentence"]
    assert n3["N3c"]["holds"]


def test_family_folds_stratified():
    types = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
    fold = T.family_folds(list(range(30)), types, 5, seed=1)
    assert np.bincount(fold).tolist() == [6] * 5
    for t in "abc":
        idx = [i for i, x in enumerate(types) if x == t]
        assert np.bincount(fold[idx], minlength=5).max() == 2


# ------------------------------------------------------------------------------------------------ N4
LIN = {m: META[m]["lineage"] for m in MODELS}


def lineage_profile(seed=0, noise=0.02):
    rng = np.random.default_rng(seed)
    eff = {l: rng.uniform(0.15, 0.85, 30) for l in set(LIN.values())}
    return lambda mi, fi, ti: eff[LIN[MODELS[mi]]][fi]


def tabs_from(onset_fn, noise=0.01, with_trained=False, **kw):
    un = T.Tab(make_rows(onset_fn, noise=noise, **kw), "unmodified")
    out = {"unmodified": un}
    if with_trained:
        out["trained"] = T.Tab(make_rows(onset_fn, kind="trained", noise=noise, rng_seed=9,
                                         acc_fn=lambda mi, fi, ti: 0.6, **kw), "trained")
    return out


def test_n4_lineage_only_gives_l4m():
    r = T.read_n4(tabs_from(lineage_profile()), gate=1, n_null=60, n_boot=30)
    assert r["N4b"]["mae_mean"]["i"] < r["N4b"]["mae_mean"]["o"]
    assert r["leaf"] == "L4m", (r["leaf"], r["N4b"]["mae_mean"], r["N4b"]["wins"])
    assert not r["splits_lineage"] or not r["groups_exist"]


def test_n4_size_gradient_gives_l4g():
    rng = np.random.default_rng(1)
    base, slope = rng.uniform(0.3, 0.6, 30), rng.uniform(-0.08, 0.08, 30)
    lp = {m: np.log(META[m]["n_params"] / 1e9) for m in MODELS}
    f = lambda mi, fi, ti: base[fi] + slope[fi] * lp[MODELS[mi]]
    r = T.read_n4(tabs_from(f, noise=0.002), gate=1, n_null=40, n_boot=20)
    assert r["leaf"] == "L4g" and r["variant"] is None, (r["leaf"], r["N4b"]["mae_mean"])
    assert r["N4b"]["winner"] == "ii"


def test_n4_cross_lineage_groups_give_l4plus():
    rng = np.random.default_rng(3)
    grp = np.arange(14) % 3  # groups cut across lineages
    cen = rng.uniform(0.15, 0.85, (3, 30))
    f = lambda mi, fi, ti: cen[grp[mi], fi]
    r = T.read_n4(tabs_from(f, noise=0.01), gate=1, n_null=60, n_boot=30)
    assert r["groups_exist"] and r["splits_lineage"]
    assert r["leaf"] == "L4+", (r["leaf"], r["N4b"]["mae_mean"], r["N4b"]["wins"])
    assert "group matches its profile" in r["sentence"]


def test_n4_x_and_u():
    f = lambda mi, fi, ti: 0.4
    r = T.read_n4(tabs_from(f, valid_fn=lambda mi, fi, ti: fi < 8 or mi < 5), gate=1, n_null=5, n_boot=5)
    assert r["leaf"] == "L4x" and r["n_probe_core"] < 12
    rng = np.random.default_rng(5)
    noise = rng.uniform(0.15, 0.85, (14, 30))
    r = T.read_n4(tabs_from(lambda mi, fi, ti: noise[mi, fi], noise=0.0), gate=1, n_null=30, n_boot=10)
    assert r["leaf"] == "L4u"


def test_n4_gate2_features():
    f = lambda mi, fi, ti: 0.3 + 0.02 * (fi % 10)
    tabs = tabs_from(f, with_trained=True, use_fn=lambda mi, fi, ti, rel: min(rel + 0.1, 0.95),
                     twin_fn=lambda mi, fi, ti, rel: rel)
    prof = T.build_profile(tabs, gate=2)
    assert set(prof["features_used"]) == {"probe_onset", "probe_width", "twin_onset", "use_onset", "training_shift",
                                          "learnability"}
    assert prof["X"].shape == (14, 6 * 30 - 0) or prof["X"].shape[1] >= 5 * 30
    g1 = T.build_profile(tabs, gate=1)
    assert set(g1["features_used"]) == {"probe_onset", "probe_width", "twin_onset"}


def test_n4b_estimators_on_toy_matrix():
    # two groups of 7 models with identical rows inside each group: the cluster mean is exact
    rng = np.random.default_rng(0)
    cen = rng.normal(size=(2, 40))
    X = np.vstack([np.tile(cen[0], (7, 1)), np.tile(cen[1], (7, 1))]) + rng.normal(0, 0.01, (14, 40))
    names = [f"m{i}" for i in range(14)]

    class Tb:
        n_params = np.exp(rng.normal(size=14)) * 1e9
        n_blocks = np.full(14, 24.0)
        release = [f"2024-0{1 + i % 9}-01" for i in range(14)]
        fam_type = ["a", "b"] * 5

    lineage = ["L%d" % (i % 2) for i in range(14)]  # lineages = alternating, not the groups
    prof = dict(X=X, tab=Tb, lineage=lineage, col_fam=np.repeat(np.arange(10), 4), models=names)
    b = T.n4b(prof, k_groups=2)
    assert b["winner"] == "iii", (b["mae_mean"], b["wins"])
    assert b["mae_mean"]["iii"] < 0.1 * b["mae_mean"]["o"]
    assert b["p"]["iii>o"] < 0.05


# ------------------------------------------------------------------------------------------------ N5
def test_n5_fraction_of_depth_absolute_layer_and_partial():
    rng = np.random.default_rng(0)
    task = rng.uniform(0.3, 0.7, 30)
    B = {m: META[m]["n_blocks"] for m in MODELS}

    def fn(beta):
        # layer onset = exp(task) * blocks^beta, expressed as rel = layers / blocks
        return lambda mi, fi, ti: np.exp(task[fi] * 0 + np.log(task[fi])) * B[MODELS[mi]] ** beta / B[MODELS[mi]]

    for beta, leaf in ((1.0, "L5r"), (0.0, "L5a"), (0.5, "L5p")):
        r = T.read_n5(T.Tab(make_rows(fn(beta) if beta else (lambda mi, fi, ti: task[fi] * 6 / B[MODELS[mi]] * 2),
                                      noise=0.0, half_noise=0.0), "unmodified"), n_boot=300)
        assert r["leaf"] == leaf, (beta, r["beta"], r["ci"])
    r = T.read_n5(T.Tab(make_rows(lambda mi, fi, ti: task[fi]), "unmodified"), n_boot=50, power=0.5)
    assert r["descriptive"]


def test_n5_contains_both_when_noisy():
    rng = np.random.default_rng(1)
    task = rng.uniform(0.3, 0.7, 6)
    rows = make_rows(lambda mi, fi, ti: 0.5, noise=0.3)
    rows = [r for r in rows if int(r["family"][3:]) < 3 and r["model"] in MODELS[:5]]  # tiny, noisy
    tab = T.Tab(rows, "unmodified")
    r = T.read_n5(tab, n_boot=100)
    assert r["leaf"] in ("L5n", "L5r", "L5a", "L5p")  # runs on a tiny table without error


# ------------------------------------------------------------------------------------------------ N8
def test_n8_leaves():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 0.7, 30)
    probe = lambda mi, fi, ti: base[fi] + 0.01 * (mi % 3)
    mk = lambda use: T.Tab(make_rows(probe, use_fn=use, kind="trained", noise=0.0), "trained")
    a = T.read_n8(mk(lambda mi, fi, ti, rel: rel), icc=0.9, n_boot=100)
    assert a["leaf"] == "L8a" and a["n_m"] == 14
    g = T.read_n8(mk(lambda mi, fi, ti, rel: rel + 0.2), icc=0.9, n_boot=100)
    assert g["leaf"] == "L8g" and g["gap"] == pytest.approx(0.2, abs=0.01) and "differ by 0.20" in g["sentence"]
    ind = np.random.default_rng(9).uniform(0.2, 0.8, (14, 30))
    d = T.read_n8(mk(lambda mi, fi, ti, rel: ind[mi, fi]), icc=0.9, n_boot=100)
    assert d["leaf"] == "L8d", (d["rho"], d["rho_ci"])
    # precondition: ICC too low, too few families
    assert T.read_n8(mk(lambda mi, fi, ti, rel: rel), icc=0.3)["leaf"] == "L8x"
    few = T.Tab(make_rows(probe, use_fn=lambda mi, fi, ti, rel: rel, kind="trained",
                          valid_fn=lambda mi, fi, ti: fi < 11), "trained")
    assert T.read_n8(few, icc=0.9)["leaf"] == "L8x"
    # ICC from the table's half onsets
    r = T.read_n8(mk(lambda mi, fi, ti, rel: rel), n_boot=20)
    assert r["icc_use"] > 0.9 and r["leaf"] == "L8a"


# ------------------------------------------------------------------------------------------------ form + sentences
GOOD = dict(n2=dict(power_d3_worst_k=0.9, d_star_worst=3.0), n4=dict(fpr_max=0.05), n8=dict(fpr_max=0.05),
            n5=dict(power=0.9))


def R(**nodes):
    return {k: dict(leaf=v) for k, v in nodes.items()}


def test_form_rules():
    f = T.form
    assert f(dict(could_not_test=True, unmodified={}), GOOD)["rule"] == 1
    r = dict(unmodified=R(N1="L1b"), trained=R(N1=None, N2=None, N3="L3"))
    assert f(r, GOOD)["form"].startswith("paper") and f(r, GOOD)["rule"] == 3
    # L2c needs power >= 80% at d = 3
    r = dict(unmodified=R(N1=None, N2="L2c"), trained=R(N1="L1"))
    assert f(r, GOOD)["form"].startswith("paper")
    low = dict(GOOD, n2=dict(power_d3_worst_k=0.6))
    assert f(r, low)["form"] == "short note" and f(r, low)["rule"] == 4
    # N4 FPR > 10% makes L4+ descriptive -> L4u -> internal
    r = dict(unmodified=R(N1="L1b"), trained=R(N4="L4+"))
    assert f(r, GOOD)["form"].startswith("paper")
    bad = dict(GOOD, n4=dict(fpr_max=0.2))
    out = f(r, bad)
    assert out["form"] == "internal report" and out["rule"] == 5 and out["leaves"]["trained"] == ["L4u"]
    # N8 FPR > 10%: L8d -> L8w
    r = dict(unmodified=R(N1="L1b"), trained=R(N8="L8d"))
    assert f(r, GOOD)["form"].startswith("paper")
    assert f(r, dict(GOOD, n8=dict(fpr_max=0.3)))["form"] == "internal report"
    # N5 power < 80% -> L5n (internal)
    r = dict(unmodified=R(N1="L1b"), trained=R(N5="L5p"))
    assert f(r, GOOD)["form"] == "internal report"
    assert f(r, dict(GOOD, n5=dict(power=0.5)))["leaves"]["trained"] == ["L5n"]
    # short note: L4m / L8a / L8g / L1 / L2s / L2x
    for leaf, node in (("L4m", "N4"), ("L8a", "N8"), ("L8g", "N8"), ("L1", "N1"), ("L2s", "N2"), ("L2x", "N2")):
        r = dict(unmodified=R(N1="L1b"), trained=R(**{node: leaf}))
        assert f(r, GOOD)["rule"] == 4, leaf
    # no C-sim: nothing is strengthened
    r = dict(unmodified=R(N1="L1b"), trained=R(N4="L4g"))
    assert f(r, None)["form"] == "internal report"
    # rule 6: anything else (contrast leaf alone)
    r = dict(unmodified=dict(N1=dict(leaf="L1b"), contrast=dict(leaf="L2t")), trained=None)
    assert f(r, GOOD)["rule"] == 6
    # rule 2: stop after gate 1 reads the unmodified kind only, N4 from its gate-1 reading
    r = dict(stop_after_gate1=True, unmodified=R(N1="L1b", N4="L4g"), trained=R(N3="L3"))
    out = f(r, GOOD)
    assert out["form"].startswith("paper") and out["rule"] == 3 and "trained" not in out["leaves"]
    r = dict(stop_after_gate1=True, unmodified=R(N1="L1b"), trained=R(N3="L3"))
    assert f(r, GOOD)["form"] == "internal report"
    # N4 is read on the trained kind only when both exist
    r = dict(unmodified=R(N1="L1b", N4="L4g"), trained=R(N4="L4u"))
    assert f(r, GOOD)["form"] == "internal report"


def test_leaf_sentences():
    s = T.leaf_sentence("L1", "trained", "probe", n_f=24, n_m=14, rho=0.12)
    assert "14 small base models" in s and "24 synthetic task families" in s and "0.12" in s and "trained" not in "x"
    assert "after training" in s and "probe" in s
    s = T.leaf_sentence("L2c", "unmodified", "probe", 25, 14, censored=True, weak=True, d_star=3.2)
    assert "d* = 3.20" in s and s.endswith("(weakly powered)") and "late end is under-sampled" in s
    assert T.leaf_sentence("L4g", "trained", "profile", variant="smooth").count("smooth, not grouped") == 1
    assert "release date, not lineage" in T.leaf_sentence("L4m", "trained", "profile", variant="release")
    s = T.leaf_sentence("L4+", "trained", "profile", n_m=14, k=3, x=0.2, y=0.3, z=0.4)
    assert "error 0.20 vs 0.30, 0.40" in s
    s = T.leaf_sentence("L8a", "unmodified", "probe vs use", 20, 12, rho=0.71)
    assert "rho = 0.71" in s and "20 families, 12 models" in s
    assert "ICC = 0.55" in T.leaf_sentence("L-meas", icc=0.55)
    for leaf in T.TEMPLATES:
        assert T.leaf_sentence(leaf.split("-")[0] if leaf.endswith(("smooth", "release")) else leaf, "trained", "probe",
                               n_f=1, n_m=2, other="x")
    banned = ("prove", "show", "demonstrate", "confirm")  # strengthening verbs; none appear in templates
    assert not any(b in " ".join(T.TEMPLATES.values()).lower() for b in ("prove", "demonstrat", "confirm"))


# ------------------------------------------------------------------------------------------------ loader
def _write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def test_rows_from_results(tmp_path):
    base = dict(n_blocks=24, family="fam00", type_label="arithmetic", test_item_acc=0.7,
                valid=dict(probe=True, twin=False, use=True, arrival=True),
                onset=dict(probe=dict(rel=0.05, layers=1.2, width=0.2, peak=0.9, peak_block=3), twin=None,
                           use=dict(rel=0.8, layers=19.2, width=0.05), arrival=None))
    _write(tmp_path / "g1/plain/Qwen2.5-0.5B/fam00_a.json", dict(base, task="fam00_a"))
    _write(tmp_path / "g1/twoshot/Qwen2.5-0.5B/fam00_a.json", dict(base, task="fam00_a"))
    half = dict(base, task="fam00_b", onset_half=dict(probe=[0.05, 0.07], use=[dict(rel=0.7), dict(rel=0.8)]))
    _write(tmp_path / "g2/trained/s1/pythia-1b/fam00_b.json", half)
    bad = dict(base, task="x", valid=dict(probe=False, twin=False, use=False),
               onset=dict(probe=None, twin=None, use=None), probe=dict(selectivity=[0, 0.1, 0.3, 0.2, 0.1]))
    _write(tmp_path / "g1/plain/pythia-1b/x.json", bad)
    (tmp_path / "g1/plain/pythia-1b/broken.json").write_text("{not json")
    rows = T.rows_from_results(tmp_path, models_json="/nonexistent")
    assert len(rows) == 4
    by = {(r["kind"], r["prompt"], r["model"], r["task"]): r for r in rows}
    a = by[("unmodified", "plain", "Qwen2.5-0.5B", "fam00_a")]
    assert a["lineage"] == "Qwen2.5" and a["n_blocks"] == 24 and a["early_read"] is True and a["seed"] == 0
    assert a["onset_half"] == {"probe": None, "twin": None, "use": None} and a["trained_acc"] is None
    assert ("unmodified", "twoshot", "Qwen2.5-0.5B", "fam00_a") in by
    t = by[("trained", "plain", "pythia-1b", "fam00_b")]
    assert t["seed"] == 1 and t["onset_half"]["use"] == [0.7, 0.8] and t["trained_acc"] == 0.7
    x = by[("unmodified", "plain", "pythia-1b", "x")]
    assert x["onset"]["probe"] is None and x["onset_dropped"]["probe"] is not None and not x["valid"]["probe"]


def test_model_meta_reads_release_date(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"Qwen/Qwen2.5-0.5B": {"hub": {"created_at": "2024-09-15T12:15:39+00:00"}}}))
    assert T.model_meta(p)["Qwen2.5-0.5B"]["release_date"] == "2024-09-15"
    assert T.model_meta("/nonexistent")["Qwen2.5-0.5B"]["release_date"] is None


def test_run_tree_gate1_smoke():
    rows = make_rows(stages3, noise=0.02, half_noise=0.003, use_fn=lambda mi, fi, ti, rel: rel + 0.01)
    rows += make_rows(stages3, noise=0.02, half_noise=0.003, prompt="twoshot", use_fn=lambda mi, fi, ti, rel: rel + 0.01)
    out = T.run_tree(rows, gate=1, csim=dict(n2=dict(power_d3_worst_k=0.9), n4=dict(fpr_max=0.0),
                                             n8=dict(fpr_max=0.0), n5=dict(power=0.9)), n_null=30, n_boot=20)
    r = out["readings"]["unmodified"]
    assert r["N1"]["passed"] and r["N2"]["verdict"] == "discrete" and r["N3"]["leaf"] in ("L3", "L3m", "L3f", "L3w")
    assert r["N8"]["leaf"] in ("L8a", "L8g", "L8d", "L8w", "L8x")
    assert out["form"]["form"].startswith("paper")
    json.dumps(T.to_jsonable(out))


def test_run_tree_gate2_and_script(tmp_path, capsys):
    import sys
    sys.path.insert(0, "scripts")
    import atlas_tree_run as run

    f = lambda mi, fi, ti: stages3(mi, fi, ti)
    kw = dict(noise=0.02, half_noise=0.003, use_fn=lambda mi, fi, ti, rel: rel + 0.01,
              twin_fn=lambda mi, fi, ti, rel: rel)
    rows = make_rows(f, **kw) + make_rows(f, prompt="twoshot", **kw)
    rows += make_rows(f, kind="trained", acc_fn=lambda mi, fi, ti: 0.6, rng_seed=5, **kw)
    # write the rows back out as the runner's result files
    for r in rows:
        d = dict(task=r["task"], family=r["family"], type_label=r["type_label"], n_blocks=r["n_blocks"], valid=r["valid"],
                 onset=r["onset"], onset_half=r["onset_half"], test_item_acc=r["trained_acc"] or 0.7)
        sub = f"g1/{r['prompt']}" if r["kind"] == "unmodified" else f"g2/trained/s{r['seed']}"
        p = tmp_path / sub / r["model"] / f"{r['task']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d))
    csim = tmp_path / "csim" / "csim.json"
    csim.parent.mkdir()
    csim.write_text(json.dumps(dict(n2=dict(power_d3_worst_k=0.9, d_star_worst_text="3.0"), n4=dict(fpr_max=0.0),
                                    n8=dict(fpr_max=0.0), n5=dict(power=0.9))))
    out = tmp_path / "tree" / "g2.json"
    rc = run.main(["read", "--gate", "2", "--root", str(tmp_path), "--out", str(out), "--n-null", "20", "--n-boot", "15",
                   "--models-json", "/nonexistent"])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["readings"]["trained"]["N2"]["verdict"] == "discrete" and res["readings"]["trained"]["N4"]["leaf"]
    assert res["form"]["form"]
    txt = capsys.readouterr().out
    assert "form:" in txt and "[trained] N2" in txt
