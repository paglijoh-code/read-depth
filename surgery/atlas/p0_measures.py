"""P0 pilot measures (proposal 4.3): pure numpy/sklearn, no models. States are [n, B+1, d] (block 0 = embedding)."""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
from scipy.stats import binom
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CHANCE = 0.25


def _fit(X, y, C=1.0):
    sc = StandardScaler().fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lr = LogisticRegression(C=C, max_iter=500).fit(sc.transform(X), y)
    return sc, lr


def _acc(sc, lr, X, y):
    return float(np.mean(lr.predict(sc.transform(X)) == y))


def _one_block(Xtr, y_tr, groups_tr, Xte, y_te, perms, cv_folds):
    sc, lr = _fit(Xtr, y_tr)
    acc = _acc(sc, lr, Xte, y_te)
    fit = (sc.scale_.copy(), lr.coef_.copy())
    folds = []
    for a, v in GroupKFold(cv_folds).split(Xtr, y_tr, groups_tr):
        s2, l2 = _fit(Xtr[a], y_tr[a])
        folds.append(_acc(s2, l2, Xtr[v], y_tr[v]))
    null = []
    for p in perms:
        s3, l3 = _fit(Xtr, y_tr[p])
        null.append(_acc(s3, l3, Xte, y_te))
    return acc, fit, float(np.mean(folds)), null


def _run_blocks(S_tr, y_tr, groups_tr, S_te, y_te, perms, cv_folds, nb):
    """One task per block. ATLAS_PROBE_JOBS > 1 spreads blocks over processes (for CPU-quota pods); each fit is
    unchanged, so results match the serial path. Default 1 = serial."""
    jobs = int(os.environ.get("ATLAS_PROBE_JOBS", "1"))
    args = [(S_tr[:, b], y_tr, groups_tr, S_te[:, b], y_te, perms, cv_folds) for b in range(nb)]
    if jobs <= 1:
        return [_one_block(*a) for a in args]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=jobs)(delayed(_one_block)(*a) for a in args)


def probe_curve(S_tr, y_tr, groups_tr, S_te, y_te, n_perm=20, seed=0, cv_folds=5, bow_acc=None, alpha=0.01):
    """Probe at each block (proposal 4.3 items 1 and 6). Control accuracy = the larger of the mean permuted-label test
    accuracy (`n_perm` fits per block) and the bag-of-words rival `bow_acc`; selectivity = (acc - control)/(1 - chance).
    Valid if peak selectivity >= 0.20 and the one-sided binomial test of the peak block's accuracy against its control
    is below `alpha` after Bonferroni over the blocks. Curves are lists; `fits` = [(scale, coef)] per block."""
    t_start = time.time()
    y_tr, y_te = np.asarray(y_tr), np.asarray(y_te)
    nb, n_te = S_tr.shape[1], len(y_te)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(len(y_tr)) for _ in range(n_perm)]
    blocks = _run_blocks(S_tr, y_tr, groups_tr, S_te, y_te, perms, cv_folds, nb)
    acc = [r[0] for r in blocks]
    fits = [r[1] for r in blocks]
    cv = [r[2] for r in blocks]
    null = np.array([r[3] for r in blocks]).T.reshape(len(perms), nb)
    acc = np.array(acc)
    perm_mean = null.mean(0) if len(perms) else np.full(nb, CHANCE)
    ctrl = np.maximum(perm_mean, bow_acc) if bow_acc is not None else perm_mean
    sel = (acc - ctrl) / (1 - CHANCE)
    peak_b = int(np.argmax(sel))
    p_perm = float((1 + np.sum(null.max(1) >= acc.max())) / (1 + len(perms))) if len(perms) else None
    k = int(round(acc[peak_b] * n_te))
    p0 = float(np.clip(ctrl[peak_b], CHANCE, 1 - 1e-9))
    p_binom = float(binom.sf(k - 1, n_te, p0))
    p_bonf = float(min(1.0, nb * p_binom))
    peak_sel = float(sel[peak_b])
    return {
        "acc": acc.tolist(), "cv_acc": cv, "control_acc": ctrl.tolist(), "perm_mean": perm_mean.tolist(),
        "bow_acc": None if bow_acc is None else float(bow_acc), "selectivity": sel.tolist(),
        "peak_block": peak_b, "peak_acc_block": int(np.argmax(acc)), "peak_selectivity": peak_sel,
        "p_perm": p_perm, "p_binom": p_binom, "p_binom_bonf": p_bonf, "test": "binomial_bonferroni",
        "n_perm": len(perms), "n_fits": nb * (1 + cv_folds + len(perms)), "fit_s": time.time() - t_start,
        "valid": bool(peak_sel >= 0.20 and p_bonf < alpha), "fits": fits,
    }


def probe_subspace(scale, coef, k=3):
    """Class directions in the raw basis (coef / scale), centred over classes; top-k left singular vectors, [d, k]."""
    raw = coef / scale
    raw = raw - raw.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(raw, full_matrices=False)
    return vt[:k].T.copy()


def random_subspaces(d, n_blocks, n=10, k=3, seed=0):
    """[n_blocks+1, n, d, k] orthonormal random subspaces, seeded per block (same for all items)."""
    out = np.zeros((n_blocks + 1, n, d, k), dtype=np.float32)
    for L in range(n_blocks + 1):
        rng = np.random.default_rng(seed * 10007 + L)
        for r in range(n):
            q, _ = np.linalg.qr(rng.standard_normal((d, k)))
            out[L, r] = q
    return out


def _cos_dist(a, b):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return 1.0 - num / den


def twin_divergence(S_item, S_ans, S_surf, n_draw=1000, seed=0):
    """t(L) = (mean d_ans - mean d_surf)/(mean d_ans + mean d_surf + 1e-6); states centred per block on the item mean."""
    mu = S_item.mean(0, keepdims=True)
    I, A, F = S_item - mu, S_ans - mu, S_surf - mu
    da, ds = _cos_dist(I, A), _cos_dist(I, F)  # [n, B+1]
    da[np.isnan(da)] = 0.0
    ds[np.isnan(ds)] = 0.0

    def t(x, y):
        return (x.mean(0) - y.mean(0)) / (x.mean(0) + y.mean(0) + 1e-6)

    curve = t(da, ds)
    rng = np.random.default_rng(seed)
    flip = rng.random((n_draw, da.shape[0])) < 0.5
    ma = np.where(flip[:, :, None], ds[None], da[None]).mean(1)
    ms = np.where(flip[:, :, None], da[None], ds[None]).mean(1)
    null_max = ((ma - ms) / (ma + ms + 1e-6)).max(1)
    p = float((1 + np.sum(null_max >= curve.max())) / (1 + n_draw))
    peak = float(curve.max())
    return {"t": curve.tolist(), "d_ans": da.mean(0).tolist(), "d_surf": ds.mean(0).tolist(),
            "peak": peak, "peak_block": int(np.argmax(curve)), "p_perm": p,
            "valid": bool(peak >= 0.05 and p < 0.01 and peak > 0),
            # C-check repair (proposed, not v11): the peak must exceed the deepest dip, so a curve with the twins
            # swapped (exactly -t) can never pass when the original does
            "trough": float(curve.min()),
            "valid_sym": bool(peak >= 0.05 and p < 0.01 and peak > 0 and peak > abs(float(curve.min())))}


def pair_mask(logit_item, logit_twin, ans_item, ans_twin, min_gap=1.0):
    """Pairs: model answers both correctly (argmax over 4 options) and |D(twin) - D(item)| >= min_gap.
    D = logit(twin answer) - logit(item answer); logit_* are [n, 4]."""
    n = len(ans_item)
    ar = np.arange(n)
    ok = (logit_item.argmax(1) == ans_item) & (logit_twin.argmax(1) == ans_twin) & (ans_item != ans_twin)
    d_item = logit_item[ar, ans_twin] - logit_item[ar, ans_item]
    d_twin = logit_twin[ar, ans_twin] - logit_twin[ar, ans_item]
    return ok & (np.abs(d_twin - d_item) >= min_gap)


def delta(logits, ans_item, ans_twin):
    """logits [..., n, 4] -> logit(twin answer) - logit(item answer), [..., n]."""
    return np.take_along_axis(logits, np.asarray(ans_twin)[..., None], -1)[..., 0] - \
        np.take_along_axis(logits, np.asarray(ans_item)[..., None], -1)[..., 0]


def use_scores(d_item, d_twin, d_probe, d_rand, d_surf, n_flip=10000, seed=0, min_pairs=40):
    """d_item, d_twin: [P]; d_probe, d_surf: [P, K] (blocks 1..K); d_rand: [P, R, K].
    Returns curves over blocks 0..K (block 0 = 0) plus the sign-flip test and validity."""
    P = len(d_item)
    den = float(np.mean(d_twin - d_item))
    rec_p = np.mean(d_probe - d_item[:, None], 0) / den
    rec_r = np.mean(d_rand - d_item[:, None, None], (0, 1)) / den
    use = rec_p - rec_r
    surf = np.mean(d_surf - d_item[:, None], 0) / den
    c = ((d_probe - d_item[:, None]) - np.mean(d_rand - d_item[:, None, None], 1)) / den  # [P, K]
    obs = float(c.mean(0).max()) if P else 0.0
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_flip)
    for i in range(0, n_flip, 2000):
        m = min(2000, n_flip - i)
        s = rng.choice([-1.0, 1.0], size=(m, P))
        null_max[i:i + m] = (s @ c / P).max(1)
    p = float((1 + np.sum(null_max >= obs)) / (1 + n_flip))
    z = np.zeros(1)
    peak = float(use.max())
    valid = bool(P >= min_pairs and peak >= 0.10 and p < 0.01 and np.abs(surf).max() < 0.05)
    return {"n_pairs": int(P), "use": np.concatenate([z, use]).tolist(), "probe_recovery": np.concatenate([z, rec_p]).tolist(),
            "random_recovery": np.concatenate([z, rec_r]).tolist(), "surface_recovery": np.concatenate([z, surf]).tolist(),
            "peak": peak, "peak_block": int(np.argmax(use)) + 1, "p_signflip": p, "valid": valid}


def arrival_scores(d_item, d_twin, d_whole, d_wsurf, B, min_pairs=1):
    """Whole-state swap, same pairs and score. Curves over blocks 0..K, block 0 = 0."""
    den = float(np.mean(d_twin - d_item))
    arr = np.mean(d_whole - d_item[:, None], 0) / den
    surf = np.mean(d_wsurf - d_item[:, None], 0) / den
    z = np.zeros(1)
    curve = np.concatenate([z, arr])
    on = onset(curve, B)
    valid = bool(len(d_item) >= min_pairs and np.abs(surf).max() < 0.05 and on is not None
                 and on["rel"] <= (B - 2) / B + 1e-12)
    return {"arrival": curve.tolist(), "surface_recovery": np.concatenate([z, surf]).tolist(), "valid": valid}


def onset(curve, B):
    """First point the curve reaches half its peak (baseline 0 at block 0), linear interpolation; also rise width
    (first 90% crossing minus first 10% crossing, relative depth). None if the peak is not positive."""
    c = np.asarray(curve, dtype=float)
    peak = c.max()
    if not peak > 0:
        return None

    def cross(frac):
        thr = frac * peak
        for i in range(len(c)):
            if c[i] >= thr:
                if i == 0 or c[i] == c[i - 1]:
                    return float(i)
                return (i - 1) + (thr - c[i - 1]) / (c[i] - c[i - 1])
        return None

    x = cross(0.5)
    return {"layers": x, "rel": x / B, "width": (cross(0.9) - cross(0.1)) / B,
            "peak": float(peak), "peak_block": int(np.argmax(c))}


def logit_lens_acc(lens_logits, y):
    """lens_logits [n, B+1, 4] -> accuracy per block."""
    return (lens_logits.argmax(-1) == np.asarray(y)[:, None]).mean(0).tolist()
