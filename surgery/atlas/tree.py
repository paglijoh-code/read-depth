"""The depth-atlas tree (docs/PROPOSAL-depth-stages.md v13, sections 4.3, 6, 7): N1-N5, N8, the form rules (R2) and
the leaf sentences. Pure numpy/scipy, no models. Input is a plain list of dict rows (see `rows_from_results`).

Row: {model, lineage, n_params, n_blocks, release_date, task, family, type_label, kind ('unmodified'|'trained'),
prompt ('plain'|'twoshot'), seed, valid{probe,twin,use}, onset{probe,twin,use: {rel,layers,width}|None},
onset_half{probe,twin,use: [a,b]|None}, onset_dropped{method: rel|None}, early_read, trained_acc, seed}.

Interpretations the spec leaves open (all exposed as parameters or recorded in each reader's output):
- ICC form: ICC(2,1), two-way random effects, single measure, absolute agreement (Shrout and Fleiss).
- "Early-read" per cell: probe onset rel <= 0.10 or layers <= 1 (block 1). N1 drops early-read tasks before
  forming family values; N2 keeps them (its trivial verdict needs them).
- Variance floor 0.05 is on the variance of the standardised onset. Separation (0.5 SD) is in SD of the task-level
  values being clustered (`sep_unit='data'`; 'unit' uses 1.0).
- Every reader returns a dict; keys starting with "_" hold bulky arrays and are dropped by `to_jsonable`.
"""
from __future__ import annotations

import itertools
import json
import math
import warnings
from pathlib import Path

import numpy as np
from scipy import optimize, stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

for _m in ("Mean of empty slice", "All-NaN slice", "Degrees of freedom", "invalid value encountered"):
    warnings.filterwarnings("ignore", message=_m)

# ----------------------------------------------------------------------------------------------- model table
# short name -> (lineage, blocks, approximate total parameters). Lineage and blocks: proposal 4.1. Parameters:
# model-card totals, used only for ordering ("3 largest") and the size regression; var/ccheck/models.json has none.
MODEL_META = {
    "Qwen3-0.6B-Base": ("Qwen3", 28, 0.75e9), "Qwen3-1.7B-Base": ("Qwen3", 28, 2.03e9),
    "Qwen2.5-0.5B": ("Qwen2.5", 24, 0.49e9), "Qwen2.5-1.5B": ("Qwen2.5", 28, 1.54e9),
    "Qwen2.5-3B": ("Qwen2.5", 36, 3.09e9),
    "SmolLM2-135M": ("SmolLM2", 30, 0.135e9), "SmolLM2-360M": ("SmolLM2", 32, 0.36e9),
    "SmolLM2-1.7B": ("SmolLM2", 24, 1.71e9),
    "pythia-410m": ("Pythia", 24, 0.405e9), "pythia-1b": ("Pythia", 16, 1.01e9),
    "pythia-1.4b": ("Pythia", 24, 1.41e9), "pythia-2.8b": ("Pythia", 32, 2.78e9),
    "OLMo-2-0425-1B": ("OLMo", 16, 1.48e9), "OLMo-1B-0724-hf": ("OLMo", 16, 1.18e9),
    "TinyLlama_v1.1": ("TinyLlama", 22, 1.10e9),
    "pythia-160m": ("screening", 12, 0.16e9), "SmolLM-135M": ("screening", 30, 0.135e9),
    "SmolLM-1.7B": ("screening", 24, 1.71e9),
}
FITTED = [m for m, v in MODEL_META.items() if v[0] not in ("TinyLlama", "screening")]
NOT_FITTED = {m for m in MODEL_META if m not in FITTED}
HELDOUT = "TinyLlama_v1.1"
KINDS = ("unmodified", "trained")
METHODS = ("probe", "twin", "use")

EARLY_REL = 0.10


def model_meta(models_json="var/ccheck/models.json"):
    """short name -> dict(lineage, n_blocks, n_params, release_date). Release date = Hub `created_at` (date part)
    from var/ccheck/models.json when present; parameters from there only if a numeric `n_params` exists."""
    meta = {m: dict(lineage=v[0], n_blocks=v[1], n_params=v[2], release_date=None) for m, v in MODEL_META.items()}
    try:
        d = json.loads(Path(models_json).read_text())
    except Exception:
        return meta
    for mid, v in d.items():
        short = mid.split("/")[-1]
        if short not in meta or not isinstance(v, dict):
            continue
        ca = (v.get("hub") or {}).get("created_at")
        if ca:
            meta[short]["release_date"] = str(ca)[:10]
        if isinstance(v.get("n_params"), (int, float)):
            meta[short]["n_params"] = float(v["n_params"])
        elif isinstance((v.get("config") or {}).get("num_hidden_layers"), int):
            meta[short]["n_blocks"] = meta[short]["n_blocks"] or v["config"]["num_hidden_layers"]
    return meta


# ----------------------------------------------------------------------------------------------- loader
def onset_curve(curve, B):
    """Same rule as p0_measures.onset: first point reaching half the peak, linear interpolation. None if peak <= 0."""
    c = np.asarray(curve, dtype=float)
    if c.size == 0 or not c.max() > 0:
        return None
    peak = c.max()

    def cross(frac):
        thr = frac * peak
        for i in range(len(c)):
            if c[i] >= thr:
                if i == 0 or c[i] == c[i - 1]:
                    return float(i)
                return (i - 1) + (thr - c[i - 1]) / (c[i] - c[i - 1])
        return None

    x = cross(0.5)
    return {"layers": x, "rel": x / B, "width": (cross(0.9) - cross(0.1)) / B}


def _on(o):
    if not isinstance(o, dict) or o.get("rel") is None:
        return None
    return {"rel": float(o["rel"]), "layers": None if o.get("layers") is None else float(o["layers"]),
            "width": None if o.get("width") is None else float(o["width"])}


def _half(v):
    """[a, b] from a list of numbers or {rel:..} dicts; None if absent or incomplete."""
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return None
    out = []
    for x in v:
        x = x.get("rel") if isinstance(x, dict) else x
        if x is None:
            return None
        out.append(float(x))
    return out


def _row(j, model, kind, prompt, seed, meta):
    mm = meta.get(model, {})
    onset = {m: _on((j.get("onset") or {}).get(m)) for m in METHODS}
    valid = {m: bool((j.get("valid") or {}).get(m)) for m in METHODS}
    oh = j.get("onset_half") or j.get("half") or {}
    B = j.get("n_blocks") or mm.get("n_blocks")
    dropped = {}
    curves = {"probe": ("probe", "selectivity"), "twin": ("twin", "t"), "use": ("use", "use")}
    for m, (a, b) in curves.items():
        dropped[m] = None
        if not valid[m] and B and isinstance(j.get(a), dict) and j[a].get(b) is not None:
            oc = onset_curve(j[a][b], B)
            dropped[m] = None if oc is None else oc["rel"]
    p = onset["probe"]
    early = j.get("early_read")
    if early is None:
        early = bool(p is not None and (p["rel"] <= EARLY_REL or (p["layers"] is not None and p["layers"] <= 1.0)))
    acc = j.get("trained_acc")
    if acc is None and kind == "trained":
        acc = j.get("test_item_acc")
    return dict(model=model, lineage=mm.get("lineage"), n_params=mm.get("n_params"), n_blocks=B,
                release_date=mm.get("release_date"), task=j.get("task"), family=j.get("family"),
                type_label=j.get("type_label"), kind=kind, prompt=prompt, seed=seed, valid=valid, onset=onset,
                onset_half={m: _half(oh.get(m)) for m in METHODS}, onset_dropped=dropped,
                early_read=bool(early), trained_acc=None if acc is None else float(acc))


def rows_from_results(root, models_json="var/ccheck/models.json"):
    """Rows from <root>/g1/<prompt>/<model>/<task>.json (kind unmodified) and
    <root>/g2/trained/s<seed>/<model>/<task>.json (kind trained, plain prompt). Tolerant: unreadable files are skipped,
    absent half onsets stay None."""
    root = Path(root)
    meta = model_meta(models_json)
    rows = []

    def add(path, model, kind, prompt, seed):
        try:
            j = json.loads(path.read_text())
        except Exception:
            return
        if isinstance(j, dict) and isinstance(j.get("valid"), dict):
            j.setdefault("task", path.stem)
            rows.append(_row(j, model, kind, prompt, seed, meta))

    for p in sorted(root.glob("g1/*/*/*.json")):
        add(p, p.parent.name, "unmodified", p.parent.parent.name, 0)
    for p in sorted(root.glob("g2/trained/s*/*/*.json")):
        sd = p.parent.parent.name[1:]
        add(p, p.parent.name, "trained", "plain", int(sd) if sd.isdigit() else 0)
    return rows


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ----------------------------------------------------------------------------------------------- small stats
def icc_2_1(X):
    """ICC(2,1): two-way random effects, single measure, absolute agreement, for an n-targets x k-raters matrix
    (rows with a NaN are dropped). Returns nan if fewer than 3 complete rows."""
    X = np.asarray(X, dtype=float)
    X = X[~np.isnan(X).any(1)]
    n, k = X.shape if X.ndim == 2 else (0, 0)
    if n < 3 or k < 2:
        return float("nan")
    g = X.mean()
    ssr = k * ((X.mean(1) - g) ** 2).sum()
    ssc = n * ((X.mean(0) - g) ** 2).sum()
    sse = ((X - g) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else float("nan")


def ari(a, b):
    """Adjusted Rand index of two label vectors."""
    a, b = np.unique(np.asarray(a), return_inverse=True)[1], np.unique(np.asarray(b), return_inverse=True)[1]
    n = len(a)
    if n < 2:
        return 1.0
    ct = np.bincount(a * (b.max() + 1) + b, minlength=(a.max() + 1) * (b.max() + 1)).astype(float)
    c2 = lambda x: x * (x - 1) / 2.0
    s_ij, s_a, s_b = c2(ct).sum(), c2(np.bincount(a).astype(float)).sum(), c2(np.bincount(b).astype(float)).sum()
    exp = s_a * s_b / c2(n)
    mx = 0.5 * (s_a + s_b)
    return 1.0 if mx == exp else float((s_ij - exp) / (mx - exp))


def _rank_corr(a, b, min_n=3):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n:
        return float("nan")
    ra, rb = stats.rankdata(a[ok]), stats.rankdata(b[ok])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def signflip_p(d, n_draw=20000, seed=0, exact_max=20):
    """One-sided paired sign-flip permutation p for mean(d) > 0 (d = baseline error - estimator error). Exact
    enumeration of the 2^n flips when n <= exact_max (14 models -> 16,384), else n_draw random flips."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    obs = d.mean()
    if n <= exact_max:
        bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        m = (bits @ d) / n
    else:
        m = (np.random.default_rng(seed).choice([-1.0, 1.0], (n_draw, n)) @ d) / n
    return float((m >= obs - 1e-12).mean())


def _boot_ci(vals, lo=2.5, hi=97.5):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


# ----------------------------------------------------------------------------------------------- 1-D Gaussian EM
def gmm_fit_batch(X, k, n_starts=20, floor=0.05, max_iter=100, tol=1e-6, rng=None, max_elems=1_500_000):
    """Vectorised 1-D Gaussian mixture EM over a batch of datasets X [R, n] (a 1-D X is one dataset). Each dataset
    gets `n_starts` random starts (means = random data points, equal weights, variance = data variance); the best start
    by log-likelihood is returned. Variances are floored at `floor` (variance units). Returns dict of arrays:
    ll [R], mu/var/w [R, k]."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    R, n = X.shape
    rng = np.random.default_rng(0) if rng is None else rng
    if k == 1:
        mu = X.mean(1)
        v = np.maximum(X.var(1), floor)
        ll = -0.5 * n * np.log(2 * np.pi * v) - 0.5 * ((X - mu[:, None]) ** 2).sum(1) / v
        return dict(ll=ll, mu=mu[:, None], var=v[:, None], w=np.ones((R, 1)))
    out = {key: [] for key in ("ll", "mu", "var", "w")}
    chunk = max(1, int(max_elems // (n_starts * n * k)))
    for s0 in range(0, R, chunk):
        Xc = X[s0:s0 + chunk]
        r_ = len(Xc)
        idx = rng.integers(0, n, (r_, n_starts, k))
        mu = np.take_along_axis(Xc[:, None, :], idx, 2).astype(float)
        mu = mu + rng.normal(0, 1e-3, mu.shape) * (Xc.std(1)[:, None, None] + 1e-9)
        var = np.maximum(np.repeat(Xc.var(1)[:, None, None], k, 2) * np.ones((r_, n_starts, k)), floor)
        w = np.full((r_, n_starts, k), 1.0 / k)
        x = Xc[:, None, :, None]
        prev = None
        for it in range(max_iter + 1):
            d2 = (x - mu[:, :, None, :]) ** 2
            lp = np.log(w)[:, :, None, :] - 0.5 * np.log(2 * np.pi * var)[:, :, None, :] - 0.5 * d2 / var[:, :, None, :]
            mx = lp.max(-1, keepdims=True)
            e = np.exp(lp - mx)
            sm = e.sum(-1, keepdims=True)
            ll = (mx + np.log(sm))[..., 0].sum(-1)
            if it == max_iter or (prev is not None and np.max(ll - prev) < tol):
                break
            prev = ll
            r = e / sm
            nk = np.maximum(r.sum(2), 1e-8)
            w = nk / n
            mu = (r * x).sum(2) / nk
            var = np.maximum((r * (x - mu[:, :, None, :]) ** 2).sum(2) / nk, floor)
        b = ll.argmax(1)
        ar = np.arange(r_)
        out["ll"].append(ll[ar, b])
        out["mu"].append(mu[ar, b])
        out["var"].append(var[ar, b])
        out["w"].append(w[ar, b])
    return {key: np.concatenate(v) for key, v in out.items()}


def gmm_logpdf(x, mu, var, w):
    """log mixture density of points x [n] under one fitted mixture (mu, var, w each [k])."""
    x = np.asarray(x, dtype=float)[:, None]
    lp = np.log(w)[None] - 0.5 * np.log(2 * np.pi * var)[None] - 0.5 * (x - mu[None]) ** 2 / var[None]
    mx = lp.max(1, keepdims=True)
    return (mx + np.log(np.exp(lp - mx).sum(1, keepdims=True)))[:, 0]


def gmm_labels(x, mu, var, w):
    x = np.asarray(x, dtype=float)[:, None]
    lp = np.log(w)[None] - 0.5 * np.log(2 * np.pi * var)[None] - 0.5 * (x - mu[None]) ** 2 / var[None]
    return lp.argmax(1)


def bic_select(X, ks=(1, 2, 3), n_starts=20, floor=0.05, rng=None, max_iter=100):
    """Fit every k in ks to each dataset in X [R, n]; BIC = -2 ll + (3k - 1) ln n. Returns dict: bic [R, len(ks)],
    kstar [R], dbic [R] (= BIC(k=1) - min BIC, >= 0), fits {k: gmm_fit_batch output}."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[1]
    rng = np.random.default_rng(0) if rng is None else rng
    fits = {k: gmm_fit_batch(X, k, n_starts, floor, max_iter=max_iter, rng=rng) for k in ks}
    bic = np.stack([-2 * fits[k]["ll"] + (3 * k - 1) * math.log(n) for k in ks], 1)
    j = bic.argmin(1)
    b1 = bic[:, list(ks).index(1)] if 1 in ks else bic[:, 0]
    return dict(bic=bic, kstar=np.asarray(ks)[j], dbic=b1 - bic.min(1), fits=fits)


# ----------------------------------------------------------------------------------------------- log-concave MLE
def _J(a, b):
    """Integral over [0,1] of exp((1-u)a + u b), with partials, stable for b ~ a."""
    d = b - a
    ea = np.exp(a)
    small = np.abs(d) < 1e-3
    ds = np.where(small, 1.0, d)
    eb = np.exp(b)
    J = np.where(small, ea * (1 + d / 2 + d ** 2 / 6 + d ** 3 / 24), (eb - ea) / ds)
    Jb = np.where(small, ea * (0.5 + d / 3 + d ** 2 / 8), eb / ds - (eb - ea) / ds ** 2)
    Ja = np.where(small, ea * (0.5 + d / 6 + d ** 2 / 24), -ea / ds + (eb - ea) / ds ** 2)
    return J, Ja, Jb


class LogConcave:
    """Dumbgen-Rufibach log-concave MLE of a 1-D sample: a piecewise-linear concave log-density with knots at the
    distinct sorted data points, support = [min, max]. Solved as a convex problem (L-BFGS-B on
    phi = P theta, theta = (phi_1, slope_1, decrements >= 0) with an analytic gradient).
    `smoothed=True` is the variance-preserving version (Chen and Samworth): the MLE convolved with
    N(0, s^2 - var(f_hat)), s^2 the sample variance, so the sampler's variance equals the sample variance.
    Fewer than 3 distinct points: falls back to a normal with the sample moments."""

    def __init__(self, x, smoothed=True):
        x = np.asarray(x, dtype=float)
        self.smoothed = smoothed
        self.sample_var = float(x.var())
        self.xs, cnt = np.unique(x, return_counts=True)
        self.normal = len(self.xs) < 3
        if self.normal:
            self.mean_hat, self.var_hat = float(x.mean()), self.sample_var
            self.sigma_s = 0.0
            return
        w = cnt / cnt.sum()
        xs = self.xs
        m = len(xs)
        dx = np.diff(xs)
        P = np.zeros((m, m))
        P[:, 0] = 1.0
        # slopes s_j = s1 - sum_{i=2..j} delta_i  (j = 1..m-1); phi_{j+1} = phi_j + dx_j s_j
        S = np.zeros((m - 1, m - 1))
        S[:, 0] = 1.0
        for j in range(1, m - 1):
            S[j:, j] = -1.0  # theta index j+1 is delta_{j+1}; affects slopes j.. onward (0-based slope index j..)
        C = np.tril(np.ones((m - 1, m - 1))) * dx[None, :]  # phi_{i+1} - phi_1 = sum_{j<=i} dx_j s_j
        P[1:, 1:] = C @ S
        self._P, self._dx, self._w = P, dx, w
        mu0, v0 = float((w * xs).sum()), self.sample_var
        phi0 = -0.5 * (xs - mu0) ** 2 / v0
        s0 = np.diff(phi0) / dx
        th0 = np.concatenate([[phi0[0], s0[0]], np.maximum(s0[:-1] - s0[1:], 0)])
        bounds = [(None, None), (None, None)] + [(0, None)] * (m - 2)

        def f(th):
            phi = P @ th
            J, Ja, Jb = _J(phi[:-1], phi[1:])
            val = -(w * phi).sum() + (dx * J).sum()
            g = -w.copy()
            g[:-1] += dx * Ja
            g[1:] += dx * Jb
            return val, P.T @ g

        res = optimize.minimize(f, th0, jac=True, method="L-BFGS-B", bounds=bounds,
                                options=dict(maxiter=500, ftol=1e-12, gtol=1e-8))
        phi = P @ res.x
        J, _, _ = _J(phi[:-1], phi[1:])
        Z = (dx * J).sum()
        self.phi = phi - math.log(Z)
        self.seg_mass = dx * J / Z
        self.slope = np.diff(self.phi) / dx
        # moments by Gauss-Legendre per segment
        gx, gw = np.polynomial.legendre.leggauss(24)
        t = 0.5 * (gx + 1)[None, :] * dx[:, None] + xs[:-1, None]
        wt = 0.5 * gw[None, :] * dx[:, None]
        dens = np.exp(self.phi[:-1, None] + self.slope[:, None] * (t - xs[:-1, None]))
        self.mean_hat = float((wt * dens * t).sum())
        self.var_hat = float((wt * dens * (t - self.mean_hat) ** 2).sum())
        self.sigma_s = math.sqrt(max(self.sample_var - self.var_hat, 0.0)) if smoothed else 0.0

    def logpdf_hat(self, x):
        """log of the (unsmoothed) MLE density; -inf outside the data range."""
        x = np.asarray(x, dtype=float)
        if self.normal:
            return stats.norm.logpdf(x, self.mean_hat, math.sqrt(max(self.var_hat, 1e-12)))
        j = np.clip(np.searchsorted(self.xs, x, side="right") - 1, 0, len(self.xs) - 2)
        lp = self.phi[j] + self.slope[j] * (x - self.xs[j])
        return np.where((x >= self.xs[0]) & (x <= self.xs[-1]), lp, -np.inf)

    def pdf(self, x, n_grid=4001):
        """Density of the (smoothed) distribution, by numerical convolution on a grid for the smoothed version."""
        x = np.asarray(x, dtype=float)
        if self.normal:
            return stats.norm.pdf(x, self.mean_hat, math.sqrt(max(self.sample_var, 1e-12)))
        if self.sigma_s <= 0:
            return np.exp(self.logpdf_hat(x))
        g = np.linspace(self.xs[0], self.xs[-1], n_grid)
        f = np.exp(self.logpdf_hat(g))
        dg = g[1] - g[0]
        return (f[None, :] * stats.norm.pdf(x[:, None], g[None, :], self.sigma_s)).sum(1) * dg

    def logpdf(self, x):
        return np.log(np.maximum(self.pdf(x), 1e-300))

    def sample(self, size, rng):
        size = tuple(np.atleast_1d(size).tolist()) if not isinstance(size, int) else (size,)
        n = int(np.prod(size))
        if self.normal:
            return rng.normal(self.mean_hat, math.sqrt(max(self.sample_var, 1e-12)), size)
        seg = rng.choice(len(self.seg_mass), n, p=self.seg_mass / self.seg_mass.sum())
        s, dx = self.slope[seg], self.dx_seg(seg)
        U = rng.random(n)
        sd = s * dx
        flat = np.abs(sd) < 1e-9
        u = np.where(flat, U * dx, np.log1p(U * np.expm1(np.where(flat, 1.0, np.clip(sd, -700, 700)))) /
                     np.where(flat, 1.0, s))
        out = self.xs[seg] + np.clip(u, 0, dx)
        if self.sigma_s > 0:
            out = out + rng.normal(0, self.sigma_s, n)
        return out.reshape(size)

    def dx_seg(self, seg):
        return self._dx[seg]


# ----------------------------------------------------------------------------------------------- table
class Tab:
    """Arrays for one (kind, prompt, seed) slice of the rows: models x tasks."""

    def __init__(self, rows, kind, prompt="plain", seed=0, models=None):
        sel = [r for r in rows if r["kind"] == kind and r.get("prompt", "plain") == prompt and r.get("seed", 0) == seed]
        if models is None:
            present = {r["model"] for r in sel} - NOT_FITTED
            models = sorted(present, key=lambda m: (FITTED.index(m) if m in FITTED else len(FITTED), m))
        self.kind, self.prompt, self.seed = kind, prompt, seed
        self.models = list(models)
        sel = [r for r in sel if r["model"] in self.models]
        self.tasks = sorted({r["task"] for r in sel})
        mi = {m: i for i, m in enumerate(self.models)}
        ti = {t: i for i, t in enumerate(self.tasks)}
        M, T = len(self.models), len(self.tasks)
        self.M, self.T = M, T
        first = {}
        self.lineage = [None] * M
        self.n_blocks = np.full(M, np.nan)
        self.n_params = np.full(M, np.nan)
        self.release = [None] * M
        nan = lambda *s: np.full(s, np.nan)
        self.ons = {m: nan(M, T) for m in METHODS}
        self.lay = {m: nan(M, T) for m in METHODS}
        self.wid = {m: nan(M, T) for m in METHODS}
        self.drop = {m: nan(M, T) for m in METHODS}
        self.half = {m: nan(M, T, 2) for m in METHODS}
        self.valid = {m: np.zeros((M, T), bool) for m in METHODS}
        self.early_cell = np.zeros((M, T), bool)
        self.acc = nan(M, T)
        for r in sel:
            i, j = mi[r["model"]], ti[r["task"]]
            first.setdefault(r["task"], r)
            self.lineage[i] = r.get("lineage")
            self.n_blocks[i] = np.nan if r.get("n_blocks") is None else r["n_blocks"]
            self.n_params[i] = np.nan if r.get("n_params") is None else r["n_params"]
            self.release[i] = r.get("release_date")
            for m in METHODS:
                v = bool(r["valid"].get(m)) and r["onset"].get(m) is not None
                self.valid[m][i, j] = v
                if v:
                    o = r["onset"][m]
                    self.ons[m][i, j] = o["rel"]
                    self.lay[m][i, j] = np.nan if o.get("layers") is None else o["layers"]
                    self.wid[m][i, j] = np.nan if o.get("width") is None else o["width"]
                dr = (r.get("onset_dropped") or {}).get(m)
                if dr is not None and not v:
                    self.drop[m][i, j] = dr
                h = (r.get("onset_half") or {}).get(m)
                if h is not None and v:
                    self.half[m][i, j] = h
            self.early_cell[i, j] = bool(r.get("early_read")) and self.valid["probe"][i, j]
            if r.get("trained_acc") is not None:
                self.acc[i, j] = r["trained_acc"]
        self.family = [first[t].get("family") for t in self.tasks]
        self.type_label = [first[t].get("type_label") for t in self.tasks]
        self.families = sorted(set(self.family))
        fi = {f: i for i, f in enumerate(self.families)}
        self.task_fam = np.array([fi[f] for f in self.family], int)
        self.fam_type = [None] * len(self.families)
        for t in range(T):
            self.fam_type[self.task_fam[t]] = self.type_label[t]
        nv = self.valid["probe"].sum(0)
        self.early_task = (self.early_cell.sum(0) >= 0.5 * np.maximum(nv, 1)) & (nv > 0)

    def models_by_size(self):
        order = np.argsort(-np.nan_to_num(self.n_params, nan=-1), kind="stable")
        return [int(i) for i in order]


def family_values(tab, method="probe", drop_early=False, field="ons", min_models=9):
    """Family value = mean over a family's valid tasks (dropping early-read tasks if asked), per model.
    Returns (F [M, nfam] with NaN where undefined, counts [nfam] = models with a value, keep [nfam] = counts >= min)."""
    V = getattr(tab, field)[method].copy()
    if drop_early:
        V[:, tab.early_task] = np.nan
    nf = len(tab.families)
    F = np.full((tab.M, nf), np.nan)
    for f in range(nf):
        c = V[:, tab.task_fam == f]
        n = np.isfinite(c).sum(1)
        s = np.where(np.isfinite(c), c, 0).sum(1)
        F[:, f] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    counts = np.isfinite(F).sum(0)
    return F, counts, counts >= min_models


def family_counts(tab, method="probe", min_models=9):
    F, counts, keep = family_values(tab, method, min_models=min_models)
    return dict(counts=counts.tolist(), families=list(tab.families), n_counting=int(keep.sum()),
                counting=[f for f, k in zip(tab.families, keep) if k])


def icc_halves(tab, method="probe"):
    """ICC(2,1) of family onset between the two item halves, over (model, family) cells (mean of a family's valid
    tasks that have both halves). nan if no half data."""
    H = tab.half[method]
    cells = []
    for f in range(len(tab.families)):
        c = H[:, tab.task_fam == f, :]  # [M, t, 2]
        ok = np.isfinite(c).all(2)
        n = ok.sum(1)
        s = np.where(ok[..., None], c, 0).sum(1)
        cells.append(np.where(n[:, None] > 0, s / np.maximum(n, 1)[:, None], np.nan))
    X = np.concatenate(cells, 0) if cells else np.zeros((0, 2))
    return icc_2_1(X)


def icc_seeds(tab0, tab1, method="probe"):
    """Trained kind, seed 0 vs seed 1, task-level cells valid in both."""
    common = [m for m in tab0.models if m in tab1.models]
    ct = [t for t in tab0.tasks if t in tab1.tasks]
    if not common or not ct:
        return float("nan")
    a = np.array([[tab0.ons[method][tab0.models.index(m), tab0.tasks.index(t)] for t in ct] for m in common])
    b = np.array([[tab1.ons[method][tab1.models.index(m), tab1.tasks.index(t)] for t in ct] for m in common])
    return icc_2_1(np.stack([a.ravel(), b.ravel()], 1))


def censoring_check(tab, method="probe", n_large=3, gap=0.15, max_invalid=3):
    """Item 12. On the n_large largest models: mean onset of the curves the signal filter dropped (computed from the
    curve ignoring the filter; absent -> None) minus the mean of the kept ones; and the families with no valid task on
    any of them. Under-sampled if gap >= 0.15 or more than 3 families invalid on all."""
    big = tab.models_by_size()[:n_large]
    ons, drp, val = tab.ons[method][big], tab.drop[method][big], tab.valid[method][big]
    kept = float(np.nanmean(ons)) if np.isfinite(ons).any() else None
    dropped = float(np.nanmean(drp)) if np.isfinite(drp).any() else None
    n_inv = 0
    for f in range(len(tab.families)):
        if not val[:, tab.task_fam == f].any():
            n_inv += 1
    g = None if kept is None or dropped is None else dropped - kept
    flag = bool((g is not None and g >= gap) or n_inv > max_invalid)
    return dict(node="censoring", method=method, kind=tab.kind, models=[tab.models[i] for i in big],
                kept_onset=kept, dropped_onset=dropped, gap=g, n_families_invalid_on_all=n_inv,
                bar=dict(gap=gap, max_invalid_families=max_invalid), dropped_onsets_available=dropped is not None,
                undersampled=flag)


def early_read_flags(tab):
    return {t: bool(e) for t, e in zip(tab.tasks, tab.early_task)}


# ----------------------------------------------------------------------------------------------- leaf sentences
KIND_WORDS = {"unmodified": "in the unmodified models", "trained": "after training"}

TEMPLATES = {
    "L0": "Could not be tested: too few model pairs shared enough families readable after the earliest layers ({kind}).",
    "L1": "Across {n_m} small base models, the depth order of {n_f} synthetic task families {kind} is not shared "
          "(median rho = {rho}).",
    "L1b": "Agreement in family depth order {kind} lies between {a} and {b}, with reserves added; not established.",
    "L-meas": "Read depth {kind} was not reliable between item halves after smoothing and fresh items (ICC = {icc}); "
              "no stage result is reported.",
    "L2s": "The only stage structure {kind} separates tasks read in the earliest layers from the rest.",
    "L2x": "Tasks {kind} form {k} stable groups in read depth, but the groups are closer than the preset separation; "
           "not reported as stages.",
    "L2c": "Read depth {kind} varies on a smooth scale for {n_f} families across {n_m} models; stages closer than "
           "d* = {d_star} are not excluded.",
    "L2t": "Discrete stages appear after training; in the unmodified models, depth {other}.",
    "L2u": "Discrete stages appear in the unmodified models; after training, depth {other}.",
    "L2w": "Evidence for stages {kind} is weak, with reserves added; not established.",
    "L3": "In this designed set of {n_f} families, read depth {kind} falls into {k} stages shared by {n_m} models "
          "from five lineages; the stages hold when a lineage is left out and when families are left out.",
    "L3m": "In this designed set of {n_f} families, read depth {kind} falls into {k} stages shared by {n_m} models "
           "from five lineages; the stages hold when a lineage is left out, but do not hold for held-out families.",
    "L3f": "{k} depth stages fit the pooled models {kind}, but they do not hold when a lineage is left out.",
    "L3w": "Whether the stages {kind} hold on held-out lineages is not established.",
    "L4+": "The {n_m} models form {k} groups by where they read and use tasks {kind}; on held-out models and families, "
           "a model's group matches its profile more closely than its lineage or its size and depth do "
           "(error {x} vs {y}, {z}).",
    "L4m": "Models' depth profiles {kind} are matched best by their lineage; groups add nothing beyond it.",
    "L4g": "Models' depth profiles {kind} follow size and depth on a gradient; no grouping matches held-out profiles "
           "better.",
    "L4g-smooth": "Models' depth profiles {kind} are matched best by their nearest models on other families; the "
                  "similarity is smooth, not grouped, and not explained by size and depth.",
    "L4m-release": "Models' depth profiles {kind} are matched best by their lineage; groups add nothing beyond it "
                   "(release date, not lineage).",
    "L4u": "Neither lineage, size and depth, nor grouping matches held-out profiles better than the others {kind}.",
    "L4x": "Too few families were readable in every model to compare profiles {kind}.",
    "L5r": "Read depth {kind} carries across models as a fraction of depth.",
    "L5a": "Read depth {kind} carries across models as an absolute layer.",
    "L5p": "Read depth {kind} carries across models partly (beta = {beta}).",
    "L5n": "Read depth {kind} fits neither a fraction of depth nor an absolute layer better (beta = {beta}).",
    "L8a": "Where a probe reads a task and where swapping only the probe's direction changes the answer agree {kind} "
           "(rho = {rho}, gap within +-0.05).",
    "L8g": "Probe depth and the depth at which the probe's direction changes the answer agree in order "
           "(rho = {rho}) but differ by {g} of depth {kind}.",
    "L8d": "Where a probe reads a task and where its direction changes the answer do not agree {kind} (rho = {rho}); "
           "probe depth alone is not a guide to where that direction is used.",
    "L8w": "Agreement of read and use depth {kind} is not established.",
    "L8x": "Agreement of read and use depth {kind} could not be tested (too few families or unreliable use onsets).",
}


class _Fmt(dict):
    def __missing__(self, k):
        return "?"


def _num(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float, np.floating)) and math.isfinite(float(v)) else ("?" if v is None else str(v))


def leaf_sentence(leaf, kind="unmodified", method="probe", n_f=None, n_m=None, censored=False, weak=False, **fields):
    """Section 7 template for `leaf`, filled. Always names the kind and method; appends the family and model counts
    when the template does not carry them, the censoring clause when `censored`, and "(weakly powered)" when `weak`.
    Unknown leaf names raise KeyError."""
    key = leaf
    if leaf == "L4g" and fields.get("variant") == "smooth":
        key = "L4g-smooth"
    if leaf == "L4m" and fields.get("variant") == "release":
        key = "L4m-release"
    tpl = TEMPLATES[key]
    kw = _Fmt(kind=f"{KIND_WORDS.get(kind, kind)} by {method}", n_f=n_f if n_f is not None else "?",
              n_m=n_m if n_m is not None else "?")
    for k, v in fields.items():
        kw[k] = "?" if v is None else _num(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v
    s = tpl.format_map(kw)
    if "{n_f}" not in tpl and "{n_m}" not in tpl and (n_f is not None or n_m is not None):
        s += f" ({'?' if n_f is None else n_f} families, {'?' if n_m is None else n_m} models.)"
    if censored:
        s += " The late end is under-sampled."
    if weak:
        s += " (weakly powered)"
    return s


# ----------------------------------------------------------------------------------------------- N1
def read_n1(tab, method="probe", min_common=15, min_pairs=45, n_boot=1000, seed=0, reserves_available=False,
            min_models=9, hold=(0.5, 0.25), fail=(0.3, 0.5), censored=False, weak=False):
    """N1: median pairwise Spearman of family onset over fitted-model pairs with >= min_common common non-early
    families, family-bootstrap 95% CI. HOLDS: median >= 0.5 and lower >= 0.25. FAILS: median < 0.3 and upper < 0.5.
    Between -> reserves (if available) else L1b. Fewer than min_pairs pairs -> L0."""
    F, counts, keep = family_values(tab, method, drop_early=True, min_models=min_models)
    F = F[:, keep]
    nf = F.shape[1]
    pairs = list(itertools.combinations(range(tab.M), 2))
    fin = np.isfinite(F)
    ok_pairs = [(i, j) for i, j in pairs if (fin[i] & fin[j]).sum() >= min_common]
    res = dict(node="N1", kind=tab.kind, method=method, n_f=int(nf), n_m=int(tab.M), n_pairs=len(ok_pairs),
               n_pairs_possible=len(pairs), bar=dict(min_common=min_common, min_pairs=min_pairs, holds=dict(median=hold[0],
               lower=hold[1]), fails=dict(median=fail[0], upper=fail[1])), median=None, ci=None, verdict=None,
               leaf=None)
    if len(ok_pairs) < min_pairs:
        res.update(verdict="L0", leaf="L0", reserves_pending=False)
    else:
        rho = np.array([_rank_corr(F[i], F[j]) for i, j in ok_pairs])
        med = float(np.nanmedian(rho))
        rng = np.random.default_rng(seed)
        bs = []
        for _ in range(n_boot):
            idx = rng.integers(0, nf, nf)
            Fb, fb = F[:, idx], fin[:, idx]
            r = [_rank_corr(Fb[i], Fb[j]) for i, j in ok_pairs if (fb[i] & fb[j]).sum() >= min_common]
            bs.append(np.nanmedian(r) if len(r) else np.nan)
        lo, hi = _boot_ci(bs)
        res.update(median=med, ci=[lo, hi], pair_rho=rho.tolist())
        if med >= hold[0] and lo >= hold[1]:
            res.update(verdict="HOLDS", reserves_pending=False)
        elif med < fail[0] and hi < fail[1]:
            res.update(verdict="FAILS", leaf="L1", reserves_pending=False)
        else:
            res.update(verdict="between", reserves_pending=bool(reserves_available))
            if not reserves_available:
                res["leaf"] = "L1b"
    if res["leaf"]:
        ci = res["ci"] or [None, None]
        res["sentence"] = leaf_sentence(res["leaf"], tab.kind, method, res["n_f"], res["n_m"], censored, weak,
                                        rho=res["median"], a=ci[0], b=ci[1])
    res["passed"] = res["verdict"] == "HOLDS"
    return res


# ----------------------------------------------------------------------------------------------- N2
def task_level(tab, method="probe", min_models=9, models=None):
    """Task unit: onset standardised within each model across its valid tasks, averaged over the models where the
    task is valid (>= min_models). Returns (values [T] with NaN for dropped tasks, n_models_per_task)."""
    O = tab.ons[method]
    sel = np.arange(tab.M) if models is None else np.asarray(models, int)
    O = O[sel]
    mu = np.nanmean(O, 1, keepdims=True)
    sd = np.nanstd(O, 1, ddof=1, keepdims=True)
    Z = (O - mu) / np.where(sd > 0, sd, 1.0)
    n = np.isfinite(Z).sum(0)
    v = np.where(n > 0, np.where(np.isfinite(Z), Z, 0).sum(0) / np.maximum(n, 1), np.nan)
    return np.where(n >= min_models, v, np.nan), n


def _fam_layout(fam_of_point):
    """Families (in order of first appearance) and the point indices of each."""
    fams = list(dict.fromkeys(fam_of_point.tolist()))
    return fams, [np.flatnonzero(fam_of_point == f) for f in fams]


def pair_null_datasets(x, fam_of_point, n_draw, rng, smoothed=True):
    """N2's null draws with the pair structure: family centres from the smoothed log-concave null fitted to the family
    means, plus each family's within-family deviations resampled from the data (a two-task family takes a random data
    family's deviation pair, with a random swap; one-task families get no deviation). Returns X [n_draw, n] in the
    data's point order, and the fitted LogConcave."""
    x = np.asarray(x, float)
    fams, idxs = _fam_layout(np.asarray(fam_of_point))
    means = np.array([x[i].mean() for i in idxs])
    lc = LogConcave(means, smoothed=smoothed)
    centres = lc.sample((n_draw, len(fams)), rng)
    pool = np.array([x[i] - x[i].mean() for i in idxs if len(i) == 2])  # [P, 2], second = -first
    X = np.zeros((n_draw, len(x)))
    for f, i in enumerate(idxs):
        if len(i) == 2 and len(pool):
            e = pool[rng.integers(0, len(pool), n_draw), 0] * rng.choice([-1.0, 1.0], n_draw)
            X[:, i[0]] = centres[:, f] + e
            X[:, i[1]] = centres[:, f] - e
        else:
            X[:, i] = centres[:, [f]]
    return X, lc


def n2_test(x, fam_of_point, n_null=1000, n_starts=20, floor=0.05, seed=0, ks=(1, 2, 3), max_iter=100):
    """The N2 fit and null test on task-level values x: BIC-selected k*, observed DBIC(k*, 1) against the 95th
    percentile of null draws, each refitted the same way. Returns dict incl. the mixture fit and labels."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    sel = bic_select(x, ks, n_starts, floor, rng, max_iter)
    kstar, dbic = int(sel["kstar"][0]), float(sel["dbic"][0])
    fit = sel["fits"][kstar]
    mu, var, w = fit["mu"][0], fit["var"][0], fit["w"][0]
    labels = gmm_labels(x, mu, var, w)
    out = dict(kstar=kstar, dbic=dbic, bic=sel["bic"][0].tolist(), mu=mu, var=var, w=w, labels=labels)
    if n_null > 0:
        Xn, lc = pair_null_datasets(x, fam_of_point, n_null, rng)
        nulls = bic_select(Xn, ks, n_starts, floor, rng, max_iter)["dbic"]
        q95 = float(np.percentile(nulls, 95))
        out.update(null_q95=q95, p_null=float((1 + (nulls >= dbic).sum()) / (1 + n_null)), null_beaten=bool(dbic > q95),
                   _null_dbic=nulls, _lc=lc)
    return out


def _boot_stability(x, fam_of_point, k, labels, n_boot, n_starts, floor, rng, max_iter=100):
    """Median ARI between the original labels (of the resampled points) and the refit's labels, over family
    bootstraps (families resampled with both their tasks)."""
    x = np.asarray(x, float)
    fams, idxs = _fam_layout(np.asarray(fam_of_point))
    nf = len(fams)
    draws = [np.concatenate([idxs[f] for f in rng.integers(0, nf, nf)]) for _ in range(n_boot)]
    out = np.full(n_boot, np.nan)
    by_n = {}
    for b, d in enumerate(draws):
        by_n.setdefault(len(d), []).append(b)
    for n, bl in by_n.items():
        Xb = np.stack([x[draws[b]] for b in bl])
        fit = gmm_fit_batch(Xb, k, n_starts, floor, rng=rng, max_iter=max_iter)
        for r, b in enumerate(bl):
            lab = gmm_labels(Xb[r], fit["mu"][r], fit["var"][r], fit["w"][r])
            out[b] = ari(labels[draws[b]], lab)
    return float(np.nanmedian(out)), out


def n2_decide(xs, fam, early, n_null=1000, n_boot=1000, n_starts=20, floor=0.05, seed=0, stab_min=0.70, sep_min=0.5,
              sep_unit="data", reserves_available=False):
    """N2's fit, null test, stability and separation on task-level values, and the first-match verdict
    (trivial -> L2s, discrete -> N3, close -> L2x, continuous -> L2c, between -> L2w). Also used by C-sim."""
    t = n2_test(xs, fam, n_null, n_starts, floor, seed)
    rng = np.random.default_rng(seed + 1)
    k = t["kstar"]
    if k >= 2 and n_boot > 0:
        stab, _ = _boot_stability(xs, fam, k, t["labels"], n_boot, n_starts, floor, rng)
    else:
        stab = float("nan")
    lab = t["labels"]
    meds = np.array([np.median(xs[lab == c]) for c in np.unique(lab)])
    sd = float(xs.std(ddof=1)) if sep_unit == "data" else 1.0
    gaps = np.diff(np.sort(meds))
    sep_ok = bool(len(gaps) > 0 and gaps.min() >= sep_min * sd)
    only_early = any(early[lab == c].all() for c in np.unique(lab)) if k == 2 else False
    beaten = bool(t.get("null_beaten"))
    stab_ok = bool(np.isfinite(stab) and stab >= stab_min)
    if k == 2 and beaten and stab_ok and only_early:
        v, leaf = "trivial", "L2s"
    elif k in (2, 3) and beaten and stab_ok and sep_ok:
        v, leaf = "discrete", None
    elif k in (2, 3) and beaten and stab_ok:
        v, leaf = "close", "L2x"
    elif not beaten:
        v, leaf = "continuous", "L2c"
    else:
        v, leaf = "between", (None if reserves_available else "L2w")
    return dict(verdict=v, leaf=leaf, k=k, dbic=t["dbic"], bic=t["bic"], null_q95=t.get("null_q95"),
                p_null=t.get("p_null"), null_beaten=beaten, stability=stab, separation_ok=sep_ok,
                cluster_medians=sorted(meds.tolist()), sd=sd, only_early_cluster=bool(only_early),
                reserves_pending=bool(v == "between" and reserves_available), _fit=t)


def read_n2(tab, method="probe", min_models=9, n_null=1000, n_boot=1000, n_starts=20, floor=0.05, seed=0,
            icc=None, icc_fresh=None, icc_min=0.70, stab_min=0.70, sep_min=0.5, sep_unit="data",
            reserves_available=False, d_star=None, n_families_expected=None, censored=False, weak=False):
    """N2. `icc`: reliability between item halves (computed from the table's half onsets if None); `icc_fresh`: the
    ICC after the spline / fresh-item rung (optional). If the first ICC fails and the second is absent or fails
    -> L-meas (flag `ladder_run` records whether a second value was given). Then the fit, null, stability and
    separation, and the first-match verdict."""
    icc0 = icc_halves(tab, method) if icc is None else icc
    res = dict(node="N2", kind=tab.kind, method=method, icc=icc0, icc_fresh=icc_fresh, bar=dict(icc=icc_min,
               stability=stab_min, separation_sd=sep_min, null_percentile=95), leaf=None, verdict=None)
    icc_used = icc0
    ok = (icc0 is not None and math.isfinite(icc0) and icc0 >= icc_min)
    if not ok:
        res["ladder_run"] = icc_fresh is not None
        if icc_fresh is not None:
            icc_used = icc_fresh
            ok = math.isfinite(icc_fresh) and icc_fresh >= icc_min
    res["icc_used"] = icc_used
    x, nmod = task_level(tab, method, min_models)
    good = np.isfinite(x)
    res.update(n_tasks=int(good.sum()), n_f=int(len(set(tab.task_fam[good]))), n_m=int(tab.M))
    if not ok:
        res.update(verdict="L-meas", leaf="L-meas", passed=False)
    elif good.sum() < 8:
        res.update(verdict="L-meas", leaf="L-meas", passed=False, note="too few tasks")
    else:
        xs, fam = x[good], tab.task_fam[good]
        dec = n2_decide(xs, fam, tab.early_task[good], n_null, n_boot, n_starts, floor, seed, stab_min, sep_min,
                        sep_unit, reserves_available)
        res.update({k_: v_ for k_, v_ in dec.items() if k_ != "_fit"}, passed=True, _x=xs, _fam=fam, _fit=dec["_fit"],
                   _tasks=[tab.tasks[i] for i in np.flatnonzero(good)])
    res["discrete"] = res.get("verdict") == "discrete"
    if res["leaf"]:
        res["sentence"] = leaf_sentence(res["leaf"], tab.kind, method, res.get("n_f"), res.get("n_m"), censored, weak,
                                        icc=icc_used, k=res.get("k"), d_star=d_star)
    return res


def contrast(n2_by_kind):
    """If one kind is discrete and the other reached L2c or L2x -> L2t (trained discrete) or L2u (unmodified
    discrete). n2_by_kind: {kind: read_n2 result}."""
    d = {k: v for k, v in n2_by_kind.items() if v}
    for a, b, leaf in (("trained", "unmodified", "L2t"), ("unmodified", "trained", "L2u")):
        if a in d and b in d and d[a].get("verdict") == "discrete" and d[b].get("leaf") in ("L2c", "L2x"):
            other = "is smooth" if d[b]["leaf"] == "L2c" else "forms groups too close to call stages"
            return dict(node="N2-contrast", leaf=leaf, sentence=leaf_sentence(leaf, a, "probe", other=other))
    return None


# ----------------------------------------------------------------------------------------------- N3
def family_folds(families, types, n_folds=5, seed=0):
    """Seeded fold index per family, stratified by type (each type's shuffled families dealt round-robin)."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(families), int)
    off = 0
    for t in sorted(set(map(str, types))):
        ids = [i for i, tt in enumerate(types) if str(tt) == t]
        rng.shuffle(ids)
        for r, i in enumerate(ids):
            fold[i] = (r + off) % n_folds
        off += len(ids)
    return fold


def _fit_k(x, k, n_starts, floor, seed):
    f = gmm_fit_batch(x, k, n_starts, floor, rng=np.random.default_rng(seed))
    return f["mu"][0], f["var"][0], f["w"][0]


def read_n3(tab, n2, method="probe", min_models=9, n_null=1000, n_starts=20, floor=0.05, seed=0, n_perm=1000,
            tiny_tab=None, censored=False, weak=False):
    """N3 for a kind whose N2 verdict is discrete (needs n2['_x'] etc.). a: leave one lineage out (5 folds), b: leave
    families out (5 folds, stratified by type; mixture minus null held-out log-likelihood against null draws), c:
    TinyLlama clause. In 3b the null density is the smoothed log-concave fit to the training task values."""
    k = n2["k"]
    lineages = sorted(set(l for l in tab.lineage if l))
    # --- N3a
    aris = []
    for L in lineages:
        rest = [i for i in range(tab.M) if tab.lineage[i] != L]
        held = [i for i in range(tab.M) if tab.lineage[i] == L]
        if not rest or not held:
            continue
        xr, _ = task_level(tab, method, min_models, rest)
        xh, _ = task_level(tab, method, 1, held)
        ok = np.isfinite(xr)
        mu, var, w = _fit_k(xr[ok], k, n_starts, floor, seed)
        lab_fit = np.full(tab.T, -1)
        lab_fit[ok] = gmm_labels(xr[ok], mu, var, w)
        both = ok & np.isfinite(xh)
        if both.sum() < 3:
            aris.append(dict(lineage=L, ari=float("nan"), n_tasks=int(both.sum())))
            continue
        lab_h = gmm_labels(xh[both], mu, var, w)
        aris.append(dict(lineage=L, ari=ari(lab_fit[both], lab_h), n_tasks=int(both.sum())))
    av = np.array([a["ari"] for a in aris], float)
    med = float(np.nanmedian(av)) if np.isfinite(av).any() else float("nan")
    n_low = int(np.sum(av < 0.20))
    if med >= 0.50 and n_low <= 1:
        a_v = "HOLDS"
    elif med < 0.30 or not np.isfinite(med):
        a_v = "FAILS"
    else:
        a_v = "between"
    # --- N3b
    x, fam = n2["_x"], n2["_fam"]
    fams, idxs = _fam_layout(fam)
    ftypes = [tab.fam_type[f] for f in fams]
    folds = family_folds(fams, ftypes, 5, seed)
    point_fold = np.zeros(len(x), int)
    for f, i in enumerate(idxs):
        point_fold[i] = folds[f]

    def stat(X):  # X [R, n]; returns T per dataset
        T = np.zeros(len(X))
        for fo in range(5):
            tr, te = point_fold != fo, point_fold == fo
            if not te.any():
                continue
            fit = gmm_fit_batch(X[:, tr], k, n_starts, floor, rng=np.random.default_rng(seed + fo))
            for r in range(len(X)):
                lm = gmm_logpdf(X[r, te], fit["mu"][r], fit["var"][r], fit["w"][r]).sum()
                ln = LogConcave(X[r, tr]).logpdf(X[r, te]).sum()
                T[r] += lm - ln
        return T

    obs = float(stat(x[None])[0])
    rng = np.random.default_rng(seed + 7)
    Xn, _ = pair_null_datasets(x, fam, n_null, rng)
    nul = stat(Xn) if n_null > 0 else np.array([])
    p_b = float((1 + (nul >= obs).sum()) / (1 + len(nul)))
    b_v = "HOLDS" if p_b < 0.05 else "FAILS"
    # clause: type label vs stage
    lab = n2["_fit"]["labels"]
    types = np.array([tab.fam_type[f] for f in fam])

    def match(tp):
        acc = 0
        for t in np.unique(tp):
            acc += np.bincount(lab[tp == t], minlength=k).max()
        return acc / len(lab)

    m_obs = match(types)
    ft = np.array(ftypes)
    perm = []
    for _ in range(n_perm):
        sh = rng.permutation(len(ft))
        mp = {f: ft[sh[i]] for i, f in enumerate(fams)}
        perm.append(match(np.array([mp[f] for f in fam])))
    chance = float(np.mean(perm))
    p_t = float((1 + (np.array(perm) >= m_obs).sum()) / (1 + n_perm))
    type_clause = bool(m_obs >= chance + 0.20 and p_t < 0.05)
    # --- N3c
    c = None
    if tiny_tab is not None and tiny_tab.M:
        xt, _ = task_level(tiny_tab, method, 1)
        tmap = {t: i for i, t in enumerate(tiny_tab.tasks)}
        pairs = [(i, tmap[t]) for i, t in enumerate(n2["_tasks"]) if t in tmap and np.isfinite(xt[tmap[t]])]
        if len(pairs) >= 3:
            ix, it = zip(*pairs)
            lt = gmm_labels(xt[list(it)], n2["_fit"]["mu"], n2["_fit"]["var"], n2["_fit"]["w"])
            a_t = ari(lab[list(ix)], lt)
            c = dict(ari=a_t, holds=bool(a_t >= 0.5), n_tasks=len(pairs))
    leaf = {("HOLDS", "HOLDS"): "L3", ("HOLDS", "FAILS"): "L3m"}.get((a_v, b_v))
    if a_v == "FAILS":
        leaf = "L3f"
    elif a_v == "between":
        leaf = "L3w"
    res = dict(node="N3", kind=tab.kind, method=method, k=k, n_f=n2.get("n_f"), n_m=n2.get("n_m"), leaf=leaf,
               N3a=dict(verdict=a_v, median_ari=med, n_below_0_20=n_low, folds=aris,
                        bar=dict(median=0.5, max_low_folds=1, fails_below=0.3)),
               N3b=dict(verdict=b_v, p=p_b, statistic=obs, n_null=len(nul), bar=dict(p=0.05),
                        type_clause=dict(holds=type_clause, match=float(m_obs), chance=chance, p=p_t)),
               N3c=c)
    res["sentence"] = leaf_sentence(leaf, tab.kind, method, res["n_f"], res["n_m"], censored, weak, k=k)
    return res


# ----------------------------------------------------------------------------------------------- N4
def _zcols(X):
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 1e-12, sd, 1.0)


def silhouette(D, labels):
    """Mean silhouette from a square distance matrix (singleton clusters score 0)."""
    labels = np.asarray(labels)
    cl = np.unique(labels)
    if len(cl) < 2:
        return -1.0
    n = len(labels)
    s = np.zeros(n)
    mean_to = np.stack([D[:, labels == c].sum(1) / (labels == c).sum() for c in cl], 1)  # [n, ncl] incl. self
    for i in range(n):
        ci = int(np.flatnonzero(cl == labels[i])[0])
        ni = (labels == labels[i]).sum()
        if ni < 2:
            continue
        a = D[i, labels == labels[i]].sum() / (ni - 1)
        b = np.min(np.delete(mean_to[i], ci))
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def cluster_best(X, ks=(2, 3, 4)):
    """Average-linkage on Euclidean distance between rows, k by silhouette. Returns (k, silhouette, labels)."""
    n = len(X)
    d = pdist(X)
    Z = hierarchy.linkage(d, "average")
    D = squareform(d)
    best = (None, -2.0, None)
    for k in ks:
        if k >= n:
            continue
        lab = hierarchy.fcluster(Z, k, "maxclust") - 1
        s = silhouette(D, lab)
        if s > best[1]:
            best = (k, s, lab)
    return best


def build_profile(tabs, gate=2, min_core=12, models=None):
    """Profile matrix, models x (feature, family) columns, standardised per column. `tabs`: {'unmodified': Tab,
    'trained': Tab}. Gate 1: unmodified probe (onset, width) and twin onset only. Gate 2: trained kind: probe onset
    and width, twin onset and use onset (each only if its own core has >= min_core families), training shift (probe
    valid in both kinds in every model), and learnability (fraction of a family's tasks with trained accuracy >=
    0.45; all families). Core = families with a value in every model. Returns dict incl. `n_probe_core`."""
    main = tabs["unmodified"] if gate == 1 else tabs["trained"]
    M = main.M
    cols, data = [], []

    def core(F):
        return np.flatnonzero(np.isfinite(F).all(0))

    Fp, _, _ = family_values(main, "probe")
    cp = core(Fp)
    Fw, _, _ = family_values(main, "probe", field="wid")
    n_probe_core = len(cp)
    out = dict(models=list(main.models), lineage=list(main.lineage), n_probe_core=int(n_probe_core), cols=[],
               features_used=[], features_skipped=[], gate=gate, tab=main)
    if n_probe_core < min_core:
        out["X"] = None
        return out

    def add(name, F, idx):
        for f in idx:
            cols.append((name, int(f)))
            data.append(F[:, f])
        out["features_used"].append(name)

    add("probe_onset", Fp, cp)
    add("probe_width", Fw, cp)
    Ft, _, _ = family_values(main, "twin")
    ct = core(Ft)
    if len(ct) >= min_core:
        add("twin_onset", Ft, ct)
    else:
        out["features_skipped"].append("twin_onset")
    if gate == 2:
        Fu, _, _ = family_values(main, "use")
        cu = core(Fu)
        if len(cu) >= min_core:
            add("use_onset", Fu, cu)
        else:
            out["features_skipped"].append("use_onset")
        un = tabs.get("unmodified")
        if un is not None and un.models == main.models:
            acc = np.zeros((M, len(main.families)))
            cnt = np.zeros((M, len(main.families)))
            for t in main.tasks:
                if t not in un.tasks:
                    continue
                a, b = main.tasks.index(t), un.tasks.index(t)
                d = main.ons["probe"][:, a] - un.ons["probe"][:, b]
                ok = np.isfinite(d)
                acc[ok, main.task_fam[a]] += d[ok]
                cnt[ok, main.task_fam[a]] += 1
            sh = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
            cs = core(sh)
            if len(cs):
                add("training_shift", sh, cs)
            else:
                out["features_skipped"].append("training_shift")
        else:
            out["features_skipped"].append("training_shift")
        # learnability: all families, always defined
        L = np.full((M, len(main.families)), np.nan)
        for f in range(len(main.families)):
            a = main.acc[:, main.task_fam == f]
            L[:, f] = np.where(np.isfinite(a).any(1), np.nanmean(np.where(np.isfinite(a), a >= 0.45, np.nan), 1), np.nan)
        cl = np.flatnonzero(np.isfinite(L).all(0))
        if len(cl):
            add("learnability", L, cl)
        else:
            out["features_skipped"].append("learnability")
    Xraw = np.stack(data, 1)
    out.update(X=_zcols(Xraw), cols=cols, col_fam=np.array([c[1] for c in cols]),
               col_feature=[c[0] for c in cols])
    return out


def _cluster_null_stat(X, rng, n_null, ks):
    """One-factor continuum null: scores on PC1 x loadings + independent Gaussian residuals with each column's
    residual variance; each draw re-standardised and clustered the same way. Returns the null's best silhouettes."""
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    t, v = U[:, 0] * S[0], Vt[0]
    fit = np.outer(t, v)
    sd = (X - X.mean(0) - fit).std(0)
    out = np.empty(n_null)
    for b in range(n_null):
        Xn = fit + rng.normal(0, 1, X.shape) * sd
        out[b] = cluster_best(_zcols(Xn), ks)[1]
    return out


def n4a(prof, n_null=1000, n_boot=1000, seed=0, ks=(2, 3, 4), stab_min=0.60):
    X = prof["X"]
    rng = np.random.default_rng(seed)
    k, sil, lab = cluster_best(X, ks)
    nul = _cluster_null_stat(X, rng, n_null, ks) if n_null > 0 else np.array([-2.0])
    q95 = float(np.percentile(nul, 95))
    fams = np.unique(prof["col_fam"])
    st = []
    for _ in range(n_boot):
        pick = rng.choice(fams, len(fams))
        ci = np.concatenate([np.flatnonzero(prof["col_fam"] == f) for f in pick])
        Xb = _zcols(X[:, ci])
        lb = hierarchy.fcluster(hierarchy.linkage(pdist(Xb), "average"), k, "maxclust") - 1
        st.append(ari(lab, lb))
    stab = float(np.median(st)) if st else float("nan")
    return dict(k=int(k), silhouette=float(sil), null_q95=q95, p_null=float((1 + (nul >= sil).sum()) / (1 + len(nul))),
                stability=stab, groups=bool(sil > q95 and np.isfinite(stab) and stab >= stab_min), labels=lab,
                bar=dict(null_percentile=95, stability=stab_min))


def _nearest(Xo, x, k):
    return np.argsort(np.sqrt(((Xo - x) ** 2).sum(1)), kind="stable")[:k]


def release_ordinal(s):
    if not s:
        return None
    try:
        y, m, d = (int(v) for v in str(s)[:10].split("-"))
        return float(y * 372 + m * 31 + d)
    except Exception:
        return None


ESTIMATORS = ("o", "i", "ii", "iii", "iv", "v")


def n4b(prof, k_groups, n_folds=5, seed=0, n_near=3, alpha=0.05, pairwise_test=False, hide_fold=None):
    """N4b: for each model m and family fold f, hide m's entries on f's families and estimate them by (o) the mean of
    the other models, (i) m's lineage partners, (ii) regression on log parameters and blocks (other models), (iii) the
    mean of m's cluster (clusters refit on the other models, m assigned to the nearest centre by its other folds), (iv)
    the 3 nearest models by its other folds, (v) the 3 nearest in release date. Per-model MAE over hidden entries;
    paired sign-flip permutation over models. An estimator wins if it beats (o) at p < alpha and has the lower mean MAE
    than every other estimator in play (`pairwise_test=True` also demands p < alpha against each of them; off by
    default because (i) and (iii) are near-identical when the clusters are the lineages, so no test could separate
    them, and exit 3 needs (iii) to be able to win there). All pairwise p-values are reported. Columns are the standardised
    profile columns. `hide_fold` (testing): array col->fold overriding the family folds."""
    X, tab = prof["X"], prof["tab"]
    M, P = X.shape
    lineage = np.array(prof["lineage"])
    fam_ids = np.unique(prof["col_fam"])
    ftype = [tab.fam_type[f] for f in fam_ids]
    ffold = dict(zip(fam_ids.tolist(), family_folds(fam_ids, ftype, n_folds, seed).tolist()))
    colfold = np.array([ffold[f] for f in prof["col_fam"]]) if hide_fold is None else np.asarray(hide_fold)
    npar, nbl = tab.n_params, tab.n_blocks
    reg_ok = bool(np.isfinite(npar).all() and np.isfinite(nbl).all())
    rel = np.array([np.nan if release_ordinal(r) is None else release_ordinal(r) for r in tab.release])
    rel_ok = bool(np.isfinite(rel).all())
    Z = np.column_stack([np.ones(M), np.log(np.where(np.isfinite(npar), npar, 1.0)), np.where(np.isfinite(nbl), nbl, 0.0)])
    est_in_play = [e for e in ESTIMATORS if not (e == "ii" and not reg_ok) and not (e == "v" and not rel_ok)]
    err = {e: np.zeros(M) for e in est_in_play}
    cnt = np.zeros(M)
    for m in range(M):
        others = np.array([j for j in range(M) if j != m])
        partners = [j for j in others if lineage[j] == lineage[m]]
        kk = min(k_groups, len(others) - 1)
        cl = hierarchy.fcluster(hierarchy.linkage(pdist(X[others]), "average"), kk, "maxclust") - 1
        for f in np.unique(colfold):
            H = np.flatnonzero(colfold == f)
            O = np.flatnonzero(colfold != f)
            if len(H) == 0 or len(O) == 0:
                continue
            truth = X[m, H]
            pred = {"o": X[others][:, H].mean(0)}
            if partners:
                pred["i"] = X[partners][:, H].mean(0)
            else:
                pred["i"] = pred["o"]
            if "ii" in est_in_play:
                coef = np.linalg.lstsq(Z[others], X[others][:, H], rcond=None)[0]
                pred["ii"] = Z[m] @ coef
            cents = np.stack([X[others][cl == c][:, O].mean(0) for c in np.unique(cl)])
            cm = int(np.argmin(((cents - X[m, O]) ** 2).sum(1)))
            members = others[cl == np.unique(cl)[cm]]
            pred["iii"] = X[members][:, H].mean(0)
            nn = others[_nearest(X[others][:, O], X[m, O], n_near)]
            pred["iv"] = X[nn][:, H].mean(0)
            if "v" in est_in_play:
                nv = others[np.argsort(np.abs(rel[others] - rel[m]), kind="stable")[:n_near]]
                pred["v"] = X[nv][:, H].mean(0)
            for e in est_in_play:
                err[e][m] += np.abs(pred[e] - truth).sum()
            cnt[m] += len(H)
    mae = {e: err[e] / np.maximum(cnt, 1) for e in est_in_play}
    p = {}
    for a in est_in_play:
        for b in est_in_play:
            if a != b:
                p[(a, b)] = signflip_p(mae[b] - mae[a], seed=seed)  # a better than b
    wins = []
    for a in est_in_play:
        if a == "o":
            continue
        beats_o = p[(a, "o")] < alpha
        if pairwise_test:
            beats_all = all(p[(a, b)] < alpha for b in est_in_play if b not in (a, "o"))
        else:
            beats_all = all(mae[a].mean() < mae[b].mean() for b in est_in_play if b not in (a, "o"))
        if beats_o and beats_all:
            wins.append(a)
    return dict(mae_mean={e: float(mae[e].mean()) for e in est_in_play}, mae_per_model={e: mae[e].tolist() for e in est_in_play},
                p={f"{a}>{b}": v for (a, b), v in p.items()}, in_play=est_in_play, wins=wins,
                winner=wins[0] if len(wins) == 1 else None, n_models=M, bar=dict(p=alpha))


def n4_exit(groups, splits_lineage, winner):
    """N4b exits, first match wins: 1 groups and (iii) wins and a lineage is split -> L4+; 2 (ii) -> L4g; 3 (i), or
    groups and (iii) with every lineage whole -> L4m; 4 (iv) -> L4g (smooth clause); 4a (v) -> L4m (release clause);
    5 otherwise L4u. Returns (leaf, variant)."""
    w = winner
    if groups and w == "iii" and splits_lineage:
        return "L4+", None
    if w == "ii":
        return "L4g", None
    if w == "i" or (groups and w == "iii"):
        return "L4m", None
    if w == "iv":
        return "L4g", "smooth"
    if w == "v":
        return "L4m", "release"
    return "L4u", None


def read_n4(tabs, gate=2, n_null=1000, n_boot=1000, seed=0, min_core=12, min_models=None, censored=False, weak=False,
            profile=None, k_groups=None):
    """N4. gate 1: unmodified profile (probe + twin); gate 2: the trained-kind profile (the verdict). < min_core probe
    core families -> L4x."""
    prof = profile or build_profile(tabs, gate, min_core)
    kind = prof["tab"].kind
    res = dict(node="N4", kind=kind, method="profile", gate=gate, n_m=len(prof["models"]),
               n_probe_core=prof["n_probe_core"], features_used=prof["features_used"],
               features_skipped=prof["features_skipped"], bar=dict(min_core=min_core), leaf=None)
    if prof["X"] is None:
        res.update(verdict="L4x", leaf="L4x", n_f=prof["n_probe_core"])
        res["sentence"] = leaf_sentence("L4x", kind, "profile", res["n_f"], res["n_m"], censored, weak)
        return res
    res["n_f"] = int(len(np.unique(prof["col_fam"])))
    a = n4a(prof, n_null, n_boot, seed)
    labels = a.pop("labels")
    k = a["k"] if k_groups is None else k_groups
    b = n4b(prof, k, seed=seed)
    lin = np.array(prof["lineage"])
    split = bool(any(len(set(labels[lin == l])) > 1 for l in set(lin)))
    w, groups = b["winner"], a["groups"]
    leaf, clause = n4_exit(groups, split, w)
    mae = b["mae_mean"]
    res.update(N4a=a, N4b=b, groups_exist=groups, splits_lineage=split, labels=labels.tolist(), leaf=leaf,
               variant=clause, verdict=leaf, _profile=prof)
    res["sentence"] = leaf_sentence(leaf, kind, "profile", res["n_f"], res["n_m"], censored, weak, k=a["k"],
                                    x=mae.get("iii"), y=mae.get("i"), z=mae.get("ii"), variant=clause)
    return res


# ----------------------------------------------------------------------------------------------- N5
def _slope_w(task, x, y, w):
    """Weighted within-task slope of y on x (task effects removed)."""
    sw = np.bincount(task, w)
    mx = np.bincount(task, w * x) / np.maximum(sw, 1e-12)
    my = np.bincount(task, w * y) / np.maximum(sw, 1e-12)
    dx, dy = x - mx[task], y - my[task]
    den = (w * dx * dx).sum()
    return float((w * dx * dy).sum() / den) if den > 0 else float("nan")


def n5_points(tab, method="probe", models=None):
    L = tab.lay[method]
    ti, mi = np.where(np.isfinite(L.T) & (L.T >= 1.0))
    keep = np.ones(len(mi), bool) if models is None else np.isin(mi, models)
    ti, mi = ti[keep], mi[keep]
    return ti, mi, np.log(L[mi, ti]), np.log(tab.n_blocks[mi])


def n5_fit(ti, mi, y, xb, n_task, n_model, n_boot=1000, seed=0):
    """beta of log layer-onset = task effect + beta log blocks and its two-way (task x model) bootstrap 95% CI
    (tasks and models each resampled with replacement; a resampled unit enters as an integer weight)."""
    _, tcode = np.unique(ti, return_inverse=True)
    beta = _slope_w(tcode, xb, y, np.ones(len(y)))
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        ct = np.bincount(rng.integers(0, n_task, n_task), minlength=n_task)
        cm = np.bincount(rng.integers(0, n_model, n_model), minlength=n_model)
        bs[b] = _slope_w(tcode, xb, y, (ct[ti] * cm[mi]).astype(float))
    return beta, _boot_ci(bs), bs


def read_n5(tab, method="probe", n_boot=1000, seed=0, power=None, power_min=0.80, n2_passed=True, censored=False,
            weak=False):
    """N5: beta CI vs 1 (fraction of depth) and 0 (absolute layer). `power`: C-sim power at beta = 0.5; below
    `power_min` the leaf is flagged descriptive (form() then substitutes L5n). Pythia reported separately."""
    ti, mi, y, xb = n5_points(tab, method)
    res = dict(node="N5", kind=tab.kind, method=method, n_points=int(len(y)), n_m=int(tab.M), n_f=int(len(tab.families)),
               bar=dict(contains_1_excludes_0="L5r", contains_0_excludes_1="L5a", excludes_both="L5p",
                        contains_both="L5n", power_min=power_min), power=power, leaf=None)
    if len(y) < 10 or len(np.unique(mi)) < 3:
        res.update(verdict="L5n", leaf="L5n", beta=None, ci=None, note="too few points")
    else:
        beta, ci, _ = n5_fit(ti, mi, y, xb, tab.T, tab.M, n_boot, seed)
        has1, has0 = ci[0] <= 1 <= ci[1], ci[0] <= 0 <= ci[1]
        leaf = "L5r" if has1 and not has0 else "L5a" if has0 and not has1 else "L5p" if not has0 and not has1 else "L5n"
        res.update(beta=beta, ci=ci, leaf=leaf, verdict=leaf)
        pi = [i for i in range(tab.M) if tab.lineage[i] == "Pythia"]
        if len(pi) >= 3:
            t2, m2, y2, x2 = n5_points(tab, method, pi)
            if len(y2) >= 6 and np.ptp(x2) > 0:
                b2, c2, _ = n5_fit(t2, m2, y2, x2, tab.T, tab.M, max(n_boot // 5, 50), seed + 3)
                res["pythia"] = dict(beta=b2, ci=c2, n_points=int(len(y2)))
    res["descriptive"] = bool(power is not None and power < power_min)
    res["passed"] = True
    res["sentence"] = leaf_sentence(res["leaf"], tab.kind, method, res["n_f"], res["n_m"], censored,
                                    weak or res["descriptive"], beta=res.get("beta"))
    return res


# ----------------------------------------------------------------------------------------------- N8
def n8_stats(P, U, min_fams=12):
    """Per-model Spearman and median gap from family value matrices P, U [M, nfam] (NaN = invalid)."""
    rho, gap, n = [], [], []
    for i in range(len(P)):
        ok = np.isfinite(P[i]) & np.isfinite(U[i])
        n.append(int(ok.sum()))
        rho.append(_rank_corr(P[i][ok], U[i][ok]) if ok.sum() >= min_fams else np.nan)
        gap.append(float(np.median(U[i][ok] - P[i][ok])) if ok.sum() >= min_fams else np.nan)
    return np.array(rho), np.array(gap), np.array(n)


def n8_leaf(rho_med, rho_hi, gap_ci, margin=0.05):
    if rho_med >= 0.5 and gap_ci[0] >= -margin and gap_ci[1] <= margin:
        return "L8a"
    if rho_med >= 0.5 and (gap_ci[0] > margin or gap_ci[1] < -margin):
        return "L8g"
    if rho_hi < 0.3:
        return "L8d"
    return "L8w"


def read_n8(tab, min_models=9, min_fams=12, icc_min=0.60, n_boot=1000, seed=0, icc=None, descriptive=False,
            censored=False, weak=False):
    """N8 on one Tab (trained plain prompt, or the unmodified 2-shot reading). Precondition: >= 9 models with >= 12
    families valid for both probe and use, and the use-onset ICC (between item halves, over the entering cells) >= 0.60;
    otherwise L8x. `icc` overrides the computed value."""
    P, _, _ = family_values(tab, "probe")
    U, _, _ = family_values(tab, "use")
    both = np.isfinite(P) & np.isfinite(U)
    enter = np.flatnonzero(both.sum(1) >= min_fams)
    res = dict(node="N8", kind=tab.kind, prompt=tab.prompt, method="probe-vs-use", n_m=int(len(enter)),
               n_f=int(both[enter].any(0).sum()) if len(enter) else 0, bar=dict(min_models=min_models, min_fams=min_fams,
               icc=icc_min, L8a="rho>=0.5, gap CI within +-0.05", L8g="rho>=0.5, gap CI beyond +-0.05",
               L8d="rho CI upper < 0.3"), leaf=None, descriptive=descriptive)
    ic = icc
    if ic is None and len(enter):
        sub = tab.half["use"]
        H = []
        for f in range(len(tab.families)):
            c = sub[enter][:, tab.task_fam == f, :]
            ok = np.isfinite(c).all(2)
            n = ok.sum(1)
            s = np.where(ok[..., None], c, 0).sum(1)
            cell = np.where(n[:, None] > 0, s / np.maximum(n, 1)[:, None], np.nan)
            cell[~both[enter][:, f]] = np.nan
            H.append(cell)
        ic = icc_2_1(np.concatenate(H, 0))
    res["icc_use"] = ic
    if len(enter) < min_models or ic is None or not math.isfinite(ic) or ic < icc_min:
        res.update(leaf="L8x", verdict="L8x", reason=("too few models" if len(enter) < min_models else "use-onset ICC"))
    else:
        rho, gap, n = n8_stats(P[enter], U[enter], min_fams)
        rng = np.random.default_rng(seed)
        mr, mg = [], []
        for _ in range(n_boot):
            ix = rng.integers(0, len(enter), len(enter))
            mr.append(np.nanmedian(rho[ix]))
            mg.append(np.nanmedian(gap[ix]))
        rci, gci = _boot_ci(mr), _boot_ci(mg)
        med_r, med_g = float(np.nanmedian(rho)), float(np.nanmedian(gap))
        leaf = n8_leaf(med_r, rci[1], gci)
        res.update(rho=med_r, rho_ci=rci, gap=med_g, gap_ci=gci, per_model_rho=rho.tolist(), per_model_gap=gap.tolist(),
                   models=[tab.models[i] for i in enter], leaf=leaf, verdict=leaf)
    # twin vs probe, descriptive
    T, _, _ = family_values(tab, "twin")
    bt = np.isfinite(P) & np.isfinite(T)
    rt = [_rank_corr(P[i][bt[i]], T[i][bt[i]]) for i in range(tab.M) if bt[i].sum() >= min_fams]
    res["twin_vs_probe_rho_median"] = float(np.nanmedian(rt)) if len(rt) else None
    res["sentence"] = leaf_sentence(res["leaf"], tab.kind, "probe vs use", res["n_f"], res["n_m"], censored, weak,
                                    rho=res.get("rho"), g=res.get("gap"))
    return res


# ----------------------------------------------------------------------------------------------- form (R2)
def _csim_get(csim, *path, default=None):
    cur = csim
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def csim_summary(csim):
    """Reduce a csim.json dict to the four numbers form() needs (None = not available)."""
    if not csim:
        return dict(n2_power_d3=None, n4_fpr=None, n8_fpr=None, n5_power=None, d_star=None)
    p3 = _csim_get(csim, "n2", "power_d3_worst_k")
    return dict(n2_power_d3=p3, n4_fpr=_csim_get(csim, "n4", "fpr_max"), n8_fpr=_csim_get(csim, "n8", "fpr_max"),
                n5_power=_csim_get(csim, "n5", "power"),
                d_star=_csim_get(csim, "n2", "d_star_worst_text", default=_csim_get(csim, "n2", "d_star_worst")))


PAPER_LEAVES = {"L3", "L3m", "L3f"}
SHORT_LEAVES = {"L1", "L2s", "L2x", "L2c", "L4m", "L8a", "L8g"}
INTERNAL_LEAVES = {"L0", "L1b", "L-meas", "L2w", "L3w", "L4u", "L4x", "L5r", "L5a", "L5p", "L5n", "L8w", "L8x"}


def _leaves_for_kind(r, cs, notes, n4_here=True):
    """(effective leaf list, paper-eligible flags) for one kind's readings, with C-sim substitutions."""
    out = []
    for node in ("N1", "N2", "N3", "N5", "N4", "N8"):
        x = r.get(node)
        if not x or not x.get("leaf"):
            continue
        leaf = x["leaf"]
        if node == "N4":
            if not n4_here:
                continue
            fpr = cs["n4_fpr"]
            if leaf in ("L4+", "L4g", "L4m") and (fpr is None or fpr > 0.10):
                notes.append(f"N4 {leaf} made descriptive (C-sim FPR {fpr}) -> L4u")
                leaf = "L4u"
        elif node == "N8":
            fpr = cs["n8_fpr"]
            if leaf in ("L8a", "L8g", "L8d") and (fpr is None or fpr > 0.10):
                notes.append(f"N8 {leaf} made descriptive (C-sim FPR {fpr}) -> L8w")
                leaf = "L8w"
        elif node == "N5":
            pw = cs["n5_power"]
            if leaf in ("L5r", "L5a", "L5p") and (pw is None or pw < 0.80):
                notes.append(f"N5 {leaf} made descriptive (C-sim power {pw}) -> L5n")
                leaf = "L5n"
        elif node == "N2" and leaf == "L2c":
            pw = cs["n2_power_d3"]
            if pw is None or pw < 0.80:
                leaf = "L2c-underpowered"
        out.append(leaf)
    c = r.get("contrast")
    if c and c.get("leaf"):
        out.append(c["leaf"])
    return out


def form(readings, csim=None):
    """R2. readings: {'could_not_test': bool, 'stop_after_gate1': bool, 'unmodified': {N1,N2,N3,N4,N5,N8,contrast},
    'trained': {...} | None}. Each node dict carries 'leaf' (None when the node passes on). csim: csim.json dict (or
    anything with the keys csim_summary reads). Missing C-sim numbers never strengthen a leaf. Returns
    {form, rule, notes, leaves}."""
    cs = csim_summary(csim)
    notes = []
    if readings.get("could_not_test"):
        return dict(form="internal report", rule=1, notes=["C-check or C-screen ended at COULD NOT BE TESTED"], leaves={})
    stop = bool(readings.get("stop_after_gate1"))
    kinds = ["unmodified"] if stop or not readings.get("trained") else ["unmodified", "trained"]
    if stop:
        notes.append("rule 2: stopped after gate 1; read on the unmodified kind, trained not run")
    leaves = {}
    for k in kinds:
        n4_here = (k == "unmodified") if len(kinds) == 1 else (k == "trained")
        leaves[k] = _leaves_for_kind(readings[k], cs, notes, n4_here)
    allv = [l for v in leaves.values() for l in v]
    paper = [l for l in allv if l in PAPER_LEAVES]
    if any(l == "L2c" for l in allv):
        paper.append("L2c")
    for l in allv:
        if l in ("L4+", "L4g") or l == "L8d":
            paper.append(l)
    if paper:
        return dict(form="paper", rule=3, notes=notes + [f"reached: {sorted(set(paper))}"], leaves=leaves)
    short = [l for l in allv if l in SHORT_LEAVES or l == "L2c-underpowered"]
    if short:
        return dict(form="short note", rule=4, notes=notes + [f"best: {sorted(set(short))}"], leaves=leaves)
    if all(l in INTERNAL_LEAVES for l in allv):
        return dict(form="internal report", rule=5, notes=notes, leaves=leaves)
    return dict(form="short note", rule=6, notes=notes, leaves=leaves)


# ----------------------------------------------------------------------------------------------- orchestration
def run_tree(rows, gate=1, csim=None, n_null=1000, n_boot=1000, seed=0, reserves_available=False,
             stop_after_gate1=False, could_not_test=False, models=None, quick=False):
    """Read the whole tree on the rows. gate 1: unmodified kind; gate 2: both kinds, trained N4 as the verdict.
    Returns {'readings': {...}, 'form': ..., 'sentences': [...]}."""
    cs = csim_summary(csim)
    weak2 = cs["n2_power_d3"] is not None and cs["n2_power_d3"] < 0.80
    readings = dict(could_not_test=could_not_test, stop_after_gate1=stop_after_gate1, trained=None, gate=gate)
    kinds = ["unmodified"] if gate == 1 else ["unmodified", "trained"]
    tabs = {k: Tab(rows, k, "plain", 0, models) for k in kinds}
    for k in kinds:
        tab = tabs[k]
        r = dict(tab_models=tab.models, n_tasks=tab.T, n_families=len(tab.families))
        r["family_counts"] = {m: family_counts(tab, m) for m in METHODS}
        r["early_read_tasks"] = [t for t, e in early_read_flags(tab).items() if e]
        r["censoring"] = {m: censoring_check(tab, m) for m in METHODS}
        cens = r["censoring"]["probe"]["undersampled"]
        r["N1"] = read_n1(tab, "probe", n_boot=n_boot, seed=seed, reserves_available=reserves_available,
                          censored=cens)
        r["N1_descriptive"] = {m: read_n1(tab, m, n_boot=min(n_boot, 200), seed=seed, reserves_available=True)
                               for m in ("twin", "use")}
        if r["N1"]["passed"]:
            r["N2"] = read_n2(tab, "probe", n_null=n_null, n_boot=n_boot, seed=seed, reserves_available=reserves_available,
                              d_star=cs["d_star"], censored=cens, weak=weak2)
            if r["N2"].get("discrete"):
                tiny = Tab(rows, k, "plain", 0, [HELDOUT]) if any(x["model"] == HELDOUT and x["kind"] == k for x in rows) else None
                r["N3"] = read_n3(tab, r["N2"], "probe", n_null=n_null, seed=seed, tiny_tab=tiny, censored=cens)
            if r["N2"].get("passed"):
                r["N5"] = read_n5(tab, "probe", n_boot=n_boot, seed=seed, power=cs["n5_power"], censored=cens)
        # N8: trained kind (plain), or unmodified 2-shot
        if k == "trained":
            r["N8"] = read_n8(tab, n_boot=n_boot, seed=seed, censored=cens)
        else:
            t2 = Tab(rows, "unmodified", "twoshot", 0, models)
            r["N8"] = read_n8(t2, n_boot=n_boot, seed=seed) if t2.T else dict(
                node="N8", leaf="L8x", reason="no 2-shot rows", verdict="L8x")
        readings[k] = r
    if gate == 2 and "trained" in kinds:
        readings["unmodified"]["N4"] = read_n4(tabs, 1, n_null, n_boot, seed)
        readings["trained"]["N4"] = read_n4(tabs, 2, n_null, n_boot, seed)
    else:
        readings["unmodified"]["N4"] = read_n4(tabs, 1, n_null, n_boot, seed)
    c = contrast({k: readings[k].get("N2") for k in kinds})
    if c:
        readings[kinds[-1]]["contrast"] = c
    out = dict(readings=readings, form=form(readings if gate == 2 else dict(readings, trained=None), csim))
    out["sentences"] = [x["sentence"] for k in kinds for x in
                        [readings[k].get(n) for n in ("N1", "N2", "N3", "N4", "N5", "N8")] if x and x.get("sentence")]
    return out
