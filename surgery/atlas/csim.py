"""C-sim (proposal section 5): power and false-positive simulations for N2, N4, N5 and N8, on measured spreads.

  spreads = spreads_from_data()            # reads var/cscreen/<model>/<task>.json and var/p0/results*/
  out = run_csim(spreads, reps=200, n_null=200, jobs=13)

Output schema (read by tree.csim_summary / tree.form):
  n2: power[k][d] = {power, se, power_null, ...}, d_star{k}, d_star_worst(_text), power_d3_worst_k
  n4: power_by_delta, fpr_continuum, fpr_lineage, fpr_max
  n8: levels{name: {fpr_L8g, fpr_L8d, fpr}}, fpr_max (the measured-noise level)
  n5: power, se
Every simulation is seeded per (cell, replicate) with numpy SeedSequence, so results do not depend on n_jobs.
All replicate counts are parameters; Monte Carlo SEs are reported.
"""
from __future__ import annotations

import glob
import json
import math
import zlib
from pathlib import Path

import numpy as np

from surgery.atlas import tree as T

D_GRID = (2.5, 3.0, 3.5)
KS = (2, 3)


def _rng(*key):
    return np.random.default_rng(np.random.SeedSequence([int(k) for k in key]))


def _se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / max(n, 1))


def _par(fn, args, n_jobs):
    if n_jobs == 1 or len(args) <= 1:
        return [fn(*a) for a in args]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(delayed(fn)(*a) for a in args)


# ----------------------------------------------------------------------------------------------- measured spreads
def _load(path):
    try:
        j = json.loads(Path(path).read_text())
    except Exception:
        return None
    return j if isinstance(j, dict) else None


def spreads_from_data(cscreen_root="var/cscreen", p0_roots=("var/p0/results", "var/p0/results_trained"),
                      trained_root="var/p0/results_trained"):
    """Measured spreads. C-screen files (`probe_onset`, `twin_onset`: {rel, layers, width}; family; n_blocks) give
    the probe-onset distribution, the within-family share of task-onset variance and the noise of log layer onsets.
    P0 files give the use-onset range (0.71-0.85 in the pilot), use rise width and peak, and, from the trained pilot
    curves' `use.random_recovery` (the random-subspace baseline per block), the per-block noise SD (RMS about 0)."""
    cs = [(p, _load(p)) for p in sorted(glob.glob(f"{cscreen_root}/*/*.json"))]
    cs = [(p, j) for p, j in cs if j and "probe_onset" in j]
    probe = [j["probe_onset"]["rel"] for _, j in cs if j.get("probe_onset")]
    twin = [j["twin_onset"]["rel"] for _, j in cs if j.get("twin_onset")]
    # within-family share of task variance: per model, families with two valid tasks
    shares, fam_pairs = [], 0
    by_model = {}
    for _, j in cs:
        if j.get("probe_onset"):
            by_model.setdefault(j["model"], {}).setdefault(j["family"], []).append(j["probe_onset"]["rel"])
    for m, fams in by_model.items():
        pairs = [v for v in fams.values() if len(v) == 2]
        allv = [x for v in fams.values() for x in v]
        if len(pairs) >= 5 and np.var(allv) > 0:
            vw = float(np.mean([(a - b) ** 2 / 2 for a, b in pairs]))
            shares.append(min(max(vw / np.var(allv), 0.02), 0.98))
            fam_pairs += len(pairs)
    # noise of log layer onsets: residual SD after removing task and model means (tasks valid in >= 2 models)
    L = {}
    for _, j in cs:
        o = j.get("probe_onset")
        if o and o.get("layers") and o["layers"] >= 1:
            L.setdefault(j["task"], {})[j["model"]] = math.log(o["layers"])
    mods = sorted({m for v in L.values() for m in v})
    rows = [[v.get(m, np.nan) for m in mods] for v in L.values() if len(v) >= 2]
    ll_sd = None
    if len(rows) >= 5 and len(mods) >= 2:
        A = np.array(rows)
        A = A - np.nanmean(A, 1, keepdims=True)
        A = A - np.nanmean(A, 0, keepdims=True)
        ll_sd = float(np.sqrt(np.nanmean(A ** 2) * len(mods) / max(len(mods) - 1, 1)))
    # P0
    use_on, use_w, use_pk = [], [], []
    for root in p0_roots:
        for p in sorted(glob.glob(f"{root}/*/*.json")):
            j = _load(p)
            if not j:
                continue
            o = (j.get("onset") or {}).get("use")
            if o and o.get("rel") is not None:
                use_on.append(o["rel"])
                use_w.append(o.get("width"))
                use_pk.append(o.get("peak"))
    rr = []
    for p in sorted(glob.glob(f"{trained_root}/*/*.json")):
        j = _load(p)
        r = ((j or {}).get("use") or {}).get("random_recovery")
        if r:
            rr.append(np.asarray(r, float))
    noise = float(np.sqrt(np.mean(np.concatenate(rr) ** 2))) if rr else None
    use_w = [w for w in use_w if w is not None]
    use_pk = [w for w in use_pk if w is not None]
    return dict(
        probe_onsets=[float(x) for x in probe], probe_sd=float(np.std(probe)) if probe else None,
        twin_sd=float(np.std(twin)) if twin else None,
        within_family_share=float(np.mean(shares)) if shares else None, n_family_pairs=fam_pairs,
        log_layer_noise_sd=ll_sd,
        use_onsets=[float(x) for x in use_on], use_range=[min(use_on), max(use_on)] if use_on else None,
        use_sd=float(np.std(use_on)) if use_on else None,
        use_width=float(np.mean(use_w)) if use_w else None, use_peak=float(np.mean(use_pk)) if use_pk else None,
        block_noise_sd=noise, n_trained_curves=len(rr),
        sources=dict(cscreen=len(cs), p0_use=len(use_on)))


DEFAULTS = dict(within_family_share=0.30, log_layer_noise_sd=0.30, block_noise_sd=5e-4, use_width=0.05, use_peak=0.8,
                probe_onsets=list(np.linspace(0.05, 0.85, 40)))


def _sp(spreads, key):
    v = (spreads or {}).get(key)
    return DEFAULTS[key] if v is None or (isinstance(v, list) and not v) else v


# ----------------------------------------------------------------------------------------------- N2 power
def sim_stage_data(k, d, share_w, rng, n_fam=30):
    """Task-level standardised onsets for k equal stages with centres d apart (in units of the within-stage SD, which is
    1 = family part + within-family part with variance share `share_w`). Two tasks per family. Returns (x [2 n_fam],
    fam ids, stage per point)."""
    stage = np.repeat(np.arange(k), int(math.ceil(n_fam / k)))[:n_fam]
    fam_eff = rng.normal(0, math.sqrt(1 - share_w), n_fam)
    x = np.empty(2 * n_fam)
    fam = np.repeat(np.arange(n_fam), 2)
    for t in range(2):
        x[t::2] = stage * d + fam_eff + rng.normal(0, math.sqrt(share_w), n_fam)
    return x, fam, np.repeat(stage, 2)


def _n2_rep(k, d, rep, share_w, n_null, n_boot, n_starts, seed):
    rng = _rng(seed, 2, k, int(round(d * 10)), rep)
    x, fam, _ = sim_stage_data(k, d, share_w, rng)
    dec = T.n2_decide(x, fam, np.zeros(len(x), bool), n_null, n_boot, n_starts, 0.05, int(rng.integers(1 << 30)))
    return dict(discrete=dec["verdict"] == "discrete", beaten=bool(dec["null_beaten"]), kstar=int(dec["k"]),
                verdict=dec["verdict"])


def n2_power(spreads=None, reps=200, n_null=200, n_boot=20, n_starts=10, d_grid=D_GRID, ks=KS, seed=0, n_jobs=1,
             share_w=None, level=0.80):
    """Power of N2 (verdict `discrete`: null beaten, k* in {2, 3}, stability >= 0.70, separation met) when the data are
    k equal stages d apart. d* = smallest d in the grid with power >= `level`; the L2c condition uses the worse k
    (the larger d*)."""
    sw = _sp(spreads, "within_family_share") if share_w is None else share_w
    power, dstar = {}, {}
    for k in ks:
        power[str(k)] = {}
        for d in d_grid:
            rs = _par(_n2_rep, [(k, d, r, sw, n_null, n_boot, n_starts, seed) for r in range(reps)], n_jobs)
            p = float(np.mean([r["discrete"] for r in rs]))
            pn = float(np.mean([r["beaten"] for r in rs]))
            power[str(k)][str(d)] = dict(power=p, se=_se(p, reps), power_null=pn, se_null=_se(pn, reps),
                                         p_kstar_equals_k=float(np.mean([r["kstar"] == k for r in rs])), reps=reps)
        ok = [d for d in d_grid if power[str(k)][str(d)]["power"] >= level]
        dstar[str(k)] = min(ok) if ok else None
    worst = None if any(v is None for v in dstar.values()) else max(dstar.values())
    p3 = [power[str(k)][str(3.0)]["power"] for k in ks if str(3.0) in power[str(k)]]
    return dict(power=power, d_star=dstar, d_star_worst=worst,
                d_star_worst_text=(f"{worst}" if worst is not None else f">{max(d_grid)}"),
                power_d3_worst_k=min(p3) if p3 else None, reps=reps, n_null=n_null, n_boot=n_boot,
                within_family_share=sw, level=level)


# ----------------------------------------------------------------------------------------------- N4
LINEAGES14 = [T.MODEL_META[m][0] for m in T.FITTED]


def synthetic_profile(rng=None, n_fam=20, n_feat=3, factor_share=0.5, lineages=None):
    """A stand-in 14 x (n_feat n_fam) standardised profile with a one-factor structure, used when no real profile is
    supplied (C-sim runs before the data exist). Real profiles can be passed as `profile` (dict with X, lineage,
    col_fam, tab-like n_params / n_blocks / release / fam_type)."""
    rng = np.random.default_rng(0) if rng is None else rng
    lin = lineages or LINEAGES14
    M, P = len(lin), n_feat * n_fam
    t = rng.normal(size=M)
    load = rng.normal(size=P) * math.sqrt(factor_share)
    X = np.outer(t, load) + rng.normal(size=(M, P)) * math.sqrt(1 - factor_share)
    return _profile_dict(T._zcols(X), lin, np.tile(np.arange(n_fam), n_feat))


class _Meta:
    """Tab stand-in for n4b on a profile without a table: sizes and blocks of the 14 fitted models, no release dates."""

    def __init__(self, lin):
        ok = len(lin) == len(T.FITTED)
        self.n_params = np.array([T.MODEL_META[m][2] for m in T.FITTED]) if ok else np.full(len(lin), np.nan)
        self.n_blocks = np.array([T.MODEL_META[m][1] for m in T.FITTED], float) if ok else np.full(len(lin), np.nan)
        self.release = [None] * len(lin)
        self.fam_type = [f"t{i % 6}" for i in range(400)]


def _profile_dict(X, lin, col_fam, meta=None):
    return dict(X=X, lineage=list(lin), col_fam=np.asarray(col_fam), models=[f"m{i}" for i in range(len(lin))],
                tab=meta or _Meta(lin))


def _n4_read(prof, n_null, n_boot, seed):
    a = T.n4a(prof, n_null, n_boot, seed)
    labels = a.pop("labels")
    b = T.n4b(prof, a["k"], seed=seed)
    lin = np.array(prof["lineage"])
    split = any(len(set(labels[lin == l])) > 1 for l in set(lin))
    return T.n4_exit(a["groups"], split, b["winner"])[0]


def _fit_one_factor(X):
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    t, v = U[:, 0] * S[0], Vt[0]
    sd = (X - X.mean(0) - np.outer(t, v)).std(0)
    return t.std(), v, sd


def _fit_lineage(X, lin):
    lin = np.array(lin)
    mu = np.zeros_like(X)
    for l in set(lin):
        mu[lin == l] = X[lin == l].mean(0)
    return mu, (X - mu).std(0)


def _n4_rep(kind, rep, prof, delta, n_null, n_boot, seed):
    rng = _rng(seed, 4, {"power": 1, "continuum": 2, "lineage": 3}[kind], int(round(delta * 100)), rep)
    X, lin = prof["X"], prof["lineage"]
    M, P = X.shape
    if kind == "power":  # 3 groups that cut across lineages (model i -> group i mod 3), unit noise
        g = np.arange(M) % 3
        C = rng.normal(0, delta, (3, P))
        Xn = C[g] + rng.normal(size=(M, P))
    elif kind == "continuum":  # one-factor continuum fitted to the profile
        ts, v, sd = _fit_one_factor(X)
        Xn = np.outer(rng.normal(0, ts, M), v) + rng.normal(size=(M, P)) * sd
    else:  # lineage-only structure fitted to the profile: lineage means + independent residuals
        mu, sd = _fit_lineage(X, lin)
        Xn = mu + rng.normal(size=(M, P)) * sd
    p = dict(prof, X=T._zcols(Xn))
    return _n4_read(p, n_null, n_boot, int(rng.integers(1 << 30)))


def n4_sim(spreads=None, reps=100, n_null=100, n_boot=30, seed=0, n_jobs=1, profile=None, deltas=(0.75, 1.0, 1.5)):
    """N4 C-sim. Power: 3 groups across lineages (group effect SD `delta` per column, unit noise) read as L4+.
    False positives: under (a) a one-factor continuum and (b) lineage-only structure, both fitted to `profile` (a
    synthetic one-factor profile if None). Positive = L4+ under (a) (a continuum has no groups; L4g with the smooth
    clause is the right reading there); L4+ or L4g under (b). Both counts are reported."""
    prof = profile or synthetic_profile(_rng(seed, 40))
    out = dict(power_by_delta={}, reps=reps, n_null=n_null, n_boot=n_boot, profile="given" if profile else "synthetic",
               positive_definition="continuum: L4+; lineage-only: L4+ or L4g")
    for dlt in deltas:
        rs = _par(_n4_rep, [("power", r, prof, dlt, n_null, n_boot, seed) for r in range(reps)], n_jobs)
        p = float(np.mean([x == "L4+" for x in rs]))
        out["power_by_delta"][str(dlt)] = dict(power=p, se=_se(p, reps), leaves=_count(rs))
    for name, key in (("continuum", "fpr_continuum"), ("lineage", "fpr_lineage")):
        rs = _par(_n4_rep, [(name, r, prof, 0.0, n_null, n_boot, seed) for r in range(reps)], n_jobs)
        pos = [x == "L4+" for x in rs] if name == "continuum" else [x in ("L4+", "L4g") for x in rs]
        p = float(np.mean(pos))
        out[key] = dict(fpr=p, se=_se(p, reps), leaves=_count(rs), any_L4plus_or_L4g=float(np.mean([x in ("L4+", "L4g") for x in rs])))
    out["fpr_max"] = max(out["fpr_continuum"]["fpr"], out["fpr_lineage"]["fpr"])
    return out


def _count(xs):
    return {k: int(sum(1 for x in xs if x == k)) for k in sorted(set(xs))}


# ----------------------------------------------------------------------------------------------- N8
def onset_batch(curves):
    """Half-peak onset (relative depth) of each curve in curves [..., B], the rule of tree.onset_curve; NaN if the
    peak is not positive."""
    c = np.asarray(curves, float)
    B = c.shape[-1]
    peak = c.max(-1)
    thr = 0.5 * peak
    ge = c >= thr[..., None]
    i = ge.argmax(-1)
    ci = np.take_along_axis(c, i[..., None], -1)[..., 0]
    cp = np.take_along_axis(c, np.maximum(i - 1, 0)[..., None], -1)[..., 0]
    flat = (i == 0) | (ci == cp)
    x = np.where(flat, i, (i - 1) + (thr - cp) / np.where(ci == cp, 1.0, ci - cp))
    return np.where(peak > 0, x / B, np.nan)


def sim_use_curves(probe, blocks, noise_sd, width, peak, rng):
    """Use curves whose noiseless onset is the probe onset: a logistic ramp in relative depth with the measured rise
    width and peak, plus N(0, noise_sd) per block. probe [M, F]; blocks [M]. Returns noisy onsets [M, F]."""
    M, F = probe.shape
    Bmax = int(max(blocks))
    out = np.full((M, F), np.nan)
    s = width / 4.394  # 10-90% width of a logistic = 4.394 scale
    for m in range(M):
        B = int(blocks[m])
        x = (np.arange(B) / B)[None, None, :]
        curve = peak / (1 + np.exp(-(x - probe[m][None, :, None]) / s))[0]
        curve = curve + rng.normal(0, noise_sd, curve.shape)
        out[m] = onset_batch(curve)
    return out


def _n8_rep(rep, level_name, noise_sd, spreads, n_fam, n_boot, seed):
    rng = _rng(seed, 8, zlib.crc32(level_name.encode()) % 100003, rep)
    pool = np.asarray(_sp(spreads, "probe_onsets"), float)
    models = T.FITTED
    blocks = np.array([T.MODEL_META[m][1] for m in models])
    probe = rng.choice(pool, (len(models), n_fam))
    use = sim_use_curves(probe, blocks, noise_sd, _sp(spreads, "use_width"), _sp(spreads, "use_peak"), rng)
    rho, gap, _ = T.n8_stats(probe, use, min_fams=min(12, n_fam))
    ix = rng.integers(0, len(models), (n_boot, len(models)))
    mr, mg = np.nanmedian(rho[ix], 1), np.nanmedian(gap[ix], 1)
    leaf = T.n8_leaf(float(np.nanmedian(rho)), T._boot_ci(mr)[1], T._boot_ci(mg))
    return leaf


def n8_sim(spreads=None, reps=100, n_fam=25, n_boot=100, seed=0, n_jobs=1, levels=None):
    """N8 false-positive rate of L8g and L8d when use onset = probe onset + noise: each model's use curve is a ramp at
    the family's probe onset with the measured per-block noise SD (RMS of P0's trained `random_recovery`), and its onset
    is recomputed. The headline (`fpr_max`) is the measured-noise level; larger levels are reported as sensitivity
    (x10 of measured, and 10% of the curve's peak)."""
    meas = _sp(spreads, "block_noise_sd")
    peak = _sp(spreads, "use_peak")
    levels = levels or {"measured": meas, "x10": 10 * meas, "peak10pct": 0.10 * peak}
    out = dict(levels={}, reps=reps, n_fam=n_fam, noise_sd_measured=meas)
    for name, sd in levels.items():
        rs = _par(_n8_rep, [(r, name, sd, spreads, n_fam, n_boot, seed) for r in range(reps)], n_jobs)
        g, d = float(np.mean([x == "L8g" for x in rs])), float(np.mean([x == "L8d" for x in rs]))
        out["levels"][name] = dict(noise_sd=sd, fpr_L8g=g, fpr_L8d=d, fpr=max(g, d), fpr_either=g + d,
                                   se=_se(g + d, reps), leaves=_count(rs))
    out["fpr_max"] = out["levels"]["measured"]["fpr_either"]
    return out


# ----------------------------------------------------------------------------------------------- N5
def _n5_rep(rep, beta, n_task, noise_sd, spreads, n_boot, seed):
    rng = _rng(seed, 5, int(beta * 100), rep)
    models = T.FITTED
    B = np.array([T.MODEL_META[m][1] for m in models], float)
    pool = np.asarray(_sp(spreads, "probe_onsets"), float)
    task_eff = np.log(np.maximum(rng.choice(pool, n_task), 0.05) * 24.0)  # log layer onset at 24 blocks, beta-free part
    ti, mi = np.meshgrid(np.arange(n_task), np.arange(len(models)), indexing="ij")
    ti, mi = ti.ravel(), mi.ravel()
    y = task_eff[ti] + beta * (np.log(B[mi]) - np.log(24.0)) + rng.normal(0, noise_sd, len(ti))
    keep = y >= 0  # layer onset >= 1
    ti, mi, y = ti[keep], mi[keep], y[keep]
    xb = np.log(B[mi])
    _, ci, _ = T.n5_fit(ti, mi, y, xb, n_task, len(models), n_boot, int(rng.integers(1 << 30)))
    return bool(not (ci[0] <= 0 <= ci[1]) and not (ci[0] <= 1 <= ci[1]))


def n5_power(spreads=None, reps=200, beta=0.5, n_task=50, n_boot=200, seed=0, n_jobs=1):
    """Power at beta = 0.5: the share of simulated (task x model) tables whose two-way bootstrap CI for beta excludes
    both 0 and 1 (the L5p exit). Task onsets are resampled from the measured probe onsets; the noise SD of log layer
    onset is the measured one (C-screen residual after task and model means)."""
    sd = _sp(spreads, "log_layer_noise_sd")
    rs = _par(_n5_rep, [(r, beta, n_task, sd, spreads, n_boot, seed) for r in range(reps)], n_jobs)
    p = float(np.mean(rs))
    return dict(power=p, se=_se(p, reps), beta=beta, reps=reps, noise_sd=sd, n_task=n_task)


# ----------------------------------------------------------------------------------------------- everything
def run_csim(spreads=None, reps=100, n_null=200, n_boot=20, n_jobs=1, seed=0, n_reps_n4=None, n_reps_n8=None,
             n_reps_n5=None, d_grid=D_GRID, profile=None):
    """All four sections. Returns the csim.json dict (see module docstring); `spreads` None -> measured spreads are read
    from disk (and defaults fill any that are missing; the used values are echoed under `spreads_used`)."""
    sp = spreads if spreads is not None else spreads_from_data()
    out = dict(seed=seed, reps=reps, n_null=n_null)
    out["spreads_used"] = {k: (_sp(sp, k) if k in DEFAULTS else sp.get(k)) for k in sorted(set(DEFAULTS) | set(sp or {}))
                           if k != "probe_onsets" and k != "use_onsets"}
    out["spreads_used"]["n_probe_onsets"] = len(_sp(sp, "probe_onsets"))
    out["n2"] = n2_power(sp, reps, n_null, n_boot, d_grid=d_grid, seed=seed, n_jobs=n_jobs)
    out["n4"] = n4_sim(sp, n_reps_n4 or max(reps // 2, 2), max(n_null // 2, 20), max(n_boot, 10), seed, n_jobs, profile)
    out["n8"] = n8_sim(sp, n_reps_n8 or reps, seed=seed, n_jobs=n_jobs)
    out["n5"] = n5_power(sp, n_reps_n5 or reps, seed=seed, n_jobs=n_jobs)
    return out
