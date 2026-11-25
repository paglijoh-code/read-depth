"""M9 atlas: the secondary "task overlap" analysis (docs/ATLAS-overlap.md, fixed 2026-11-10).

Pure functions over arrays and dicts, numpy only. File reading and the report live in scripts/atlas_overlap.py.
Conventions:
  - Drop vectors cover blocks 1-26 (`BLOCKS`); blocks 0 and 27 are never compared.
  - A similarity that cannot be computed (a constant drop vector) is nan, and nan pairs are left out of every mean.
  - Permutation p-values are one-sided (statistic >= observed). Where the permutation space has at most `N_PERM`
    members it is enumerated and p = count / space (the observed labelling is one member); otherwise p = (1 + count)
    / (1 + N_PERM) over `N_PERM` random draws of seed `SEED`.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from surgery.atlas.heal import _ranks, spearman

BLOCKS = slice(1, 27)       # blocks 1..26 of a 28-block drop vector
N_BLOCKS = 28
FRAGILE = 0.05
EPS = 1e-9
N_PERM = 10_000
N_BOOT = 2_000
SEED = 9
N_SHUFFLE = 100
SV_TOL = 1e-6
ALPHA = 0.05
K_LOAD = 12
LAYERS = tuple(range(29))
MAP_LAYERS = (6, 12, 18, 24)


# ---------------------------------------------------------------- block drop vectors and similarity

def block_vector(drops) -> np.ndarray:
    """Lesion drops of blocks 1..26 from the 28-entry drop list (blocks 0..27)."""
    d = np.asarray(drops, dtype=float)
    if d.shape != (N_BLOCKS,):
        raise ValueError(f"expected {N_BLOCKS} lesion drops (blocks 0..27), got shape {d.shape}")
    return d[BLOCKS]


def fragile_set(drops, threshold: float = FRAGILE) -> set[int]:
    """Blocks in 1..26 whose drop is at least `threshold` (block ids, not positions)."""
    v = block_vector(drops)
    return {i + 1 for i, x in enumerate(v) if x >= threshold - EPS}


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    u = a | b
    return float("nan") if not u else len(a & b) / len(u)


def similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Spearman rho between the rows of `vectors` (tasks x blocks); diagonal 1, nan where a row is constant."""
    n = len(vectors)
    s = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            s[i, j] = s[j, i] = spearman(vectors[i], vectors[j])
    return s


def jaccard_matrix(sets) -> np.ndarray:
    n = len(sets)
    j = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            j[a, b] = j[b, a] = jaccard(sets[a], sets[b])
    return j


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


# ---------------------------------------------------------------- O1a, O1b

def kind_test(sim: np.ndarray, kinds, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """O1a: mean similarity over same-kind pairs minus mean over cross-kind pairs; p over relabellings that keep the
    group sizes (enumerated when there are at most n_perm of them)."""
    kinds = np.asarray(kinds)
    n = len(kinds)
    iu = np.triu_indices(n, 1)
    v = sim[iu]
    ok = np.isfinite(v)
    v0 = np.where(ok, v, 0.0)

    def stats(labels):  # labels: (m, n) -> (m,)
        same = (labels[:, iu[0]] == labels[:, iu[1]]) & ok
        cross = (labels[:, iu[0]] != labels[:, iu[1]]) & ok
        ns, nc = same.sum(1), cross.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (same * v0).sum(1) / ns - (cross * v0).sum(1) / nc

    obs = float(stats(kinds[None, :])[0])
    first = kinds == kinds[0]
    m = int(first.sum())
    space = math.comb(n, m)
    if space <= n_perm:
        labels = np.zeros((space, n), dtype=int)
        for r, idx in enumerate(itertools.combinations(range(n), m)):
            labels[r, list(idx)] = 1
        dist = stats(labels)
        p = float((dist >= obs - EPS).sum() / space)
        method, total = "exact", space
    else:
        rng = np.random.default_rng(seed)
        labels = np.stack([rng.permutation(kinds) for _ in range(n_perm)])
        dist = stats(labels)
        p = float((1 + (dist >= obs - EPS).sum()) / (1 + n_perm))
        method, total = "permutation", n_perm
    labs = kinds[None, :]
    same = (labs[:, iu[0]] == labs[:, iu[1]])[0] & ok
    cross = (labs[:, iu[0]] != labs[:, iu[1]])[0] & ok
    return {"effect": obs, "p": p, "method": method, "n_perm": total,
            "same_mean": _nanmean(v[same]), "cross_mean": _nanmean(v[cross]),
            "n_same": int(same.sum()), "n_cross": int(cross.sum())}


def matching_test(sim: np.ndarray, appraise, execute) -> dict:
    """O1b: `appraise[i]` and `execute[i]` are a matched pair (indices into `sim`). Statistic: mean similarity over the
    matched pairs minus mean over the other appraise-execute pairs among these tasks. p: exact over every way of
    matching the appraise tasks to the execute tasks (m! of them, the true matching is one)."""
    b = sim[np.ix_(list(appraise), list(execute))]
    m = len(b)
    ok = np.isfinite(b)
    b0 = np.where(ok, b, 0.0)
    rows = np.arange(m)

    def stat(perm):
        pick = np.zeros((m, m), dtype=bool)
        pick[rows, list(perm)] = True
        a, c = pick & ok, ~pick & ok
        return (b0[a].sum() / a.sum() if a.any() else float("nan")) - (b0[c].sum() / c.sum() if c.any() else float("nan"))

    obs = stat(range(m))
    dist = np.array([stat(p) for p in itertools.permutations(range(m))])
    n_match = len(dist)
    return {"effect": float(obs), "p": float((dist >= obs - EPS).sum() / n_match), "method": "exact", "n_perm": n_match,
            "matched_mean": _nanmean(b[rows, rows]),
            "other_mean": _nanmean(b[~np.eye(m, dtype=bool)]), "n_other": int((~np.eye(m, dtype=bool) & ok).sum())}


# ---------------------------------------------------------------- O2

def lesion_load(drops, order, k: int = K_LOAD) -> float:
    """Sum of max(drop, 0) over the first k blocks of the cut order (block ids index the 28-entry drop list)."""
    d = np.asarray(drops, dtype=float)
    return float(sum(max(d[b], 0.0) for b in order[:k]))


def spearman_test(x, y, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """Spearman rho of x against y with the one-sided permutation p (y permuted over items)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho = spearman(x, y)
    if not math.isfinite(rho):
        return {"rho": rho, "p": float("nan"), "n": len(x)}
    rx, ry = _ranks(x), _ranks(y)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        if np.corrcoef(rx, rng.permutation(ry))[0, 1] >= rho - EPS:
            cnt += 1
    return {"rho": rho, "p": (1 + cnt) / (1 + n_perm), "n": len(x)}


def bootstrap_interval(x, y, n_boot: int = N_BOOT, seed: int = SEED) -> list:
    """2.5 and 97.5 percentiles of the finite Spearman rhos over task resamples (with replacement)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        r = spearman(x[idx], y[idx])
        if math.isfinite(r):
            rhos.append(r)
    if not rhos:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return [float(lo), float(hi)]


def lesion_test(tasks, load: dict, damage: dict, rivals: dict) -> dict:
    """O2 over `tasks`: Spearman of lesion load against damage (permutation p, bootstrap interval), plus the same rho
    and interval for the rivals {name: {task: value}} against the same damage."""
    dm = np.array([damage[t] for t in tasks])
    out = {"tasks": list(tasks), "damage": {t: damage[t] for t in tasks}, "load": {t: load[t] for t in tasks}}
    ld = np.array([load[t] for t in tasks])
    out["lesion_load"] = {**spearman_test(ld, dm), "interval": bootstrap_interval(ld, dm)}
    for name, vals in rivals.items():
        v = np.array([vals[t] for t in tasks], dtype=float)
        out[name] = {"rho": spearman(v, dm), "interval": bootstrap_interval(v, dm)}
    return out


# ---------------------------------------------------------------- O3

def symmetrise(transfer: dict, tasks) -> np.ndarray:
    """(T[a][b] + T[b][a]) / 2; nan where either direction is missing. Diagonal nan."""
    n = len(tasks)
    out = np.full((n, n), np.nan)
    for i, a in enumerate(tasks):
        for j, b in enumerate(tasks):
            if i == j:
                continue
            x, y = transfer.get(a, {}).get(b), transfer.get(b, {}).get(a)
            if x is not None and y is not None:
                out[i, j] = (x + y) / 2
    return out


def mantel_test(sim: np.ndarray, tmat: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """O3: Spearman rho between similarity and symmetrised transfer over task pairs with both defined; p permutes the
    task labels of the similarity matrix."""
    n = len(sim)
    iu = np.triu_indices(n, 1)
    t = tmat[iu]

    def rho_of(s_full):
        s = s_full[iu]
        ok = np.isfinite(s) & np.isfinite(t)
        return spearman(s[ok], t[ok]), int(ok.sum())

    rho, npairs = rho_of(sim)
    if not math.isfinite(rho):
        return {"rho": rho, "p": float("nan"), "n_pairs": npairs}
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        r, _ = rho_of(sim[np.ix_(p, p)])
        if math.isfinite(r) and r >= rho - EPS:
            cnt += 1
    return {"rho": rho, "p": (1 + cnt) / (1 + n_perm), "n_pairs": npairs, "n_perm": n_perm}


# ---------------------------------------------------------------- multiple tests

def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values; nan counts as 1."""
    items = sorted(((1.0 if (p is None or not math.isfinite(p)) else p), k) for k, p in pvals.items())
    m = len(items)
    adj, run = {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


def verdicts(effects: dict, pvals: dict, alpha: float = ALPHA) -> dict:
    """{test: {effect, p, p_holm, verdict}}: supported only if the effect is positive and the Holm p is under alpha."""
    adj = holm(pvals)
    out = {}
    for k in pvals:
        e = effects[k]
        good = e is not None and math.isfinite(e) and e > 0 and adj[k] < alpha
        out[k] = {"effect": e, "p": pvals[k], "p_holm": adj[k], "verdict": "supported" if good else "not supported"}
    return out


# ---------------------------------------------------------------- descriptive: hubs, first hit

def hub_blocks(fragile_sets, min_sets: int) -> dict:
    """{block: number of fragile sets containing it} for blocks in at least `min_sets` sets."""
    count: dict[int, int] = {}
    for f in fragile_sets:
        for b in f:
            count[b] = count.get(b, 0) + 1
    return {b: c for b, c in sorted(count.items()) if c >= min_sets}


def first_fragile_hit(fragile: set, order, grid):
    """Smallest k in `grid` at which order[:k] contains a block of `fragile`; None if none does."""
    for k in sorted(grid):
        if fragile & set(order[:k]):
            return k
    return None


# ---------------------------------------------------------------- descriptive: shared directions

def class_means(x: np.ndarray, labels, n_classes: int) -> np.ndarray:
    """Per-class mean rows of x minus the grand mean of x; classes absent from `labels` are dropped. Computed in float64
    from the (fp16 or fp32) inputs: in fp32 the rounding noise of the means, against activations with huge entries,
    exceeds the 1e-6 singular-value cut and the rank comes out as the number of classes instead of at most classes - 1."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    grand = x.mean(0)
    rows = [x[labels == c].mean(0) - grand for c in range(n_classes) if (labels == c).any()]
    return np.stack(rows)


def row_basis(m: np.ndarray, tol: float = SV_TOL) -> np.ndarray:
    """Orthonormal basis of the row space of m (r x d) by SVD, dropping singular values below tol * the largest.
    A zero matrix has rank 0 (shape (0, d))."""
    m = np.asarray(m, dtype=np.float64)
    if m.size == 0:
        return np.zeros((0, m.shape[-1]))
    _, s, vt = np.linalg.svd(m, full_matrices=False)
    if s[0] <= 0:
        return np.zeros((0, m.shape[1]))
    return vt[s >= tol * s[0]]


def subspace_overlap(qa: np.ndarray, qb: np.ndarray) -> float:
    """||Qa Qb^T||_F^2 / min(rank_a, rank_b); nan if either basis is empty."""
    r = min(len(qa), len(qb))
    if r == 0:
        return float("nan")
    return float(np.sum((qa @ qb.T) ** 2) / r)


def project_out(m: np.ndarray, q: np.ndarray) -> np.ndarray:
    """m with the span of the orthonormal rows `q` removed. Linear, so it acts on class means, not on residuals."""
    return m if len(q) == 0 else m - (m @ q.T) @ q


def _onehot_means(x, labels, n_classes, grand):
    counts = np.bincount(labels, minlength=n_classes)
    h = np.zeros((n_classes, len(labels)), dtype=np.float64)
    h[labels, np.arange(len(labels))] = 1.0
    present = counts > 0
    return ((h @ x)[present] / counts[present, None]) - grand


def subspace_analysis(load_layer, tasks, answers: dict, n_classes: dict, unembed: dict | None, layers=LAYERS,
                      n_shuffle: int = N_SHUFFLE, seed: int = SEED) -> dict:
    """Overlap of every task pair's class-mean subspaces at every layer.

    load_layer(task, layer) -> (n, d) residuals. answers[task] = integer labels. unembed[task] = option-token
    unembedding rows (None skips the projected variant). Returns arrays over (layer, task, task):
    raw, raw_null, and (if unembed) proj, proj_null, where *_null is the mean over `n_shuffle` within-task label
    shuffles (the same shuffles at every layer, drawn from one generator of `seed`)."""
    n = len(tasks)
    rng = np.random.default_rng(seed)
    labs = {t: np.asarray(answers[t], dtype=int) for t in tasks}
    shuf = {t: [rng.permutation(labs[t]) for _ in range(n_shuffle)] for t in tasks}
    union = {}
    if unembed is not None:
        for i in range(n):
            for j in range(i + 1, n):
                union[i, j] = row_basis(np.vstack([unembed[tasks[i]], unembed[tasks[j]]]))
    shape = (len(layers), n, n)
    res = {k: np.full(shape, np.nan) for k in (("raw", "raw_null") + (("proj", "proj_null") if unembed is not None else ()))}
    for li, layer in enumerate(layers):
        xs = {t: np.asarray(load_layer(t, layer), dtype=np.float64) for t in tasks}
        grand = {t: xs[t].mean(0) for t in tasks}
        sums = {k: np.zeros((n, n)) for k in res if k.endswith("null")}
        cnts = {k: np.zeros((n, n)) for k in sums}
        for s in range(-1, n_shuffle):
            ms = [_onehot_means(xs[t], labs[t] if s < 0 else shuf[t][s], n_classes[t], grand[t]) for t in tasks]
            qs = [row_basis(m) for m in ms]
            kinds = [("raw", "raw_null")] + ([("proj", "proj_null")] if unembed is not None else [])
            for kind, nullk in kinds:
                for i in range(n):
                    for j in range(i + 1, n):
                        if kind == "raw":
                            v = subspace_overlap(qs[i], qs[j])
                        else:
                            u = union[i, j]
                            v = subspace_overlap(row_basis(project_out(ms[i], u)), row_basis(project_out(ms[j], u)))
                        if s < 0:
                            res[kind][li, i, j] = res[kind][li, j, i] = v
                        elif math.isfinite(v):
                            sums[nullk][i, j] += v
                            cnts[nullk][i, j] += 1
            del ms, qs
        for nullk in sums:
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(cnts[nullk] > 0, sums[nullk] / np.maximum(cnts[nullk], 1), np.nan)
            res[nullk][li] = np.where(np.isnan(mean), mean.T, mean)
    return res


# ---------------------------------------------------------------- JSON helper

def jsonable(x):
    """numpy to plain Python; nan and inf to None."""
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in (sorted(x) if isinstance(x, set) else x)]
    return x
