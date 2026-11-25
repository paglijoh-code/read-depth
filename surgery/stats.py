"""Classification statistics with template-cluster percentile bootstrap CIs.

Labels are integer column indices. Accuracy uses group sufficient statistics;
other callables (including macro-F1 and ECE) use a general resampling loop.
The latter is intended for modest classification datasets (3,000 rows / 10,000
resamples), where a runtime up to 60 seconds is acceptable.
"""

import math

import numpy as np


def _pair(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim != 1 or p.shape != y.shape or not y.size:
        raise ValueError("expected nonempty, equally sized label vectors")
    return y, p


def accuracy(y, p):
    y, p = _pair(y, p)
    return float(np.mean(y == p))


def macro_f1(y, p, n_classes=6):
    y, p = _pair(y, p)
    if (n_classes < 1 or y.dtype.kind not in "iu" or p.dtype.kind not in "iu"
            or np.any(y < 0) or np.any(p < 0)
            or np.any(y >= n_classes) or np.any(p >= n_classes)):
        raise ValueError("labels must be integer indices within n_classes")
    support = np.bincount(y, minlength=n_classes)
    predicted = np.bincount(p, minlength=n_classes)
    tp = np.bincount(y[y == p], minlength=n_classes)
    denom = support + predicted
    present = denom > 0
    return float(np.mean(2 * tp[present] / denom[present]))


def ece(y, probs, n_bins=10):
    """Equal-width [left, right) bins, with confidence 1 in the final bin."""
    y, probs = np.asarray(y), np.asarray(probs, dtype=float)
    if (y.ndim != 1 or not y.size or probs.ndim != 2
            or probs.shape[0] != y.size or probs.shape[1] == 0
            or not isinstance(n_bins, (int, np.integer)) or n_bins < 1):
        raise ValueError("expected labels, probability rows, and positive n_bins")
    if (not np.all(np.isfinite(probs)) or np.any(probs < 0)
            or np.any(probs > 1) or not np.allclose(probs.sum(axis=1), 1)):
        raise ValueError("probability rows must be finite and sum to one")
    confidence = probs.max(axis=1)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    correct = probs.argmax(axis=1) == y
    # Weighted bin error simplifies to abs(sum(correct) - sum(confidence))/n.
    return float(np.abs(np.bincount(bins, weights=correct, minlength=n_bins)
                        - np.bincount(bins, weights=confidence,
                                      minlength=n_bins)).sum() / y.size)


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    if (logits.ndim != 2 or not logits.shape[1]
            or not np.all(np.isfinite(logits))):
        raise ValueError("expected a finite matrix of logits")
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _inputs(groups, arrays, n_boot, alpha):
    groups = np.asarray(groups)
    arrays = tuple(np.asarray(a) for a in arrays)
    if (groups.ndim != 1 or not groups.size or not arrays
            or any(a.ndim == 0 or len(a) != len(groups) for a in arrays)):
        raise ValueError("groups and arrays must have the same nonzero row count")
    if (not isinstance(n_boot, (int, np.integer)) or n_boot < 1
            or not 0 < alpha < 1):
        raise ValueError("n_boot must be positive and 0 < alpha < 1")
    _, inverse = np.unique(groups, return_inverse=True)
    return inverse, arrays


def _distribution(metric_fn, inverse, arrays, n_boot, seed, paired=False):
    rng = np.random.default_rng(seed)
    n_groups = int(inverse.max()) + 1
    sizes = np.bincount(inverse)
    values = np.empty(n_boot)
    if metric_fn is accuracy:
        y, a = _pair(*arrays[:2])
        scores = (y == a).astype(float)
        if paired:
            _, b = _pair(y, arrays[2])
            scores = (y == b).astype(float) - scores
        sums = np.bincount(inverse, weights=scores)
        # Bounded memory; no repeated row arrays or Python loop per replicate.
        for start in range(0, n_boot, 256):
            counts = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups),
                                     size=min(256, n_boot - start))
            values[start:start + len(counts)] = (
                (counts * sums).sum(axis=1) / (counts * sizes).sum(axis=1))
    else:
        order = np.argsort(inverse, kind="stable")
        rows = np.split(order, np.cumsum(sizes)[:-1])
        for i in range(n_boot):
            draws = rng.integers(n_groups, size=n_groups)
            idx = np.concatenate([rows[g] for g in draws])
            if paired:
                y, a, b = (arr[idx] for arr in arrays)
                values[i] = metric_fn(y, b) - metric_fn(y, a)
            else:
                values[i] = metric_fn(*(arr[idx] for arr in arrays))
    return values


def _interval(point, values, n_groups, alpha):
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return dict(point=float(point), lo=float(lo), hi=float(hi),
                half_width=float((hi - lo) / 2), n_groups=int(n_groups),
                n_boot=len(values))


def grouped_bootstrap(metric_fn, groups, *arrays, n_boot=10_000, seed=0,
                      alpha=0.05):
    """Draw G clusters uniformly with replacement; retain every drawn row.

    Unequal cluster sizes thus produce variable-size replicate datasets.
    Pass ``accuracy`` directly to use its vectorized sufficient-statistic path.
    """
    inverse, arrays = _inputs(groups, arrays, n_boot, alpha)
    point = metric_fn(*arrays)
    values = _distribution(metric_fn, inverse, arrays, n_boot, seed)
    return _interval(point, values, inverse.max() + 1, alpha)


def paired_grouped_bootstrap(metric_fn, groups, y, pred_a, pred_b,
                             n_boot=10_000, seed=0, alpha=0.05):
    """CI for B minus A using shared clusters; zero draws are not opposite-sign.

    At a zero point estimate p_two_sided is defined as 1.
    """
    inverse, arrays = _inputs(groups, (y, pred_a, pred_b), n_boot, alpha)
    y, a, b = arrays
    point = metric_fn(y, b) - metric_fn(y, a)
    values = _distribution(metric_fn, inverse, arrays, n_boot, seed, paired=True)
    result = _interval(point, values, inverse.max() + 1, alpha)
    result["p_two_sided"] = (1.0 if point == 0 else
                             float(min(1, 2 * np.mean(values * np.sign(point) < 0))))
    return result


def mcnemar(correct_a, correct_b):
    a, b = _pair(correct_a, correct_b)
    a, b = a.astype(bool), b.astype(bool)
    right_wrong = int(np.sum(a & ~b))
    wrong_right = int(np.sum(~a & b))
    n = right_wrong + wrong_right
    tail = sum(math.comb(n, k) for k in range(min(right_wrong, wrong_right) + 1))
    return dict(b=right_wrong, c=wrong_right,
                p_exact=min(1.0, 2 * tail / (1 << n)))


def summarize(records_y, logits, groups, seed=0, n_boot=10_000):
    """Summarize integer true labels and logits in the same class-column order."""
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    kwargs = dict(seed=seed, n_boot=n_boot)
    return dict(
        accuracy=grouped_bootstrap(accuracy, groups, records_y, pred, **kwargs),
        macro_f1=grouped_bootstrap(macro_f1, groups, records_y, pred, **kwargs),
        ece=grouped_bootstrap(ece, groups, records_y, probs, **kwargs),
        n=len(records_y), n_groups=len(np.unique(groups)))
