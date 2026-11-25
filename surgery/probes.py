"""Linear probes on the last-token residual, one per layer, grouped by template.

Each layer gets StandardScaler + LogisticRegression(C=1.0, max_iter=2000) under
GroupKFold(5) on template ids, so no template appears in both train and test.
The label probe reports out-of-fold accuracy; the correctness probe reports
out-of-fold AUROC. The control refits with targets permuted within the training
folds and scores against the true test targets (chance if the probe reads
nothing but fold structure). CIs are grouped bootstraps over the out-of-fold
predictions.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from surgery import stats

KINDS = ("label", "correct")


def _auroc(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return 0.5
    return float(roc_auc_score(y, s))


def _oof(X, target, groups, kind, seed, permute, n_splits=5):
    """Out-of-fold predictions: class index (label) or P(correct) (correct)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(target), dtype=float if kind == "correct" else int)
    for train, test in GroupKFold(n_splits=n_splits).split(X, target, groups):
        t = target[train]
        if permute:
            t = rng.permutation(t)
        classes = np.unique(t)
        if len(classes) == 1:  # degenerate fold: constant prediction
            out[test] = float(classes[0]) if kind == "correct" else classes[0]
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=2000))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(X[train], t)
        if kind == "correct":
            out[test] = clf.predict_proba(X[test])[:, list(clf.classes_).index(1)]
        else:
            out[test] = clf.predict(X[test])
    return out


def _score(kind, target, pred, groups, n_boot, seed):
    metric = stats.accuracy if kind == "label" else _auroc
    ci = stats.grouped_bootstrap(metric, groups, target, pred,
                                 n_boot=n_boot, seed=seed)
    return ci["point"], ci["lo"], ci["hi"]


def _probe_layer(Xl, target, groups, kind, seed, n_boot):
    pred = _oof(Xl, target, groups, kind, seed, permute=False)
    ctrl = _oof(Xl, target, groups, kind, seed, permute=True)
    m, lo, hi = _score(kind, target, pred, groups, n_boot, seed)
    cm, clo, chi = _score(kind, target, ctrl, groups, n_boot, seed)
    return dict(metric=m, lo=lo, hi=hi, control=cm, control_lo=clo,
                control_hi=chi)


def probe_by_layer(X, target, groups, kind="label", seed=0, n_boot=500,
                   n_jobs=1):
    """X (N, L+1, d) -> list of L+1 dicts {layer, metric, lo, hi, control,
    control_lo, control_hi}. target: label indices (label) or 0/1 correctness
    (correct). groups: template ids."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    target, groups = np.asarray(target), np.asarray(groups)
    if X.ndim != 3 or len(X) != len(target) or len(groups) != len(target):
        raise ValueError("X must be (N, L+1, d) matching target and groups")
    n_layers = X.shape[1]

    def one(i):
        Xl = np.asarray(X[:, i], dtype=np.float32)
        return dict(layer=i, **_probe_layer(Xl, target, groups, kind, seed + i,
                                            n_boot))

    if n_jobs == 1:
        return [one(i) for i in range(n_layers)]
    from joblib import Parallel, delayed
    return list(Parallel(n_jobs=n_jobs)(delayed(one)(i) for i in range(n_layers)))
