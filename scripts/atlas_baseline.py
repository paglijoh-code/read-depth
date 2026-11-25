#!/usr/bin/env python3
"""Bag-of-words baseline for the atlas tasks (docs/ATLAS-phase0.md, Brief A, choice 6).

    uv run python scripts/atlas_baseline.py [--data DIR] [--out FILE] [--tasks T ...]

TF-IDF (word 1-2 grams + character 3-5 grams, as scripts/routing_rivals.py) with logistic regression on the user text
only. C is chosen on lrsel (fit on train), then the model is refit on train+lrsel and scored on selection, for the
whole task and for each rung. A task is surface-leaky if BoW >= 0.90 or within 0.05 of the probe peak; the probe peak is
filled later by the report (`probe_peak` is null here, so only the 0.90 rule is applied until then).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import FeatureUnion, make_pipeline  # noqa: E402

from surgery.atlas import spec  # noqa: E402

CS = [0.1, 1.0, 10.0, 100.0]
LEAKY_ABS, LEAKY_GAP = 0.90, 0.05


def model(C: float):
    feats = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)),
                          ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))])
    return make_pipeline(feats, LogisticRegression(C=C, max_iter=2000))


def surface_leaky(bow: float, probe_peak: float | None) -> bool:
    return bow >= LEAKY_ABS or (probe_peak is not None and bow >= probe_peak - LEAKY_GAP)


def _acc(m, recs: list[dict]) -> float:
    pred = m.predict([r["prompt"] for r in recs])
    return float(np.mean(pred == np.array([r["answer"] for r in recs])))


def baseline(task: str, data: Path) -> dict:
    tr, lr, sel = (spec.read_records(task, s, data) for s in ("train", "lrsel", "selection"))
    y_tr = [r["answer"] for r in tr]
    if len(set(y_tr)) < 2:
        raise ValueError(f"{task}: one answer in train")
    scores = {}
    for C in CS:
        m = model(C).fit([r["prompt"] for r in tr], y_tr)
        scores[C] = _acc(m, lr)
    best = max(CS, key=lambda c: (scores[c], -c))     # ties: the smaller C
    full = tr + lr
    m = model(best).fit([r["prompt"] for r in full], [r["answer"] for r in full])
    out = {"task": task, "number": spec.TASKS[task].number, "C": best, "lrsel_by_C": scores,
           "chance": spec.chance([r["answer"] for r in sel]), "bow": _acc(m, sel), "n_selection": len(sel),
           "rungs": {}, "probe_peak": None}
    for rung in sorted({r["rung"] for r in sel if r["rung"]}):
        sub = [r for r in sel if r["rung"] == rung]
        out["rungs"][rung] = {"n": len(sub), "chance": spec.chance([r["answer"] for r in sub]), "bow": _acc(m, sub)}
    out["surface_leaky"] = surface_leaky(out["bow"], None)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=spec.DATA_DIR)
    ap.add_argument("--out", type=Path, default=Path("var/results/atlas/baseline.json"))
    ap.add_argument("--tasks", nargs="*", default=None)
    a = ap.parse_args(argv)
    names = a.tasks or list(spec.PHASE1_TASKS)
    res = {"rule": f"surface-leaky if BoW >= {LEAKY_ABS} or within {LEAKY_GAP} of the probe peak (peak filled by the report)",
           "tasks": {}}
    for t in names:
        res["tasks"][t] = r = baseline(t, a.data)
        print(f"{r['number']:>2} {t:<16} bow={r['bow']:.3f} chance={r['chance']:.3f} C={r['C']} leaky={r['surface_leaky']} "
              + " ".join(f"{k}={v['bow']:.3f}" for k, v in r["rungs"].items()), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
