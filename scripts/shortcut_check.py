"""Bag-of-words shortcut check: can surface words predict the label?

usage: shortcut_check.py FILE [--generate N --seed S]
Unigram+bigram counts + logistic regression, stratified k-fold CV accuracy. Exits 1 if
CV accuracy > majority-class rate + MARGIN.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from surgery.task.generator import generate, read_jsonl, write_jsonl  # noqa: E402

MARGIN = 0.15


def check(records: list[dict], folds: int | None = None, seed: int = 0) -> dict:
    X = [r["prompt"] for r in records]
    y = [r["label"] for r in records]
    folds = folds or (5 if len(y) <= 100 else 5)
    pipe = make_pipeline(CountVectorizer(ngram_range=(1, 2), min_df=2),
                         LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    acc = float(np.mean(cross_val_score(pipe, X, y, cv=cv)))
    maj = max(Counter(y).values()) / len(y)
    return {"cv_accuracy": acc, "majority_rate": maj, "margin": MARGIN, "ok": acc <= maj + MARGIN}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--generate", type=int, default=None, help="generate N examples into FILE first")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.generate:
        write_jsonl(generate(a.generate, a.seed), a.file)
    r = check(read_jsonl(a.file), seed=a.seed)
    print(f"shortcut check: cv_accuracy={r['cv_accuracy']:.3f} majority={r['majority_rate']:.3f} "
          f"limit={r['majority_rate'] + MARGIN:.3f}")
    if not r["ok"]:
        print("FAIL: bag-of-words predicts the label above the limit", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
