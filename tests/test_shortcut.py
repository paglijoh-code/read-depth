"""Bag-of-words must not predict the label from surface words (the real shortcut test)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from shortcut_check import check  # noqa: E402

from surgery.task.generator import generate  # noqa: E402


def test_bow_cannot_predict_label():
    r = check(generate(2000, 1))
    assert r["ok"], r
