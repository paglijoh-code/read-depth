"""Canary file and perplexity-text integrity (fast), plus model-guarded scoring tests."""
import math
import os
from collections import Counter

import pytest

from surgery.canaries import (CATEGORIES, CANARIES_PATH, PPL_PATH, load_canaries, load_ppl_text, perplexity,
                              score_canaries)

ITEMS = load_canaries()


def test_counts():
    assert len(ITEMS) == 500
    assert Counter(i["category"] for i in ITEMS) == {c: 100 for c in CATEGORIES}


def test_item_validity():
    assert len({i["id"] for i in ITEMS}) == 500
    for i in ITEMS:
        assert set(i) == {"id", "category", "prompt", "options", "answer", "source"}
        assert 2 <= len(i["options"]) <= 4
        assert len(set(i["options"])) == len(i["options"]), i["id"]
        assert isinstance(i["answer"], int) and 0 <= i["answer"] < len(i["options"])


def test_no_duplicate_items():
    keys = {(i["prompt"], tuple(sorted(i["options"]))) for i in ITEMS}
    assert len(keys) == 500
    for c in CATEGORIES:
        assert len({(i["prompt"], tuple(sorted(i["options"]))) for i in ITEMS if i["category"] == c}) == 100


def test_answer_positions_vary():
    for c in CATEGORIES:
        assert len({i["answer"] for i in ITEMS if i["category"] == c}) > 1, c


def test_ppl_text_hash():
    if not PPL_PATH.exists():
        pytest.skip(f"{PPL_PATH} absent; run scripts/build_canaries.py")
    assert len(load_ppl_text()) > 100_000  # also verifies PPL_SHA256


@pytest.mark.skipif(os.environ.get("SURGERY_MODEL_TESTS") != "1", reason="set SURGERY_MODEL_TESTS=1 to run model tests")
def test_model_scoring_and_ppl():
    from surgery.eval import load_model
    model, tok = load_model()
    sub = ITEMS[::50]
    r = score_canaries(model, tok, sub, batch_size=4)
    assert r["n"] == len(sub) and len(r["correct"]) == len(sub)
    assert 0.0 <= r["accuracy"] <= 1.0
    assert all(math.isfinite(v) and v <= 0 for s in r["scores"] for v in s)
    # batch size must not change the scores materially
    r2 = score_canaries(model, tok, sub, batch_size=1)
    assert r["pred"] == r2["pred"]
    p = perplexity(model, tok, load_ppl_text()[:3000], window=128, stride=64)
    assert math.isfinite(p["ppl"]) and p["ppl"] > 1 and p["n_tokens"] > 100


def test_eligible_at_exact_limit():
    from surgery.canaries import eligible
    cats = {"syntax": 0.63, "factual": 0.63, "reasoning": 0.63, "instruction": 0.63, "code": 0.63}
    got = {**cats, "factual": 0.58}  # 5 of 100 items: 5.000000000000004 in floats
    ok, drops = eligible({"accuracy": 0.634, "per_category": cats}, {"accuracy": 0.624, "per_category": got})
    assert ok and drops["factual"] == 5.0
