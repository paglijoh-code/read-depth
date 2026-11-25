"""Eval tests. Fast ones need no model; model tests skip unless SURGERY_MODEL_TESTS=1."""
import os
from types import SimpleNamespace

import pytest
import torch

from surgery import eval as ev
from surgery.task.generator import LABELS


def test_parse_dtype():
    assert ev.parse_dtype("fp32") is torch.float32
    assert ev.parse_dtype("bf16") is torch.bfloat16
    assert ev.parse_dtype(torch.float16) is torch.float16
    with pytest.raises(ValueError):
        ev.parse_dtype("fp8")
    assert ev.dtype_name(torch.bfloat16) == "bf16"


def test_evaluate_result_keys(monkeypatch):
    recs = [{"prompt": "a", "label": LABELS[i % 6], "wrong_label": LABELS[(i + 1) % 6]} for i in range(5)]
    fake = torch.eye(6)[[0, 1, 2, 3, 0]] * 5

    class Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(ev, "apply_spec", lambda m, s: Ctx())
    monkeypatch.setattr(ev, "label_logits", lambda *a, **k: fake)
    model = SimpleNamespace(config=SimpleNamespace(_name_or_path="x/y"), dtype=torch.bfloat16)
    res = ev.evaluate(model, None, recs, ev.NO_SPEC)
    assert res["model"] == "x/y" and res["dtype"] == "bf16"
    assert res["pred"] == [0, 1, 2, 3, 0]
    assert len(res["logits"]) == 5 and len(res["logits"][0]) == 6
    assert res["accuracy"] == pytest.approx(4 / 5)
    for k in ("n", "mean_logit_diff", "per_label_accuracy", "spec", "spec_hash", "data_path"):
        assert k in res


@pytest.mark.skipif(os.environ.get("SURGERY_MODEL_TESTS") != "1",
                    reason="set SURGERY_MODEL_TESTS=1 to run model tests")
def test_two_passes_identical():
    from surgery.task.generator import read_jsonl
    model, tok = ev.load_model()
    recs = read_jsonl("var/data/eval500.jsonl")[:16]
    a = ev.evaluate(model, tok, recs, ev.NO_SPEC)
    b = ev.evaluate(model, tok, recs, ev.NO_SPEC)
    assert a["pred"] == b["pred"]
    assert a["dtype"] == "fp32" and a["model"] == ev.MODEL_NAME
    assert len(a["logits"]) == 16
