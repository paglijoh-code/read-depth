"""Hook tests on the real Qwen3-0.6B. Skipped unless SURGERY_MODEL_TESTS=1."""
import os

import pytest

from surgery.hooks import apply_spec, spec_hash

pytestmark = pytest.mark.skipif(os.environ.get("SURGERY_MODEL_TESTS") != "1",
                                reason="set SURGERY_MODEL_TESTS=1 to run model tests")

PROMPTS = ["Task: fix the parser.\nTests:\n- test_a: fail", "Task: add a cache.\nFiles:\n- a.py: read=no"]


def test_spec_hash_stable():
    assert spec_hash({"op": "skip", "layers": [1, 2]}) == spec_hash({"layers": [1, 2], "op": "skip"})
    assert spec_hash({"op": "skip", "layers": [1]}) != spec_hash({"op": "skip", "layers": [2]})


@pytest.fixture(scope="module")
def mt():
    from surgery.eval import load_model
    return load_model()


def _logits(model, tok):
    import torch
    enc = tok(PROMPTS, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        return model(**enc).logits[:, -1, :].float().cpu()


def test_unknown_op_rejected(mt):
    model, _ = mt
    with pytest.raises(ValueError):
        apply_spec(model, {"op": "ablate", "layers": [1]})
    with pytest.raises(ValueError):
        apply_spec(model, {"op": "skip", "layers": [99]})


def test_skip_empty_is_exact(mt):
    import torch
    model, tok = mt
    base = _logits(model, tok)
    with apply_spec(model, {"op": "skip", "layers": []}):
        hooked = _logits(model, tok)
    assert torch.equal(base, hooked)


def test_skip_changes_logits_and_removes_cleanly(mt):
    import torch
    model, tok = mt
    base = _logits(model, tok)
    with apply_spec(model, {"op": "skip", "layers": [5]}):
        changed = _logits(model, tok)
    assert not torch.equal(base, changed)
    assert torch.equal(base, _logits(model, tok))  # hooks removed


class _Identity(__import__("torch").nn.Module):
    def forward(self, hidden_states, *a, **k):
        return hidden_states


@pytest.mark.parametrize("layers", [[3], [0, 10]])
def test_skip_equals_layer_removed(mt, layers):
    import torch
    model, tok = mt
    with apply_spec(model, {"op": "skip", "layers": layers}):
        hooked = _logits(model, tok)
    orig = {i: model.model.layers[i] for i in layers}
    try:
        for i in layers:
            model.model.layers[i] = _Identity()
        manual = _logits(model, tok)
    finally:
        for i, l in orig.items():
            model.model.layers[i] = l
    assert torch.allclose(hooked, manual, atol=1e-4, rtol=1e-4)
