"""Offline ablation checks against explicit residual and projection calculations."""
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from surgery import stats
from surgery.hooks import apply_spec, spec_hash, validate_spec
from surgery.subset import sweep_indices
from surgery.task.generator import LABELS


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=32, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16)
    model = Qwen3ForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    return model


@pytest.fixture
def inputs():
    return dict(input_ids=torch.tensor([[0, 0, 3, 7, 9], [1, 2, 4, 5, 6]]),
                attention_mask=torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]))


@pytest.fixture
def means():
    generator = torch.Generator().manual_seed(12)
    # Deliberately double precision: hooks must convert to the model activation dtype.
    return {"attn": torch.randn(2, 64, generator=generator, dtype=torch.float64) / 10,
            "mlp": torch.randn(2, 64, generator=generator, dtype=torch.float64) / 10,
            "heads": torch.randn(2, 4, 16, generator=generator, dtype=torch.float64) / 10}


@torch.no_grad()
def _logits(model, inputs):
    return model(**inputs, use_cache=False).logits


@torch.no_grad()
def _manual_residual_forward(model, inputs, site=None, layer_index=None, mean=None, skip=()):
    hidden = model.model.embed_tokens(inputs["input_ids"])
    length = hidden.shape[1]
    positions = torch.arange(length).unsqueeze(0)
    rotary = model.model.rotary_emb(hidden, positions)
    allowed = torch.ones(length, length, dtype=torch.bool).tril()[None, None]
    allowed = allowed & inputs["attention_mask"][:, None, None, :].bool()
    mask = torch.zeros_like(allowed, dtype=hidden.dtype).masked_fill(
        ~allowed, torch.finfo(hidden.dtype).min)
    for index, layer in enumerate(model.model.layers):
        if index in skip:
            continue
        attn, _ = layer.self_attn(layer.input_layernorm(hidden),
                                  position_embeddings=rotary, attention_mask=mask)
        if site == "attn" and index == layer_index:
            attn = torch.zeros_like(attn) if mean is None else mean.to(attn).expand_as(attn)
        hidden = hidden + attn
        mlp = layer.mlp(layer.post_attention_layernorm(hidden))
        if site == "mlp" and index == layer_index:
            mlp = torch.zeros_like(mlp) if mean is None else mean.to(mlp).expand_as(mlp)
        hidden = hidden + mlp
    return model.lm_head(model.model.norm(hidden))


@pytest.mark.parametrize("site", ["attn", "mlp"])
@pytest.mark.parametrize("mode", ["zero", "mean"])
@pytest.mark.parametrize("layer", [0, 1])
def test_sublayer_matches_explicit_residual(tiny, inputs, means, site, mode, layer):
    baseline = _logits(tiny, inputs)
    # Check the independently built forward before using it as the reference.
    torch.testing.assert_close(baseline, _manual_residual_forward(tiny, inputs))
    expected = _manual_residual_forward(
        tiny, inputs, site, layer, means[site][layer] if mode == "mean" else None)
    spec = dict(op="ablate", site=site, mode=mode, layers=[layer])
    with apply_spec(tiny, spec, means):
        actual = _logits(tiny, inputs)
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, baseline)
    assert torch.equal(_logits(tiny, inputs), baseline)


class _ManualProjection(torch.nn.Module):
    """Remove selected column contributions, then add the means' projections."""
    def __init__(self, projection, heads, means):
        super().__init__()
        weight = projection.weight.detach().clone()
        bias = torch.zeros(weight.shape[0])
        for head in heads:
            span = slice(head * 16, (head + 1) * 16)
            if means is not None:
                bias += F.linear(means[head].to(weight), weight[:, span])
            weight[:, span] = 0
        if projection.bias is not None:
            bias += projection.bias.detach()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, value):
        return F.linear(value, self.weight, self.bias)


@pytest.mark.parametrize("mode", ["zero", "mean"])
@pytest.mark.parametrize("op,entries,heads", [
    ("ablate_heads", [[0, 1], [0, 3], [0, 1]], [1, 3]),
    ("ablate_kv", [[0, 1]], [2, 3]),
    ("ablate_kv", [[0, 0]], [0, 1]),
])
def test_head_projection_and_untouched_slices(tiny, inputs, means, mode, op, entries, heads):
    projection = tiny.model.layers[0].self_attn.o_proj
    before, after = [], []
    first = projection.register_forward_pre_hook(lambda m, args: before.append(args[0].clone()))
    spec = dict(op=op, mode=mode, **{"heads" if op == "ablate_heads" else "groups": entries})
    try:
        with apply_spec(tiny, spec, means):
            last = projection.register_forward_pre_hook(lambda m, args: after.append(args[0].clone()))
            try:
                actual = _logits(tiny, inputs)
            finally:
                last.remove()
    finally:
        first.remove()
    for head in range(4):
        span = slice(head * 16, (head + 1) * 16)
        if head not in heads:
            assert torch.equal(before[0][..., span], after[0][..., span])
        elif mode == "zero":
            assert torch.count_nonzero(after[0][..., span]) == 0
        else:
            torch.testing.assert_close(after[0][..., span],
                                       means["heads"][0, head].to(after[0]).expand_as(after[0][..., span]))
    tiny.model.layers[0].self_attn.o_proj = _ManualProjection(
        projection, heads, means["heads"][0] if mode == "mean" else None)
    try:
        expected = _logits(tiny, inputs)
    finally:
        tiny.model.layers[0].self_attn.o_proj = projection
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_heads_in_multiple_layers_and_keyword_input(tiny, means):
    spec = dict(op="ablate_heads", mode="mean", heads=[[0, 0], [1, 3]])
    value = torch.randn(2, 3, 64)
    original = value.clone()
    with apply_spec(tiny, spec, means):
        for index, head in [(0, 0), (1, 3)]:
            projection = tiny.model.layers[index].self_attn.o_proj
            expected = value.clone()
            expected[..., head * 16:(head + 1) * 16] = means["heads"][index, head].float()
            torch.testing.assert_close(projection(input=value), F.linear(expected, projection.weight))
    assert torch.equal(value, original)


def test_skip_and_hash_unchanged(tiny, inputs):
    spec = {"op": "skip", "layers": [1, 1]}
    canon = b'{"layers":[1,1],"op":"skip"}'
    assert spec_hash(spec) == hashlib.sha256(canon).hexdigest()[:12]
    assert spec_hash(spec) == spec_hash({"layers": [1, 1], "op": "skip"})
    base = _logits(tiny, inputs)
    with apply_spec(tiny, {"op": "skip", "layers": []}):
        assert torch.equal(_logits(tiny, inputs), base)
    with apply_spec(tiny, spec):
        torch.testing.assert_close(_logits(tiny, inputs),
                                   _manual_residual_forward(tiny, inputs, skip=[1]))
    assert torch.equal(_logits(tiny, inputs), base)


@pytest.mark.parametrize("spec", [
    {"op": "unknown"},
    {"op": "ablate", "site": "attn", "layers": [0]},
    {"op": "ablate", "site": "norm", "mode": "zero", "layers": [0]},
    {"op": "ablate", "site": "mlp", "mode": "zero", "layers": [-1]},
    {"op": "ablate", "site": "mlp", "mode": "zero", "layers": [2]},
    {"op": "ablate", "site": "mlp", "mode": "zero", "layers": [True]},
    {"op": "ablate_heads", "mode": "zero", "heads": [[0, 4]]},
    {"op": "ablate_heads", "mode": "zero", "heads": [[0]]},
    {"op": "ablate_heads", "mode": "zero", "heads": [[0, -1]]},
    {"op": "ablate_heads", "mode": "zero", "heads": [0]},
    {"op": "ablate_kv", "mode": "zero", "groups": [[0, 2]]},
    {"op": "ablate_kv", "mode": "median", "groups": [[0, 0]]},
    {"op": "ablate_kv", "mode": "zero"},
])
def test_validation(tiny, spec):
    with pytest.raises(ValueError):
        validate_spec(spec, 2, 4, 2)
    with pytest.raises(ValueError):
        apply_spec(tiny, spec)
    assert not tiny.model.layers[0].self_attn._forward_hooks
    assert not tiny.model.layers[0].self_attn.o_proj._forward_pre_hooks


@pytest.mark.parametrize("spec", [
    dict(op="ablate", site="attn", mode="mean", layers=[0]),
    dict(op="ablate_heads", mode="mean", heads=[[0, 0]]),
    dict(op="ablate_kv", mode="mean", groups=[[0, 0]]),
])
def test_mean_required_and_checked(tiny, spec):
    for bad in [None, {}, {"attn": torch.zeros(2, 63), "heads": torch.zeros(2, 4, 15)}]:
        with pytest.raises(ValueError, match="mean"):
            apply_spec(tiny, spec, bad)


def test_context_exception_removes_hooks(tiny, inputs):
    base = _logits(tiny, inputs)
    with pytest.raises(RuntimeError, match="deliberate"):
        with apply_spec(tiny, dict(op="ablate", site="attn", mode="zero", layers=[0, 1])):
            raise RuntimeError("deliberate")
    assert torch.equal(_logits(tiny, inputs), base)


def test_seeded_subset():
    assert sweep_indices(1000) == sorted(random.Random(0).sample(range(1000), 500))
    assert sweep_indices(30, 16, 7) == sorted(random.Random(7).sample(range(30), 16))
    assert sweep_indices(3, 3) == [0, 1, 2]
    with pytest.raises(ValueError):
        sweep_indices(3, 4)


def _sweep_module():
    path = Path(__file__).resolve().parents[1] / "scripts/ablation_sweep.py"
    module_spec = importlib.util.spec_from_file_location("ablation_sweep", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_accuracy_lost_ci_has_baseline_minus_ablated_sign():
    sweep = _sweep_module()
    y = [0, 0, 0, 0]
    base_pred = [0, 0, 0, 1]
    logits = torch.tensor([[2., 0.], [0., 2.], [0., 2.], [0., 2.]])
    groups = [0, 0, 1, 1]
    row = sweep._row(dict(op="ablate", site="attn", mode="zero", layers=[0]),
                     logits, y, groups, base_pred, np.zeros(4), 0.1)
    direct = stats.paired_grouped_bootstrap(stats.accuracy, groups, y,
                                           logits.argmax(-1).tolist(), base_pred)
    assert row["accuracy_lost"] == 0.5
    for key in ["point", "lo", "hi", "half_width", "p_two_sided"]:
        assert row["accuracy_lost_ci"][key] == pytest.approx(direct[key])


def test_sweep_tiny_end_to_end_resume_heads_and_smoke(tiny, means, tmp_path, monkeypatch):
    sweep = _sweep_module()
    monkeypatch.chdir(tmp_path)
    data = Path("var/data/m2/rule_s0_discovery.jsonl")
    data.parent.mkdir(parents=True)
    records = [dict(id=f"r{i}", template_id=f"t{i // 2}", label=LABELS[i % 6], prompt=str(i))
               for i in range(20)]
    data.write_text("".join(json.dumps(r) + "\n" for r in records))
    store = Path("var/acts/tiny_rule_s0_discovery")
    store.mkdir(parents=True)
    meta = dict(model="test/tiny", variant="rule", split="discovery", data_path=str(data),
                n=20, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
                ids=[r["id"] for r in records], template_ids=[r["template_id"] for r in records],
                y=[i % 6 for i in range(20)])
    (store / "meta.json").write_text(json.dumps(meta))
    torch.save(means, store / "means.pt")
    calls, frees = [], []

    def load(name, dtype):
        assert name == "test/tiny" and dtype == "fp32"
        return tiny, None

    @torch.no_grad()
    def label_logits(model, tok, prompts, batch_size):
        calls.append(list(prompts))
        # Real tiny-model inference, no tokenizer or weights downloads.
        ids = torch.tensor([[1, 2, 3, int(p) + 4] for p in prompts])
        return torch.cat([model(input_ids=batch, logits_to_keep=1, use_cache=False).logits[:, -1, :6]
                          for batch in ids.split(batch_size)]).float().cpu()

    monkeypatch.setattr(sweep, "load_model", load)
    monkeypatch.setattr(sweep, "label_logits", label_logits)
    monkeypatch.setattr(sweep, "_free", lambda m: frees.append(m))
    args = ["--model", "test/tiny", "--variant", "rule", "--n", "12", "--batch-size", "4",
            "--sites", "attn"]
    sweep.main(args)
    path = Path("var/results/m3/ablation_tiny_rule.json")
    result = json.loads(path.read_text())
    assert result["indices"] == sweep_indices(20, 12)
    assert result["ids"] == [records[i]["id"] for i in result["indices"]]
    assert len(result["rows"]) == 4
    assert len(calls) == len(frees) == 5
    baseline = result["baseline"]["accuracy"]["point"]
    for row in result["rows"]:
        assert row["accuracy_lost"] == pytest.approx(baseline - row["accuracy"])
        assert row["seconds"] >= 0
        assert row["accuracy_ci"]["lo"] <= row["accuracy"] <= row["accuracy_ci"]["hi"]
        assert np.isfinite(row["mean_correct_label_log_prob_drop"])
    original_rows = result["rows"]
    sweep.main(args)
    assert len(calls) == len(frees) == 6  # Only baseline is rerun.
    assert json.loads(path.read_text())["rows"] == original_rows
    with pytest.raises(ValueError, match="incompatible"):
        sweep.main(args + ["--n", "10"])

    head_file = tmp_path / "heads.json"
    head_file.write_text(json.dumps([[0, 0], [1, 3], [0, 0]]))
    sweep.main(args + ["--heads-file", str(head_file)])
    head_result = json.loads(Path("var/results/m3/ablation_tiny_rule_heads.json").read_text())
    assert len(head_result["rows"]) == 2
    assert all(row["spec"]["mode"] == "mean" for row in head_result["rows"])
    assert len(calls) == len(frees) == 9
    sweep.main(args + ["--smoke", "--modes", "zero"])
    smoke = json.loads(Path("var/results/m3/ablation_tiny_rule_smoke.json").read_text())
    assert smoke["n"] == 16 and len(smoke["rows"]) == 2
    assert json.loads(path.read_text())["rows"] == original_rows
    assert not list(path.parent.glob("*.tmp"))
