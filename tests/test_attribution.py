"""Attribution checks on a randomly initialized tiny Qwen3, entirely offline."""
import json

import pytest
import torch
import torch.nn.functional as F
from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM

from surgery.attribution import head_attribution
from surgery.prompt import build_prompt, label_token_ids
from surgery.task.generator import LABELS


class TinyTokenizer:
    """Only the shared prompt/scoring interface; no downloaded tokenizer needed."""
    padding_side = "left"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return messages[-1]["content"]

    def encode(self, text, add_special_tokens=False):
        return [LABELS.index(text) + 1]

    def __call__(self, prompts, return_tensors="pt", padding=True):
        sequences = [[8 + ord(c) % 23 for c in p] + [7] for p in prompts]
        length = max(map(len, sequences))
        ids = [[0] * (length - len(s)) + s for s in sequences]
        mask = [[0] * (length - len(s)) + [1] * len(s) for s in sequences]
        return BatchEncoding({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)})


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        pad_token_id=0,
    )).double().eval()
    tok = TinyTokenizer()
    records = [{"id": i, "template_id": str(i), "prompt": prompt, "label": LABELS[i]}
               for i, prompt in enumerate(["a", "bc", "defg"])]
    means = torch.randn(2, 4, 16, dtype=torch.float64) * 0.02
    return model, tok, records, means


@torch.no_grad()
def mean_ablation_loss(model, tok, records, means, layer=None, head=None, epsilon=0.0):
    enc = tok([build_prompt(tok, r["prompt"]) for r in records], padding=True, return_tensors="pt")
    handles = []
    if layer is not None:
        def ablate(module, args):
            z = args[0].clone().reshape(*args[0].shape[:2], 4, 16)
            z[:, :, head] += (epsilon * enc["attention_mask"][..., None]
                             * (means[layer, head] - z[:, :, head]))
            return (z.flatten(-2),)
        handles.append(model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(ablate))
    try:
        logits = model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1, label_token_ids(tok)]
        y = torch.tensor([LABELS.index(r["label"]) for r in records])
        return F.cross_entropy(logits, y).item()
    finally:
        for handle in handles:
            handle.remove()


def test_each_head_matches_small_mean_ablation(tiny):
    model, tok, records, means = tiny
    scores = head_attribution(model, tok, records, means, batch_size=3)["heads"]
    baseline = mean_ablation_loss(model, tok, records, means)
    epsilon = 1e-3
    # Ablate each head separately in an actual forward, moving its mean toward z
    # by epsilon. This checks sign, NLL normalization, head layout, and the graph
    # through later attention layers, rather than repeating attribution's code.
    for layer in range(2):
        for head in range(4):
            loss = mean_ablation_loss(model, tok, records, means, layer, head, epsilon)
            actual = loss - baseline
            predicted = scores[layer, head].item() * epsilon
            assert actual * predicted > 0
            assert predicted == pytest.approx(actual, rel=1e-2, abs=1e-10)


def test_batching_padding_kv_and_parameter_grads(tiny):
    model, tok, records, means = tiny
    model.float()
    means = means.float()
    # Exercise restoration of mixed parameter flags and module training flags.
    next(model.parameters()).requires_grad_(False)
    model.train()
    model.model.layers[0].mlp.eval()
    flags = [p.requires_grad for p in model.parameters()]
    training = [m.training for m in model.modules()]
    frozen_passes = []

    def assert_frozen(module, args):
        assert not any(p.requires_grad for p in model.parameters())
        frozen_passes.append(True)

    handle = model.register_forward_pre_hook(assert_frozen)
    try:
        # The function must enable grad even inside a caller's no_grad context.
        with torch.no_grad():
            batched = head_attribution(model, tok, records, means, batch_size=2)
        single = head_attribution(model, tok, records, means, batch_size=1)
    finally:
        handle.remove()
    assert len(frozen_passes) == 5
    assert batched["heads"].shape == (2, 4)
    assert batched["kv"].shape == (2, 2)
    assert batched["heads"].device.type == "cpu"
    assert batched["heads"].dtype == torch.float32
    torch.testing.assert_close(batched["heads"], single["heads"], atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(batched["kv"], batched["heads"].reshape(2, 2, 2).sum(-1))
    assert [p.requires_grad for p in model.parameters()] == flags
    assert [m.training for m in model.modules()] == training
    assert all(p.grad is None for p in model.parameters())
    assert all(not block.self_attn.o_proj._forward_pre_hooks for block in model.model.layers)


def test_cleanup_after_failure_preserves_existing_gradients(tiny, monkeypatch):
    model, tok, records, means = tiny
    param = next(model.parameters())
    param.grad = torch.ones_like(param)
    old_grad = param.grad
    model.train()

    def fail(*args, **kwargs):
        raise RuntimeError("failed forward")

    monkeypatch.setattr(model.model.layers[1].mlp, "forward", fail)
    with pytest.raises(RuntimeError, match="failed forward"):
        head_attribution(model, tok, records, means)
    assert model.training
    assert all(p.requires_grad for p in model.parameters())
    assert param.grad is old_grad
    assert torch.equal(param.grad, torch.ones_like(param))
    assert all(not block.self_attn.o_proj._forward_pre_hooks for block in model.model.layers)


def test_invalid_inputs(tiny):
    model, tok, records, means = tiny
    with pytest.raises(ValueError, match="batch_size"):
        head_attribution(model, tok, records, means, batch_size=0)
    with pytest.raises(ValueError, match="empty"):
        head_attribution(model, tok, [], means)
    with pytest.raises(ValueError, match="shape"):
        head_attribution(model, tok, records, means[:, :1])


def test_cli_writes_ranked_heads_and_companion_file(tiny, tmp_path, monkeypatch):
    from scripts import attribute_heads as cli

    model, tok, records, means = tiny
    monkeypatch.chdir(tmp_path)
    store = tmp_path / "var/acts/tiny_demo_s0_discovery"
    store.mkdir(parents=True)
    torch.save({"heads": means.float()}, store / "means.pt")
    data = tmp_path / "var/data/m2/demo_s0_discovery.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text("".join(json.dumps(r) + "\n" for r in records))
    seen = {}

    def load(name, dtype):
        assert (name, dtype) == ("test/tiny", "fp32")
        return model, tok

    def attribute(m, t, recs, heads, batch_size):
        assert m is model and t is tok
        assert recs == records[:2]
        torch.testing.assert_close(heads, means.float())
        assert batch_size == 1
        seen["called"] = True
        values = torch.arange(-12, 12).reshape(3, 8).float()
        return {"heads": values, "kv": values.reshape(3, 2, 4).sum(-1)}

    monkeypatch.setattr(cli, "load_model", load)
    monkeypatch.setattr(cli, "head_attribution", attribute)
    cli.main(["--model", "test/tiny", "--variant", "demo", "--n", "2", "--batch-size", "1"])
    assert seen["called"]
    result = json.loads((tmp_path / "var/results/m3/attribution_tiny_demo.json").read_text())
    checks = json.loads((tmp_path / "var/results/m3/check_heads_tiny_demo.json").read_text())
    assert (result["model"], result["variant"], result["n"]) == ("test/tiny", "demo", 2)
    assert len(result["heads"]) == len(result["kv"]) == 3
    assert result["seconds"] >= 0
    top, bottom = result["check_heads"]["top"], result["check_heads"]["bottom"]
    assert len(top) == len(bottom) == 10
    assert top[0] == [2, 7]  # largest positive score
    assert bottom[0] == [0, 0]  # most negative score, not closest to zero
    assert checks == top + bottom
    assert len({tuple(pair) for pair in checks}) == 20
