"""Layer plans, FLOP counter and KV-bias equivalence on a tiny random Qwen3."""
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from surgery import structure
from surgery.hooks import apply_spec, spec_hash, validate_spec


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    model = Qwen3ForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    return model


@pytest.fixture
def inputs():
    return dict(input_ids=torch.tensor([[0, 0, 3, 7, 9], [1, 2, 4, 5, 6]]),
                attention_mask=torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]))


@torch.no_grad()
def fwd(model, inputs):
    return model(**inputs).logits


@torch.no_grad()
def manual(model, inputs, sequence):
    """Run blocks in `sequence` by hand (no cache), using the model's own mask construction."""
    from transformers.masking_utils import create_causal_mask
    emb = model.model.embed_tokens(inputs["input_ids"])
    pos = torch.arange(emb.shape[1]).unsqueeze(0)
    mask = create_causal_mask(config=model.config, inputs_embeds=emb, attention_mask=inputs["attention_mask"],
                              past_key_values=None, position_ids=pos)
    rot = model.model.rotary_emb(emb, pos)
    h = emb
    for i in sequence:
        h = model.model.layers[i](h, attention_mask=mask, position_embeddings=rot, position_ids=pos)
    return model.lm_head(model.model.norm(h))


def test_identity_plan_exact(tiny, inputs):
    base = fwd(tiny, inputs)
    with apply_spec(tiny, {"op": "plan", "order": [0, 1, 2, 3], "scales": None}):
        assert torch.equal(fwd(tiny, inputs), base)


def test_drop_one_equals_skip_hook(tiny, inputs):
    with apply_spec(tiny, {"op": "skip", "layers": [2]}):
        skipped = fwd(tiny, inputs)
    with apply_spec(tiny, {"op": "plan", "order": [0, 1, 3]}):
        torch.testing.assert_close(fwd(tiny, inputs), skipped)
    assert not torch.equal(skipped, fwd(tiny, inputs))


def test_repeat_scale_zero_is_identity_and_scale_one_runs_twice(tiny, inputs):
    base = fwd(tiny, inputs)
    with apply_spec(tiny, {"op": "plan", "order": [0, 1, 1, 2, 3], "scales": [1, 1, 0, 1, 1]}):
        torch.testing.assert_close(fwd(tiny, inputs), base)
    with apply_spec(tiny, {"op": "plan", "order": [0, 1, 1, 2, 3]}):
        twice = fwd(tiny, inputs)
    torch.testing.assert_close(twice, manual(tiny, inputs, [0, 1, 1, 2, 3]))
    assert not torch.allclose(twice, base)


def test_half_scale_blends(tiny, inputs):
    with apply_spec(tiny, {"op": "plan", "order": [0, 1, 2, 3], "scales": [1, 0.5, 1, 1]}) as _:
        half = fwd(tiny, inputs)
    assert not torch.allclose(half, fwd(tiny, inputs))


def test_no_cache_under_plan(tiny, inputs):
    assert tiny.config.use_cache
    with apply_spec(tiny, {"op": "plan", "order": [0, 1, 1, 2, 3]}):
        out = tiny(**inputs)
        assert out.past_key_values is None
        out = tiny(**inputs, use_cache=False)
        assert out.past_key_values is None
        assert tiny.config.use_cache is False


def test_remove_restores(tiny, inputs):
    layers, base = tiny.model.layers, fwd(tiny, inputs)
    types, n = list(tiny.config.layer_types), tiny.config.num_hidden_layers
    h = apply_spec(tiny, {"op": "plan", "order": [3, 0, 0, 1, 2], "scales": [1, 1, 0.5, 1, 1]})
    assert len(tiny.model.layers) == 5 and tiny.config.num_hidden_layers == 5
    h.remove()
    h.remove()
    assert tiny.model.layers is layers and len(layers) == 4
    assert tiny.config.num_hidden_layers == n and tiny.config.use_cache is True
    assert tiny.config.layer_types == types
    assert torch.equal(fwd(tiny, inputs), base)


def test_swap(tiny, inputs):
    with apply_spec(tiny, {"op": "plan", "order": [0, 2, 1, 3]}):
        swapped = fwd(tiny, inputs)
    torch.testing.assert_close(swapped, manual(tiny, inputs, [0, 2, 1, 3]))


def test_compose_and_validation(tiny, inputs):
    abl = {"op": "ablate", "site": "mlp", "mode": "zero", "layers": [1]}
    plan = {"op": "plan", "order": [0, 1, 3]}
    both = {"op": "compose", "specs": [plan, abl]}
    validate_spec(both, 4, 4, 2)
    with apply_spec(tiny, abl):
        expected = None
        with apply_spec(tiny, plan):
            expected = fwd(tiny, inputs)
    with apply_spec(tiny, both):
        torch.testing.assert_close(fwd(tiny, inputs), expected)
    assert len(tiny.model.layers) == 4 and len(tiny.model.layers[1].mlp._forward_hooks) == 0
    assert spec_hash(both) != spec_hash(plan)
    for bad in [{"op": "compose", "specs": [plan, plan]}, {"op": "compose", "specs": []},
                {"op": "compose", "specs": [{"op": "compose", "specs": [abl]}]},
                {"op": "plan", "order": [0, 9]}, {"op": "plan", "order": []},
                {"op": "plan", "order": [0, 1], "scales": [1.0]}]:
        with pytest.raises(ValueError):
            validate_spec(bad, 4, 4, 2)


def hand_config():
    return Qwen3Config(vocab_size=100, hidden_size=8, intermediate_size=16, num_hidden_layers=3,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=4)


def test_flops_hand_computed():
    cfg, T = hand_config(), 7  # ctx = 4
    # q 2*8*16=256, k and v 2*8*8=128 each, o 256, scores+values 4*4*4*4=256 -> attn 1024; mlp 3*2*8*16=768
    f = structure.flops_per_token(cfg, T)
    assert f["attn"] == 3 * 1024 and f["mlp"] == 3 * 768 and f["total"] == 3 * 1792
    assert f["relative"] == 1.0 and f["lm_head"] == 2 * 8 * 100
    # repeat block 1, drop block 2: blocks 0,1,1
    assert structure.flops_per_token(cfg, T, order=[0, 1, 1])["total"] == 3 * 1792
    assert structure.flops_per_token(cfg, T, order=[0, 1, 1, 1])["relative"] == pytest.approx(4 / 3)
    assert structure.flops_per_token(cfg, T, order=[0])["relative"] == pytest.approx(1 / 3)
    # prune group (1, 0): -k,v rows 128, -q rows (2 heads) 128, -o cols 128, -core 4*2*4*4=128 -> 512
    g = structure.flops_per_token(cfg, T, pruned_groups=[(1, 0)])
    assert g["attn"] == 3 * 1024 - 512 and g["mlp"] == 3 * 768
    # pruned layer 1 appears twice in the plan, layer 2 removed
    r = structure.flops_per_token(cfg, T, order=[0, 1, 1], pruned_groups=[(1, 0), (2, 1)])
    assert r["total"] == 3 * 1792 - 2 * 512
    p = structure.param_count(cfg)
    assert p["attn"] == 3 * (8 * 16 * 2 + 8 * 8 * 2) and p["mlp"] == 3 * 3 * 8 * 16


def test_param_count_pruned_group():
    cfg = hand_config()
    # one group: k,v rows 2*8*4, q rows + o cols 2 * (8*4*2)
    group = 2 * 8 * 4 + 2 * 8 * 4 * 2
    full = structure.param_count(cfg)
    assert structure.param_count(cfg, pruned_groups=[(1, 0)])["total"] == full["total"] - group
    assert structure.param_count(cfg, order=[0, 1, 1], pruned_groups=[(1, 0)])["total"] == \
        full["total"] - group * 2
    assert structure.param_count(cfg, order=[0, 1, 1])["unique"] == full["unique"] - (full["unique"] // 3)


def test_kv_bias_equals_mean_ablate_kv(tiny, inputs):
    g = torch.Generator().manual_seed(3)
    heads = torch.randn(4, 4, 16, generator=g) / 5
    means = {"heads": heads}
    groups = [[1, 0], [2, 1], [2, 0]]
    with apply_spec(tiny, {"op": "ablate_kv", "mode": "mean", "groups": groups}, means):
        expected = fwd(tiny, inputs)
    bias = structure.kv_bias(tiny, heads, groups)
    assert sorted(bias) == [1, 2]
    hooks = []
    for layer, b in bias.items():
        sel = [h for l, gi in groups if l == layer for h in range(gi * 2, gi * 2 + 2)]

        def zero(module, args, sel=sel):
            x = args[0].clone()
            for h in sel:
                x[..., h * 16:(h + 1) * 16] = 0
            return (x,)

        def add(module, args, output, b=b):
            return output + b
        o = tiny.model.layers[layer].self_attn.o_proj
        hooks += [o.register_forward_pre_hook(zero), o.register_forward_hook(add)]
    try:
        actual = fwd(tiny, inputs)
    finally:
        for h in hooks:
            h.remove()
    torch.testing.assert_close(actual, expected)
