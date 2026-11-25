"""M7 pieces on a tiny random Qwen3: merged LoRA, physical excision, KV-group fold, counts."""
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from surgery import adapters, structure
from surgery.hooks import apply_spec

L, H, KV, HD, D = 4, 4, 2, 16, 64


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=32, hidden_size=D, intermediate_size=128, num_hidden_layers=L,
                         num_attention_heads=H, num_key_value_heads=KV, head_dim=HD)
    model = Qwen3ForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    return model


@pytest.fixture
def enc():
    return dict(input_ids=torch.tensor([[0, 0, 3, 7, 9], [1, 2, 4, 5, 6]]),
                attention_mask=torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]))


def fwd(model, enc):
    with torch.no_grad():
        return model(**enc).logits


def randomize_lora(handle):
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for m in handle.modules:
            m.B.copy_(torch.randn(m.B.shape, generator=g) * 0.1)


def test_merged_lora_equals_unmerged(tiny, enc):
    modules = ("q_proj", "k_proj", "v_proj", "o_proj")
    h = adapters.add_lora(tiny, range(L), modules, r=3)
    randomize_lora(h)
    unmerged = fwd(tiny, enc)
    base = adapters.LoRALinear
    assert isinstance(tiny.model.layers[0].self_attn.o_proj, base)
    assert adapters.merge_lora(h) == 4 * L
    assert all(type(getattr(l.self_attn, m)) is torch.nn.Linear for l in tiny.model.layers for m in modules)
    assert (fwd(tiny, enc) - unmerged).abs().max() < 1e-4
    h.remove()  # no-op: the merged weights stay
    assert (fwd(tiny, enc) - unmerged).abs().max() < 1e-4


def test_excise_equals_apply_plan(tiny, enc):
    alive = [0, 2, 3]
    with apply_spec(tiny, {"op": "plan", "order": alive}):
        planned = fwd(tiny, enc)
    structure.excise(tiny, alive)
    assert len(tiny.model.layers) == 3 and tiny.config.num_hidden_layers == 3
    assert len(tiny.config.layer_types) == 3
    assert [l.self_attn.layer_idx for l in tiny.model.layers] == [0, 1, 2]
    torch.testing.assert_close(fwd(tiny, enc), planned)
    with torch.no_grad():  # the KV cache works on the renumbered model
        out = tiny(**enc, use_cache=True)
    assert out.past_key_values.get_seq_length() == 5


def make_means(n_layers):
    return torch.randn(n_layers, H, HD, generator=torch.Generator().manual_seed(3)) / 5


@pytest.mark.parametrize("alive", [None, [0, 2, 3]])
def test_fold_equals_mean_ablate_hook(tiny, enc, alive):
    if alive is not None:
        structure.excise(tiny, alive)
    n = len(tiny.model.layers)
    means = make_means(n)
    groups = [(0, 1), (1, 0)]
    with apply_spec(tiny, {"op": "ablate_kv", "mode": "mean", "groups": [list(g) for g in groups]},
                    {"heads": means}):
        expected = fwd(tiny, enc)
    structure.fold_kv_groups(tiny, means, groups)
    assert (fwd(tiny, enc) - expected).abs().max() < 1e-4
    attn = tiny.model.layers[1].self_attn
    assert attn.k_proj.out_features == HD and attn.q_proj.out_features == 2 * HD and attn.o_proj.bias is not None
    assert structure.kv_cache_bytes_per_token(tiny) == 2 * (n * KV - 2) * HD * 2


def test_counts_after_excise_and_fold(tiny):
    structure.excise(tiny, [0, 2, 3])
    structure.fold_kv_groups(tiny, make_means(3), [(0, 1), (2, 0)])
    # 3 blocks x (q,o 2*D*H*HD + k,v 2*D*KV*HD + mlp 3*D*128 + 2 norms D + q/k norm 2*HD) minus 2 groups, plus 2 biases
    block = 2 * D * H * HD + 2 * D * KV * HD + 3 * D * 128 + 2 * D + 2 * HD
    group = 2 * D * HD + 2 * (D * 2 * HD)
    embed = 2 * 32 * D  # the tiny config is untied: embed_tokens and lm_head
    expected = 3 * block - 2 * group + 2 * D + D + embed  # + final norm D
    sz = structure.model_size(tiny)
    assert sz["total"] == expected and sz["embedding"] == embed
    assert sz["non_embedding"] == expected - embed and sz["bf16_bytes"] == 2 * expected
    assert structure.kv_cache_bytes_per_token(tiny) == 2 * (3 * KV - 2) * HD * 2
    assert structure.kv_cache_bytes_per_token(tiny, 4) == 2 * (3 * KV - 2) * HD * 4
    cfg, pg = tiny.config, [(0, 1), (2, 0)]
    assert structure.kv_cache_bytes_per_token(cfg, pruned_groups=pg) == 2 * (3 * KV - 2) * HD * 2
    pc = structure.param_count(cfg, pruned_groups=pg)
    block_w = sum(p.numel() for n, p in tiny.model.layers.named_parameters()
                  if n.endswith("proj.weight"))
    assert pc["total"] == block_w


def test_lora_trains_on_folded_model(tiny, enc):
    structure.excise(tiny, [0, 2, 3])
    structure.fold_kv_groups(tiny, make_means(3), [(0, 1), (2, 0)])
    h = adapters.add_lora(tiny, range(3), ("q_proj", "k_proj", "v_proj", "o_proj"), r=2)
    ids = torch.tensor([3, 5, 7, 9, 11, 13])
    opt = torch.optim.AdamW([p for p in tiny.parameters() if p.requires_grad], lr=1e-2)
    before = adapters.full_logits(tiny, enc, ids)
    loss = torch.nn.functional.cross_entropy(before, torch.tensor([0, 1]))
    loss.backward()
    opt.step()
    assert (adapters.full_logits(tiny, enc, ids) - before).abs().max() > 0
    adapters.merge_lora(h)
    assert structure.model_size(tiny)["total"] > 0


def test_model_size_counts_tied_embedding_once():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=32, hidden_size=D, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=H, num_key_value_heads=KV, head_dim=HD, tie_word_embeddings=True)
    model = Qwen3ForCausalLM(config)
    sz = structure.model_size(model)
    assert sz["embedding"] == 32 * D and sz["total"] == sz["embedding"] + sz["non_embedding"]
    assert sz["total"] == sum(p.numel() for p in model.parameters())
