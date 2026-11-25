"""LoRA, LoReFT, the prefix-cached LoReFT path and the held-out split on a tiny random Qwen3."""
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from surgery import adapters
from surgery.hooks import apply_spec


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    model = Qwen3ForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    return model


@pytest.fixture
def enc():
    return dict(input_ids=torch.tensor([[0, 0, 3, 7, 9], [1, 2, 4, 5, 6]]),
                attention_mask=torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]))


IDS = torch.tensor([3, 5, 7, 9, 11, 13])


def fwd(model, enc):
    with torch.no_grad():
        return model(**enc).logits


def test_lora_init_is_exact_and_remove_restores(tiny, enc):
    base = fwd(tiny, enc)
    originals = [tiny.model.layers[i].self_attn.q_proj for i in range(4)]
    h = adapters.add_lora(tiny, [0, 1, 2, 3], r=2)
    assert isinstance(tiny.model.layers[0].self_attn.q_proj, adapters.LoRALinear)
    assert torch.equal(fwd(tiny, enc), base)
    h.remove()
    assert [tiny.model.layers[i].self_attn.q_proj for i in range(4)] == originals
    assert torch.equal(fwd(tiny, enc), base)


def test_trainable_counts(tiny):
    v_out = tiny.model.layers[0].self_attn.v_proj.out_features
    q_out = tiny.model.layers[0].self_attn.q_proj.out_features
    h = adapters.add_lora(tiny, [2, 3], modules=("v_proj",), r=1)
    assert adapters.count_trainable(tiny) == 2 * 1 * (64 + v_out)
    h.remove()
    h = adapters.add_lora(tiny, [0, 1, 2, 3], r=3)
    assert adapters.count_trainable(tiny) == 4 * 3 * ((64 + q_out) + (64 + v_out))
    h.remove()
    for d, r in ((64, 5), (64, 49)):
        m = adapters.LoReFT(d, r)
        assert adapters.count_trainable(m) == m.n_params == r * (2 * d + 1)


def test_gradients_reach_only_adapter_params(tiny, enc):
    h = adapters.add_lora(tiny, [1, 2], r=2)
    opt = torch.optim.AdamW([p for p in tiny.parameters() if p.requires_grad], lr=1e-2)
    adapters.full_logits(tiny, enc, IDS).sum().backward()
    opt.step()
    with_grad = {n for n, p in tiny.named_parameters() if p.grad is not None}
    assert with_grad and all(n.endswith((".A", ".B")) for n in with_grad)
    assert all(not p.requires_grad for n, p in tiny.named_parameters() if not n.endswith((".A", ".B")))
    h.remove()


def test_loreft_init_is_identity_and_keeps_orthonormal(tiny, enc):
    init = torch.randn(3, 64)
    m = adapters.LoReFT(64, 8, init_rows=init, seed=1)
    h = torch.randn(4, 64)
    assert (m(h) - h).abs().max() < 1e-5
    R = m.R.weight.detach()
    assert (R @ R.T - torch.eye(8)).abs().max() < 1e-5
    q, _ = torch.linalg.qr(init.T.double())  # first rows span the same space as init (in order)
    assert (R[:3].double() @ q[:, :3]).abs().det() > 0.99
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    (m(h) ** 2).sum().backward()
    opt.step()
    R = m.R.weight.detach()
    assert (R @ R.T - torch.eye(8)).abs().max() < 1e-4


def test_cached_last_token_path_equals_full_forward(tiny, enc):
    torch.manual_seed(2)
    m = adapters.LoReFT(64, 8, seed=0)
    with torch.no_grad():
        m.proj.weight.add_(0.1 * torch.randn_like(m.proj.weight))
        m.proj.bias.add_(0.1 * torch.randn_like(m.proj.bias))
    spec = {"op": "ablate", "site": "attn", "mode": "zero", "layers": [1]}
    with apply_spec(tiny, spec):
        with adapters.apply_loreft(tiny, 2, m):
            full = adapters.full_logits(tiny, enc, IDS)
        cached = adapters.cached_logits(tiny, enc, IDS, edit=lambda: adapters.apply_loreft(tiny, 2, m))
    assert (full - cached).abs().max() <= 1e-4
    with apply_spec(tiny, spec):
        assert (adapters.full_logits(tiny, enc, IDS) - full).abs().max() > 1e-4  # the edit does something


def test_heldout_is_template_disjoint_from_every_seed():
    recs = [{"id": i, "template_id": f"t{i % 40}", "label": "DONE", "prompt": "p"} for i in range(1200)]
    held, _ = adapters.split_discovery(recs, 0, n_heldout=200, n_train=500)
    assert len(held) == 200
    held_t = {r["template_id"] for r in held}
    for seed in range(3):
        h2, train = adapters.split_discovery(recs, seed, n_heldout=200, n_train=500)
        assert h2 == held and len(train) == 500
        assert not held_t & {r["template_id"] for r in train}
    assert adapters.split_discovery(recs, 0)[1] != adapters.split_discovery(recs, 1)[1]
