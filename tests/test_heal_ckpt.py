"""Gradient checkpointing in train_whole leaves the loss and LoRA gradients unchanged (D3 run 1 memory fix)."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from surgery.atlas import heal


def _model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    m = LlamaForCausalLM(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m.model.layers[1].self_attn.q_proj.weight.requires_grad_(True)
    return m


def _grad(on):
    m = _model()
    ids = torch.randint(0, 64, (2, 12))
    enc = (ids, torch.ones_like(ids), torch.where(torch.arange(12) >= 10, ids, torch.full_like(ids, -100)), 2)
    heal._checkpointing(m, on)
    loss = heal.whole_loss(m, enc)
    loss.backward()
    g = m.model.layers[1].self_attn.q_proj.weight.grad.clone()
    heal._checkpointing(m, False)
    assert not m.training
    return float(loss), g


def test_checkpointing_same_loss_and_grad():
    l0, g0 = _grad(False)
    l1, g1 = _grad(True)
    assert abs(l0 - l1) < 1e-6
    assert torch.allclose(g0, g1, atol=1e-6)
