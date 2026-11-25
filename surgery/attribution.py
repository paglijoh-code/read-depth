"""First-order loss changes from mean-ablating attention heads."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from surgery.prompt import build_prompt, label_token_ids
from surgery.task.generator import LABELS


@torch.enable_grad()
def head_attribution(model, tok, records, means_heads, batch_size=8):
    """Return CPU fp32 ``heads`` (L, H) and ``kv`` (L, KV) score tensors.

    Scores are loss derivatives toward the supplied per-head means, summed over
    non-padding positions and averaged over examples. Positive scores predict
    increased six-label NLL after mean ablation. Parameters are temporarily
    frozen; their existing gradients and requires_grad flags are preserved.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not records:
        raise ValueError("records must not be empty")
    layers = model.model.layers
    config = model.config
    n_heads, n_kv = config.num_attention_heads, config.num_key_value_heads
    head_dim = layers[0].self_attn.head_dim
    shape = (len(layers), n_heads, head_dim)
    if tuple(means_heads.shape) != shape:
        raise ValueError(f"means_heads must have shape {shape}")
    if n_heads % n_kv:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    means = means_heads.detach().to(device=model.device, dtype=model.dtype)
    label_ids = torch.tensor(label_token_ids(tok), device=model.device)
    # MPS does not support fp64; accumulate on CPU without retaining batch graphs.
    totals = torch.zeros(len(layers), n_heads, dtype=torch.float64)
    captured = {}
    handles = []
    params = list(model.parameters())
    flags = [p.requires_grad for p in params]
    training = [(module, module.training) for module in model.modules()]

    def capture(layer):
        def hook(module, args):
            z = args[0]
            # The first layer starts a graph; later layers must keep their path
            # back to earlier z's, including through keys and values.
            z.requires_grad_(True)
            z.retain_grad()
            captured[layer] = z
        return hook

    try:
        model.eval()
        for p in params:
            p.requires_grad_(False)
        for layer, block in enumerate(layers):
            handles.append(block.self_attn.o_proj.register_forward_pre_hook(capture(layer)))
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            enc = tok([build_prompt(tok, r["prompt"]) for r in batch],
                      return_tensors="pt", padding=True).to(model.device)
            y = torch.tensor([LABELS.index(r["label"]) for r in batch], device=model.device)
            captured.clear()
            logits = model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1, label_ids]
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            loss = F.cross_entropy(logits, y, reduction="sum")
            loss.backward()
            with torch.no_grad():
                mask = enc["attention_mask"][..., None, None]
                for layer in range(len(layers)):
                    z = captured.pop(layer)
                    grad = z.grad.reshape(*z.shape[:2], n_heads, head_dim)
                    delta = means[layer] - z.reshape_as(grad)
                    scores = (delta * grad * mask).sum(dim=(0, 1, 3))
                    totals[layer] += scores.detach().cpu().double()
                    del z, grad, delta, scores
            del enc, logits, loss, y, mask
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)
        for module, flag in training:
            module.training = flag

    heads = (totals / len(records)).float()
    kv = heads.reshape(len(layers), n_kv, n_heads // n_kv).sum(dim=-1)
    return {"heads": heads, "kv": kv}
