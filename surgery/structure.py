"""Structural surgery: layer plans (remove / repeat / reorder / scale blocks), analytic FLOPs, KV-prune bias.

A plan temporarily replaces `model.model.layers` with the original blocks in a new order. Qwen3Model.forward
iterates `self.layers[:config.num_hidden_layers]` and picks each mask with `config.layer_types[i]`, so the
config is patched too (restored by remove()). No KV cache is used under a plan: a repeated block would
otherwise append to its own cache.
"""
from __future__ import annotations

import torch
from torch import nn

from surgery.hooks import Handle


class _Scaled(nn.Module):
    """output = x + s * (layer(x) - x)"""

    def __init__(self, layer, scale: float):
        super().__init__()
        self.layer, self.scale = layer, float(scale)

    def forward(self, hidden_states, *args, **kwargs):
        out = self.layer(hidden_states, *args, **kwargs)
        value = out[0] if isinstance(out, tuple) else out
        value = hidden_states + self.scale * (value - hidden_states)
        return (value,) + tuple(out[1:]) if isinstance(out, tuple) else value


class _Restore:
    def __init__(self, model, layers, config_state):
        self.model, self.layers, self.state = model, layers, config_state

    def remove(self):
        if self.model is None:
            return
        self.model.model.layers = self.layers
        for key, value in self.state.items():
            setattr(self.model.config, key, value)
        self.model = None


def apply_plan(model, order, scales=None) -> Handle:
    layers = model.model.layers
    n = len(layers)
    order = list(order)
    if not order or any(type(i) is not int or not 0 <= i < n for i in order):
        raise ValueError(f"plan order needs ints in 0..{n - 1}")
    if scales is not None and len(scales) != len(order):
        raise ValueError("scales must match order in length")
    cfg = model.config
    state = {"num_hidden_layers": cfg.num_hidden_layers, "use_cache": cfg.use_cache,
             "layer_types": getattr(cfg, "layer_types", None)}
    new = []
    for j, i in enumerate(order):
        s = 1.0 if scales is None else scales[j]
        new.append(layers[i] if s == 1 else _Scaled(layers[i], s))
    types = state["layer_types"]
    handle = Handle([_Restore(model, layers, state)])
    model.model.layers = nn.ModuleList(new)
    cfg.num_hidden_layers = len(order)
    cfg.use_cache = False
    if types is not None:
        cfg.layer_types = [types[i] for i in order]
    return handle


def _dims(config):
    nh, nkv = config.num_attention_heads, config.num_key_value_heads
    hd = getattr(config, "head_dim", None) or config.hidden_size // nh
    return config.hidden_size, nh, nkv, hd, config.intermediate_size


def _block_cost(config, seq_len, pruned_in_layer: int):
    """(attn, mlp) FLOPs per token for one block with `pruned_in_layer` KV groups removed."""
    d, nh, nkv, hd, ff = _dims(config)
    r = nh // nkv
    ctx = (seq_len + 1) / 2
    heads, kv = nh - r * pruned_in_layer, nkv - pruned_in_layer
    attn = 2 * d * heads * hd + 2 * (2 * d * kv * hd) + 2 * heads * hd * d + 4 * heads * hd * ctx
    return attn, 3 * 2 * d * ff


def _plan_layers(config, order, pruned_groups):
    order = list(range(config.num_hidden_layers)) if order is None else list(order)
    pruned = {}
    for layer, g in pruned_groups:
        pruned.setdefault(layer, set()).add(g)
    return order, {k: len(v) for k, v in pruned.items()}


def flops_per_token(config, seq_len, order=None, pruned_groups=()) -> dict:
    """Prefill FLOPs per token at `seq_len` (mean causal context (seq_len+1)/2), embedding and LM head excluded."""
    order, pruned = _plan_layers(config, order, pruned_groups)
    attn = mlp = 0.0
    for i in order:
        a, m = _block_cost(config, seq_len, pruned.get(i, 0))
        attn, mlp = attn + a, mlp + m
    full = sum(_block_cost(config, seq_len, 0)) * config.num_hidden_layers
    total = attn + mlp
    return {"total": total, "attn": attn, "mlp": mlp, "relative": total / full,
            "lm_head": 2 * config.hidden_size * config.vocab_size}


def param_count(config, order=None, pruned_groups=()) -> dict:
    """Block weight matrices only (norms, embedding excluded). A repeated block counts once per occurrence in
    `total`; `unique` counts shared weights once."""
    d, nh, nkv, hd, ff = _dims(config)
    r = nh // nkv
    order, pruned = _plan_layers(config, order, pruned_groups)

    def block(i):
        k = pruned.get(i, 0)
        heads, kv = nh - r * k, nkv - k
        return 2 * d * heads * hd + 2 * d * kv * hd, 3 * d * ff
    attn = sum(block(i)[0] for i in order)
    mlp = sum(block(i)[1] for i in order)
    unique = sum(sum(block(i)) for i in set(order))
    full = sum(sum(block(i)) for i in range(config.num_hidden_layers))
    return {"total": attn + mlp, "attn": attn, "mlp": mlp, "unique": unique, "relative": (attn + mlp) / full,
            "embed": config.vocab_size * d}


@torch.no_grad()
def kv_bias(model, means_heads, groups) -> dict:
    """{layer: tensor(d)}: o_proj of the mean head values of the pruned groups. Zeroing those heads' o_proj
    input slice and adding this bias equals mean-ablating them (o_proj has no bias)."""
    cfg = model.config
    _, nh, nkv, hd, _ = _dims(cfg)
    r = nh // nkv
    out = {}
    for layer, g in groups:
        w = model.model.layers[layer].self_attn.o_proj.weight
        acc = out.get(layer, torch.zeros(w.shape[0], dtype=w.dtype, device=w.device))
        for h in range(g * r, (g + 1) * r):
            m = means_heads[layer, h].to(device=w.device, dtype=w.dtype)
            acc = acc + w[:, h * hd:(h + 1) * hd] @ m
        out[layer] = acc
    return out


# ---------------------------------------------------------------- permanent surgery (M7)

def excise(model, alive) -> None:
    """Physically drop every block not in `alive` (original indices, kept in the given order). Same logits as
    `apply_plan(model, alive)`, but permanent: config and `self_attn.layer_idx` are renumbered so the KV cache
    works and the model can be saved."""
    layers = model.model.layers
    alive = list(alive)
    n = len(layers)
    if not alive or any(type(i) is not int or not 0 <= i < n for i in alive) or len(set(alive)) != len(alive):
        raise ValueError(f"excise needs distinct ints in 0..{n - 1}")
    model.model.layers = nn.ModuleList([layers[i] for i in alive])
    cfg = model.config
    types = getattr(cfg, "layer_types", None)
    cfg.num_hidden_layers = len(alive)
    if types is not None:
        cfg.layer_types = [types[i] for i in alive]
    for j, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = j


def _slim(old: nn.Linear, rows=None, cols=None, extra_bias=None) -> nn.Linear:
    w = old.weight.detach()
    if rows is not None:
        w = w[rows]
    if cols is not None:
        w = w[:, cols]
    bias = old.bias.detach() if old.bias is not None else None
    if bias is not None and rows is not None:
        bias = bias[rows]
    if extra_bias is not None:
        bias = extra_bias if bias is None else bias + extra_bias
    new = nn.Linear(w.shape[1], w.shape[0], bias=bias is not None, device=w.device, dtype=w.dtype)
    with torch.no_grad():
        new.weight.copy_(w)
        if bias is not None:
            new.bias.copy_(bias)
    return new


@torch.no_grad()
def fold_kv_groups(model, head_means, groups) -> None:
    """Physical KV-group pruning on the current layer numbering. `head_means[layer]` is (n_heads, head_dim), the
    mean o_proj input per head; `groups` is [(layer, group)] with group indices as in the unpruned layer. For each
    pruned group the k/v rows, its query heads' q rows and o_proj input columns are dropped and o_proj gets a bias
    equal to o_proj applied to the dropped heads' means (as `kv_bias`), which equals the mean-ablation hook.
    Qwen3 attention reshapes with -1 heads and a constant `num_key_value_groups`, so per-layer head counts
    work as long as each layer keeps at least one group. Fold each layer once."""
    cfg = model.config
    _, nh, nkv, hd, _ = _dims(cfg)
    r = nh // nkv
    bias = kv_bias(model, head_means, [tuple(g) for g in groups])
    pruned = {}
    for layer, g in groups:
        if not 0 <= g < nkv:
            raise ValueError(f"group {g} out of range 0..{nkv - 1}")
        pruned.setdefault(layer, set()).add(g)
    for layer, gs in pruned.items():
        attn = model.model.layers[layer].self_attn
        if attn.k_proj.out_features != nkv * hd:
            raise ValueError(f"layer {layer} is already folded")
        keep = [g for g in range(nkv) if g not in gs]
        if not keep:
            raise ValueError(f"layer {layer}: cannot prune every KV group")
        kv_rows = torch.tensor([g * hd + j for g in keep for j in range(hd)])
        q_rows = torch.tensor([(g * r + i) * hd + j for g in keep for i in range(r) for j in range(hd)])
        attn.q_proj = _slim(attn.q_proj, rows=q_rows)
        attn.k_proj = _slim(attn.k_proj, rows=kv_rows)
        attn.v_proj = _slim(attn.v_proj, rows=kv_rows)
        attn.o_proj = _slim(attn.o_proj, cols=q_rows, extra_bias=bias[layer])


@torch.no_grad()
def head_means(model, tok, records, batch_size: int = 8) -> torch.Tensor:
    """(n_layers, n_heads, head_dim) mean o_proj input per head over non-pad prompt tokens, computed as in
    `store.build_store` (fp32 per-batch sums on the device, fp64 running total), on the model as it is."""
    from surgery.eval import _free
    from surgery.prompt import build_prompt
    layers = model.model.layers
    hd = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
    H = layers[0].self_attn.o_proj.in_features // hd
    total = torch.zeros(len(layers), H, hd, dtype=torch.float64)
    state = {"mask": None}

    def make(i):
        def hook(m, args):
            z = args[0].detach().float()
            z = z.reshape(z.shape[0], z.shape[1], H, hd)
            total[i] += (z * state["mask"][..., None, None]).sum((0, 1)).cpu().double()
        return hook

    handles = [l.self_attn.o_proj.register_forward_pre_hook(make(i)) for i, l in enumerate(layers)]
    count = 0
    try:
        for b in range(0, len(records), batch_size):
            enc = tok([build_prompt(tok, r["prompt"]) for r in records[b:b + batch_size]],
                      return_tensors="pt", padding=True).to(model.device)
            state["mask"] = enc["attention_mask"].float()
            count += int(enc["attention_mask"].sum())
            model(**enc, logits_to_keep=1)
            del enc
            _free(model)
    finally:
        for h in handles:
            h.remove()
    return (total / count).float().cpu()


def kv_cache_bytes_per_token(config_or_model, bytes_per_value: int = 2, pruned_groups=()) -> int:
    """2 (k and v) x sum over layers of KV heads left x head_dim x bytes. A model is read off its k_proj shapes
    (so it reflects excision and folds); a config uses `pruned_groups` [(layer, group)] on its own numbering."""
    layers = getattr(getattr(config_or_model, "model", None), "layers", None)
    if layers is not None:
        cfg = config_or_model.config
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        kv = sum(l.self_attn.k_proj.out_features // hd for l in layers)
    else:
        cfg = config_or_model
        _, _, nkv, hd, _ = _dims(cfg)
        kv = nkv * cfg.num_hidden_layers - len({tuple(g) for g in pruned_groups})
    return 2 * kv * hd * bytes_per_value


def model_size(model) -> dict:
    """Parameter counts (tied embedding once) and bf16 bytes. Merge any LoRA first."""
    total = sum(p.numel() for p in model.parameters())
    embed = model.model.embed_tokens.weight.numel()
    head = model.lm_head.weight
    if head.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        embed += head.numel()
    return {"total": total, "embedding": embed, "non_embedding": total - embed, "bf16_bytes": 2 * total}
