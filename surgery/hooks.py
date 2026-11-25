"""Intervention hooks on HF Qwen3 decoder layers (plain PyTorch forward hooks).

For {"op": "skip", "layers": [..]}, a skipped layer's output equals its
input hidden states (residual passthrough). In transformers 5.x a decoder layer
receives hidden_states as the first positional arg (or keyword) and returns a tensor;
older versions returned a tuple, so both are handled.
"""
from __future__ import annotations

import hashlib
import json

import torch

SUPPORTED_OPS = {"skip", "ablate", "ablate_heads", "ablate_kv", "plan", "compose"}
COMPOSABLE_OPS = SUPPORTED_OPS - {"compose"}


def spec_hash(spec: dict) -> str:
    canon = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def validate_spec(spec: dict, n_layers: int | None = None,
                  n_heads: int | None = None, n_kv_heads: int | None = None) -> None:
    op = spec.get("op")
    if op not in SUPPORTED_OPS:
        raise ValueError(f"unknown op {op!r}; supported: {sorted(SUPPORTED_OPS)}")
    if op == "plan":
        _validate_plan(spec, n_layers)
        return
    if op == "compose":
        _validate_compose(spec, n_layers, n_heads, n_kv_heads)
        return
    if op != "skip":
        _validate_ablation(spec, n_layers, n_heads, n_kv_heads)
        return
    layers = spec.get("layers")
    if not isinstance(layers, (list, tuple)) or not all(isinstance(i, int) for i in layers):
        raise ValueError("skip spec needs 'layers': list of ints")
    if n_layers is not None and any(i < 0 or i >= n_layers for i in layers):
        raise ValueError(f"layer index out of range 0..{n_layers - 1}: {layers}")


def _validate_plan(spec, n_layers):
    order, scales = spec.get("order"), spec.get("scales")
    if (not isinstance(order, (list, tuple)) or not order
            or not all(type(i) is int and i >= 0 for i in order)):
        raise ValueError("plan spec needs 'order': non-empty list of block indices")
    if n_layers is not None and any(i >= n_layers for i in order):
        raise ValueError(f"layer index out of range 0..{n_layers - 1}: {list(order)}")
    if scales is not None and (not isinstance(scales, (list, tuple)) or len(scales) != len(order)
                               or not all(isinstance(s, (int, float)) and not isinstance(s, bool) for s in scales)):
        raise ValueError("plan 'scales' must be null or one number per order entry")


def _validate_compose(spec, n_layers, n_heads, n_kv_heads):
    specs = spec.get("specs")
    if not isinstance(specs, (list, tuple)) or not specs:
        raise ValueError("compose spec needs 'specs': non-empty list")
    if sum(isinstance(s, dict) and s.get("op") == "plan" for s in specs) > 1:
        raise ValueError("compose allows at most one plan")
    for s in specs:
        if not isinstance(s, dict) or s.get("op") not in COMPOSABLE_OPS:
            raise ValueError(f"compose entries must be non-compose specs: {s!r}")
        validate_spec(s, n_layers, n_heads, n_kv_heads)


def _skip_hook(module, args, kwargs, output):
    hs = args[0] if args else kwargs["hidden_states"]
    if isinstance(output, tuple):
        return (hs,) + tuple(output[1:])
    return hs


class Handle:
    """Removable handle; also a context manager."""

    def __init__(self, handles):
        self._handles = handles

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def apply_spec(model, spec: dict, means=None) -> Handle:
    layers = model.model.layers
    if spec.get("op") in ("plan", "compose"):
        return _apply_structural(model, spec, means)
    if spec.get("op") != "skip":
        return _apply_ablation(model, spec, means)
    validate_spec(spec, len(layers))
    handles = [layers[i].register_forward_hook(_skip_hook, with_kwargs=True)
               for i in sorted(set(spec["layers"]))]
    return Handle(handles)


def _apply_structural(model, spec, means):
    config = model.config
    validate_spec(spec, len(model.model.layers), config.num_attention_heads, config.num_key_value_heads)
    from surgery.structure import apply_plan
    subs = spec["specs"] if spec["op"] == "compose" else [spec]
    handles = []
    try:  # hooks attach to modules by original index, so they go on before the plan reorders anything
        for s in subs:
            if s["op"] != "plan":
                handles.append(apply_spec(model, s, means))
        for s in subs:
            if s["op"] == "plan":
                handles.append(apply_plan(model, s["order"], s.get("scales")))
    except Exception:
        Handle(handles[::-1]).remove()
        raise
    return Handle(handles[::-1])


def _validate_ablation(spec, n_layers, n_heads, n_kv_heads):
    if spec.get("mode") not in {"zero", "mean"}:
        raise ValueError("ablation mode must be 'zero' or 'mean'")
    op = spec["op"]
    if op == "ablate":
        if spec.get("site") not in {"attn", "mlp"}:
            raise ValueError("ablation site must be 'attn' or 'mlp'")
        key, width = "layers", None
    else:
        key, width = ("heads", n_heads) if op == "ablate_heads" else ("groups", n_kv_heads)
    entries = spec.get(key)
    if not isinstance(entries, (list, tuple)):
        raise ValueError(f"ablation spec needs '{key}': list of indices")
    for entry in entries:
        pair = (entry,) if op == "ablate" else entry
        expected = 1 if op == "ablate" else 2
        if (not isinstance(pair, (list, tuple)) or len(pair) != expected
                or any(type(i) is not int or i < 0 for i in pair)):
            raise ValueError(f"invalid {key} entry: {entry!r}")
        if n_layers is not None and pair[0] >= n_layers:
            raise ValueError(f"layer index out of range 0..{n_layers - 1}: {pair[0]}")
        if expected == 2 and width is not None and pair[1] >= width:
            raise ValueError(f"{key} index out of range 0..{width - 1}: {pair[1]}")


def _output_ablation(mean):
    def hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        replacement = (torch.zeros_like(value) if mean is None else
                       mean.to(device=value.device, dtype=value.dtype).expand_as(value))
        return (replacement,) + output[1:] if isinstance(output, tuple) else replacement
    return hook


def _head_ablation(heads, head_dim, means):
    def hook(module, args, kwargs):
        value = args[0] if args else kwargs["input"]
        replacement = value.clone()
        for head in heads:
            span = slice(head * head_dim, (head + 1) * head_dim)
            replacement[..., span] = (0 if means is None else
                                     means[head].to(device=value.device, dtype=value.dtype))
        if args:
            return (replacement,) + args[1:], kwargs
        return args, {**kwargs, "input": replacement}
    return hook


def _apply_ablation(model, spec, means):
    layers = model.model.layers
    config = model.config
    n_heads, n_kv = config.num_attention_heads, config.num_key_value_heads
    head_dim = config.head_dim or config.hidden_size // n_heads
    validate_spec(spec, len(layers), n_heads, n_kv)
    if n_heads % n_kv:
        raise ValueError("query heads must divide evenly into KV groups")
    site = spec["site"] if spec["op"] == "ablate" else "heads"
    values = None
    if spec["mode"] == "mean":
        if means is None or site not in means:
            raise ValueError(f"mean ablation requires means[{site!r}]")
        values = means[site]
        shape = ((len(layers), n_heads, head_dim) if site == "heads" else
                 (len(layers), config.hidden_size))
        if not isinstance(values, torch.Tensor) or tuple(values.shape) != shape:
            raise ValueError(f"means[{site!r}] must be a tensor of shape {shape}")
        values = values.detach()

    handles = []
    if spec["op"] == "ablate":
        for i in sorted(set(spec["layers"])):
            module = layers[i].self_attn if site == "attn" else layers[i].mlp
            handles.append(module.register_forward_hook(
                _output_ablation(None if values is None else values[i])))
    else:
        selected = {}
        key = "heads" if spec["op"] == "ablate_heads" else "groups"
        for layer, index in spec[key]:
            heads = ([index] if key == "heads" else
                     range(index * (n_heads // n_kv), (index + 1) * (n_heads // n_kv)))
            selected.setdefault(layer, set()).update(heads)
        for i, heads in sorted(selected.items()):
            handles.append(layers[i].self_attn.o_proj.register_forward_pre_hook(
                _head_ablation(sorted(heads), head_dim, None if values is None else values[i]),
                with_kwargs=True))
    return Handle(handles)
