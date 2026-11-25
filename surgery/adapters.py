"""Adapters for the M6 adapter axis: LoRA on attention projections and LoReFT at the last prompt token.

LoRA swaps `layers[i].self_attn.<m>` for a `LoRALinear` (B starts at zero, so the model is unchanged at init).
LoReFT edits the output of decoder layer L at the last prompt token: h + R^T (W h + b - R h), the same site
and convention as `steer.apply_steer`. Training is one forward per batch; for LoReFT the prompt prefix (all
tokens but the last) never changes, so it runs once without gradients and only the last token is re-run.
"""
from __future__ import annotations

import math
import random
import time

import torch
from torch import nn
from torch.nn.utils import parametrizations

from surgery.eval import _free
from surgery.hooks import Handle
from surgery.prompt import build_prompt, label_token_ids
from surgery.task.generator import LABELS


# ---------------------------------------------------------------- LoRA

class LoRALinear(nn.Module):
    """base(x) + scale * B(A x); A (r, in) kaiming-uniform, B (out, r) zero."""

    def __init__(self, base: nn.Linear, r: int, scale: float = 2.0):
        super().__init__()
        self.base_layer, self.r, self.scale = base, r, float(scale)
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base_layer(x) + self.scale * ((x @ self.A.T) @ self.B.T)


MLP_MODULES = ("gate_proj", "up_proj", "down_proj")      # live on layer.mlp; the rest on layer.self_attn


class _RestoreLoRA:
    def __init__(self, swaps):
        self.swaps = swaps

    def remove(self):
        for attn, name, original in self.swaps:
            setattr(attn, name, original)
        self.swaps = []


class FusedQKVLoRA(nn.Module):
    """LoRA on a fused query_key_value projection (GPT-NeoX), updating only the q and v rows. The fused output is
    viewed as (n_heads, 3, head_size) per token, so q and v are interleaved per head; `mask` ([out]) is 1 on those rows.
    base(x) + scale * mask * (B (A x)); the k rows never change (no gradient reaches B's k rows either)."""

    def __init__(self, base: nn.Linear, r: int, n_heads: int, scale: float = 2.0):
        super().__init__()
        assert base.out_features % (3 * n_heads) == 0
        self.base_layer, self.r, self.scale = base, r, float(scale)
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.register_buffer("mask", qv_row_mask(n_heads, base.out_features // (3 * n_heads)).to(dev, dt), persistent=False)

    def delta(self):
        return self.scale * (self.mask[:, None] * self.B) @ self.A

    def forward(self, x):
        return self.base_layer(x) + self.scale * (((x @ self.A.T) @ self.B.T) * self.mask)


def qv_row_mask(n_heads: int, head_size: int) -> torch.Tensor:
    """[3 * n_heads * head_size] float mask: 1 on the q and v rows of a GPT-NeoX fused QKV output, 0 on k."""
    m = torch.zeros(n_heads, 3, head_size)
    m[:, 0], m[:, 2] = 1.0, 1.0
    return m.reshape(-1)


def add_lora_fused_qkv(model, layer_ids, r: int = 1, scale: float = 2.0) -> Handle:
    """As `add_lora` for GPT-NeoX: a FusedQKVLoRA on `layers[i].attention.query_key_value` (q and v rows only)."""
    model.requires_grad_(False)
    n_heads = model.config.num_attention_heads
    swaps, loras = [], []
    for i in layer_ids:
        attn = model.gpt_neox.layers[i].attention
        original = attn.query_key_value
        lora = FusedQKVLoRA(original, r, n_heads, scale)
        attn.query_key_value = lora
        swaps.append((attn, "query_key_value", original))
        loras.append(lora)
    handle = Handle([_RestoreLoRA(swaps)])
    handle.modules = nn.ModuleList(loras)
    return handle


def add_lora(model, layer_ids, modules=("q_proj", "v_proj"), r: int = 1, scale: float = 2.0) -> Handle:
    """Freeze the model and swap LoRA modules in on the original layer indices. The handle has `.modules`
    (the LoRALinear list); remove() restores the originals."""
    model.requires_grad_(False)
    swaps, loras = [], []
    for i in layer_ids:
        layer = model.model.layers[i]
        for m in modules:
            attn = layer.mlp if m in MLP_MODULES else layer.self_attn
            original = getattr(attn, m)
            lora = LoRALinear(original, r, scale)
            setattr(attn, m, lora)
            swaps.append((attn, m, original))
            loras.append(lora)
    handle = Handle([_RestoreLoRA(swaps)])
    handle.modules = nn.ModuleList(loras)
    return handle


@torch.no_grad()
def merge_lora(handle: Handle) -> int:
    """Fold scale * B @ A into each wrapped base layer's weight in place and put the base `nn.Linear` back on
    its module; remove() is a no-op afterwards. Mutates the model: callers start every run from a fresh one.
    Returns the number of modules merged."""
    n = 0
    for h in handle._handles:
        for attn, name, original in getattr(h, "swaps", []):
            lora = getattr(attn, name)
            if lora is not original:
                d = lora.delta() if hasattr(lora, "delta") else lora.scale * (lora.B @ lora.A)
                original.weight.add_(d.to(original.weight.dtype))
                n += 1
            setattr(attn, name, original)
        if hasattr(h, "swaps"):
            h.swaps = []
    return n


# ---------------------------------------------------------------- LoReFT

def orthonormal_rows(r: int, d: int, init_rows=None, seed: int = 0) -> torch.Tensor:
    """(r, d) with orthonormal rows. The first rows follow init_rows (Gram-Schmidt in order, truncated to r),
    the rest are a random orthonormal completion."""
    g = torch.Generator().manual_seed(seed)
    parts = []
    if init_rows is not None:
        parts.append(torch.as_tensor(init_rows, dtype=torch.float64)[:r])
    k = sum(len(p) for p in parts)
    if k < r:
        parts.append(torch.randn(r - k, d, generator=g, dtype=torch.float64))
    M = torch.cat(parts)
    Q, Rm = torch.linalg.qr(M.T)  # columns of Q span the rows of M in order
    sign = torch.sign(torch.diagonal(Rm))
    sign[sign == 0] = 1
    Q = Q * sign.unsqueeze(0)
    return Q.T.contiguous().float()


class LoReFT(nn.Module):
    """h + R^T (W h + b - R h). R (r, d) keeps orthonormal rows; W starts as a copy of R and b at zero."""

    def __init__(self, d: int, r: int, init_rows=None, seed: int = 0):
        super().__init__()
        rows = orthonormal_rows(r, d, init_rows, seed)
        self.R = nn.Linear(d, r, bias=False)
        parametrizations.orthogonal(self.R, orthogonal_map="cayley")  # householder degenerates at a QR start
        self.R.weight = rows
        self.proj = nn.Linear(d, r)
        with torch.no_grad():
            self.proj.weight.copy_(rows)
            self.proj.bias.zero_()

    @property
    def n_params(self) -> int:
        return self.R.weight.shape[0] * (2 * self.R.weight.shape[1] + 1)

    def forward(self, h):
        R = self.R.weight
        return h + (self.proj(h) - h @ R.T) @ R


def apply_loreft(model, layer: int, module: LoReFT, positions=None) -> Handle:
    """Forward hook on layer `layer`: edit the last position (the last prompt token with left padding, or the
    only token when the prefix is cached). `positions`: callable giving a LongTensor (B,) for right-padded
    batches (canaries), as in `steer.apply_steer`."""
    def hook(m, args, output):
        out = output[0] if isinstance(output, tuple) else output
        new = out.clone()
        if positions is None:
            new[:, -1] = module(out[:, -1].float()).to(out.dtype)
        else:
            pos = positions().to(out.device)
            rows = torch.arange(out.shape[0], device=out.device)
            new[rows, pos] = module(out[rows, pos].float()).to(out.dtype)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

    return Handle([model.model.layers[layer].register_forward_hook(hook)])


# ---------------------------------------------------------------- forwards

def encode(model, tok, records):
    return tok([build_prompt(tok, r["prompt"]) for r in records], return_tensors="pt", padding=True).to(model.device)


def label_ids(model, tok):
    return torch.tensor(label_token_ids(tok), device=model.device)


def full_logits(model, enc, ids):
    """(B, 6) label logits from one full forward."""
    return model(**enc, logits_to_keep=1).logits[:, -1, :][:, ids].float()


def cached_logits(model, enc, ids, edit=None):
    """Same logits via the prefix cache: the prefix runs once without gradients, then only the last token
    runs. `edit()` returns a Handle (e.g. `apply_loreft`) installed for the last-token run only, so the
    cached prefix is never edited. Hooks already on the model (the surgery spec) cover both runs."""
    with torch.no_grad():
        past = model(input_ids=enc["input_ids"][:, :-1], attention_mask=enc["attention_mask"][:, :-1],
                     use_cache=True, logits_to_keep=1).past_key_values
    handle = edit() if edit else Handle([])
    try:
        lg = model(input_ids=enc["input_ids"][:, -1:], attention_mask=enc["attention_mask"], past_key_values=past,
                   use_cache=True, logits_to_keep=1).logits[:, -1, :]
    finally:
        handle.remove()
    return lg[:, ids].float()


# ---------------------------------------------------------------- data

def split_discovery(records, seed: int, n_heldout: int = 200, n_train: int = 800, heldout_seed: int = 100):
    """(heldout, train): the held-out set takes whole templates drawn with Random(100) until it has at least
    n_heldout examples (then the first n_heldout); train is a seeded sample from the other templates."""
    templates = sorted({r["template_id"] for r in records})
    random.Random(heldout_seed).shuffle(templates)
    held, count = set(), 0
    for t in templates:
        if count >= n_heldout:
            break
        held.add(t)
        count += sum(r["template_id"] == t for r in records)
    heldout = [r for r in records if r["template_id"] in held][:n_heldout]
    rest = [r for r in records if r["template_id"] not in held]
    train = random.Random(seed).sample(rest, min(n_train, len(rest)))
    return heldout, train


# ---------------------------------------------------------------- training

def count_trainable(x) -> int:
    mods = x if isinstance(x, (list, tuple)) else [x]
    return sum(p.numel() for m in mods for p in m.parameters() if p.requires_grad)


def adapter_state(params: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in params.state_dict().items() if "base_layer." not in k}


def load_adapter_state(params: nn.Module, state: dict) -> None:
    params.load_state_dict(state, strict=False)


@torch.no_grad()
def heldout_accuracy(forward_fn, records, batch_size: int = 8) -> float:
    y = torch.tensor([LABELS.index(r["label"]) for r in records])
    pred = [forward_fn(records[i:i + batch_size]).argmax(-1).cpu() for i in range(0, len(records), batch_size)]
    return float((torch.cat(pred) == y).float().mean())


def train_adapter(model, tok, train_recs, heldout_recs, params: nn.Module, forward_fn, steps: int,
                  lr: float = 1e-3, batch_size: int = 8, seed: int = 0, n_evals: int = 4):
    """One pass of AdamW (10% warm-up, linear decay) on cross-entropy over the six label logits.
    Returns (best_state, curve); the best of n_evals evenly spaced held-out evaluations is kept."""
    g = torch.Generator().manual_seed(seed)
    trainable = [p for p in params.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    warm = max(1, round(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    eval_at = sorted({max(1, round(steps * (i + 1) / n_evals)) for i in range(n_evals)})
    y_all = torch.tensor([LABELS.index(r["label"]) for r in train_recs])
    order, best, curve, eval_s = [], None, [], 0.0
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        if len(order) < batch_size:
            order = torch.randperm(len(train_recs), generator=g).tolist() + order
        idx, order = order[:batch_size], order[batch_size:]
        loss = nn.functional.cross_entropy(forward_fn([train_recs[i] for i in idx]), y_all[idx].to(model.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step in eval_at:
            te = time.perf_counter()
            acc = heldout_accuracy(forward_fn, heldout_recs, batch_size)
            eval_s += time.perf_counter() - te
            curve.append({"step": step, "heldout_accuracy": acc, "train_loss": float(loss.detach()),
                          "seconds": time.perf_counter() - t0, "eval_seconds": eval_s})
            if best is None or acc > best[0]:
                best = (acc, adapter_state(params), step)
            _free(model)
            print(f"  step {step}/{steps} loss {float(loss.detach()):.3f} heldout {acc:.3f}", flush=True)
    return best[1], curve
