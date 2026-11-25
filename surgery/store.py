"""Activation store: last-token residuals, logit lens, and mean sublayer outputs.

Layout of `out_dir` is specified in M3_PLAN.md (resid.npy, lens.npy, final_logits.npy,
means.pt, meta.json). Built with forward hooks over left-padded batches.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from surgery.eval import _free  # noqa: E402
from surgery.prompt import build_prompt, label_token_ids  # noqa: E402
from surgery.task.generator import LABELS  # noqa: E402


def short_name(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1]


def _first(x):
    return x[0] if isinstance(x, tuple) else x


@torch.no_grad()
def build_store(model, tok, records: list[dict], out_dir, batch_size: int = 8,
                meta_extra: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = model.config
    layers = model.model.layers
    L, d = len(layers), cfg.hidden_size
    H, KV = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or d // H
    N = len(records)
    dev = model.device
    ids = label_token_ids(tok)
    ids_t = torch.tensor(ids, device=dev)

    resid = np.lib.format.open_memmap(out_dir / "resid.npy", mode="w+", dtype=np.float16, shape=(N, L + 1, d))
    lens = np.lib.format.open_memmap(out_dir / "lens.npy", mode="w+", dtype=np.float32, shape=(N, L + 1, 6))
    final = np.zeros((N, 6), dtype=np.float32)

    # per-batch sums are fp32 on the device (MPS has no fp64); the running total is fp64 on CPU
    s_attn = torch.zeros(L, d, dtype=torch.float64)
    s_mlp = torch.zeros(L, d, dtype=torch.float64)
    s_heads = torch.zeros(L, H, hd, dtype=torch.float64)
    count = 0
    state = {"mask": None, "last": [None] * (L + 1)}

    def embed_hook(m, i, o):
        state["last"][0] = _first(o)[:, -1].detach().float()

    def make_layer_hook(i):
        def hook(m, inp, o):
            state["last"][i + 1] = _first(o)[:, -1].detach().float()
        return hook

    def make_sum_hook(buf, i):
        def hook(m, inp, o):
            x = _first(o).detach().float()
            buf[i] += (x * state["mask"][..., None]).sum((0, 1)).cpu().double()
        return hook

    def make_head_hook(i):
        def hook(m, args):
            z = args[0].detach().float()
            z = z.reshape(z.shape[0], z.shape[1], H, hd)
            s_heads[i] += (z * state["mask"][..., None, None]).sum((0, 1)).cpu().double()
        return hook

    handles = [model.model.embed_tokens.register_forward_hook(embed_hook)]
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_layer_hook(i)))
        handles.append(layer.self_attn.register_forward_hook(make_sum_hook(s_attn, i)))
        handles.append(layer.mlp.register_forward_hook(make_sum_hook(s_mlp, i)))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_head_hook(i)))

    W = model.lm_head.weight[ids_t].float()
    try:
        for b in range(0, N, batch_size):
            batch = records[b:b + batch_size]
            enc = tok([build_prompt(tok, r["prompt"]) for r in batch],
                      return_tensors="pt", padding=True).to(dev)
            state["mask"] = enc["attention_mask"].float()
            count += int(enc["attention_mask"].sum())
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
            final[b:b + len(batch)] = logits[:, ids_t].float().cpu().numpy()
            h = torch.stack(state["last"], dim=1)  # (B, L+1, d) fp32
            normed = model.model.norm(h)
            lens[b:b + len(batch)] = (normed @ W.T).cpu().numpy()
            resid[b:b + len(batch)] = h.cpu().numpy().astype(np.float16)
            state["last"] = [None] * (L + 1)
            del enc, logits, h, normed
            _free(model)
    finally:
        for hnd in handles:
            hnd.remove()
    resid.flush()
    lens.flush()
    del resid, lens
    np.save(out_dir / "final_logits.npy", final)
    torch.save({"attn": (s_attn / count).float().cpu(), "mlp": (s_mlp / count).float().cpu(),
                "heads": (s_heads / count).float().cpu()}, out_dir / "means.pt")
    y = [LABELS.index(r["label"]) for r in records]
    meta = {
        "model": getattr(cfg, "_name_or_path", None), "variant": None, "split": "discovery",
        "data_path": None, "n": N, "n_layers": L, "d_model": d, "n_heads": H, "n_kv_heads": KV,
        "head_dim": hd, "ids": [r["id"] for r in records],
        "template_ids": [r["template_id"] for r in records], "y": y,
        "pred": final.argmax(-1).tolist(), "label_token_ids": ids,
    }
    meta.update(meta_extra or {})
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta
