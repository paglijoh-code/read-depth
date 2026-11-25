"""M9 atlas measurements: activation store, whole-block lesion sweep, block influence and perplexity under skips.

The store is `surgery/store.py` with the atlas prompt and a per-task option set instead of the six planner labels.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from surgery.atlas import spec  # noqa: E402
from surgery.eval import _free  # noqa: E402
from surgery.hooks import apply_spec  # noqa: E402

BI_WINDOW = 1024
PPL_TOKENS = 16384


def _first(x):
    return x[0] if isinstance(x, tuple) else x


def _write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _encode(tok, task, records, device):
    return tok([spec.build_prompt(tok, task, r["prompt"]) for r in records], return_tensors="pt", padding=True).to(device)


@torch.no_grad()
def build_store(model, tok, task: str, records: list[dict], out_dir, batch_size: int = 8,
                meta_extra: dict | None = None) -> dict:
    """Writes resid.npy (N, L+1, d) fp16, lens.npy (N, L+1, n_options) fp32, final_logits.npy (N, n_options),
    means.pt {attn, mlp} (L, d) and meta.json under `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = model.model.layers
    L, d = len(layers), model.config.hidden_size
    N = len(records)
    dev = model.device
    ids = spec.option_token_ids(tok, task)
    n_opt = len(ids)
    ids_t = torch.tensor(ids, device=dev)

    resid = np.lib.format.open_memmap(out_dir / "resid.npy", mode="w+", dtype=np.float16, shape=(N, L + 1, d))
    lens = np.lib.format.open_memmap(out_dir / "lens.npy", mode="w+", dtype=np.float32, shape=(N, L + 1, n_opt))
    final = np.zeros((N, n_opt), dtype=np.float32)
    s_attn = torch.zeros(L, d, dtype=torch.float64)
    s_mlp = torch.zeros(L, d, dtype=torch.float64)
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

    handles = [model.model.embed_tokens.register_forward_hook(embed_hook)]
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_layer_hook(i)))
        handles.append(layer.self_attn.register_forward_hook(make_sum_hook(s_attn, i)))
        handles.append(layer.mlp.register_forward_hook(make_sum_hook(s_mlp, i)))

    W = model.lm_head.weight[ids_t].float()
    try:
        for b in range(0, N, batch_size):
            batch = records[b:b + batch_size]
            enc = _encode(tok, task, batch, dev)
            state["mask"] = enc["attention_mask"].float()
            count += int(enc["attention_mask"].sum())
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
            final[b:b + len(batch)] = logits[:, ids_t].float().cpu().numpy()
            h = torch.stack(state["last"], dim=1)  # (B, L+1, d) fp32
            lens[b:b + len(batch)] = (model.model.norm(h) @ W.T).cpu().numpy()
            resid[b:b + len(batch)] = h.cpu().numpy().astype(np.float16)
            state["last"] = [None] * (L + 1)
            del enc, logits, h
            _free(model)
    finally:
        for hnd in handles:
            hnd.remove()
    resid.flush()
    lens.flush()
    del resid, lens
    np.save(out_dir / "final_logits.npy", final)
    torch.save({"attn": (s_attn / count).float().cpu(), "mlp": (s_mlp / count).float().cpu()}, out_dir / "means.pt")
    answers = [r["answer"] for r in records]
    meta = {
        "model": getattr(model.config, "_name_or_path", None), "task": task, "split": "discovery",
        "n": N, "n_layers": L, "d_model": d, "option_token_ids": ids,
        "ids": [r["id"] for r in records], "template_ids": [r["template_id"] for r in records],
        "answers": answers, "rungs": [r.get("rung") for r in records], "chance": spec.chance(answers),
        "pred": final.argmax(-1).tolist(),
    }
    meta.update(meta_extra or {})
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta


@torch.no_grad()
def option_logits(model, tok, task: str, records: list[dict], batch_size: int = 8) -> np.ndarray:
    """(N, n_options) logits over the task's option first tokens, under whatever hooks are active."""
    ids_t = torch.tensor(spec.option_token_ids(tok, task), device=model.device)
    out = []
    for b in range(0, len(records), batch_size):
        enc = _encode(tok, task, records[b:b + batch_size], model.device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        out.append(logits[:, ids_t].float().cpu().numpy())
        del enc, logits
    _free(model)
    return np.concatenate(out) if out else np.zeros((0, len(ids_t)), dtype=np.float32)


def block_spec(i: int) -> dict:
    """Both sublayer outputs of block i replaced by their means at once."""
    return {"op": "compose", "specs": [{"op": "ablate", "site": s, "mode": "mean", "layers": [i]}
                                       for s in ("attn", "mlp")]}


def lesion_sweep(model, tok, task: str, records: list[dict], means, batch_size: int = 8,
                 path=None, blocks=None) -> dict:
    """Mean-ablate attn and mlp of each block in turn; accuracy over the task's options. Resumable row by row when
    `path` is given (a finished block is not recomputed). Returns {base_accuracy, n, rows: [{block, accuracy, drop}], drops}."""
    answers = np.array([r["answer"] for r in records])
    blocks = list(range(len(model.model.layers))) if blocks is None else list(blocks)
    ident = {"task": task, "ids": [r["id"] for r in records]}
    result = {**ident, "n": len(records), "base_accuracy": None, "rows": []}
    if path is not None and Path(path).exists():
        old = json.loads(Path(path).read_text())
        if any(old.get(k) != v for k, v in ident.items()):
            raise ValueError(f"incompatible existing lesion sweep at {path}")
        result = old
    if result["base_accuracy"] is None:
        result["base_accuracy"] = float((option_logits(model, tok, task, records, batch_size).argmax(-1) == answers).mean())
        if path is not None:
            _write_json(path, result)
    done = {row["block"] for row in result["rows"]}
    for i in blocks:
        if i in done:
            continue
        t0 = time.perf_counter()
        with apply_spec(model, block_spec(i), means=means):
            acc = float((option_logits(model, tok, task, records, batch_size).argmax(-1) == answers).mean())
        result["rows"].append({"block": i, "accuracy": acc, "drop": result["base_accuracy"] - acc,
                               "seconds": time.perf_counter() - t0})
        if path is not None:
            _write_json(path, result)
    by_block = {row["block"]: row["drop"] for row in result["rows"]}
    result["drops"] = [by_block[i] for i in blocks if i in by_block]
    return result


def _token_ids(tok, text, max_tokens=None):
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return ids if max_tokens is None else ids[:, :max_tokens]


@torch.no_grad()
def block_influence(model, tok, text: str, window: int = BI_WINDOW) -> list[float]:
    """ShortGPT block influence per block: 1 - mean cosine(block input, block output) over every token of
    non-overlapping `window`-token pieces of `text` (a trailing piece of one token is dropped)."""
    layers = model.model.layers
    L = len(layers)
    cos_sum = torch.zeros(L, dtype=torch.float64)
    n_tok = 0

    def make_hook(i):
        def hook(module, args, kwargs, output):
            x = args[0] if args else kwargs["hidden_states"]
            y = _first(output)
            c = torch.nn.functional.cosine_similarity(x.float(), y.float(), dim=-1)
            cos_sum[i] += float(c.sum())
        return hook

    handles = [layer.register_forward_hook(make_hook(i), with_kwargs=True) for i, layer in enumerate(layers)]
    ids = _token_ids(tok, text)
    try:
        for b in range(0, ids.size(1), window):
            x = ids[:, b:b + window]
            if x.size(1) < 2:
                continue
            model(input_ids=x.to(model.device))
            n_tok += x.size(1)
        _free(model)
    finally:
        for h in handles:
            h.remove()
    return (1.0 - cos_sum / max(n_tok, 1)).tolist()


@torch.no_grad()
def _ppl(model, ids, window: int = 1024, stride: int = 512) -> float:
    """Strided sliding-window perplexity over token ids (same scheme as surgery.canaries.perplexity)."""
    n = ids.size(1)
    nll, count, prev_end = 0.0, 0, 0
    for begin in range(0, n, stride):
        end = min(begin + window, n)
        trg = end - prev_end
        x = ids[:, begin:end].to(model.device)
        logp = torch.log_softmax(model(input_ids=x).logits[0].float()[:-1], dim=-1)
        tgt = x[0, 1:]
        k = min(trg, tgt.numel())
        sel = torch.arange(tgt.numel() - k, tgt.numel(), device=x.device)
        nll += float(-logp[sel, tgt[sel]].sum())
        count += k
        prev_end = end
        if end == n:
            break
    return float(np.exp(nll / count))


def ppl_drop(model, tok, text: str, max_tokens: int = PPL_TOKENS, blocks=None, done: dict | None = None,
             on_row=None) -> dict:
    """Perplexity with each block skipped minus the unmodified perplexity, on the first `max_tokens` tokens.
    `done` is {block: ppl} from an earlier partial run; `on_row(base_ppl, ppl_by_block)` is called after each block."""
    ids = _token_ids(tok, text, max_tokens)
    base = _ppl(model, ids)
    blocks = list(range(len(model.model.layers))) if blocks is None else list(blocks)
    ppl = {int(k): v for k, v in (done or {}).items()}
    for i in blocks:
        if i in ppl:
            continue
        with apply_spec(model, {"op": "skip", "layers": [i]}):
            ppl[i] = _ppl(model, ids)
        _free(model)
        if on_row is not None:
            on_row(base, ppl)
    return {"base_ppl": base, "n_tokens": int(ids.size(1)), "ppl": ppl,
            "drop": [ppl[i] - base for i in blocks]}


def abs_positions(mask: torch.Tensor, positions) -> torch.Tensor:
    """(B, S) absolute indices into a left-padded batch. `positions[i]` holds S token indices of record i counted in its own
    unpadded sequence (negative counts from the end, so -1 is the last prompt token)."""
    B, T = mask.shape
    n = mask.sum(1).long().cpu()
    out = torch.zeros(B, len(positions[0]), dtype=torch.long)
    for i, ps in enumerate(positions):
        for j, p in enumerate(ps):
            if not -int(n[i]) <= p < int(n[i]):
                raise ValueError(f"position {p} outside record {i} of {int(n[i])} tokens")
            out[i, j] = T - int(n[i]) + p if p >= 0 else T + p
    return out.to(mask.device)


@torch.no_grad()
def site_states(model, tok, task: str, records: list[dict], positions, batch_size: int = 8,
                dtype=np.float16) -> np.ndarray:
    """(N, S, L+1, d): the residual stream (embedding output, then each block's output, as `build_store`) at S chosen token
    positions per record. `positions[i]` lists record i's S token indices in its unpadded prompt (-1: the last token)."""
    layers = model.model.layers
    L, d = len(layers), model.config.hidden_size
    S = len(positions[0])
    out = np.zeros((len(records), S, L + 1, d), dtype=dtype)
    state = {"pos": None, "rows": None, "h": [None] * (L + 1)}

    def grab(i):
        def hook(m, inp, o):
            x = _first(o)
            state["h"][i] = x[state["rows"][:, None], state["pos"]].detach().float()   # (B, S, d)
        return hook

    handles = [model.model.embed_tokens.register_forward_hook(grab(0))]
    handles += [layer.register_forward_hook(grab(i + 1)) for i, layer in enumerate(layers)]
    try:
        for b in range(0, len(records), batch_size):
            batch = records[b:b + batch_size]
            enc = _encode(tok, task, batch, model.device)
            state["pos"] = abs_positions(enc["attention_mask"], positions[b:b + batch_size])
            state["rows"] = torch.arange(len(batch), device=model.device)
            model(**enc, logits_to_keep=1)
            out[b:b + len(batch)] = torch.stack(state["h"], dim=2).cpu().numpy().astype(dtype)
            state["h"] = [None] * (L + 1)
            del enc
            _free(model)
    finally:
        for hnd in handles:
            hnd.remove()
    return out
