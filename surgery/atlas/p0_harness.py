"""P0 harness: model adapter, last-token raw residual states, cached-prefix patched runs, use/arrival rows."""
from __future__ import annotations

import copy
import os
import time
from collections import defaultdict

import numpy as np
import torch

from surgery.atlas.p0_measures import probe_subspace


def kv_row_bytes(cfg, n_tokens: int, itemsize: int) -> int:
    """Bytes of one row's KV cache (keys and values, every layer) for a prefix of n_tokens."""
    n_heads = cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", None) or n_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    return 2 * cfg.num_hidden_layers * n_tokens * kv_heads * head_dim * itemsize


def kv_chunk(cfg, n_tokens: int, itemsize: int, chunk: int, budget_bytes: float) -> int:
    """`chunk` capped so chunk x one row's KV cache stays within budget_bytes; at least 1."""
    return max(1, min(chunk, int(budget_bytes // max(1, kv_row_bytes(cfg, n_tokens, itemsize)))))


class Adapter:
    """Base causal LM (GPTNeoX, Llama, Qwen2, Qwen3, OLMo, OLMo2). Block 0 = embedding output; block L = output of layer L-1."""

    def __init__(self, model_id=None, device="cpu", dtype=torch.float32, model=None, tok=None):
        t0 = time.time()
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.model, self.tok, self.device = model.to(device).eval(), tok, device
        mt = model.config.model_type
        if mt == "gpt_neox":
            self.layers, self.final_norm, self.unembed = model.gpt_neox.layers, model.gpt_neox.final_layer_norm, model.get_output_embeddings()  # embed_out / lm_head
        elif mt in ("llama", "qwen2", "qwen3", "olmo", "olmo2"):
            self.layers, self.final_norm, self.unembed = model.model.layers, model.model.norm, model.get_output_embeddings()
        else:
            raise ValueError(f"unsupported architecture {mt}")
        self.embed = model.get_input_embeddings()
        self.n_blocks = len(self.layers)
        self.d = model.config.hidden_size
        self.load_seconds = time.time() - t0

    # ---- tokens ----
    def encode(self, prompts):
        return [self.tok(p, add_special_tokens=False).input_ids for p in prompts]

    def option_ids(self, options):
        return [self.tok(o, add_special_tokens=False).input_ids[0] for o in options]

    def filter_items(self, items):
        """Drop items whose item / answer twin / surface twin token lengths differ (or whose last tokens differ)."""
        kept, dropped = [], 0
        for x in items:
            a, b, c = self.encode([x["prompt"], x["ans_twin"]["prompt"], x["surf_twin"]["prompt"]])
            if len(a) == len(b) == len(c) and a[-1] == b[-1] == c[-1]:
                kept.append(x)
            else:
                dropped += 1
        return kept, dropped

    # ---- states ----
    @torch.no_grad()
    def collect_ids(self, ids_list, opt_ids, batch_size=32):
        """Last-token raw residual at block 0..B and option logits. Batches share one token length (no padding)."""
        n, B = len(ids_list), self.n_blocks
        states = np.zeros((n, B + 1, self.d), dtype=np.float32)
        logits = np.zeros((n, len(opt_ids)), dtype=np.float32)
        by_len = defaultdict(list)
        for i, ids in enumerate(ids_list):
            by_len[len(ids)].append(i)
        for T, idx in by_len.items():
            for s in range(0, len(idx), batch_size):
                sel = idx[s:s + batch_size]
                x = torch.tensor([ids_list[i] for i in sel], device=self.device)
                cap, hooks = {}, []
                hooks.append(self.embed.register_forward_hook(lambda m, i, o: cap.__setitem__(0, o[:, -1].float())))
                for L, layer in enumerate(self.layers, start=1):
                    def f(m, i, o, L=L):
                        h = o[0] if isinstance(o, tuple) else o
                        cap[L] = h[:, -1].float()
                    hooks.append(layer.register_forward_hook(f))
                try:
                    out = self.model(x).logits[:, -1, opt_ids].float()
                finally:
                    for h in hooks:
                        h.remove()
                states[sel] = torch.stack([cap[L] for L in range(B + 1)], 1).cpu().numpy()
                logits[sel] = out.cpu().numpy()
        return states, logits

    def collect(self, prompts, opt_ids, batch_size=32):
        return self.collect_ids(self.encode(prompts), opt_ids, batch_size)

    # ---- cached-prefix patched run ----
    @torch.no_grad()
    def patched_logits(self, ids, rows, opt_ids, chunk=128):
        """Prefix ids[:-1] once with a cache; then only the last token, batched over rows. Each row is (L, vec):
        replace the last-token output of block L (layer L-1) with vec; (None, None) = unpatched. -> [len(rows), 4].
        The chunk is capped so the chunk's copies of the KV cache fit ATLAS_PATCH_KV_GB (default 4 GB); the chunk only
        batches rows, so the result does not depend on it beyond float noise."""
        itemsize = next(self.model.parameters()).element_size()
        budget = float(os.environ.get("ATLAS_PATCH_KV_GB", "4")) * 2**30
        chunk = kv_chunk(self.model.config, len(ids), itemsize, chunk, budget)
        x = torch.tensor([ids], device=self.device)
        cache = self.model(x[:, :-1], use_cache=True).past_key_values
        last = x[:, -1:]
        out = []
        for s in range(0, len(rows), chunk):
            part = rows[s:s + chunk]
            m = len(part)
            c = copy.deepcopy(cache)
            c.batch_repeat_interleave(m)
            by_L = defaultdict(list)
            for j, (L, v) in enumerate(part):
                if L is not None:
                    by_L[L].append((j, v))
            hooks = []
            for L, lst in by_L.items():
                idx = torch.tensor([j for j, _ in lst], device=self.device)
                vecs = torch.stack([torch.as_tensor(v, dtype=torch.float32) for _, v in lst]).to(self.device)

                def f(mod, i, o, idx=idx, vecs=vecs):
                    h = (o[0] if isinstance(o, tuple) else o).clone()
                    h[idx, -1] = vecs.to(h.dtype)
                    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
                hooks.append(self.layers[L - 1].register_forward_hook(f))
            try:
                lg = self.model(last.repeat(m, 1), past_key_values=c, use_cache=True).logits[:, -1, opt_ids].float()
            finally:
                for h in hooks:
                    h.remove()
            out.append(lg.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(opt_ids)), dtype=np.float32)

    # ---- answer-token frequency (C-scan, proposal 4.3 item 8) ----
    @torch.no_grad()
    def answer_logfreq(self, opt_ids):
        """Unigram log-probability of each option from a one-token "empty" prompt ([BOS], else [EOS]), two readings:
        "lm" = full forward, last position; "block0" = the prompt's embedding state through final norm + unembed.
        Both are log_softmax over the full vocabulary."""
        pid = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        if pid is None:
            raise ValueError("tokenizer has neither a BOS nor an EOS token")
        opt = [int(o) for o in opt_ids]
        x = torch.tensor([[pid]], device=self.device)
        lm = torch.log_softmax(self.model(x).logits[0, -1].float(), -1)
        h = self.embed(x)[:, -1]
        b0 = torch.log_softmax(self.unembed(self.final_norm(h))[0].float(), -1)
        out = {"prompt_token_id": int(pid)}
        for k, lp in (("lm", lm), ("block0", b0)):
            v = [float(lp[o]) for o in opt]
            out[k] = {"values": v, "mean": float(np.mean(v))}
        return out

    # ---- logit lens ----
    @torch.no_grad()
    def lens_logits(self, states, opt_ids, batch_size=64):
        """Final norm + unembed on stored raw states, restricted to the options: [n, B+1, 4]."""
        out = []
        for s in range(0, len(states), batch_size):
            h = torch.as_tensor(states[s:s + batch_size], device=self.device).to(next(self.model.parameters()).dtype)
            out.append(self.unembed(self.final_norm(h))[..., opt_ids].float().cpu().numpy())
        return np.concatenate(out)


def swap_subspace(h_item, h_twin, Q):
    """h' = h_item - Q Q^T h_item + Q Q^T h_twin, Q = [d, k] orthonormal."""
    return h_item - Q @ (Q.T @ h_item) + Q @ (Q.T @ h_twin)


def make_rows(h_item, h_ans, h_surf, Qprobe, Qrand, B):
    """Rows for blocks L = 1..B-1, in a fixed order per block: probe-ans, R random-ans, probe-surf, whole-ans,
    whole-surf. h_*: [B+1, d]; Qprobe: [B+1, d, k] (or dict-like by L); Qrand: [B+1, R, d, k]."""
    rows = []
    R = Qrand.shape[1]
    for L in range(1, B):
        rows.append((L, swap_subspace(h_item[L], h_ans[L], Qprobe[L])))
        for r in range(R):
            rows.append((L, swap_subspace(h_item[L], h_ans[L], Qrand[L, r])))
        rows.append((L, swap_subspace(h_item[L], h_surf[L], Qprobe[L])))
        rows.append((L, h_ans[L]))
        rows.append((L, h_surf[L]))
    return rows


def split_rows(lg, B, R):
    """Inverse of make_rows ordering: -> dict of probe [K,4], rand [K,R,4], psurf [K,4], whole [K,4], wsurf [K,4]."""
    g = lg.reshape(B - 1, R + 4, -1)
    return {"probe": g[:, 0], "rand": g[:, 1:1 + R], "psurf": g[:, 1 + R], "whole": g[:, 2 + R], "wsurf": g[:, 3 + R]}


def probe_subspaces(fits, k=3):
    """Per block [B+1, d, k] from the (scale, coef) fits."""
    return np.stack([probe_subspace(s, c, k) for s, c in fits]).astype(np.float32)


def run_use_arrival(ad, ids, h_item, h_ans, h_surf, Qprobe, Qrand, opt_ids, chunk=128):
    rows = make_rows(h_item, h_ans, h_surf, Qprobe, Qrand, ad.n_blocks)
    lg = ad.patched_logits(ids, rows, opt_ids, chunk=chunk)
    return split_rows(lg, ad.n_blocks, Qrand.shape[1])


def analyze_task(ad, ids_tr, y_tr, g_tr, ids_te, y_te, ids_an, y_an, ids_sf, opt_ids,
                 n_perm=20, max_pairs=200, seed=0, n_flip=10000, n_draw=1000, log=None, bow_acc=None):
    """The whole P0 pipeline for one task on one adapter (real model or toy). ids_* are token-id lists; twins
    are aligned with the test items (ids_an[i] is the answer twin of ids_te[i], y_an[i] its answer)."""
    from surgery.atlas import p0_measures as M
    B = ad.n_blocks
    say = log or (lambda *_: None)
    y_tr, y_te, y_an = np.asarray(y_tr), np.asarray(y_te), np.asarray(y_an)
    tim = {}
    t0 = time.time()
    S_tr, _ = ad.collect_ids(ids_tr, opt_ids)
    S_te, L_te = ad.collect_ids(ids_te, opt_ids)
    S_an, L_an = ad.collect_ids(ids_an, opt_ids)
    S_sf, _ = ad.collect_ids(ids_sf, opt_ids)
    tim["states_s"] = time.time() - t0
    out = {"n_train": len(ids_tr), "n_test": len(ids_te), "test_item_acc": float(np.mean(L_te.argmax(1) == y_te)),
           "n_blocks": B}

    t0 = time.time()
    pc = M.probe_curve(S_tr, y_tr, g_tr, S_te, y_te, n_perm=n_perm, seed=seed, bow_acc=bow_acc)
    fits = pc.pop("fits")
    tim["probe_s"] = time.time() - t0
    tim["probe_fits"] = pc["n_fits"]
    say("probe done", tim["probe_s"])

    t0 = time.time()
    td = M.twin_divergence(S_te, S_an, S_sf, n_draw=n_draw, seed=seed)
    tim["twin_s"] = time.time() - t0

    lens = ad.lens_logits(S_te, opt_ids)
    out["logit_lens_acc"] = M.logit_lens_acc(lens, y_te)

    mask = M.pair_mask(L_te, L_an, y_te, y_an)
    sel = np.flatnonzero(mask)
    n_qual = len(sel)
    if n_qual > max_pairs:
        sel = np.sort(np.random.default_rng(seed).choice(sel, max_pairs, replace=False))
    out["n_pairs_qualifying"], out["n_pairs_used"] = int(n_qual), int(len(sel))

    t0 = time.time()
    use = arr = None
    if len(sel) > 0:
        Qp = probe_subspaces(fits)
        Qr = M.random_subspaces(ad.d, B, seed=seed)
        res = []
        for i in sel:
            res.append(run_use_arrival(ad, ids_te[i], S_te[i], S_an[i], S_sf[i], Qp, Qr, opt_ids))
        yi, yt = y_te[sel], y_an[sel]
        st = lambda k: np.stack([r[k] for r in res])  # [P, K(,R), 4]
        d_item = M.delta(L_te[sel], yi, yt)
        d_twin = M.delta(L_an[sel], yi, yt)
        dd = lambda k: M.delta(st(k), yi[:, None], yt[:, None])
        d_rand = M.delta(st("rand"), yi[:, None, None], yt[:, None, None]).swapaxes(1, 2)
        use = M.use_scores(d_item, d_twin, dd("probe"), d_rand, dd("psurf"), n_flip=n_flip, seed=seed)
        arr = M.arrival_scores(d_item, d_twin, dd("whole"), dd("wsurf"), B)
    tim["use_arrival_s"] = time.time() - t0

    out["probe"], out["twin"], out["use"], out["arrival"] = pc, td, use, arr
    on = {}
    on["probe"] = M.onset(pc["selectivity"], B) if pc["valid"] else None
    on["twin"] = M.onset(td["t"], B) if td["valid"] else None
    on["use"] = M.onset(use["use"], B) if use and use["valid"] else None
    on["arrival"] = M.onset(arr["arrival"], B) if arr and arr["valid"] else None
    out["onset"] = on
    out["valid"] = {"probe": pc["valid"], "twin": td["valid"], "use": bool(use and use["valid"]),
                    "arrival": bool(arr and arr["valid"])}
    out["timing"] = tim
    return out
