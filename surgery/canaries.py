"""Canaries (500 multiple-choice items) and held-out perplexity, for detecting collateral damage.

Items are scored by option log-likelihood (no generation). Never optimize on these.
Callers wrap calls in `surgery.hooks.apply_spec(model, spec)` to score under an intervention.

CLI: uv run python -m surgery.canaries [--model Qwen/Qwen3-0.6B] [--dtype fp32]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CANARIES_PATH = ROOT / "data" / "canaries500.jsonl"
PPL_PATH = ROOT / "data" / "ppl_heldout.txt"
PPL_SHA256 = "26e11645fc4d8f3ffe8190531e1510e686535a8535ef51866ac1f6d891eb9abd"
CATEGORIES = ["syntax", "factual", "reasoning", "instruction", "code"]


def load_canaries(path: str | Path = CANARIES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_ppl_text(path: str | Path = PPL_PATH, verify: bool = True) -> str:
    raw = Path(path).read_bytes()
    if verify and hashlib.sha256(raw).hexdigest() != PPL_SHA256:
        raise ValueError(f"{path} does not match PPL_SHA256")
    return raw.decode("utf-8")


def _question(prompt: str) -> str:
    return f"Question: {prompt}\nAnswer:"


@torch.no_grad()
def score_canaries(model, tok, items: list[dict], batch_size: int = 8, on_batch=None) -> dict:
    """Sum of option-token log-probs given the prompt; argmax over options. Prompt and option are tokenized
    separately (option with a leading space) and the ids concatenated, so the split is exact.
    `on_batch(n_prompt_list)`, if given, is called with each row's prompt length just before the forward."""
    seqs = []  # (item index, option index, ids, n_prompt)
    for i, it in enumerate(items):
        p = tok(_question(it["prompt"]), add_special_tokens=False)["input_ids"]
        for j, o in enumerate(it["options"]):
            seqs.append((i, j, p + tok(" " + o, add_special_tokens=False)["input_ids"], len(p)))
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][2]))  # similar lengths per batch
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    lp = [0.0] * len(seqs)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        L = max(len(seqs[k][2]) for k in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for r, k in enumerate(idx):
            s = seqs[k][2]
            ids[r, :len(s)] = torch.tensor(s)
            mask[r, :len(s)] = 1
        if on_batch is not None:
            on_batch([seqs[k][3] for k in idx])
        logits = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device)).logits
        logp = torch.log_softmax(logits.float(), dim=-1).cpu()
        for r, k in enumerate(idx):
            _, _, s, n_p = seqs[k]
            tgt = torch.tensor(s[n_p:])
            pos = torch.arange(n_p - 1, len(s) - 1)
            lp[k] = float(logp[r, pos, tgt].sum())
    scores = [[0.0] * len(it["options"]) for it in items]
    for (i, j, _, _), v in zip(seqs, lp):
        scores[i][j] = v
    pred = [max(range(len(s)), key=s.__getitem__) for s in scores]
    correct = [p == it["answer"] for p, it in zip(pred, items)]
    per = {}
    for c in sorted({it["category"] for it in items}, key=lambda c: CATEGORIES.index(c) if c in CATEGORIES else 99):
        ok = [x for x, it in zip(correct, items) if it["category"] == c]
        per[c] = sum(ok) / len(ok)
    return {"accuracy": sum(correct) / len(correct) if correct else float("nan"), "per_category": per,
            "correct": correct, "pred": pred, "scores": scores, "n": len(items)}


def eligible(base: dict, got: dict) -> tuple[bool, dict]:
    """M3 canary gate: overall drop <= 2 points, syntax/factual/reasoning/code drops <= 5 points each."""
    # Rounded so a drop of exactly the limit (e.g. 5 of 100 items -> 5.000000000000004) passes, as written.
    drops = {"overall": round(100 * (base["accuracy"] - got["accuracy"]), 6)}
    for c in CATEGORIES:
        drops[c] = round(100 * (base["per_category"][c] - got["per_category"][c]), 6)
    ok = drops["overall"] <= 2 and all(drops[c] <= 5 for c in ("syntax", "factual", "reasoning", "code"))
    return ok, drops


@torch.no_grad()
def perplexity(model, tok, text: str, window: int = 1024, stride: int = 512) -> dict:
    """Strided sliding-window perplexity; each token is scored once, by the first window that reaches it
    (the first token of the text has no context and is not scored)."""
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    n = ids.size(1)
    nll, count, prev_end = 0.0, 0, 0
    for begin in range(0, n, stride):
        end = min(begin + window, n)
        trg = end - prev_end  # tokens not scored by earlier windows
        x = ids[:, begin:end].to(model.device)
        logits = model(input_ids=x).logits[0].float()
        logp = torch.log_softmax(logits[:-1], dim=-1)
        tgt = x[0, 1:]
        k = min(trg, tgt.numel())  # score only the last k targets of this window
        sel = torch.arange(tgt.numel() - k, tgt.numel(), device=x.device)
        nll += float(-logp[sel, tgt[sel]].sum())
        count += k
        prev_end = end
        if end == n:
            break
    return {"ppl": float(torch.exp(torch.tensor(nll / count))), "n_tokens": count}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args(argv)
    from surgery.eval import load_model
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
    model, tok = load_model(a.model, dtype=dt)
    t0 = time.time()
    res = score_canaries(model, tok, load_canaries(), a.batch_size)
    t1 = time.time()
    print(f"canaries: accuracy {res['accuracy']:.3f} (n={res['n']}, {t1 - t0:.0f}s)")
    for c, v in res["per_category"].items():
        print(f"  {c:12s} {v:.3f}")
    pp = perplexity(model, tok, load_ppl_text())
    print(f"perplexity: {pp['ppl']:.3f} on {pp['n_tokens']} tokens ({time.time() - t1:.0f}s)")


if __name__ == "__main__":
    main()
