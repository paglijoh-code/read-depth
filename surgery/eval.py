"""Evaluate Qwen3 on next-action records under an intervention spec.

CLI: uv run python -m surgery.eval --data var/data/eval500.jsonl [--n N] [--skip 3,4]
Writes var/results/<spec_hash>.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

from surgery.hooks import apply_spec, spec_hash  # noqa: E402
from surgery.prompt import build_prompt, label_token_ids  # noqa: E402
from surgery.task.generator import LABELS, read_jsonl  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-0.6B"
PAD_MULTIPLE = None  # set to 64 if unbucketed shapes prove to slow repeated passes
NO_SPEC = {"op": "skip", "layers": []}


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def parse_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; use one of {sorted(DTYPES)}")
    return DTYPES[dtype]


def dtype_name(dtype: torch.dtype) -> str:
    return {v: k for k, v in DTYPES.items()}.get(dtype, str(dtype))


def load_model(name: str = MODEL_NAME, device: str | None = None, dtype="fp32"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=parse_dtype(dtype)).to(device).eval()
    return model, tok


def _free(model) -> None:
    gc.collect()
    if str(model.device).startswith("mps"):
        torch.mps.empty_cache()
    elif str(model.device).startswith("cuda"):
        torch.cuda.empty_cache()


@torch.no_grad()
def label_logits(model, tok, prompts: list[str], batch_size: int = 8,
                 pad_multiple: int | None = PAD_MULTIPLE) -> torch.Tensor:
    """(len(prompts), 6) float32 CPU logits over the first token of each label, left-padded batches."""
    ids = torch.tensor(label_token_ids(tok), device=model.device)
    out = []
    kw = {"pad_to_multiple_of": pad_multiple} if pad_multiple else {}
    for i in range(0, len(prompts), batch_size):
        enc = tok([build_prompt(tok, p) for p in prompts[i:i + batch_size]],
                  return_tensors="pt", padding=True, **kw).to(model.device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        out.append(logits[:, ids].float().cpu())
        del enc, logits
    _free(model)
    return torch.cat(out)


def evaluate(model, tok, records: list[dict], spec: dict, batch_size: int = 8,
             data_path: str | None = None, pad_multiple: int | None = PAD_MULTIPLE) -> dict:
    with apply_spec(model, spec):
        lg = label_logits(model, tok, [r["prompt"] for r in records], batch_size, pad_multiple)
    pred = lg.argmax(-1).tolist()
    correct, diffs = [], []
    per = {l: [0, 0] for l in LABELS}
    for r, p, row in zip(records, pred, lg):
        y = LABELS.index(r["label"])
        w = LABELS.index(r["wrong_label"])
        correct.append(p == y)
        diffs.append(float(row[y] - row[w]))
        per[r["label"]][0] += p == y
        per[r["label"]][1] += 1
    n = len(records)
    return {
        "accuracy": sum(correct) / n, "n": n, "mean_logit_diff": sum(diffs) / n,
        "per_label_accuracy": {l: (c / t if t else None) for l, (c, t) in per.items()},
        "spec": spec, "spec_hash": spec_hash(spec),
        "model": getattr(model.config, "_name_or_path", None) or MODEL_NAME,
        "dtype": dtype_name(model.dtype), "data_path": data_path,
        "pred": pred, "logits": lg.tolist(),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="var/data/eval500.jsonl")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--skip", default="", help="comma list of layers to skip")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    a = ap.parse_args(argv)
    recs = read_jsonl(a.data)[: a.n]
    spec = {"op": "skip", "layers": [int(x) for x in a.skip.split(",") if x]}
    model, tok = load_model(a.model, dtype=a.dtype)
    t0 = time.time()
    res = evaluate(model, tok, recs, spec, a.batch_size, a.data)
    out = Path("var/results") / f"{res['spec_hash']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"accuracy {res['accuracy']:.3f} on {res['n']} ({time.time() - t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
