"""M8 step 3: MLX fp32 against a fresh PyTorch fp32 rebuild on selection, then bf16 accuracy.

  uv run python scripts/mlx_parity.py [--models ...] [--n 1500] [--variant rule|sensible] [--mlx var/mlx]
Same single-token label scoring as M7 (`build_prompt`, `label_token_ids`); MLX runs batch 1 unpadded, PyTorch
batches of 8. Gate (CONTRACT protocol): agreement >= 99% and accuracy within 0.5 points of M7. A run on fewer than
all 1500 items (the 50-prompt spike) is compared with PyTorch's own accuracy on the same items and is not logged.
bf16 is reported, not gated; a drop of more than 1 point is flagged. Results: var/results/m8/parity.json; each full
pass is appended to var/results/selection_log.jsonl in M7's format. Selection split only.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from surgery import export as E  # noqa: E402
from surgery.prompt import build_prompt, label_token_ids  # noqa: E402
from surgery.task.generator import LABELS, read_jsonl  # noqa: E402

DATA = Path("var/data/m2")
RESULT = Path("var/results/m8/parity.json")
SELECTION_LOG = Path("var/results/selection_log.jsonl")
FULL_N = 1500
BF16_DROP = 0.01


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def accuracy(pred, recs) -> float:
    return sum(p == LABELS.index(r["label"]) for p, r in zip(pred, recs)) / len(recs)


def mlx_predict(model_dir: Path, tok, recs) -> list[int]:
    import mlx.core as mx
    from mlx_lm import load
    config = json.loads((model_dir / "config.json").read_text())
    model, _ = load(str(model_dir), trust_remote_code="model_file" in config)
    ids = mx.array(label_token_ids(tok))
    pred = []
    for r in recs:
        x = mx.array(tok(build_prompt(tok, r["prompt"]))["input_ids"])[None]
        logits = model(x)[0, -1, ids].astype(mx.float32)
        mx.eval(logits)
        pred.append(int(mx.argmax(logits).item()))
    del model
    return pred


def _free_device(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def write(obj) -> None:
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(RESULT)


def log_selection(name, v, dtype, n, acc, row) -> None:
    SELECTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SELECTION_LOG.open("a") as fh:
        fh.write(json.dumps({"ts": now(), "model": E.MODEL, "dtype": dtype, "variant": v, "split": "selection",
                             "seed": 0, "n": n, "result": str(RESULT), "intervention": f"m8-mlx-{name}-{dtype}",
                             "config": {"runtime": "mlx", "pt_agreement": row.get("agreement")},
                             "accuracy": acc}, sort_keys=True) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(E.MODELS))
    ap.add_argument("--n", type=int, default=FULL_N)
    ap.add_argument("--variant", choices=["rule", "sensible"])
    ap.add_argument("--mlx", default="var/mlx")
    a = ap.parse_args(argv)
    from surgery.eval import label_logits, load_model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pristine, tok = load_model(E.MODEL, device="cpu")
    res = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    for name in a.models.split(","):
        for v in [x for x in E.variants_of(name) if a.variant in (None, x)]:
            recs = read_jsonl(DATA / f"{v}_s0_selection.jsonl")[:a.n]
            full = len(recs) == FULL_N
            d32 = Path(a.mlx) / name / "fp32"
            if not d32.exists():
                sys.exit(f"{d32} missing: run scripts/mlx_export.py first")
            t0 = time.time()
            model = E.load_point(name, pristine=copy.deepcopy(pristine)).to(device)
            pt = label_logits(model, tok, [r["prompt"] for r in recs], 8).argmax(-1).tolist()
            del model
            _free_device(device)
            m32 = mlx_predict(d32, tok, recs)
            pt_acc, acc32 = accuracy(pt, recs), accuracy(m32, recs)
            ref = E.m7_accuracy(name, v) if full else pt_acc
            gate = E.parity_gate(pt, m32, acc32, ref)
            row = {"n": len(recs), "full": full, "pt_accuracy": pt_acc, "mlx_fp32_accuracy": acc32,
                   "reference_accuracy": ref, "reference": "m7" if full else "pytorch on the same items", **gate}
            d16 = Path(a.mlx) / name / "bf16"
            if d16.exists():
                acc16 = accuracy(mlx_predict(d16, tok, recs), recs)
                row.update(mlx_bf16_accuracy=acc16, bf16_drop=acc32 - acc16, bf16_flag=(acc32 - acc16) > BF16_DROP)
            row["seconds"] = time.time() - t0
            res.setdefault(name, {})[v] = row
            write(res)
            if full:
                log_selection(name, v, "fp32", len(recs), acc32, row)
                if "mlx_bf16_accuracy" in row:
                    log_selection(name, v, "bf16", len(recs), row["mlx_bf16_accuracy"], row)
            print(f"{name}/{v}: n={len(recs)} agreement {gate['agreement']:.3f} fp32 {acc32:.3f} (ref {ref:.3f}) "
                  f"{'PASS' if gate['passed'] else 'FAIL'}"
                  + (f", bf16 {row['mlx_bf16_accuracy']:.3f}{' FLAG' if row['bf16_flag'] else ''}"
                     if "mlx_bf16_accuracy" in row else ""), flush=True)


if __name__ == "__main__":
    main()
