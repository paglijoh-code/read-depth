"""Time repeated evaluate passes in one process and check determinism.

uv run python scripts/speed_check.py --n 500 --passes 4 [--dtype bf16] [--model M]
uv run python scripts/speed_check.py --n 500 --compare-dtype
Exit 1 if any pass > 1.2x pass 1 or predictions differ across passes.
--compare-dtype only reports (exit 0).
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from surgery.eval import MODEL_NAME, NO_SPEC, evaluate, load_model  # noqa: E402
from surgery.task.generator import read_jsonl  # noqa: E402

LIMIT = 1.2


def compare_dtype(a, recs) -> int:
    res = {}
    for dt in ("fp32", "bf16"):
        model, tok = load_model(a.model, dtype=dt)
        res[dt] = evaluate(model, tok, recs, NO_SPEC, a.batch_size, a.data)
        print(f"{dt}: accuracy {res[dt]['accuracy']:.3f} on {res[dt]['n']}", flush=True)
        del model, tok
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    agree = sum(x == y for x, y in zip(res["fp32"]["pred"], res["bf16"]["pred"])) / len(recs)
    print(f"prediction agreement fp32 vs bf16: {agree:.2%}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--data", default="var/data/eval500.jsonl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--compare-dtype", action="store_true")
    a = ap.parse_args(argv)
    recs = read_jsonl(a.data)[: a.n]
    if a.compare_dtype:
        return compare_dtype(a, recs)
    model, tok = load_model(a.model, dtype=a.dtype)
    times, preds = [], []
    for k in range(a.passes):
        t0 = time.time()
        r = evaluate(model, tok, recs, NO_SPEC, a.batch_size, a.data)
        times.append(time.time() - t0)
        preds.append(r["pred"])
        print(f"pass {k + 1}: {times[-1]:.1f}s ({times[-1] / times[0]:.2f}x) acc {r['accuracy']:.3f}", flush=True)
    slow = [k + 1 for k, t in enumerate(times) if t > LIMIT * times[0]]
    same = all(p == preds[0] for p in preds)
    print(f"slow passes (>{LIMIT}x pass 1): {slow or 'none'}; predictions identical: {same}")
    return 1 if slow or not same else 0


if __name__ == "__main__":
    sys.exit(main())
