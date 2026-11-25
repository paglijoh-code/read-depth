"""M2 baseline: one unmodified-model pass over one split, in a fresh process.

  uv run python scripts/baseline.py --model Qwen/Qwen3-0.6B --variant rule --split discovery [--seed 0]
  uv run python scripts/baseline.py --model Qwen/Qwen3-0.6B --canaries
Writes var/results/baseline/<model>_<variant>_s<seed>_<split>[_<tag>].json with per-example
predictions and label logits, metrics with template-grouped 95% bootstrap CIs, latency and
peak memory. Selection-split runs append to var/results/selection_log.jsonl and test-split
runs to var/results/test_ledger.jsonl (CONTRACT.md "Splits").
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from surgery import stats  # noqa: E402
from surgery.eval import NO_SPEC, evaluate, load_model  # noqa: E402
from surgery.task.generator import LABELS, read_jsonl  # noqa: E402

OUT = Path("var/results/baseline")
DATA = Path("var/data/m2")


def short(model: str) -> str:
    return model.split("/")[-1]


def peak_mem() -> dict:
    m = {"max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}  # bytes on macOS
    if torch.backends.mps.is_available():
        m["mps_driver_mb"] = torch.mps.driver_allocated_memory() / 2**20
    return m


def append(path: str, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def run_split(a) -> Path:
    path = DATA / f"{a.variant}_s{a.seed}_{a.split}.jsonl"
    recs = read_jsonl(path)[: a.n]
    model, tok = load_model(a.model, dtype=a.dtype)
    t0 = time.time()
    res = evaluate(model, tok, recs, NO_SPEC, a.batch_size, str(path))
    secs = time.time() - t0
    y = [LABELS.index(r["label"]) for r in recs]
    groups = [r["template_id"] for r in recs]
    summ = stats.summarize(y, res["logits"], groups, seed=0, n_boot=a.n_boot)
    out = {
        "model": a.model, "dtype": a.dtype, "variant": a.variant, "seed": a.seed, "split": a.split,
        "data_path": str(path), "n": len(recs), "n_templates": len(set(groups)),
        "metrics": summ, "per_label_accuracy": res["per_label_accuracy"],
        "mean_logit_diff": res["mean_logit_diff"],
        "seconds": secs, "ms_per_example": 1000 * secs / len(recs), "batch_size": a.batch_size,
        "memory": peak_mem(),
        "ids": [r["id"] for r in recs], "template_ids": groups, "y": y,
        "pred": res["pred"], "logits": res["logits"],
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tag = f"_{a.tag}" if a.tag else ""
    dest = OUT / f"{short(a.model)}_{a.variant}_s{a.seed}_{a.split}{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out))
    row = {"ts": out["finished"], "model": a.model, "dtype": a.dtype, "variant": a.variant,
           "split": a.split, "seed": a.seed, "intervention": "baseline", "n": len(recs),
           "accuracy": summ["accuracy"]["point"], "result": str(dest)}
    if a.split == "selection":
        append("var/results/selection_log.jsonl", row)
    elif a.split == "test" and not a.tag:
        append("var/results/test_ledger.jsonl", row)
    acc = summ["accuracy"]
    print(f"{short(a.model)} {a.variant} s{a.seed} {a.split}{tag}: acc {acc['point']:.3f} "
          f"[{acc['lo']:.3f}, {acc['hi']:.3f}] n={len(recs)} ({secs:.0f}s) -> {dest}")
    return dest


def run_canaries(a) -> Path:
    from surgery import canaries
    model, tok = load_model(a.model, dtype=a.dtype)
    items = read_jsonl("data/canaries500.jsonl")
    t0 = time.time()
    can = canaries.score_canaries(model, tok, items, batch_size=a.batch_size)
    t1 = time.time()
    ppl = canaries.perplexity(model, tok, canaries.load_ppl_text())
    t2 = time.time()
    out = {"model": a.model, "dtype": a.dtype, "canaries": can, "perplexity": ppl,
           "canary_seconds": t1 - t0, "ppl_seconds": t2 - t1, "memory": peak_mem(),
           "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    dest = OUT / f"{short(a.model)}_canaries.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out))
    print(f"{short(a.model)} canaries {can['accuracy']:.3f} ppl {ppl['ppl']:.2f} -> {dest}")
    return dest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--variant", choices=["rule", "sensible"])
    ap.add_argument("--split", choices=["discovery", "selection", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=None, help="first N examples only (smoke)")
    ap.add_argument("--tag", default="", help="suffix for repeat runs (determinism gate)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--canaries", action="store_true")
    a = ap.parse_args(argv)
    if a.canaries:
        run_canaries(a)
    else:
        if not (a.variant and a.split):
            ap.error("--variant and --split are required unless --canaries")
        run_split(a)


if __name__ == "__main__":
    main()
