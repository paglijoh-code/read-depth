"""Per-layer skip sweep: baseline, then each listed layer skipped alone.

usage: sweep_skip.py [--data PATH] [--n N] [--layers 0,1,..|all] [--smoke] [--baseline-only]
Writes var/results/sensitivity.json (sensitivity_smoke.json with --smoke,
sensitivity_baseline.json with --baseline-only, so those never overwrite the real result).
--smoke also asserts skip-[] logits == unhooked logits exactly on one batch.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from surgery.eval import MODEL_NAME, NO_SPEC, evaluate, label_logits, load_model  # noqa: E402
from surgery.hooks import apply_spec  # noqa: E402
from surgery.task.generator import generate, read_jsonl, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="var/data/eval500.jsonl")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    if not Path(a.data).exists():
        print(f"{a.data} missing; generating 500 examples with seed 0")
        write_jsonl(generate(500, 0), a.data)
    recs = read_jsonl(a.data)[: a.n]
    model, tok = load_model()
    n_layers = len(model.model.layers)
    layers = [] if a.baseline_only else (
        list(range(n_layers)) if a.layers == "all" else [int(x) for x in a.layers.split(",") if x])

    if a.smoke:
        prompts = [r["prompt"] for r in recs[: a.batch_size]]
        base = label_logits(model, tok, prompts, a.batch_size)
        with apply_spec(model, NO_SPEC):
            hooked = label_logits(model, tok, prompts, a.batch_size)
        if not torch.equal(base, hooked):
            print("SMOKE FAIL: skip-[] logits differ from unhooked logits", file=sys.stderr)
            sys.exit(1)
        print("smoke: skip-[] logits identical to unhooked logits")

    t0 = time.time()
    base = evaluate(model, tok, recs, NO_SPEC, a.batch_size, a.data)
    print(f"baseline accuracy {base['accuracy']:.3f} on n={base['n']} ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for L in layers:
        r = evaluate(model, tok, recs, {"op": "skip", "layers": [L]}, a.batch_size, a.data)
        rows.append({"layer": L, "accuracy": r["accuracy"],
                     "accuracy_lost": base["accuracy"] - r["accuracy"],
                     "mean_logit_diff": r["mean_logit_diff"]})
        print(f"layer {L:2d}: acc {r['accuracy']:.3f} lost {rows[-1]['accuracy_lost']:+.3f} "
              f"[{len(rows)}/{len(layers)}, {time.time() - t0:.0f}s]", flush=True)
    out = {"model": MODEL_NAME, "n": base["n"], "data": a.data, "knobs": recs[0]["knobs"],
           "baseline_accuracy": base["accuracy"], "baseline_mean_logit_diff": base["mean_logit_diff"],
           "layers": rows, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    name = "sensitivity_smoke" if a.smoke else "sensitivity_baseline" if a.baseline_only else "sensitivity"
    path = Path("var/results") / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
