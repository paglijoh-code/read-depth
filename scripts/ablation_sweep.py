"""Single-process, resumable discovery ablations using the M3 activation store."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from surgery import stats  # noqa: E402
from surgery.eval import _free, label_logits, load_model  # noqa: E402
from surgery.hooks import apply_spec, spec_hash, validate_spec  # noqa: E402
from surgery.subset import sweep_indices  # noqa: E402
from surgery.task.generator import LABELS, read_jsonl  # noqa: E402


def _pass(model, tok, prompts, batch_size, spec=None, means=None):
    start = time.perf_counter()
    try:
        if spec is None:
            logits = label_logits(model, tok, prompts, batch_size)
        else:
            with apply_spec(model, spec, means=means):
                logits = label_logits(model, tok, prompts, batch_size)
        return logits, time.perf_counter() - start
    finally:
        _free(model)


def _correct_logp(logits, y):
    return logits.log_softmax(-1)[torch.arange(len(y)), torch.as_tensor(y)].numpy()


def _row(spec, logits, y, groups, base_pred, base_logp, seconds):
    pred = logits.argmax(-1).tolist()
    accuracy = stats.grouped_bootstrap(stats.accuracy, groups, y, pred)
    # The shared function returns B - A; report baseline - ablated instead.
    delta = stats.paired_grouped_bootstrap(stats.accuracy, groups, y, base_pred, pred)
    lost = {**delta, "point": -delta["point"], "lo": -delta["hi"], "hi": -delta["lo"]}
    return dict(spec=spec, spec_hash=spec_hash(spec), accuracy=accuracy["point"],
                accuracy_ci=accuracy, accuracy_lost=lost["point"], accuracy_lost_ci=lost,
                mean_correct_label_log_prob_drop=float(np.mean(base_logp - _correct_logp(logits, y))),
                seconds=seconds, pred=pred)


def _write(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True, choices=["rule", "sensible"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--sites", default="attn,mlp")
    ap.add_argument("--modes", default="zero,mean")
    ap.add_argument("--heads-file", type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    sites, modes = args.sites.split(","), args.modes.split(",")
    if args.n < 1 or args.batch_size < 1:
        ap.error("--n and --batch-size must be positive")
    if set(sites) - {"attn", "mlp"} or set(modes) - {"zero", "mean"}:
        ap.error("--sites must be attn,mlp; --modes must be zero,mean")

    short = args.model.rsplit("/", 1)[-1]
    store = Path("var/acts") / f"{short}_{args.variant}_s0_discovery"
    meta = json.loads((store / "meta.json").read_text())
    if (meta["model"] != args.model or meta["variant"] != args.variant
            or meta["split"] != "discovery"):
        raise ValueError("store model, variant, and discovery split must match the sweep")
    means = torch.load(store / "means.pt", map_location="cpu", weights_only=True)
    n = 16 if args.smoke else args.n
    indices = sweep_indices(meta["n"], n)
    data_path = Path(meta["data_path"])
    records = read_jsonl(data_path)
    if len(records) < meta["n"] or [r["id"] for r in records[:meta["n"]]] != meta["ids"]:
        raise ValueError("discovery records do not match the store's ids")
    records = [records[i] for i in indices]
    y = [LABELS.index(r["label"]) for r in records]
    groups = [r["template_id"] for r in records]
    if y != [meta["y"][i] for i in indices] or groups != [meta["template_ids"][i] for i in indices]:
        raise ValueError("discovery labels or template ids do not match the store")
    n_layers = min(2, meta["n_layers"]) if args.smoke else meta["n_layers"]
    if args.heads_file:
        heads = json.loads(args.heads_file.read_text())
        validate_spec(dict(op="ablate_heads", mode="mean", heads=heads),
                      meta["n_layers"], meta["n_heads"], meta["n_kv_heads"])
        specs = [dict(op="ablate_heads", mode="mean", heads=[pair]) for pair in heads
                 if pair[0] < n_layers]
    else:
        specs = [dict(op="ablate", site=site, mode=mode, layers=[layer])
                 for site in sites for mode in modes for layer in range(n_layers)]

    # Smoke output is separate so a small run cannot replace or contaminate a full sweep.
    suffix = ("_heads" if args.heads_file else "") + ("_smoke" if args.smoke else "")
    path = Path("var/results/m3") / f"ablation_{short}_{args.variant}{suffix}.json"
    identity = dict(model=args.model, variant=args.variant, split="discovery", dtype="fp32",
                    n=n, indices=indices, ids=[r["id"] for r in records],
                    template_ids=groups, y=y, data_path=str(data_path), batch_size=args.batch_size)
    result = {**identity, "rows": []}
    if path.exists():
        result = json.loads(path.read_text())
        if any(result.get(key) != value for key, value in identity.items()):
            raise ValueError(f"incompatible existing sweep at {path}; use the original settings")

    model, tok = load_model(args.model, dtype="fp32")
    if len(model.model.layers) != meta["n_layers"]:
        raise ValueError("model layer count does not match the store")
    prompts = [r["prompt"] for r in records]
    base, seconds = _pass(model, tok, prompts, args.batch_size)
    base_pred = base.argmax(-1).tolist()
    base_logp = _correct_logp(base, y)
    if "baseline" in result and result["baseline"]["pred"] != base_pred:
        raise ValueError("baseline predictions changed; cannot resume this sweep")
    result["baseline"] = dict(
        accuracy=stats.grouped_bootstrap(stats.accuracy, groups, y, base_pred),
        pred=base_pred, mean_correct_label_log_prob=float(base_logp.mean()), seconds=seconds)
    del base
    _write(path, result)
    print(f"baseline accuracy {result['baseline']['accuracy']['point']:.3f} on n={n}", flush=True)
    done = {spec_hash(row["spec"]) for row in result["rows"]}
    for spec in specs:
        if spec_hash(spec) in done:
            continue
        logits, seconds = _pass(model, tok, prompts, args.batch_size, spec, means)
        row = _row(spec, logits, y, groups, base_pred, base_logp, seconds)
        del logits
        result["rows"].append(row)
        _write(path, result)
        done.add(spec_hash(spec))
        print(f"{spec}: accuracy {row['accuracy']:.3f}, lost {row['accuracy_lost']:+.3f} "
              f"({seconds:.1f}s) -> {path}", flush=True)


if __name__ == "__main__":
    main()
