"""M9 atlas: map one task. Store on the discovery split, probes (whole task and each rung), logit lens, block lesions.

  uv run python scripts/atlas_map.py --task arith [--n-lesion 500] [--smoke]

Each part is saved as it finishes in var/results/atlas/map_{task}.partial.json (under --out) and the lesion sweep resumes
block by block; map_{task}.json is written only when every part is done. --smoke uses 64 items, 20 boots,
two lesion blocks, and separate output names.
"""
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

from surgery.atlas import measure, metrics, spec  # noqa: E402
from surgery.eval import load_model  # noqa: E402
from surgery.probes import probe_by_layer  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
OUT = Path("var/results/atlas")
ACTS = Path("var/acts/atlas")


def _write(path, obj):
    measure._write_json(path, obj)


def _probe(X, answers, groups, idx, boots, seed):
    """Probe curve on the rows `idx`; None if the subset cannot be cross-validated (under 5 templates or one class)."""
    a = np.asarray(answers)[idx]
    g = np.asarray(groups)[idx]
    if len(set(g)) < 5 or len(set(a)) < 2:
        return None
    rows = probe_by_layer(X[idx], a, g, kind="label", seed=seed, n_boot=boots)
    chance = spec.chance(a.tolist())
    curve = [r["metric"] for r in rows]
    peak = int(np.argmax(curve))
    return {"n": int(len(a)), "chance": chance, "curve": curve, "lo": [r["lo"] for r in rows], "hi": [r["hi"] for r in rows],
            "control": [r["control"] for r in rows], "peak": curve[peak], "peak_layer": peak,
            "L_on": metrics.l_on(curve, chance)}


def _lens(lens, answers, idx, final_pred):
    a = np.asarray(answers)[idx]
    curve = [float((lens[idx, l].argmax(-1) == a).mean()) for l in range(lens.shape[1])]
    chance = spec.chance(a.tolist())
    final = float((np.asarray(final_pred)[idx] == a).mean())
    return {"chance": chance, "curve": curve, "final": final, "L_lens": metrics.l_lens(curve, chance, final)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(spec.TASKS))
    ap.add_argument("--n-lesion", type=int, default=500)
    ap.add_argument("--boots", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT, help="results folder (default var/results/atlas)")
    ap.add_argument("--model", default=MODEL, help=f"model to map (default {MODEL}; another model keeps its store in its own subfolder)")
    args = ap.parse_args(argv)
    task, ts = args.task, spec.TASKS[args.task]
    suffix = "_smoke" if args.smoke else ""
    boots = 20 if args.smoke else args.boots
    n_lesion = 8 if args.smoke else args.n_lesion
    out = args.out
    final_path = out / f"map_{task}{suffix}.json"
    part_path = out / f"map_{task}{suffix}.partial.json"
    if final_path.exists():
        print(f"{final_path} exists; nothing to do")
        return
    records = spec.read_records(task, "discovery")
    if args.smoke:
        records = records[:64]
    store = (ACTS if args.model == MODEL else ACTS / args.model.split("/")[-1]) / f"{task}_discovery{suffix}"
    result = json.loads(part_path.read_text()) if part_path.exists() else {}
    result.update(task=task, number=ts.number, kind=ts.kind, needs=ts.needs, steps=ts.steps, n=len(records), smoke=args.smoke)
    if args.model != MODEL:
        result["model"] = args.model

    model = tok = None

    def get_model():
        nonlocal model, tok
        if model is None:
            model, tok = load_model(args.model, dtype="fp32")
        return model, tok

    if not (store / "meta.json").exists():
        t0 = time.time()
        m, t = get_model()
        measure.build_store(m, t, task, records, store, batch_size=args.batch_size)
        print(f"store built in {time.time() - t0:.0f}s", flush=True)
    meta = json.loads((store / "meta.json").read_text())
    if meta["ids"] != [r["id"] for r in records]:
        raise ValueError(f"store at {store} does not match the discovery records")
    answers, groups, rungs = meta["answers"], meta["template_ids"], meta["rungs"]
    n = len(answers)
    everything = np.arange(n)
    rung_idx = {r: np.array([i for i, x in enumerate(rungs) if x == r]) for r in ts.rungs}

    if "lens" not in result:
        lens = np.load(store / "lens.npy", mmap_mode="r")
        lens = np.asarray(lens, dtype=np.float32)
        result["lens"] = {"all": _lens(lens, answers, everything, meta["pred"]),
                          "rungs": {r: _lens(lens, answers, ix, meta["pred"]) for r, ix in rung_idx.items() if len(ix)}}
        result["base_accuracy"] = result["lens"]["all"]["final"]
        _write(part_path, result)
        print("lens done", flush=True)

    X = np.load(store / "resid.npy", mmap_mode="r")
    if "probe" not in result:
        t0 = time.time()
        result["probe"] = {"all": _probe(X, answers, groups, everything, boots, seed=0)}
        _write(part_path, result)
        print(f"probe all done ({time.time() - t0:.0f}s)", flush=True)
    for r, ix in rung_idx.items():
        if r not in result["probe"]:
            result["probe"][r] = _probe(X, answers, groups, ix, boots, seed=0) if len(ix) else None
            _write(part_path, result)
            print(f"probe {r} done", flush=True)

    lesion_path = out / f"lesion_{task}{suffix}.json"
    m, t = get_model()
    means = torch.load(store / "means.pt", map_location="cpu", weights_only=True)
    blocks = [0, 1] if args.smoke else None
    les = measure.lesion_sweep(m, t, task, records[:n_lesion], means, batch_size=args.batch_size, path=lesion_path, blocks=blocks)
    drops = les["drops"]
    result["lesion"] = {"n": les["n"], "base_accuracy": les["base_accuracy"], "drops": drops,
                        "fragile": metrics.fragile(drops, 0.05)}  # smoke blocks are 0,1 so positions equal indices

    p = result["probe"]["all"]
    result.update(chance=result["lens"]["all"]["chance"], L_on=p["L_on"] if p else None,
                  peak_layer=p["peak_layer"] if p else None, peak=p["peak"] if p else None,
                  control_curve=p["control"] if p else None, L_lens=result["lens"]["all"]["L_lens"],
                  fragile=result["lesion"]["fragile"])
    _write(final_path, result)
    part_path.unlink(missing_ok=True)
    print(f"{task}: base {result['base_accuracy']:.3f}  L_on {result['L_on']}  L_lens {result['L_lens']}  "
          f"fragile {result['fragile']} -> {final_path}", flush=True)


if __name__ == "__main__":
    main()
