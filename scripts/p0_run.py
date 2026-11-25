#!/usr/bin/env python
"""P0 pilot runner: one model, one or all tasks -> var/p0/results/<model_short>/<task>.json (numbers only).

  .venv/bin/python scripts/p0_run.py --model EleutherAI/pythia-160m --task surf_copy --out var/p0/results --n-perm 20

Resumable: tasks whose JSON exists are skipped unless --force.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgery.atlas import p0_tasks as T  # noqa: E402


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def build_task(ad, name):
    """Items, token ids and labels for one task on one adapter (everything analyze_task needs, plus the item lists)."""
    task = T.get_task(name)
    opt_ids = T.check_options(ad.tok, task)
    items = T.generate(name, seed=0)
    kept, dropped = ad.filter_items(items)
    tr, te = T.split_items(kept)
    tmpl = {t: i for i, t in enumerate(sorted({x["template"] for x in tr}))}
    enc = lambda ps: ad.encode(ps)
    return {"task": task, "opt_ids": opt_ids, "kept": kept, "dropped": dropped, "tr": tr, "te": te,
            "ids_tr": enc([x["prompt"] for x in tr]), "ids_te": enc([x["prompt"] for x in te]),
            "ids_an": enc([x["ans_twin"]["prompt"] for x in te]), "ids_sf": enc([x["surf_twin"]["prompt"] for x in te]),
            "y_tr": [x["answer"] for x in tr], "g_tr": [tmpl[x["template"]] for x in tr],
            "y_te": [x["answer"] for x in te], "y_an": [x["ans_twin"]["answer"] for x in te]}


def run_built(ad, name, b, args, rivals, log=print, **over):
    """analyze_task on a built task; `over` replaces any of the built fields (tamper tests)."""
    from surgery.atlas.p0_harness import analyze_task
    b = {**b, **over}
    return analyze_task(ad, b["ids_tr"], b["y_tr"], b["g_tr"], b["ids_te"], b["y_te"], b["ids_an"], b["y_an"],
                        b["ids_sf"], b["opt_ids"], n_perm=args.n_perm, max_pairs=args.max_pairs,
                        bow_acc=rivals["bow"], log=lambda *a: log("   ", *a))


def run_task(ad, name, args, model_short, log=print):
    b = build_task(ad, name)
    rivals = T.rivals(b["kept"], name)  # before analyze_task: the bag-of-words rival is part of the probe control
    res = run_built(ad, name, b, args, rivals, log)
    task = b["task"]
    res["timing"]["load_s"] = ad.load_seconds
    res["answer_logfreq"] = ad.answer_logfreq(b["opt_ids"])
    res.update(model=args.model, task=name, family=task.family, type_label=task.type_label,
               options=list(task.options), items_dropped=b["dropped"], items_kept=len(b["kept"]),
               n_train_kept=len(b["tr"]), n_test_kept=len(b["te"]), rivals=rivals)
    return res


def summary(res):
    pc, td, us, ar, on = res["probe"], res["twin"], res["use"], res["arrival"], res["onset"]
    fmt = lambda o: "-" if o is None else f"{o['rel']:.2f}"
    t = res["timing"]
    return (f"{res['task']:15s} acc={res['test_item_acc']:.2f} pairs={res['n_pairs_qualifying']:3d} "
            f"probe[sel={pc['peak_selectivity']:.2f} p={pc['p_binom_bonf']:.3g} onset={fmt(on['probe'])}] "
            f"twin[t={td['peak']:.2f} onset={fmt(on['twin'])}] "
            f"use[peak={us['peak'] if us else float('nan'):.2f} onset={fmt(on['use'])}] "
            f"arrival[onset={fmt(on['arrival'])}] valid={ {k: int(v) for k, v in res['valid'].items()} } "
            f"time(s) load={t['load_s']:.1f} states={t['states_s']:.1f} probe={t['probe_s']:.1f} "
            f"twin={t['twin_s']:.1f} use+arr={t['use_arrival_s']:.1f} dropped={res['items_dropped']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="all")
    ap.add_argument("--out", default=None)
    ap.add_argument("--adapter", default=None, help="adapter.safetensors from p0_heal.py: merged into the base model; "
                    "results go to var/p0/results_trained")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    names = T.TASK_NAMES if args.task == "all" else args.task.split(",")
    short = args.model.split("/")[-1]
    out_dir = Path(args.out or ("var/p0/results_trained" if args.adapter else "var/p0/results")) / short
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if args.force or not (out_dir / f"{n}.json").exists()]
    for n in set(names) - set(todo):
        print(f"skip {n} (exists)")
    if not todo:
        return
    import torch
    from surgery.atlas.p0_harness import Adapter
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    ad = Adapter(args.model, device=device)
    if args.adapter:
        from surgery import adapters
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import p0_heal
        handle = p0_heal.attach_lora(ad.model)
        p0_heal.load_adapter(handle, args.adapter)
        print(f"merged {adapters.merge_lora(handle)} LoRA modules from {args.adapter}", flush=True)
    print(f"loaded {args.model} on {device}: B={ad.n_blocks} d={ad.d} in {ad.load_seconds:.1f}s", flush=True)
    for n in todo:
        t0 = time.time()
        res = run_task(ad, n, args, short)
        (out_dir / f"{n}.json").write_text(json.dumps(res, default=_json))
        print(summary(res), f"total={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
