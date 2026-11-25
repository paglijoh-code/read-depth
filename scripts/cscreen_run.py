#!/usr/bin/env python
"""C-screen runner. Results -> var/cscreen/<model_short>/<task>.json, status -> var/cscreen/status.txt.

  PYTHONPATH=. uv run python scripts/cscreen_run.py run --models EleutherAI/pythia-160m,HuggingFaceTB/SmolLM-135M
  PYTHONPATH=. uv run python scripts/cscreen_run.py verdict --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgery.atlas import cscreen as C  # noqa: E402
from surgery.atlas import ccheck  # noqa: E402

OUT = Path("var/cscreen")
SCREENING = [m for m, _, role in ccheck.MODELS if role == "screening"]
# v13: probe validity only (rivals and twins stay on SCREENING). SmolLM v1, Apache-2.0, outside the fit.
PROBE_SCREENING = ["HuggingFaceTB/SmolLM-1.7B"]


def short(mid):
    return mid.split("/")[-1]


def _say(msg):
    ccheck.status(msg, OUT)


def _verdict_words(r):
    p, t = r["probe"], r["twin"]
    head = (f"acc={r['test_item_acc']:.2f} probe[sel={p['peak_selectivity']:.2f} valid={int(p['valid'])}"
            f"{' early' if r['early_read'] else ''}]")
    if t is None:
        return head + f" drop[align={r['dropped_alignment']} filt={r['dropped_filter']}]"
    return (head + f" twin[peak={t['peak']:.2f} valid_sym={int(t['valid_sym'])}"
            f" swap_valid={int(r['swapped']['valid_sym'])}] drop[align={r['dropped_alignment']} filt={r['dropped_filter']}]")


def cmd_run(args):
    import torch
    from surgery.atlas import families as fam
    from surgery.atlas.families import tokcheck  # noqa: F401  (sets fam.tokcheck)
    from surgery.atlas.p0_harness import Adapter
    t_start = time.time()
    names = fam.TASK_NAMES if args.tasks == "all" else args.tasks.split(",")
    models = args.models.split(",")
    _say(f"START models={models} tasks={len(names)} force={args.force} probe_only={args.probe_only}")
    toks = fam.tokcheck.load_tokenizers()
    for name in names:
        p = OUT / "alignment" / f"{name}.json"
        if args.force or not p.exists():
            try:
                task = fam.get_task(name)
                res = fam.tokcheck.check_items(fam.generate(name, seed=0), list(task.options), toks)
                ccheck.write_json(f"alignment/{name}.json", res, OUT)
            except Exception as e:  # noqa: BLE001
                _say(f"ALIGNMENT ERROR {name}: {type(e).__name__}: {e}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for mid in models:
        todo = [n for n in names if args.force or not (OUT / short(mid) / f"{n}.json").exists()]
        if not todo:
            _say(f"{short(mid)}: nothing to do")
            continue
        ad = Adapter(mid, device=device)
        ad.model_id = mid
        _say(f"{short(mid)}: loaded in {ad.load_seconds:.0f}s on {device}, {len(todo)} tasks")
        for name in todo:
            t0 = time.time()
            try:
                r = C.run_task(ad, name, fam, toks=toks, n_perm=args.n_perm, probe_only=args.probe_only)
                _say(f"{short(mid)} {name} {time.time() - t0:.0f}s {_verdict_words(r)}")
            except Exception as e:  # noqa: BLE001
                tb = traceback.extract_tb(e.__traceback__)[-1]
                r = {"error": f"{type(e).__name__}: {e}", "where": f"{tb.filename}:{tb.lineno}", "task": name, "model": mid}
                _say(f"{short(mid)} {name} ERROR {r['error']} at {Path(tb.filename).name}:{tb.lineno}")
            ccheck.write_json(f"{short(mid)}/{name}.json", r, OUT)
        del ad
        if device == "mps":
            torch.mps.empty_cache()
    _say(f"DONE {(time.time() - t_start) / 60:.1f} min")


def cmd_verdict(args):
    from surgery.atlas import families as fam
    from surgery.atlas.families import tokcheck  # noqa: F401  (sets fam.tokcheck)
    results = {}
    for mid in SCREENING:
        results[mid] = {}
        for _, (_, tn) in fam.FAMILIES.items():
            for t in tn:
                p = OUT / short(mid) / f"{t}.json"
                results[mid][t] = json.loads(p.read_text()) if p.exists() else {"error": "missing"}
    v = C.verdicts(results, fam)
    pk = C.pick(v["families"], seed=args.seed)
    ccheck.write_json("verdicts.json", v, OUT)
    ccheck.write_json("pick.json", pk, OUT)
    for f, fv in v["families"].items():
        print(f"{f:20s} {fv['type_label']:22s} {'KEEP' if fv['kept'] else 'drop'}{'+' if fv['kept_strict'] else ' '} "
              f"sig={fv['signal']:.2f} early={fv['early_tasks']} {'; '.join(fv['reasons'])}")
    print(f"\nstatus={pk['status']} seed={pk['seed']} kept={pk['n_kept']} main={len(pk['main'])} reserve={len(pk['reserve'])}")
    print("main:", ", ".join(pk["main"]))
    print("reserve:", ", ".join(pk["reserve"]))
    print("early-read tasks:", pk["early_read_tasks"], "per type:", pk["per_type"])


def _load(models, fam):
    res = {}
    for mid in models:
        res[mid] = {}
        for _, (_, tn) in fam.FAMILIES.items():
            for t in tn:
                p = OUT / short(mid) / f"{t}.json"
                res[mid][t] = json.loads(p.read_text()) if p.exists() else {"error": "missing"}
    return res


def cmd_verdict13(args):
    """v13 kept rule + joint early-read cap -> verdicts_v13.json, pick_v13.json (v12 files untouched)."""
    from surgery.atlas import families as fam
    rd = lambda n: json.loads((OUT / n).read_text()) if (OUT / n).exists() else {}  # noqa: E731
    v = C.verdicts_v13(_load(SCREENING, fam), fam, probe_results=_load(PROBE_SCREENING, fam),
                       fresh_rivals=rd(f"rivals_seed{args.fresh_seed}.json"), artefacts=rd("artefact.json"))
    pk = C.pick_v13(v["families"], seed=args.seed)
    ccheck.write_json("verdicts_v13.json", v, OUT)
    ccheck.write_json("pick_v13.json", pk, OUT)
    for f, fv in v["families"].items():
        print(f"{f:20s} {fv['type_label']:22s} {'KEEP' if fv['kept'] else 'drop'}{'+' if fv['kept_strict'] else ' '} "
              f"v12={'K' if fv['kept_v12'] else '-'} sig={fv['signal']:.2f} early={fv['early_tasks']} "
              f"{'; '.join(fv['reasons'])}")
    print(f"\nstatus={pk['status']} kept={pk['n_kept']} main={len(pk['main'])} reserve={len(pk['reserve'])} "
          f"early={pk['early_read_tasks']} cap={pk['early_cap_tasks']}")
    print("main:", ", ".join(pk["main"]))
    print("reserve:", ", ".join(pk["reserve"]))
    print("skipped (early cap):", ", ".join(pk["skipped_early"]))


def cmd_rivals(args):
    """v13 fresh-seed rival check: items from `generate(name, seed)`, the same alignment filter and the per-model
    twin-length filter (both tokenizer-only, as in `run`), rivals on each small screening model's kept items."""
    from surgery.atlas import families as fam
    from surgery.atlas.families import tokcheck  # noqa: F401  (sets fam.tokcheck)
    toks = fam.tokcheck.load_tokenizers()
    mtok = fam.tokcheck.load_tokenizers(SCREENING, strict=True)
    out = {}
    for name in fam.TASK_NAMES:
        task = fam.get_task(name)
        items = fam.generate(name, seed=args.seed)
        bad = set(fam.tokcheck.check_items(items, list(task.options), toks)["bad_ids"])
        items = [x for x in items if x["id"] not in bad]
        per = {}
        for mid, tok in mtok.items():
            def same(x, tok=tok):  # as Adapter.filter_items: equal lengths and equal last tokens
                e = [tok(p, add_special_tokens=False).input_ids for p in
                     (x["prompt"], x["ans_twin"]["prompt"], x["surf_twin"]["prompt"])]
                return len({len(a) for a in e}) == 1 and len({a[-1] for a in e}) == 1
            kept = [x for x in items if same(x)]
            per[mid] = {"rivals": fam.rivals(kept, name), "n_kept": len(kept), "dropped_alignment": len(bad)}
        out[name] = per
        r = next(iter(per.values()))["rivals"]
        print(f"{name:22s} bow={r['bow']:.2f} pos={r['position']} nm={r['near_miss']}")
    ccheck.write_json(f"rivals_seed{args.seed}.json", out, OUT)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", default=",".join(SCREENING))
    r.add_argument("--tasks", default="all")
    r.add_argument("--n-perm", type=int, default=20)
    r.add_argument("--force", action="store_true")
    r.add_argument("--probe-only", action="store_true", help="v13: probe curve only (no twin states)")
    rv = sub.add_parser("rivals")
    rv.add_argument("--seed", type=int, default=1)
    v13 = sub.add_parser("verdict13")
    v13.add_argument("--seed", type=int, default=0)
    v13.add_argument("--fresh-seed", type=int, default=1)
    v = sub.add_parser("verdict")
    v.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"run": cmd_run, "verdict": cmd_verdict, "rivals": cmd_rivals, "verdict13": cmd_verdict13}[args.cmd](args)


if __name__ == "__main__":
    main()
