#!/usr/bin/env python
"""C-check CLI (proposal section 5): writes JSON to var/ccheck/<name>.json and timestamped progress to var/ccheck/status.txt.

  models [--ids ID ...] [--no-load] [--force]     hub info, config, base check, tokenizer, load + answer-token frequency
  tasks                                           leaks, template and class-word splits, rivals, single-token summary
  onset                                           v11 onset vs paper one's stored L_on (read-only)
  tamper --model ID [--tasks a,b] | --toy         shuffled-label and swapped-twin tamper tests
  toy                                             planted toy under the v11 probe test
  timing --model ID [--tasks all]                 p0_run.run_task per task with v11 defaults
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from surgery.atlas import ccheck as C  # noqa: E402
from surgery.atlas import p0_tasks as T  # noqa: E402


def _short(model):
    return model.split("/")[-1]


def _load_models():
    p = C.OUT / "models.json"
    return json.loads(p.read_text()) if p.exists() else None


def _adapter(model, device=None):
    import torch
    from surgery.atlas.p0_harness import Adapter
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    return Adapter(model, device=device)


def cmd_models(a):
    done = C.run_models(a.ids or None, no_load=a.no_load, force=a.force, device=a.device or "mps")
    print(f"models.json: {len(done)} records")


def cmd_tasks(a):
    r = C.check_tasks(_load_models())
    C.write_json("tasks.json", r)
    for n, v in r["tasks"].items():
        print(f"{n:15s} {'PASS' if v['pass'] else 'FAIL'} {v['checks']}")
    C.status(f"tasks: {r['n_pass']}/{r['n_tasks']} pass" )
    print("TASKS", "PASS" if r["pass"] else "FAIL")
    return 0 if r["pass"] else 1


def cmd_onset(a):
    r = C.onset_check()
    C.write_json("onset.json", r)
    C.status(f"onset: {r['n_pass']}/{r['n_checked']} within 1 layer, {len(r['skipped_no_control'])} skipped (no control), "
             f"{r['n_recompute_mismatch']} L_on recompute mismatches")
    print(f"ONSET {r['n_pass']}/{r['n_checked']} pass; {r['n_fail']} fail; skipped {len(r['skipped_no_control'])}; "
          f"recompute mismatches {r['n_recompute_mismatch']}")
    return 0


def cmd_toy(a):
    r = C.toy_check()
    C.write_json("toy.json", r)
    for x in r["runs"]:
        print(x)
    C.status(f"toy: {'PASS' if r['pass'] else 'FAIL'}")
    print("TOY", "PASS" if r["pass"] else "FAIL")
    return 0 if r["pass"] else 1


def _print_tamper(name, r):
    for k, v in r["runs"].items():
        print(f"  {name} {k:16s} valid={v['valid']} probe_sel={v['probe']['peak_selectivity']:.2f} "
              f"p={v['probe']['p_binom_bonf']:.3g} twin={v['twin']['peak']:.2f} pairs={v['n_pairs']}")
    print(f"  {name} TAMPER", "PASS" if r["pass"] else "FAIL", r["checks"])


def cmd_tamper(a):
    if a.toy:
        r = C.toy_tamper()
        C.write_json("tamper_toy.json", r)
        _print_tamper("toy", r)
        C.status(f"tamper toy: {'PASS' if r['pass'] else 'FAIL'}")
        return 0 if r["pass"] else 1
    import p0_run
    ad = _adapter(a.model, a.device)
    args = types.SimpleNamespace(model=a.model, n_perm=a.n_perm, max_pairs=200)
    names = T.TASK_NAMES if a.tasks == "all" else a.tasks.split(",")
    out, ok = {}, True
    for n in names:
        t0 = time.time()
        b = p0_run.build_task(ad, n)
        riv = T.rivals(b["kept"], n)
        r = C.tamper(ad, b, lambda **over: p0_run.run_built(ad, n, b, args, riv, log=lambda *x: None, **over))
        out[n] = r
        ok &= r["pass"]
        _print_tamper(n, r)
        C.write_json(f"tamper_{_short(a.model)}.json", out)
        C.status(f"tamper {_short(a.model)} {n}: {'PASS' if r['pass'] else 'FAIL'} in {time.time() - t0:.0f}s")
    return 0 if ok else 1


def cmd_timing(a):
    import p0_run
    ad = _adapter(a.model, a.device)
    args = types.SimpleNamespace(model=a.model, n_perm=a.n_perm, max_pairs=200)
    short = _short(a.model)
    names = T.TASK_NAMES if a.tasks == "all" else a.tasks.split(",")
    summ = {}
    for n in names:
        res = p0_run.run_task(ad, n, args, short, log=lambda *x: None)
        C.write_json(f"timing_{short}/{n}.json", res)
        t = res["timing"]
        summ[n] = {**{k: t[k] for k in ("states_s", "probe_s", "twin_s", "use_arrival_s")}, "probe_fits": t["probe_fits"]}
        C.write_json(f"timing_{short}.json", {"model": a.model, "n_blocks": ad.n_blocks, "n_perm": a.n_perm, "tasks": summ})
        C.status(f"timing {short} {n}: " + " ".join(f"{k}={v:.1f}" for k, v in summ[n].items()))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("models")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--no-load", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default=None)
    p.set_defaults(f=cmd_models)
    for name, f in (("tasks", cmd_tasks), ("onset", cmd_onset), ("toy", cmd_toy)):
        sub.add_parser(name).set_defaults(f=f)
    p = sub.add_parser("tamper")
    p.add_argument("--model")
    p.add_argument("--toy", action="store_true")
    p.add_argument("--tasks", default="all")
    p.add_argument("--n-perm", type=int, default=20)
    p.add_argument("--device", default=None)
    p.set_defaults(f=cmd_tamper)
    p = sub.add_parser("timing")
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", default="all")
    p.add_argument("--n-perm", type=int, default=20)
    p.add_argument("--device", default=None)
    p.set_defaults(f=cmd_timing)
    a = ap.parse_args()
    if a.cmd == "tamper" and not a.toy and not a.model:
        ap.error("tamper needs --model or --toy")
    sys.exit(a.f(a) or 0)


if __name__ == "__main__":
    main()
