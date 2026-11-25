#!/usr/bin/env python
"""Depth-atlas tree runner (docs/PROPOSAL-depth-stages.md v13 sections 5-7).

  python scripts/atlas_tree_run.py csim --out var/results/atlas_gates/csim/csim.json [--reps 200 --n-null 200 --jobs 13]
  python scripts/atlas_tree_run.py read --gate 1 --root var/results/atlas_gates --csim var/results/atlas_gates/csim/csim.json
  python scripts/atlas_tree_run.py read --gate 2 --root var/results/atlas_gates --out var/results/atlas_gates/tree/g2.json

`read` loads <root>/g1/<prompt>/<model>/<task>.json (and, for gate 2, <root>/g2/trained/s<seed>/...), reads the tree,
writes the readings to --out (default <root>/tree/g<gate>.json), and prints the leaves, their sentences and the form.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgery.atlas import csim as C  # noqa: E402
from surgery.atlas import tree as T  # noqa: E402

ROOT = "var/results/atlas_gates"


def _dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(T.to_jsonable(obj), indent=1))


def cmd_csim(a):
    t0 = time.time()
    sp = C.spreads_from_data(a.cscreen, tuple(a.p0), a.trained)
    out = C.run_csim(sp, reps=a.reps, n_null=a.n_null, n_boot=a.n_boot, n_jobs=a.jobs, seed=a.seed,
                     n_reps_n4=a.reps_n4, n_reps_n8=a.reps_n8, n_reps_n5=a.reps_n5)
    out["spreads"] = {k: v for k, v in sp.items() if k not in ("probe_onsets", "use_onsets")}
    out["seconds"] = round(time.time() - t0, 1)
    _dump(out, a.out)
    n2 = out["n2"]
    print(f"csim -> {a.out} ({out['seconds']} s)")
    print(f"  N2 d* per k: {n2['d_star']}  worst: {n2['d_star_worst_text']}  power at d=3 (worse k): {n2['power_d3_worst_k']}")
    print(f"  N4 FPR max: {out['n4']['fpr_max']}  N8 FPR (measured noise): {out['n8']['fpr_max']}  N5 power: {out['n5']['power']}")


def cmd_read(a):
    t0 = time.time()
    rows = T.rows_from_results(a.root, a.models_json)
    if not rows:
        print(f"no rows under {a.root}", file=sys.stderr)
        return 1
    csim = None
    cpath = a.csim or str(Path(a.root) / "csim" / "csim.json")
    if Path(cpath).exists():
        csim = json.loads(Path(cpath).read_text())
    else:
        print(f"note: no C-sim file at {cpath}; nothing is made stronger than 'descriptive'", file=sys.stderr)
    res = T.run_tree(rows, gate=a.gate, csim=csim, n_null=a.n_null, n_boot=a.n_boot, seed=a.seed,
                     stop_after_gate1=a.stop_after_gate1)
    res["rows"] = len(rows)
    res["seconds"] = round(time.time() - t0, 1)
    out = a.out or str(Path(a.root) / "tree" / f"g{a.gate}.json")
    _dump(res, out)
    print(f"tree gate {a.gate}: {len(rows)} rows -> {out} ({res['seconds']} s)")
    for kind, r in res["readings"].items():
        if kind not in T.KINDS or not r:
            continue
        for node in ("N1", "N2", "N3", "N4", "N5", "N8"):
            x = r.get(node)
            if x:
                print(f"  [{kind}] {node}: {x.get('verdict') or x.get('leaf')}  leaf={x.get('leaf')}")
                if x.get("sentence"):
                    print(f"      {x['sentence']}")
        if r.get("contrast"):
            print(f"  [{kind}] {r['contrast']['leaf']}: {r['contrast']['sentence']}")
    f = res["form"]
    print(f"form: {f['form']} (rule {f['rule']})")
    for n in f["notes"]:
        print(f"  - {n}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("csim")
    c.add_argument("--out", default=f"{ROOT}/csim/csim.json")
    c.add_argument("--reps", type=int, default=200)
    c.add_argument("--n-null", type=int, default=200)
    c.add_argument("--n-boot", type=int, default=20)
    c.add_argument("--reps-n4", type=int, default=None)
    c.add_argument("--reps-n8", type=int, default=None)
    c.add_argument("--reps-n5", type=int, default=None)
    c.add_argument("--jobs", type=int, default=1)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--cscreen", default="var/cscreen")
    c.add_argument("--p0", nargs="*", default=["var/p0/results", "var/p0/results_trained"])
    c.add_argument("--trained", default="var/p0/results_trained")
    r = sub.add_parser("read")
    r.add_argument("--gate", type=int, choices=(1, 2), required=True)
    r.add_argument("--root", default=ROOT)
    r.add_argument("--csim", default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--n-null", type=int, default=1000)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--stop-after-gate1", action="store_true")
    r.add_argument("--models-json", default="var/ccheck/models.json")
    a = ap.parse_args(argv)
    return cmd_csim(a) if a.cmd == "csim" else cmd_read(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
