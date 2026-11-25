#!/usr/bin/env python
"""v13 C-screen: token-position artefact check (proposal section 5, "any task whose probe reads far above the model's
own accuracy ... is checked for a token-position artefact").

Flag: a (task, model) whose probe is valid with peak selectivity >= 0.6 while the model's own 4-way test accuracy is
<= 0.30 (chance 0.25). Test: every item (train and test) gets a seeded random prefix of k in {0,1,2,3} copies of
"Note. ", which moves every token's absolute position by a varying amount without changing the answer. The probe is
refitted with the same harness and rule (probe_curve, 20 permutations, binomial test). If the answer is read from
absolute token positions the shifted probe collapses; if it is computed, it survives.
Confirmed artefact: the shifted probe is invalid, or its peak selectivity is below half the unshifted one.

  PYTHONPATH=. uv run python scripts/cscreen_artefact.py [--tasks a,b] [--models m1,m2]
Results -> var/cscreen/artefact.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgery.atlas import ccheck  # noqa: E402
from surgery.atlas import p0_measures as M  # noqa: E402

OUT = Path("var/cscreen")
SMALL = [m for m, _, role in ccheck.MODELS if role == "screening"]
SEL_MIN, ACC_MAX = 0.6, 0.30


def short(mid):
    return mid.split("/")[-1]


def flagged(models):
    out = []
    for mid in models:
        for p in sorted((OUT / short(mid)).glob("*.json")):
            r = json.loads(p.read_text())
            if "probe" in r and r["probe"]["valid"] and r["probe"]["peak_selectivity"] >= SEL_MIN \
                    and r["test_item_acc"] <= ACC_MAX:
                out.append((mid, r["task"], r["probe"]["peak_selectivity"], r["test_item_acc"]))
    return out


def shifted(items, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for x in items:
        k = int(rng.integers(0, 4))
        y = dict(x)
        y["prompt"] = "Note. " * k + x["prompt"]
        y["shift"] = k
        out.append(y)
    return out


def check(ad, name, fam, n_perm=20):
    task = fam.get_task(name)
    items = fam.generate(name, seed=0)
    tr, te = fam.split_items(shifted(items))
    opt_ids = fam.check_options(ad.tok, task)
    tmpl = {t: i for i, t in enumerate(sorted({x["template"] for x in tr}))}
    S_tr, _ = ad.collect_ids(ad.encode([x["prompt"] for x in tr]), opt_ids)
    S_te, L_te = ad.collect_ids(ad.encode([x["prompt"] for x in te]), opt_ids)
    y_tr, y_te = np.asarray([x["answer"] for x in tr]), np.asarray([x["answer"] for x in te])
    bow = fam.rival_bow(tr + te)
    pc = M.probe_curve(S_tr, y_tr, [tmpl[x["template"]] for x in tr], S_te, y_te, n_perm=n_perm, seed=0, bow_acc=bow)
    pc.pop("fits", None)
    return {"valid": bool(pc["valid"]), "peak_selectivity": float(pc["peak_selectivity"]),
            "peak_block": int(pc["peak_block"]), "test_item_acc": float(np.mean(L_te.argmax(1) == y_te)), "bow": bow}


def main():
    import torch
    from surgery.atlas import families as fam
    from surgery.atlas.p0_harness import Adapter
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(SMALL))
    ap.add_argument("--tasks", default="flagged")
    args = ap.parse_args()
    models = args.models.split(",")
    todo = flagged(models) if args.tasks == "flagged" else [(m, t, None, None) for m in models for t in args.tasks.split(",")]
    path = OUT / "artefact.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for mid in sorted({m for m, *_ in todo}):
        ad = Adapter(mid, device=device)
        for _, name, sel, acc in [x for x in todo if x[0] == mid]:
            t0 = time.time()
            orig = json.loads((OUT / short(mid) / f"{name}.json").read_text())
            s = check(ad, name, fam)
            o_sel = orig["probe"]["peak_selectivity"]
            s["orig_selectivity"], s["orig_acc"] = o_sel, orig["test_item_acc"]
            s["artefact"] = bool(not s["valid"] or s["peak_selectivity"] < 0.5 * o_sel)
            s["seconds"] = time.time() - t0
            res.setdefault(name, {})[short(mid)] = s
            ccheck.write_json("artefact.json", res, OUT)
            print(f"{short(mid):12s} {name:18s} orig sel={o_sel:.2f} acc={orig['test_item_acc']:.2f} -> shifted "
                  f"sel={s['peak_selectivity']:.2f} valid={int(s['valid'])} artefact={int(s['artefact'])} "
                  f"({s['seconds']:.0f}s)", flush=True)
        del ad
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
