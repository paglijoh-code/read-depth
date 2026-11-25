#!/usr/bin/env python3
"""Pre-training gates for the steps-v3 Stage 2 families (docs/ATLAS-steps3.md, "Stage 2: new families").

    HF_HUB_OFFLINE=1 uv run python scripts/steps3_gate.py [--data DIR] [--out FILE]

No model is run; tokenizers only. Per cell on the selection split, a rule passes when its accuracy is at most chance + 0.10
(chance = frequency of the commonest answer in the cell). Rules are in surgery/atlas/steps3.py (RULES); a rule that is the
real solution in a cell (steps3.expected, e.g. last-operation-only at m = 1) is reported and left out of pass/fail.
Bag-of-words is trained on train (scripts/atlas_baseline.py). Token report: mean prompt tokens per cell (full chat prompt,
Qwen3) and the relative spread (max - min) / grand mean per family, which must be at most 2%; SmolLM2 is reported, and
option first tokens must be distinct under both tokenizers. Writes var/results/steps3/gate.json; always exits 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

import atlas_baseline  # noqa: E402
import steps_gate  # noqa: E402
from surgery.atlas import spec, steps3  # noqa: E402

MARGIN, SPREAD_MAX = 0.10, 0.02


def acc(vals: list[int], recs: list[dict]) -> float:
    return sum(v == r["answer"] for v, r in zip(vals, recs)) / len(recs)


def discovery_cv(recs: list[dict]) -> dict:
    """5-fold CV accuracy of the bag-of-words model (C = 1) on the discovery split: per cell and pooled. Report only."""
    from sklearn.model_selection import KFold
    X, y = [r["prompt"] for r in recs], np.array([r["answer"] for r in recs])
    pred = np.zeros(len(recs), dtype=int)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        pred[te] = atlas_baseline.model(1.0).fit([X[i] for i in tr], y[tr]).predict([X[i] for i in te])
    ok = pred == y
    cells = {c: float(ok[[i for i, r in enumerate(recs) if r["rung"] == c]].mean()) for c in sorted({r["rung"] for r in recs})}
    return {"cells": cells, "pooled": float(ok.mean()), "n": len(recs)}


def cell_gate(task: str, cell: str, rs: list[dict], bow: float) -> dict:
    ch = spec.chance([r["answer"] for r in rs])
    skip = steps3.expected(task, cell)
    rules = {name: acc([fn(r) for r in rs], rs) for name, fn in steps3.RULES[task].items()}
    gated = {k: v for k, v in rules.items() if k not in skip}
    worst = max(gated, key=gated.get)
    ok = bow <= ch + MARGIN + 1e-9 and all(v <= ch + MARGIN + 1e-9 for v in gated.values())
    return {"n": len(rs), "chance": ch, "bow": bow, "rules": rules, "expected": sorted(skip),
            "worst_rule": worst, "worst_margin": gated[worst] - ch, "bow_margin": bow - ch, "pass": bool(ok)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=spec.DATA_DIR)
    ap.add_argument("--out", type=Path, default=Path("var/results/steps3/gate.json"))
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    toks = {k: AutoTokenizer.from_pretrained(v) for k, v in steps_gate.TOKENIZERS.items()}
    res = {"margin": MARGIN, "spread_max": SPREAD_MAX, "families": {}}
    for task in steps3.TASKS3:
        sets = {s: spec.read_records(task, s, a.data) for s in steps3.SPLIT_ORDER}
        bow = atlas_baseline.baseline(task, a.data)["rungs"]
        by = defaultdict(list)
        for r in sets["selection"]:
            by[r["rung"]].append(r)
        cells = {c: cell_gate(task, c, by[c], bow[c]["bow"]) for c in spec.TASKS[task].rungs}
        cv = discovery_cv(sets["discovery"])
        for c in cells:
            cells[c]["bow_cv_discovery"] = cv["cells"][c]
        tk = steps_gate.token_report(task, [r for s in sets.values() for r in s], toks)
        tk["qwen_spread_pass"] = bool(tk["qwen"]["spread"] <= SPREAD_MAX)
        tk["token_pass"] = bool(tk["qwen_spread_pass"] and tk["qwen"]["options_distinct"] and tk["smol"]["options_distinct"])
        cells_pass = all(c["pass"] for c in cells.values())
        res["families"][task] = {"cells": cells, "bow_cv_discovery_pooled": cv["pooled"], "tokens": tk, "cells_pass": cells_pass,
                                 "pass": bool(cells_pass and tk["token_pass"])}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    for task, t in res["families"].items():
        tk = t["tokens"]
        print(f"\n### {task}: {'PASS' if t['pass'] else 'FAIL'} (cells {'pass' if t['cells_pass'] else 'fail'}, tokens "
              f"{'pass' if tk['token_pass'] else 'fail'}: qwen spread {tk['qwen']['spread']:.4f}, max dev {tk['qwen']['max_abs_dev']:.4f}; "
              f"smol spread {tk['smol']['spread']:.4f}; options distinct qwen {tk['qwen']['options_distinct']} smol {tk['smol']['options_distinct']})")
        print("mean Qwen tokens per cell: " + ", ".join(f"{c} {m:.1f}" for c, m in tk["qwen"]["cell_means"].items()))
        names = list(steps3.RULES[task])
        print("| cell | n | chance | bow | " + " | ".join(names) + " | bow cv disc | worst (margin) | pass |")
        print("|" + "---|" * (len(names) + 7))
        for cell, c in t["cells"].items():
            vals = [f"{c['rules'][k]:.2f}" + (" (exp)" if k in c["expected"] else "") for k in names]
            print(f"| {cell} | {c['n']} | {c['chance']:.2f} | {c['bow']:.2f} | " + " | ".join(vals) +
                  f" | {c['bow_cv_discovery']:.2f} | {c['worst_rule']} ({c['worst_margin']:+.2f}) | {'pass' if c['pass'] else 'FAIL'} |")
        print(f"pooled bow cv on discovery (n={len(sets['discovery'])}): {t['bow_cv_discovery_pooled']:.3f} (chance 0.100)")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
