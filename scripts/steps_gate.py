#!/usr/bin/env python3
"""Pre-training gates for the steps study (docs/ATLAS-steps.md, "Gates" and "Revision 2026-11-17").

    HF_HUB_OFFLINE=1 uv run python scripts/steps_gate.py [--data DIR] [--out FILE]

No model is run; tokenizers only. Per cell, a rule passes when its accuracy is at most chance + 0.10
(chance = frequency of the commonest answer in that cell and split).
  F1 steps_copy  (selection): bag-of-words, first digit, last digit, nearest binding; each also scored on twin pairs
                 (base and flip both right), pair accuracy at most 0.10 (not for the single-record pad cell).
  F2 steps_ops   (selection): bag-of-words, first digit, last digit, last operand, first operand (of the queried
                 expression), sign-pattern classifier (fit on train+lrsel).
  F4 steps_describe: best single-clue classifier (selection, levels 2-4), answer prior per cell and clue-combination
                 lookup (both scored on test). Bag-of-words and the strongest-single-clue rule are reported, not gated.
  All: prompt token counts (full chat prompt) per cell must have means within +-2% of the family mean under Qwen3;
       SmolLM2's spread is reported; option first tokens must be distinct under both tokenizers.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import atlas_baseline  # noqa: E402
from surgery.atlas import spec, steps  # noqa: E402

MARGIN, PAIR_MAX, TOKEN_TOL = 0.10, 0.10, 0.02
PAD_CELLS = (steps.F1_PAD_A, steps.F1_PAD_B)
TOKENIZERS = {"qwen": "Qwen/Qwen3-1.7B", "smol": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}


def load(task: str, data: Path) -> dict[str, list[dict]]:
    return {s: spec.read_records(task, s, data) for s in steps.SPLIT_ORDER}


def acc(pred: list[int], recs: list[dict]) -> float:
    return float(np.mean([p == r["answer"] for p, r in zip(pred, recs)])) if recs else float("nan")


def by_cell(recs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        out[r["rung"]].append(r)
    return dict(out)


def digits(text: str) -> list[int]:
    return [int(c) for c in text if c.isdigit()]


# ---------------------------------------------------------------- F1 rules
def f1_nearest(text: str) -> int:
    for line in reversed(text.split("\n")[:-1]):
        m = re.fullmatch(r"\w+ = (\d)", line)
        if m:
            return int(m.group(1))
    return -1


F1_RULES = {"first_digit": lambda t: (digits(t) or [-1])[0], "last_digit": lambda t: (digits(t) or [-1])[-1],
            "nearest_binding": f1_nearest}


def f1_gate(recs_sel: list[dict], bow: dict) -> dict:
    cells = {}
    for cell, rs in by_cell(recs_sel).items():
        ch = spec.chance([r["answer"] for r in rs])
        row = {"n": len(rs), "chance": ch, "bow": bow[cell]["bow"], "rules": {}, "pairs": {}}
        groups = defaultdict(dict)
        for r in rs:
            groups[r["meta"]["twin_group"]][r["meta"]["twin_role"]] = r
        pairs = [g for g in groups.values() if "flip" in g]
        row["n_pairs"] = len(pairs)
        for name, fn in F1_RULES.items():
            row["rules"][name] = acc([fn(r["prompt"]) for r in rs], rs)
            if pairs:
                row["pairs"][name] = float(np.mean([fn(g["base"]["prompt"]) == g["base"]["answer"]
                                                    and fn(g["flip"]["prompt"]) == g["flip"]["answer"] for g in pairs]))
        # bag-of-words pair accuracy needs the fitted model; filled by the caller
        ok = row["bow"] <= ch + MARGIN + 1e-9 and all(v <= ch + MARGIN + 1e-9 for v in row["rules"].values())
        ok = ok and all(v <= PAIR_MAX for v in row["pairs"].values())
        row["pass"] = bool(ok)
        row["exempt"] = cell in PAD_CELLS      # padding removes the competing bindings by design: report only
        cells[cell] = row
    return cells


# ---------------------------------------------------------------- F2 rules
def f2_parse(r: dict):
    m = r["meta"]
    return m["qops"], m["qsigns"]


def sign_features(signs: str) -> list[float]:
    f = [0.0] * 4
    for i, s in enumerate(signs):
        f[i] = 1.0 if s == "+" else -1.0
    return f


def f2_gate(sets: dict, bow: dict) -> dict:
    fit = sets["train"] + sets["lrsel"]
    clf = LogisticRegression(max_iter=2000, C=1.0).fit([sign_features(r["meta"]["qsigns"]) for r in fit],
                                                       [r["answer"] for r in fit])
    cells = {}
    for cell, rs in by_cell(sets["selection"]).items():
        ch = spec.chance([r["answer"] for r in rs])
        sub = [r for r in rs]
        rules = {"first_digit": acc([(digits(r["prompt"]) or [-1])[0] for r in sub], sub),
                 "last_digit": acc([(digits(r["prompt"].split("\n")[-2])[-1:] or [-1])[0] for r in sub], sub),
                 "last_operand": acc([r["meta"]["qops"][-1] for r in sub], sub),
                 "first_operand": acc([r["meta"]["qops"][0] for r in sub], sub),
                 "sign_pattern": acc(list(clf.predict([sign_features(r["meta"]["qsigns"]) for r in sub])), sub)}
        ok = bow[cell]["bow"] <= ch + MARGIN + 1e-9 and all(v <= ch + MARGIN + 1e-9 for v in rules.values())
        cells[cell] = {"n": len(sub), "chance": ch, "bow": bow[cell]["bow"], "rules": rules, "pass": bool(ok)}
    return cells


# ---------------------------------------------------------------- F4 rules
def literals(r: dict) -> set[tuple]:
    return set(zip(r["meta"]["features"], r["meta"]["polarities"]))


def f4_gate(sets: dict, bow: dict) -> dict:
    fit = sets["train"] + sets["lrsel"]
    lits = sorted({l for r in fit + sets["selection"] for l in literals(r)})
    X = lambda rs: np.array([[1.0 if l in literals(r) else 0.0 for l in lits] for r in rs])
    Xf, Xs = X(fit), X(sets["selection"])
    yf = [r["answer"] for r in fit]
    single_pred = {}
    for j, l in enumerate(lits):
        if Xf[:, j].sum() < 5:
            continue
        m = LogisticRegression(max_iter=1000).fit(Xf[:, [j]], yf)
        single_pred[l] = m.predict(Xs[:, [j]])
    # strongest-single-clue rule (comparator): the shown clue leaving the fewest concepts, predict its commonest train answer
    left = {l: {c for c in range(len(steps.CONCEPTS)) if steps.truth(c, l[0]) == l[1]} for l in lits}
    ans_by_lit = defaultdict(Counter)
    for r in fit:
        for l in literals(r):
            ans_by_lit[l][r["answer"]] += 1

    def strongest(r):
        l = min(literals(r), key=lambda x: (len(left[x]), x))
        c = ans_by_lit[l]
        return max(sorted(left[l]), key=lambda k: c.get(k, 0))
    # combination lookup and prior (scored on test)
    table: dict = {}
    for r in fit:
        table.setdefault((r["rung"], frozenset(literals(r))), Counter())[r["answer"]] += 1
    cells = {}
    test = by_cell(sets["test"])
    for cell, rs in by_cell(sets["selection"]).items():
        lvl = rs[0]["meta"]["n_needed"]
        ch_sel = spec.chance([r["answer"] for r in rs])
        idx = [i for i, r in enumerate(sets["selection"]) if r["rung"] == cell]
        row = {"n": len(rs), "chance": ch_sel, "bow_comparator": bow[cell]["bow"],
               "strongest_clue_comparator": acc([strongest(r) for r in rs], rs), "rules": {}}
        gated = {}
        if lvl >= 2:
            best = max(((acc(list(p[idx]), rs), l) for l, p in single_pred.items()), key=lambda x: x[0])
            row["rules"]["best_single_clue"] = best[0]
            row["best_clue"] = f"{best[1][0]}{'+' if best[1][1] else '-'}"
            gated["best_single_clue"] = best[0] <= ch_sel + MARGIN + 1e-9
        tr = test[cell]
        ch_t = spec.chance([r["answer"] for r in tr])
        maj = Counter(r["answer"] for r in fit if r["rung"] == cell).most_common(1)[0][0]
        prior = acc([maj] * len(tr), tr)
        look = acc([(table[(cell, frozenset(literals(r)))].most_common(1)[0][0]
                     if (cell, frozenset(literals(r))) in table else maj) for r in tr], tr)
        row["rules"]["answer_prior_test"], row["rules"]["combo_lookup_test"] = prior, look
        row["chance_test"] = ch_t
        gated["answer_prior_test"], gated["combo_lookup_test"] = prior <= ch_t + MARGIN + 1e-9, look <= ch_t + MARGIN + 1e-9
        row["pass"] = all(gated.values())
        cells[cell] = row
    return cells


# ---------------------------------------------------------------- tokens
def token_report(task: str, recs: list[dict], toks: dict) -> dict:
    out = {}
    for name, tok in toks.items():
        counts = defaultdict(list)
        for r in recs:
            full = spec.build_prompt(tok, task, r["prompt"]) if name == "qwen" else tok.apply_chat_template(
                [{"role": "system", "content": spec.TASKS[task].system}, {"role": "user", "content": r["prompt"]}],
                tokenize=False, add_generation_prompt=True)
            counts[r["rung"]].append(len(tok(full, add_special_tokens=False)["input_ids"]))
        means = {c: float(np.mean(v)) for c, v in counts.items()}
        grand = float(np.mean([x for v in counts.values() for x in v]))
        dev = {c: m / grand - 1 for c, m in means.items()}
        out[name] = {"grand_mean": grand, "cell_means": means, "max_abs_dev": max(abs(d) for d in dev.values()),
                     "spread": (max(means.values()) - min(means.values())) / grand,
                     "within_record_range": {c: [min(v), max(v)] for c, v in counts.items()}}
        try:
            ids = spec.option_token_ids(tok, task)
            out[name]["options_distinct"] = True
        except ValueError as e:
            out[name]["options_distinct"] = False
            out[name]["options_error"] = str(e)
    out["token_pass"] = bool(out["qwen"]["max_abs_dev"] <= TOKEN_TOL and out["qwen"]["options_distinct"]
                             and out["smol"]["options_distinct"])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=spec.DATA_DIR)
    ap.add_argument("--out", type=Path, default=Path("var/results/steps/gate.json"))
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    toks = {k: AutoTokenizer.from_pretrained(v) for k, v in TOKENIZERS.items()}
    res = {"margin": MARGIN, "pair_max": PAIR_MAX, "token_tol": TOKEN_TOL, "tasks": {}}
    for task in steps.STEPS_TASKS:
        sets = load(task, a.data)
        bow = atlas_baseline.baseline(task, a.data)["rungs"]
        if task == "steps_copy":
            cells = f1_gate(sets["selection"], bow)
        elif task == "steps_ops":
            cells = f2_gate(sets, bow)
        else:
            cells = f4_gate(sets, bow)
        tk = token_report(task, [r for s in sets.values() for r in s], toks)
        cell_pass = all(c["pass"] for c in cells.values() if not c.get("exempt"))
        res["tasks"][task] = {"cells": cells, "tokens": tk, "cells_pass": cell_pass, "pass": bool(cell_pass and tk["token_pass"])}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    for task, t in res["tasks"].items():
        print(f"\n### {task}: {'PASS' if t['pass'] else 'FAIL'} (cells {'pass' if t['cells_pass'] else 'fail'}, tokens "
              f"{'pass' if t['tokens']['token_pass'] else 'fail'}: qwen max dev {t['tokens']['qwen']['max_abs_dev']:.3f} "
              f"spread {t['tokens']['qwen']['spread']:.3f}; smol max dev {t['tokens']['smol']['max_abs_dev']:.3f} "
              f"spread {t['tokens']['smol']['spread']:.3f})\n")
        names = sorted({k for c in t["cells"].values() for k in c["rules"]})
        extra = [k for k in ("bow", "bow_comparator", "strongest_clue_comparator") if any(k in c for c in t["cells"].values())]
        print("| cell | n | chance | " + " | ".join(extra + names) + (" | max pair | " if task == "steps_copy" else " | ") + "pass |")
        print("|" + "---|" * (3 + len(extra) + len(names) + (2 if task == "steps_copy" else 1)))
        for cell, c in t["cells"].items():
            vals = [f"{c[k]:.2f}" if k in c else "" for k in extra] + [f"{c['rules'][k]:.2f}" if k in c["rules"] else "" for k in names]
            mp = (f" {max(c['pairs'].values()):.2f} |" if c.get("pairs") else " n/a |") if task == "steps_copy" else ""
            print(f"| {cell} | {c['n']} | {c['chance']:.2f} | " + " | ".join(vals) + " |" + mp + f" {'exempt (' + ('pass' if c['pass'] else 'fail') + ')' if c.get('exempt') else 'pass' if c['pass'] else 'FAIL'} |")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
