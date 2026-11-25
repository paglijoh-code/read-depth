#!/usr/bin/env python3
"""POST HOC (round-5 review, 2026-11-20): cross-model read-depth agreement under ONE depth rule before and after task training.

Unmodified depth (`surgery/atlas/metrics.py` `l_on`) is the first layer reaching chance + half the gap to the PROBE's
peak. Trained depth (`scripts/atlas2_healed.py` `l_on_healed`) is the first layer reaching chance + half the gap to
the trained MODEL's own accuracy. This script recomputes both models' depths from the saved probe curves under each
rule applied to both conditions, then the Qwen3-1.7B vs SmolLM2-1.7B Spearman agreement (24 tasks, 22 learned tasks,
after-layer-1 subset), task bootstrap CIs (5,000 draws, seed 0, as `scripts/depth_agreement_unmod.py`), and a paired
bootstrap of after-minus-before on the same tasks.

Inputs (model-free): unmodified curves `var/results/atlas3/{1.7B,smol17}/map_<task>.json` (probe.all.curve,
base_accuracy); trained curves `var/results/atlas3/{c1,d3}/healed_probe_<task>.json` (curve, healed_accuracy).

  .venv/bin/python scripts/depth_agreement_samerule.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from surgery.atlas import metrics  # noqa: E402

R = ROOT / "var/results/atlas3"
MODELS = {"qwen": ("1.7B", "c1"), "smol": ("smol17", "d3")}


def rule_peak(curve, chance):
    return metrics.l_on(curve, chance)


def rule_model(curve, chance, acc):
    if acc - chance < metrics.MIN_GAP - metrics.EPS:
        return None
    hit = metrics._first_reaching(curve, chance + 0.5 * (acc - chance))
    return len(curve) if hit is None else hit


def load(model):
    mdir, hdir = MODELS[model]
    out = {}
    for p in sorted((R / mdir).glob("map_*.json")):
        m = json.loads(p.read_text())
        t = m["task"]
        h = R / hdir / f"healed_probe_{t}.json"
        if not h.exists():
            continue
        h = json.loads(h.read_text())
        pc = m["probe"]["all"]
        out[t] = {
            "unmod_curve": pc["curve"], "chance": m["chance"], "base_acc": m["base_accuracy"], "unmod_stored": m["L_on"],
            "tr_curve": h["curve"], "tr_chance": h["chance"], "tr_acc": h["healed_accuracy"], "tr_stored": h["L_on_healed"],
        }
    return out


def rho(a, b):
    return float(spearmanr(np.asarray(a, float), np.asarray(b, float)).correlation)


def boot(a, b, n=5000):
    rng = np.random.default_rng(0)
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        if len(set(a[i])) > 1 and len(set(b[i])) > 1:
            out.append(spearmanr(a[i], b[i]).correlation)
    return np.percentile(out, [2.5, 97.5])


def paired(a0, b0, a1, b1, n=5000):
    """Bootstrap of rho(a1,b1) - rho(a0,b0), resampling the same tasks for both."""
    rng = np.random.default_rng(0)
    a0, b0, a1, b1 = (np.asarray(x, float) for x in (a0, b0, a1, b1))
    out = []
    for _ in range(n):
        i = rng.integers(0, len(a0), len(a0))
        if min(len(set(x[i])) for x in (a0, b0, a1, b1)) > 1:
            out.append(spearmanr(a1[i], b1[i]).correlation - spearmanr(a0[i], b0[i]).correlation)
    out = np.asarray(out)
    return np.percentile(out, [2.5, 97.5]), float(np.mean(out <= 0))


def f(x):
    return f"{x:+.2f}"


def main() -> int:
    q, s = load("qwen"), load("smol")
    tasks = sorted(set(q) & set(s))
    dropped = {r["task"] for r in json.loads((R / "d3/summary.json").read_text())["rows"] if r.get("dropped")}
    print(f"{len(tasks)} shared tasks; SmolLM2 not learned: {sorted(dropped & set(tasks))}")

    # 1. reproduce stored depths
    for name, d in (("qwen", q), ("smol", s)):
        bad_u = [t for t in tasks if rule_peak(d[t]["unmod_curve"], d[t]["chance"]) != d[t]["unmod_stored"]]
        bad_t = [t for t in tasks if rule_model(d[t]["tr_curve"], d[t]["tr_chance"], d[t]["tr_acc"]) != d[t]["tr_stored"]]
        print(f"{name}: unmodified stored = peak rule except {bad_u}; trained stored = model rule except {bad_t}")

    depth = {}
    for name, d in (("qwen", q), ("smol", s)):
        depth[name] = {
            "unmod_peak": [rule_peak(d[t]["unmod_curve"], d[t]["chance"]) for t in tasks],
            "unmod_model": [rule_model(d[t]["unmod_curve"], d[t]["chance"], d[t]["base_acc"]) for t in tasks],
            "tr_peak": [rule_peak(d[t]["tr_curve"], d[t]["tr_chance"]) for t in tasks],
            "tr_model": [rule_model(d[t]["tr_curve"], d[t]["tr_chance"], d[t]["tr_acc"]) for t in tasks],
        }
    print("\nper task: task | qwen unmod peak/model, trained peak/model | smol unmod peak/model, trained peak/model")
    for i, t in enumerate(tasks):
        print(t, *(depth["qwen"][k][i] for k in ("unmod_peak", "unmod_model", "tr_peak", "tr_model")), "|",
              *(depth["smol"][k][i] for k in ("unmod_peak", "unmod_model", "tr_peak", "tr_model")),
              " base acc q/s %.2f/%.2f" % (q[t]["base_acc"], s[t]["base_acc"]))
    for k in depth["qwen"]:
        nn = sum(depth["qwen"][k][i] is None or depth["smol"][k][i] is None for i in range(len(tasks)))
        print(f"{k}: tasks undefined in either model {nn}")

    def block(label, idx_all):
        print(f"\n== {label} ==")
        for cond_b, cond_a, rule in (("unmod_peak", "tr_peak", "probe-peak rule, both conditions"),
                                     ("unmod_peak", "tr_model", "as published (peak before, model after)")):
            idx = [i for i in idx_all if all(depth[m][c][i] is not None for m in depth for c in (cond_b, cond_a))]
            QB = [depth["qwen"][cond_b][i] for i in idx]; SB = [depth["smol"][cond_b][i] for i in idx]
            QA = [depth["qwen"][cond_a][i] for i in idx]; SA = [depth["smol"][cond_a][i] for i in idx]
            lo_b, hi_b = boot(QB, SB); lo_a, hi_a = boot(QA, SA)
            (dlo, dhi), p0 = paired(QB, SB, QA, SA)
            print(f"[{rule}] n={len(idx)} before {f(rho(QB, SB))} [{lo_b:.2f}, {hi_b:.2f}]  after {f(rho(QA, SA))} "
                  f"[{lo_a:.2f}, {hi_a:.2f}]  after-before {f(rho(QA, SA) - rho(QB, SB))} [{dlo:+.2f}, {dhi:+.2f}] "
                  f"P(diff<=0) {p0:.3f}")
            # after-layer-1 subset, the SAME tasks before and after: depth > 1 in both models in both conditions
            deep = [j for j in range(len(idx)) if min(QB[j], SB[j], QA[j], SA[j]) > 1]
            if len(deep) >= 4:
                qb, sb, qa, sa = ([x[j] for j in deep] for x in (QB, SB, QA, SA))
                lb, hb = boot(qb, sb); la, ha = boot(qa, sa)
                (dl, dh), pp = paired(qb, sb, qa, sa)
                print(f"   after-layer-1 in all four, n={len(deep)}: before {f(rho(qb, sb))} [{lb:.2f}, {hb:.2f}] "
                      f"after {f(rho(qa, sa))} [{la:.2f}, {ha:.2f}] diff {f(rho(qa, sa) - rho(qb, sb))} "
                      f"[{dl:+.2f}, {dh:+.2f}] P(diff<=0) {pp:.3f}")
            for c, lab in ((cond_b, "before"), (cond_a, "after")):
                dd = [j for j in range(len(idx)) if depth["qwen"][c][idx[j]] > 1 and depth["smol"][c][idx[j]] > 1]
                a = [depth["qwen"][c][idx[j]] for j in dd]; b = [depth["smol"][c][idx[j]] for j in dd]
                if len(dd) >= 4:
                    lo, hi = boot(a, b)
                    print(f"   {lab}, own after-layer-1 subset n={len(dd)}: {f(rho(a, b))} [{lo:.2f}, {hi:.2f}]")
            ties = [sum(depth[m][c][i] == 1 for i in idx) for m in ("qwen", "smol") for c in (cond_b, cond_a)]
            print(f"   layer-1 ties qwen before/after {ties[0]}/{ties[1]}, smol {ties[2]}/{ties[3]}")

    block("24 shared tasks", list(range(len(tasks))))
    block("22 tasks SmolLM2 learned", [i for i, t in enumerate(tasks) if t not in dropped])
    for name in ("qwen", "smol"):
        d = depth[name]
        print(f"within {name}: unmod peak vs trained peak {f(rho(d['unmod_peak'], d['tr_peak']))}; "
              f"trained peak vs trained model {f(rho(d['tr_peak'], d['tr_model']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
