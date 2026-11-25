#!/usr/bin/env python
"""E3 shared-scale check (docs/ATLAS-robustness.md, "E3"; review point M1). $0, stored curves only, no model.

`L_on_healed` and R both use the healed k0 accuracy a0. S1 measures depth from half the probe's own peak, S2 from a fixed
threshold (half of chance-to-perfect), S3 is the ranked partial rho(L_on_healed, R | a0). Rows and curves come from
`atlas_robust.py`'s profiles, which re-check every stored L_on_healed and R before use.

    .venv/bin/python scripts/atlas_e3_scale.py
Writes var/results/atlas3/e3_scale.json and prints a markdown table.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import atlas_robust as E  # noqa: E402
from surgery.atlas import metrics  # noqa: E402

D3 = f"SmolLM2-1.7B,{ROOT}/var/results/atlas3/d3/summary.json,{ROOT}/var/results/atlas3/d3,10"
MAIN = {"Qwen3-0.6B": "weighted", "Qwen3-1.7B": "weighted", "SmolLM2-1.7B": "weighted"}  # D3-A used X.d1_size (weighted within-kind), not pooled


def first(curve, thr):
    hit = metrics._first_reaching(curve, thr)
    return len(curve) + 1 if hit is None else hit


def depths(r):
    c, curve = r["pchance"], r["curve"]
    s1 = first(curve, c + 0.5 * (max(curve) - c))
    s2 = first(curve, c + 0.5 * (1.0 - c))
    return s1, s2


def both(x, y, kinds):
    return {"weighted": E.wrho(x, y, kinds), "pooled": E.spearman_plain(x, y)}


def run(label, p):
    rows, k = p["rows"], str(p.get("kref", 12))
    d = [r["L_on_healed"] for r in rows]
    y = [r["r12"] for r in rows]
    kinds = [r["kind"] for r in rows]
    a0 = [r["acc"]["0"]["mean"] for r in rows]
    ak = [r["acc"][k]["mean"] for r in rows]
    s1, s2 = zip(*(depths(r) for r in rows))
    out = {"label": label, "n": len(rows), "k": int(k), "main": MAIN[label],
           "base": both(d, y, kinds), "S1": both(list(s1), y, kinds), "S2": both(list(s2), y, kinds),
           "S3": E.ranked_partial(d, y, a0),
           "rho_depth_a0": E.spearman_plain(d, a0), "a0_min": min(a0), "a0_median": float(np.median(a0)),
           "n_a0_ge_0.99": sum(a >= 0.99 for a in a0), "rho_depth_raw_cut_acc": E.spearman_plain(d, ak),
           "tasks": [{"task": r["task"], "L_on_healed": r["L_on_healed"], "S1": a, "S2": b, "a0": x, "ak": z, "R": v}
                     for r, a, b, x, z, v in zip(rows, s1, s2, a0, ak, y)]}
    m = out["main"]
    vals = [out["S1"][m], out["S2"][m], out["S3"]]
    out["verdict"] = ("survives" if all(v is not None and v <= -0.40 for v in vals)
                      else "not robust" if any(v is None or v > -0.20 for v in vals) else "weakened")
    return out


def main():
    d1 = E.X.load(E.OUT / "d1.json")
    del d1
    profs = [("Qwen3-0.6B", E.profile_06()), ("Qwen3-1.7B", E.profile_17()), ("SmolLM2-1.7B", E.profile_extra(D3))]
    res = [run(lab, p) for lab, p in profs]
    (E.OUT / "e3_scale.json").write_text(json.dumps(res, indent=1, default=str))
    f = lambda v: "—" if v is None else f"{v:+.3f}"  # noqa: E731
    print("| model | n | main estimator | paper ρ | S1 own-peak | S2 fixed | S3 partial given a0 | ρ(depth, a0) | a0 ≥ 0.99 | ρ(depth, raw cut acc) | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for o in res:
        m = o["main"]
        v = o["verdict"] if o["label"] != "Qwen3-0.6B" else f"({o['verdict']}, report only)"
        print(f"| {o['label']} | {o['n']} | {m} | {f(o['base'][m])} | {f(o['S1'][m])} | {f(o['S2'][m])} | {f(o['S3'])} | "
              f"{f(o['rho_depth_a0'])} | {o['n_a0_ge_0.99']} of {o['n']} | {f(o['rho_depth_raw_cut_acc'])} | {v} |")
    print("\nOther estimator:", {o["label"]: {s: f(o[s]["pooled" if o["main"] == "weighted" else "weighted"]) for s in ("base", "S1", "S2")} for o in res})


if __name__ == "__main__":
    main()
