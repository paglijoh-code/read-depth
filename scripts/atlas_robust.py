"""Step E1 (docs/ATLAS-robustness.md, "E1: free checks"): robustness checks on stored results; $0, Mac only, no model.

E1-a single-task influence, E1-b depth threshold, E1-c kind through depth, E1-d cut size. E1-e is skipped.
Depth is `L_on_healed`, R is R12 (D1's definition: `rows_of` rows, weighted within-kind Spearman `B.weighted_rho`, as D1-A).
Writes var/results/atlas3/e1.json and e1.md. Seeds: bootstrap and permutations 0, 10,000 each.

    .venv/bin/python scripts/atlas_robust.py
    .venv/bin/python scripts/atlas_robust.py --extra D3,PATH/summary.json,PATH/curve_dir,10   # a later profile (not run yet)

An extra profile reads rows like C1's summary.json and `healed_probe_<task>.json` curves from the curve dir; the last field is
the cut k the profile's R uses (its summary's own `r12` is checked against it only for k = 12).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
_argv, sys.argv = sys.argv, [sys.argv[0]]  # atlas3_extra reads positional argv at import
import atlas2_cuts as B  # noqa: E402
import atlas3_extra as X  # noqa: E402

sys.argv = _argv
from surgery.atlas import metrics  # noqa: E402

OUT = ROOT / "var/results/atlas3"
SEED = 0
N = 10_000
FRACTIONS = (0.3, 0.5, 0.7)
KS = (9, 12, 16)
DEEP_STEP = 7   # E2's deep bar, used only to count deep judging tasks


# ------------------------------------------------------------------ pure pieces

def healed_l_on(curve, chance, healed_acc, fraction=0.5):
    """D1's L_on_healed (`atlas2_healed.l_on_healed`) with a fraction: first layer reaching chance + fraction * (healed accuracy
    - chance). Healed accuracy under 5 points above chance: None. Never reached: len(curve)."""
    if healed_acc - chance < metrics.MIN_GAP - metrics.EPS:
        return None
    hit = metrics._first_reaching(curve, chance + fraction * (healed_acc - chance))
    return len(curve) if hit is None else hit


def r_at(acc_k, k0, chance):
    """R as R12 defines it (`atlas3_extra.r12`), at any k."""
    return X.r12(acc_k, k0, chance)


def wrho(pred, y, kinds):
    return B.weighted_rho(list(pred), list(y), list(kinds))[0]


def ranked_partial(x, y, z):
    """Pearson partial correlation of x and y given z, all three on average ranks; None if degenerate."""
    r = [rankdata(np.asarray(v, float)) for v in (x, y, z)]
    if any(v.std() == 0 for v in r):
        return None
    c = np.corrcoef(r)
    a, b, d = c[0, 1], c[0, 2], c[1, 2]
    den = math.sqrt((1 - b * b) * (1 - d * d))
    return None if den < 1e-12 else float((a - b * d) / den)


def spearman_plain(x, y):
    rx, ry = rankdata(np.asarray(x, float)), rankdata(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def e1a(depth, r, kinds, tasks, n_boot=N, seed=SEED):
    full = wrho(depth, r, kinds)
    loo = []
    for i in range(len(tasks)):
        keep = [j for j in range(len(tasks)) if j != i]
        loo.append({"task": tasks[i], "rho": wrho([depth[j] for j in keep], [r[j] for j in keep], [kinds[j] for j in keep])})
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(tasks), len(tasks))
        v = wrho([depth[j] for j in idx], [r[j] for j in idx], [kinds[j] for j in idx])
        if v is not None:
            boots.append(v)
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (None, None)
    deep = [i for i, d in enumerate(depth) if d > 1]
    rho_deep = wrho([depth[i] for i in deep], [r[i] for i in deep], [kinds[i] for i in deep])
    vals = [x["rho"] for x in loo if x["rho"] is not None]
    fragile = [x["task"] for x in loo if x["rho"] is not None and x["rho"] > -0.20]
    moved = sorted(loo, key=lambda x: -abs((x["rho"] if x["rho"] is not None else full) - full))[:3]
    if fragile:
        verdict = "Fragile"
    elif vals and all(v <= -0.40 for v in vals) and hi is not None and hi < 0:
        verdict = "Robust"
    else:
        verdict = "Mostly robust"
    return {"rho": full, "loo": loo, "loo_min": min(vals), "loo_max": max(vals), "boot_n": len(boots), "boot_lo": lo, "boot_hi": hi,
            "rho_depth_gt1": rho_deep, "n_depth_gt1": len(deep), "fragile_tasks": fragile,
            "most_moving": [{"task": m["task"], "rho": m["rho"]} for m in moved], "verdict": verdict}


def e1b(rows, n_perm=N, seed=SEED):
    out = {}
    for f in FRACTIONS:
        use = [(r, healed_l_on(r["curve"], r["pchance"], r["healed_acc"], f)) for r in rows]
        use = [(r, d) for r, d in use if d is not None]
        t = B.h2a_test([d for _, d in use], [r["r12"] for r, _ in use], [r["kind"] for r, _ in use], n_perm, seed)
        out[str(f)] = {"rho": t["rho"], "p": t["p"], "n": len(use), "depths": {r["task"]: d for r, d in use}}
    ok = {f: (v["rho"] is not None and v["rho"] <= -0.40 and v["p"] is not None and v["p"] < 0.05) for f, v in out.items()
          if f != "0.5"}
    out["verdict"] = "Holds" if all(ok.values()) else "Threshold-dependent"
    out["failing_fractions"] = [f for f, v in ok.items() if not v]
    return out


def e1c(depth, r, kinds, n_perm=N, seed=SEED):
    kind = [1.0 if k == "appraise" else 0.0 for k in kinds]
    obs = ranked_partial(kind, r, depth)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        s = ranked_partial(rng.permutation(kind), r, depth)
        if s is not None and obs is not None and abs(s) >= abs(obs) - 1e-12:
            hits += 1
    p = (hits + 1) / (n_perm + 1)
    if obs is not None and abs(obs) < 0.20 and p >= 0.05:
        verdict = "Kind adds nothing measurable beyond depth"
    elif obs is not None and abs(obs) >= 0.30 and p < 0.05:
        verdict = "Kind has its own effect"
    else:
        verdict = "Unclear"
    return {"rho_kind_depth": spearman_plain(kind, depth), "rho_kind_r12": spearman_plain(kind, r), "partial": obs, "p": p,
            "verdict": verdict}


def e1d(rows, ks=KS, kref=12, n_perm=N, seed=SEED):
    out = {}
    depth = [r["L_on_healed"] for r in rows]
    for k in ks:
        rk = [r_at(r["acc"][str(k)]["mean"], r["acc"]["0"]["mean"], r["chance"]) for r in rows]
        t = B.h2a_test(depth, rk, [r["kind"] for r in rows], n_perm, seed)
        grp = {}
        for name, sel in (("shallow (<=3)", lambda d: d <= 3), ("middle (4-10)", lambda d: 4 <= d <= 10), ("deep (>10)", lambda d: d > 10)):
            v = [x for x, d in zip(rk, depth) if sel(d)]
            grp[name] = {"n": len(v), "mean_R": float(np.mean(v)) if v else None}
        out[str(k)] = {"rho": t["rho"], "p": t["p"], "groups": grp, "R": dict(zip([r["task"] for r in rows], rk))}
    sig = sum(1 for v in out.values() if v["p"] is not None and v["p"] < 0.05)
    neg = all(v["rho"] is not None and v["rho"] < 0 for v in out.values())
    out["verdict"] = "Dose-response" if neg and sig >= 2 else "Not dose-response"
    out["n_sig"] = sig
    return out


# ------------------------------------------------------------------ files

def _curve_rows(rows, curve_path, kref):
    """Attach curve, chance, healed_acc and the check against `r12` to `rows` (already D1-usable)."""
    out = []
    for r in rows:
        h = X.load(curve_path(r["task"]))
        r = dict(r, curve=h["curve"], pchance=h["chance"], healed_acc=h["healed_accuracy"])  # `chance` stays the summary's (R)
        if kref == 12:
            calc = r_at(r["acc"]["12"]["mean"], r["acc"]["0"]["mean"], r["chance"])
            if calc is None or abs(calc - r["r12"]) > 1e-6:
                raise SystemExit(f"SANITY FAIL: {r['task']} R12 recomputed {calc} != stored {r['r12']}")
        out.append(r)
    return out


def profile_17():
    c1 = X.C1
    rows, _ = X.d1_split(X.d1_rows_17(c1))
    return {"label": "1.7B", "primary": True, "rows": _curve_rows(rows, lambda t: c1 / f"healed_probe_{t}.json", 12),
            "expect_rho": -0.721, "d1_key": "1.7B"}


def profile_06():
    def path(t):
        for d in ("atlas2b", "atlas2"):
            p = ROOT / f"var/results/{d}/healed_{t}.json"
            if p.exists():
                return p
        raise SystemExit(f"no 0.6B healed file for {t}")
    rows, _ = X.d1_split(X.d1_rows_06(ROOT))
    return {"label": "0.6B", "primary": False, "rows": _curve_rows(rows, path, 12), "expect_rho": -0.658, "d1_key": "0.6B"}


def profile_extra(spec):
    label, summary, curves, kref = spec.split(",")
    kref, curves = int(kref), Path(curves)
    rows = []
    s = X.load(summary)
    # D3's summary names R `r` (its k is kref, not 12) and stores only k0 and the cut level's accuracy.
    s["rows"] = [dict(r, r12=r.get("r12", r.get("r")),
                      acc=r.get("acc") or {"0": {"mean": r.get("healed_k0")}, str(kref): {"mean": r.get("acc_cut")}})
                 for r in s["rows"]]
    for r in X.rows_of(s):
        p = curves / f"healed_probe_{r['task']}.json"
        h = X.load(p)
        rows.append(dict(r, L_on_healed=h["L_on_healed"], never=bool(h["never"]), healed_accuracy=h["healed_accuracy"]))
    usable, _ = X.d1_split(rows)
    return {"label": label, "primary": False, "kref": kref, "ks": (kref,),
            "rows": _curve_rows(usable, lambda t: curves / f"healed_probe_{t}.json", kref), "expect_rho": None, "d1_key": None}


def run_profile(p, d1, n=N):
    rows = p["rows"]
    kref = p.get("kref", 12)
    depth_at_half = []
    for r in rows:
        again = healed_l_on(r["curve"], r["pchance"], r["healed_acc"], 0.5)
        if again != r["L_on_healed"]:
            raise SystemExit(f"SANITY FAIL: {p['label']} {r['task']} recomputed L_on_healed {again} != stored {r['L_on_healed']}")
        depth_at_half.append(again)
    if d1 and p["d1_key"]:
        for t in d1[p["d1_key"]]["x1"]["per_task"]:
            mine = next(r for r in rows if r["task"] == t["task"])["L_on_healed"]
            if mine != t["L_on"]:
                raise SystemExit(f"SANITY FAIL: {p['label']} {t['task']} L_on_healed {mine} != d1.json {t['L_on']}")
    tasks = [r["task"] for r in rows]
    kinds = [r["kind"] for r in rows]
    y = [r["r12"] for r in rows]
    full = wrho(depth_at_half, y, kinds)
    if p["expect_rho"] is not None and (full is None or abs(full - p["expect_rho"]) > 0.0006):
        raise SystemExit(f"SANITY FAIL: {p['label']} full-set rho {full} != {p['expect_rho']}")
    deep_judging = [r["task"] for r in rows if r["kind"] == "appraise" and r["L_on_healed"] >= DEEP_STEP]
    return {"label": p["label"], "primary": p["primary"], "n_tasks": len(rows), "kref": kref, "full_rho": full,
            "tasks": [{"task": r["task"], "kind": r["kind"], "L_on_healed": r["L_on_healed"], "R": r["r12"]} for r in rows],
            "E1a": e1a(depth_at_half, y, kinds, tasks, n), "E1b": e1b(rows, n), "E1c": e1c(depth_at_half, y, kinds, n),
            "E1d": e1d(rows, p.get("ks", KS), kref, n),
            "deep_judging_tasks": deep_judging}


# ------------------------------------------------------------------ report

def _f(v, n=3):
    return "n/a" if v is None else f"{v:.{n}f}"


def markdown(results):
    md = ["# Step E1: free robustness checks (docs/ATLAS-robustness.md)", "",
          "Depth = `L_on_healed`; R = R12 (R_k for E1-d). rho = D1-A's weighted within-kind Spearman (appraise and execute separately, "
          "combined by task count); p one-sided by permutation within kind. Bootstrap and permutation seed 0, 10,000 each. "
          "E1-e skipped. `L_on_healed` at fraction f is `chance + f * (healed accuracy - chance)`, the same top D1 used.", ""]
    for res in results:
        tag = "primary, bars apply" if res["primary"] else "reported, no bar"
        a, b, c, d = res["E1a"], res["E1b"], res["E1c"], res["E1d"]
        md += [f"## {res['label']} ({tag}); {res['n_tasks']} tasks, full-set rho {_f(res['full_rho'])}", ""]
        v = (lambda w: f"**{w}**") if res["primary"] else (lambda w: f"({w}, reported, no bar)")
        md += [f"### E1-a single-task influence: {v(a['verdict'])}", "",
               f"- leave-one-out rho range {_f(a['loo_min'])} to {_f(a['loo_max'])}; bootstrap 95% interval over tasks "
               f"[{_f(a['boot_lo'])}, {_f(a['boot_hi'])}] ({a['boot_n']} valid resamples)",
               f"- rho over tasks with L_on_healed > 1 only (reported): {_f(a['rho_depth_gt1'])} (n {a['n_depth_gt1']})",
               f"- tasks with leave-one-out rho > -0.20: {a['fragile_tasks'] or 'none'}",
               "- tasks that move rho most: " + ", ".join(f"{m['task']} ({_f(m['rho'])})" for m in a["most_moving"]), "",
               "| left out | rho |", "|---|---|"]
        md += [f"| {x['task']} | {_f(x['rho'])} |" for x in a["loo"]]
        md += ["", f"### E1-b depth threshold: {v(b['verdict'])}", "", "| fraction | rho | p | n |", "|---|---|---|---|"]
        md += [f"| {f} | {_f(b[str(f)]['rho'])} | {_f(b[str(f)]['p'], 4)} | {b[str(f)]['n']} |" for f in FRACTIONS]
        if b["failing_fractions"]:
            md += ["", f"Fractions failing the bar: {b['failing_fractions']}"]
        md += ["", f"### E1-c kind through depth: {v(c['verdict'])}", "",
               f"- rho(kind, depth) {_f(c['rho_kind_depth'])}; rho(kind, R12) {_f(c['rho_kind_r12'])}; "
               f"partial rho(kind, R12 | depth) {_f(c['partial'])}, permutation p (two-sided) {_f(c['p'], 4)}; kind = 1 appraise, 0 execute",
               f"- deep judging tasks (appraise, L_on_healed >= {DEEP_STEP}): {len(res['deep_judging_tasks'])} {res['deep_judging_tasks']}"
               " (the paper says at most 3; E2 decides the wording of claim 6)", "",
               f"### E1-d cut size: {v(d['verdict'])}", "", "| k | rho(L_on_healed, R_k) | p | " +
               " | ".join(f"mean R {g}" for g in next(x for x in d.values() if isinstance(x, dict))["groups"]) + " |",
               "|---|---|---|" + "---|" * 3]
        for k in sorted((x for x in d if x.isdigit()), key=int):
            e = d[k]
            md.append(f"| {k} | {_f(e['rho'])} | {_f(e['p'], 4)} | " +
                      " | ".join(f"{_f(g['mean_R'], 2)} (n {g['n']})" for g in e["groups"].values()) + " |")
        md += ["", f"rho < 0 at every k and p < 0.05 at {d['n_sig']} of {len(d) - 2}.", ""]
    return "\n".join(md)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extra", action="append", default=[], help="LABEL,SUMMARY.json,CURVE_DIR,K (e.g. D3 later)")
    ap.add_argument("--n", type=int, default=N, help="resamples and permutations (tests only)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    d1 = X.load(OUT / "d1.json")
    profiles = [profile_17(), profile_06()] + [profile_extra(s) for s in args.extra]
    results = [run_profile(p, d1, args.n) for p in profiles]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e1.json").write_text(json.dumps({r["label"]: r for r in results}, indent=1, default=str))
    md = markdown(results)
    (out / "e1.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
