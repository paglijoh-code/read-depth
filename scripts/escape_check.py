"""Escape check (docs/ATLAS-escape.md): post hoc, exploratory, saved data only.

Usage: python scripts/escape_check.py  -> prints markdown, writes var/results/atlas3/escape.json
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "var/results/atlas3"


def load_runs(paths):
    out = {}
    for p in paths:
        for v in json.load(open(p))["runs"].values():
            out[(v.get("arm", ""), v["task"], v["k"], v["seed"], v["lr"])] = v
    return out


def avg_ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = avg_ranks(x), avg_ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def weighted_within(rows):
    """rows: (kind, depth, value). Task-count-weighted mean of within-kind Spearman."""
    by = collections.defaultdict(list)
    for kind, d, v in rows:
        by[kind].append((d, v))
    num = den = 0.0
    per = {}
    for kind, xs in by.items():
        rho = spearman([a for a, _ in xs], [b for _, b in xs])
        per[kind] = rho
        if rho is not None:
            num += rho * len(xs)
            den += len(xs)
    return (num / den if den else None), per, spearman([d for _, d, _ in rows], [v for _, _, v in rows])


def winner_lr(runs, arm, task, k):
    lrs = {key[4] for key in runs if key[:3] == (arm, task, k) and key[3] == 1}
    return lrs.pop() if lrs else None


def triple(runs, arm, task, k):
    lr = winner_lr(runs, arm, task, k)
    if lr is None:
        return None
    hs = [runs.get((arm, task, k, s, lr)) for s in (0, 1, 2)]
    return None if None in hs else hs


def escape_step(h, m):
    for p in h["curve"]:
        if p["lrsel_exact"] >= m:
            return p["step"]
    return None


def plateau(h):
    ex = [e["text"] for e in h.get("examples", [])]
    return len(ex) >= 5 and len(set(ex)) == 1


def analyse(name, runs, k0runs, k, arm, tasks):
    """tasks: dict task -> (kind, depth)."""
    out = {"name": name, "tasks": {}}
    stuck = sig = 0
    splits = []
    rows = []
    for t, (kind, depth) in tasks.items():
        hs = triple(runs, arm, t, k)
        k0 = triple(k0runs, "", t, 0)
        if hs is None or k0 is None:
            continue
        target = sum(h["best_lrsel"] for h in k0) / 3
        chance = hs[0]["chance"]
        m = chance + 0.5 * (target - chance)
        steps = [escape_step(h, m) for h in hs]
        for h, s in zip(hs, steps):
            if s is None:
                stuck += 1
                sig += plateau(h)
        esc = [s for s in steps if s is not None]
        if esc and len(esc) < 3:
            splits.append(t)
        mean_esc = sum(esc) / len(esc) if esc else None
        out["tasks"][t] = {"kind": kind, "depth": depth, "m": round(m, 3), "steps": steps,
                           "plateau": [plateau(h) for h in hs], "mean_escape": mean_esc}
        if mean_esc is not None:
            rows.append((kind, depth, mean_esc))
    w, per, pooled = weighted_within(rows)
    out.update({"stuck": stuck, "plateau_sig": sig, "T1": sig / stuck if stuck else None,
                "T2_splits": splits, "T3": w, "T3_by_kind": per, "T3_pooled": pooled,
                "T3_n": len(rows)})
    return out


def main():
    c1 = load_runs([RES / "c1/runs.shard0of1.json"])
    d3 = load_runs([RES / "d3/runs.shard0of1.json"])
    e4 = load_runs(glob.glob(str(RES / "e4/e4_*.json")))
    d2 = load_runs(glob.glob(str(RES / "d2/d2_long.*.json")))
    e1 = json.load(open(RES / "e1.json"))["1.7B"]["tasks"]
    q17 = {r["task"]: (r["kind"], r["L_on_healed"]) for r in e1}
    smol = {r["task"]: (r["kind"], r["L_on_healed"]) for r in json.load(open(RES / "d3/summary.json"))["rows"]
            if not r.get("dropped") and r.get("L_on_healed") is not None}
    res = [analyse("Qwen3-1.7B C1 k12 (primary)", c1, c1, 12, "", q17),
           analyse("Qwen3-1.7B C1 k9", c1, c1, 9, "", q17),
           analyse("Qwen3-1.7B C1 k16", c1, c1, 16, "", q17),
           analyse("SmolLM2-1.7B D3 k10", d3, d3, 10, "", smol)]
    for arm in ("late17", "late15"):
        qual = {t: v for t, v in q17.items() if t not in ("hard", "code_output")}
        res.append(analyse(f"Qwen3-1.7B E4 {arm} (n_pre 0 tasks)", e4, c1, 10, arm, qual))
    # T4: D2 long arm
    t4 = {}
    for key, h in d2.items():
        arm, t, k, s, lr = key
        if lr != winner_lr(d2, arm, t, k):
            continue
        t4.setdefault(t, {})[s] = [(p["step"], p["lrsel_exact"]) for p in h["curve"]]
    (RES / "escape.json").write_text(json.dumps({"results": res, "T4": t4}, indent=1, default=str))

    print("# Escape check results (post hoc, exploratory; criteria `docs/ATLAS-escape.md`)\n")
    print("| dataset | stuck heals | with plateau signature | T1 share | seed splits (T2) | T3 weighted ρ(depth, escape step) | by kind | pooled | tasks in T3 |")
    print("|---|---|---|---|---|---|---|---|---|")
    f = lambda x: "—" if x is None else f"{x:.3f}"
    for r in res:
        print(f"| {r['name']} | {r['stuck']} | {r['plateau_sig']} | {f(r['T1'])} | {len(r['T2_splits'])} ({', '.join(r['T2_splits'])}) | {f(r['T3'])} | "
              + ", ".join(f"{k} {f(v)}" for k, v in r["T3_by_kind"].items()) + f" | {f(r['T3_pooled'])} | {r['T3_n']} |")
    print("\n## Per task, primary (C1 k12): depth, escape step per seed (— = stuck), plateau signature\n")
    print("| task | kind | depth | midpoint | escape steps | plateau |")
    print("|---|---|---|---|---|---|")
    for t, v in sorted(res[0]["tasks"].items(), key=lambda kv: kv[1]["depth"]):
        print(f"| {t} | {v['kind']} | {v['depth']} | {v['m']} | {', '.join('—' if s is None else str(s) for s in v['steps'])} | {', '.join('yes' if p else 'no' for p in v['plateau'])} |")
    print("\n## T4: D2 4x budget, lrsel accuracy by step (winning lr)\n")
    for t, ss in sorted(t4.items()):
        print(f"- {t}: " + "; ".join(f"s{s} " + " ".join(f"{st}:{a}" for st, a in c) for s, c in sorted(ss.items())))


if __name__ == "__main__":
    main()
