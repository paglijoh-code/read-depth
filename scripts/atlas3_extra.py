"""Stage C extra analyses X1 and X2 (docs/ATLAS-phase2-stageC.md, last section; post hoc, $0, stored results only).

X1: is H2-A mechanical? n_pre = cut blocks (first 12 of the shared order) that sit before the read point (block < L_on).
X2: why H2-B moved between 0.6B (stage B) and 1.7B (C1): scoring/healing, depth grouping, or the tasks themselves.
Writes var/results/atlas3/extra.json and extra.md.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import atlas2_cuts as B  # noqa: E402
from surgery.atlas import heal  # noqa: E402

_POS = [a for a in sys.argv[1:] if not a.startswith("--")]
C1 = Path(_POS[0]) if _POS else ROOT / "var/results/atlas3/c1"
OUT = ROOT / "var/results/atlas3"
N_PERM = 10000
K = 12


def load(p):
    return json.loads(Path(p).read_text())


def r12(acc, k0, chance):
    if k0 - chance <= 1e-9:
        return None
    return float(min(1.0, max(0.0, (acc - chance) / (k0 - chance))))


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y holding z: Pearson on ranks, then the usual partial formula."""
    rx, ry, rz = (np.argsort(np.argsort(v)).astype(float) for v in (x, y, z))
    c = np.corrcoef([rx, ry, rz])
    a, b, d = c[0, 1], c[0, 2], c[1, 2]
    den = math.sqrt(max(1e-12, (1 - b * b) * (1 - d * d)))
    return float((a - b * d) / den)


def rows_of(summary):
    return [r for r in summary["rows"] if r.get("r12") is not None and r.get("L_on") is not None and not r.get("dropped")]


def x1(rows, cut, label):
    lon = [r["L_on"] for r in rows]
    y = [r["r12"] for r in rows]
    kinds = [r["kind"] for r in rows]
    npre = [sum(1 for b in cut if b < L) for L in lon]
    main = B.h2a_test(lon, y, kinds, N_PERM)
    mech = B.h2a_test(npre, y, kinds, N_PERM)
    zero = [(r["task"], r["kind"], r["L_on"], round(r["r12"], 3)) for r, n in zip(rows, npre) if n == 0]
    zl = [z[2] for z in zero]
    zy = [z[3] for z in zero]
    rho_zero = heal.spearman(zl, zy) if len(set(zl)) > 1 else None
    return {"model": label, "cut": sorted(cut), "n_tasks": len(rows),
            "h2a_Lon": main, "h2a_npre": mech,
            "partial_Lon_given_npre": partial_spearman(lon, y, npre),
            "partial_npre_given_Lon": partial_spearman(npre, y, lon),
            "npre_zero": {"tasks": zero, "n": len(zero), "rho_Lon": rho_zero,
                          "n_below_0.6": sum(1 for v in zy if v < 0.6), "n_at_least_0.8": sum(1 for v in zy if v >= 0.8)},
            "per_task": [{"task": r["task"], "kind": r["kind"], "L_on": r["L_on"], "n_pre": n, "r12": r["r12"]}
                         for r, n in zip(rows, npre)]}


def h2b(rows, groups=None):
    g = groups or [B.l_on_group(r["L_on"]) for r in rows]
    return B.h2b_test(g, [r["r12"] for r in rows], [r["kind"] for r in rows], N_PERM)


def whole_r12_06():
    """0.6B doing tasks: R12 from stage B's whole-answer exact-match arm (seed means, best lr per k by mean)."""
    runs = {}
    for i in (0, 1):
        runs.update(load(ROOT / f"var/results/atlas2b/whole.shard{i}of2.json")["runs"])
    by = {}
    for r in runs.values():
        by.setdefault((r["task"], r["k"], r["lr"]), []).append(r["exact_match"])
    out = {}
    for t in sorted({k[0] for k in by}):
        def best(k):
            c = [np.mean(v) for (tt, kk, _), v in by.items() if tt == t and kk == k]
            return float(max(c)) if c else None
        out[t] = (best(0), best(K))
    return out


def main():
    sb = load(ROOT / "var/results/atlas2b/summary.json")
    sc = load(C1 / "summary.json")
    o6 = load(ROOT / "var/results/atlas/order.json")["shared_order"][:K]
    o17 = load(ROOT / "var/results/atlas3/1.7B/order.json")["shared_order"][:K]
    r6, r17 = rows_of(sb), rows_of(sc)
    res = {"x1": [x1(r6, o6, "0.6B"), x1(r17, o17, "1.7B")]}

    # X2
    base6, base17 = h2b(r6), h2b(r17)
    w = whole_r12_06()
    r6w = []
    for r in r6:
        r = dict(r)
        if r["kind"] == "execute" and r["task"] in w and None not in w[r["task"]]:
            k0, a12 = w[r["task"]]
            r["r12_first_token"] = r["r12"]
            r["r12"] = r12(a12, k0, r["chance"])
        r6w.append(r)
    r6w = [r for r in r6w if r["r12"] is not None]
    a_scoring = h2b(r6w)
    g6 = {r["task"]: B.l_on_group(r["L_on"]) for r in r6}
    g17 = {r["task"]: B.l_on_group(r["L_on"]) for r in r17}
    common = sorted(set(g6) & set(g17))
    c17 = [r for r in r17 if r["task"] in common]
    c6 = [r for r in r6 if r["task"] in common]
    b_17_with_06groups = h2b(c17, [g6[r["task"]] for r in c17])
    b_17_common = h2b(c17)
    b_06_with_17groups = h2b(c6, [g17[r["task"]] for r in c6])
    A6, A17 = {r["task"]: r for r in r6}, {r["task"]: r for r in r17}
    W6 = {r["task"]: r for r in r6w}
    moves = [{"task": t, "kind": A6[t]["kind"], "L_on_06": A6[t]["L_on"], "L_on_17": A17[t]["L_on"],
              "r12_06": A6[t]["r12"], "r12_06_whole": W6[t]["r12"] if t in W6 else None, "r12_17": A17[t]["r12"],
              "delta": A17[t]["r12"] - A6[t]["r12"]} for t in common]
    mean_move = {k: float(np.mean([m["delta"] for m in moves if m["kind"] == k])) for k in ("appraise", "execute")}
    gap = base6["stat"] - base17["stat"]
    res["x2"] = {"h2b_06": base6, "h2b_17": base17, "gap": gap,
                 "a_06_whole_answer": a_scoring, "b_17_with_06_groups": b_17_with_06groups,
                 "b_17_common_tasks": b_17_common, "b_06_with_17_groups": b_06_with_17groups,
                 "c_moves": moves, "c_mean_move_by_kind": mean_move,
                 "explains": {"a_scoring": abs(base6["stat"] - a_scoring["stat"]) >= gap / 2,
                              "b_grouping_on_17": abs(b_17_with_06groups["stat"] - b_17_common["stat"]) >= gap / 2,
                              "b_grouping_on_06": abs(b_06_with_17groups["stat"] - base6["stat"]) >= gap / 2}}
    (OUT / "extra.json").write_text(json.dumps(res, indent=1))

    f = lambda v, n=3: "n/a" if v is None else f"{v:.{n}f}"  # noqa: E731
    md = ["# Stage C extra analyses (post hoc)", ""]
    for x in res["x1"]:
        md += [f"## X1 {x['model']} (cut blocks {x['cut']}, {x['n_tasks']} tasks)",
               f"- H2-A with L_on: rho {f(x['h2a_Lon']['rho'])}, p {f(x['h2a_Lon']['p'], 4)}",
               f"- H2-A with n_pre (mechanical rival): rho {f(x['h2a_npre']['rho'])}, p {f(x['h2a_npre']['p'], 4)}",
               f"- partial rho L_on | n_pre {f(x['partial_Lon_given_npre'])}; n_pre | L_on {f(x['partial_npre_given_Lon'])}",
               f"- n_pre = 0 tasks: {x['npre_zero']['n']}, R12 < 0.6: {x['npre_zero']['n_below_0.6']}, "
               f">= 0.8: {x['npre_zero']['n_at_least_0.8']}, rho(L_on, R12) inside {f(x['npre_zero']['rho_Lon'])}",
               "", "| task | kind | L_on | n_pre | R12 |", "|---|---|---|---|---|"]
        md += [f"| {p['task']} | {p['kind']} | {p['L_on']} | {p['n_pre']} | {p['r12']:.2f} |" for p in x["per_task"]]
        md.append("")
    x = res["x2"]
    md += ["## X2 why H2-B changed",
           f"- H2-B 0.6B {f(x['h2b_06']['stat'])} (p {f(x['h2b_06']['p'], 4)}); 1.7B {f(x['h2b_17']['stat'])} "
           f"(p {f(x['h2b_17']['p'], 4)}); gap {f(x['gap'])}",
           f"- (a) 0.6B with whole-answer exact match for doing tasks: {f(x['a_06_whole_answer']['stat'])} "
           f"(p {f(x['a_06_whole_answer']['p'], 4)}) groups {json.dumps(x['a_06_whole_answer']['groups'])}",
           f"- (b) 1.7B common tasks own groups {f(x['b_17_common_tasks']['stat'])}; with 0.6B groups "
           f"{f(x['b_17_with_06_groups']['stat'])}; 0.6B with 1.7B groups {f(x['b_06_with_17_groups']['stat'])}",
           f"- (c) mean R12 move 0.6B to 1.7B by kind: {json.dumps({k: round(v, 3) for k, v in x['c_mean_move_by_kind'].items()})}",
           f"- explains (moves stat by at least half the gap): {json.dumps(x['explains'])}",
           "", "| task | kind | L_on 0.6B | L_on 1.7B | R12 0.6B | 0.6B whole | R12 1.7B |", "|---|---|---|---|---|---|---|"]
    md += [f"| {m['task']} | {m['kind']} | {m['L_on_06']} | {m['L_on_17']} | {m['r12_06']:.2f} | "
           f"{f(m['r12_06_whole'], 2)} | {m['r12_17']:.2f} |" for m in x["c_moves"]]
    (OUT / "extra.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ---------------------------------------------------------------- step D1: depth measured on the healed model
# docs/ATLAS-phase2-stageC.md, "Step D1". Pure functions take plain lists/dicts; d1_main() does the file work.

D1_MARGIN = 0.10
D1_TOTAL = 24


def d1_compare(healed, other, margin=D1_MARGIN):
    """"sharper" if the healed rho is more negative by at least `margin`, "weaker" if less negative by at least it, else "same"."""
    if healed is None or other is None:
        return None
    if healed <= other - margin + 1e-9:
        return "sharper"
    if healed >= other + margin - 1e-9:
        return "weaker"
    return "same"


def d1_split(rows):
    """(usable, excluded {reason: [tasks]}): usable has an `L_on_healed` and is not `never`."""
    usable, ex = [], {"never": [], "none": [], "missing": []}
    for r in rows:
        if r.get("healed_missing"):
            ex["missing"].append(r["task"])
        elif r.get("never"):
            ex["never"].append(r["task"])
        elif r.get("L_on_healed") is None:
            ex["none"].append(r["task"])
        else:
            usable.append(r)
    return usable, ex


def d1_size(rows, cut, label, n_perm=N_PERM):
    """D1-A, D1-X1 and D1-B for one model size. `rows` carry the `rows_of` fields plus L_on_healed, never, healed_accuracy
    and (1.7B, file absent) healed_missing."""
    usable, ex = d1_split(rows)
    kinds, y = [r["kind"] for r in usable], [r["r12"] for r in usable]
    healed = B.h2a_test([r["L_on_healed"] for r in usable], y, kinds, n_perm)
    same = B.h2a_test([r["L_on"] for r in usable], y, kinds, n_perm)
    full = B.h2a_test([r["L_on"] for r in rows], [r["r12"] for r in rows], [r["kind"] for r in rows], n_perm)
    sr = [r for r in usable if r.get("surface")]
    rival = B.h2a_test([-r["surface"]["gap"] for r in sr], [r["r12"] for r in sr], [r["kind"] for r in sr], 0)
    cmp_same, cmp_full = d1_compare(healed["rho"], same["rho"]), d1_compare(healed["rho"], full["rho"])
    swapped = [{**r, "L_on_unmodified": r["L_on"], "L_on": r["L_on_healed"]} for r in usable]
    out = {"model": label, "n_rows": len(rows), "n_usable": len(usable), "excluded": ex, "tasks": [r["task"] for r in usable],
           "healed": healed, "unmod_same": same, "unmod_full": full, "rival_rho": rival["rho"], "rival_n": rival["n"],
           "label": B.label_a(healed["rho"], healed["p"], rival["rho"]), "compare": cmp_same, "compare_vs_full": cmp_full,
           "compare_differs": cmp_same != cmp_full}
    if len(usable) >= 2:
        out["x1"] = x1(swapped, cut, label)
        out["h2b"] = h2b(swapped)
    else:
        out["x1"] = out["h2b"] = None
    return out


def d1_decision(d17, d06):
    """The doc's rule. `d17` / `d06` are d1_size results. Returns (decision or None, text)."""
    miss = d17["excluded"]["missing"]
    if miss:
        n = d17["n_rows"] - len(miss)
        return None, f"INCOMPLETE {n}/{d17['n_rows']} (1.7B missing: {miss})"
    rho = d17["healed"]["rho"]
    if d17["label"] == "supported" and d17["compare"] != "weaker" and d06["compare"] != "weaker":
        return "healed_primary", "1.7B D1-A supported and not weaker at either size: L_on_healed is the primary depth measure"
    if d17["compare"] == "weaker" and rho is not None and rho <= -0.2:
        return "unmodified_primary_report_both", "1.7B weaker but rho <= -0.2: unmodified L_on stays primary, report both, call the measure fragile"
    if rho is not None and rho > -0.2:
        return "depth_claim_unmodified_only_reconsider_D3", "1.7B healed rho > -0.2: depth claim holds only for the unmodified measure; reconsider D3"
    return "no_rule_matched", (f"no rule matched: 1.7B label {d17['label']}, compare {d17['compare']}, healed rho {rho}, "
                               f"0.6B compare {d06['compare']}")


def d1_markdown(d17, d06, rows17, rows06, decision, text):
    f = lambda v, n=3: "n/a" if v is None else f"{v:.{n}f}"  # noqa: E731
    md = ["# Step D1: depth measured on the trained model", ""]
    for d, rows, tag in ((d17, rows17, "test (blind)"), (d06, rows06, "description (already seen)")):
        h, u, fu = d["healed"], d["unmod_same"], d["unmod_full"]
        md += [f"## {d['model']} - {tag}", "",
               f"- tasks {d['n_rows']}, usable {d['n_usable']}; excluded: never {d['excluded']['never']}, "
               f"no healed value {d['excluded']['none']}, missing {d['excluded']['missing']}",
               f"- D1-A healed L_on: rho {f(h['rho'])}, p {f(h['p'], 4)}, rival (negated surface) rho {f(d['rival_rho'])} "
               f"(n {d['rival_n']}): **{d['label']}**",
               f"- unmodified L_on, same {d['n_usable']} tasks: rho {f(u['rho'])}, p {f(u['p'], 4)}; full set ({d['n_rows']}): "
               f"rho {f(fu['rho'])}, p {f(fu['p'], 4)}",
               f"- healed vs unmodified (same tasks): **{d['compare']}**; vs full-set unmodified: {d['compare_vs_full']}"
               + (" (differs)" if d["compare_differs"] else "")]
        if d["x1"]:
            x = d["x1"]
            md += [f"- D1-X1 (n_pre from L_on_healed, cut {x['cut']}): H2-A rho {f(x['h2a_Lon']['rho'])} p {f(x['h2a_Lon']['p'], 4)}; "
                   f"n_pre rho {f(x['h2a_npre']['rho'])} p {f(x['h2a_npre']['p'], 4)}; partial L_on|n_pre "
                   f"{f(x['partial_Lon_given_npre'])}, n_pre|L_on {f(x['partial_npre_given_Lon'])}; n_pre = 0 tasks "
                   f"{x['npre_zero']['n']}, rho inside {f(x['npre_zero']['rho_Lon'])}",
                   f"- D1-B (groups from L_on_healed, report only): stat {f(d['h2b']['stat'])}, p {f(d['h2b']['p'], 4)}, "
                   f"groups {json.dumps(d['h2b']['groups'])}"]
        npre = {p["task"]: p["n_pre"] for p in (d["x1"] or {"per_task": []})["per_task"]}
        md += ["", "| task | kind | L_on | L_on_healed | never | healed acc | n_pre (healed) | R12 |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['task']} | {r['kind']} | {r['L_on']} | {'missing' if r.get('healed_missing') else r.get('L_on_healed')} | "
                      f"{r.get('never')} | {f(r.get('healed_accuracy'))} | {npre.get(r['task'], '-')} | {f(r['r12'], 2)} |")
        md.append("")
    md.append(f"decision: {decision} - {text}" if decision else f"{text}\ndecision: null")
    return "\n".join(md) + "\n"


def d1_rows_17(c1):
    rows = []
    for r in rows_of(load(c1 / "summary.json")):
        p = c1 / f"healed_probe_{r['task']}.json"
        r = dict(r)
        if p.exists():
            h = load(p)
            r.update(L_on_healed=h["L_on_healed"], never=bool(h["never"]), healed_accuracy=h["healed_accuracy"])
        else:
            r.update(L_on_healed=None, never=False, healed_accuracy=None, healed_missing=True)
        rows.append(r)
    return rows


def d1_rows_06(root):
    rows = []
    for r in rows_of(load(root / "var/results/atlas2b/summary.json")):
        r = dict(r)
        r.update(never=False, healed_accuracy=None)
        for d in ("atlas2b", "atlas2"):
            p = root / f"var/results/{d}/healed_{r['task']}.json"
            if p.exists():
                h = load(p)
                r.update(never=bool(h.get("never")), healed_accuracy=h.get("healed_accuracy"))
                break
        rows.append(r)
    return rows


def d1_main():
    o6 = load(ROOT / "var/results/atlas/order.json")["shared_order"][:K]
    o17 = load(ROOT / "var/results/atlas3/1.7B/order.json")["shared_order"][:K]
    rows17, rows06 = d1_rows_17(C1), d1_rows_06(ROOT)
    d17, d06 = d1_size(rows17, o17, "1.7B"), d1_size(rows06, o6, "0.6B")
    decision, text = d1_decision(d17, d06)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d1.json").write_text(json.dumps({"1.7B": d17, "0.6B": d06, "decision": decision, "decision_text": text},
                                            indent=1, default=str))
    md = d1_markdown(d17, d06, rows17, rows06, decision, text)
    (OUT / "d1.md").write_text(md)
    print(md)


if __name__ == "__main__":
    if "--d1" in sys.argv[1:]:
        d1_main()
    else:
        main()
