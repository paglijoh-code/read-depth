"""Tables for the M5 write-up from var/results/m5/*.json (and the two m4 test runs).

  uv run python scripts/m5_report.py   # prints markdown, writes var/results/m5/report_tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path("var/results/m5")
M4 = Path("var/results/m4")
MODEL = "Qwen3-0.6B"
VARIANTS = ["rule", "sensible"]
CATS = ["syntax", "factual", "reasoning", "instruction", "code"]
M4_TEST = {"rule": "d37b428cc779", "sensible": "5d5d48675301"}


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def m5(name):
    return load(R / f"{name}.json")


def pc(x, d=1):
    return f"{100 * x:.{d}f}"


def sgn(x, d=1):
    return f"{100 * x:+.{d}f}"


def ci(c, signed=False, d=1):
    f = sgn if signed else pc
    return f"[{f(c['lo'], d)}, {f(c['hi'], d)}]"


def describe(spec):
    if spec is None:
        return "-"
    op = spec.get("op")
    if op == "ablate":
        return f"{spec['site']} {spec['mode']} L{','.join(map(str, spec['layers']))}"
    if op == "ablate_kv":
        g = spec["groups"]
        return f"kv {spec['mode']} " + (", ".join(f"{a}.{b}" for a, b in g) if len(g) <= 3 else f"{len(g)} groups")
    if op == "compose":
        return " + ".join(describe(s) for s in spec["specs"])
    if op == "plan":
        order = spec["order"]
        missing = [b for b in range(28) if b not in order]
        reps = sorted({b for b in order if order.count(b) > 1})
        bits = ([f"drop {missing}"] if missing else []) + ([f"repeat {reps}"] if reps else [])
        if not bits and order != sorted(order):
            bits = ["swap " + str([i for i, (a, b) in enumerate(zip(order, sorted(order))) if a != b])]
        if spec.get("scales") and any(s != 1.0 for s in spec["scales"]):
            bits.append("scaled")
        return "plan " + " ".join(bits)
    if op == "steer" or "direction" in spec:
        return f"steer L{spec.get('layer')} {spec.get('direction')}"
    return json.dumps(spec)[:60]


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def excision():
    out = ["## (a) Greedy excision path (discovery, n=500)", ""]
    for V in VARIANTS:
        e = m5(f"excise_{MODEL}_{V}")
        if not e:
            continue
        by = {r["spec_hash"]: r for r in e["rows"]}
        base = e["baseline"]["accuracy"]
        out += [f"### {V} (baseline {pc(base)}%)", ""]
        rows = []
        for s in e["steps"]:
            r = by.get(s["spec_hash"])
            rows.append([s["step"], s["chosen"], pc(s["accuracy"]), sgn(s["accuracy"] - base),
                         f"{r['flops_relative']:.3f}" if r else "-"])
        out += table(["Step", "Block removed", "Discovery acc %", "vs baseline (pts)", "FLOPs rel"], rows)
    return out


def pareto():
    out = ["## (b) Pareto points on selection (n=1500)", ""]
    for V in VARIANTS:
        p = m5(f"pareto_{MODEL}_{V}")
        if not p:
            continue
        cb = p["canary_baseline"]["accuracy"]
        out += [f"### {V} (baseline {pc(p['baseline_accuracy'])}%, canary baseline {pc(cb)}%)", ""]
        rows = [[x["label"] + (f" ({x['groups']})" if x.get("groups") else ""), f"{x['flops_relative']:.3f}",
                 f"{pc(x['selection_accuracy'])} {ci(x['selection_accuracy_ci'])}", sgn(x["selection_gain"]),
                 sgn(x["canary_accuracy"] - cb), "pass" if x["eligible"] else "fail"] for x in p["points"]]
        out += table(["Point", "FLOPs rel", "Acc % [CI]", "Gain (pts)", "Canary overall (pts)", "Gate"], rows)
    return out


def tests():
    out = ["## (c) Test runs (n=1500)", ""]
    rows = []
    for V in VARIANTS:
        for h in ("h1", "h3"):
            t = m5(f"test_{MODEL}_{V}_{h}")
            if not t:
                continue
            base = t["baseline"]["accuracy"]
            for r in t["runs"].values():
                rows.append([V, h.upper(), r.get("label", describe(r["spec"])), pc(base), pc(r["accuracy"]),
                             f"{sgn(r['gain'])} {ci(r['gain_ci'], True)}", t["verdict"]])
        d = load(M4 / f"test_{MODEL}_{V}_test_{M4_TEST[V]}.json")
        if d:
            c, pr = d["cfg"], d["summary"]["paired"]
            rows.append([V, "H4 (m4, deferred)", f"steer L{c['layer']} {c['direction']} {c['op']} a={c['alpha']}",
                         pc(d["baseline"]["accuracy"]), pc(d["summary"]["accuracy"]),
                         f"{sgn(pr['point'])} {ci(pr, True)}", d["h4"]])
    return out + table(["Variant", "Hypothesis", "Intervention", "Base %", "After %", "Gain pts [CI]", "Verdict"], rows)


def h3():
    out = ["## (d) H3 candidates (selection split)", ""]
    for V in VARIANTS:
        p, s = m5(f"pareto_{MODEL}_{V}"), m5(f"select_{MODEL}_{V}")
        if not (p and s):
            continue
        by = {r["spec_hash"]: r for r in s["rows"]}
        cb = p["canary_baseline"]["per_category"]
        out += [f"### {V}", ""]
        rows = []
        for x in p["points"]:
            if x["kind"] != "h3":
                continue
            r = by.get(x["spec_hash"])
            drops = ", ".join(f"{c[:4]} {100 * (x['canary_per_category'][c] - cb[c]):+.0f}" for c in CATS)
            rows.append([x["label"], describe(r["spec"]) if r else x["spec_hash"],
                         f"{sgn(r['gain'])} {ci(r['gain_ci'], True)}" if r else sgn(x["selection_gain"]),
                         drops, "eligible" if x["eligible"] else "failed gate"])
        out += table(["Label", "Spec", "Selection gain pts [CI]", "Canary change (pts)", "Eligibility"], rows)
    return out


def kv():
    out = ["## (e) KV-group scan (discovery, n=200) and greedy path", ""]
    for V in VARIANTS:
        k = m5(f"kvscan_{MODEL}_{V}")
        if not k:
            continue
        rows_ = sorted(k["rows"], key=lambda r: -r["gain"])
        fmt = lambda r: [describe(r["spec"]), pc(r["accuracy"]), f"{sgn(r['gain'])} {ci(r['gain_ci'], True)}"]
        out += [f"### {V}: top 5 gains (of {len(rows_)})", ""] + table(["Group", "Acc %", "Gain pts [CI]"], [fmt(r) for r in rows_[:5]])
        out += [f"### {V}: top 5 losses", ""] + table(["Group", "Acc %", "Gain pts [CI]"], [fmt(r) for r in rows_[::-1][:5]])
        p = m5(f"kvpath_{MODEL}_{V}")
        if p:
            out += [f"### {V}: greedy KV removal path (discovery, n={p['n']}, baseline {pc(p['baseline']['accuracy'])}%)", ""]
            out += table(["Groups removed", "Acc %", "Gain pts [CI]", "FLOPs rel"],
                         [[len(r["spec"]["groups"]), pc(r["accuracy"]), f"{sgn(r['gain'])} {ci(r['gain_ci'], True)}", f"{r['flops_relative']:.3f}"]
                          for r in p["rows"]])
    return out


def repeat_ratio(rs):
    """Max per-position norm ratio of a repeat row vs the identity plan.

    There is no identity row; the repeat-of-block-21 row's norms[0..21] are the identity norms
    (positions before the repeat are untouched), so ratios cover logical blocks <= 21 only.
    """
    ref = next((r for r in rs["rows"] if rs["meta"][r["spec_hash"]] == {"kind": "repeat", "block": 21, "scale": 1.0}), None)
    return ref["norms"][:22] if ref and ref.get("norms") else None


def repswap():
    out = ["## (f) Repeat / swap (discovery, n=200): top 5 by accuracy", ""]
    for V in VARIANTS:
        rs = m5(f"repswap_{MODEL}_{V}")
        if not rs:
            continue
        ident = repeat_ratio(rs)
        base = rs["baseline"]["accuracy"]
        out += [f"### {V} (baseline {pc(base)}%)", ""]
        for kind in ("repeat", "swap"):
            rows_ = sorted((r for r in rs["rows"] if rs["meta"][r["spec_hash"]]["kind"] == kind), key=lambda r: -r["accuracy"])[:5]
            tab = []
            for r in rows_:
                m = rs["meta"][r["spec_hash"]]
                ratio = "-"
                if kind == "repeat" and ident and r.get("norms"):
                    b = m["block"]
                    rat = [(j, r["norms"][j] / ident[j if j <= b else j - 1]) for j in range(len(r["norms"])) if (j if j <= b else j - 1) <= 21]
                    j, v = max(rat, key=lambda t: t[1])
                    ratio = f"{v:.2f}x at pos {j}"
                tab.append([f"{kind} block {m['block']}" + (f" scale {m['scale']}" if "scale" in m else ""), pc(r["accuracy"]),
                            f"{sgn(r['gain'])} {ci(r['gain_ci'], True)}", f"{r['flops_relative']:.3f}", ratio])
            out += table([kind.capitalize(), "Acc %", "Gain pts [CI]", "FLOPs rel", "Max norm ratio vs identity"], tab)
    out += ["Norm ratio compares each position's last-token residual norm with the identity plan's (taken from the repeat-block-21 row's unchanged prefix), so it covers logical blocks up to 21 only.", ""]
    return out


def attrib():
    out = ["## (g) Attribution re-test (Spearman rho vs real mean ablation)", ""]
    rows = []
    for V in VARIANTS:
        a = m5(f"attrib_{MODEL}_{V}")
        if not a:
            continue
        for key, name in (("heads_excluding_layer0", "heads (excl. layer 0)"), ("kv_groups", "KV groups")):
            d = a[key]
            rows.append([V, name, d["log_prob"]["n"], f"{d['log_prob']['rho']:.2f}", f"{d['accuracy_lost']['rho']:.2f}",
                         f"{d['accuracy_lost']['p']:.3g}", "pass" if d["gate_passed"] else "fail"])
    return out + table(["Variant", "Set", "n", "rho vs log-prob drop", "rho vs accuracy lost", "p (acc)", f"Gate (rho >= {a['gate_rho']})"], rows)


def anatomy():
    out = ["## (h) Anatomy matrix (points vs baseline)", ""]
    a = m5(f"anatomy_{MODEL}")
    if not a:
        return out
    rows = [[f"{r['variant']}: {r['label']}", f"{r['delta_rule_pts']:+.1f}", f"{r['delta_sensible_pts']:+.1f}"]
            + [f"{r['delta_canary_pts'][c]:+.0f}" for c in CATS]
            + [f"{r['delta_canary_overall_pts']:+.1f}" if "delta_canary_overall_pts" in r else "-"] for r in a["rows"]]
    return out + table(["Row", "Δ rule", "Δ sensible"] + [f"Δ {c}" for c in CATS] + ["Δ canary overall"], rows)


def main():
    out = ["# M5 report tables", ""]
    for f in (excision, pareto, tests, h3, kv, repswap, attrib, anatomy):
        out += f()
    md = "\n".join(out)
    print(md)
    (R / "report_tables.md").write_text(md + "\n")


if __name__ == "__main__":
    main()
