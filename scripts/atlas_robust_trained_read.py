"""Scores the trained robustness arm (post hoc; frozen rule in DECISIONS 2026-11-24) from var/results/atlas_gates/g2/robust."""
import collections
import json
from pathlib import Path

R = Path("var/results/atlas_gates/g2/robust")
G1 = ("Qwen2.5-1.5B", "SmolLM2-1.7B", "pythia-1.4b")
VAR = ("C=0.01", "C=0.1", "C=10", "meandiff")


def main():
    draw = json.loads((R / "draw.json").read_text())
    out, passed = {}, []
    for mid in draw["models"]:
        m, fam = mid.split("/")[-1], collections.defaultdict(list)
        for t in draw["tasks"]:
            r = json.loads((R / m / f"{t}.json").read_text())
            fam[r["family"]].append(r)
        res = {}
        for f, rs in fam.items():
            tb = [r for r in rs if r["onsets"].get("C=1") is not None]
            res[f] = "none" if not tb else ("ok" if all(r["robust"] for r in tb) else "FAIL")
        n = sum(v == "ok" for v in res.values())
        out[m] = (n, res)
        if n >= 5:
            passed.append(m)
        print(f"{m:18s} {n}/6 {'PASS' if n >= 5 else 'fail'}  not ok: { {f: v for f, v in res.items() if v != 'ok'} }")
    print("models passing:", len(passed), "of", len(out), "| overall (>= 12):", len(passed) >= 12,
          "| v13 three:", {m: out[m][0] for m in G1}, all(out[m][0] >= 5 for m in G1))
    nt, bad, cause = 0, [], collections.Counter()
    for m in out:
        for t in draw["tasks"]:
            r = json.loads((R / m / f"{t}.json").read_text())
            b = r["onsets"].get("C=1")
            if b is None:
                print("untestable", m, t, "pairs", r["n_pairs_used"], "learnable", r["learnable"], r["onsets"])
                continue
            nt += 1
            if not r["robust"]:
                bad.append((m, t))
                for v in VAR:
                    s = r["onset_shift_vs_C1"][v]
                    if s is None or s > 0.10:
                        cause[v] += 1
                print("not robust", m, t, {k: (None if v is None else round(v, 2)) for k, v in r["onsets"].items()})
    print("tasks testable", nt, "robust", nt - len(bad), "cause by variant", dict(cause))
    c_only = sum(1 for m in out for t in draw["tasks"] for r in [json.loads((R / m / f"{t}.json").read_text())]
                 if r["onsets"].get("C=1") is not None and all(r["onset_shift_vs_C1"][v] is not None and r["onset_shift_vs_C1"][v] <= .10 for v in VAR[:3]))
    print("robust to C alone", c_only)


main()
