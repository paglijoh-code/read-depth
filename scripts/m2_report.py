"""Tables and gates for reports/m2_baseline.md from var/results/baseline/*.json.

  uv run python scripts/m2_report.py            prints markdown tables and gate verdicts
  uv run python scripts/m2_report.py --json     also writes var/results/baseline/summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from surgery import stats  # noqa: E402

R = Path("var/results/baseline")
MODELS = ["Qwen3-0.6B", "Qwen3-1.7B"]
VARIANTS = ["rule", "sensible"]
SPLITS = ["discovery", "selection", "test"]


def load(name: str) -> dict | None:
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ci(m: dict, scale: float = 100) -> str:
    return f"{scale * m['point']:.1f} [{scale * m['lo']:.1f}, {scale * m['hi']:.1f}]"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    out, summary = [], {"gates": {}, "half_widths": {}}

    out += ["## Task accuracy, macro-F1, ECE (95% template-grouped bootstrap CI, 10,000 resamples)", "",
            "| Model | Variant | Split | n | Templates | Accuracy % | Macro-F1 % | ECE % | ms/example |",
            "|---|---|---|---|---|---|---|---|---|"]
    for M in MODELS:
        for V in VARIANTS:
            for S in SPLITS + ["s1-discovery"]:
                r = load(f"{M}_{V}_s1_discovery" if S == "s1-discovery" else f"{M}_{V}_s0_{S}")
                if not r:
                    out.append(f"| {M} | {V} | {S} | missing | | | | | |")
                    continue
                mt = r["metrics"]
                out.append(f"| {M} | {V} | {S} | {r['n']} | {r['n_templates']} | {ci(mt['accuracy'])} | "
                           f"{ci(mt['macro_f1'])} | {ci(mt['ece'])} | {r['ms_per_example']:.0f} |")

    out += ["", "## Gates", "", "| Gate | Model | Variant | Measured | Pass |", "|---|---|---|---|---|"]
    for M in MODELS:
        for V in VARIANTS:
            d0, d1 = load(f"{M}_{V}_s0_discovery"), load(f"{M}_{V}_s1_discovery")
            rep = load(f"{M}_{V}_s0_discovery_rep")
            g = summary["gates"].setdefault(f"{M}/{V}", {})
            if d0 and d1:
                a0, a1, hw = (d0["metrics"]["accuracy"]["point"], d1["metrics"]["accuracy"]["point"],
                              d0["metrics"]["accuracy"]["half_width"])
                ok = abs(a0 - a1) < hw
                g["seed_agreement"] = {"s0": a0, "s1": a1, "diff": abs(a0 - a1), "s0_half_width": hw, "pass": ok}
                out.append(f"| Seed agreement | {M} | {V} | |{100 * a0:.1f} − {100 * a1:.1f}| = {100 * abs(a0 - a1):.1f} "
                           f"vs half-width {100 * hw:.1f} | {'yes' if ok else 'NO'} |")
            if d0 and rep:
                same = d0["pred"] == rep["pred"]
                maxdiff = float(np.max(np.abs(np.array(d0["logits"]) - np.array(rep["logits"]))))
                g["determinism"] = {"identical_predictions": same, "max_abs_logit_diff": maxdiff, "pass": same}
                out.append(f"| Determinism | {M} | {V} | predictions identical: {same}; max |Δlogit| {maxdiff:.2g} "
                           f"| {'yes' if same else 'NO'} |")
            accs = {S: load(f"{M}_{V}_s0_{S}") for S in SPLITS}
            if all(accs.values()):
                vals = [accs[S]["metrics"]["accuracy"]["point"] for S in SPLITS]
                ok = all(0.5 <= v <= 0.95 for v in vals)
                g["worth_specializing"] = {"accuracies": vals, "pass": ok}
                out.append(f"| Worth specializing (50–95%) | {M} | {V} | "
                           f"{', '.join(f'{100 * v:.1f}' for v in vals)} | {'yes' if ok else 'NO'} |")

    out += ["", "## Test-split CI half-widths for threshold locking (points)", "",
            "| Variant | Model | Unpaired accuracy half-width | Paired half-width, 1.7B − 0.6B | Paired point |",
            "|---|---|---|---|---|"]
    for V in VARIANTS:
        t = {M: load(f"{M}_{V}_s0_test") for M in MODELS}
        if not all(t.values()):
            continue
        a_, b_ = t["Qwen3-0.6B"], t["Qwen3-1.7B"]
        assert a_["ids"] == b_["ids"]
        pr = stats.paired_grouped_bootstrap(stats.accuracy, a_["template_ids"], a_["y"], a_["pred"], b_["pred"])
        mc = stats.mcnemar(np.array(a_["pred"]) == np.array(a_["y"]), np.array(b_["pred"]) == np.array(b_["y"]))
        for M in MODELS:
            hw = t[M]["metrics"]["accuracy"]["half_width"]
            summary["half_widths"][f"{M}/{V}"] = {"unpaired": hw, "paired_model_pair": pr["half_width"]}
            out.append(f"| {V} | {M} | {100 * hw:.2f} | {100 * pr['half_width']:.2f} | "
                       f"{100 * pr['point']:+.1f} (McNemar p={mc['p_exact']:.2g}) |")
    if summary["half_widths"]:
        worst = max(max(v.values()) for v in summary["half_widths"].values())
        summary["lock_points"] = math.ceil(100 * worst)
        out.append(f"\nLargest half-width: {100 * worst:.2f} points → threshold floor {summary['lock_points']} points.")

    out += ["", "## Canaries (accuracy %, chance ≈ 33%) and perplexity", ""]
    cats = ["syntax", "factual", "reasoning", "instruction", "code"]
    out += ["| Model | " + " | ".join(cats) + " | Overall | Perplexity (tokens) | Peak RSS MB |",
            "|---" * (len(cats) + 4) + "|"]
    for M in MODELS:
        c = load(f"{M}_canaries")
        if c:
            pc = c["canaries"]["per_category"]
            out.append(f"| {M} | " + " | ".join(f"{100 * pc[k]:.0f}" for k in cats) +
                       f" | {100 * c['canaries']['accuracy']:.1f} | {c['perplexity']['ppl']:.2f} "
                       f"({c['perplexity']['n_tokens']}) | {c['memory']['max_rss_mb']:.0f} |")

    out += ["", "## Latency and memory per pass", "", "| Model | Pass | Seconds | ms/example | Peak RSS MB | MPS driver MB |",
            "|---|---|---|---|---|---|"]
    for M in MODELS:
        for V in VARIANTS:
            r = load(f"{M}_{V}_s0_discovery")
            if r:
                out.append(f"| {M} | {V} discovery | {r['seconds']:.0f} | {r['ms_per_example']:.1f} | "
                           f"{r['memory']['max_rss_mb']:.0f} | {r['memory'].get('mps_driver_mb', 0):.0f} |")
    print("\n".join(out))
    if a.json:
        (R / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
