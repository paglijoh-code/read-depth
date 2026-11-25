"""Tables for reports/m3_maps.md from var/results/m3/*.json.

  uv run python scripts/m3_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

R = Path("var/results/m3")
MODELS = ["Qwen3-0.6B", "Qwen3-1.7B"]
VARIANTS = ["rule", "sensible"]


def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pt(x):
    return x["point"] if isinstance(x, dict) else x


def main():
    out = ["## Probes, logit lens and residual norm by layer", ""]
    for M in MODELS:
        for V in VARIANTS:
            m = load(f"maps_{M}_{V}")
            if not m:
                continue
            L = m["layers"]
            best = max(L, key=lambda r: r["label_probe"]["metric"])
            first_lens = next((r["layer"] for r in L if r["lens_accuracy"] >= 0.5 * m["accuracy"] + 0.5 * (1 / 6) and r["layer"] > 0), None)
            mid = (len(L) - 1) // 2
            corr_mid = max(r["correct_probe"]["metric"] for r in L[: mid + 1])
            out += [f"### {M} · {V} (model accuracy {100 * m['accuracy']:.1f}%, n={m['n']}, probes on {m['probe_subset']['n']})", "",
                    f"Best label probe: layer {best['layer']}, {100 * best['label_probe']['metric']:.1f}% "
                    f"[{100 * best['label_probe']['lo']:.1f}, {100 * best['label_probe']['hi']:.1f}]. "
                    f"Lens first halfway from chance to final accuracy at layer {first_lens}. "
                    f"Best correctness AUROC at layer ≤ {mid}: {corr_mid:.3f}.", "",
                    "| Layer | Label probe % [CI] | Control % | Correctness AUROC | Lens acc % | Lens p(correct) | Norm mean (p5–p95) |",
                    "|---|---|---|---|---|---|---|"]
            step = 1 if len(L) <= 29 else 2
            for r in L[::step]:
                lp, cp = r["label_probe"], r["correct_probe"]
                out.append(f"| {r['layer']} | {100 * lp['metric']:.1f} [{100 * lp['lo']:.1f}, {100 * lp['hi']:.1f}] | "
                           f"{100 * lp['control']:.1f} | {cp['metric']:.3f} | {100 * r['lens_accuracy']:.1f} | "
                           f"{r['lens_p_correct']:.3f} | {r['norm_mean']:.1f} ({r['norm_p5']:.1f}–{r['norm_p95']:.1f}) |")
            out.append("")

    out += ["## Sublayer ablation, Qwen3-0.6B (500-example discovery subset; points lost, paired 95% CI)", ""]
    for V in VARIANTS:
        s = load(f"ablation_Qwen3-0.6B_{V}")
        if not s:
            continue
        base = pt(s["baseline"]["accuracy"])
        cells = {}
        for r in s["rows"]:
            sp = r["spec"]
            ci = r["accuracy_lost_ci"] if "accuracy_lost_ci" in r else {}
            cells[(sp["site"], sp["mode"], sp["layers"][0])] = (pt(r["accuracy_lost"]), ci)
        n_layers = 1 + max(k[2] for k in cells)
        out += [f"### {V} (baseline {100 * base:.1f}% on {len(s['baseline']['pred'])})", "",
                "| Layer | attn zero | attn mean | mlp zero | mlp mean |", "|---|---|---|---|---|"]
        for l in range(n_layers):
            row = []
            for site in ("attn", "mlp"):
                for mode in ("zero", "mean"):
                    v, ci = cells.get((site, mode, l), (None, {}))
                    if v is None:
                        row.append("—")
                    else:
                        sig = ci and (ci.get("lo", 0) > 0 or ci.get("hi", 0) < 0)
                        row.append(f"{100 * v:+.1f}{'*' if sig else ''}")
            out.append(f"| {l} | " + " | ".join(row) + " |")
        out += ["", "`*` = paired CI excludes 0. Positive = accuracy lost.", ""]

    out += ["## Head attribution vs real mean ablation (0.6B, 20 checked heads per variant)", "",
            "| Variant | Spearman ρ vs log-prob drop | ρ vs accuracy lost | Largest real loss (head) | Its attribution |",
            "|---|---|---|---|---|"]
    for V in VARIANTS:
        a, h = load(f"attribution_Qwen3-0.6B_{V}"), load(f"ablation_Qwen3-0.6B_{V}_heads")
        if not (a and h):
            continue
        H = np.array(a["heads"])
        xs, acc, lp = [], [], []
        for r in h["rows"]:
            l, hd = r["spec"]["heads"][0]
            xs.append(H[l, hd]); acc.append(pt(r["accuracy_lost"])); lp.append(pt(r["mean_correct_label_log_prob_drop"]))
        i = int(np.argmax(acc))
        l, hd = h["rows"][i]["spec"]["heads"][0]
        out.append(f"| {V} | {spearmanr(xs, lp).statistic:.2f} | {spearmanr(xs, acc).statistic:.2f} | "
                   f"{100 * acc[i]:+.1f} pts ({l}.{hd}) | {xs[i]:+.3f} |")
        kv = np.array(a["kv"])
        top = np.dstack(np.unravel_index(np.argsort(-kv, axis=None)[:5], kv.shape))[0]
        out.append(f"| | top KV groups by attribution: " + ", ".join(f"{a_}.{b_} ({kv[a_, b_]:+.2f})" for a_, b_ in top) + " | | | |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
