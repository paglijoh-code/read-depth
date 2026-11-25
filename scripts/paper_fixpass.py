"""Zero-cost analyses and audits for the round-2 paper reviews (reports/paper_review2_astra.md, _argon.md).

Saved results only: no training, no GPU, no network. Everything here is post hoc. Estimators are the published ones
(`escape_check.weighted_within` / `spearman`, which agree with `atlas2_cuts.weighted_rho` and `atlas_robust.spearman_plain`;
the agreement is asserted on the headline before anything else is computed).

    uv run python scripts/paper_fixpass.py           # writes var/results/paper_fixpass/*.json and reports/paper_fixpass.md

Pure helpers (no numpy, no torch) are at the top and are unit-tested in tests/test_paper_fixpass.py; the loaders import the
project's analysis scripts lazily.
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "var/results"
OUT = RES / "paper_fixpass"
REPORT = ROOT / "reports/paper_fixpass.md"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from escape_check import avg_ranks, spearman, weighted_within  # noqa: E402

N_BOOT = 10_000
SEED = 0
MODELS = ("Qwen3-0.6B", "Qwen3-1.7B", "SmolLM2-1.7B")
# The seven families of paper/sec_appendix.tex (Appendix A), fixed here before any result: the code's FAMILY_OF has one
# family per generator, which makes most leave-one-family-out runs leave-one-task-out.
PAPER_FAMILY = {
    "route_judgement": "routing", "route_work": "routing",
    "code_output": "code", "code_literal": "code", "code_shape": "code", "code_uses_first": "code", "code_traces_first": "code",
    "fact_check": "facts", "fact_topic": "facts",
    "rule": "workflow rules", "sensible": "workflow rules", "hard": "workflow rules", "hard_hops": "workflow rules",
    "hard_target_read": "workflow rules", "target_read_bare": "workflow rules",
    "arith": "arithmetic", "arith_id": "arithmetic", "arith_shape": "arithmetic", "arith_order": "arithmetic",
    "arith_long": "arithmetic",
    "syntax_valid": "syntax", "syntax_depth": "syntax", "syntax_first_close": "syntax", "syntax_outer_pair": "syntax",
    "bind_compare": "binding",
}


# ------------------------------------------------------------------ pure helpers

def estimators(kinds, x, y):
    """(weighted within-kind rho, pooled rho) of predictor x against y, as the paper's headline (average ranks)."""
    w, _, p = weighted_within(list(zip(kinds, x, y)))
    return w, p


def leave_one_family_out(rows, family_of):
    """rows: dicts with task, kind, depth, y. One entry per family present: the estimators on the rows without it."""
    out = []
    for fam in sorted({family_of[r["task"]] for r in rows}):
        keep = [r for r in rows if family_of[r["task"]] != fam]
        w, p = estimators([r["kind"] for r in keep], [r["depth"] for r in keep], [r["y"] for r in keep])
        out.append({"left_out": fam, "tasks": sorted(r["task"] for r in rows if family_of[r["task"]] == fam),
                    "n": len(keep), "weighted": w, "pooled": p})
    return out


def binary_depth(depth):
    """The layer-1 predictor: 0 for L_on = 1, 1 for L_on > 1."""
    return [0 if d <= 1 else 1 for d in depth]


def _abs(v):
    return None if v is None else abs(v)


def paired_bootstrap_abs_diff(rows, n_boot=N_BOOT, seed=SEED):
    """Task-resampling bootstrap (one resample of tasks, both predictors scored on it) of |rho(binary)| - |rho(continuous)|
    against y, for the weighted and the pooled estimator. Resamples where either rho is undefined are skipped and counted."""
    rng = random.Random(seed)
    n = len(rows)
    out = {}
    obs = {}
    for name, idx in (("weighted", 0), ("pooled", 1)):
        kinds = [r["kind"] for r in rows]
        y = [r["y"] for r in rows]
        c = estimators(kinds, [r["depth"] for r in rows], y)[idx]
        b = estimators(kinds, binary_depth([r["depth"] for r in rows]), y)[idx]
        obs[name] = {"rho_continuous": c, "rho_binary": b, "abs_diff": None if c is None or b is None else abs(b) - abs(c)}
    diffs = {"weighted": [], "pooled": []}
    skipped = {"weighted": 0, "pooled": 0}
    for _ in range(n_boot):
        s = [rows[rng.randrange(n)] for _ in range(n)]
        kinds, d, y = [r["kind"] for r in s], [r["depth"] for r in s], [r["y"] for r in s]
        cw, cp = estimators(kinds, d, y)
        bw, bp = estimators(kinds, binary_depth(d), y)
        for name, c, b in (("weighted", cw, bw), ("pooled", cp, bp)):
            if c is None or b is None:
                skipped[name] += 1
            else:
                diffs[name].append(abs(b) - abs(c))
    for name in ("weighted", "pooled"):
        v = sorted(diffs[name])
        out[name] = dict(obs[name], n_boot=n_boot, n_defined=len(v), n_skipped=skipped[name],
                         lo=_pct(v, 2.5), hi=_pct(v, 97.5), share_binary_larger=(sum(1 for x in v if x > 0) / len(v)) if v else None)
    return out


def _pct(sorted_vals, q):
    """Linear-interpolation percentile (numpy's default) of a sorted list."""
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def multiset_permutations(values):
    """Every distinct arrangement of `values` (a list with repeats), each once."""
    counts = Counter(values)
    keys = sorted(counts)
    n = len(values)

    def rec(prefix):
        if len(prefix) == n:
            yield tuple(prefix)
            return
        for k in keys:
            if counts[k]:
                counts[k] -= 1
                prefix.append(k)
                yield from rec(prefix)
                prefix.pop()
                counts[k] += 1
    yield from rec([])


def n_distinct_permutations(values):
    out = math.factorial(len(values))
    for c in Counter(values).values():
        out //= math.factorial(c)
    return out


def exact_permutation_p(x, y, cap=5_000_000):
    """Exact permutation test of Spearman rho(x, y), shuffling x over positions. Every distinct arrangement of x is equally
    likely under a uniform shuffle, so enumerating the distinct arrangements gives the exact p. Returns None if more than
    `cap` arrangements. One-sided: share with rho <= observed; two-sided: share with |rho| >= |observed|."""
    total = n_distinct_permutations(list(x))
    if total > cap:
        return None
    obs = spearman(list(x), list(y))
    one = two = 0
    best_pos = None
    for arr in multiset_permutations(list(x)):
        s = spearman(list(arr), list(y))
        if s is None:
            continue
        one += s <= obs + 1e-12
        two += abs(s) >= abs(obs) - 1e-12
        best_pos = s if best_pos is None else max(best_pos, s)
    return {"observed": obs, "n_arrangements": total, "hits_one_sided": one, "hits_two_sided": two,
            "p_one_sided": one / total, "p_two_sided": two / total, "max_rho_over_arrangements": best_pos}


def monte_carlo_p(x, y, n=N_BOOT, seed=SEED):
    """The paper's permutation p as t8_partials computes it (shuffle y, (hits + 1) / (n + 1)), for comparison; stdlib RNG here,
    so it differs from the paper's numpy stream in the last digits."""
    rng = random.Random(seed)
    obs = spearman(list(x), list(y))
    one = two = 0
    yy = list(y)
    for _ in range(n):
        rng.shuffle(yy)
        s = spearman(list(x), yy)
        one += s <= obs + 1e-12
        two += abs(s) >= abs(obs) - 1e-12
    return {"p_one_sided": (one + 1) / (n + 1), "p_two_sided": (two + 1) / (n + 1)}


def seed_survival(acc_seed, a0, chance):
    """R of one seed's accuracy against a fixed uncut reference, clipped to 0..1 (`atlas3_extra.r12`)."""
    if a0 - chance <= 1e-9:
        return None
    return min(1.0, max(0.0, (acc_seed - chance) / (a0 - chance)))


def n_pre(depth, cut):
    return sum(1 for b in cut if b < depth)


# ------------------------------------------------------------------ loaders (project scripts imported lazily)

def _imports():
    _argv, sys.argv = sys.argv, [sys.argv[0]]  # atlas3_extra reads positional argv at import
    try:
        import atlas2_cuts as B
        import atlas_e3_scale as S
        import atlas_robust as E
    finally:
        sys.argv = _argv
    return B, S, E


def load_profiles():
    """{model: rows} with the keys of atlas_robust's profile rows (task, kind, L_on_healed, r12, chance, healed_acc, pchance, acc)."""
    B, S, E = _imports()
    return {"Qwen3-0.6B": E.profile_06()["rows"], "Qwen3-1.7B": E.profile_17()["rows"],
            "SmolLM2-1.7B": E.profile_extra(S.D3)["rows"]}


def flat(rows):
    return [{"task": r["task"], "kind": r["kind"], "depth": r["L_on_healed"], "y": r["r12"]} for r in rows]


def jload(rel):
    return json.loads((RES / rel).read_text())


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=str) + "\n")


# ------------------------------------------------------------------ the nine items

def item0_headline(prof):
    """Reproduce the published headline (and check the two estimator implementations agree) before anything else."""
    B, S, E = _imports()
    published = {"Qwen3-0.6B": (-0.658, -0.686), "Qwen3-1.7B": (-0.721, -0.711), "SmolLM2-1.7B": (-0.744, -0.770)}
    out = {}
    for m in MODELS:
        rows = flat(prof[m])
        w, p = estimators([r["kind"] for r in rows], [r["depth"] for r in rows], [r["y"] for r in rows])
        bw = B.weighted_rho([r["depth"] for r in rows], [r["y"] for r in rows], [r["kind"] for r in rows])[0]
        bp = E.spearman_plain([r["depth"] for r in rows], [r["y"] for r in rows])
        assert abs(w - bw) < 1e-9 and abs(p - bp) < 1e-9, (m, w, bw, p, bp)
        assert round(w, 3) == published[m][0] and round(p, 3) == published[m][1], (m, w, p, published[m])
        out[m] = {"n": len(rows), "weighted": w, "pooled": p, "published_weighted": published[m][0],
                  "published_pooled": published[m][1], "agrees_with_atlas2_cuts_and_atlas_robust": True}
    return out


def item1_provenance():
    """`rule` at Qwen3-0.6B, k = 12: which runs make 0.785 (section 9) and which make 0.28 (Table 8)."""
    p1 = jload("atlas1/runs.shard0of1.json")["runs"]
    plan = jload("atlas1/plan.json")
    sweep = {k: v for k, v in p1.items() if v["task"] == "rule" and v["k"] in (0, 12)}
    whole = {}
    for f in sorted(glob.glob(str(RES / "atlas2b/whole.shard*of*.json"))):
        whole.update(json.loads(Path(f).read_text())["runs"])
    whole = {k: v for k, v in whole.items() if v["task"] == "rule" and v["k"] in (0, 12)}
    summ = jload("atlas2b/summary.json")
    srow = next(r for r in summ["rows"] if r["task"] == "rule")
    sweep_rows = [{"key": k, "lr": v["lr"], "seed": v["seed"], "k": v["k"], "n_scored": v["n"], "sel_accuracy_500": v["sel_accuracy"],
                   "sel_accuracy_first200": v["sel_accuracy200"], "best_lrsel": v["best_lrsel"], "steps": v["steps"],
                   "ts": v["ts"], "ckpt": v["ckpt"]} for k, v in sorted(sweep.items())]
    whole_rows = [{"key": k, "lr": v["lr"], "seed": v["seed"], "k": v["k"], "n_scored": v["n"], "exact_match": v["exact_match"],
                   "first_token_accuracy": v["first_token_accuracy"], "best_lrsel_exact": v["best_lrsel"],
                   "ts": v["ts"], "ckpt": v["ckpt"]} for k, v in sorted(whole.items())]
    k12 = [r for r in sweep_rows if r["k"] == 12]
    # winner lr per the Phase 1 rule: seed 0's two runs at (task, k), higher best_lrsel, tie to 3e-4
    s0 = [r for r in k12 if r["seed"] == 0]
    win_lr = max(s0, key=lambda r: (r["best_lrsel"], r["lr"] == 3e-4))["lr"]
    win = sorted((r for r in k12 if math.isclose(r["lr"], win_lr)), key=lambda r: r["seed"])
    mean = lambda v: sum(v) / len(v)
    sd = lambda v: (sum((a - mean(v)) ** 2 for a in v) / (len(v) - 1)) ** 0.5
    w12 = sorted((r for r in whole_rows if r["k"] == 12), key=lambda r: r["seed"])
    w0 = sorted((r for r in whole_rows if r["k"] == 0), key=lambda r: r["seed"])
    s0k0 = [r for r in sweep_rows if r["k"] == 0 and r["seed"] == 0]
    win0 = max(s0k0, key=lambda r: (r["best_lrsel"], r["lr"] == 3e-4))["lr"]
    k0win = sorted((r for r in sweep_rows if r["k"] == 0 and math.isclose(r["lr"], win0)), key=lambda r: r["seed"])
    out = {
        "phase1_plan_lr_rule": plan.get("lr_rule"),
        "first_token_sweep_k12": {
            "source": "var/results/atlas1/runs.shard0of1.json (Phase 1 heals, trained by heal.train_heal: cross-entropy over the task's option logits, i.e. first-token loss; scored by option-logit argmax)",
            "winner_lr": win_lr, "rows_all_lrs": k12,
            "seed_values_500": [r["sel_accuracy_500"] for r in win], "mean_500": mean([r["sel_accuracy_500"] for r in win]),
            "sd_500": sd([r["sel_accuracy_500"] for r in win]),
            "seed_values_first200": [r["sel_accuracy_first200"] for r in win], "mean_first200": mean([r["sel_accuracy_first200"] for r in win]),
            "summary_acc12_mean": srow["acc"]["12"]["mean"], "summary_acc12_min_max": [srow["acc"]["12"]["min"], srow["acc"]["12"]["max"]],
            "summary_chance": srow["chance"], "summary_healed_k0": srow["healed_k0"], "summary_r12": srow["r12"],
            "k0_winner_lr": win0, "k0_seed_values_500": [r["sel_accuracy_500"] for r in k0win],
        },
        "whole_answer_heals_k12": {
            "source": "var/results/atlas2b/whole.shard{0,1}of2.json (stage B H2-C, scripts/atlas2_cuts.py run_whole: heal.train_whole = token cross-entropy on the whole answer + end of turn; scored by greedy exact match on selection[:200] = ac.TRANSFER_N)",
            "rows": w12, "seed_values_exact": [r["exact_match"] for r in w12], "mean_exact": mean([r["exact_match"] for r in w12]),
            "sd_exact": sd([r["exact_match"] for r in w12]), "lrs": sorted({r["lr"] for r in w12}),
            "first_token_equals_exact": all(r["exact_match"] == r["first_token_accuracy"] for r in w12),
            "k0_rows": w0, "k0_mean_exact": mean([r["exact_match"] for r in w0]),
        },
        "n_whole_answer_scored_on_500": "none: the whole-answer heals of stage B were scored on selection[:200] only (n = 200 in every row)",
    }
    a = out["first_token_sweep_k12"]["mean_500"]
    out["checks"] = {
        "0.785_matches_paper_section_9": abs(out["whole_answer_heals_k12"]["mean_exact"] - 0.785) < 5e-4,
        "0.287_matches_paper_section_9": abs(out["first_token_sweep_k12"]["mean_first200"] - 0.287) < 5e-4,
        "0.28_pm_0.15_matches_table_8": round(a, 2) == 0.28 and round(out["first_token_sweep_k12"]["sd_500"], 2) == 0.15,
        "lower_bound_if_same_checkpoints": 0.785 * 200 / 500,
        "table_8_R12": srow["r12"],
        "R12_if_whole_answer_0.785_were_used": seed_survival(0.785, srow["healed_k0"], srow["chance"]),
    }
    return out


def item1b_whole_sensitivity(prof):
    """Supplementary (not asked for; post hoc): the 0.6B headline with whole-answer R in place of first-token R for the doing tasks
    that have whole-answer heals (stage B H2-C: k = 0 and 12, seeds 0-2, lr 3e-4, scored on selection[:200]); depth and every
    other task unchanged. Chance is the task's selection-split chance, as for R; the 200 vs 500 split mix is a caveat."""
    whole = {}
    for f in sorted(glob.glob(str(RES / "atlas2b/whole.shard*of*.json"))):
        whole.update(json.loads(Path(f).read_text())["runs"])
    rows, swapped = [], []
    for r in prof["Qwen3-0.6B"]:
        t = r["task"]
        y = r["r12"]
        w12 = [v["exact_match"] for v in whole.values() if v["task"] == t and v["k"] == 12]
        w0 = [v["exact_match"] for v in whole.values() if v["task"] == t and v["k"] == 0]
        yw = None
        if len(w12) == 3 and len(w0) == 3:
            yw = seed_survival(sum(w12) / 3, sum(w0) / 3, r["chance"])
        swapped.append({"task": t, "kind": r["kind"], "L_on_healed": r["L_on_healed"], "R12_first_token_500": y,
                        "R12_whole_answer_200": yw, "em12": None if yw is None else sum(w12) / 3,
                        "em0": None if yw is None else sum(w0) / 3, "chance": r["chance"]})
        rows.append({"task": t, "kind": r["kind"], "depth": r["L_on_healed"], "y": y if yw is None else yw})
    w, p = estimators([x["kind"] for x in rows], [x["depth"] for x in rows], [x["y"] for x in rows])
    base = flat(prof["Qwen3-0.6B"])
    bw, bp = estimators([x["kind"] for x in base], [x["depth"] for x in base], [x["y"] for x in base])
    return {"n_tasks": len(rows), "n_swapped": sum(1 for x in swapped if x["R12_whole_answer_200"] is not None),
            "published": {"weighted": bw, "pooled": bp}, "with_whole_answer_R": {"weighted": w, "pooled": p}, "tasks": swapped}


def item2_chance(prof):
    """Chance and the L_on threshold inputs, per task, at the three models."""
    from surgery.atlas import spec
    rows = {}
    for m in MODELS:
        out = []
        for r in prof[m]:
            a0 = (r.get("acc") or {}).get("0", {}).get("mean") if r.get("acc") else None
            if a0 is None:
                a0 = r.get("healed_k0")
            c_p, ft = r["pchance"], r["healed_acc"]
            out.append({"task": r["task"], "kind": r["kind"], "chance_R_selection": r["chance"], "chance_probe_discovery": c_p,
                        "a0_healed_k0_selection": a0, "first_token_discovery_acc": ft,
                        "threshold": c_p + 0.5 * (ft - c_p), "L_on_healed": r["L_on_healed"],
                        "ft_minus_a0": None if a0 is None else ft - a0})
        rows[m] = out
    return {"spec_chance_doc": spec.chance.__doc__, "tasks": rows}


def item3_lofo(prof):
    from surgery.atlas import tasks as T
    out = {"family_of_source": "surgery/atlas/tasks.py FAMILY_OF (one family per generator)",
           "paper_families_source": "paper/sec_appendix.tex Appendix A, seven families (mapping PAPER_FAMILY in scripts/paper_fixpass.py)"}
    for m in MODELS:
        rows = flat(prof[m])
        full = estimators([r["kind"] for r in rows], [r["depth"] for r in rows], [r["y"] for r in rows])
        entry = {"n": len(rows), "full_weighted": full[0], "full_pooled": full[1]}
        for label, fam in (("FAMILY_OF", T.FAMILY_OF), ("paper_seven", PAPER_FAMILY)):
            res = leave_one_family_out(rows, fam)
            ws = [x["weighted"] for x in res if x["weighted"] is not None]
            ps = [x["pooled"] for x in res if x["pooled"] is not None]
            entry[label] = {"n_families": len(res), "results": res, "weighted_min": min(ws), "weighted_max": max(ws),
                            "pooled_min": min(ps), "pooled_max": max(ps)}
        out[m] = entry
    return out


def item4_binary(prof):
    out = {}
    for m in MODELS:
        rows = flat(prof[m])
        b = paired_bootstrap_abs_diff(rows)
        gt1 = [r for r in rows if r["depth"] > 1]
        w, p = estimators([r["kind"] for r in gt1], [r["depth"] for r in gt1], [r["y"] for r in gt1])
        out[m] = {"n": len(rows), "n_L_on_gt_1": len(gt1), "n_L_on_eq_1": len(rows) - len(gt1),
                  "binary_vs_continuous": b, "continuous_L_on_gt_1": {"weighted": w, "pooled": p, "n": len(gt1)},
                  "kinds_in_gt_1": dict(Counter(r["kind"] for r in gt1)),
                  "binary_by_kind_note": "binary predictor inside each kind: 0 for L_on = 1, 1 for L_on > 1"}
    return out


def _runs_for(model):
    B, S, E = _imports()
    ac = B.ac
    if model == "Qwen3-1.7B":
        runs = json.loads((RES / "atlas3/c1/runs.shard0of1.json").read_text())["runs"]
        rule = "per_k"
        return lambda tasks: ac.collect(runs, tasks, [0, 12], rule)
    old = B.merged_dir(RES / "atlas1", "runs", "", "runs")
    new = {}
    for f in sorted(glob.glob(str(RES / "atlas2b/runs.shard*of*.json"))):
        new.update(json.loads(Path(f).read_text()).get("runs", {}))
    rule = json.loads((RES / "atlas1/plan.json").read_text()).get("lr_rule", "per_k")

    def coll(tasks):
        o = {}
        o.update(ac.collect(old, [t for t in tasks if t not in B.NEW6], [0, 12], rule))
        o.update(ac.collect(new, [t for t in tasks if t in B.NEW6], [0, 12], "per_k"))
        return o
    return coll


def item5_per_seed(prof):
    """Association per cut-heal seed. Depth fixed at the published L_on_healed. Primary R_s = (a12_s - c) / (a0 - c) with the
    published seed-mean a0 (only the cut-heal seed varies); secondary R_s uses that seed's own k = 0 accuracy as well.
    Both declared before the numbers were seen."""
    out = {}
    for m in ("Qwen3-1.7B", "Qwen3-0.6B"):
        rows = prof[m]
        coll = _runs_for(m)([r["task"] for r in rows])
        per = {s: [] for s in (0, 1, 2)}
        check = []
        for r in rows:
            t, c = r["task"], r["chance"]
            a0 = r["acc"]["0"]["mean"]
            k12 = {x["seed"]: x["sel_accuracy"] for x in coll[t][12]}
            k0 = {x["seed"]: x["sel_accuracy"] for x in coll[t][0]}
            assert abs(sum(k12.values()) / 3 - r["acc"]["12"]["mean"]) < 1e-9, (m, t)
            assert abs(sum(k0.values()) / 3 - a0) < 1e-9, (m, t)
            check.append(t)
            for s in (0, 1, 2):
                per[s].append({"task": t, "kind": r["kind"], "depth": r["L_on_healed"], "chance": c, "a0_mean": a0, "a0_seed": k0[s],
                               "a12_seed": k12[s], "y": seed_survival(k12[s], a0, c), "y_own_k0": seed_survival(k12[s], k0[s], c)})
        res = {}
        for s in (0, 1, 2):
            ent = {"n": len(per[s])}
            for key, lab in (("y", "primary_fixed_a0"), ("y_own_k0", "secondary_own_k0")):
                use = [x for x in per[s] if x[key] is not None]
                w, p = estimators([x["kind"] for x in use], [x["depth"] for x in use], [x[key] for x in use])
                ent[lab] = {"weighted": w, "pooled": p, "n": len(use)}
            ent["rows"] = per[s]
            res[str(s)] = ent
        out[m] = {"seeds": res, "tasks_checked_against_summary_mean": len(check),
                  "lr_rule_note": "winner lr per task and k (seed 0 picks it); seeds 1 and 2 ran at the winner only"}
    return out


def item6_fig6():
    s = jload("atlas3/e2chk/summary.json")["sizes"]["1.7B"]["pairs"]
    post = jload("atlas3/e2chk_mac/1.7B/summary_post_hoc.json")
    rows = []
    for p in s:
        chk = post.get(p["check"])
        rows.append({"do": p["do"], "n_common_ids": p["n_common"], "depth_do": p["depth_do"],
                     "R_do_matched_seed0_test": p["R_do"], "R_do_selection_3seed": p["R_do_selection"],
                     "acc_do_k0_k12_matched": [p["acc_do"]["0"], p["acc_do"]["12"]], "chance_do_matched": p["chance_do"],
                     "check": p["check"], "R_check_matched_seed0_test": p["R_check"],
                     "depth_check_post_hoc": None if chk is None else chk["L_on_healed"],
                     "depth_check_boot": None if chk is None else [chk["boot"]["lo"], chk["boot"]["hi"]],
                     "acc_check_k0_k12": [p["acc_check"]["0"], p["acc_check"]["12"]], "chance_check": p["chance_check"],
                     "bin": p["bin"], "R_bin_matched_seed0_test": p["R_bin"]})
    return {"source_values": "var/results/atlas3/e2chk/summary.json sizes.1.7B.pairs[*] (R_do, R_check, depth_do, acc_*, chance_*; the same rows are in stage1.json) "
                             "and var/results/atlas3/e2chk_mac/1.7B/summary_post_hoc.json (check depth); paper/tab_e2.tex is generated from these by paper/figs/t9_e2.py",
            "rows": rows}


def item7_table11():
    e1 = jload("atlas3/e1.json")
    small = {t["task"]: t for t in e1["0.6B"]["tasks"]}
    big = {t["task"]: t for t in e1["1.7B"]["tasks"]}
    rows = []
    for t, b in big.items():
        s = small.get(t)
        rows.append({"task": t, "L_on_0.6B": None if s is None else s["L_on_healed"], "L_on_1.7B": b["L_on_healed"],
                     "differs": None if s is None else s["L_on_healed"] != b["L_on_healed"]})
    return {"source": "var/results/atlas3/e1.json 0.6B.tasks[*].L_on_healed and 1.7B.tasks[*].L_on_healed (paper/figs/t4_cuts.py prints the 1.7B value only)",
            "rows": rows, "n_differ": sum(1 for r in rows if r["differs"]), "n_both": sum(1 for r in rows if r["differs"] is not None)}


def item8_smol_npre0(prof):
    B, S, E = _imports()
    cut = S.D3  # noqa: F841 (spec string; the cut itself comes from d3/summary.json)
    blocks = sorted(jload("atlas3/d3/summary.json")["cut"])
    rows = flat(prof["SmolLM2-1.7B"])
    grp = [dict(r, n_pre=n_pre(r["depth"], blocks)) for r in rows]
    g = [r for r in grp if r["n_pre"] == 0]
    d, y = [r["depth"] for r in g], [r["y"] for r in g]
    ex = exact_permutation_p(d, y)
    mc = monte_carlo_p(d, y)
    kinds = Counter(r["kind"] for r in g)
    by_kind = {}
    for k in kinds:
        sub = [r for r in g if r["kind"] == k]
        by_kind[k] = {"n": len(sub), "depths": sorted(r["depth"] for r in sub), "within_kind_rho": spearman([r["depth"] for r in sub], [r["y"] for r in sub])}
    ties = Counter(round(v, 12) for v in y)
    sorted_y = sorted(y)
    return {"cut_blocks": blocks, "group": [{k: r[k] for k in ("task", "kind", "depth", "y")} for r in sorted(g, key=lambda r: r["y"])],
            "n": len(g), "n_depth_1": sum(1 for x in d if x == 1), "exact": ex, "monte_carlo_stdlib_seed0": mc,
            "published_monte_carlo": {"p_two_sided": 0.0818, "p_one_sided": 0.0818, "source": "paper/tab_partials.tex via paper/figs/t8_partials.py (numpy default_rng(0), 10,000 shuffles of R)"},
            "by_kind": by_kind, "R_values_tied_at_top": ties.get(round(sorted_y[-1], 12)), "R_sorted": sorted_y,
            "estimator_note": "t8_partials.within_group uses the pooled Spearman over the group (S.heal.spearman), not the weighted within-kind estimator"}


def item9_deep_17(prof):
    from surgery.atlas import tasks as T  # noqa: F401
    B, S, E = _imports()
    rows = prof["Qwen3-1.7B"]
    blocks = json.loads((RES / "atlas3/1.7B/order.json").read_text())["shared_order"][:12]
    e4 = jload("atlas3/e4/summary.json")
    e4_set = set(e4["arms"]["late17"]["qualifying"])  # the 22 tasks with n_pre = 0 under both late cuts
    out = []
    for r in sorted(rows, key=lambda r: -r["L_on_healed"]):
        L = r["L_on_healed"]
        out.append({"task": r["task"], "kind": r["kind"], "L_on_healed": L, "ge_10": L >= 10, "gt_10": L > 10,
                    "n_pre_shared_k12": n_pre(L, blocks), "in_n_pre0_group_shared_k12": n_pre(L, blocks) == 0,
                    "in_E4_22_task_set": r["task"] in e4_set,
                    "E4_late17_R": e4["arms"]["late17"]["tasks"].get(r["task"], {}).get("r"),
                    "E4_late15_R": e4["arms"]["late15"]["tasks"].get(r["task"], {}).get("r"),
                    "R12_shared": r["r12"], "L_on_unmodified": r.get("L_on")})
    deep = [o for o in out if o["ge_10"]]
    return {"shared_cut_k12": sorted(blocks), "E4_task_set_size": len(e4_set), "rows_ge_10": deep,
            "n_ge_10": len(deep), "n_gt_10": sum(1 for o in deep if o["gt_10"]),
            "tasks_ge_10_in_E4_set": [o["task"] for o in deep if o["in_E4_22_task_set"]],
            "tasks_ge_10_not_in_E4_set": [o["task"] for o in deep if not o["in_E4_22_task_set"]],
            "tasks_gt_10_in_E4_set": [o["task"] for o in deep if o["gt_10"] and o["in_E4_22_task_set"]],
            "n_in_n_pre0_group_shared_k12_ge_10": sum(1 for o in deep if o["in_n_pre0_group_shared_k12"]),
            "n_pre0_group_size_shared_k12": sum(1 for o in out if o["in_n_pre0_group_shared_k12"])}


def _f(v, n=3):
    return "n/a" if v is None else f"{v:+.{n}f}"


def _g(v, n=3):
    return "n/a" if v is None else f"{v:.{n}f}"


def _iv(lo, hi):
    return f"[{_f(lo)}, {_f(hi)}]"


def render(res):
    L = []
    w = L.append
    h, pv, sens, ch, lo, bi, ps, f6, t11, sm, dp = (res[k] for k in (
        "headline", "provenance", "provenance_sensitivity", "chance", "lofo", "binary", "per_seed", "fig6", "table11",
        "smol_npre0", "deep_1p7B"))
    w("# Paper fix pass: zero-cost audits and analyses (round-2 reviews)\n")
    w("Written by `scripts/paper_fixpass.py` (run `uv run python scripts/paper_fixpass.py`); numbers in `var/results/paper_fixpass/*.json`. "
      "Saved results only: no training, no GPU, no network, no paper edits. **Everything below is post hoc.** Estimators are the published ones "
      "(`scripts/escape_check.py` `weighted_within` and `spearman`, average ranks; weighted = task-count-weighted mean of within-kind Spearman, "
      "pooled = Spearman over all tasks); choices that are mine are named where they occur and were fixed before the numbers were seen. "
      "Resampling uses Python's `random.Random(0)`, 10,000 draws, so intervals differ from numpy-based ones in the last digits.\n")
    w("## 0. Headline reproduced first\n")
    w("Rows are `atlas_robust.profile_06/profile_17/profile_extra(D3)` (`L_on_healed`, `r12`), the rows the paper's tests use. "
      "The two estimator implementations (`escape_check` and `atlas2_cuts.weighted_rho` / `atlas_robust.spearman_plain`) agree to 1e-9 on every model, and both match the published values at three decimals (asserted in the script).\n")
    w("| Model | n | weighted rho | pooled rho |\n|---|---|---|---|")
    for m in MODELS:
        w(f"| {m} | {h[m]['n']} | {_f(h[m]['weighted'])} | {_f(h[m]['pooled'])} |")
    w("\nPublished: -0.658/-0.686, -0.721/-0.711, -0.744/-0.770 (`reports/claims.md`, `paper/tab_estimators.tex`). Reproduced.\n")

    # ---- 1
    ft, wh, chk = pv["first_token_sweep_k12"], pv["whole_answer_heals_k12"], pv["checks"]
    w("## 1. `rule` provenance (AI reviewer error 1)\n")
    w("**Answer: the two numbers come from different heals trained with different losses and scored differently, on different records. They do not contradict each other. "
      "Table 8's 0.28 +/- 0.15 is a first-token-trained, first-token-scored value; Section 9's 0.785 is a whole-answer-trained, exact-match value. "
      "But Equation 1 and Section 3.2 define a_k as exact-match accuracy of the whole generated answer, and Section 9 says later studies train and score the whole answer; that does not describe any Qwen3-0.6B cell of Table 8 nor the 0.6B headline, which come from first-token heals.**\n")
    w("| | Table 8 (`paper/tab_pertask_q06.tex`) | Section 9 (`paper/sec_nulls.tex`) |\n|---|---|---|")
    w(f"| number | a_12 = {_g(ft['mean_500'], 2)} +/- {_g(ft['sd_500'], 2)} (seeds {', '.join(_g(v, 3) for v in ft['seed_values_500'])}); R_12 = {_g(ft['summary_r12'], 2)} | exact match {_g(wh['mean_exact'], 3)} (seeds {', '.join(_g(v, 3) for v in wh['seed_values_exact'])}); first-token run beside it: {_g(ft['mean_first200'], 3)} (seeds {', '.join(_g(v, 3) for v in ft['seed_values_first200'])}) |")
    w("| runs | `var/results/atlas1/runs.shard0of1.json`, keys `rule\\|k12\\|s{0,1,2}\\|lr0.0003` (Phase 1 sweep; lr 3e-4 won at seed 0: lrsel 0.5 vs 0.18), via `var/results/atlas2b/summary.json` row `rule`, `acc.12` | `var/results/atlas2b/whole.shard{0,1}of2.json`, keys `whole\\|rule\\|k12\\|s{0,1,2}\\|lr0.0003` |")
    w("| loss | `heal.train_heal`: cross-entropy over the task's option logits (first token) | `heal.train_whole`: token cross-entropy on the whole answer plus end of turn |")
    w("| scoring | `heal.accuracy_and_preds`: argmax over option logits (first token) | `heal.exact_match_scores`: greedy generation, exact match (`first_token_accuracy` equals `exact_match` in all six rows) |")
    w(f"| records | whole selection split, n = 500 (`sel_accuracy`; `sel_accuracy200` = first 200 gives {_g(ft['mean_first200'], 3)}) | first 200 selection records (`ac.TRANSFER_N`, n = 200 in every row) |")
    w("| checkpoints | `var/ckpt/atlas1/rule_k12_s{0,1,2}_lr0.0003.pt` | `var/ckpt/atlas2b/whole_rule_k12_s{0,1,2}_lr0.0003.pt` |")
    w("| lr, seeds, steps | 3e-4 (winner of {1e-3, 3e-4}), seeds 0-2, 300 steps | 3e-4 only (no lr search), seeds 0-2, 300 steps |")
    w("| cut, chance, k=0 | shared order, first 12 blocks; chance 0.168 (selection); healed k=0 0.998 (`summary.json` `healed_k0`) | same cut; k=0 whole-answer exact match 0.997 |\n")
    w(f"AI reviewer's bound (0.785 on 200 implies at least {_g(chk['lower_bound_if_same_checkpoints'], 3)} on 500) holds only for the same checkpoints; these are separate heals, so there is no inconsistency. "
      f"No whole-answer score exists on all 500 records: the stage B whole heals were scored on 200 only (not found for 500; looked in `atlas2b/whole.*.json`, `atlas2b/summary.json`, `atlas2b/test_results.json`). "
      f"The 0.28 is close to 0.284 because it is that same first-token run (0.284 is the mean over the 500; the `claims.md` fact (j) 0.287 is the mean over the first 200). "
      f"Both statements in the paper are therefore right about what they say; Table 8's `rule` row is the first-token measurement and gives R_12 = {_g(chk['table_8_R12'], 2)}, while a whole-answer R would be {_g(chk['R12_if_whole_answer_0.785_were_used'], 2)} (0.785 against k=0 0.997, chance 0.168).\n")
    w("**Scope of the problem.** `var/results/atlas2b/summary.json` rows (the source of `tab_pertask_q06.tex`, via `paper/figs/_pertask.py` `rows_for('q06')`, and of the 0.6B survival R in `e1.json`) take `acc` from `sel_accuracy` of the Phase 1 / stage B sweep heals (`scripts/atlas2_cuts.py` `task_rows`, `scripts/atlas_cuts.py` `run_heal`), i.e. first-token-trained and first-token-scored for all 20 tasks. "
      "The 1.7B and SmolLM2 values come from whole-answer heals (`scripts/atlas3_cuts.py` `run_whole_heal`, `sel_accuracy` = exact match). So the paper's description of scoring holds for two of three models.\n")
    w("**Supplementary, post hoc (not asked for).** The 0.6B headline with whole-answer R in place of first-token R for the doing tasks that have whole-answer heals "
      f"({sens['n_swapped']} of {sens['n_tasks']} tasks; k=0 and k=12, three seeds, scored on 200 records; chance from the selection split; depth and the judging tasks unchanged): "
      f"weighted {_f(sens['with_whole_answer_R']['weighted'])}, pooled {_f(sens['with_whole_answer_R']['pooled'])} (published {_f(sens['published']['weighted'])} / {_f(sens['published']['pooled'])}). Association weaker, same sign.\n")
    w("| task | L_on | R_12 first-token (Table 8) | R_12 whole-answer (n=200) |\n|---|---|---|---|")
    for t in sens["tasks"]:
        if t["R12_whole_answer_200"] is not None:
            w(f"| {t['task']} | {t['L_on_healed']} | {_g(t['R12_first_token_500'], 3)} | {_g(t['R12_whole_answer_200'], 3)} |")
    w("\nSource fields: `provenance.json`, `provenance_sensitivity.json`; inputs `atlas1/runs.shard0of1.json` (`runs[*].sel_accuracy`, `sel_accuracy200`, `n`, `lr`, `seed`, `ckpt`), `atlas1/plan.json` (`lr_rule`), `atlas2b/whole.shard*of2.json` (`runs[*].exact_match`, `first_token_accuracy`, `n`), `atlas2b/summary.json` (`rows[rule].acc`, `chance`, `healed_k0`, `r12`).\n")

    # ---- 2
    w("## 2. What chance is, and which accuracy the L_on threshold uses (AI reviewer error 3, AI reviewer 6)\n")
    w("**Definition (code).** `surgery/atlas/spec.py` `chance(answers)` = frequency of the commonest answer option in the given list of records (majority-class rate), not 1/(number of options). It is evaluated on different splits for different quantities:\n")
    w("| quantity | split | code |\n|---|---|---|")
    w("| `chance` in R = (a_k - c)/(a_0 - c) and in Tables 8-10 | the task's 500 selection records (all 500, whole answer-option distribution) | 0.6B: `scripts/atlas_cuts.py` `score_selection` line 234 (`spec.chance(answers)`); 1.7B and SmolLM2: `scripts/atlas3_cuts.py` `run_whole_heal` line 245 (`spec.chance([r['answer'] for r in recs])`, recs = selection) |")
    w("| `chance` in the L_on threshold | the 1000 discovery records (the probe items) | `scripts/atlas3_cuts.py` `stage_healedprobe` line 910 (`spec.chance(answers)`, answers from `measure.build_store` over discovery), 0.6B: `scripts/atlas2_healed.py` line 78 |")
    w("| chance in E2 (Tables 12) | the test split, restricted to the ids common to the do, bin and check tasks (`chance_do`, `chance_check`, `chance_bin`) | `scripts/atlas3_cuts.py` lines 359 and 791 (`spec.chance` over test records), then common ids in `e2chk/summary.json` |")
    w("| chance of BoW (surface) | selection split | `scripts/atlas_baseline.py` line 60 |\n")
    w("**Threshold.** `L_on_healed` is the first layer whose grouped-CV probe accuracy reaches `chance_probe + 0.5 * (healed_acc - chance_probe)` (`scripts/atlas2_healed.py` `l_on_healed`, same in `atlas3_cuts.py`); `chance_probe` is the discovery-split chance above, and `healed_acc` is the trained (healed k=0, seed 0, winner lr, LoRA merged) model's **first-token option accuracy on the 1000 discovery records** (`np.mean(meta['pred'] == answers)` in `stage_healedprobe`). "
      "It is not the selection-split a_0 of Equation 1 (exact match for 1.7B and SmolLM2, first-token for 0.6B). A task is not given an L_on if healed_acc - chance < 0.05 (`metrics.MIN_GAP`).\n")
    for m in MODELS:
        rows = ch["tasks"][m]
        gaps = [abs(r["ft_minus_a0"]) for r in rows if r["ft_minus_a0"] is not None]
        w(f"### {m}: per task (n = {len(rows)}; tasks dropped from the analysis have no row). Largest |first-token discovery accuracy - a_0| = {_g(max(gaps), 3)}\n")
        w("| task | kind | chance R (selection) | chance probe (discovery) | a_0 (selection, summary) | first-token discovery acc (threshold input) | threshold | L_on |\n|---|---|---|---|---|---|---|---|")
        for r in rows:
            w(f"| {r['task']} | {r['kind']} | {_g(r['chance_R_selection'])} | {_g(r['chance_probe_discovery'])} | {_g(r['a0_healed_k0_selection'])} | {_g(r['first_token_discovery_acc'])} | {_g(r['threshold'])} | {r['L_on_healed']} |")
        w("")
    w("Source fields: `chance.json`; `atlas_robust.profile_*` row fields `chance` (R), `pchance` and `healed_acc` (from `healed_probe_<task>.json` / `healed_<task>.json` `chance`, `healed_accuracy`), `acc['0'].mean` or `healed_k0` (a_0). "
      "Files: 1.7B `var/results/atlas3/c1/healed_probe_*.json`, `c1/summary.json`; SmolLM2 `var/results/atlas3/d3/healed_probe_*.json`, `d3/summary.json`; 0.6B `var/results/atlas2/healed_*.json` and `atlas2b/healed_*.json`, `atlas2b/summary.json`.\n")

    # ---- 3
    w("## 3. Leave-one-generator-family-out (AI reviewer missing check 1, AI reviewer 1)\n")
    w("Two groupings, both fixed before computing: (A) `surgery/atlas/tasks.py` `FAMILY_OF` (one family per generator; 'hard', 'arith', 'syntax', 'code', 'fact' hold the matched judging/doing pairs, every other task is its own family, so most of those runs are leave-one-task-out); "
      "(B) the paper's seven families of Appendix A (`paper/sec_appendix.tex`; mapping `PAPER_FAMILY` in the script; a model with no task in a family has fewer; 0.6B has no binding task and SmolLM2 no routing task). Same task sets and estimators as the headline.\n")
    for m in MODELS:
        e = lo[m]
        w(f"### {m}: full weighted {_f(e['full_weighted'])}, pooled {_f(e['full_pooled'])} (n = {e['n']})\n")
        a, b = e["FAMILY_OF"], e["paper_seven"]
        w(f"(A) FAMILY_OF, {a['n_families']} families: weighted range [{_f(a['weighted_min'])}, {_f(a['weighted_max'])}], pooled range [{_f(a['pooled_min'])}, {_f(a['pooled_max'])}].  "
          f"(B) paper seven, {b['n_families']} families: weighted [{_f(b['weighted_min'])}, {_f(b['weighted_max'])}], pooled [{_f(b['pooled_min'])}, {_f(b['pooled_max'])}].\n")
        w("| (B) left out | tasks | n left | weighted | pooled |\n|---|---|---|---|---|")
        for r in b["results"]:
            w(f"| {r['left_out']} | {', '.join(r['tasks'])} | {r['n']} | {_f(r['weighted'])} | {_f(r['pooled'])} |")
        w("")
        w("(A) rows with the weakest weighted rho: " + "; ".join(f"{r['left_out']} {_f(r['weighted'])}" for r in sorted(a["results"], key=lambda r: -(r["weighted"] if r["weighted"] is not None else -9))[:3]) + ". Full list in `lofo.json`.\n")
    w("Reading: no family's removal changes the sign, and none leaves either estimator weaker than -0.497 (weakest: 0.6B without workflow rules, weighted). The association is not carried by one family. This addresses family dependence of the point estimate only; it is not a family-level permutation test.\n")
    w("Source: `lofo.json`; rows from `atlas_robust` profiles (`task`, `kind`, `L_on_healed`, `r12`).\n")

    # ---- 4
    w("## 4. Binary layer-1 predictor against continuous L_on (AI reviewer missing check 4)\n")
    w("Predictor b = 0 if L_on = 1 else 1. Estimators as the headline. Interval: bootstrap over tasks (resample tasks with replacement, both predictors scored on the same resample, 10,000 draws, seed 0, percentile 2.5/97.5) of the difference d = |rho(b, R)| - |rho(L_on, R)|; resamples where either rho is undefined are skipped and counted. Positive d means the binary predictor is the stronger one.\n")
    w("| Model | n | n at L_on = 1 | estimator | rho continuous | rho binary | d | 95% interval | share of resamples d > 0 | skipped |\n|---|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        e = bi[m]
        for k in ("weighted", "pooled"):
            v = e["binary_vs_continuous"][k]
            w(f"| {m} | {e['n']} | {e['n_L_on_eq_1']} | {k} | {_f(v['rho_continuous'])} | {_f(v['rho_binary'])} | {_f(v['abs_diff'])} | {_iv(v['lo'], v['hi'])} | {_g(v['share_binary_larger'], 3)} | {v['n_skipped']} |")
    w("\nrho restricted to L_on > 1 (continuous L_on; the same estimators as `paper/tab_estimators.tex`):\n")
    w("| Model | n | kinds | weighted | pooled |\n|---|---|---|---|---|")
    for m in MODELS:
        e = bi[m]
        c = e["continuous_L_on_gt_1"]
        w(f"| {m} | {e['n_L_on_gt_1']} | {e['kinds_in_gt_1']} | {_f(c['weighted'])} | {_f(c['pooled'])} |")
    w("\nReading: at both Qwen3 models the continuous depth has the larger point estimate, at SmolLM2 the binary one does; every interval for d includes 0 (at 1.7B, both estimators, and at 0.6B pooled the upper end is 0.006 to 0.022, so the continuous predictor is ahead by a point estimate of 0.1 to 0.15 with an interval that only just reaches 0). "
      "The data do not support a claim that continuous depth beats the layer-1 split, nor the reverse. Restricted to L_on > 1 the published values reproduce (-0.430/-0.677, -0.660/-0.705, -0.246/+0.043).\n")
    w("Source: `binary.json`.\n")

    # ---- 5
    w("## 5. Per-seed headline, depth fixed (AI reviewer missing check 3)\n")
    w("For each cut-heal seed s in {0, 1, 2}, R_s = clip((a_12,s - c) / (a_0 - c)) with a_12,s the seed's own accuracy at k = 12 on the 500 selection records (`sel_accuracy` of the winner-lr run), c the summary chance, and **a_0 the published seed-mean healed k=0 accuracy** (primary, fixed before computing; only the cut-heal seed varies). "
      "Secondary: a_0 replaced by seed s's own k=0 accuracy. Depth = the published `L_on_healed` (read from the seed-0 healed model). Winner lr per task from seed 0 as in the paper; the per-seed means reproduce the summary means (asserted).\n")
    w("| Model | seed | weighted (primary) | pooled (primary) | weighted (own k=0) | pooled (own k=0) | n |\n|---|---|---|---|---|---|---|")
    for m in ("Qwen3-1.7B", "Qwen3-0.6B"):
        for sd in ("0", "1", "2"):
            x = ps[m]["seeds"][sd]
            p1, p2 = x["primary_fixed_a0"], x["secondary_own_k0"]
            w(f"| {m} | {sd} | {_f(p1['weighted'])} | {_f(p1['pooled'])} | {_f(p2['weighted'])} | {_f(p2['pooled'])} | {p1['n']} |")
    w("\nReading: negative at every seed and both models; ranges -0.61 to -0.76 weighted at 1.7B (published -0.721) and -0.35 to -0.59 at 0.6B (published -0.658). The 0.6B values are first-token (see section 1). Seed 2 at 0.6B is the weakest (weighted -0.354). "
      "Per-task rows (`a12_seed`, `a0_seed`, `y`) are in `per_seed.json`.\n")
    w("Source: `per_seed.json`; `var/results/atlas3/c1/runs.shard0of1.json` (1.7B), `var/results/atlas1/runs.shard0of1.json` and `var/results/atlas2b/runs.shard*of*.json` (0.6B), `runs[*].sel_accuracy`, `seed`, `lr`, `k`.\n")

    # ---- 6
    w("## 6. Figure 6 matched values (AI reviewer error 5)\n")
    w("File: `var/results/atlas3/e2chk/summary.json`, `sizes['1.7B'].pairs[*]` (identical rows in `e2chk/stage1.json`, which `paper/figs/f6_kind.py` reads): `R_do` (seed 0, test split, common ids, n = `n_common`), `R_check`, `depth_do`, `acc_do`, `chance_do`; check read depth post hoc from `var/results/atlas3/e2chk_mac/1.7B/summary_post_hoc.json` (`L_on_healed`). `paper/tab_e2.tex` is generated from these. "
      "`R_do_selection` (3-seed mean on the selection split) is what the figure currently plots for the doing tasks.\n")
    w("| doing task | n common ids | depth (do) | R do matched (seed 0, test) | R do plotted now (3-seed, selection) | acc do k0 / k12 | chance do | check task | depth check (post hoc) | R check matched | acc check k0 / k12 |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for r in f6["rows"]:
        w(f"| {r['do']} | {r['n_common_ids']} | {r['depth_do']} | {_g(r['R_do_matched_seed0_test'], 3)} | {_g(r['R_do_selection_3seed'], 3)} | {_g(r['acc_do_k0_k12_matched'][0], 3)} / {_g(r['acc_do_k0_k12_matched'][1], 3)} | {_g(r['chance_do_matched'], 3)} | {r['check']} | {r['depth_check_post_hoc']} | {_g(r['R_check_matched_seed0_test'], 3)} | {_g(r['acc_check_k0_k12'][0], 3)} / {_g(r['acc_check_k0_k12'][1], 3)} |")
    w("\n`hard` has no check depth (`hard_check` was never learned). Source: `fig6.json`.\n")

    # ---- 7
    w("## 7. Table 11: L_on at 0.6B next to 1.7B (AI reviewer error 5)\n")
    w(f"Source: `var/results/atlas3/e1.json` `0.6B.tasks[*].L_on_healed` and `1.7B.tasks[*].L_on_healed`. `paper/tab_cuts.tex` (`paper/figs/t4_cuts.py`) prints the 1.7B value only. {t11['n_differ']} of the {t11['n_both']} tasks that exist at both models differ.\n")
    w("| task | L_on 0.6B | L_on 1.7B |\n|---|---|---|")
    for r in t11["rows"]:
        mark = " (differs)" if r["differs"] else ""
        w(f"| {r['task']} | {r['L_on_0.6B'] if r['L_on_0.6B'] is not None else '---'} | {r['L_on_1.7B']}{mark} |")
    w("\nSource: `table11.json`.\n")

    # ---- 8
    ex, mc = sm["exact"], sm["monte_carlo_stdlib_seed0"]
    w("## 8. SmolLM2 n_pre = 0 permutation p (AI reviewer error 4)\n")
    w(f"Group: tasks of the 22 whose first {len(sm['cut_blocks'])} cut blocks {sm['cut_blocks']} (`var/results/atlas3/d3/summary.json` `cut`) all sit at or above L_on_healed; n = {sm['n']}, {sm['n_depth_1']} at L_on = 1, one (`code_uses_first`, R = {_g(sm['group'][0]['y'], 3)}, the group's lowest R) at L_on = 2. "
      "Estimator of the paper's within-group test (`paper/figs/t8_partials.py` `within_group`): pooled Spearman over the group, permuting R over the tasks.\n")
    w(f"**Exact enumeration** (all {ex['n_arrangements']} distinct arrangements of the depth labels over the tasks, each equally likely under a shuffle): observed rho {_f(ex['observed'])}; one-sided p = {ex['hits_one_sided']}/{ex['n_arrangements']} = {_g(ex['p_one_sided'], 4)}; two-sided p = {ex['hits_two_sided']}/{ex['n_arrangements']} = {_g(ex['p_two_sided'], 4)}. "
      f"Monte Carlo with this script's RNG (10,000 shuffles, seed 0): {_g(mc['p_one_sided'], 4)} one- and two-sided. Published (numpy stream, same procedure): 0.0818 both.\n")
    w(f"**Why they match.** The observed rho ({_f(ex['observed'])}) is the minimum over all arrangements (the one depth-2 label sits on the lowest-R task), and the largest positive rho any arrangement can reach is {_f(ex['max_rho_over_arrangements'])}, below |{_f(ex['observed'])}|, because {sm['R_values_tied_at_top']} tasks tie exactly at R = 1.000 and three more are above 0.995 (so the best positive arrangement gives only the smaller extreme). So no shuffle counts toward the opposite tail, and the two-sided count equals the one-sided count. The AI reviewer's 1/13 = 0.0769 is the exact value; 0.0818 is Monte Carlo noise around it (817 hits in 10,000 against an expected 769, about 1.8 SD). "
      "`claims.md` and the paper table should give exact p = 0.0769 for both, with a note that the one-sided and two-sided counts coincide by construction.\n")
    ek = sm["by_kind"]
    w(f"AI reviewer's second point confirmed: in this group the doing side ({ek['execute']['n']} tasks: `sensible`, `syntax_valid`, `arith_id`) is all at L_on = 1, so the weighted within-kind estimator is undefined there (within-kind rho: judging {_f(ek['appraise']['within_kind_rho'])}, n = {ek['appraise']['n']}; doing n/a). The table's group rho {_f(ex['observed'])} is the pooled estimator, as `t8_partials.py` states in its docstring but the paper's table does not.\n")
    w("Source: `smol_npre0.json`.\n")

    # ---- 9
    w("## 9. Qwen3-1.7B tasks with L_on >= 10 and > 10 (AI reviewer error 2)\n")
    w(f"Source: `var/results/atlas3/c1/healed_probe_*.json` `L_on_healed` via the profile rows; n_pre from the first 12 shared-order blocks {dp['shared_cut_k12']}; E4 set = `var/results/atlas3/e4/summary.json` `arms.late17.qualifying` ({dp['E4_task_set_size']} tasks, excluded: `hard`, `code_output`, which have n_pre > 0 under the shared cut).\n")
    w("| task | kind | L_on | >= 10 | > 10 | n_pre (shared k=12) | in n_pre=0 group (18 tasks, shared cut) | in E4 22-task set | R12 shared | R E4 blocks 17-26 | R E4 blocks 15-24 |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for o in dp["rows_ge_10"]:
        w(f"| {o['task']} | {o['kind']} | {o['L_on_healed']} | {'yes' if o['ge_10'] else 'no'} | {'yes' if o['gt_10'] else 'no'} | {o['n_pre_shared_k12']} | {'yes' if o['in_n_pre0_group_shared_k12'] else 'no'} | {'yes' if o['in_E4_22_task_set'] else 'no'} | {_g(o['R12_shared'], 3)} | {_g(o['E4_late17_R'], 3)} | {_g(o['E4_late15_R'], 3)} |")
    w(f"\n**Counts.** L_on >= 10: {dp['n_ge_10']} tasks; L_on > 10: {dp['n_gt_10']} tasks (`hard_target_read` is exactly 10). Of the L_on >= 10 tasks, {len(dp['tasks_ge_10_in_E4_set'])} are in the E4 22-task set ({', '.join(dp['tasks_ge_10_in_E4_set'])}); `hard` and `code_output` are not. "
      f"Of the L_on > 10 tasks, {len(dp['tasks_gt_10_in_E4_set'])} are in E4's set. Only {dp['n_in_n_pre0_group_shared_k12_ge_10']} of the {dp['n_ge_10']} (`fact_check` 11, `hard_target_read` 10) are in the n_pre = 0 group of the shared k=12 cut. "
      "So AI reviewer is right on two points and the paper's E4 sentence is right on its own subset: (i) 'six tasks that read at layer 10 or later' is correct for the E4 22-task set (arith, bind_compare, rule, target_read_bare, fact_check, hard_target_read) and the text should say that set, since eight tasks read at 10 or later overall; "
      "(ii) the Figure 2 caption's 'after layer 10' for `hard_target_read` (L_on 10) is wrong: it reads at layer 10, not after it. The E4 R values quoted in `claims.md` (bind_compare 0.571 and 0.022, arith 0.13 and 0.05, the other four at 0.86 or above) reproduce from the E4 columns above.\n")
    w("Source: `deep_1p7B.json`.\n")
    return "\n".join(L) + "\n"


def main():
    prof = load_profiles()
    res = {
        "headline": item0_headline(prof),
        "provenance": item1_provenance(),
        "provenance_sensitivity": item1b_whole_sensitivity(prof),
        "chance": item2_chance(prof),
        "lofo": item3_lofo(prof),
        "binary": item4_binary(prof),
        "per_seed": item5_per_seed(prof),
        "fig6": item6_fig6(),
        "table11": item7_table11(),
        "smol_npre0": item8_smol_npre0(prof),
        "deep_1p7B": item9_deep_17(prof),
    }
    for k, v in res.items():
        dump(f"{k}.json", v)
    if "--json-only" not in sys.argv:
        REPORT.write_text(render(res))
    print(f"wrote {OUT.relative_to(ROOT)}/*.json and {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
