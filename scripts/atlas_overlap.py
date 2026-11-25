"""M9 atlas: the secondary task-overlap analysis (docs/ATLAS-overlap.md, plan fixed 2026-11-10).

  uv run python scripts/atlas_overlap.py [--atlas0 var/results/atlas] [--atlas1 var/results/atlas1]
                                         [--acts var/acts/atlas] [--unembed var/results/atlas1/unembed_options.npz]
                                         [--out var/results/atlas1]

Reads Phase 0 and Phase 1 files, writes overlap.json (all numbers) and overlap.md (a report section). numpy only: no
model is built. Without the unembed file the projected variant is skipped and the report says so. Exits 1 with a message
if a required Phase 1 input is missing.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from surgery.atlas import overlap as ov  # noqa: E402

TITLE = "How the tasks overlap (secondary, plan fixed 2026-11-10 in docs/ATLAS-overlap.md)"


class MissingInput(Exception):
    pass


def _json(path: Path):
    if not path.exists():
        raise MissingInput(f"missing {path}")
    return json.loads(path.read_text())


def read_runs(atlas1: Path) -> dict:
    """Every `runs.shard{I}of{N}.json` merged (the layout of atlas_cuts.read_merged, for any directory)."""
    files = sorted(atlas1.glob("runs.shard*of*.json"))
    if not files:
        raise MissingInput(f"no runs.shard*of*.json in {atlas1}")
    runs = {}
    for p in files:
        runs.update(json.loads(p.read_text()).get("runs", {}))
    return runs


def default_tasks():
    from surgery.atlas import spec
    tasks = {t.name: {"kind": t.kind, "n_options": len(t.options), "number": t.number} for t in (spec.TASKS[n] for n in spec.PHASE1_TASKS)}
    return tasks, list(spec.PAIRS)


def load_inputs(atlas0: Path, atlas1: Path, names: list[str], cuts) -> dict:
    """Everything the analysis needs; raises MissingInput naming the first thing absent."""
    order = _json(atlas0 / "order.json")["shared_order"]
    summary = _json(atlas1 / "summary.json")
    rivals = _json(atlas1 / "rivals.json")
    plan = json.loads((atlas1 / "plan.json").read_text()) if (atlas1 / "plan.json").exists() else {}
    rule = plan.get("lr_rule", "per_k")
    drops = {}
    for t in names:
        les = _json(atlas1 / f"secondary_lesion_{t}.json")
        # measure.py writes per-block rows; older plan text assumed a flat "drops" list
        drops[t] = les["drops"] if "drops" in les else [r["drop"] for r in sorted(les["rows"], key=lambda r: r["block"])]
        if rivals.get(t) is None or rivals[t].get("bi") is None or rivals[t].get("ce_rise") is None:
            raise MissingInput(f"rivals.json has no bi / ce_rise for {t}")
    runs = read_runs(atlas1)
    # the same lr choice and seed-mean table as atlas_cuts.build_summary
    coll = cuts.collect(runs, names, [0, ov.K_LOAD], rule)
    damage = {}
    for t in names:
        by_k = coll.get(t, {})
        if 0 not in by_k or ov.K_LOAD not in by_k:
            raise MissingInput(f"no healed runs at k=0 and k={ov.K_LOAD} for {t} (lr rule {rule})")
        acc = {k: float(np.mean([r["sel_accuracy"] for r in rows])) for k, rows in by_k.items()}
        damage[t] = acc[0] - acc[ov.K_LOAD]
    transfer = summary.get("transfer", {}).get(str(ov.K_LOAD))
    if not transfer:
        raise MissingInput(f"summary.json has no transfer at k={ov.K_LOAD}")
    kstar = {r["task"]: r["k_star"] for r in summary.get("rows", []) if r.get("rung") is None}
    return {"order": order, "drops": drops, "rivals": rivals, "damage": damage, "transfer": transfer,
            "kstar": kstar, "grid": summary.get("ks", []), "lr_rule": rule}


def analyse(inp: dict, tasks: dict, pairs: list) -> dict:
    """The four tests, Holm verdicts and the descriptive outputs that need only Phase 0/1 result files."""
    names = sorted(tasks, key=lambda t: tasks[t]["number"])
    idx = {t: i for i, t in enumerate(names)}
    n = len(names)
    drops, order = inp["drops"], inp["order"]
    vec = np.stack([ov.block_vector(drops[t]) for t in names])
    sets = [ov.fragile_set(drops[t]) for t in names]
    sim, jac = ov.similarity_matrix(vec), ov.jaccard_matrix(sets)
    kinds = [tasks[t]["kind"] for t in names]
    o1a = ov.kind_test(sim, [int(k == "execute") for k in kinds])
    app, exe = [idx[a] for a, _ in pairs], [idx[b] for _, b in pairs]
    o1b = ov.matching_test(sim, app, exe)
    load = {t: ov.lesion_load(drops[t], order) for t in names}
    o2 = ov.lesion_test(names, load, inp["damage"],
                        {"bi": {t: inp["rivals"][t]["bi"] for t in names},
                         "ce_rise": {t: inp["rivals"][t]["ce_rise"] for t in names}})
    tmat = ov.symmetrise(inp["transfer"], names)
    o3 = ov.mantel_test(sim, tmat)
    effects = {"O1a": o1a["effect"], "O1b": o1b["effect"], "O2": o2["lesion_load"]["rho"], "O3": o3["rho"]}
    pvals = {"O1a": o1a["p"], "O1b": o1b["p"], "O2": o2["lesion_load"]["p"], "O3": o3["p"]}
    iu = np.triu_indices(n, 1)
    same = np.array([kinds[i] == kinds[j] for i, j in zip(*iu)])
    jv = jac[iu]
    jac_summary = {"same_kind": ov._nanmean(jv[same]), "cross_kind": ov._nanmean(jv[~same]),
                   "matched": ov._nanmean([jac[idx[a], idx[b]] for a, b in pairs])}
    hubs = ov.hub_blocks(sets, math.ceil(n / 2))
    pos = {b: i for i, b in enumerate(order)}
    first = {t: {"first_fragile_hit": ov.first_fragile_hit(sets[idx[t]], order, inp["grid"]),
                 "k_star": inp["kstar"].get(t), "fragile": sorted(sets[idx[t]])} for t in names}
    return {"tasks": names, "kinds": kinds, "blocks": [1, 26], "lr_rule": inp["lr_rule"],
            "similarity": sim, "jaccard": jac, "jaccard_summary": jac_summary, "symmetrised_T": tmat,
            "O1a": o1a, "O1b": o1b, "O2": o2, "O3": o3, "verdicts": ov.verdicts(effects, pvals),
            "hubs": {"min_sets": math.ceil(n / 2), "blocks": {b: {"n_sets": c, "order_position": pos.get(b)}
                                                              for b, c in hubs.items()}},
            "first_hit": first, "rivals": {t: {"bi": inp["rivals"][t]["bi"], "ce_rise": inp["rivals"][t]["ce_rise"]} for t in names},
            "grid": inp["grid"], "order": order}


def run_subspace(names, acts: Path, unembed, log=print) -> tuple:
    stores, answers, ncls = {}, {}, {}
    for t in names:
        d = acts / f"{t}_discovery"
        if not (d / "resid.npy").exists():
            raise MissingInput(f"missing {d / 'resid.npy'}")
        stores[t] = np.load(d / "resid.npy", mmap_mode="r")
        meta = json.loads((d / "meta.json").read_text())
        answers[t] = meta["answers"]
        ncls[t] = max(meta["answers"]) + 1
    layers = ov.LAYERS if stores[names[0]].shape[1] >= len(ov.LAYERS) else tuple(range(stores[names[0]].shape[1]))

    def load_layer(t, l):
        if t == names[0] and l % 4 == 0:
            log(f"subspace: layer {l}", flush=True)
        return np.asarray(stores[t][:, l, :], dtype=np.float32)

    return ov.subspace_analysis(load_layer, names, answers, ncls, unembed, layers=layers), layers


def subspace_summary(res: dict, layers, names, pairs) -> dict:
    idx = {t: i for i, t in enumerate(names)}
    li = {l: i for i, l in enumerate(layers)}
    out = {"layers": list(layers), "pairs": [list(p) for p in pairs], "raw": {}, "excess_raw": {}, "excess_projected": {},
           "projected": {}, "null_raw": {}, "null_projected": {}, "maps": {}}
    variants = [("raw", "raw_null", "excess_raw", "null_raw"), ("proj", "proj_null", "excess_projected", "null_projected")]
    for key, nullk, exk, nk in variants:
        if key not in res:
            continue
        for a, b in pairs:
            name = f"{a}|{b}"
            raw = res[key][:, idx[a], idx[b]]
            nl = res[nullk][:, idx[a], idx[b]]
            out["raw" if key == "raw" else "projected"][name] = raw
            out[nk][name] = nl
            out[exk][name] = raw - nl
        out["maps"][exk] = {str(l): (res[key][li[l]] - res[nullk][li[l]]) for l in ov.MAP_LAYERS if l in li}
    return out


# ---------------------------------------------------------------- report

def _f(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def render(a: dict, sub: dict | None, sub_note: str | None) -> str:
    v = a["verdicts"]
    out = [f"## {TITLE}", "",
           "Secondary analysis. It changes no locked measure, hypothesis or bar. Four tests (O1a, O1b, O2, O3) form one "
           "family, Holm-corrected at 0.05, one-sided (positive). A test is supported only if its effect is positive "
           "and its Holm p is under 0.05. Blocks 1-26 are compared; fragile means drop of at least 0.05. "
           "Where the permutation space is at most 10,000 the p is exact, else 10,000 permutations, seed 9.", "",
           "### Tests", ""]
    desc = {"O1a": "same-kind minus cross-kind mean block similarity",
            "O1b": "matched pairs minus other cross-kind pairs among the 10 paired tasks",
            "O2": "Spearman rho, lesion load vs healed damage (k=0 minus k=12)",
            "O3": "Mantel rho, block similarity vs symmetrised T_12"}
    meth = {"O1a": a["O1a"]["method"], "O1b": a["O1b"]["method"], "O2": "permutation", "O3": "permutation"}
    out += _table(["test", "what", "effect", "p", "p (Holm)", "p method", "verdict"],
                  [[k, desc[k], _f(v[k]["effect"], 3, True), _f(v[k]["p"], 4), _f(v[k]["p_holm"], 4), meth[k],
                    v[k]["verdict"]] for k in ("O1a", "O1b", "O2", "O3")])
    o1a, o1b, o2, o3 = a["O1a"], a["O1b"], a["O2"], a["O3"]
    out += ["", f"O1a: same-kind mean {_f(o1a['same_mean'])} ({o1a['n_same']} pairs), cross-kind mean "
            f"{_f(o1a['cross_mean'])} ({o1a['n_cross']} pairs), {o1a['n_perm']} labellings.",
            f"O1b: matched mean {_f(o1b['matched_mean'])}, other cross-kind mean {_f(o1b['other_mean'])} "
            f"({o1b['n_other']} pairs), {o1b['n_perm']} matchings.",
            f"O3: {o3['n_pairs']} task pairs with a defined similarity and transfer.",
            f"Jaccard of fragile sets, beside block similarity: same-kind {_f(a['jaccard_summary']['same_kind'])}, "
            f"cross-kind {_f(a['jaccard_summary']['cross_kind'])}, matched pairs {_f(a['jaccard_summary']['matched'])}.",
            "", "### O2: lesion load against damage, with the rivals", ""]
    rows = [["lesion load", _f(o2["lesion_load"]["rho"], 3, True), _f(o2["lesion_load"]["p"], 4),
             f"[{_f(o2['lesion_load']['interval'][0], 2)}, {_f(o2['lesion_load']['interval'][1], 2)}]"],
            ["BI", _f(o2["bi"]["rho"], 3, True), "-", f"[{_f(o2['bi']['interval'][0], 2)}, {_f(o2['bi']['interval'][1], 2)}]"],
            ["CE rise", _f(o2["ce_rise"]["rho"], 3, True), "-",
             f"[{_f(o2['ce_rise']['interval'][0], 2)}, {_f(o2['ce_rise']['interval'][1], 2)}]"]]
    out += _table(["predictor", "rho vs damage", "p", "bootstrap 95% (2,000 draws)"], rows)
    out += ["", "Per task:", ""]
    out += _table(["task", "kind", "lesion load D", "damage", "BI", "CE rise"],
                  [[t, a["kinds"][i], _f(o2["load"][t]), _f(o2["damage"][t], 3, True),
                    _f(a["rivals"][t]["bi"], 4), _f(a["rivals"][t]["ce_rise"], 3)]
                   for i, t in enumerate(a["tasks"])])
    out += ["", "### Hub blocks", ""]
    h = a["hubs"]
    if h["blocks"]:
        out += [f"Blocks in at least {h['min_sets']} of {len(a['tasks'])} fragile sets, with position in the shared cut order "
                "(1 = cut first):", ""]
        out += _table(["block", "fragile sets", "cut-order position"],
                      [[b, d["n_sets"], "-" if d["order_position"] is None else d["order_position"] + 1]
                       for b, d in h["blocks"].items()])
    else:
        out.append(f"No block is in {h['min_sets']} or more fragile sets.")
    out += ["", "### First fragile hit against k*", "",
            "First grid k at which the order removes a block of the task's fragile set (- = never on the grid).", ""]
    out += _table(["task", "fragile blocks", "first hit k", "k*"],
                  [[t, " ".join(map(str, d["fragile"])) or "-", d["first_fragile_hit"] if d["first_fragile_hit"] is not None else "-",
                    d["k_star"] if d["k_star"] is not None else "-"] for t, d in a["first_hit"].items()])
    out += ["", "### Same directions (descriptive)", ""]
    if sub is None:
        out.append(f"Skipped: {sub_note}")
        return "\n".join(out) + "\n"
    out += ["Overlap of class-mean subspaces (class means minus the task grand mean, unmodified model, last prompt token), "
            "as excess over the mean of 100 within-task label shuffles (seed 9). Layer 0 is the embedding output. "
            "Projected = the span of both tasks' option-token unembedding rows removed first.", ""]
    if sub_note:
        out += [sub_note, ""]
    layers = sub["layers"]
    for title, key in (("Matched pairs, excess overlap, raw", "excess_raw"),
                       ("Matched pairs, excess overlap, projected", "excess_projected")):
        if not sub.get(key):
            continue
        names = list(sub[key])
        out += [f"#### {title}", ""]
        out += _table(["layer"] + [nm.replace("|", " / ") for nm in names],
                      [[l] + [_f(sub[key][nm][i], 3, True) for nm in names] for i, l in enumerate(layers)])
        out.append("")
    for key, title in (("excess_raw", "raw"), ("excess_projected", "projected")):
        for l, m in sub["maps"].get(key, {}).items():
            m = np.asarray(m, dtype=float)
            out += [f"#### Excess overlap map, {title}, layer {l}", ""]
            out += _table(["task"] + [str(i + 1) for i in range(len(a["tasks"]))],
                          [[f"{i + 1} {t}"] + ["" if i == j else _f(m[i, j], 2) for j in range(len(a["tasks"]))]
                           for i, t in enumerate(a["tasks"])])
            out.append("")
    return "\n".join(out) + "\n"


def run(args, tasks=None, pairs=None) -> int:
    import atlas_cuts as cuts
    from export_unembed import load_unembed
    if tasks is None:
        tasks, pairs = default_tasks()
    atlas0, atlas1, acts, out = Path(args.atlas0), Path(args.atlas1), Path(args.acts), Path(args.out)
    names = sorted(tasks, key=lambda t: tasks[t]["number"])
    try:
        inp = load_inputs(atlas0, atlas1, names, cuts)
        a = analyse(inp, tasks, pairs)
        unembed, sub_note = None, None
        upath = Path(args.unembed)
        if upath.exists():
            unembed = load_unembed(upath)[0]
            miss = [t for t in names if t not in unembed]
            if miss:
                raise MissingInput(f"{upath} has no rows for {miss}")
        else:
            sub_note = (f"The projected variant was skipped: {upath} is missing. "
                        "Only the raw overlap is reported.")
        res, layers = run_subspace(names, acts, unembed)
    except MissingInput as e:
        print(f"atlas_overlap: {e}", file=sys.stderr)
        return 1
    sub = subspace_summary(res, layers, names, pairs)
    doc = {"plan": "docs/ATLAS-overlap.md", "projected_available": unembed is not None, **a, "subspace": sub}
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlap.json").write_text(json.dumps(ov.jsonable(doc), indent=1, allow_nan=False) + "\n")
    md = render(a, sub, sub_note)
    (out / "overlap.md").write_text(md)
    print(md, flush=True)
    return 0


def main(argv=None, tasks=None, pairs=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas0", default="var/results/atlas")
    ap.add_argument("--atlas1", default="var/results/atlas1")
    ap.add_argument("--acts", default="var/acts/atlas")
    ap.add_argument("--unembed", default="var/results/atlas1/unembed_options.npz")
    ap.add_argument("--out", default="var/results/atlas1")
    return run(ap.parse_args(argv), tasks, pairs)


if __name__ == "__main__":
    sys.exit(main())
