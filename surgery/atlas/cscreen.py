"""C-screen (proposal 4.2 / 4.3 / 5): per (task, model) probe + twin measurements on the plain prompt, the per-task and
per-family verdicts over the two screening models, and the seeded stratified pick of main and reserve families.

The task package (`surgery.atlas.families`, or any module with the same interface) is passed in as `fam`.
"""
from __future__ import annotations

import time

import numpy as np

from surgery.atlas import p0_measures as M

CHANCE = 0.25
RIVAL_TOL = 0.10
EARLY_REL, EARLY_LAYERS = 0.10, 1.0


def early_read(probe, onset):
    """Item 11: probe valid and its onset at <= 10% depth or <= 1 block."""
    return bool(probe["valid"] and onset is not None and (onset["rel"] <= EARLY_REL or onset["layers"] <= EARLY_LAYERS))


def run_task(ad, name, fam, toks=None, n_perm=20, seed=0, probe_only=False):
    """One (task, model) on the plain prompt. Returns a JSON-able dict (no use/arrival curves).
    `probe_only` (v13 probe-validity screen): skip the twin states and twin curves (twin, swapped = None)."""
    t_all = time.time()
    task = fam.get_task(name)
    items = fam.generate(name, seed=0)
    n_gen = len(items)
    if toks:
        bad = set(fam.tokcheck.check_items(items, list(task.options), toks)["bad_ids"])
        items = [x for x in items if x["id"] not in bad]
    n_align = n_gen - len(items)
    kept, n_filter = ad.filter_items(items)
    rivals = fam.rivals(kept, name)
    opt_ids = fam.check_options(ad.tok, task)
    tr, te = fam.split_items(kept)
    tmpl = {t: i for i, t in enumerate(sorted({x["template"] for x in tr}))}
    enc = ad.encode
    y_tr, y_te = np.asarray([x["answer"] for x in tr]), np.asarray([x["answer"] for x in te])
    g_tr = [tmpl[x["template"]] for x in tr]

    t0 = time.time()
    S_tr, _ = ad.collect_ids(enc([x["prompt"] for x in tr]), opt_ids)
    S_te, L_te = ad.collect_ids(enc([x["prompt"] for x in te]), opt_ids)
    if not probe_only:
        S_an, _ = ad.collect_ids(enc([x["ans_twin"]["prompt"] for x in te]), opt_ids)
        S_sf, _ = ad.collect_ids(enc([x["surf_twin"]["prompt"] for x in te]), opt_ids)
    states_s = time.time() - t0

    B = ad.n_blocks
    t0 = time.time()
    pc = M.probe_curve(S_tr, y_tr, g_tr, S_te, y_te, n_perm=n_perm, seed=seed, bow_acc=rivals["bow"])
    pc.pop("fits")
    probe_s = time.time() - t0
    on_probe = M.onset(pc["selectivity"], B) if pc["valid"] else None

    t0 = time.time()
    td = sw = on_twin = None
    if not probe_only:
        td = M.twin_divergence(S_te, S_an, S_sf, seed=seed)
        sw = M.twin_divergence(S_te, S_sf, S_an, seed=seed)
        on_twin = M.onset(td["t"], B) if td["valid_sym"] else None
    twin_s = time.time() - t0

    return {
        "model": getattr(ad, "model_id", None), "task": name, "probe_only": probe_only, "family": task.family, "type_label": task.type_label,
        "options": list(task.options), "n_blocks": B,
        "n_generated": n_gen, "dropped_alignment": n_align, "dropped_filter": n_filter,
        "n_train": len(tr), "n_test": len(te), "test_item_acc": float(np.mean(L_te.argmax(1) == y_te)),
        "rivals": rivals, "probe": pc, "probe_onset": on_probe, "early_read": early_read(pc, on_probe),
        "twin": td, "twin_onset": on_twin, "swapped": sw,
        "answer_logfreq": ad.answer_logfreq(opt_ids),
        "timing": {"states_s": states_s, "probe_s": probe_s, "twin_s": twin_s, "total_s": time.time() - t_all},
    }


# ---------------------------------------------------------------- verdicts
def _ok(r):
    return isinstance(r, dict) and "error" not in r and "probe" in r


def _near_max(nm):
    if nm is None:
        return None
    return max(nm.values()) if isinstance(nm, dict) else float(nm)


def _rivals_ok(runs):
    """Rivals depend on the kept items only (identical up to the per-model filter): use the worst case over models."""
    vals, reasons = [], []
    for r in runs:
        riv = r["rivals"]
        for k, v in (("bow", riv.get("bow")), ("near_miss", _near_max(riv.get("near_miss"))), ("position", riv.get("position"))):
            if v is not None:
                vals.append((k, float(v)))
    bad = sorted({f"{k}={v:.2f}" for k, v in vals if abs(v - CHANCE) > RIVAL_TOL})
    return (not bad and bool(vals)), bad


def task_verdict(per_model):
    """per_model = {model: run_task dict or {"error":..}} for one task."""
    runs = [r for r in per_model.values() if _ok(r)]
    n_models = len(per_model)
    all_ok = len(runs) == n_models and n_models > 0
    out = {"errors": sorted(m for m, r in per_model.items() if not _ok(r))}
    out["rivals_ok"], out["rival_bad"] = _rivals_ok(runs) if runs else (False, ["no_runs"])
    out["aligned_ok"] = bool(all_ok and all(r["dropped_alignment"] == 0 and r["dropped_filter"] == 0 for r in runs))
    out["twin_peak_gt_dip"] = bool(all_ok and all(r["twin"]["peak"] > abs(r["twin"]["trough"]) for r in runs))
    out["swap_fails"] = bool(all_ok and all(not r["swapped"]["valid_sym"] for r in runs))
    out["twins_ok"] = bool(out["aligned_ok"] and out["twin_peak_gt_dip"] and out["swap_fails"])
    valid = [m for m, r in per_model.items() if _ok(r) and r["probe"]["valid"]]
    out["probe_valid_models"] = valid
    flagged = [m for m in valid if per_model[m].get("early_read")]
    out["early_read"] = bool(valid and len(flagged) * 2 >= len(valid))
    out["peak_selectivity"] = {m: (r["probe"]["peak_selectivity"] if _ok(r) else None) for m, r in per_model.items()}
    return out


def verdicts(results, fam):
    """results = {model: {task: run_task dict}}. -> {"tasks": {task: ..}, "families": {family: ..}}."""
    models = list(results)
    tasks = {}
    for fname, (_, tnames) in fam.FAMILIES.items():
        for t in tnames:
            tasks[t] = task_verdict({m: results[m].get(t, {"error": "missing"}) for m in models})
    fams = {}
    for fname, (label, tnames) in fam.FAMILIES.items():
        tv = [tasks[t] for t in tnames]
        sel = [max(0.0, v) if v is not None else 0.0 for t in tv for v in t["peak_selectivity"].values()]
        reasons = []
        for t, v in zip(tnames, tv):
            for k in ("rivals_ok", "aligned_ok", "twin_peak_gt_dip", "swap_fails"):
                if not v[k]:
                    reasons.append(f"{t}:{k}" + (f"({','.join(v['rival_bad'])})" if k == "rivals_ok" else ""))
            if v["errors"]:
                reasons.append(f"{t}:error[{','.join(v['errors'])}]")
        any_valid = any(v["probe_valid_models"] for v in tv)
        if not any_valid:
            reasons.append("probe_valid_nowhere")
        both_valid = all(v["probe_valid_models"] for v in tv)
        base = all(v["rivals_ok"] and v["twins_ok"] for v in tv)
        kept = bool(base and any_valid)
        fams[fname] = {
            "type_label": label, "tasks": list(tnames), "kept": kept, "kept_strict": bool(kept and both_valid),
            "signal": float(np.mean(sel)) if sel else 0.0,
            "early_tasks": sum(1 for v in tv if v["early_read"]),
            "reasons": [] if kept else reasons,
        }
    return {"tasks": tasks, "families": fams}


# ---------------------------------------------------------------- pick
def _round_robin(order, queues, fv, count, cap):
    """Take each type's best remaining fitting family in turn until `count` are taken. Mutates queues."""
    taken, early = [], 0
    while len(taken) < count:
        progress = False
        for ty in order:
            if len(taken) >= count:
                break
            for i, f in enumerate(queues[ty]):
                if early + fv[f]["early_tasks"] <= cap:
                    taken.append(f)
                    early += fv[f]["early_tasks"]
                    del queues[ty][i]
                    progress = True
                    break
        if not progress:
            break
    return taken, early


def pick(fam_verdicts, seed=0, n_main=30, n_reserve=6, early_cap=0.15, n_min_main=24):
    """Seeded stratified pick (4.2). Families ranked by signal within type (seeded tie-break), seeded type order,
    round-robin with an early-read cap. If fewer than n_main + n_reserve families are kept but at least
    n_min_main + n_reserve, main shrinks (down to n_min_main) so that the reserve is still full (OK_MIN)."""
    rng = np.random.default_rng(seed)
    kept = sorted(f for f, v in fam_verdicts.items() if v["kept"])
    labels = sorted({fam_verdicts[f]["type_label"] for f in kept})
    order = [labels[i] for i in rng.permutation(len(labels))]
    tie = {f: float(k) for f, k in zip(kept, rng.random(len(kept)))}
    queues = {ty: sorted((f for f in kept if fam_verdicts[f]["type_label"] == ty),
                         key=lambda f: (-fam_verdicts[f]["signal"], tie[f])) for ty in labels}
    target = min(n_main, max(n_min_main, len(kept) - n_reserve)) if len(kept) >= n_min_main + n_reserve else n_main
    main, e_main = _round_robin(order, queues, fam_verdicts, target, int(np.floor(early_cap * 2 * target)))
    reserve, e_res = _round_robin(order, queues, fam_verdicts, n_reserve, int(np.floor(early_cap * 2 * n_reserve)))
    if len(main) == n_main and len(reserve) == n_reserve:
        status = "OK"
    elif len(main) >= n_min_main and len(reserve) == n_reserve:
        status = "OK_MIN"
    else:
        status = "SHORT"
    per_type = lambda lst: {ty: sum(1 for f in lst if fam_verdicts[f]["type_label"] == ty) for ty in sorted({v["type_label"] for v in fam_verdicts.values()})}
    return {"main": main, "reserve": reserve, "status": status, "seed": seed, "n_kept": len(kept),
            "type_order": order, "early_read_tasks": {"main": e_main, "reserve": e_res},
            "per_type": {"main": per_type(main), "reserve": per_type(reserve)}}


# ---------------------------------------------------------------- v13 (proposal v13, section 5 and 4.2)
def _band_bad(riv):
    bad = []
    for k, v in (("bow", riv.get("bow")), ("near_miss", _near_max(riv.get("near_miss"))), ("position", riv.get("position"))):
        if v is not None and abs(float(v) - CHANCE) > RIVAL_TOL:
            bad.append(f"{k}={float(v):.2f}")
    return bad


def verdicts_v13(results, fam, probe_results=None, fresh_rivals=None, artefacts=None):
    """v13 kept rule. `results` = {small model: {task: run}} (rivals, twins, probe); `probe_results` = {model: {task:
    run}} for probe-validity-only models (SmolLM-1.7B); `fresh_rivals` = {task: {small model: {"rivals": ..}}} on a
    fresh seed; `artefacts` = {task: {model_short: {"artefact": bool}}}: a confirmed artefact makes that (task, model)
    probe-invalid. Kept: both tasks pass rivals (seed 0 and the fresh seed) and twins on the small models, and the probe
    is valid on at least one (task, screening model) counting the probe-only models. Early read and signal use every
    model whose probe is valid / was run."""
    base = verdicts(results, fam)
    probe_results = probe_results or {}
    artefacts = artefacts or {}
    fresh_rivals = fresh_rivals or {}

    def art(t, m):
        return bool(artefacts.get(t, {}).get(m.split("/")[-1], {}).get("artefact"))

    tasks = {}
    for t, v in base["tasks"].items():
        v = dict(v)
        per = {m: results[m].get(t, {"error": "missing"}) for m in results}
        per.update({m: probe_results[m].get(t, {"error": "missing"}) for m in probe_results})
        valid = [m for m, r in per.items() if _ok(r) and r["probe"]["valid"] and not art(t, m)]
        v["artefact_models"] = sorted(m for m in per if art(t, m))
        v["probe_valid_models"] = valid
        v["early_read"] = bool(valid and sum(1 for m in valid if per[m].get("early_read")) * 2 >= len(valid))
        v["peak_selectivity"] = {m: (r["probe"]["peak_selectivity"] if _ok(r) else None) for m, r in per.items()}
        fr = fresh_rivals.get(t)
        v["fresh_rival_bad"] = (sorted({b for x in fr.values() for b in _band_bad(x["rivals"])}) if fr else ["missing"])
        v["fresh_rivals_ok"] = not v["fresh_rival_bad"]
        v["errors_probe_only"] = sorted(m for m in probe_results if not _ok(per[m]))
        tasks[t] = v
    fams = {}
    for fname, (label, tnames) in fam.FAMILIES.items():
        tv = [tasks[t] for t in tnames]
        sel = [max(0.0, x) if x is not None else 0.0 for t in tv for x in t["peak_selectivity"].values()]
        reasons = list(base["families"][fname]["reasons"])
        reasons = [r for r in reasons if r != "probe_valid_nowhere"]
        for t, v in zip(tnames, tv):
            if not v["fresh_rivals_ok"]:
                reasons.append(f"{t}:fresh_rivals({','.join(v['fresh_rival_bad'])})")
            for m in v["artefact_models"]:
                reasons.append(f"{t}:artefact[{m}]")
        any_valid = any(v["probe_valid_models"] for v in tv)
        if not any_valid:
            reasons.append("probe_valid_nowhere")
        base_ok = all(v["rivals_ok"] and v["twins_ok"] and v["fresh_rivals_ok"] for v in tv)
        kept = bool(base_ok and any_valid)
        fams[fname] = {
            "type_label": label, "tasks": list(tnames), "kept": kept,
            "kept_strict": bool(kept and all(v["probe_valid_models"] for v in tv)),
            "kept_v12": base["families"][fname]["kept"],
            "valid_only_on_probe_models": bool(kept and not any(
                m in results for v in tv for m in v["probe_valid_models"])),
            "signal": float(np.mean(sel)) if sel else 0.0,
            "early_tasks": sum(1 for v in tv if v["early_read"]),
            "reasons": [] if kept else reasons,
        }
    return {"tasks": tasks, "families": fams}


def pick_v13(fam_verdicts, seed=0, n_main=30, n_reserve=6, early_cap=0.15, n_min_main=24):
    """v13 pick: as `pick`, but the early-read cap counts tasks over main and reserve together
    (floor(0.15 x 2 x (main + reserve)) = 10 of 72); a family that would break the cap is skipped (the pick moves on)
    and listed in `skipped_early`."""
    rng = np.random.default_rng(seed)
    kept = sorted(f for f, v in fam_verdicts.items() if v["kept"])
    labels = sorted({fam_verdicts[f]["type_label"] for f in kept})
    order = [labels[i] for i in rng.permutation(len(labels))]
    tie = {f: float(k) for f, k in zip(kept, rng.random(len(kept)))}
    queues = {ty: sorted((f for f in kept if fam_verdicts[f]["type_label"] == ty),
                         key=lambda f: (-fam_verdicts[f]["signal"], tie[f])) for ty in labels}
    cap = int(np.floor(early_cap * 2 * (n_main + n_reserve)))
    target = min(n_main, max(n_min_main, len(kept) - n_reserve)) if len(kept) >= n_min_main + n_reserve else n_main
    early = [0]

    def rr(count):
        taken = []
        while len(taken) < count:
            progress = False
            for ty in order:
                if len(taken) >= count:
                    break
                for i, f in enumerate(queues[ty]):
                    if early[0] + fam_verdicts[f]["early_tasks"] <= cap:
                        taken.append(f)
                        early[0] += fam_verdicts[f]["early_tasks"]
                        del queues[ty][i]
                        progress = True
                        break
            if not progress:
                break
        return taken

    main = rr(target)
    e_main = early[0]
    reserve = rr(n_reserve)
    skipped = sorted(f for q in queues.values() for f in q if fam_verdicts[f]["early_tasks"] > cap - early[0])
    if len(main) == n_main and len(reserve) == n_reserve:
        status = "OK"
    elif len(main) >= n_min_main and len(reserve) == n_reserve:
        status = "OK_MIN"
    else:
        status = "SHORT"
    tys = sorted({v["type_label"] for v in fam_verdicts.values()})
    per_type = lambda lst: {ty: sum(1 for f in lst if fam_verdicts[f]["type_label"] == ty) for ty in tys}  # noqa: E731
    return {"main": main, "reserve": reserve, "status": status, "seed": seed, "n_kept": len(kept),
            "type_order": order, "early_cap_tasks": cap,
            "early_read_tasks": {"main": e_main, "reserve": early[0] - e_main, "total": early[0]},
            "skipped_early": skipped, "unpicked_kept": sorted(f for q in queues.values() for f in q),
            "per_type": {"main": per_type(main), "reserve": per_type(reserve)}}
