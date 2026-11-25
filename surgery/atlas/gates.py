"""Depth-atlas gates 1 and 2 (docs/ATLAS-gates-run.md): measurement and heal runners built on the P0 harness.

Library only (paths are arguments). Pieces: family-package task build for the plain and 2-shot prompts, fresh items,
the `lora` split, `measure_arrays` (analyze_task plus per-item arrays for the item-half and fresh-item rereads),
half-curve helpers (pure numpy), the robustness arm, a heal for every architecture (GPT-NeoX gets a fused-QKV LoRA on
the q and v rows), and the seeded draws.
"""
from __future__ import annotations

import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from surgery.atlas import families as fam
from surgery.atlas import p0_measures as M
from surgery.atlas.p0_harness import (Adapter, probe_subspaces, run_use_arrival, swap_subspace)  # noqa: F401

CHANCE = 0.25
N_LORA, N_TRAIN = 2300, 2000
RANK, SCALE, BATCH, STEPS = 10, 2.0, 8, 300
EX_OFFSET = 10 ** 6  # exemplar stream index offset for fresh items (never collides with a generated item index)
LINEAGES = ("Qwen2.5", "SmolLM2", "pythia")


def _crc(name: str) -> int:
    return zlib.crc32(name.encode())


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------- templates, items

def train_test_templates(name: str, seed: int = 0):
    """(train templates, test templates) exactly as `generate(name, seed)` picks them."""
    spec = fam.get_task(name).spec
    return spec.pick_templates(np.random.default_rng(seed * 1000003 + _crc(name)), 4)


def _balanced_aux(rng, cls):
    aux = np.zeros(len(cls), dtype=int)
    for c in range(4):  # aux balanced within each class, as core.generate
        idx = np.flatnonzero(cls == c)
        aux[idx] = rng.permutation(np.tile(np.arange(4), len(idx) // 4 + 1)[:len(idx)])
    return aux


def _item(spec, name, tag, i, tpl, c, s, split, rng):
    s_a, c_a = spec.ans_twin(rng, tpl, s, c, split)
    s_s = spec.surf_twin(rng, tpl, s, c, split)
    return {"id": f"{name}-{tag}-{i}", "prompt": spec.render(tpl, s), "answer": c, "template": tpl, "split": tag,
            "slots": s, "ans_twin": {"prompt": spec.render(tpl, s_a), "answer": c_a},
            "surf_twin": {"prompt": spec.render(tpl, s_s), "answer": c}}


def fresh_items(name: str, n: int = 200, seed: int = 0) -> list:
    """N2 bucket 2: `n` new test items. The same 4 test templates as `generate(name, 0)`, test-side words, own rng stream
    ([seed, crc, 5]), ids `<name>-fresh-<i>`, class-balanced. A prompt that already occurs in `generate(name, 0)` (or
    earlier in this list) is redrawn up to 30 times, so small pools still give mostly unseen prompts."""
    spec = fam.get_task(name).spec
    _, te_t = train_test_templates(name, 0)
    taken = {x["prompt"] for x in fam.generate(name, 0)}
    rng = np.random.default_rng([seed, _crc(name), 5])
    cls = np.tile(np.arange(4), n // 4 + 1)[:n]
    rng.shuffle(cls)
    aux = _balanced_aux(rng, cls)
    tp = np.tile(np.arange(len(te_t)), n // len(te_t) + 1)[:n]
    rng.shuffle(tp)
    items = []
    for i in range(n):
        tpl, c = te_t[int(tp[i])], int(cls[i])
        for _ in range(30):
            s = spec.sample(rng, tpl, c, "test", int(aux[i]))
            if spec.render(tpl, s) not in taken:
                break
        taken.add(spec.render(tpl, s))
        items.append(_item(spec, name, "fresh", i, tpl, c, s, "test", rng))
    return items


def lora_items(name: str, n: int = N_LORA, seed: int = 0) -> list:
    """Heal data for any family task (analogue of `p0_tasks.generate_lora`): the train templates of `generate(name, seed)`
    only, train-side words, own rng stream ([seed, crc, 77]), ids `<name>-lora-<i>`, split "lora"; class-balanced, `aux`
    balanced within class, shuffled. Duplicate prompts are possible."""
    spec = fam.get_task(name).spec
    tr_t, _ = train_test_templates(name, seed)
    rng = np.random.default_rng([seed, _crc(name), 77])
    cls = np.tile(np.arange(4), n // 4 + 1)[:n]
    rng.shuffle(cls)
    aux = _balanced_aux(rng, cls)
    tp = np.tile(np.arange(len(tr_t)), n // len(tr_t) + 1)[:n]
    rng.shuffle(tp)
    items = []
    for i in range(n):
        tpl, c = tr_t[int(tp[i])], int(cls[i])
        s = spec.sample(rng, tpl, c, "train", int(aux[i]))
        items.append({"id": f"{name}-lora-{i}", "prompt": spec.render(tpl, s), "answer": c, "template": tpl,
                      "split": "lora", "slots": s})
    return items


# ---------------------------------------------------------------- 2-shot prompts

def exemplar_classes(name: str, i: int, seed: int = 0) -> tuple:
    """Answer classes of the two exemplars of item index `i`: a seeded balanced cycle over the 4 classes (each block of 4
    consecutive indices is a fresh permutation; exemplar 1 and 2 take adjacent entries), independent of the item's answer."""
    perm = np.random.default_rng([seed, _crc(name), 3, i // 4]).permutation(4)
    j = i % 4
    return int(perm[j]), int(perm[(j + 1) % 4])


def exemplar_prefix(name: str, i: int, own_tpl: str, seed: int = 0) -> str:
    """`ex1_prompt + ex1_option + "\\n\\n" + ex2_prompt + ex2_option + "\\n\\n"`. Exemplars are fresh items from the task's
    TRAIN templates only (never the test templates), excluding `own_tpl`, train-side words, own rng stream
    ([seed, crc, 2, i]); the option text keeps its leading space."""
    task = fam.get_task(name)
    tr_t, _ = train_test_templates(name, 0)
    pool = [t for t in tr_t if t != own_tpl]
    rng = np.random.default_rng([seed, _crc(name), 2, i])
    parts = []
    for c in exemplar_classes(name, i, seed):
        tpl = pool[int(rng.integers(len(pool)))]
        s = task.spec.sample(rng, tpl, c, "train", int(rng.integers(4)))
        parts.append(f"{task.spec.render(tpl, s)}{task.options[c]}\n\n")
    return "".join(parts)


def _prefixed(x: dict, prefix: str) -> dict:
    y = dict(x)
    y["prompt"] = prefix + x["prompt"]
    y["ans_twin"] = {**x["ans_twin"], "prompt": prefix + x["ans_twin"]["prompt"]}
    y["surf_twin"] = {**x["surf_twin"], "prompt": prefix + x["surf_twin"]["prompt"]}
    y["prefix"] = prefix
    return y


def twoshot(items: list, name: str, seed: int = 0, offset: int = 0) -> list:
    """Item, answer twin and surface twin share one prefix (same exemplars), so twins stay position-matched."""
    return [_prefixed(x, exemplar_prefix(name, offset + i, x["template"], seed)) for i, x in enumerate(items)]


def build_task(ad, name: str, prompt: str = "plain", seed: int = 0, with_fresh: bool = False, n_fresh: int = 200) -> dict:
    """Everything `analyze_task` needs for one (task, adapter): as `cscreen.run_task`'s setup (generate(name, 0), token
    alignment filter, option check, template split, rivals), on the plain or the 2-shot prompt. `seed` only feeds the
    exemplar streams. `with_fresh` adds the fresh items (same prompt kind) with the bag-of-words rival on them."""
    assert prompt in ("plain", "twoshot"), prompt
    task = fam.get_task(name)
    items = fam.generate(name, 0)
    if prompt == "twoshot":
        items = twoshot(items, name, seed)
    kept, dropped = ad.filter_items(items)
    rivals = fam.rivals(kept, name)
    opt_ids = fam.check_options(ad.tok, task)
    tr, te = fam.split_items(kept)
    tmpl = {t: i for i, t in enumerate(sorted({x["template"] for x in tr}))}
    enc = ad.encode
    b = {"task": task, "name": name, "prompt": prompt, "opt_ids": opt_ids, "kept": kept, "dropped": dropped,
         "n_generated": len(items), "tr": tr, "te": te, "rivals": rivals,
         "ids_tr": enc([x["prompt"] for x in tr]), "ids_te": enc([x["prompt"] for x in te]),
         "ids_an": enc([x["ans_twin"]["prompt"] for x in te]), "ids_sf": enc([x["surf_twin"]["prompt"] for x in te]),
         "y_tr": [x["answer"] for x in tr], "g_tr": [tmpl[x["template"]] for x in tr],
         "y_te": [x["answer"] for x in te], "y_an": [x["ans_twin"]["answer"] for x in te]}
    if with_fresh:
        fr = fresh_items(name, n_fresh)
        if prompt == "twoshot":
            fr = twoshot(fr, name, seed, offset=EX_OFFSET)
        fr, fr_dropped = ad.filter_items(fr)
        b.update(fr=fr, fr_dropped=fr_dropped, ids_fr=enc([x["prompt"] for x in fr]), y_fr=[x["answer"] for x in fr],
                 bow_fr=fam.rival_bow(tr + [dict(x, split="test") for x in fr]))
    return b


# ---------------------------------------------------------------- measure (analyze_task + per-item arrays)

def _block_pred(Xtr, y_tr, groups_tr, Xte, y_te, perms, cv_folds, Xfr):
    """`p0_measures._one_block` (same fits) that also returns the test and fresh-item predictions of the true-label fit and
    of every permuted-label fit: (acc, fit, cv, null, extras)."""
    from sklearn.model_selection import GroupKFold
    sc, lr = M._fit(Xtr, y_tr)
    p_te = lr.predict(sc.transform(Xte))
    acc = float(np.mean(p_te == y_te))
    fit = (sc.scale_.copy(), lr.coef_.copy())
    p_fr = lr.predict(sc.transform(Xfr)) if Xfr is not None else None
    folds = []
    for a, v in GroupKFold(cv_folds).split(Xtr, y_tr, groups_tr):
        s2, l2 = M._fit(Xtr[a], y_tr[a])
        folds.append(M._acc(s2, l2, Xtr[v], y_tr[v]))
    null, pp_te, pp_fr = [], [], []
    for p in perms:
        s3, l3 = M._fit(Xtr, y_tr[p])
        q = l3.predict(s3.transform(Xte))
        null.append(float(np.mean(q == y_te)))
        pp_te.append(q.astype(np.int8))
        if Xfr is not None:
            pp_fr.append(l3.predict(s3.transform(Xfr)).astype(np.int8))
    ex = {"p_te": p_te.astype(np.int8), "pp_te": np.array(pp_te, dtype=np.int8).reshape(len(perms), len(y_te)),
          "p_fr": None if p_fr is None else p_fr.astype(np.int8),
          "pp_fr": None if Xfr is None else np.array(pp_fr, dtype=np.int8).reshape(len(perms), len(Xfr))}
    return acc, fit, float(np.mean(folds)), null, ex


def _probe_curve_preds(S_tr, y_tr, g_tr, S_te, y_te, S_fr, n_perm, seed, bow_acc):
    """`p0_measures.probe_curve` with identical numbers (its block runner is swapped for `_block_pred`, so the curve
    assembly is the original code), plus the per-item predictions. Respects ATLAS_PROBE_JOBS."""
    store = {}

    def run(S_tr_, y_tr_, groups, S_te_, y_te_, perms, cv_folds, nb):
        jobs = int(os.environ.get("ATLAS_PROBE_JOBS", "1"))
        args = [(S_tr_[:, b], y_tr_, groups, S_te_[:, b], y_te_, perms, cv_folds, None if S_fr is None else S_fr[:, b])
                for b in range(nb)]
        if jobs <= 1:
            res = [_block_pred(*a) for a in args]
        else:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=jobs)(delayed(_block_pred)(*a) for a in args)
        store["res"] = res
        return [r[:4] for r in res]

    orig = M._run_blocks
    M._run_blocks = run
    try:
        pc = M.probe_curve(S_tr, y_tr, g_tr, S_te, y_te, n_perm=n_perm, seed=seed, bow_acc=bow_acc)
    finally:
        M._run_blocks = orig
    ex = [r[4] for r in store["res"]]
    arr = {"pred_te": np.stack([e["p_te"] for e in ex]), "pred_perm": np.stack([e["pp_te"] for e in ex], 1)}
    if S_fr is not None:
        arr["pred_fr"] = np.stack([e["p_fr"] for e in ex])
        arr["pred_fr_perm"] = np.stack([e["pp_fr"] for e in ex], 1)
    return pc, arr


def _select_pairs(L_te, L_an, y_te, y_an, max_pairs, seed):
    sel = np.flatnonzero(M.pair_mask(L_te, L_an, y_te, y_an))
    n_qual = len(sel)
    if n_qual > max_pairs:
        sel = np.sort(np.random.default_rng(seed).choice(sel, max_pairs, replace=False))
    return sel, n_qual


def measure_arrays(ad, b, n_perm=20, max_pairs=200, seed=0, n_flip=10000, n_draw=1000, log=None):
    """`p0_harness.analyze_task` on a built task `b` (keys as `build_task`; optional `ids_fr`, `y_fr`, `bow_fr`), with
    the same numbers in the JSON fields (twin validity is `valid_sym`, the v13 rule), plus the per-item arrays.
    Returns (json dict, arrays dict)."""
    say = log or (lambda *_: None)
    B = ad.n_blocks
    opt_ids = b["opt_ids"]
    y_tr, y_te, y_an = np.asarray(b["y_tr"]), np.asarray(b["y_te"]), np.asarray(b["y_an"])
    tim = {}
    t0 = time.time()
    S_tr, _ = ad.collect_ids(b["ids_tr"], opt_ids)
    S_te, L_te = ad.collect_ids(b["ids_te"], opt_ids)
    S_an, L_an = ad.collect_ids(b["ids_an"], opt_ids)
    S_sf, _ = ad.collect_ids(b["ids_sf"], opt_ids)
    has_fr = b.get("ids_fr") is not None
    S_fr = ad.collect_ids(b["ids_fr"], opt_ids)[0] if has_fr else None
    tim["states_s"] = time.time() - t0
    out = {"n_train": len(b["ids_tr"]), "n_test": len(b["ids_te"]), "test_item_acc": float(np.mean(L_te.argmax(1) == y_te)),
           "n_blocks": B}

    t0 = time.time()
    bow = (b.get("rivals") or {}).get("bow")
    pc, arr = _probe_curve_preds(S_tr, y_tr, b["g_tr"], S_te, y_te, S_fr, n_perm, seed, bow)
    fits = pc.pop("fits")
    tim["probe_s"], tim["probe_fits"] = time.time() - t0, pc["n_fits"]
    say("probe done", tim["probe_s"])

    t0 = time.time()
    td = M.twin_divergence(S_te, S_an, S_sf, n_draw=n_draw, seed=seed)
    tim["twin_s"] = time.time() - t0
    mu = S_te.mean(0, keepdims=True)  # per-item cosine distances behind twin_divergence (same code path)
    d_ans, d_surf = M._cos_dist(S_te - mu, S_an - mu), M._cos_dist(S_te - mu, S_sf - mu)
    d_ans[np.isnan(d_ans)] = 0.0
    d_surf[np.isnan(d_surf)] = 0.0

    out["logit_lens_acc"] = M.logit_lens_acc(ad.lens_logits(S_te, opt_ids), y_te)
    sel, n_qual = _select_pairs(L_te, L_an, y_te, y_an, max_pairs, seed)
    out["n_pairs_qualifying"], out["n_pairs_used"] = int(n_qual), int(len(sel))

    t0 = time.time()
    use = ar = None
    P, K, R = len(sel), B - 1, 10
    pair = {k: np.zeros(s, dtype=np.float32) for k, s in
            (("d_item", (0,)), ("d_twin", (0,)), ("d_probe", (0, K)), ("d_rand", (0, R, K)), ("d_psurf", (0, K)),
             ("d_whole", (0, K)), ("d_wsurf", (0, K)))}
    if P > 0:
        Qp = probe_subspaces(fits)
        Qr = M.random_subspaces(ad.d, B, seed=seed)
        res = [run_use_arrival(ad, b["ids_te"][i], S_te[i], S_an[i], S_sf[i], Qp, Qr, opt_ids) for i in sel]
        yi, yt = y_te[sel], y_an[sel]
        st = lambda k: np.stack([r[k] for r in res])  # noqa: E731  [P, K(,R), 4]
        dd = lambda k: M.delta(st(k), yi[:, None], yt[:, None])  # noqa: E731
        pair = {"d_item": M.delta(L_te[sel], yi, yt), "d_twin": M.delta(L_an[sel], yi, yt), "d_probe": dd("probe"),
                "d_rand": M.delta(st("rand"), yi[:, None, None], yt[:, None, None]).swapaxes(1, 2),
                "d_psurf": dd("psurf"), "d_whole": dd("whole"), "d_wsurf": dd("wsurf")}
        use = M.use_scores(pair["d_item"], pair["d_twin"], pair["d_probe"], pair["d_rand"], pair["d_psurf"],
                           n_flip=n_flip, seed=seed)
        ar = M.arrival_scores(pair["d_item"], pair["d_twin"], pair["d_whole"], pair["d_wsurf"], B)
    tim["use_arrival_s"] = time.time() - t0

    out["probe"], out["twin"], out["use"], out["arrival"] = pc, td, use, ar
    out["onset"] = {"probe": M.onset(pc["selectivity"], B) if pc["valid"] else None,
                    "twin": M.onset(td["t"], B) if td["valid_sym"] else None,
                    "use": M.onset(use["use"], B) if use and use["valid"] else None,
                    "arrival": M.onset(ar["arrival"], B) if ar and ar["valid"] else None}
    out["valid"] = {"probe": pc["valid"], "twin": td["valid_sym"], "use": bool(use and use["valid"]),
                    "arrival": bool(ar and ar["valid"])}
    out["timing"] = tim

    ids = lambda xs: np.array([x["id"] for x in xs], dtype=str)  # noqa: E731
    arrays = {"n_blocks": np.array(B), "bow_acc": np.array(np.nan if bow is None else bow),
              "y_te": y_te, "y_an": y_an, "test_ids": ids(b["te"]) if "te" in b else np.array([], dtype=str),
              **arr, "d_ans": d_ans.astype(np.float32), "d_surf": d_surf.astype(np.float32),
              "pair_idx": sel, "pair_ids": ids([b["te"][i] for i in sel]) if "te" in b else np.array([], dtype=str),
              **{k: np.asarray(v, dtype=np.float32) for k, v in pair.items()}}
    if has_fr:
        arrays.update(y_fr=np.asarray(b["y_fr"]), bow_fr=np.array(b.get("bow_fr", np.nan)),
                      fresh_ids=ids(b["fr"]) if "fr" in b else np.array([], dtype=str))
        out["n_fresh"] = len(b["ids_fr"])
    return out, arrays


def _write_atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_npz(path, arrays: dict):
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    _write_atomic(Path(path), buf.getvalue())


def run_measure(ad, name, prompt, out_json, out_npz, n_perm=20, max_pairs=200, seed=0, n_fresh=200, log=print,
                extra=None):
    """One (task, adapter) on the plain or 2-shot prompt: the full analyze_task pipeline plus per-item arrays (npz,
    written first; the JSON is written last, so an existing JSON means done). `extra` is merged into the JSON."""
    t_all = time.time()
    b = build_task(ad, name, prompt, seed=seed, with_fresh=True, n_fresh=n_fresh)
    res, arrays = measure_arrays(ad, b, n_perm=n_perm, max_pairs=max_pairs, seed=seed, log=lambda *a: log("   ", *a))
    task = b["task"]
    res["timing"]["load_s"] = getattr(ad, "load_seconds", None)
    res["timing"]["total_s"] = time.time() - t_all
    res["answer_logfreq"] = ad.answer_logfreq(b["opt_ids"])
    res.update(model=getattr(ad, "model_id", None), task=name, prompt=prompt, family=task.family,
               type_label=task.type_label, options=list(task.options), items_dropped=b["dropped"],
               items_kept=len(b["kept"]), n_train_kept=len(b["tr"]), n_test_kept=len(b["te"]), rivals=b["rivals"],
               fresh_dropped=b.get("fr_dropped"), bow_fresh=b.get("bow_fr"))
    res.update(extra or {})
    save_npz(out_npz, arrays)
    _write_atomic(Path(out_json), json.dumps(res, default=_json).encode())
    return res


# ---------------------------------------------------------------- half curves and fresh curve (pure numpy)

def _halves(n: int, seed: int):
    p = np.random.default_rng(seed).permutation(n)
    h = n // 2
    return np.sort(p[:h]), np.sort(p[h:])


def _bow(npz, key):
    v = float(npz[key]) if key in npz else float("nan")
    return None if np.isnan(v) else v


def probe_half_curves(npz, seed: int = 0) -> dict:
    """The two seeded halves of the test items: per half the probe accuracy per block, the control (mean permuted-probe
    accuracy on that half's items, max with the bag-of-words rival), the selectivity and its onset."""
    pt, pp, y = np.asarray(npz["pred_te"]), np.asarray(npz["pred_perm"]), np.asarray(npz["y_te"])
    B, bow = int(npz["n_blocks"]), _bow(npz, "bow_acc")
    out = []
    for idx in _halves(len(y), seed):
        acc = (pt[:, idx] == y[idx]).mean(1)
        null = (pp[:, :, idx] == y[idx]).mean(2)  # [P, B+1]
        ctrl = null.mean(0) if len(null) else np.full(len(acc), CHANCE)
        if bow is not None:
            ctrl = np.maximum(ctrl, bow)
        sel = (acc - ctrl) / (1 - CHANCE)
        out.append({"idx": idx, "n": int(len(idx)), "acc": acc, "control": ctrl, "selectivity": sel,
                    "onset": M.onset(sel, B)})
    return {"seed": seed, "halves": out}


def use_half_curves(npz, seed: int = 0) -> dict:
    """The two seeded halves of the qualifying pairs: the use curve (as `use_scores`, no sign-flip test) and onset."""
    B, P = int(npz["n_blocks"]), len(npz["d_item"])
    out = []
    for idx in _halves(P, seed):
        if len(idx) < 2:
            out.append({"idx": idx, "n_pairs": int(len(idx)), "use": None, "onset": None})
            continue
        u = M.use_scores(npz["d_item"][idx], npz["d_twin"][idx], npz["d_probe"][idx], npz["d_rand"][idx],
                         npz["d_psurf"][idx], n_flip=0, seed=seed, min_pairs=0)["use"]
        out.append({"idx": idx, "n_pairs": int(len(idx)), "use": np.asarray(u), "onset": M.onset(u, B)})
    return {"seed": seed, "halves": out}


def fresh_probe_curve(npz) -> dict:
    """The same probe fits scored on the fresh items: accuracy, control (mean permuted accuracy, max with the rival on
    the fresh items `bow_fr`), selectivity, onset."""
    pf, pfp, y = np.asarray(npz["pred_fr"]), np.asarray(npz["pred_fr_perm"]), np.asarray(npz["y_fr"])
    B, bow = int(npz["n_blocks"]), _bow(npz, "bow_fr")
    acc = (pf == y).mean(1)
    ctrl = (pfp == y).mean(2).mean(0)
    if bow is not None:
        ctrl = np.maximum(ctrl, bow)
    sel = (acc - ctrl) / (1 - CHANCE)
    return {"n": int(len(y)), "acc": acc, "control": ctrl, "selectivity": sel, "onset": M.onset(sel, B)}


# ---------------------------------------------------------------- robustness arm

def meandiff_subspace(X, y, k: int = 3):
    """[d, k] orthonormal: class means of the raw train states minus the grand mean, top-k right singular vectors."""
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
    means = np.stack([X[y == c].mean(0) for c in range(4)]) - X.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(means, full_matrices=False)
    return vt[:k].T.astype(np.float32)


def _fit_dir(X, y, C):
    sc, lr = M._fit(X, y, C=C)
    return M.probe_subspace(sc.scale_, lr.coef_).astype(np.float32)


def robust_subspaces(S_tr, y_tr, Cs):
    """{variant: [B+1, d, 3]}: probes refit at each C (one fit per block, no CV, no permutations) and the mean-difference
    subspace. Variant names: "C=<C:g>" and "meandiff"."""
    y_tr = np.asarray(y_tr)
    nb = S_tr.shape[1]
    jobs = int(os.environ.get("ATLAS_PROBE_JOBS", "1"))
    todo = [(C, b) for C in Cs for b in range(nb)]
    if jobs <= 1:
        fits = [_fit_dir(S_tr[:, b], y_tr, C) for C, b in todo]
    else:
        from joblib import Parallel, delayed
        fits = Parallel(n_jobs=jobs)(delayed(_fit_dir)(S_tr[:, b], y_tr, C) for C, b in todo)
    out = {f"C={C:g}": np.stack(fits[j * nb:(j + 1) * nb]) for j, C in enumerate(Cs)}
    out["meandiff"] = np.stack([meandiff_subspace(S_tr[:, b], y_tr) for b in range(nb)])
    return out


def robust_built(ad, b, Cs=(0.01, 0.1, 1, 10), max_pairs=200, seed=0, n_flip=10000, log=None):
    """Use curve and onset per subspace variant on a built task (same pairs and random-subspace control as the main run)."""
    B, opt_ids = ad.n_blocks, b["opt_ids"]
    y_te, y_an = np.asarray(b["y_te"]), np.asarray(b["y_an"])
    S_tr, _ = ad.collect_ids(b["ids_tr"], opt_ids)
    S_te, L_te = ad.collect_ids(b["ids_te"], opt_ids)
    S_an, L_an = ad.collect_ids(b["ids_an"], opt_ids)
    S_sf, _ = ad.collect_ids(b["ids_sf"], opt_ids)
    sel, n_qual = _select_pairs(L_te, L_an, y_te, y_an, max_pairs, seed)
    subs = robust_subspaces(S_tr, b["y_tr"], Cs)
    names = list(subs)
    out = {"n_pairs_qualifying": int(n_qual), "n_pairs_used": int(len(sel)), "n_blocks": B, "variants": {}}
    if len(sel) == 0:
        out["onsets"] = {v: None for v in names}
        return out
    Qr = M.random_subspaces(ad.d, B, seed=seed)
    R, V = Qr.shape[1], len(names)
    res = []
    for i in sel:
        rows = []
        for L in range(1, B):
            for r in range(R):
                rows.append((L, swap_subspace(S_te[i, L], S_an[i, L], Qr[L, r])))
            for v in names:
                rows.append((L, swap_subspace(S_te[i, L], S_an[i, L], subs[v][L])))
                rows.append((L, swap_subspace(S_te[i, L], S_sf[i, L], subs[v][L])))
        lg = ad.patched_logits(b["ids_te"][i], rows, opt_ids)
        res.append(lg.reshape(B - 1, R + 2 * V, -1))
    g = np.stack(res)  # [P, K, R+2V, 4]
    yi, yt = y_te[sel], y_an[sel]
    d_item, d_twin = M.delta(L_te[sel], yi, yt), M.delta(L_an[sel], yi, yt)
    d_rand = M.delta(g[:, :, :R], yi[:, None, None], yt[:, None, None]).swapaxes(1, 2)
    for j, v in enumerate(names):
        d_p = M.delta(g[:, :, R + 2 * j], yi[:, None], yt[:, None])
        d_s = M.delta(g[:, :, R + 2 * j + 1], yi[:, None], yt[:, None])
        u = M.use_scores(d_item, d_twin, d_p, d_rand, d_s, n_flip=n_flip, seed=seed)
        out["variants"][v] = {"use": u["use"], "peak": u["peak"], "p_signflip": u["p_signflip"], "valid": u["valid"],
                              "onset": M.onset(u["use"], B)}
    out["onsets"] = {v: (x["onset"]["rel"] if x["onset"] else None) for v, x in out["variants"].items()}
    base = out["onsets"].get("C=1")
    out["onset_shift_vs_C1"] = {v: (None if base is None or o is None else abs(o - base)) for v, o in out["onsets"].items()}
    shifts = [s for v, s in out["onset_shift_vs_C1"].items() if v != "C=1"]
    out["robust"] = bool(base is not None and all(s is not None and s <= 0.10 for s in shifts))
    return out


def run_robust(ad, name, Cs=(0.01, 0.1, 1, 10), prompt="plain", max_pairs=200, seed=0, n_flip=10000, log=None):
    """Robustness arm (v13 section 5) for one (task, adapter): JSON dict with the onsets per variant."""
    t0 = time.time()
    b = build_task(ad, name, prompt, seed=seed)
    res = robust_built(ad, b, Cs, max_pairs=max_pairs, seed=seed, n_flip=n_flip, log=log)
    task = b["task"]
    res.update(model=getattr(ad, "model_id", None), task=name, prompt=prompt, family=task.family, Cs=list(Cs),
               total_s=time.time() - t0)
    return res


# ---------------------------------------------------------------- heal for every architecture

def _p0h():
    """scripts/p0_heal.py (adapter io, device, pick_lr), imported by path."""
    d = str(Path(__file__).resolve().parents[2] / "scripts")
    if d not in sys.path:
        sys.path.insert(0, d)
    import p0_heal
    return p0_heal


def pick_lr(rows) -> float:
    return _p0h().pick_lr(rows)


def base_module(model):
    return model.gpt_neox if model.config.model_type == "gpt_neox" else model.model


def attach_lora_any(model, seed=0):
    """q_proj/v_proj LoRA (rank 10, scale 2.0, as `heal.add_heal`); GPT-NeoX: fused-QKV LoRA on the q and v rows."""
    import torch
    from surgery import adapters
    if model.config.model_type == "gpt_neox":
        torch.manual_seed(seed)
        return adapters.add_lora_fused_qkv(model, list(range(len(model.gpt_neox.layers))), r=RANK, scale=SCALE)
    from surgery.atlas.heal import add_heal
    return add_heal(model, len(model.model.layers), rank=RANK, seed=seed, scale=SCALE)


def option_logits_any(model, ids_list, opt_ids, device):
    """[B, 4] float32 logits over the options at the last real token (right padded), any supported architecture."""
    import torch
    n = max(len(x) for x in ids_list)
    x = torch.zeros(len(ids_list), n, dtype=torch.long)
    m = torch.zeros(len(ids_list), n, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        x[i, :len(ids)], m[i, :len(ids)] = torch.tensor(ids), 1
    x, m = x.to(device), m.to(device)
    h = base_module(model)(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state
    last = h[torch.arange(len(ids_list), device=device), m.sum(1) - 1]
    w = model.get_output_embeddings().weight[torch.tensor(opt_ids, device=device)]
    return (last @ w.T).float()


def accuracy_any(model, ids, y, opt_ids, device, bs=32) -> float:
    import torch
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    pred = np.zeros(len(ids), dtype=int)
    with torch.no_grad():
        for s in range(0, len(order), bs):
            sel = order[s:s + bs]
            pred[sel] = option_logits_any(model, [ids[i] for i in sel], opt_ids, device).argmax(1).cpu().numpy()
    return float((pred == np.asarray(y)).mean())


def filter_by_tok(tok, items):
    """`Adapter.filter_items` from the tokenizer alone: keep items whose item / answer twin / surface twin token lengths
    and last tokens agree."""
    kept = []
    for x in items:
        a, b, c = (tok(p, add_special_tokens=False).input_ids for p in
                   (x["prompt"], x["ans_twin"]["prompt"], x["surf_twin"]["prompt"]))
        if len(a) == len(b) == len(c) and a[-1] == b[-1] == c[-1]:
            kept.append(x)
    return kept


def heal_data(tok, name, limit_train=None):
    """(train, dev, test) as (token id lists, answers): `lora_items` first 2000 / last 300; test = the probe test split
    (plain prompt, after the token-alignment filter)."""
    lora = lora_items(name, N_LORA)
    tr = lora[:N_TRAIN] if limit_train is None else lora[:limit_train]
    dev = lora[N_TRAIN:]
    te = filter_by_tok(tok, fam.split_items(fam.generate(name, 0))[1])
    enc = lambda xs: ([tok(x["prompt"], add_special_tokens=False).input_ids for x in xs], [x["answer"] for x in xs])  # noqa: E731
    return enc(tr), enc(dev), enc(te)


def heal_task_any(model, tok, name, lr, seed, out, device, steps=STEPS, limit_train=None, base_cache=None, log=print):
    """`p0_heal.heal_task` for family tasks and every architecture (same optimiser, schedule, 300 steps, batch 8, bf16
    autocast on cuda). The model is left clean. Writes <out>/<task>_lr<lr>_s<seed>/{adapter.safetensors,heal.json}."""
    import torch
    from surgery.atlas.heal import _schedule
    p0h = _p0h()
    opt_ids = fam.check_options(tok, name)
    (x_tr, y_tr), (x_dev, y_dev), (x_te, y_te) = heal_data(tok, name, limit_train)
    cache = base_cache if base_cache is not None else {}
    model.eval()
    if name not in cache:
        cache[name] = accuracy_any(model, x_te, y_te, opt_ids, device)
    handle = attach_lora_any(model, seed)
    try:
        params = [p for p in handle.modules.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr)
        sched = _schedule(opt, steps)
        g = torch.Generator().manual_seed(seed)
        y_all = torch.tensor(y_tr)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
        order, losses = [], []
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for step in range(1, steps + 1):
            if len(order) < BATCH:
                order = torch.randperm(len(x_tr), generator=g).tolist() + order
            idx, order = order[:BATCH], order[BATCH:]
            with amp:
                lg = option_logits_any(model, [x_tr[i] for i in idx], opt_ids, device)
            loss = torch.nn.functional.cross_entropy(lg, y_all[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
            if step % 50 == 0:
                log(f"  {name} lr{lr:g} s{seed} step {step}/{steps} loss {np.mean(losses[-50:]):.3f}")
        if device == "cuda":
            torch.cuda.synchronize()
        train_s = time.perf_counter() - t0
        with amp:
            dev_acc = accuracy_any(model, x_dev, y_dev, opt_ids, device)
            test_acc = accuracy_any(model, x_te, y_te, opt_ids, device)
        wall = time.perf_counter() - t0
        d = Path(out) / f"{name}_lr{lr:g}_s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        p0h.save_adapter(handle, d / "adapter.safetensors")
    finally:
        handle.remove()
    res = {"task": name, "arch": model.config.model_type, "wall_seconds": wall, "train_seconds": train_s,
           "step_seconds": train_s / max(steps, 1), "steps": steps, "lr": lr, "seed": seed, "dev_acc": dev_acc,
           "test_acc": test_acc, "base_test_acc": cache[name], "device": device, "n_train": len(x_tr),
           "n_dev": len(x_dev), "n_test": len(x_te), "final_train_loss": float(np.mean(losses[-20:]))}
    (d / "heal.json").write_text(json.dumps(res, indent=1))
    return res


def load_trained(model_id, adapter_path, device="cpu", dtype=None):
    """A fresh base model with the adapter's LoRA merged into its weights, as a `p0_harness.Adapter`."""
    import torch
    from surgery import adapters
    ad = Adapter(model_id, device=device, dtype=dtype or torch.float32)
    ad.model_id = model_id
    handle = attach_lora_any(ad.model)
    _p0h().load_adapter(handle, adapter_path)
    ad.merged_modules = adapters.merge_lora(handle)
    return ad


# ---------------------------------------------------------------- seeded draws

def _draw(items, k, seed):
    s = sorted(items)
    return [s[i] for i in np.random.default_rng(seed).permutation(len(s))[:min(k, len(s))]]


def lr_tasks(tasks, seed: int = 0, k: int = 6) -> list:
    """The 6 learning-rate tasks: `default_rng(seed).permutation(sorted(tasks))[:6]`, in drawn order (the first is the
    C-adapt task)."""
    return _draw(tasks, k, seed)


def seed_tasks(tasks, seed: int = 0, k: int = 12) -> list:
    """The 12 seed-check tasks: the same permutation as `lr_tasks`, first 12 (so the 6 LR tasks are among them)."""
    return _draw(tasks, k, seed)


def robust_draw(models, families, seed: int = 0, k_fam: int = 6) -> dict:
    """One model from each of the Qwen2.5, SmolLM2 and Pythia lineages (substring match on sorted ids; one uniform draw per
    lineage whether or not it has a candidate) and `k_fam` families (`default_rng(seed)` permutation of sorted)."""
    rng = np.random.default_rng(seed)
    picked = []
    for lin in LINEAGES:
        u = rng.random()
        cand = sorted(m for m in models if lin.lower() in m.lower())
        if cand:
            picked.append(cand[int(u * len(cand))])
    fams = sorted(families)
    idx = rng.permutation(len(fams))[:min(k_fam, len(fams))]
    return {"models": picked, "families": [fams[i] for i in sorted(idx)]}
