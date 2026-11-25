"""Steps study v2 run pipeline (docs/ATLAS-steps.md): heals, probes, prediction, survival scores, twin patching, bars S1-S6.

  uv run python scripts/steps_run.py --stage sweep --seed S [--shard I/N] [--model qwen17|smol17]
  uv run python scripts/steps_run.py --stage probe [--task T] --seed S [--model ...]
  uv run python scripts/steps_run.py --stage predict | cutscore | patch | table | project   [--smoke]

Reuses scripts/atlas3_cuts.py (`make_cfg`, `run_whole_heal`, `stage_sweep`, the shared order, the C1 recipe: q/v LoRA rank 10,
300 steps, seed 0 tries C1's two lrs and picks on lrsel, seeds 1-2 use that winner) and surgery/probes.py (the linear probe
fit). Results: var/results/steps (SmolLM2 in var/results/steps/smol17), checkpoints var/ckpt/steps, activations var/acts/steps.
--smoke writes under var/results/steps/smoke, var/ckpt/steps_smoke and var/acts/steps_smoke, uses 4 training steps and a few
records per cell, and never writes the test ledger. Every stage resumes (a heal is skipped when its run key is in a shard
file, a probe or cutscore model when its result is already saved). `predict` writes predicted.json and its sha256 into the
stream log; `cutscore` refuses to run without it (S5 is judged against depths fixed before any survival score is opened).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import atlas2_cuts as B  # noqa: E402
import atlas3_cuts as C  # noqa: E402
import atlas_cuts as ac  # noqa: E402
from atlas2_healed import l_on_healed  # noqa: E402

from surgery import adapters, probes as P  # noqa: E402
from surgery.atlas import heal, measure, metrics, spec, steps as ST  # noqa: E402

TASKS = ST.STEPS_TASKS
COPY, OPS, DESC = TASKS
SEEDS = (0, 1, 2)
KCUT = 12
ONSET_ACC = 0.55
EPS = 1e-9
DROP_MARGIN = 0.30                      # cell dropped if uncut selection accuracy < chance + 0.30
EFFECT, NO_EFFECT = 3.0, 1.0
S5_RHO, S5_MIN_CELLS, S5_SPREAD, S5_CEIL, S5_FLOOR = -0.6, 8, 0.10, 0.95, 0.05
S6_MIN_BLOCKS = 3.0 * 24 / 28           # 3 blocks scaled from 28 to 24: 2.6
MIN_REFITS = 20                         # fewer refit replicates than this: the intervals come from the OOF bootstrap
MLP_HIDDEN, MLP_ALPHAS = 64, (1e-3, 1e-2, 1e-1, 1.0)
BORDERLINE = ("tall", "garden", "pot", "small", "branches", "desert", "petals", "woody")   # F4 truth-table cells people split on
PATCH_SITES = ("last", "flip_after", "final_binding")
SITE_KEYS = {"flip_after": "flip_after_tok", "final_binding": "final_binding_tok", "query": "query_tok"}
SMOKE_PER_CELL = {"discovery": 30, "train": 6, "lrsel": 3, "selection": 6, "test": 6}

ROOT_OUT = Path("var/results/steps")
ROOT_CKPT = Path("var/ckpt/steps")
ROOT_ACTS = Path("var/acts/steps")
PROFILES = {
    "qwen17": SimpleNamespace(tasks=TASKS, ks=(0, KCUT), seeds=SEEDS, n_layers=28, sub=""),
    "smol17": SimpleNamespace(tasks=(COPY,), ks=(0,), seeds=(0,), n_layers=24, sub="smol17"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- directories, config, context

def root_out(smoke: bool) -> Path:
    return ROOT_OUT / "smoke" if smoke else ROOT_OUT


def dirs(model: str, smoke: bool) -> SimpleNamespace:
    sub = PROFILES[model].sub
    out, ck, ax = root_out(smoke), (Path(str(ROOT_CKPT) + "_smoke") if smoke else ROOT_CKPT), \
        (Path(str(ROOT_ACTS) + "_smoke") if smoke else ROOT_ACTS)
    if sub:
        out, ck, ax = out / sub, ck / sub, ax / sub
    return SimpleNamespace(root=root_out(smoke), out=out, ckpt=ck, acts=ax)


def make_cfg(model: str, smoke: bool, boots=None, refits=None) -> SimpleNamespace:
    """C1's configuration (atlas3_cuts.make_cfg) with this study's tasks and k grid. Smoke: 4 steps, few records per cell."""
    prof = PROFILES[model]
    base = C.make_cfg(False, C.resolve_profile(model) if model != "qwen17" else None)
    cfg = SimpleNamespace(**vars(base))
    cfg.smoke, cfg.sfx, cfg.ledger, cfg.n = smoke, "", not smoke, None
    cfg.tasks = (COPY,) if smoke else prof.tasks
    cfg.table_tasks = cfg.tasks
    cfg.ks = prof.ks
    cfg.kcut = KCUT
    cfg.steps = 4 if smoke else 300
    cfg.boots = boots if boots is not None else (20 if smoke else 500)
    cfg.refits = refits if refits is not None else (3 if smoke else 20)
    cfg.ckpt_tokens = C.D3_CKPT_TOKENS if model == "smol17" and not smoke else None
    cfg.model = model
    cfg.n_layers = prof.n_layers
    return cfg


def configure(model: str, smoke: bool) -> SimpleNamespace:
    """Point the shared modules at this study's folders and the model's profile (done in main, never at import)."""
    C.configure(None if model == "qwen17" else C.PROFILES[model])
    d = dirs(model, smoke)
    for m in (ac, B, C):
        m.OUT, m.CKPT, m.ACTS = d.out, d.ckpt, d.acts
    d.out.mkdir(parents=True, exist_ok=True)
    return d


def smoke_subset(recs: list[dict], n: int) -> list[dict]:
    """The first `n` records of every cell (cells in order of first appearance)."""
    seen, out = {}, []
    for r in recs:
        c = r.get("rung")
        if seen.get(c, 0) < n:
            seen[c] = seen.get(c, 0) + 1
            out.append(r)
    return out


class StepsCtx(ac.Ctx):
    """`atlas_cuts.Ctx` whose records keep a few of each cell in smoke mode (the files are ordered by cell)."""

    def records(self, task: str, split: str) -> list[dict]:
        key = (task, split)
        if key not in self._rec:
            recs = spec.read_records(task, split)
            self._rec[key] = smoke_subset(recs, SMOKE_PER_CELL[split]) if self.cfg.smoke else recs
        return self._rec[key]


def make_ctx(cfg, order=None) -> StepsCtx:
    ctx = StepsCtx(cfg, order=order, device=os.environ.get("ATLAS3_DEVICE") or None)
    if cfg.model != "qwen17" and set(cfg.ks) == {0} and not ac.ORDER_PATH.exists():
        ctx._order = []                         # nothing is cut at k=0, so the order is never read
    return ctx


def stream(root: Path, msg: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "stream.log").open("a") as fh:
        fh.write(f"{now()} {msg}\n")


def winner_row(runs: dict, task: str, k: int, seed: int):
    rows = [r for r in ac.collect(runs, [task], [k], "per_k").get(task, {}).get(k, []) if r["seed"] == seed]
    return rows[0] if rows else None


def cells_of(task: str) -> list[str]:
    return list(ST.cell_sizes(task)["test"])


# ---------------------------------------------------------------- pure helpers: depth, onset, bootstrap

def first_reaching(curve, thr):
    for i, v in enumerate(curve):
        if v >= thr - EPS:
            return i
    return None


def onset_fixed(curve, acc: float = ONSET_ACC):
    """First layer whose probe accuracy reaches the fixed `acc`; None if never."""
    return first_reaching(curve, acc)


def first_reaching_vec(curves: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Row-wise first index with value >= thr (row-wise); curves.shape[1] if none (the 'never' depth)."""
    hit = curves >= thr[:, None] - EPS
    return np.where(hit.any(1), hit.argmax(1), curves.shape[1]).astype(float)


def group_tables(groups, answers, correct_layers: np.ndarray, correct_ft: np.ndarray, n_classes: int = 10):
    """Per-template counts for the vectorised grouped bootstrap: sizes (G,), correct per layer (G, L+1), first-token correct
    (G,), class counts (G, n_classes)."""
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    G = int(inv.max()) + 1
    size = np.bincount(inv, minlength=G).astype(float)
    corr = np.zeros((G, correct_layers.shape[0]))
    np.add.at(corr, inv, correct_layers.T.astype(float))
    ft = np.bincount(inv, weights=correct_ft.astype(float), minlength=G)
    cls = np.zeros((G, n_classes))
    np.add.at(cls, (inv, np.asarray(answers)), 1.0)
    return {"size": size, "corr": corr, "ft": ft, "cls": cls}


def oof_boot_depths(tab: dict, n_boot: int, seed: int = 0):
    """Grouped bootstrap of the stored out-of-fold predictions (no refits): resample templates with replacement, recompute the
    curve, the first-token accuracy and the chance, then `l_on_healed`'s rule and the fixed-accuracy onset. Returns
    (depth, onset) arrays of length n_boot; depth is nan when the resample's accuracy is under chance + 0.05."""
    rng = np.random.default_rng(seed)
    G = len(tab["size"])
    idx = rng.integers(0, G, (n_boot, G))
    n = tab["size"][idx].sum(1)
    acc = tab["ft"][idx].sum(1) / n
    chance = tab["cls"][idx].sum(1).max(1) / n
    curve = tab["corr"][idx].sum(1) / n[:, None]
    thr = chance + 0.5 * (acc - chance)
    depth = first_reaching_vec(curve, thr)
    depth[acc - chance < metrics.MIN_GAP - metrics.EPS] = np.nan
    return depth, first_reaching_vec(curve, np.full(n_boot, ONSET_ACC))


def linear_oof(Xc, a, g, seed: int, jobs: int = 1, layers=None) -> np.ndarray:
    """Out-of-fold predicted labels per layer, (len(layers), n): the fit `probes.probe_by_layer` makes (StandardScaler +
    logistic regression, GroupKFold(5) on templates, seed + layer)."""
    layers = range(Xc.shape[1]) if layers is None else layers

    def one(l):
        return P._oof(np.asarray(Xc[:, l], dtype=np.float32), a, g, "label", seed + l, False)

    if jobs > 1:
        from joblib import Parallel, delayed
        return np.stack(Parallel(n_jobs=jobs)(delayed(one)(l) for l in layers))
    return np.stack([one(l) for l in layers])


def linear_curve(Xc, a, g, seed: int, jobs: int = 1) -> tuple[list[float], np.ndarray]:
    oof = linear_oof(Xc, a, g, seed, jobs)
    return [float((o == a).mean()) for o in oof], oof


def depth_of(curve, chance: float, acc: float):
    """(L_on_healed, never) of D1's rule for this curve."""
    return l_on_healed(curve, chance, acc)


def refit_replicate(Xc, a, g, pred_ft, seed: int):
    """One bootstrap replicate with refits: resample templates with replacement (a repeated template keeps one id, so it stays
    in one fold), refit the probe layer by layer from layer 0 and stop at the first layer reaching the replicate's threshold
    and the fixed accuracy. Returns (depth, onset); depth nan if the replicate's accuracy is under chance + 0.05 or it cannot
    be cross-validated, onset = n_layers + 1 when never reached."""
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    pick = rng.integers(0, len(ug), len(ug))
    by = {u: np.flatnonzero(g == u) for u in ug}
    rows = np.concatenate([by[ug[i]] for i in pick])
    gg, aa = g[rows], a[rows]
    if len(set(gg.tolist())) < 5 or len(set(aa.tolist())) < 2:
        return float("nan"), float("nan")
    acc = float((pred_ft[rows] == aa).mean())
    chance = spec.chance(aa.tolist())
    valid = acc - chance >= metrics.MIN_GAP - metrics.EPS
    thr = chance + 0.5 * (acc - chance)
    nL = Xc.shape[1]
    depth, onset = (float("nan") if not valid else None), None
    for l in range(nL):
        o = P._oof(np.asarray(Xc[rows, l], dtype=np.float32), aa, gg, "label", seed + l, False)
        v = float((o == aa).mean())
        if depth is None and v >= thr - EPS:
            depth = float(l)
        if onset is None and v >= ONSET_ACC - EPS:
            onset = float(l)
        if depth is not None and onset is not None:
            break
    return (float(nL) if depth is None else depth), (float(nL) if onset is None else onset)


# ---------------------------------------------------------------- MLP probe and the Hewitt-Liang control

def control_labels(answers, templates, n_classes: int = 10, seed: int = 0) -> np.ndarray:
    """Hewitt and Liang's control task: each answer re-mapped by a fixed random permutation of the classes, one permutation
    per template (derived from the template id, so it is the same wherever the records are)."""
    a, t = np.asarray(answers), np.asarray(templates)
    out = np.empty_like(a)
    for u in dict.fromkeys(t.tolist()):
        perm = np.random.default_rng([seed, zlib.crc32(str(u).encode())]).permutation(n_classes)
        m = t == u
        out[m] = perm[a[m]]
    return out


def mlp_fit_predict(Xtr, ytr, Xte, alpha: float, seed: int):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(ytr)) == 1:
        return np.full(len(Xte), ytr[0])
    clf = make_pipeline(StandardScaler(), MLPClassifier((MLP_HIDDEN,), alpha=alpha, max_iter=300, random_state=seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(Xtr, ytr)
    return clf.predict(Xte)


def mlp_oof_acc(Xl, y, groups, alpha: float, seed: int, grouped: bool) -> float:
    from sklearn.model_selection import GroupKFold, KFold
    splits = (GroupKFold(5).split(Xl, y, groups) if grouped else KFold(5, shuffle=True, random_state=seed).split(Xl, y))
    hits = 0
    for tr, te in splits:
        hits += int((mlp_fit_predict(Xl[tr], y[tr], Xl[te], alpha, seed) == y[te]).sum())
    return hits / len(y)


def mlp_cell(Xc, a, g, alpha: float, seed: int, jobs: int = 1) -> dict:
    """Per layer: MLP accuracy with templates held out (the onset curve), and with a record-level split on the real answers
    and on the control labels (their difference is the selectivity)."""
    ctrl = control_labels(a, g)

    def one(l):
        Xl = np.asarray(Xc[:, l], dtype=np.float32)
        return (mlp_oof_acc(Xl, a, g, alpha, seed + l, True), mlp_oof_acc(Xl, a, g, alpha, seed + l, False),
                mlp_oof_acc(Xl, ctrl, g, alpha, seed + l, False))

    layers = range(Xc.shape[1])
    if jobs > 1:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=jobs)(delayed(one)(l) for l in layers)
    else:
        rows = [one(l) for l in layers]
    grouped, rec, con = (list(map(float, c)) for c in zip(*rows))
    return {"curve": grouped, "real_record": rec, "control_record": con,
            "selectivity": [r - c for r, c in zip(rec, con)]}


def choose_alpha(score_fn, alphas=MLP_ALPHAS) -> float:
    """The weight decay with the best lrsel accuracy; ties go to the smaller one. `score_fn(alpha) -> float`."""
    best = None
    for al in alphas:
        s = score_fn(al)
        if best is None or s > best[0] + EPS:
            best = (s, al)
    return best[1]


# ---------------------------------------------------------------- stage: sweep

def stage_sweep(ctx, seed: int, shard) -> int:
    return C.stage_sweep(ctx, seed, shard)


# ---------------------------------------------------------------- stage: probe

def probe_path(d, task: str, seed: int) -> Path:
    return d.out / f"probe_{task}_s{seed}.json"


def exact_by_cell(ctx, model, task: str, recs: list[dict]) -> tuple[dict, dict]:
    """Exact-match accuracy per cell (and per record id) with the study's generation settings."""
    _, texts = heal.exact_match_scores(model, ctx.tokenizer, task, recs, max_new_tokens=C.max_new_tokens(ctx, task),
                                       batch_size=ac.EVAL_BS)
    ok = [heal.exact_match(t, heal.answer_text(task, r)) for t, r in zip(texts, recs)]
    cells = {}
    for c, idx in cell_index([r["rung"] for r in recs]).items():
        cells[c] = {"acc": float(np.mean([ok[i] for i in idx])), "n": len(idx),
                    "chance": spec.chance([recs[i]["answer"] for i in idx])}
    return cells, {r["id"]: int(o) for r, o in zip(recs, ok)}


def cell_index(rungs) -> dict:
    """{cell: indices}, cells in order of first appearance."""
    out = {}
    for i, c in enumerate(rungs):
        out.setdefault(c, []).append(i)
    return {c: np.asarray(v) for c, v in out.items()}


def avoid_mask(recs: list[dict]) -> np.ndarray:
    """F4 items whose shown clues touch none of the borderline truth-table features (a shown feature touches the cells it
    names for every candidate concept, so any shown borderline feature excludes the item)."""
    bad = set(BORDERLINE)
    return np.asarray([not (set(r["meta"]["features"]) & bad) for r in recs], dtype=bool)


def probe_cell_linear(Xc, a, g, pred_ft, cfg, seed: int, jobs: int, with_refits: bool = True) -> dict | None:
    """Linear probe curve of one cell, D1's depth, onset at 0.55, the OOF bootstrap and the refit bootstrap."""
    if len(set(g.tolist())) < 5 or len(set(a.tolist())) < 2:
        return None
    chance = spec.chance(a.tolist())
    acc = float((pred_ft == a).mean())
    curve, oof = linear_curve(Xc, a, g, seed, jobs)
    depth, never = depth_of(curve, chance, acc)
    tab = group_tables(g, a, oof == a[None, :], pred_ft == a)
    bd, bo = oof_boot_depths(tab, cfg.boots, seed)
    res = {"n": int(len(a)), "chance": chance, "acc": acc, "curve": curve, "L_on_healed": depth, "never": bool(never),
           "onset055": onset_fixed(curve), "peak": max(curve), "L_on_peak": metrics.l_on(curve, chance),
           "boot": {"depth": bd.tolist(), "onset055": bo.tolist()}}
    if with_refits and cfg.refits:
        seeds = [seed * 1000 + j for j in range(cfg.refits)]
        if jobs > 1:
            from joblib import Parallel, delayed
            reps = Parallel(n_jobs=jobs)(delayed(refit_replicate)(Xc, a, g, pred_ft, s) for s in seeds)
        else:
            reps = [refit_replicate(Xc, a, g, pred_ft, s) for s in seeds]
        res["refit"] = {"depth": [r[0] for r in reps], "onset055": [r[1] for r in reps]}
    return res


def select_alpha(Xd, meta_d, Xl, meta_l, seed: int) -> float:
    """MLP weight decay chosen on the lrsel states: fit on each cell's discovery states at a few layers, accuracy pooled."""
    cd, cl = cell_index(meta_d["rungs"]), cell_index(meta_l["rungs"])
    ad, gd = np.asarray(meta_d["answers"]), np.asarray(meta_d["template_ids"])
    al = np.asarray(meta_l["answers"])
    layers = [l for l in range(4, Xd.shape[1], 4)] or [Xd.shape[1] - 1]

    def score(alpha):
        hit = tot = 0
        for c, idx in cd.items():
            jl = cl.get(c)
            if jl is None or len(jl) == 0:
                continue
            for l in layers:
                pred = mlp_fit_predict(np.asarray(Xd[idx, l], dtype=np.float32), ad[idx],
                                       np.asarray(Xl[jl, l], dtype=np.float32), alpha, seed + l)
                hit += int((pred == al[jl]).sum())
                tot += len(jl)
        return hit / max(tot, 1)

    return choose_alpha(score)


def site_probe(ctx, model, task: str, recs: list[dict], cfg, seed: int, jobs: int) -> dict:
    """F1, report only: probes on twin pairs (base and flip) at the token after the first differing link, the final binding
    and the query token. Linear onset per site per cell (peak rule and the fixed 0.55)."""
    out = {}
    for c, idx in cell_index([r["rung"] for r in recs]).items():
        pair = [i for i in idx if recs[i]["meta"]["twin_role"] in ("base", "flip") and recs[i]["meta"].get("flip_after_tok")]
        if len(pair) < 20:
            continue
        sub = [recs[i] for i in pair]
        pos = [[r["meta"][SITE_KEYS[s]] for s in SITE_KEYS] for r in sub]
        S = measure.site_states(model, ctx.tokenizer, task, sub, pos, batch_size=ac.EVAL_BS)
        a, g = np.asarray([r["answer"] for r in sub]), np.asarray([r["template_id"] for r in sub])
        out[c] = {}
        if len(set(g.tolist())) < 5:
            continue
        chance = spec.chance(a.tolist())
        for j, name in enumerate(SITE_KEYS):
            curve, _ = linear_curve(S[:, j], a, g, seed, jobs)
            out[c][name] = {"n": len(sub), "chance": chance, "curve": curve, "onset055": onset_fixed(curve),
                            "L_on_peak": metrics.l_on(curve, chance), "peak": max(curve)}
        del S
    return out


def stage_probe(ctx, d, task: str, seed: int, args) -> int:
    cfg = ctx.cfg
    final = probe_path(d, task, seed)
    if final.exists():
        print(f"probe {task} s{seed}: {final} exists; nothing to do", flush=True)
        return 0
    runs = ac.read_merged("runs", cfg.sfx, "runs")
    row = winner_row(runs, task, 0, seed)
    if row is None:
        print(f"probe {task} s{seed}: no healed k=0 run with a known lr winner; skipped", flush=True)
        return 1
    jobs = args.jobs
    t0 = time.perf_counter()
    recs, lrs_recs, sel = ctx.records(task, "discovery"), ctx.records(task, "lrsel"), ctx.records(task, "selection")
    sd, sl = ac.ACTS / f"{task}_s{seed}_disc", ac.ACTS / f"{task}_s{seed}_lrsel"
    model, handle = ac.rebuild_healed(ctx, row)
    adapters.merge_lora(handle)
    timings = {}
    try:
        meta = measure.build_store(model, ctx.tokenizer, task, recs, sd, batch_size=ac.EVAL_BS)
        measure.build_store(model, ctx.tokenizer, task, lrs_recs, sl, batch_size=ac.EVAL_BS)
        sel_cells, sel_ok = exact_by_cell(ctx, model, task, sel)
        sites = {}
        if task == COPY and cfg.model == "qwen17":
            sites = site_probe(ctx, model, task, recs, cfg, seed, jobs)
        timings["store_seconds"] = time.perf_counter() - t0
    except BaseException:
        shutil.rmtree(sd, ignore_errors=True)
        shutil.rmtree(sl, ignore_errors=True)
        raise
    finally:
        del model
        ac.release(ctx)
    t1 = time.perf_counter()
    try:
        X = np.load(sd / "resid.npy", mmap_mode="r")
        Xl = np.load(sl / "resid.npy", mmap_mode="r")
        meta_l = json.loads((sl / "meta.json").read_text())
        a_all, g_all = np.asarray(meta["answers"]), np.asarray(meta["template_ids"])
        pred = np.asarray(meta["pred"])
        alpha = select_alpha(X, meta, Xl, meta_l, seed)
        cells = {}
        for c, idx in cell_index(meta["rungs"]).items():
            Xc = np.asarray(X[idx])
            r = probe_cell_linear(Xc, a_all[idx], g_all[idx], pred[idx], cfg, seed, jobs)
            if r is None:
                cells[c] = {"n": int(len(idx)), "error": "fewer than 5 templates or one class"}
                continue
            t2 = time.perf_counter()
            ml = mlp_cell(Xc, a_all[idx], g_all[idx], alpha, seed, jobs)
            md, mnever = depth_of(ml["curve"], r["chance"], r["acc"])
            ml.update(alpha=alpha, L_on_healed=md, never=bool(mnever), onset055=onset_fixed(ml["curve"]),
                      seconds=time.perf_counter() - t2)
            at = md if md is not None and not mnever else None
            ml["selectivity_at_onset"] = None if at is None else ml["selectivity"][at]
            ml["selectivity_max"] = max(ml["selectivity"])
            ml["gap"] = None if (md is None or r["L_on_healed"] is None) else r["L_on_healed"] - md
            ml["gap055"] = (None if (ml["onset055"] is None or r["onset055"] is None) else r["onset055"] - ml["onset055"])
            r["mlp"] = ml
            if task == DESC:
                keep = avoid_mask([recs[i] for i in idx])
                if keep.sum() >= 20:
                    sub = idx[keep]
                    ra = probe_cell_linear(Xc[keep], a_all[sub], g_all[sub], pred[sub], cfg, seed, jobs, with_refits=False)
                    if ra:
                        r["avoid"] = {k: v for k, v in ra.items() if k != "boot"}
            cells[c] = r
        del X, Xl
    finally:
        shutil.rmtree(sd, ignore_errors=True)
        shutil.rmtree(sl, ignore_errors=True)
    timings["probe_seconds"] = time.perf_counter() - t1
    ac.write(final, {"task": task, "seed": seed, "model": cfg.model, "k": 0, "lr": row["lr"], "ckpt": row["ckpt"],
                     "n_layers": cfg.n_layers, "n": len(meta["answers"]), "boots": cfg.boots, "refits": cfg.refits,
                     "alpha": alpha, "scoring": "option first-token accuracy (depth); exact match (selection)",
                     "cells": cells, "sites": sites, "selection": {"cells": sel_cells, "correct": sel_ok},
                     "timings": timings, "device": str(ctx.device), "ts": now()})
    print(f"probe {task} s{seed}: {len(cells)} cells, alpha {alpha:g}, store {timings['store_seconds']:.0f}s, probes "
          f"{timings['probe_seconds']:.0f}s -> {final}", flush=True)
    return 0


# ---------------------------------------------------------------- stage: predict

def load_probes(d, model: str, tasks, seeds) -> dict:
    out = {}
    for t in tasks:
        for s in seeds:
            p = ac.read_json(probe_path(d, t, s))
            if p:
                out.setdefault(t, {})[s] = p
    return out


def kept_cells(probes_t: dict, margin: float = DROP_MARGIN) -> dict:
    """{cell: kept?}: seed-mean uncut selection exact match at least chance + 0.30 (a cell with no selection score: dropped)."""
    cells = {}
    for p in probes_t.values():
        for c, v in p["selection"]["cells"].items():
            cells.setdefault(c, []).append(v)
    return {c: bool(np.mean([v["acc"] for v in vs]) >= vs[0]["chance"] + margin - EPS) for c, vs in cells.items()}


def build_predicted(d, model: str, smoke: bool, allow_partial: bool) -> dict:
    prof = PROFILES[model]
    probes = load_probes(d, model, prof.tasks, prof.seeds)
    complete = all(len(probes.get(t, {})) == len(prof.seeds) for t in (prof.tasks if not smoke else (COPY,)))
    if not complete and not (smoke or allow_partial):
        raise SystemExit("predict: probe files missing for " + str([(t, s) for t in prof.tasks for s in prof.seeds
                                                                    if s not in probes.get(t, {})]))
    depths = {}
    for t, bys in probes.items():
        for s, p in bys.items():
            for c, v in p["cells"].items():
                depths.setdefault(t, {}).setdefault(c, {})[str(s)] = {
                    "L_on_healed": v.get("L_on_healed"), "never": v.get("never"), "onset055": v.get("onset055")}
    return {"model": model, "complete": bool(complete), "n_layers": prof.n_layers, "depth_unit": "probe layer index (0 = embeddings)",
            "depths": depths, "kept": {t: kept_cells(b) for t, b in probes.items()}}


def sha_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_predict(args, smoke: bool) -> int:
    d = dirs("qwen17", smoke)
    path = d.root / "predicted.json"
    if path.exists():
        print(f"predict: {path} exists (sha256 {sha_of(path)}); not rewritten", flush=True)
        return 0
    if (d.out / "cutscore.json").exists():
        raise SystemExit("predict: cut scores already exist; predicted.json can no longer be written")
    doc = build_predicted(d, "qwen17", smoke, args.allow_partial)
    doc["ts"] = now()
    d.root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    h = sha_of(path)
    (d.root / "predicted.sha256").write_text(h + "\n")
    stream(d.root, f"predicted.json sha256 {h}")
    print(f"predict: wrote {path} sha256 {h}", flush=True)
    return 0


def check_predicted(root: Path) -> str:
    """The sha256 of predicted.json, or SystemExit unless the file exists and matches its recorded hash."""
    path, shaf = root / "predicted.json", root / "predicted.sha256"
    if not path.exists() or not shaf.exists():
        raise SystemExit("cutscore refused: predicted.json is missing (run --stage predict first)")
    h = sha_of(path)
    if shaf.read_text().strip() != h:
        raise SystemExit("cutscore refused: predicted.json differs from the hash recorded when it was written")
    log = root / "stream.log"
    if log.exists() and f"predicted.json sha256 {h}" not in log.read_text():
        raise SystemExit("cutscore refused: the stream log has no record of this predicted.json hash")
    return h


# ---------------------------------------------------------------- stage: cutscore

def cutscore_path(d) -> Path:
    return d.out / "cutscore.json"


def stage_cutscore(ctx, d, args) -> int:
    cfg = ctx.cfg
    h = check_predicted(d.root)
    runs = ac.read_merged("runs", cfg.sfx, "runs")
    path = cutscore_path(d)
    doc = ac.read_json(path) or {"predicted_sha256": h, "models": {}}
    if doc.get("predicted_sha256") != h:
        raise SystemExit("cutscore: existing results were made against another predicted.json")
    ledger = d.root / "test_ledger.jsonl"
    for task in cfg.tasks:
        recs = ctx.records(task, "test")
        for k in (0, KCUT):
            for seed in PROFILES["qwen17"].seeds:
                row = winner_row(runs, task, k, seed)
                key = f"{task}|k{k}|s{seed}"
                if row is None or key in doc["models"]:
                    continue
                model, handle = ac.rebuild_healed(ctx, row)
                adapters.merge_lora(handle)
                try:
                    cells, ok = exact_by_cell(ctx, model, task, recs)
                finally:
                    del model
                    ac.release(ctx)
                doc["models"][key] = {"task": task, "k": k, "seed": seed, "lr": row["lr"], "cells": cells, "correct": ok,
                                      "n": len(recs), "ts": now()}
                ac.write(path, doc)
                if cfg.ledger:
                    with ledger.open("a") as fh:
                        fh.write(json.dumps({"key": key, "task": task, "k": k, "seed": seed, "n": len(recs),
                                             "accuracy": float(np.mean(list(ok.values()))), "ts": now()}) + "\n")
                print(f"cutscore {key}: {np.mean(list(ok.values())):.3f}", flush=True)
    return 0


# ---------------------------------------------------------------- stage: patch

def decision_block(shares, thr: float = 0.5):
    """First block b whose share of pairs switched to the flip answer reaches `thr`, else 'never'. Also whether every later
    block stays at or above it (the rule's reading: patching there and later switches it, earlier does not)."""
    for b, s in enumerate(shares):
        if s is not None and s >= thr - EPS:
            return {"block": b, "stable": all(x is not None and x >= thr - EPS for x in shares[b:])}
    return {"block": "never", "stable": False}


def patched_preds(model, tok, task: str, recs: list[dict], positions, block: int, new_states) -> np.ndarray:
    """Option first-token predictions with the output of `block` replaced, at one position per record, by `new_states`
    (B, d). The other blocks and positions run as usual."""
    enc = measure._encode(tok, task, recs, model.device)
    pos = measure.abs_positions(enc["attention_mask"], positions)[:, 0]
    rows = torch.arange(len(recs), device=model.device)
    new = torch.as_tensor(new_states, device=model.device)

    def hook(m, inp, o):
        h = measure._first(o).clone()
        h[rows, pos] = new.to(h.dtype)
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    hd = model.model.layers[block].register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        ids = torch.tensor(spec.option_token_ids(tok, task), device=model.device)
        return logits[:, ids].argmax(-1).cpu().numpy()
    finally:
        hd.remove()


def twin_pairs(recs: list[dict], per_cell: int) -> dict:
    """{cell: [(base, flip)]} from the records' twin groups, the first `per_cell` by id; cells without twins are absent."""
    by = {}
    for r in recs:
        m = r["meta"]
        if m.get("pad") == "A" or m["twin_role"] not in ("base", "flip"):
            continue
        by.setdefault(m["twin_group"], {})[m["twin_role"]] = r
    out = {}
    for gid in sorted(by):
        p = by[gid]
        if "base" in p and "flip" in p:
            out.setdefault(p["base"]["rung"], []).append((p["base"], p["flip"]))
    return {c: v[:per_cell] for c, v in out.items()}


def stage_patch(ctx, d, args) -> int:
    cfg = ctx.cfg
    task, seed = COPY, 0
    path = d.out / "patch.json"
    doc = ac.read_json(path) or {"task": task, "seed": seed, "k": 0, "model": "uncut healed seed 0", "cells": {}}
    runs = ac.read_merged("runs", cfg.sfx, "runs")
    row = winner_row(runs, task, 0, seed)
    if row is None:
        print("patch: no healed k=0 seed-0 run", flush=True)
        return 1
    recs = [r for sp in ("discovery", "selection", "lrsel") for r in ctx.records(task, sp)]
    pairs = twin_pairs(recs, args.patch_pairs)
    if args.patch_cells:
        pairs = dict(list(pairs.items())[:args.patch_cells])
    model, handle = ac.rebuild_healed(ctx, row)
    adapters.merge_lora(handle)
    L, tok = len(model.model.layers), ctx.tokenizer
    blocks = list(range(0, L, args.patch_stride))
    try:
        for c, prs in pairs.items():
            if c in doc["cells"]:
                continue
            t0 = time.perf_counter()
            cnt = {s: np.zeros(len(blocks)) for s in PATCH_SITES}
            cln = {s: np.zeros(len(blocks)) for s in PATCH_SITES}
            n = n_clean = 0
            for i in range(0, len(prs), ac.EVAL_BS):
                chunk = prs[i:i + ac.EVAL_BS]
                br, fr = [p[0] for p in chunk], [p[1] for p in chunk]
                pos = {"last": [[-1]] * len(chunk),
                       "flip_after": [[r["meta"]["flip_after_tok"]] for r in br],
                       "final_binding": [[r["meta"]["final_binding_tok"]] for r in br]}
                if any(b["meta"]["prompt_tokens"] != f["meta"]["prompt_tokens"] for b, f in chunk):
                    raise SystemExit(f"patch: twins in cell {c} differ in token count")
                allpos = [[pos[s][j][0] for s in PATCH_SITES] for j in range(len(chunk))]
                st = measure.site_states(model, tok, task, fr, allpos, batch_size=ac.EVAL_BS, dtype=np.float32)
                pb = measure.option_logits(model, tok, task, br, ac.EVAL_BS).argmax(-1)
                pf = measure.option_logits(model, tok, task, fr, ac.EVAL_BS).argmax(-1)
                clean = (pb == np.array([r["answer"] for r in br])) & (pf == np.array([r["answer"] for r in fr]))
                fans = np.array([r["answer"] for r in fr])
                n += len(chunk)
                n_clean += int(clean.sum())
                for j, s in enumerate(PATCH_SITES):
                    for bi, b in enumerate(blocks):
                        pr = patched_preds(model, tok, task, br, pos[s], b, st[:, j, b + 1])
                        sw = pr == fans
                        cnt[s][bi] += sw.sum()
                        cln[s][bi] += (sw & clean).sum()
            res = {"n_pairs": n, "n_clean": n_clean, "blocks": blocks, "seconds": time.perf_counter() - t0, "sites": {}}
            for s in PATCH_SITES:
                share = (cnt[s] / n).tolist()
                share_c = (cln[s] / max(n_clean, 1)).tolist() if n_clean else [None] * len(blocks)
                res["sites"][s] = {"share": share, "share_clean": share_c, "decision": decision_block(share),
                                   "decision_clean": decision_block(share_c)}
            doc["cells"][c] = res
            ac.write(path, doc)
            print(f"patch {c}: {n} pairs ({n_clean} clean), decision block "
                  f"{ {s: res['sites'][s]['decision']['block'] for s in PATCH_SITES} } ({res['seconds']:.0f}s)", flush=True)
    finally:
        del model
        ac.release(ctx)
    return 0


# ---------------------------------------------------------------- bars: pure functions

def label_contrast(delta, lo, hi, seed_deltas) -> str:
    """The doc's labels. Effect: delta >= 3, interval excludes 0, every seed positive. Reverse: the mirror. No effect: the
    interval lies inside [-1, +1]. Otherwise inconclusive."""
    if delta is None:
        return "not evaluable"
    if lo is None or hi is None:
        return "inconclusive"
    if delta >= EFFECT - EPS and lo > 0 and all(x > 0 for x in seed_deltas):
        return "effect"
    if delta <= -EFFECT + EPS and hi < 0 and all(x < 0 for x in seed_deltas):
        return "reverse effect"
    if lo >= -NO_EFFECT - EPS and hi <= NO_EFFECT + EPS:
        return "no effect"
    return "inconclusive"


def _entry(dt: dict, cell: str, seed):
    e = (dt.get(cell) or {}).get(seed)
    return None if e is None or e.get("d") is None else e


def contrast(dt: dict, terms, seeds, dropped=frozenset()) -> dict:
    """A contrast between cells. `dt[cell][seed]` = {"d": depth or None, "boot": array or None}. Each term is (plus cells, minus
    cells): its value is the mean depth of the first minus the mean of the second; the contrast averages its usable terms
    (both ends kept and measured in every seed), then the seeds. Interval: percentiles of the same average over bootstrap
    replicates (each replicate index across seeds; replicates with a missing depth are left out)."""
    use, skipped = [], []
    for t in terms:
        cells = list(t[0]) + list(t[1])
        ok = all(c not in dropped and all(_entry(dt, c, s) is not None for s in seeds) for c in cells)
        (use if ok else skipped).append(t)
    base = {"n_terms": len(use), "n_skipped": len(skipped), "skipped": [list(map(list, t)) for t in skipped], "seeds": list(seeds)}
    if not use:
        return {**base, "delta": None, "lo": None, "hi": None, "seed_deltas": [], "label": "not evaluable", "reps": None,
                "ci_source": None}

    def val(cells, s, key):
        return np.mean([_entry(dt, c, s)[key] for c in cells], axis=0)

    seed_deltas = [float(np.mean([val(p, s, "d") - val(m, s, "d") for p, m in use])) for s in seeds]
    delta = float(np.mean(seed_deltas))
    reps, lo, hi = None, None, None
    boots = [_entry(dt, c, s)["boot"] for t in use for c in list(t[0]) + list(t[1]) for s in seeds]
    if all(b is not None for b in boots):
        nb = min(len(b) for b in boots)
        trim = lambda cells, s: np.mean([np.asarray(_entry(dt, c, s)["boot"][:nb], dtype=float) for c in cells], axis=0)  # noqa: E731
        per_seed = [np.mean([trim(p, s) - trim(m, s) for p, m in use], axis=0) for s in seeds]
        reps = np.mean(per_seed, axis=0)
        reps = reps[np.isfinite(reps)]
        if len(reps) >= MIN_REFITS:
            lo, hi = float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
    return {**base, "delta": delta, "lo": lo, "hi": hi, "seed_deltas": seed_deltas, "label": label_contrast(delta, lo, hi, seed_deltas),
            "reps": reps, "ci_source": None}


def s4(c_serial: dict, c_comb: dict) -> dict:
    """Serial steps cost more than combining only if (F1 L4 - L1) - (F4 n4 - n1) >= 3 with its interval excluding 0 and both
    families have all their end levels kept. One passing and one inconclusive bar is not a difference."""
    if c_serial["delta"] is None or c_comb["delta"] is None:
        return {"delta": None, "lo": None, "hi": None, "label": "not evaluable"}
    delta = c_serial["delta"] - c_comb["delta"]
    lo = hi = None
    ra, rb = c_serial.get("reps"), c_comb.get("reps")
    if ra is not None and rb is not None:
        n = min(len(ra), len(rb))
        diff = ra[:n] - rb[:n]
        if n >= MIN_REFITS:
            lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
    kept = c_serial["n_skipped"] == 0 and c_comb["n_skipped"] == 0
    ok = delta >= EFFECT - EPS and lo is not None and lo > 0 and kept
    return {"delta": float(delta), "lo": lo, "hi": hi, "ends_kept": kept,
            "label": "serial steps cost more than combining" if ok else "not shown"}


def avg_ranks(x) -> np.ndarray:
    return heal._ranks(x)


def spearman(x, y) -> float:
    return heal.spearman(list(x), list(y))


def r_value(acc12: float, acc0: float, chance: float, clip: bool = True):
    """(acc12 - chance) / (acc0 - chance); clipped to 0..1 as C1's R; None when the uncut accuracy is not above chance."""
    if acc0 - chance <= EPS:
        return None
    r = (acc12 - chance) / (acc0 - chance)
    return float(min(1.0, max(0.0, r))) if clip else float(r)


def s5(depth: dict, r: dict, lengths: dict, kept: dict, rho_boot=None) -> dict:
    """S5 over F1's main cells. `depth` and `r` are {cell: value} (seed-mean depth, C1-style R), `kept` {cell: bool} (False:
    dropped), `lengths` {cell: step count} for the within-length pooled rho. Needs at least 8 kept cells with a depth and an R;
    all R above 0.95 or below 0.05, or an R range under 0.10: uninformative. Passes if rho <= -0.6 and the interval's upper
    end is below 0. `rho_boot` is an array of replicate rhos (or None)."""
    cells = [c for c in depth if kept.get(c) and depth[c] is not None and r.get(c) is not None]
    out = {"n_cells": len(cells), "cells": cells, "rho": None, "lo": None, "hi": None, "pooled_within_length": None}
    if len(cells) < S5_MIN_CELLS:
        return {**out, "label": "not evaluable (fewer than 8 kept cells)"}
    rr = [r[c] for c in cells]
    out["r_range"] = float(max(rr) - min(rr))
    if all(v > S5_CEIL for v in rr) or all(v < S5_FLOOR for v in rr) or max(rr) - min(rr) < S5_SPREAD:
        return {**out, "label": "uninformative"}
    rho = spearman([depth[c] for c in cells], rr)
    out["rho"] = None if math.isnan(rho) else rho
    if rho_boot is not None:
        b = np.asarray(rho_boot, dtype=float)
        b = b[np.isfinite(b)]
        if len(b) >= MIN_REFITS:
            out["lo"], out["hi"] = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    # report only: depth and R compared inside each step count, pooled through within-length centred ranks
    xs, ys = [], []
    for L in sorted({lengths[c] for c in cells}):
        grp = [c for c in cells if lengths[c] == L]
        if len(grp) >= 2:
            xs += list(avg_ranks([depth[c] for c in grp]) - np.mean(avg_ranks([depth[c] for c in grp])))
            ys += list(avg_ranks([r[c] for c in grp]) - np.mean(avg_ranks([r[c] for c in grp])))
    if len(xs) >= 3 and np.std(xs) > 0 and np.std(ys) > 0:
        out["pooled_within_length"] = float(np.corrcoef(xs, ys)[0, 1])
    if out["rho"] is None:
        return {**out, "label": "not evaluable (constant depth)"}
    ok = rho <= S5_RHO + EPS and out["hi"] is not None and out["hi"] < 0
    return {**out, "label": "depth predicts survival on controlled inputs" if ok else "not shown"}


def s6(qwen: dict, smol: dict, n_layers_smol: int = 24, n_layers_qwen: int = 28) -> dict:
    """Directional agreement: Qwen's contrast passes 'effect' and the single-seed SmolLM2 contrast has the same sign, is at
    least 2.6 blocks and its interval excludes 0. Also the contrasts as a fraction of model depth."""
    out = {"qwen_label": qwen.get("label"), "qwen_delta": qwen.get("delta"), "smol_delta": smol.get("delta"),
           "smol_lo": smol.get("lo"), "smol_hi": smol.get("hi")}
    if qwen.get("delta") is not None:
        out["qwen_fraction"] = qwen["delta"] / n_layers_qwen
    if smol.get("delta") is not None:
        out["smol_fraction"] = smol["delta"] / n_layers_smol
    if qwen.get("delta") is None or smol.get("delta") is None:
        return {**out, "label": "not evaluable"}
    sd, lo, hi = smol["delta"], smol.get("lo"), smol.get("hi")
    same = sd * qwen["delta"] > 0
    excl = lo is not None and hi is not None and (lo > 0 or hi < 0)
    ok = qwen["label"] in ("effect", "reverse effect") and same and abs(sd) >= S6_MIN_BLOCKS - EPS and excl
    return {**out, "label": "directional agreement" if ok else "no directional agreement"}


# ---------------------------------------------------------------- table: assembling the numbers

def depth_table(probes_t: dict, source: str = "auto") -> tuple[dict, str]:
    """{cell: {seed: {"d", "boot"}}} from a task's probe files; the replicates are the refit bootstrap when every file has at
    least MIN_REFITS of them (source 'refit'), else the OOF bootstrap ('oof'). A depth that never reaches threshold counts as
    the number of layers (n_layers + 1), D1's convention."""
    if source == "auto":
        source = "refit" if all(len((c.get("refit") or {}).get("depth", [])) >= MIN_REFITS
                                for p in probes_t.values() for c in p["cells"].values() if "curve" in c) else "oof"
    dt = {}
    for s, p in probes_t.items():
        for c, v in p["cells"].items():
            if "curve" not in v:
                continue
            d = v["L_on_healed"]
            src = v.get("refit") if source == "refit" else v["boot"]
            boot = None if not src else np.asarray([np.nan if x is None else x for x in src["depth"]], dtype=float)
            dt.setdefault(c, {})[s] = {"d": None if d is None else float(d), "boot": boot}
    return dt, source


def f1_terms() -> dict:
    pos = ST.F1_POS
    t = {
        "S1_avg": [([f"L4_{p}"], [f"L1_{p}"]) for p in pos],
        "S2_avg": [([f"L{L}_far"], [f"L{L}_near"]) for L in (1, 2, 3, 4)],
        "S2_middle_penalty_avg": [([f"L{L}_middle"], [f"L{L}_far", f"L{L}_near"]) for L in (1, 2, 3, 4)],
        "S1_unit_avg": [([f"L4_{p}_unit"], [f"L1_{p}_unit"]) for p in pos],
        "S2_unit_avg": [([f"L{L}_far_unit"], [f"L{L}_near_unit"]) for L in (1, 4)],
        "S2_pad": [(["L1_far"], ["L1_far_pad"])],
        "S2_pad_keep": [(["L1_far"], ["L1_far_pad_keep"])],
    }
    for p in pos:
        t[f"S1_{p}"] = [([f"L4_{p}"], [f"L1_{p}"])]
    for L in (1, 2, 3, 4):
        t[f"S2_L{L}"] = [([f"L{L}_far"], [f"L{L}_near"])]
        t[f"S2_middle_penalty_L{L}"] = [([f"L{L}_middle"], [f"L{L}_far", f"L{L}_near"])]
    for L in (1, 4):
        t[f"S2_unit_L{L}"] = [([f"L{L}_far_unit"], [f"L{L}_near_unit"])]
    return t


def f4_terms() -> dict:
    t = {"S3_n_avg": [([f"n4_{p}"], [f"n1_{p}"]) for p in ("far", "near")],
         "S3_pos_avg": [([f"n{n}_far"], [f"n{n}_near"]) for n in (1, 2, 3, 4)]}
    for p in ("far", "near"):
        t[f"S3_n4-n1_{p}"] = [([f"n4_{p}"], [f"n1_{p}"])]
    for n in (1, 2, 3, 4):
        t[f"S3_far-near_n{n}"] = [([f"n{n}_far"], [f"n{n}_near"])]
    return t


def f2_terms() -> dict:
    return {"S1_ops": [(["s4"], ["s1"])]}


def correct_by_cell(doc: dict, task: str, k: int, seeds, recs: list[dict]) -> dict:
    """{cell: {seed: bool array over the cell's test records}} from the cutscore file."""
    cell_ids = {}
    for r in recs:
        cell_ids.setdefault(r["rung"], []).append(r["id"])
    out = {}
    for s in seeds:
        m = doc["models"].get(f"{task}|k{k}|s{s}")
        if m:
            for c, ids in cell_ids.items():
                out.setdefault(c, {})[s] = np.asarray([m["correct"][i] for i in ids], dtype=float)
    return out


def survival(doc: dict, task: str, recs: list[dict], seeds=SEEDS) -> dict:
    """Per cell: test chance, per-seed uncut and cut accuracy, per-seed R (clipped and not), and the seed-mean-accuracy R
    (C1's: the headline)."""
    c0, c12 = correct_by_cell(doc, task, 0, seeds, recs), correct_by_cell(doc, task, KCUT, seeds, recs)
    ans = {}
    for r in recs:
        ans.setdefault(r["rung"], []).append(r["answer"])
    out = {}
    for c, a in ans.items():
        ch = spec.chance(a)
        have = [s for s in seeds if s in c0.get(c, {}) and s in c12.get(c, {})]
        if not have:
            continue
        a0 = {s: float(c0[c][s].mean()) for s in have}
        a12 = {s: float(c12[c][s].mean()) for s in have}
        m0, m12 = float(np.mean(list(a0.values()))), float(np.mean(list(a12.values())))
        out[c] = {"chance": ch, "seeds": have, "acc0": a0, "acc12": a12,
                  "R_seed": {s: r_value(a12[s], a0[s], ch) for s in have},
                  "R_seed_unclipped": {s: r_value(a12[s], a0[s], ch, clip=False) for s in have},
                  "acc0_mean": m0, "acc12_mean": m12, "R": r_value(m12, m0, ch), "R_unclipped": r_value(m12, m0, ch, clip=False)}
    return out


def s5_bootstrap(doc: dict, recs: list[dict], depth_boot: dict, cells: list[str], n_boot: int = 1000, seed: int = 0):
    """Replicate rhos for S5: resample each cell's test templates (the same draw for the uncut and cut scores, every seed),
    recompute C1-style R, pair it with replicate b of the seed-mean depth, take Spearman across the cells."""
    rng = np.random.default_rng(seed)
    c0, c12 = correct_by_cell(doc, COPY, 0, SEEDS, recs), correct_by_cell(doc, COPY, KCUT, SEEDS, recs)
    info = {}
    for c in cells:
        rs = [r for r in recs if r["rung"] == c]
        g = np.asarray([r["template_id"] for r in rs])
        ug = np.unique(g)
        info[c] = (np.asarray([r["answer"] for r in rs]), [np.flatnonzero(g == u) for u in ug])
    nb = min(len(v) for v in depth_boot.values()) if depth_boot else 0
    if not nb:
        return None
    out = []
    for b in range(n_boot):
        d, rr = [], []
        for c in cells:
            a, groups = info[c]
            rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            ch = spec.chance(a[rows].tolist())
            ss = [s for s in SEEDS if s in c0[c] and s in c12[c]]
            m0 = np.mean([c0[c][s][rows].mean() for s in ss])
            m12 = np.mean([c12[c][s][rows].mean() for s in ss])
            r = r_value(float(m12), float(m0), ch)
            d.append(depth_boot[c][b % nb])
            rr.append(r)
        if any(v is None or not np.isfinite(v) for v in d + rr):
            continue
        rho = spearman(d, rr)
        if not math.isnan(rho):
            out.append(rho)
    return np.asarray(out)


def seed_mean_boot(dt_cell: dict) -> np.ndarray | None:
    bs = [e["boot"] for e in dt_cell.values() if e.get("boot") is not None]
    if not bs or len(bs) < len(dt_cell):
        return None
    nb = min(len(b) for b in bs)
    return np.mean([np.asarray(b[:nb], dtype=float) for b in bs], axis=0)


def fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def contrast_row(name: str, c: dict) -> str:
    ci = "-" if c["lo"] is None else f"[{c['lo']:.2f}, {c['hi']:.2f}]"
    sd = ", ".join(f"{x:.1f}" for x in c["seed_deltas"])
    return f"| {name} | {fmt(c['delta'])} | {ci} | {sd or '-'} | {c['n_terms']}/{c['n_terms'] + c['n_skipped']} | **{c['label']}** |"


def cross_model_rho() -> dict:
    """The post hoc cross-model rank correlation of healed depth (C1/D3 files), all learned tasks and with layer-1 ties named."""
    e1, d3 = Path("var/results/atlas3/e1.json"), Path("var/results/atlas3/d3/summary.json")
    if not (e1.exists() and d3.exists()):
        return {"available": False}
    q = {t["task"]: t["L_on_healed"] for t in json.loads(e1.read_text())["1.7B"]["tasks"] if t.get("L_on_healed") is not None}
    sm = {r["task"]: r for r in json.loads(d3.read_text())["rows"]}
    both = [t for t in q if t in sm and not sm[t]["dropped"] and sm[t].get("L_on_healed") is not None]
    x, y = [q[t] for t in both], [sm[t]["L_on_healed"] for t in both]
    ties = sorted(t for t in both if q[t] <= 1 or sm[t]["L_on_healed"] <= 1)
    non = [t for t in both if t not in ties]
    return {"available": True, "n": len(both), "rho": spearman(x, y), "layer1_tied_tasks": ties,
            "n_without_layer1": len(non), "rho_without_layer1": spearman([q[t] for t in non], [sm[t]["L_on_healed"] for t in non])}


def build_summary(d, smoke: bool) -> dict:
    qp = {t: load_probes(d, "qwen17", [t], SEEDS).get(t, {}) for t in TASKS}
    sd = dirs("smol17", smoke)
    sp = load_probes(sd, "smol17", [COPY], (0,)).get(COPY, {})
    cs = ac.read_json(cutscore_path(d))
    out = {"smoke": smoke, "ts": now(), "predicted_sha256": (d.root / "predicted.sha256").read_text().strip()
           if (d.root / "predicted.sha256").exists() else None, "bars": {}, "report_only": {}}
    dts, kept = {}, {}
    for t in TASKS:
        if qp[t]:
            dts[t] = depth_table(qp[t])
            kept[t] = kept_cells(qp[t])
    dropped = {t: {c for c, k in kept[t].items() if not k} for t in kept}
    out["kept"] = {t: kept[t] for t in kept}
    seeds = lambda t: sorted(qp[t])                                                           # noqa: E731
    cons = {}

    def run(name, fam_task, terms, store="bars"):
        if fam_task not in dts:
            return None
        c = contrast(dts[fam_task][0], terms, seeds(fam_task), dropped.get(fam_task, set()))
        c["ci_source"] = dts[fam_task][1]
        cons[name] = c
        out[store][name] = {k: v for k, v in c.items() if k != "reps"}
        return c

    for name, terms in f1_terms().items():
        run(name, COPY, terms, "bars" if not (name.startswith(("S1_unit", "S2_unit", "S2_pad", "S2_middle"))) else "report_only")
    for name, terms in f2_terms().items():
        run(name, OPS, terms)
    for name, terms in f4_terms().items():
        run(name, DESC, terms)
    if "S1_avg" in cons and "S3_n_avg" in cons:
        out["bars"]["S4"] = s4(cons["S1_avg"], cons["S3_n_avg"])
    # S5
    out["bars"]["S5"] = {"label": "not run (no cut scores)"}
    if cs and COPY in dts:
        recs = ctx_records(COPY, "test", smoke)
        sv = survival(cs, COPY, recs)
        dt = dts[COPY][0]
        main_cells = [c for c in ST.F1_MAIN if c in dt]
        depth = {c: float(np.mean([e["d"] for e in dt[c].values()])) if all(e["d"] is not None for e in dt[c].values()) else None
                 for c in main_cells}
        rmap = {c: (sv.get(c) or {}).get("R") for c in main_cells}
        lens = {c: int(c[1]) for c in main_cells}
        keptmap = {c: kept[COPY].get(c, False) for c in main_cells}
        boots = {c: seed_mean_boot(dt[c]) for c in main_cells}
        use = [c for c in main_cells if keptmap[c] and depth[c] is not None and rmap[c] is not None and boots[c] is not None]
        rb = s5_bootstrap(cs, recs, {c: boots[c] for c in use}, use, 200 if smoke else 1000) if len(use) >= S5_MIN_CELLS else None
        out["bars"]["S5"] = s5(depth, rmap, lens, keptmap, rb)
        out["bars"]["S5"]["depth_source"] = dts[COPY][1]
        out["survival"] = {t: survival(cs, t, ctx_records(t, "test", smoke)) for t in TASKS if any(k.startswith(t + "|") for k in cs["models"])}
        out["survival_cells"] = {c: {"depth_mean": depth.get(c), "kept": keptmap.get(c), **{k: v for k, v in (sv.get(c) or {}).items()}}
                                 for c in main_cells}
    # S6
    if sp and "S1_avg" in cons:
        sdt, ssrc = depth_table(sp)
        skept = {c for c, k in kept_cells(sp).items() if not k}
        s6_out = {}
        for name, key in (("L4-L1", "S1_avg"), ("far-near", "S2_avg")):
            cm = contrast(sdt, f1_terms()[key], [0], skept)
            cm["ci_source"] = ssrc
            s6_out[name] = s6(cons[key], cm)
            s6_out[name]["smol_contrast"] = {k: v for k, v in cm.items() if k != "reps"}
        out["bars"]["S6"] = s6_out
        out["bars"]["S6_cross_model_rho_post_hoc"] = cross_model_rho()
    # report only
    ro = out["report_only"]
    for t in TASKS:
        if not qp[t]:
            continue
        ro.setdefault("mlp", {})[t] = {c: {"linear_depth": mean_over(qp[t], c, "L_on_healed"), "mlp_depth": mean_over(qp[t], c, "mlp.L_on_healed"),
                                           "gap": mean_over(qp[t], c, "mlp.gap"), "linear_onset055": mean_over(qp[t], c, "onset055"),
                                           "mlp_onset055": mean_over(qp[t], c, "mlp.onset055"),
                                           "selectivity_at_onset": mean_over(qp[t], c, "mlp.selectivity_at_onset"),
                                           "selectivity_max": mean_over(qp[t], c, "mlp.selectivity_max")} for c in cells_of(t)}
    if qp[COPY]:
        ro["sites"] = {s: p.get("sites", {}) for s, p in qp[COPY].items()}
    pt = ac.read_json(d.out / "patch.json")
    if pt:
        ro["patch"] = {c: {s: v["decision"] for s, v in r["sites"].items()} | {"n_pairs": r["n_pairs"], "n_clean": r["n_clean"]}
                       for c, r in pt["cells"].items()}
    if qp[DESC]:
        ro["f4_avoid_borderline"] = f4_avoid(qp[DESC], smoke)
    out["ci_sources"] = {t: dts[t][1] for t in dts}
    return out


def ctx_records(task: str, split: str, smoke: bool) -> list[dict]:
    recs = spec.read_records(task, split)
    return smoke_subset(recs, SMOKE_PER_CELL[split]) if smoke else recs


def mean_over(probes_t: dict, cell: str, path: str):
    vals = []
    for p in probes_t.values():
        v = p["cells"].get(cell)
        for k in path.split("."):
            v = v.get(k) if isinstance(v, dict) else None
        vals.append(v)
    vals = [x for x in vals if x is not None]
    return float(np.mean(vals)) if vals and len(vals) == len(probes_t) else None


def f4_avoid(probes_t: dict, smoke: bool) -> dict:
    """F4 depth and selection accuracy on items that show none of the borderline truth-table features."""
    recs = {r["id"]: r for r in spec.read_records(DESC, "selection")}
    out = {}
    for c in cells_of(DESC):
        dep = [p["cells"].get(c, {}).get("avoid", {}).get("L_on_healed") for p in probes_t.values()]
        ok, okall = [], []
        for p in probes_t.values():
            for i, v in p["selection"]["correct"].items():
                r = recs.get(i)
                if r and r["rung"] == c:
                    okall.append(v)
                    if not (set(r["meta"]["features"]) & set(BORDERLINE)):
                        ok.append(v)
        out[c] = {"depth_avoid": [x for x in dep], "acc_avoid": float(np.mean(ok)) if ok else None, "n_avoid": len(ok),
                  "acc_all": float(np.mean(okall)) if okall else None, "n_all": len(okall)}
    return out


def render_tables(s: dict) -> str:
    b, ro = s["bars"], s["report_only"]
    out = [f"# Steps study v2: bars S1-S6{' (smoke)' if s['smoke'] else ''}", "",
           "Depth = `L_on_healed` (probe layer index, 0 = embeddings; a depth that never reaches its threshold counts as the "
           "number of layers + 1) per cell; delta = difference in blocks averaged over the uncut seeds. Interval: 95% "
           f"percentiles of the bootstrap over template groups (source per task: {s.get('ci_sources')}). Labels per the doc: "
           "effect (delta >= 3, interval excludes 0, all seeds same sign), reverse effect, no effect (interval inside [-1, +1]), "
           "else inconclusive. `terms` = usable / all pairs of cells (a pair needs both ends kept).", "",
           f"predicted.json sha256: {s.get('predicted_sha256')}", ""]
    head = ["| contrast | delta | 95% interval | seed deltas | terms | label |", "|---|---|---|---|---|---|"]
    for title, keys in (("S1 steps", [k for k in b if k.startswith("S1")]), ("S2 distance", [k for k in b if k.startswith("S2")]),
                        ("S3 combining evidence (F4)", [k for k in b if k.startswith("S3")])):
        out += [f"## {title}", ""] + head + [contrast_row(k, b[k]) for k in keys] + [""]
    if "S4" in b:
        x = b["S4"]
        out += ["## S4 serial vs combining", "", f"(F1 L4-L1) - (F4 n4-n1) = {fmt(x['delta'])}, interval "
                f"[{fmt(x.get('lo'))}, {fmt(x.get('hi'))}]: **{x['label']}**", ""]
    x = b["S5"]
    out += ["## S5 depth predicts survival (F1 main cells)", "",
            f"kept cells {x.get('n_cells')}; Spearman rho(depth, R) = {fmt(x.get('rho'))}, interval [{fmt(x.get('lo'))}, "
            f"{fmt(x.get('hi'))}]; R range {fmt(x.get('r_range'))}; within-length pooled rho (report only) "
            f"{fmt(x.get('pooled_within_length'))}: **{x['label']}**. The conclusion is for this cut (k = 12, shared order) and "
            "this repair budget (q/v LoRA rank 10, 300 steps), across conditions of one task family.", ""]
    if s.get("survival_cells"):
        out += ["| cell | kept | depth (seed mean) | chance | uncut acc | cut acc | R | R unclipped | R per seed |",
                "|---|---|---|---|---|---|---|---|---|"]
        for c, v in s["survival_cells"].items():
            out.append(f"| {c} | {v.get('kept')} | {fmt(v.get('depth_mean'))} | {fmt(v.get('chance'))} | {fmt(v.get('acc0_mean'))} | "
                       f"{fmt(v.get('acc12_mean'))} | {fmt(v.get('R'))} | {fmt(v.get('R_unclipped'))} | "
                       f"{', '.join(fmt(r) for r in (v.get('R_seed') or {}).values())} |")
        out.append("")
    if "S6" in b:
        out += ["## S6 cross-model (wording only)", ""]
        for name, v in b["S6"].items():
            out.append(f"- {name}: Qwen {fmt(v['qwen_delta'])} blocks ({v['qwen_label']}, {fmt(v.get('qwen_fraction'), 3)} of depth), "
                       f"SmolLM2 {fmt(v['smol_delta'])} blocks (interval [{fmt(v['smol_lo'])}, {fmt(v['smol_hi'])}], "
                       f"{fmt(v.get('smol_fraction'), 3)} of depth): **{v['label']}** (needs >= {S6_MIN_BLOCKS:.1f} blocks)")
        m = b.get("S6_cross_model_rho_post_hoc", {})
        if m.get("available"):
            out.append(f"- post hoc cross-model rho over {m['n']} tasks both models learned: {fmt(m['rho'])}; without the "
                       f"layer-1 ties ({len(m['layer1_tied_tasks'])} tasks: {', '.join(m['layer1_tied_tasks'])}): "
                       f"{fmt(m['rho_without_layer1'])} over {m['n_without_layer1']}")
        out.append("")
    out += ["## Report only", "", "### Uncut selection accuracy and dropped cells (cell kept if exact match >= chance + 0.30)", ""]
    for t, k in s.get("kept", {}).items():
        out.append(f"- {t}: dropped {[c for c, v in k.items() if not v] or 'none'}")
    for t, cells in ro.get("mlp", {}).items():
        out += ["", f"### MLP probe against linear probe, {t} (seed mean; gap = linear - MLP onset, the assembly time)", "",
                "| cell | linear depth | MLP depth | gap | linear onset .55 | MLP onset .55 | selectivity at onset | max selectivity |",
                "|---|---|---|---|---|---|---|---|"]
        out += [f"| {c} | {fmt(v['linear_depth'])} | {fmt(v['mlp_depth'])} | {fmt(v['gap'])} | {fmt(v['linear_onset055'])} | "
                f"{fmt(v['mlp_onset055'])} | {fmt(v['selectivity_at_onset'])} | {fmt(v['selectivity_max'])} |" for c, v in cells.items()]
    if ro.get("sites"):
        out += ["", "### Probe sites (F1 twin pairs; linear onset at 0.55 / half-peak rule, seed mean)", "",
                "| cell | after differing link | final binding | query token | last token (cell depth) |", "|---|---|---|---|---|"]
        cells = sorted({c for v in ro["sites"].values() for c in v}, key=lambda c: ST.F1_CELLS.index(c) if c in ST.F1_CELLS else 99)
        for c in cells:
            def site(n):
                vs = [v[c][n] for v in ro["sites"].values() if c in v and n in v[c]]
                return "-" if not vs else f"{fmt(float(np.mean([x['onset055'] for x in vs if x['onset055'] is not None])) if any(x['onset055'] is not None for x in vs) else None, 1)} / {fmt(vs[0]['L_on_peak'])}"
            out.append(f"| {c} | {site('flip_after')} | {site('final_binding')} | {site('query')} | "
                       f"{fmt(((s.get('survival_cells') or {}).get(c) or {}).get('depth_mean'))} |")
    if ro.get("patch"):
        out += ["", "### Twin patching (uncut seed-0 model; first block whose patch switches >= half the pairs; 'never' if none)", "",
                "| cell | pairs (clean) | last token | after differing link | final binding |", "|---|---|---|---|---|"]
        for c, v in ro["patch"].items():
            out.append(f"| {c} | {v['n_pairs']} ({v['n_clean']}) | {v['last']['block']} | {v['flip_after']['block']} | {v['final_binding']['block']} |")
    if ro.get("f4_avoid_borderline"):
        out += ["", "### F4 on items showing none of the 8 borderline truth-table features", "",
                "| cell | depth (avoid) | acc (avoid) | n | acc (all) | n |", "|---|---|---|---|---|---|"]
        for c, v in ro["f4_avoid_borderline"].items():
            out.append(f"| {c} | {', '.join(fmt(x) for x in v['depth_avoid'])} | {fmt(v['acc_avoid'])} | {v['n_avoid']} | "
                       f"{fmt(v['acc_all'])} | {v['n_all']} |")
    out.append("")
    return "\n".join(out)


def stage_table(args, smoke: bool) -> int:
    d = dirs("qwen17", smoke)
    s = build_summary(d, smoke)
    ac.write(d.root / "summary.json", s)
    md = render_tables(s)
    (d.root / "tables.md").write_text(md)
    print(md, flush=True)
    return 0


# ---------------------------------------------------------------- stage: project

C1_RUNS = Path("var/results/atlas3/c1/runs.shard0of1.json")
C1_ANCHORS = ("arith", "code_output", "hard_hops")       # C1 heals of 86, 110 and 685 prompt tokens, A40-class card


def c1_heal_seconds(task_tokens: float, k: int, c1_runs: dict, tokens: dict) -> float:
    """C1's seconds per heal (training + evaluations + scoring) at k, interpolated linearly in mean prompt tokens."""
    pts = []
    for t in C1_ANCHORS:
        rs = [r["timings"]["total_seconds"] + r["timings"]["score_seconds"] for r in c1_runs.values()
              if r["task"] == t and r["k"] == k]
        if rs:
            pts.append((tokens[t], float(np.mean(rs))))
    pts.sort()
    xs, ys = zip(*pts)
    if task_tokens <= xs[0]:
        return ys[0] * task_tokens / xs[0] if False else ys[0]
    return float(np.interp(task_tokens, xs, ys))


def stage_project(args, smoke: bool) -> int:
    """Heal hours for the full run, from the heals finished so far scaled against C1's per-heal times (the same card)."""
    d = dirs("qwen17", smoke)
    c1 = json.loads(C1_RUNS.read_text())["runs"]
    c1_tok = json.loads(Path("var/results/atlas3/c1/probe.json").read_text())["mean_tokens"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    tokens = dict(c1_tok)
    for t in TASKS:
        recs = spec.read_records(t, "train")[:100]
        tokens[t] = float(np.mean([len(tok(spec.build_prompt(tok, t, r["prompt"]), add_special_tokens=False)["input_ids"]) for r in recs]))
    est = {(t, k): c1_heal_seconds(tokens[t], k, c1, tokens) for t in TASKS for k in (0, KCUT)}
    runs = ac.read_merged("runs", "", "runs")
    ratios = []
    for r in runs.values():
        if (r["task"], r["k"]) in est:
            ratios.append((r["timings"]["total_seconds"] + r["timings"]["score_seconds"]) / est[(r["task"], r["k"])])
    f = float(np.median(ratios)) if ratios else 1.0
    n_heals = {(t, k): 4 for t in TASKS for k in (0, KCUT)}          # seed 0: two lrs; seeds 1 and 2: the winner
    heal_h = sum(est[tk] * n_heals[tk] for tk in est) * f / 3600
    smol = est[(COPY, 0)] * 2 * f * 1.25 / 3600                      # SmolLM2: two lrs; wider MLP, slower by an assumed 25%
    print(f"mean prompt tokens: { {t: round(tokens[t]) for t in TASKS} }", flush=True)
    print(f"C1-based seconds per heal: { {f'{t}|k{k}': round(v) for (t, k), v in est.items()} }", flush=True)
    print(f"measured/C1-estimate ratio {f:.2f} from {len(ratios)} finished heals" if ratios else "no heals yet: ratio 1.0 (C1 times)", flush=True)
    print(f"projected heal hours: Qwen {heal_h:.1f} ({sum(n_heals.values())} heals) + SmolLM2 {smol:.1f} (2 heals) = {heal_h + smol:.1f}",
          flush=True)
    stream(d.root, f"projection: heals {heal_h + smol:.1f} h (ratio {f:.2f}, {len(ratios)} finished heals)")
    return 0


# ---------------------------------------------------------------- main

STAGES = ["sweep", "probe", "predict", "cutscore", "patch", "table", "project"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--model", default="qwen17", choices=sorted(PROFILES))
    ap.add_argument("--seed", type=int, default=0, choices=SEEDS)
    ap.add_argument("--shard", default=None, help="I/N")
    ap.add_argument("--task", default=None, choices=TASKS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--boots", type=int, default=None, help="OOF bootstrap replicates (default 500; smoke 20)")
    ap.add_argument("--refits", type=int, default=None, help="bootstrap replicates with probe refits (default 20; smoke 3)")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("STEPS_JOBS") or min(8, os.cpu_count() or 1)))
    ap.add_argument("--patch-pairs", type=int, default=100)
    ap.add_argument("--patch-stride", type=int, default=1, help="patch every STRIDE-th block (the doc: every block)")
    ap.add_argument("--patch-cells", type=int, default=0, help="patch only the first N cells (smoke; 0: all)")
    ap.add_argument("--allow-partial", action="store_true", help="predict: write even if probe files are missing")
    args = ap.parse_args(argv)
    if args.stage in ("predict", "table", "project"):
        configure("qwen17", args.smoke)
        return {"predict": stage_predict, "table": stage_table, "project": stage_project}[args.stage](args, args.smoke)
    if args.model == "smol17" and args.stage not in ("sweep", "probe"):
        raise SystemExit("--model smol17 runs only the sweep and probe stages")
    d = configure(args.model, args.smoke)
    cfg = make_cfg(args.model, args.smoke, args.boots, args.refits)
    ctx = make_ctx(cfg)
    if args.model != "qwen17":
        C.check_blocks(ctx.n_layers, C.PROFILES[args.model])
    if args.stage == "sweep":
        return stage_sweep(ctx, args.seed, ac.parse_shard(args.shard))
    if args.stage == "probe":
        rc = 0
        for t in ([args.task] if args.task else cfg.tasks):
            rc |= stage_probe(ctx, d, t, args.seed, args)
        return rc
    if args.stage == "cutscore":
        return stage_cutscore(ctx, d, args)
    return stage_patch(ctx, d, args)


if __name__ == "__main__":
    sys.exit(main())
