"""M9 atlas Phase 2, stage B: the cut-and-heal sweep on the six new tasks, the order check and the whole-answer check
(docs/ATLAS-phase2-stageB.md).

  uv run python scripts/atlas2_cuts.py --stage {sweep,ordercheck,whole,healed,test,table} [--shard I/N] [--seed S]
                                       [--task T] [--smoke]

Reuses scripts/atlas_cuts.py (stages, shard files, `Ctx`, `run_heal`, `rebuild_healed`, `stage_test`) by import; this script
points that module's OUT, CKPT and ACTS at var/results/atlas2b, var/ckpt/atlas2b and var/acts/atlas2b in `configure()`.
Every stage resumes (a heal whose key is already in a shard file is skipped). --smoke uses 2 tasks, k in {0, 2}, 4 steps,
16 records, `_smoke` names, and never writes the test ledger.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import atlas_cuts as ac  # noqa: E402

from surgery import adapters  # noqa: E402
from surgery.atlas import heal, measure, metrics, spec  # noqa: E402

OUT = Path("var/results/atlas2b")
CKPT = Path("var/ckpt/atlas2b")
ACTS = Path("var/acts/atlas2b")
P1_DIR = Path("var/results/atlas1")           # Phase 1 runs (the 14 old tasks)
MAP_OLD, MAP_NEW = Path("var/results/atlas"), Path("var/results/atlas2")
HEALED_OLD = Path("var/results/atlas2")       # stage A healed probes of the 14 old tasks
M5_DIR = Path("var/results/m5")

NEW6 = ("arith_order", "syntax_first_close", "code_uses_first", "hard_target_read", "arith_id", "code_literal")
WHOLE_TASKS = ("rule", "sensible", "arith", "syntax_valid", "fact_check", "code_output", "hard", "arith_id",
               "code_literal", "arith_long")
ORDER_TASKS = ("rule", "sensible")
SEEDS = ac.SEEDS
STAGES = ["sweep", "ordercheck", "whole", "healed", "test", "table"]
WORD_TASKS = ("arith_order", "syntax_first_close", "code_uses_first")
ZERO_STEP_TASKS = ("arith_id", "code_literal")
TASK18 = "hard_target_read"
DROP_MARGIN = 0.10
N_PERM, PERM_SEED = 10_000, 9


def configure() -> None:
    """Point the Phase 1 module at this stage's directories (done in main, never at import)."""
    ac.OUT, ac.CKPT, ac.ACTS = OUT, CKPT, ACTS


def make_cfg(smoke: bool) -> SimpleNamespace:
    if smoke:
        return SimpleNamespace(smoke=True, sfx="_smoke", tasks=("arith_order", "arith_id"), ks=(0, 2), steps=4, n=16,
                               transfer=False, ledger=False, kcut=2, headline_k=2, whole_ks=(0, 2),
                               whole_tasks=("arith_id", "arith_long"), order_tasks=ORDER_TASKS,
                               table_tasks=("arith_order", "arith_id"), boots=20, probe_records=64)
    return SimpleNamespace(smoke=False, sfx="", tasks=NEW6, ks=metrics.K_GRID, steps=300, n=None, transfer=False,
                           ledger=True, kcut=12, headline_k=12, whole_ks=(0, 12), whole_tasks=WHOLE_TASKS,
                           order_tasks=ORDER_TASKS, table_tasks=tuple(spec.PHASE1_TASKS) + NEW6, boots=500,
                           probe_records=None)


# ---------------------------------------------------------------- readers

def merged_dir(directory: Path, prefix: str, sfx: str, field: str) -> dict:
    out = {}
    for p in sorted(Path(directory).glob(f"{prefix}{sfx}.shard*of*.json")):
        out.update(json.loads(p.read_text()).get(field, {}))
    return out


def load_runs(cfg) -> tuple[dict, dict]:
    """(Phase 1 runs of the old tasks, this stage's sweep runs). Smoke never reads Phase 1."""
    old = {} if cfg.smoke else merged_dir(P1_DIR, "runs", "", "runs")
    return old, ac.read_merged("runs", cfg.sfx, "runs")


def p1_rule() -> str:
    return (ac.read_json(P1_DIR / "plan.json") or {}).get("lr_rule", "per_k")


def collect_all(cfg, old: dict, new: dict, tasks, ks) -> dict:
    """{task: {k: [winner-lr rows by seed]}} over old tasks (Phase 1 rule) and new ones (this stage's rule)."""
    out = {}
    new_t = [t for t in tasks if t in NEW6]
    old_t = [t for t in tasks if t not in NEW6]
    out.update(ac.collect(old, old_t, ks, p1_rule()))
    out.update(ac.collect(new, new_t, ks, ac.plan_rule(cfg)))
    return out


def phase0(task: str) -> dict:
    return ac.read_json((MAP_NEW if task in NEW6 else MAP_OLD) / f"map_{task}.json") or {}


def surface(task: str):
    base = ac.read_json((MAP_NEW if task in NEW6 else MAP_OLD) / "baseline.json") or {}
    e = (base.get("tasks") or {}).get(task)
    return None if not e else {"bow": e["bow"], "chance": e["chance"], "gap": e["bow"] - e["chance"]}


def healed_l_on(cfg, task: str):
    path = ac.OUT / f"healed_{task}{cfg.sfx}.json" if task in NEW6 else HEALED_OLD / f"healed_{task}.json"
    return (ac.read_json(path) or {}).get("L_on_healed")


# ---------------------------------------------------------------- order check

def load_removed(task: str) -> list[int]:
    return list(json.loads((M5_DIR / f"excise_Qwen3-0.6B_{task}.json").read_text())["removed_order"])


def alive_from_removed(removed, k: int, n_layers: int) -> list[int]:
    """Choice 4: every block not among the first k removed, computed directly. `heal.alive_blocks` would refuse the last
    block, which M7's order cuts."""
    cut = list(removed)[:k]
    if len(set(cut)) != len(cut) or any(not 0 <= i < n_layers for i in cut) or len(cut) >= n_layers:
        raise ValueError(f"bad removed set {cut}")
    gone = set(cut)
    return [i for i in range(n_layers) if i not in gone]


class RemovedCtx(ac.Ctx):
    """A `Ctx` whose k-cut removes `removed[:k]` (an M5 order) instead of the shared order. Shares the pristine model."""

    def __init__(self, base: ac.Ctx, removed):
        super().__init__(base.cfg, model=base.pristine, tok=base.tokenizer, order=base.order, device=base.device)
        self._rec = base._rec
        self.removed = list(removed)

    def cut(self, k: int):
        model = self.fresh()
        alive = alive_from_removed(self.removed, k, self.n_layers)
        heal.cut_model(model, alive)
        return model, alive


@contextlib.contextmanager
def ckpt_dir(path: Path):
    old = ac.CKPT
    ac.CKPT = path
    try:
        yield
    finally:
        ac.CKPT = old


def order_ckpt_dir(task: str) -> Path:
    return ac.CKPT / f"ordercheck_{task}"


def stage_ordercheck(ctx, shard) -> int:
    cfg, (i, n) = ctx.cfg, shard
    path = ac.shard_path("ordercheck", cfg.sfx, i, n)
    mine = ac.read_json(path) or {"shard": [i, n], "runs": {}}
    k = cfg.kcut
    for ti, task in enumerate(cfg.order_tasks):
        if ti % n != i:
            continue
        rctx = RemovedCtx(ctx, load_removed(task))
        removed = rctx.removed[:k]
        for seed in SEEDS:
            have = {**ac.read_merged("ordercheck", cfg.sfx, "runs"), **mine["runs"]}
            lrs = list(ac.LRS) if seed == 0 else [w for w in [ac.winner_lr(have, task, k, "per_k")] if w is not None]
            for lr in lrs:
                key = ac.run_key(task, k, seed, lr)
                if key in have:
                    print(f"skip {key}: already done", flush=True)
                    continue
                with ckpt_dir(order_ckpt_dir(task)):
                    row = ac.run_heal(rctx, task, k, seed, lr)
                row["removed"] = removed
                mine["runs"][key] = row
                ac.write(path, mine)
    return 0


# ---------------------------------------------------------------- whole-answer check

def whole_ckpt_path(cfg, task, k, seed, lr) -> Path:
    return ac.CKPT / f"whole_{task}_k{k}_s{seed}_lr{lr:g}{cfg.sfx}.pt"


def run_whole(ctx, task: str, k: int, seed: int, lr: float) -> dict:
    cfg, tok = ctx.cfg, ctx.tokenizer
    key = f"whole|{ac.run_key(task, k, seed, lr)}"
    print(f"== {key} {ac.now()}", flush=True)
    model, alive = ctx.cut(k)
    handle = heal.add_heal(model, len(alive), ac.RANK, seed)
    try:
        state, curve, tim = heal.train_whole(
            model, tok, task, ctx.records(task, "train"), ctx.records(task, "lrsel"), handle.modules, cfg.steps, lr,
            ac.BATCH, seed, n_evals=4, eval_batch_size=ac.EVAL_BS)
        adapters.load_adapter_state(handle.modules, state)
        recs = ctx.records(task, "selection")[:ac.TRANSFER_N]
        em, texts = heal.exact_match_scores(model, tok, task, recs, max_new_tokens=8, batch_size=ac.EVAL_BS)
        ft, _ = heal.accuracy_and_preds(model, tok, task, recs, ac.EVAL_BS)
        cp = whole_ckpt_path(cfg, task, k, seed, lr)
        ac.save_ckpt(cp, state, {"key": key, "task": task, "k": k, "seed": seed, "lr": lr, "rank": ac.RANK,
                                 "n_alive": len(alive), "best_step": tim["best_step"]})
        row = {"key": key, "task": task, "k": k, "seed": seed, "lr": lr, "steps": cfg.steps, "n_alive": len(alive),
               "curve": curve, "best_lrsel": tim["best_lrsel"], "best_step": tim["best_step"], "n": len(recs),
               "exact_match": em, "first_token_accuracy": ft,
               "examples": [{"text": t, "answer": heal.answer_text(task, r)} for t, r in zip(texts[:5], recs[:5])],
               "timings": {a: b for a, b in tim.items() if a not in ("best_lrsel", "best_step")},
               "ckpt": str(cp), "ts": ac.now()}
    finally:
        handle.remove()
        del model
        ac.release(ctx)
    print(f"{key}: exact {em:.3f} first-token {ft:.3f} lrsel exact {row['best_lrsel']:.3f}", flush=True)
    return row


def stage_whole(ctx, shard) -> int:
    cfg, (i, n) = ctx.cfg, shard
    path = ac.shard_path("whole", cfg.sfx, i, n)
    mine = ac.read_json(path) or {"shard": [i, n], "runs": {}}
    old, new = load_runs(cfg)
    missing = []
    units = [(t, k) for t in cfg.whole_tasks for k in cfg.whole_ks]
    for u, (task, k) in enumerate(units):
        if u % n != i:
            continue
        if task != "arith_long":
            w = collect_all(cfg, old, new, [task], [k]).get(task, {}).get(k)
            if not w:
                missing.append(f"{task}|k{k}")
                print(f"no sweep winner for {task} k{k}; skipped", flush=True)
                continue
            sweep_lr = w[0]["lr"]
        for seed in SEEDS:
            have = {**ac.read_merged("whole", cfg.sfx, "runs"), **mine["runs"]}
            if task == "arith_long":
                lrs = list(ac.LRS) if seed == 0 else [w_ for w_ in [ac.winner_lr(have, task, k, "per_k")] if w_ is not None]
            else:
                lrs = [sweep_lr]
            for lr in lrs:
                key = f"whole|{ac.run_key(task, k, seed, lr)}"
                if key in have:
                    print(f"skip {key}: already done", flush=True)
                    continue
                row = run_whole(ctx, task, k, seed, lr)
                mine["runs"][key] = row
                ac.write(path, mine)
    if missing:
        print(f"whole incomplete: {missing}", flush=True)
        return 1
    return 0


# ---------------------------------------------------------------- healed probes (secondary)

def stage_healed(ctx, task: str) -> int:
    import atlas_map
    from atlas2_healed import l_on_healed
    cfg = ctx.cfg
    final = ac.OUT / f"healed_{task}{cfg.sfx}.json"
    if final.exists():
        print(f"{final} exists; nothing to do", flush=True)
        return 0
    runs = ac.read_merged("runs", cfg.sfx, "runs")
    rows = [r for r in ac.collect(runs, [task], [0], ac.plan_rule(cfg)).get(task, {}).get(0, []) if r["seed"] == 0]
    if not rows:
        print(f"{task}: no healed k=0 seed-0 run", flush=True)
        return 1
    row = rows[0]
    recs = ctx.records(task, "discovery")
    if cfg.probe_records:
        recs = recs[:cfg.probe_records]
    store = ac.ACTS / f"{task}_healed{cfg.sfx}"
    model, handle = ac.rebuild_healed(ctx, row)
    adapters.merge_lora(handle)
    meta = measure.build_store(model, ctx.tokenizer, task, recs, store, batch_size=ac.EVAL_BS)
    answers = meta["answers"]
    acc = float(np.mean(np.asarray(meta["pred"]) == np.asarray(answers)))
    X = np.load(store / "resid.npy", mmap_mode="r")
    p = atlas_map._probe(X, answers, meta["template_ids"], np.arange(len(answers)), cfg.boots, seed=0)
    chance = spec.chance(answers)
    curve = p["curve"] if p else None
    lo, never = l_on_healed(curve, chance, acc) if curve else (None, False)
    res = {"task": task, "smoke": cfg.smoke, "lr": row["lr"], "ckpt": row["ckpt"], "n": len(answers), "chance": chance,
           "healed_accuracy": acc, "curve": curve, "lo": p["lo"] if p else None, "hi": p["hi"] if p else None,
           "L_on_healed": lo, "never": never, "L_on_phase0": phase0(task).get("L_on")}
    ac.write(final, res)
    shutil.rmtree(store, ignore_errors=True)
    del model
    ac.release(ctx)
    print(f"{task}: healed acc {acc:.3f} chance {chance:.3f} L_on_healed {lo} never {never} "
          f"L_on_phase0 {res['L_on_phase0']} -> {final}", flush=True)
    return 0


# ---------------------------------------------------------------- test split

def stage_test(ctx) -> int:
    """Phase 1's `stage_test` on the new tasks (healed k=0 and k*), then the order-check models (`ordercheck_k{k}`)."""
    cfg = ctx.cfg
    rc = ac.stage_test(ctx)
    runs = ac.read_merged("ordercheck", cfg.sfx, "runs")
    path = ac.OUT / f"test_results{cfg.sfx}.json"
    doc = ac.read_json(path) or {"results": {}}
    ledger = ac.OUT / "test_ledger.jsonl"
    name = f"ordercheck_k{cfg.kcut}"
    for task in cfg.order_tasks:
        rctx = None
        for row in ac.collect(runs, [task], [cfg.kcut], "per_k").get(task, {}).get(cfg.kcut, []):
            key = f"ordercheck|{row['key']}"
            if key in doc["results"]:
                continue
            rctx = rctx or RemovedCtx(ctx, load_removed(task))
            model, handle = ac.rebuild_healed(rctx, row)
            recs = ctx.records(task, "test")
            a, preds = heal.accuracy_and_preds(model, ctx.tokenizer, task, recs, ac.EVAL_BS)
            rung_acc, rung_chance = ac.rung_scores(recs, preds)
            doc["results"][key] = {"task": task, "k": cfg.kcut, "seed": row["seed"], "lr": row["lr"], "names": [name],
                                   "n": len(recs), "accuracy": a, "chance": spec.chance([r["answer"] for r in recs]),
                                   "rung_acc": rung_acc, "rung_chance": rung_chance, "ts": ac.now()}
            ac.write(path, doc)
            if cfg.ledger:
                with ledger.open("a") as fh:
                    fh.write(json.dumps({"key": key, "names": [name], "task": task, "k": cfg.kcut, "seed": row["seed"],
                                         "n": len(recs), "accuracy": a, "ts": ac.now()}, sort_keys=True) + "\n")
            print(f"test {key} ({name}): {a:.3f}", flush=True)
            del model
            ac.release(ctx)
    return rc


# ---------------------------------------------------------------- statistics (choices 3, 6, 7, 4, 5)

def r12_value(a12: float, a0: float, chance: float) -> float:
    """Choice 3: (acc at k=12 - chance) / (healed k=0 acc - chance), clipped to 0..1."""
    return float(min(1.0, max(0.0, (a12 - chance) / (a0 - chance))))


def is_dropped(a0: float, chance: float) -> bool:
    """A task whose healed k=0 accuracy is under chance + 0.10 leaves H2-A and H2-B."""
    return a0 < chance + DROP_MARGIN - 1e-9


def weighted_rho(pred, y, kinds):
    """Spearman rho within appraisal and within execution (average ranks), combined as a mean weighted by task count.
    A kind with a constant side, or fewer than 3 tasks, is left out. Returns (rho or None, {kind: rho or None})."""
    num = den = 0.0
    by = {}
    for kind in ("appraise", "execute"):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        r = heal.spearman([pred[i] for i in idx], [y[i] for i in idx]) if idx else float("nan")
        by[kind] = None if math.isnan(r) else r
        if not math.isnan(r):
            num += len(idx) * r
            den += len(idx)
    return (num / den if den else None), by


def h2a_test(pred, y, kinds, n_perm: int = N_PERM, seed: int = PERM_SEED) -> dict:
    """Choice 6: weighted rho of predictor against R12, p one-sided from `n_perm` permutations of the predictor within
    kind (rho <= observed), p = (hits + 1) / (n_perm + 1)."""
    pred, y, kinds = list(pred), list(y), list(kinds)
    obs, by = weighted_rho(pred, y, kinds)
    out = {"rho": obs, "by_kind": by, "n": {k: kinds.count(k) for k in ("appraise", "execute")}, "p": None}
    if obs is None or not n_perm:
        return out
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        s, _ = weighted_rho(heal.perm_within(pred, kinds, rng), y, kinds)
        if s is not None and s <= obs + 1e-12:
            hits += 1
    out["p"] = (hits + 1) / (n_perm + 1)
    return out


def label_a(rho, p, rival_rho) -> str:
    if rho is None:
        return "undetermined"
    if rho <= -0.5 and p is not None and p < 0.05 and abs(rho) > abs(rival_rho or 0.0):
        return "supported"
    return "not_supported" if rho > -0.2 else "partial"


def l_on_group(l_on) -> int:
    return 0 if l_on <= 3 else (1 if l_on <= 10 else 2)


def h2b_test(groups, y, kinds, n_perm: int = N_PERM, seed: int = PERM_SEED) -> dict:
    """Choice 7: unweighted mean over the L_on groups holding both kinds of (mean R12 appraisal - mean R12 execution);
    p one-sided (>= observed) from permutations of kind within group."""
    groups, y, kinds = list(groups), list(y), list(kinds)

    def stat(kk):
        diffs = {}
        for g in sorted(set(groups)):
            a = [y[i] for i in range(len(y)) if groups[i] == g and kk[i] == "appraise"]
            e = [y[i] for i in range(len(y)) if groups[i] == g and kk[i] == "execute"]
            if a and e:
                diffs[g] = (float(np.mean(a)) - float(np.mean(e)), len(a), len(e))
        return (float(np.mean([d[0] for d in diffs.values()])) if diffs else None), diffs

    obs, diffs = stat(kinds)
    out = {"stat": obs, "p": None, "groups": {str(g): {"diff": d[0], "n_appraise": d[1], "n_execute": d[2]}
                                              for g, d in diffs.items()}, "label": "undetermined"}
    if obs is None:
        return out
    if n_perm:
        rng = np.random.default_rng(seed)
        hits = 0
        for _ in range(n_perm):
            s, _ = stat(list(heal.perm_within(kinds, groups, rng)))
            if s is not None and s >= obs - 1e-12:
                hits += 1
        out["p"] = (hits + 1) / (n_perm + 1)
    if obs >= 0.20 and out["p"] is not None and out["p"] < 0.05:
        out["label"] = "supported"
    else:
        out["label"] = "not_supported" if obs < 0.10 else "partial"
    return out


def reading_word(r12: dict) -> dict:
    vals = {t: r12.get(t) for t in WORD_TASKS}
    if any(v is None for v in vals.values()):
        return {"tasks": vals, "mean": None, "label": "undetermined", "task18": r12.get(TASK18)}
    m = float(np.mean(list(vals.values())))
    return {"tasks": vals, "mean": m, "task18": r12.get(TASK18),
            "label": "not_explained" if m >= 0.80 else ("rested_on" if m <= 0.50 else "mixed")}


def reading_zero_step(r12: dict) -> dict:
    vals = {t: r12.get(t) for t in ZERO_STEP_TASKS}
    if any(v is None for v in vals.values()):
        label = "undetermined"
    elif all(v >= 0.80 for v in vals.values()):
        label = "working_steps"
    elif all(v <= 0.50 for v in vals.values()):
        label = "kind_beyond_steps"
    else:
        label = "mixed"
    return {"tasks": vals, "label": label}


def label_d(rule_mean) -> str:
    if rule_mean is None:
        return "undetermined"
    return "confirmed" if rule_mean >= 0.90 else ("harness_differs" if rule_mean < 0.50 else "partial")


def whole_rows_for(whole: dict, task: str, k: int) -> list[dict]:
    rows = [r for r in whole.values() if r["task"] == task and r["k"] == k]
    if task == "arith_long":
        lr = ac.winner_lr({r["key"]: r for r in rows}, task, k, "per_k")
        rows = [] if lr is None else [r for r in rows if math.isclose(r["lr"], lr)]
    return sorted(rows, key=lambda r: r["seed"])


def h2c_cells(cfg, whole: dict, sweep_coll: dict, need: int = 16, tol: float = 0.05) -> dict:
    """Choice 5: per (task, k) cell, |seed-mean exact match - seed-mean first-token accuracy| on the same 200 records."""
    cells = []
    for task in cfg.whole_tasks:
        for k in cfg.whole_ks:
            rows = whole_rows_for(whole, task, k)
            em = float(np.mean([r["exact_match"] for r in rows])) if rows else None
            if task == "arith_long":
                ft_rows = [r["first_token_accuracy"] for r in rows]
            else:
                ft_rows = [r["sel_accuracy200"] for r in sweep_coll.get(task, {}).get(k, [])]
            ft = float(np.mean(ft_rows)) if ft_rows else None
            diff = None if em is None or ft is None else abs(em - ft)
            cells.append({"task": task, "k": k, "exact_match": em, "first_token_accuracy": ft, "diff": diff,
                          "n_seeds": len(rows), "within": None if diff is None else diff <= tol + 1e-9})
    n_within = sum(1 for c in cells if c["within"])
    complete = all(c["diff"] is not None for c in cells)
    return {"cells": cells, "n_cells": len(cells), "n_within": n_within, "need": need, "complete": complete,
            "label": "incomplete" if not complete else ("supported" if n_within >= need else "not_supported")}


# ---------------------------------------------------------------- table stage

def task_rows(cfg, coll: dict) -> list[dict]:
    grid = tuple(cfg.ks)
    out = []
    for t in sorted(cfg.table_tasks, key=lambda t: spec.TASKS[t].number):
        ts = spec.TASKS[t]
        by_k = coll.get(t)
        m, sf = phase0(t), surface(t)
        row = {"task": t, "number": ts.number, "kind": ts.kind, "L_on": m.get("L_on"), "L_on_healed": healed_l_on(cfg, t),
               "surface": sf, "r12": None, "dropped": False, "acc": {}, "healed_k0": None, "k_star": None, "chance": None,
               "acc_k12": None}
        if by_k:
            acc = {k: [r["sel_accuracy"] for r in rows] for k, rows in by_k.items()}
            chance = next(iter(by_k.values()))[0]["chance"]
            d = ac._describe(acc, chance, None, grid)
            row.update(chance=chance, acc=d["acc"], healed_k0=d["healed_k0"], k_star=d["k_star"],
                       acc_k12=(d["acc"].get(cfg.kcut) or {}).get("mean"), lr={k: rows[0]["lr"] for k, rows in by_k.items()})
            if d["healed_k0"] is not None:
                row["dropped"] = is_dropped(d["healed_k0"], chance)
                if not row["dropped"] and row["acc_k12"] is not None:
                    row["r12"] = r12_value(row["acc_k12"], d["healed_k0"], chance)
        out.append(row)
    return out


def verdicts(rows: list[dict], n_perm: int) -> dict:
    def use(drop18=False, healed=False):
        key = "L_on_healed" if healed else "L_on"
        return [r for r in rows if r["r12"] is not None and r[key] is not None and not (drop18 and r["task"] == TASK18)]

    def a_block(rs, key):
        kinds = [r["kind"] for r in rs]
        y = [r["r12"] for r in rs]
        main = h2a_test([r[key] for r in rs], y, kinds, n_perm)
        sr = [r for r in rs if r["surface"]]
        rival = h2a_test([-r["surface"]["gap"] for r in sr], [r["r12"] for r in sr], [r["kind"] for r in sr], 0)
        return {**main, "rival_rho": rival["rho"], "rival_n": rival["n"], "tasks": [r["task"] for r in rs],
                "label": label_a(main["rho"], main["p"], rival["rho"]),
                "underpowered": True}

    def b_block(rs):
        return {**h2b_test([l_on_group(r["L_on"]) for r in rs], [r["r12"] for r in rs], [r["kind"] for r in rs], n_perm),
                "tasks": [r["task"] for r in rs]}

    r12 = {r["task"]: r["r12"] for r in rows}
    return {"H2-A": a_block(use(), "L_on"),
            "H2-A_healed_L_on": a_block(use(healed=True), "L_on_healed"),
            "H2-A_drop18": a_block(use(drop18=True), "L_on"),
            "H2-B": b_block(use()), "H2-B_drop18": b_block(use(drop18=True)),
            "dropped": [r["task"] for r in rows if r["dropped"]],
            "word_shortcut": reading_word(r12), "zero_step": reading_zero_step(r12)}


def build_summary(cfg, d: dict, n_perm: int = N_PERM) -> dict:
    old, new = d["old"], d["new"]
    ks = sorted({r["k"] for r in {**old, **new}.values()})
    coll = collect_all(cfg, old, new, cfg.table_tasks, ks)
    rows = task_rows(cfg, coll)
    s = {"smoke": cfg.smoke, "ks": list(cfg.ks), "kcut": cfg.kcut, "rows": rows, "verdict": verdicts(rows, n_perm)}
    # H2-D
    od = {}
    for t in cfg.order_tasks:
        rs = ac.collect(d["order"], [t], [cfg.kcut], "per_k").get(t, {}).get(cfg.kcut, [])
        od[t] = {"mean": float(np.mean([r["sel_accuracy"] for r in rs])) if rs else None,
                 "seeds": [r["sel_accuracy"] for r in rs], "lr": rs[0]["lr"] if rs else None,
                 "removed": rs[0].get("removed") if rs else None}
    s["verdict"]["H2-D"] = {**od, "label": label_d((od.get("rule") or {}).get("mean"))}
    # H2-C
    s["verdict"]["H2-C"] = h2c_cells(cfg, d["whole"], coll)
    tests = {}
    for r in d["test"].values():
        for name in r["names"]:
            tests.setdefault(r["task"], {}).setdefault(name, []).append(r)
    s["test"] = {t: {name: {"k": rs[0]["k"], "accuracy": ac._stat([r["accuracy"] for r in rs]), "chance": rs[0]["chance"]}
                     for name, rs in by.items()} for t, by in tests.items()}
    return s


def load_all(cfg) -> dict:
    old, new = load_runs(cfg)
    return {"old": old, "new": new, "order": ac.read_merged("ordercheck", cfg.sfx, "runs"),
            "whole": ac.read_merged("whole", cfg.sfx, "runs"),
            "test": (ac.read_json(ac.OUT / f"test_results{cfg.sfx}.json") or {}).get("results", {})}


def render_tables(s: dict) -> str:
    ks, v, f = s["ks"], s["verdict"], ac._f
    out = [f"# Atlas Phase 2 stage B tables{' (smoke)' if s['smoke'] else ''}", "",
           f"R12 = (seed-mean accuracy at k={s['kcut']} - chance) / (seed-mean healed k=0 accuracy - chance), clipped 0 to 1. "
           "Cells are seed mean (min-max) of selection accuracy.", "", "## All tasks", "",
           "| # | task | kind | L_on | L_on healed | surface (BoW - chance) | chance | healed k0 | " +
           " | ".join(f"k{k}" for k in ks) + " | R12 | k* | note |", "|" + "---|" * (10 + len(ks))]
    for r in s["rows"]:
        note = "dropped (healed k0 under chance + 0.10)" if r["dropped"] else ("" if r["acc"] else "no runs")
        out.append(f"| {r['number']} | {r['task']} | {r['kind']} | {r['L_on']} | {r['L_on_healed']} | "
                   f"{f(r['surface'] and r['surface']['gap'])} | {f(r['chance'])} | {f(r['healed_k0'])} | " +
                   " | ".join(ac._cell(r["acc"].get(k)) for k in ks) + f" | {f(r['r12'])} | {r['k_star']} | {note} |")
    out += ["", "## H2-A (rank correlation of L_on with R12, within kind, weighted by task count)", "",
            "Underpowered at 0.6B (most tasks sit at L_on 1); a 'supported' here is a hint for stage C, not a finding.", ""]
    for name in ("H2-A", "H2-A_healed_L_on", "H2-A_drop18"):
        a = v[name]
        out.append(f"- {name}: rho {f(a['rho'])}, p {f(a['p'], 4)}, rival (negated surface) rho {f(a['rival_rho'])}, "
                   f"tasks {len(a['tasks'])} (appraise {a['n']['appraise']}, execute {a['n']['execute']}), "
                   f"by kind {a['by_kind']}: **{a['label']}**")
    out += ["", "## H2-B (appraisal minus execution R12 at equal L_on group)", ""]
    for name in ("H2-B", "H2-B_drop18"):
        b = v[name]
        out.append(f"- {name}: stat {f(b['stat'])}, p {f(b['p'], 4)}, groups {json.dumps(b['groups'])}: **{b['label']}**")
    d = v["H2-D"]
    out += ["", "## H2-D (order check at the M7 cut)", ""]
    for t in ("rule", "sensible"):
        if t in d:
            out.append(f"- {t}: seed mean {f(d[t]['mean'])}, seeds {d[t]['seeds']}")
    out.append(f"- H2-D: **{d['label']}** (rule seed mean: confirmed >= 0.90, harness differs < 0.50)")
    c = v["H2-C"]
    out += ["", "## H2-C (whole answer against first token, first 200 selection records)", "",
            "| task | k | exact match | first-token | |diff| | seeds |", "|---|---|---|---|---|---|"]
    out += [f"| {x['task']} | {x['k']} | {f(x['exact_match'])} | {f(x['first_token_accuracy'])} | {f(x['diff'])} | {x['n_seeds']} |"
            for x in c["cells"]]
    out.append(f"\nWithin 0.05 in {c['n_within']} of {c['n_cells']} cells (need {c['need']}): **{c['label']}**")
    w, z = v["word_shortcut"], v["zero_step"]
    out += ["", "## Readings", "",
            f"- Word shortcut: tasks {w['tasks']}, mean {f(w['mean'])}, task 18 beside them {f(w['task18'])}: **{w['label']}**",
            f"- Zero-step execution: {z['tasks']}: **{z['label']}**",
            f"- Dropped from H2-A and H2-B: {v['dropped']}", "", "## Test split (once per named model)", ""]
    if s["test"]:
        out += ["| task | model | k | accuracy (seeds) | chance |", "|---|---|---|---|---|"]
        out += [f"| {t} | {n} | {x['k']} | {ac._cell(x['accuracy'])} | {f(x['chance'])} |"
                for t, by in s["test"].items() for n, x in by.items()]
    else:
        out.append("Not run.")
    out.append("")
    return "\n".join(out)


def stage_table(cfg) -> int:
    s = build_summary(cfg, load_all(cfg))
    ac.write(ac.OUT / f"summary{cfg.sfx}.json", s)
    md = render_tables(s)
    (ac.OUT / f"tables{cfg.sfx}.md").write_text(md + "\n")
    print(md, flush=True)
    return 0


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--shard", default=None, help="I/N")
    ap.add_argument("--seed", type=int, default=0, choices=SEEDS)
    ap.add_argument("--task", default=None, help="healed: one task (default every task of the sweep)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    configure()
    cfg = make_cfg(args.smoke)
    shard = ac.parse_shard(args.shard)
    ac.OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "table":
        return stage_table(cfg)
    ctx = ac.Ctx(cfg)
    if args.stage == "sweep":
        return ac.stage_sweep(ctx, args.seed, shard)
    if args.stage == "ordercheck":
        return stage_ordercheck(ctx, shard)
    if args.stage == "whole":
        return stage_whole(ctx, shard)
    if args.stage == "healed":
        tasks = [args.task] if args.task else list(cfg.tasks)
        return int(any([stage_healed(ctx, t) for t in tasks]))
    return stage_test(ctx)


if __name__ == "__main__":
    sys.exit(main())
