"""M9 atlas Phase 1: the cut-and-heal sweep on Qwen3-0.6B (docs/ATLAS-phase1.md).

  uv run python scripts/atlas_cuts.py --stage {probe,cutonly,sweep,secondary,generative,rivals,ceiling,test,table}
                                      [--shard I/N] [--seed S] [--smoke]

Results go to var/results/atlas1/, LoRA checkpoints (state plus meta) to var/ckpt/atlas1/. Every stage resumes: a heal
whose key `{task}|k{k}|s{seed}|lr{lr}` is already in any shard file is skipped, and each shard writes only its own
`runs.shard{I}of{N}.json` (readers merge them all). --smoke uses 2 tasks, k in {0, 2}, 4 steps, 16 records, `_smoke` names,
and never writes the test ledger.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from surgery import adapters, capability  # noqa: E402
from surgery.atlas import heal, measure, metrics, spec  # noqa: E402
from surgery.eval import _free, load_model  # noqa: E402
from surgery.task.generator import read_jsonl  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
OUT = Path("var/results/atlas1")
CKPT = Path("var/ckpt/atlas1")
ACTS = Path("var/acts/atlas1")
MAP_DIR = Path("var/results/atlas")
ORDER_PATH = Path("var/results/atlas/order.json")
GEN = Path("var/data/general")
STAGES = ["probe", "cutonly", "sweep", "secondary", "generative", "rivals", "ceiling", "test", "table"]

LRS = heal.LRS
SEEDS = (0, 1, 2)
RANK, BATCH, EVAL_BS = 10, 8, 16
CEILING_LR, DEV_EVAL = 3e-5, 256
SWEEP_HOURS_CAP = 16.0
TRANSFER_N = 200
ROUTE_LONG, ROUTE_SHORT = "route_work", "arith"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean(x):
    """NaN and inf become None so the JSON writer (allow_nan=False) accepts the document."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def write(path, obj) -> None:
    measure._write_json(path, _clean(obj))


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def make_cfg(smoke: bool) -> SimpleNamespace:
    if smoke:
        return SimpleNamespace(smoke=True, sfx="_smoke", tasks=("arith", "arith_shape"), ks=(0, 2), steps=4, n=16,
                               gen_tasks=("arith",), rival_k=2, ceiling_k=2, ceiling_steps=8, headline_k=2,
                               n_lesion=8, lesion_blocks=[0, 1], probe_steps=3, ledger=False)
    return SimpleNamespace(smoke=False, sfx="", tasks=spec.PHASE1_TASKS,
                           ks=metrics.K_GRID, steps=300, n=None, gen_tasks=("arith", "code_output"), rival_k=12,
                           ceiling_k=12, ceiling_steps=1500, headline_k=12, n_lesion=500, lesion_blocks=None,
                           probe_steps=30, ledger=True)


# ---------------------------------------------------------------- context

class Ctx:
    """One shard's model access: an unmodified CPU copy kept for the run, a fresh device copy per heal."""

    def __init__(self, cfg, model=None, tok=None, order=None, device=None):
        self.cfg, self._pristine, self.tok = cfg, model, tok
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._order = order
        self._rec = {}

    @property
    def pristine(self):
        if self._pristine is None:
            self._pristine, self.tok = load_model(MODEL, device="cpu", dtype="fp32")
            if self.device.startswith("cuda"):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        return self._pristine

    @property
    def tokenizer(self):
        self.pristine  # noqa: B018
        return self.tok

    @property
    def order(self):
        if self._order is None:
            self._order = read_json(ORDER_PATH)["shared_order"]
        return self._order

    @property
    def n_layers(self):
        return self.pristine.config.num_hidden_layers

    def fresh(self):
        """A clean copy of the unmodified weights on the device (the CPU original is never touched)."""
        return copy.deepcopy(self.pristine).to(self.device).eval()

    def cut(self, k: int):
        model = self.fresh()
        alive = heal.alive_blocks(self.order, k, self.n_layers)
        heal.cut_model(model, alive)
        return model, alive

    def records(self, task: str, split: str) -> list[dict]:
        key = (task, split)
        if key not in self._rec:
            recs = spec.read_records(task, split)
            self._rec[key] = recs[:self.cfg.n] if self.cfg.n else recs
        return self._rec[key]

    def sel200(self, task: str) -> list[dict]:
        return self.records(task, "selection")[:TRANSFER_N]


# ---------------------------------------------------------------- keys, shards, readers

def run_key(task, k, seed, lr) -> str:
    return f"{task}|k{k}|s{seed}|lr{lr:g}"


def parse_shard(s: str | None) -> tuple[int, int]:
    if not s:
        return 0, 1
    i, n = s.split("/")
    i, n = int(i), int(n)
    if not 0 <= i < n:
        raise SystemExit(f"bad --shard {s}")
    return i, n


def shard_path(prefix: str, sfx: str, i: int, n: int) -> Path:
    return OUT / f"{prefix}{sfx}.shard{i}of{n}.json"


def read_merged(prefix: str, sfx: str, field: str) -> dict:
    """Merge `field` of every `{prefix}{sfx}.shard*of*.json` in the results directory."""
    merged = {}
    for p in sorted(OUT.glob(f"{prefix}{sfx}.shard*of*.json")):
        merged.update(json.loads(p.read_text()).get(field, {}))
    return merged


def plan_rule(cfg) -> str:
    return (read_json(OUT / f"plan{cfg.sfx}.json") or {}).get("lr_rule", "per_k")


def unit_shard(ti: int, ki: int, n: int, rule: str) -> int:
    """Shard of the (task, k) unit. Under the fallback all k of one task stay together, so k=0 finishes before the other
    k of that task need its lr; otherwise units rotate so every shard gets a mix of tasks and depths."""
    return ti % n if rule == "fallback" else (ti + ki) % n


def winner_lr(runs: dict, task: str, k: int, rule: str):
    """Choice 3: the lr with the higher best lrsel among seed 0's two runs at (task, k), or at (task, 0) under the
    fallback. None until both lrs have run."""
    kk = 0 if rule == "fallback" else k
    rows = [r for r in runs.values() if r["task"] == task and r["k"] == kk and r["seed"] == 0]
    if len({round(r["lr"], 12) for r in rows}) < 2:
        return None
    return heal.pick_lr(rows)


def unit_lrs(runs: dict, task: str, k: int, seed: int, rule: str) -> list[float]:
    if seed == 0 and (rule != "fallback" or k == 0):
        return list(LRS)
    w = winner_lr(runs, task, k, rule)
    return [] if w is None else [w]


def collect(runs: dict, tasks, ks, rule: str) -> dict:
    """{task: {k: [winner-lr run rows, ordered by seed]}} (only where the winner is known)."""
    out = {}
    for t in tasks:
        for k in ks:
            lr = winner_lr(runs, t, k, rule)
            if lr is None:
                continue
            rows = sorted((r for r in runs.values() if r["task"] == t and r["k"] == k and math.isclose(r["lr"], lr)),
                          key=lambda r: r["seed"])
            if rows:
                out.setdefault(t, {})[k] = rows
    return out


# ---------------------------------------------------------------- scoring helpers

def rung_scores(recs, preds):
    acc, chance = {}, {}
    for rung in sorted({r["rung"] for r in recs if r.get("rung")}):
        idx = [i for i, r in enumerate(recs) if r.get("rung") == rung]
        acc[rung] = float(np.mean([preds[i] == recs[i]["answer"] for i in idx]))
        chance[rung] = spec.chance([recs[i]["answer"] for i in idx])
    return acc, chance


def score_selection(ctx, model, task: str) -> dict:
    """Accuracy and predictions on the full selection split, rungs, and chance."""
    recs = ctx.records(task, "selection")
    acc, preds = heal.accuracy_and_preds(model, ctx.tokenizer, task, recs, EVAL_BS)
    answers = [r["answer"] for r in recs]
    acc200 = float(np.mean([p == a for p, a in zip(preds[:TRANSFER_N], answers[:TRANSFER_N])]))
    rung_acc, rung_chance = rung_scores(recs, preds)
    return {"n": len(recs), "sel_accuracy": acc, "sel_accuracy200": acc200, "sel_preds": preds,
            "chance": spec.chance(answers), "rung_acc": rung_acc, "rung_chance": rung_chance}


def score_other(ctx, model, task: str) -> float:
    return heal.accuracy_and_preds(model, ctx.tokenizer, task, ctx.sel200(task), EVAL_BS)[0]


def save_ckpt(path: Path, state: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"state": state, "meta": meta}, tmp)
    tmp.replace(path)


def ckpt_path(cfg, task, k, seed, lr) -> Path:
    return CKPT / f"{task}_k{k}_s{seed}_lr{lr:g}{cfg.sfx}.pt"


def rebuild_healed(ctx, row: dict):
    """The healed model of a sweep row: cut, LoRA structure, saved state. Returns (model, handle)."""
    model, alive = ctx.cut(row["k"])
    handle = heal.add_heal(model, len(alive), RANK, row["seed"])
    saved = torch.load(row["ckpt"], map_location="cpu")["state"]
    adapters.load_adapter_state(handle.modules, saved)
    return model, handle


def release(ctx):
    import gc
    gc.collect()
    if ctx.device.startswith("cuda"):
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- stage: sweep

def run_heal(ctx, task: str, k: int, seed: int, lr: float) -> dict:
    cfg, tok = ctx.cfg, ctx.tokenizer
    key = run_key(task, k, seed, lr)
    print(f"== {key} {now()}", flush=True)
    model, alive = ctx.cut(k)
    handle = heal.add_heal(model, len(alive), RANK, seed)
    try:
        state, curve, tim = heal.train_heal(
            model, tok, task, ctx.records(task, "train"), ctx.records(task, "lrsel"), handle.modules, cfg.steps, lr,
            BATCH, seed, n_evals=4, eval_batch_size=EVAL_BS)
        adapters.load_adapter_state(handle.modules, state)
        t0 = time.perf_counter()
        row = {"key": key, "task": task, "k": k, "seed": seed, "lr": lr, "steps": cfg.steps, "n_alive": len(alive),
               "curve": curve, "best_lrsel": tim["best_lrsel"], "best_step": tim["best_step"],
               **score_selection(ctx, model, task)}
        row["transfer"] = ({b: score_other(ctx, model, b) for b in cfg.tasks if b != task}
                           if getattr(cfg, "transfer", True) else {})  # Phase 2 stage B switches it off
        tim["score_seconds"] = time.perf_counter() - t0
        row["timings"] = {k_: v for k_, v in tim.items() if k_ not in ("best_lrsel", "best_step")}
        cp = ckpt_path(cfg, task, k, seed, lr)
        save_ckpt(cp, state, {"key": key, "task": task, "k": k, "seed": seed, "lr": lr, "rank": RANK,
                              "n_alive": len(alive), "best_step": tim["best_step"]})
        row["ckpt"] = str(cp)
        row["ts"] = now()
    finally:
        handle.remove()
        del model
        release(ctx)
    print(f"{key}: lrsel {row['best_lrsel']:.3f} (step {row['best_step']}) sel {row['sel_accuracy']:.3f} "
          f"({tim['step_seconds']:.2f}s/step, train {tim['train_seconds']:.0f}s, score {tim['score_seconds']:.0f}s)",
          flush=True)
    return row


def stage_sweep(ctx, seed: int, shard) -> int:
    cfg, (i, n) = ctx.cfg, shard
    rule = plan_rule(cfg)
    path = shard_path("runs", cfg.sfx, i, n)
    mine = read_json(path) or {"shard": [i, n], "runs": {}}
    print(f"sweep seed {seed} shard {i}/{n} lr rule {rule}", flush=True)
    missing = []
    for ti, task in enumerate(cfg.tasks):
        for ki, k in enumerate(cfg.ks):
            if unit_shard(ti, ki, n, rule) != i:
                continue
            merged = {**read_merged("runs", cfg.sfx, "runs"), **mine["runs"]}
            lrs = unit_lrs(merged, task, k, seed, rule)
            if not lrs:
                missing.append(f"{task}|k{k}")
                print(f"no seed-0 winner for {task} k{k}; skipped", flush=True)
                continue
            for lr in lrs:
                if run_key(task, k, seed, lr) in merged:
                    print(f"skip {run_key(task, k, seed, lr)}: already done", flush=True)
                    continue
                row = run_heal(ctx, task, k, seed, lr)
                mine["runs"][row["key"]] = row
                write(path, mine)
    if missing:
        print(f"sweep incomplete: {len(missing)} units lack a seed-0 winner: {missing}", flush=True)
        return 1
    return 0


# ---------------------------------------------------------------- stage: cutonly

def stage_cutonly(ctx, shard) -> int:
    cfg, (i, n) = ctx.cfg, shard
    path = shard_path("cutonly", cfg.sfx, i, n)
    mine = read_json(path) or {"shard": [i, n], "cut": {}}
    for ki, k in enumerate(cfg.ks):
        if ki % n != i:
            continue
        have = {**read_merged("cutonly", cfg.sfx, "cut"), **mine["cut"]}
        if all(f"{t}|k{k}" in have for t in cfg.tasks):
            print(f"skip cut-only k{k}: already done", flush=True)
            continue
        model, _ = ctx.cut(k)
        for task in cfg.tasks:
            key = f"{task}|k{k}"
            if key in have:
                continue
            s = score_selection(ctx, model, task)
            mine["cut"][key] = {"task": task, "k": k, "acc": s["sel_accuracy"], "acc200": s["sel_accuracy200"],
                                "n": s["n"], "chance": s["chance"], "rung_acc": s["rung_acc"],
                                "rung_chance": s["rung_chance"], "ts": now()}
            write(path, mine)
            print(f"cut-only {key}: {s['sel_accuracy']:.3f}", flush=True)
        del model
        release(ctx)
    return 0


# ---------------------------------------------------------------- stage: secondary

def _lens_curve(lens, answers, idx, final_pred):
    a = np.asarray(answers)[idx]
    curve = [float((lens[idx, l].argmax(-1) == a).mean()) for l in range(lens.shape[1])]
    chance = spec.chance(a.tolist())
    final = float((np.asarray(final_pred)[idx] == a).mean())
    return {"chance": chance, "curve": curve, "final": final, "L_lens": metrics.l_lens(curve, chance, final)}


def stage_secondary(ctx, shard) -> int:
    cfg, (i, n) = ctx.cfg, shard
    rule = plan_rule(cfg)
    runs = read_merged("runs", cfg.sfx, "runs")
    coll = collect(runs, cfg.tasks, [0], rule)
    for ti, task in enumerate(cfg.tasks):
        final_path = OUT / f"secondary_{task}{cfg.sfx}.json"
        if ti % n != i or final_path.exists():
            continue
        rows = [r for r in coll.get(task, {}).get(0, []) if r["seed"] == 0]
        if not rows:
            print(f"secondary {task}: no healed k=0 seed-0 run; skipped", flush=True)
            continue
        row = rows[0]
        recs = ctx.records(task, "discovery")
        store = ACTS / f"{task}{cfg.sfx}"
        model, handle = rebuild_healed(ctx, row)
        adapters.merge_lora(handle)
        meta = measure.build_store(model, ctx.tokenizer, task, recs, store, batch_size=EVAL_BS)
        lens = np.asarray(np.load(store / "lens.npy", mmap_mode="r"), dtype=np.float32)
        rungs = meta["rungs"]
        rung_idx = {r: np.array([j for j, x in enumerate(rungs) if x == r]) for r in spec.TASKS[task].rungs}
        lens_res = {"all": _lens_curve(lens, meta["answers"], np.arange(len(recs)), meta["pred"]),
                    "rungs": {r: _lens_curve(lens, meta["answers"], ix, meta["pred"]) for r, ix in rung_idx.items() if len(ix)}}
        means = torch.load(store / "means.pt", map_location="cpu", weights_only=True)
        les = measure.lesion_sweep(model, ctx.tokenizer, task, recs[:cfg.n_lesion], means, batch_size=EVAL_BS,
                                   path=OUT / f"secondary_lesion_{task}{cfg.sfx}.json", blocks=cfg.lesion_blocks)
        res = {"task": task, "lr": row["lr"], "ckpt": row["ckpt"], "n_discovery": len(recs), "lens": lens_res,
               "L_lens": lens_res["all"]["L_lens"], "base_accuracy": lens_res["all"]["final"],
               "lesion": {"n": les["n"], "base_accuracy": les["base_accuracy"], "drops": les["drops"],
                          "fragile": metrics.fragile(les["drops"], 0.05)},
               "fragile": metrics.fragile(les["drops"], 0.05), "ts": now()}
        write(final_path, res)
        shutil.rmtree(store, ignore_errors=True)
        del model
        release(ctx)
        print(f"secondary {task}: L_lens {res['L_lens']} fragile {res['fragile']}", flush=True)
    return 0


# ---------------------------------------------------------------- stage: generative

def stage_generative(ctx) -> int:
    cfg = ctx.cfg
    rule = plan_rule(cfg)
    runs = read_merged("runs", cfg.sfx, "runs")
    ks = [0, cfg.rival_k]
    coll = collect(runs, cfg.gen_tasks, ks, rule)
    path = OUT / f"generative{cfg.sfx}.json"
    doc = read_json(path) or {}
    for task in cfg.gen_tasks:
        recs = ctx.sel200(task)
        options = spec.TASKS[task].options
        entry = doc.setdefault(task, {})
        for name, k in (("unmodified", None), ("healed_k0", 0), (f"healed_k{cfg.rival_k}", cfg.rival_k)):
            if name in entry:
                continue
            if k is None:
                model, handle = ctx.fresh(), None
            else:
                rows = [r for r in coll.get(task, {}).get(k, []) if r["seed"] == 0]
                if not rows:
                    print(f"generative {task} {name}: no seed-0 run; skipped", flush=True)
                    continue
                model, handle = rebuild_healed(ctx, rows[0])
                adapters.merge_lora(handle)
            texts = heal.generate_answers(model, ctx.tokenizer, task, recs, max_new_tokens=8, batch_size=EVAL_BS)
            em = float(np.mean([heal.exact_match(t, options[r["answer"]]) for t, r in zip(texts, recs)]))
            acc, _ = heal.accuracy_and_preds(model, ctx.tokenizer, task, recs, EVAL_BS)
            entry[name] = {"n": len(recs), "exact_match": em, "first_token_accuracy": acc,
                           "examples": [{"text": t, "answer": options[r["answer"]]} for t, r in zip(texts[:5], recs[:5])]}
            write(path, doc)
            print(f"generative {task} {name}: exact {em:.3f} first-token {acc:.3f}", flush=True)
            del model
            release(ctx)
    return 0


# ---------------------------------------------------------------- stage: rivals

def stage_rivals(ctx) -> int:
    cfg = ctx.cfg
    path = OUT / f"rivals{cfg.sfx}.json"
    doc = read_json(path) or {}
    todo = [t for t in cfg.tasks if t not in doc]
    if not todo:
        return 0
    base = ctx.fresh()
    cut, _ = ctx.cut(cfg.rival_k)
    removed = list(ctx.order[:cfg.rival_k])
    for task in todo:
        recs = ctx.records(task, "discovery")
        text = "\n\n".join(spec.build_prompt(ctx.tokenizer, task, r["prompt"]) for r in recs)
        bi = measure.block_influence(base, ctx.tokenizer, text)
        sel = ctx.sel200(task)
        ce_base = heal.answer_ce(base, ctx.tokenizer, task, sel, EVAL_BS)
        ce_cut = heal.answer_ce(cut, ctx.tokenizer, task, sel, EVAL_BS)
        doc[task] = {"k": cfg.rival_k, "bi": float(np.mean([bi[j] for j in removed])), "ce_base": ce_base,
                     "ce_cut": ce_cut, "ce_rise": ce_cut - ce_base, "n_tokens_text": len(text)}
        write(path, doc)
        print(f"rivals {task}: BI {doc[task]['bi']:.4f} CE rise {doc[task]['ce_rise']:.3f}", flush=True)
    return 0


# ---------------------------------------------------------------- stage: ceiling

def stage_ceiling(ctx) -> int:
    cfg = ctx.cfg
    if not (GEN / "manifest.json").exists():
        print("skipped: general data absent (var/data/general/manifest.json)", flush=True)
        return 0
    train, dev = read_jsonl(GEN / "train.jsonl"), read_jsonl(GEN / "dev.jsonl")
    if cfg.smoke:
        train, dev = train[:32], dev[:32]
    dev_eval = dev[::max(1, len(dev) // DEV_EVAL)][:DEV_EVAL]  # dev is sorted by source: an even sample of all of them
    path = OUT / f"ceiling{cfg.sfx}.json"
    doc = read_json(path) or {"runs": {}}
    teacher = ctx.fresh().eval().requires_grad_(False)
    for seed in SEEDS:
        key = f"ceiling|k{cfg.ceiling_k}|s{seed}"
        if key in doc["runs"]:
            print(f"skip {key}: already done", flush=True)
            continue
        print(f"== {key} {now()}", flush=True)
        model, alive = ctx.cut(cfg.ceiling_k)
        state, curve, tim = heal.train_ceiling(model, teacher, ctx.tokenizer, train, dev_eval, cfg.ceiling_steps,
                                               CEILING_LR, BATCH, seed, n_evals=4)
        capability.load_trainables(model, state)
        _free(model)
        scores = {t: heal.accuracy_and_preds(model, ctx.tokenizer, t, ctx.sel200(t), EVAL_BS)[0] for t in cfg.tasks}
        doc["runs"][key] = {"key": key, "k": cfg.ceiling_k, "seed": seed, "lr": CEILING_LR, "steps": cfg.ceiling_steps,
                            "curve": curve, "timings": tim, "scores": scores, "n_alive": len(alive), "ts": now()}
        write(path, doc)
        print(f"{key}: dev_kl {tim['best_dev_kl']:.4f} ({tim['step_seconds']:.2f}s/step)", flush=True)
        del model
        release(ctx)
    return 0


# ---------------------------------------------------------------- stage: test

def stage_test(ctx) -> int:
    cfg = ctx.cfg
    rule = plan_rule(cfg)
    runs = read_merged("runs", cfg.sfx, "runs")
    ks = sorted({r["k"] for r in runs.values()})
    coll = collect(runs, cfg.tasks, ks, rule)
    path = OUT / f"test_results{cfg.sfx}.json"
    doc = read_json(path) or {"results": {}}
    ledger = OUT / "test_ledger.jsonl"
    acc = {t: {k: [r["sel_accuracy"] for r in rows] for k, rows in by_k.items()} for t, by_k in coll.items()}
    kstar = heal.kstar_by_task(acc, tuple(ks))
    for task in cfg.tasks:
        if task not in coll:
            continue
        for name, k in (("healed_k0", 0), ("healed_kstar", kstar.get(task))):
            if k is None or k not in coll[task]:
                continue
            for row in coll[task][k]:
                key = row["key"]
                if key in doc["results"]:
                    if name not in doc["results"][key]["names"]:
                        doc["results"][key]["names"].append(name)
                        write(path, doc)
                    continue
                model, handle = rebuild_healed(ctx, row)
                recs = ctx.records(task, "test")
                a, preds = heal.accuracy_and_preds(model, ctx.tokenizer, task, recs, EVAL_BS)
                rung_acc, rung_chance = rung_scores(recs, preds)
                doc["results"][key] = {"task": task, "k": k, "seed": row["seed"], "lr": row["lr"], "names": [name],
                                       "n": len(recs), "accuracy": a, "chance": spec.chance([r["answer"] for r in recs]),
                                       "rung_acc": rung_acc, "rung_chance": rung_chance, "ts": now()}
                write(path, doc)
                if cfg.ledger:
                    with ledger.open("a") as fh:
                        fh.write(json.dumps({"key": key, "names": [name], "task": task, "k": k, "seed": row["seed"],
                                             "n": len(recs), "accuracy": a, "ts": now()}, sort_keys=True) + "\n")
                print(f"test {key} ({name}): {a:.3f}", flush=True)
                del model
                release(ctx)
    return 0


# ---------------------------------------------------------------- stage: probe

def mean_tokens(ctx, task: str, n: int = 100) -> float:
    recs = ctx.records(task, "train")[:n]
    return float(np.mean([len(ctx.tokenizer.encode(spec.build_prompt(ctx.tokenizer, task, r["prompt"]),
                                                   add_special_tokens=False)) for r in recs]))


def time_steps(ctx, task: str, steps: int) -> float:
    model, alive = ctx.cut(0)
    handle = heal.add_heal(model, len(alive), RANK, 0)
    try:
        _, _, tim = heal.train_heal(model, ctx.tokenizer, task, ctx.records(task, "train"), [], handle.modules, steps,
                                    LRS[0], BATCH, 0, n_evals=0, record_steps=True, log=lambda *_: None)
    finally:
        handle.remove()
        del model
        release(ctx)
    ts = tim["step_times"]
    ts = ts[3:] if len(ts) > 6 else ts
    return float(np.mean(ts))


def time_ceiling(ctx, steps: int = 5) -> float:
    teacher = ctx.fresh().eval().requires_grad_(False)
    model, _ = ctx.cut(ctx.cfg.ceiling_k)
    params = [p for _, p in capability.full_trainable(model)]
    opt = torch.optim.AdamW(params, lr=CEILING_LR)
    ids = torch.randint(1000, (BATCH, 128), device=ctx.device) % model.config.vocab_size
    enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        heal.ceiling_step(model, teacher, enc, opt)
        if ctx.device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    del model, teacher, opt
    release(ctx)
    return float(np.mean(times[1:] if len(times) > 2 else times))


def stage_probe(ctx, speedup: float) -> int:
    cfg = ctx.cfg
    probed = tuple(cfg.tasks[:2]) if cfg.smoke else (ROUTE_LONG, ROUTE_SHORT)
    step_s = {t: time_steps(ctx, t, cfg.probe_steps) for t in probed}
    for t, s in step_s.items():
        print(f"probe {t}: {s:.3f}s/step at k=0", flush=True)
    tokens = {t: mean_tokens(ctx, t) for t in cfg.tasks}
    ceil_s = time_ceiling(ctx)
    print(f"probe ceiling: {ceil_s:.3f}s/step at k={cfg.ceiling_k}", flush=True)
    proj = heal.project_hours(step_s, tokens, ceil_s, ctx.n_layers, cfg.ks, cfg.steps, speedup=speedup)
    rule = "fallback" if proj["main"]["sweep"] > SWEEP_HOURS_CAP else "per_k"
    plan = {"lr_rule": rule, "sweep_hours_cap": SWEEP_HOURS_CAP, "step_seconds": step_s, "mean_tokens": tokens,
            "ceiling_step_seconds": ceil_s, "projection": proj, "ts": now(),
            "note": "rule compares the main-rule sweep projection (training plus in-run scoring) with the cap"}
    write(OUT / f"plan{cfg.sfx}.json", plan)
    print(f"projected hours: main {proj['main']}  fallback {proj['fallback']}\nlr_rule: {rule}", flush=True)
    return 0


# ---------------------------------------------------------------- stage: table

def _stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals)), "n": len(vals)}


def _describe(acc_by_k: dict, chance, unmodified, grid) -> dict:
    cells = {k: _stat(v) for k, v in acc_by_k.items() if v}
    out = {"chance": chance, "unmodified": unmodified, "acc": cells, "healed_k0": None, "k_star": None,
           "score_at_k_star": None, "rel_drop": None}
    if 0 not in cells:
        return out
    base = cells[0]["mean"]
    ks = metrics.k_star({k: v for k, v in acc_by_k.items() if v}, base, grid)
    out.update(healed_k0=base, k_star=ks)
    if ks is not None:
        out["score_at_k_star"] = cells[ks]["mean"]
        if chance is not None and base - chance > 1e-9:
            out["rel_drop"] = (base - cells[ks]["mean"]) / (base - chance)
    return out


def load_all(cfg) -> dict:
    sfx = cfg.sfx
    d = {"runs": read_merged("runs", sfx, "runs"), "cut": read_merged("cutonly", sfx, "cut"),
         "plan": read_json(OUT / f"plan{sfx}.json") or {}, "rivals": read_json(OUT / f"rivals{sfx}.json") or {},
         "generative": read_json(OUT / f"generative{sfx}.json") or {},
         "ceiling": (read_json(OUT / f"ceiling{sfx}.json") or {}).get("runs", {}),
         "test": (read_json(OUT / f"test_results{sfx}.json") or {}).get("results", {}),
         "secondary": {}, "map": {}}
    for t in cfg.tasks:
        d["secondary"][t] = read_json(OUT / f"secondary_{t}{sfx}.json")
        d["map"][t] = read_json(MAP_DIR / f"map_{t}{sfx}.json") or read_json(MAP_DIR / f"map_{t}.json")
    return d


def tmatrix(coll, cut, tasks, ks) -> dict:
    """{k: {a: {b: T}}}: seed mean of b's score after healing on a, minus b's cut-only score at the same k."""
    out = {}
    for k in ks:
        cut_k = {b: cut[f"{b}|k{k}"]["acc200"] for b in tasks if f"{b}|k{k}" in cut}
        after = {}
        for a in tasks:
            rows = coll.get(a, {}).get(k)
            if not rows:
                continue
            after[a] = {b: float(np.mean([r["transfer"][b] for r in rows if b in r["transfer"]]))
                        for b in cut_k if b != a and all(b in r["transfer"] for r in rows)}
        if after and cut_k:
            out[k] = metrics.transfer(after, cut_k)
    return out


def build_summary(cfg, d: dict) -> dict:
    rule = d["plan"].get("lr_rule", "per_k")
    runs = d["runs"]
    ks_all = sorted({r["k"] for r in runs.values()})
    coll = collect(runs, cfg.tasks, ks_all, rule)
    tasks = [t for t in cfg.tasks if t in coll]
    ks = sorted({k for t in tasks for k in coll[t]})
    grid = tuple(ks)
    hk = cfg.headline_k if cfg.headline_k in ks else (max(ks) if ks else None)
    s = {"lr_rule": rule, "tasks": tasks, "ks": ks, "headline_k": hk, "smoke": cfg.smoke, "rows": []}
    acc = {t: {k: [r["sel_accuracy"] for r in rows] for k, rows in coll[t].items()} for t in tasks}
    for t in tasks:
        ts = spec.TASKS[t]
        first = next(iter(coll[t].values()))[0]
        cut0 = d["cut"].get(f"{t}|k0")
        m = d["map"].get(t) or {}
        sec = d["secondary"].get(t) or {}
        row = {"task": t, "number": ts.number, "kind": ts.kind, "rung": None, "lr": {k: rows[0]["lr"] for k, rows in coll[t].items()},
               **_describe(acc[t], first["chance"], cut0["acc"] if cut0 else None, grid),
               "L_on": m.get("L_on"), "L_lens_phase0": m.get("L_lens"), "fragile_phase0": m.get("fragile"),
               "L_lens_healed": sec.get("L_lens"), "fragile_healed": sec.get("fragile")}
        s["rows"].append(row)
        for rung in ts.rungs:
            r_acc = {k: [r["rung_acc"][rung] for r in rows if rung in r["rung_acc"]] for k, rows in coll[t].items()}
            if not any(r_acc.values()):
                continue
            r_cut = (cut0 or {}).get("rung_acc", {}).get(rung)
            s["rows"].append({"task": t, "number": ts.number, "kind": ts.kind, "rung": rung,
                              **_describe(r_acc, first["rung_chance"].get(rung), r_cut, grid)})
    # transfer
    s["transfer"] = {str(k): v for k, v in tmatrix(coll, d["cut"], tasks, ks).items()}
    if hk is not None and str(hk) in s["transfer"]:
        cells = [{"from": a, "to": b, "T": v, "tag": f"{spec.TASKS[a].kind}->{spec.TASKS[b].kind}"}
                 for a, row in s["transfer"][str(hk)].items() for b, v in row.items() if v >= 0.05 - 1e-9]
        s["transfer_cells_ge_5pts"] = sorted(cells, key=lambda c: -c["T"])
    # ceiling control at the headline k
    if d["ceiling"] and hk is not None:
        cr = list(d["ceiling"].values())
        cut_hk = {b: d["cut"].get(f"{b}|k{hk}", {}).get("acc200") for b in tasks}
        s["ceiling"] = {"k": cr[0]["k"], "seeds": [r["seed"] for r in cr],
                        "tasks": {b: {"ceiling_acc200": _stat([r["scores"].get(b) for r in cr]),
                                      "cut_only_acc200": cut_hk.get(b),
                                      "healed_on_self_acc200": _stat([r["sel_accuracy200"] for r in coll[b].get(hk, [])]) if b in coll else None}
                                  for b in tasks}}
    s["generative"] = d["generative"]
    # test
    tests = {}
    for r in d["test"].values():
        for name in r["names"]:
            tests.setdefault(r["task"], {}).setdefault(name, []).append(r)
    s["test"] = {t: {name: {"k": rs[0]["k"], "accuracy": _stat([r["accuracy"] for r in rs]), "chance": rs[0]["chance"]}
                     for name, rs in by.items()} for t, by in tests.items()}
    # verdict numbers
    verdict = {}
    preds = {"L_on": {t: (d["map"].get(t) or {}).get("L_on") for t in tasks},
             "BI": {t: (d["rivals"].get(t) or {}).get("bi") for t in tasks},
             "loss_rise": {t: (d["rivals"].get(t) or {}).get("ce_rise") for t in tasks}}
    h1 = {name: heal.bootstrap_rho(acc, p, grid) for name, p in preds.items()}
    for name, r in h1.items():
        r.pop("kstar", None)
    beats = {}
    for name in ("BI", "loss_rise"):
        a, b = h1["L_on"]["rho"], h1[name]["rho"]
        beats[name] = None if a is None or b is None else abs(a) > abs(b)
    verdict["H-R1"] = {**h1, "L_on_beats": beats}
    kst = {r["task"]: r["k_star"] for r in s["rows"] if r["rung"] is None}
    pairs = []
    for a, b in spec.PAIRS:
        if kst.get(a) is not None and kst.get(b) is not None:
            pairs.append({"appraise": a, "execute": b, "diff": kst[a] - kst[b]})
    verdict["H-R2"] = {"pairs": pairs, "n_pairs_ge_3": sum(p["diff"] >= 3 for p in pairs)}
    ladders = {}
    for t in tasks:
        if spec.TASKS[t].rungs:
            ks_r = [next((r["k_star"] for r in s["rows"] if r["task"] == t and r["rung"] == g), None) for g in spec.TASKS[t].rungs]
            ok = None not in ks_r and len(ks_r) > 1
            ladders[t] = {"rungs": list(spec.TASKS[t].rungs), "k_star": ks_r,
                          "non_increasing": ok and all(x >= y for x, y in zip(ks_r, ks_r[1:])),
                          "last_below_first": ok and ks_r[-1] < ks_r[0]}
    verdict["H-R3"] = ladders
    verdict["H-R4"] = {"k": hk, "cells_ge_5pts": len(s.get("transfer_cells_ge_5pts", [])),
                       "ceiling_present": "ceiling" in s}
    s["verdict"] = verdict
    return s


def _f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def _cell(c):
    return "-" if not c else f"{c['mean']:.3f} ({c['min']:.3f}-{c['max']:.3f})"


def render_tables(s: dict) -> str:
    ks = s["ks"]
    out = [f"# Atlas Phase 1 tables{' (smoke)' if s['smoke'] else ''}", "",
           f"lr rule: {s['lr_rule']}; headline k: {s['headline_k']}. Cells are seed mean (min-max) of selection accuracy.", "",
           "## Accuracy by k", "",
           "| task | rung | chance | unmod | healed k0 | " + " | ".join(f"k{k}" for k in ks) +
           " | k* | rel drop | L_on | L_lens (healed) | fragile (healed) |",
           "|" + "---|" * (10 + len(ks))]
    for r in s["rows"]:
        out.append(f"| {r['task']} | {r['rung'] or ''} | {_f(r['chance'])} | {_f(r['unmodified'])} | {_f(r['healed_k0'])} | " +
                   " | ".join(_cell(r['acc'].get(k)) for k in ks) +
                   f" | {r['k_star']} | {_f(r['rel_drop'])} | {r.get('L_on')} | {r.get('L_lens_healed')} | {r.get('fragile_healed')} |")
    out += ["", "## Transfer T[a,b] in points (healed on a, scored on b, minus the cut-only score at the same k)"]
    for k, mat in s["transfer"].items():
        cols = sorted({b for row in mat.values() for b in row})
        out += ["", f"### k={k}", "", "| a \\ b | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
        for a, row in mat.items():
            out.append(f"| {a} | " + " | ".join("-" if b not in row else f"{100 * row[b]:+.1f}" for b in cols) + " |")
    if s.get("transfer_cells_ge_5pts"):
        out += ["", f"Cells of at least 5 points at k={s['headline_k']}:", ""]
        out += [f"- {c['from']} -> {c['to']} ({c['tag']}): {100 * c['T']:+.1f}" for c in s["transfer_cells_ge_5pts"]]
    if "ceiling" in s:
        c = s["ceiling"]
        out += ["", f"## Ceiling control (k={c['k']}, seeds {c['seeds']}; accuracy on the first 200 selection records)", "",
                "| task | ceiling | cut only | healed on own task |", "|---|---|---|---|"]
        for t, v in c["tasks"].items():
            out.append(f"| {t} | {_cell(v['ceiling_acc200'])} | {_f(v['cut_only_acc200'])} | {_cell(v['healed_on_self_acc200'])} |")
    else:
        out += ["", "## Ceiling control", "", "Not run (general data absent or stage not reached)."]
    out += ["", "## Generative check (first 200 selection records, greedy, 8 new tokens)", ""]
    if s["generative"]:
        out += ["| task | model | exact match | first-token acc |", "|---|---|---|---|"]
        out += [f"| {t} | {n} | {_f(v['exact_match'])} | {_f(v['first_token_accuracy'])} |"
                for t, by in s["generative"].items() for n, v in by.items()]
    else:
        out.append("Not run.")
    out += ["", "## Test split (once per named model)", ""]
    if s["test"]:
        out += ["| task | model | k | accuracy (seeds) | chance |", "|---|---|---|---|---|"]
        out += [f"| {t} | {n} | {v['k']} | {_cell(v['accuracy'])} | {_f(v['chance'])} |"
                for t, by in s["test"].items() for n, v in by.items()]
    else:
        out.append("Not run.")
    v = s["verdict"]
    out += ["", "## Verdict numbers", "", "H-R1 (Spearman rho against k*, bootstrap 2,000 draws, seed 9):", ""]
    for name in ("L_on", "BI", "loss_rise"):
        r = v["H-R1"][name]
        out.append(f"- {name}: rho {_f(r['rho'])}, interval [{_f(r['lo'])}, {_f(r['hi'])}], tasks {r['n_tasks']}")
    out.append(f"- L_on beats: {v['H-R1']['L_on_beats']}")
    out += ["", "H-R2 (k*(appraise) - k*(execute)):", ""]
    out += [f"- {p['appraise']} / {p['execute']}: {p['diff']}" for p in v["H-R2"]["pairs"]]
    out.append(f"- pairs at least 3: {v['H-R2']['n_pairs_ge_3']}")
    out += ["", "H-R3 (rung k*):", ""]
    out += [f"- {t}: {l['k_star']} non-increasing {l['non_increasing']}, last below first {l['last_below_first']}"
            for t, l in v["H-R3"].items()]
    out += ["", f"H-R4: k={v['H-R4']['k']}, cells at least 5 points: {v['H-R4']['cells_ge_5pts']}, "
                f"ceiling present: {v['H-R4']['ceiling_present']}", ""]
    return "\n".join(out)


def stage_table(cfg) -> int:
    s = build_summary(cfg, load_all(cfg))
    write(OUT / f"summary{cfg.sfx}.json", s)
    md = render_tables(s)
    (OUT / f"tables{cfg.sfx}.md").write_text(md + "\n")
    print(md, flush=True)
    return 0


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--shard", default=None, help="I/N")
    ap.add_argument("--seed", type=int, default=0, choices=SEEDS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--parallel-speedup", type=float, default=1.0, help="probe: assumed gain from side-by-side shards")
    args = ap.parse_args(argv)
    cfg = make_cfg(args.smoke)
    shard = parse_shard(args.shard)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "table":
        return stage_table(cfg)
    ctx = Ctx(cfg)
    if args.stage == "probe":
        return stage_probe(ctx, args.parallel_speedup)
    if args.stage == "sweep":
        return stage_sweep(ctx, args.seed, shard)
    if args.stage == "cutonly":
        return stage_cutonly(ctx, shard)
    if args.stage == "secondary":
        return stage_secondary(ctx, shard)
    return {"generative": stage_generative, "rivals": stage_rivals, "ceiling": stage_ceiling,
            "test": stage_test}[args.stage](ctx)


if __name__ == "__main__":
    sys.exit(main())
