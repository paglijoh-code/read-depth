"""M9 atlas Phase 2, stage C1: the cut-and-heal sweep on Qwen3-1.7B, whole-answer training and exact-match scoring for every
task, plus the own-order arm (docs/ATLAS-phase2-stageC.md, section C1).

  uv run python scripts/atlas3_cuts.py --stage {probe,sweep,ownorder,test,table} [--shard I/N] [--seed S] [--smoke]
  uv run python scripts/atlas3_cuts.py --stage d2sweep --arm {alllin,contig,long} --seed S [--shard I/N]; --stage d2table
  uv run python scripts/atlas3_cuts.py --stage e4sweep --arm {late17,late15} --seed S [--shard I/N]; --stage e4table
  uv run python scripts/atlas3_cuts.py --stage {healedlesion --task T,ownorder2,test2,table2}     # H2-E2 follow-up
  uv run python scripts/atlas3_cuts.py --model smol17 --stage {probe,sweep,healedprobe --task T,d3table}   # step D3

Reuses scripts/atlas_cuts.py (`Ctx`, shard files, lr winner, `collect`, `rebuild_healed`) and scripts/atlas2_cuts.py (the
statistics) by import; `configure()` points the Phase 1 module at the 1.7B model, the 1.7B shared order and this stage's
directories (var/results/atlas3/c1, var/ckpt/atlas3, var/acts/atlas3). Every stage resumes (a heal whose key is already in a
shard file is skipped). `--model` picks a profile (PROFILES): qwen17 (default, everything above) or smol17 (step D3,
SmolLM2-1.7B-Instruct: its own maps, results, checkpoints and activations folders). --smoke uses 2 tasks, k in {0, 2}, 4 steps, 16 records, `_smoke` names, and never writes the ledger.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import atlas2_cuts as B  # noqa: E402
import atlas_cuts as ac  # noqa: E402

from surgery import adapters  # noqa: E402
from surgery.atlas import heal, measure, spec  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"
MAPS = Path("var/results/atlas3/1.7B")
ORDER_PATH = MAPS / "order.json"
OUT = Path("var/results/atlas3/c1")
CKPT = Path("var/ckpt/atlas3")
ACTS = Path("var/acts/atlas3")
BASELINES = (Path("var/results/atlas/baseline.json"), Path("var/results/atlas2/baseline.json"),
             Path("var/results/atlas3/baseline.json"))
STAGES = ["probe", "sweep", "ownorder", "test", "table", "healedlesion", "ownorder2", "test2", "table2",
          "healedprobe"]
D2_STAGES = ["d2sweep", "d2table"]               # kept out of STAGES so the earlier stages' order is untouched
E4_STAGES = ["e4sweep", "e4table"]
D3_STAGES = ["d3table"]
SEEDS = ac.SEEDS
KS = (0, 9, 12, 16)
KS_FALLBACK = (0, 12, 16)
LONG_TASK = "arith_long"

# model profiles: one place for everything that differs between the model families the sweep runs on
PROFILES = {
    "qwen17": SimpleNamespace(name="qwen17", model=MODEL, maps=MAPS, out=OUT, ckpt=CKPT, acts=ACTS, n_layers=28, kcut=12,
                              ks=KS),
    "smol17": SimpleNamespace(name="smol17", model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                              maps=Path("var/results/atlas3/smol17"), out=Path("var/results/atlas3/d3"),
                              ckpt=Path("var/ckpt/atlas3/d3"), acts=Path("var/acts/atlas3/d3"), n_layers=24, kcut=10,
                              ks=(0, 10)),
}
PROFILE_STAGES = {"qwen17": None, "smol17": ("probe", "sweep", "healedprobe", "d3table")}   # None: every stage


def resolve_profile(name: str = "qwen17", smoke: bool = False) -> SimpleNamespace:
    """The named profile. A smoke run of a non-default profile keeps its files apart: every folder gets `_smoke` appended."""
    p = PROFILES[name]
    if not smoke or name == "qwen17":
        return p
    sm = lambda d: Path(str(d) + "_smoke")      # noqa: E731
    return SimpleNamespace(**{**vars(p), "maps": sm(p.maps), "out": sm(p.out), "ckpt": sm(p.ckpt), "acts": sm(p.acts)})


def check_blocks(n_layers: int, profile) -> None:
    """The model's block count (from its config) must be the profile's; every number in the plan scales with it."""
    if profile is not None and n_layers != profile.n_layers:
        raise SystemExit(f"profile {profile.name} expects {profile.n_layers} blocks, the loaded model has {n_layers}")
HEAL_TASKS = tuple(sorted((t.name for t in spec.TASKS.values() if t.number <= 25), key=lambda n: spec.TASKS[n].number))
OWN_DOING = ("arith", "code_output")
OWN_JUDGING = ("route_work", "code_uses_first")
OWN_TASKS = OWN_DOING + OWN_JUDGING
WORD_BASE = ("arith_order", "syntax_first_close", "code_uses_first")
WORD_CANDIDATES = ("code_traces_first", "syntax_outer_pair", "bind_compare")   # added when clean in gate.json
TASK18, TASK24 = "hard_target_read", "target_read_bare"
N_LRSEL_EVAL = 200          # train_whole's eval_n
MNT_MARGIN = 3              # max_new_tokens = longest answer tokens + end-of-turn + 2


# step D2 (docs/ATLAS-phase2-stageC.md): healing / cut-shape / budget arms on deep and shallow tasks
D2_DEEP = ("code_output", "arith", "bind_compare", "rule")
D2_SHALLOW = ("code_traces_first", "route_judgement")
D2_CONTIG = tuple(range(15, 27))                 # blocks 15..26: 12 blocks, leaves 0-14 and 27
D2_OUT = Path("var/results/atlas3/d2")
D2_ARMS = {                                      # modules, rank, step multiple of cfg.steps
    "alllin": SimpleNamespace(modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                              rank=2, mult=1),
    "contig": SimpleNamespace(modules=("q_proj", "v_proj"), rank=10, mult=1),
    "long": SimpleNamespace(modules=("q_proj", "v_proj"), rank=10, mult=4),
}
D2_K0_ARMS = ("alllin", "long")                  # arms that also heal k=0 (seed 0)
D2_PARAM_TOL = 0.10
D2_ARTIFACT, D2_HOLDS = 0.30, 0.15
OWN_K0_TOL = 0.05

# step E4 (docs/ATLAS-robustness.md, "E4: late-only cuts"): two contiguous 10-block cuts, healed as the D2 contig arm
E4_OUT = Path("var/results/atlas3/e4")
E4_BLOCKS = {"late17": tuple(range(17, 27)), "late15": tuple(range(15, 25))}   # leaves 0-16 + 27; leaves 0-14 + 25-27
E4_ARMS = {a: D2_ARMS["contig"] for a in E4_BLOCKS}                              # q/v rank 10, 300 steps, two lrs at seed 0
E4_K = 10
E1_PATH = Path("var/results/atlas3/e1.json")     # the paper's main rho: its "1.7B" tasks carry kind and L_on_healed
E4_HOLDS, E4_WEAK, E4_CEILING, E4_MIN_LOW = -0.40, -0.20, 0.9, 3
E4_PERM, E4_PERM_SEED = 10_000, 0


def configure(profile=None) -> None:
    """Point the Phase 1 module at this stage (done in main, never at import). `profile` (see PROFILES) rebinds this module's
    MODEL, MAPS, ORDER_PATH, OUT, CKPT and ACTS first; none: the module's current values (C1's)."""
    global MODEL, MAPS, ORDER_PATH, OUT, CKPT, ACTS
    if profile is not None:
        MODEL, MAPS, OUT, CKPT, ACTS = profile.model, profile.maps, profile.out, profile.ckpt, profile.acts
        ORDER_PATH = MAPS / "order.json"
    ac.MODEL, ac.ORDER_PATH = MODEL, ORDER_PATH
    ac.OUT, ac.CKPT, ac.ACTS = OUT, CKPT, ACTS
    B.OUT, B.CKPT, B.ACTS = OUT, CKPT, ACTS


# D3 run 1 (2026-11-14) ran out of memory on the first SmolLM2 heal (route_judgement, batch 8, fp32, 44 GB): its 32 KV heads
# and wider MLP need more activation memory than Qwen3-1.7B. Training batches above this many tokens use gradient
# checkpointing, which recomputes activations in the backward pass and leaves the loss and gradients unchanged.
D3_CKPT_TOKENS = 4000


def make_cfg(smoke: bool, profile=None) -> SimpleNamespace:
    """`profile` (resolve_profile) sets the cut level, k grid and, for smol17, the 24 table tasks only (step D3: no own-order
    arm, so no own-order heals in the probe's projection). Unset or qwen17: C1's configuration."""
    d3 = profile is not None and profile.name != "qwen17"
    if smoke:
        return SimpleNamespace(smoke=True, sfx="_smoke", tasks=("arith", "code_uses_first"), ks=(0, 2), steps=4, n=16,
                               kcut=2, own_tasks=("arith", "code_uses_first"), ledger=False, probe_steps=3,
                               table_tasks=("arith", "code_uses_first"), n_perm=200, boots=20, probe_records=64,
                               d2_deep=("arith",), d2_shallow=("code_uses_first",), d2_contig=D2_CONTIG[:2],
                               e4_blocks={a: b[:2] for a, b in E4_BLOCKS.items()}, e4_k=2, e4_tasks=("arith", "code_uses_first"),
                               n_own=0 if d3 else 12, fallback_grid=None if d3 else KS_FALLBACK,
                               ckpt_tokens=0 if d3 else None)
    table = tuple(t for t in HEAL_TASKS if t != LONG_TASK)
    return SimpleNamespace(smoke=False, sfx="", ckpt_tokens=D3_CKPT_TOKENS if d3 else None, tasks=table if d3 else HEAL_TASKS, ks=profile.ks if d3 else KS, steps=300,
                           n=None, kcut=profile.kcut if d3 else 12, own_tasks=OWN_TASKS,
                           ledger=True, probe_steps=5, table_tasks=table,
                           n_perm=B.N_PERM, boots=500, probe_records=None,
                           d2_deep=D2_DEEP, d2_shallow=D2_SHALLOW, d2_contig=D2_CONTIG,
                           e4_blocks=E4_BLOCKS, e4_k=E4_K, e4_tasks=table,
                           n_own=0 if d3 else 12, fallback_grid=None if d3 else KS_FALLBACK)


# ---------------------------------------------------------------- answer length

def longest_answer_tokens(tok, task: str) -> int:
    """Longest tokenised answer text of the task over all five splits (end-of-turn not counted)."""
    best = 0
    for split in spec.SPLITS:
        for r in spec.read_records(task, split):
            best = max(best, len(tok.encode(heal.answer_text(task, r), add_special_tokens=False)))
    return best


def max_new_tokens(ctx, task: str) -> int:
    """Generation budget for exact match: the longest answer, the end-of-turn token, and two spare tokens."""
    cache = ctx.__dict__.setdefault("_mnt", {})
    if task not in cache:
        cache[task] = longest_answer_tokens(ctx.tokenizer, task) + MNT_MARGIN
    return cache[task]


# ---------------------------------------------------------------- own order

def own_order(lesion: dict, n_layers: int = 28) -> list[int]:
    """Blocks 1..n-2 (block 0 and the last block may never be cut) from least to most damaging by the single-block lesion
    sweep (`drop` per block), ties broken by the lower block index. A cut of k blocks takes the first k."""
    drops = {int(r["block"]): float(r["drop"]) for r in lesion["rows"]}
    cand = [b for b in range(1, n_layers - 1) if b in drops]
    if len(cand) != n_layers - 2:
        raise ValueError(f"lesion sweep lacks blocks: has {sorted(drops)}")
    return sorted(cand, key=lambda b: (drops[b], b))


def load_own_order(task: str, n_layers: int = 28) -> list[int]:
    return own_order(json.loads((MAPS / f"lesion_{task}.json").read_text()), n_layers)


class OwnCtx(ac.Ctx):
    """A `Ctx` whose k-cut removes the task's own `order[:k]` instead of the shared order. Shares the pristine model."""

    def __init__(self, base: ac.Ctx, order):
        super().__init__(base.cfg, model=base.pristine, tok=base.tokenizer, order=list(order), device=base.device)
        self._rec = base._rec
        self.__dict__["_mnt"] = base.__dict__.setdefault("_mnt", {})


# ---------------------------------------------------------------- one heal

def own_ckpt_path(cfg, task, k, seed, lr) -> Path:
    return ac.CKPT / f"own_{task}_k{k}_s{seed}_lr{lr:g}{cfg.sfx}.pt"


def run_whole_heal(ctx, task: str, k: int, seed: int, lr: float, own: bool = False, removed=None, ckpt_fn=None,
                   arm=None, steps: int | None = None, n_evals: int = 4) -> dict:
    """Cut, LoRA, `heal.train_whole` (whole answer plus end-of-turn, checkpoint chosen by lrsel exact match), then exact
    match on `selection`. The row's `sel_accuracy` is that exact match, so `ac.collect` and `ac._describe` read it.
    `arm` (an object with `modules` and `rank`) swaps the q/v rank-10 LoRA for that module set and rank; `steps` overrides
    `cfg.steps`. Both unset: exactly the C1 heal."""
    cfg, tok = ctx.cfg, ctx.tokenizer
    steps = cfg.steps if steps is None else steps
    rank = ac.RANK if arm is None else arm.rank
    key = ac.run_key(task, k, seed, lr)
    mnt = max_new_tokens(ctx, task)
    print(f"== {'own|' if own else ''}{key} {ac.now()} (max_new_tokens {mnt})", flush=True)
    model, alive = ctx.cut(k)
    if arm is None:
        handle = heal.add_heal(model, len(alive), rank, seed)
    else:
        torch.manual_seed(seed)
        handle = adapters.add_lora(model, list(range(len(alive))), modules=arm.modules, r=rank, scale=2.0)
    try:
        state, curve, tim = heal.train_whole(
            model, tok, task, ctx.records(task, "train"), ctx.records(task, "lrsel"), handle.modules, steps, lr,
            ac.BATCH, seed, n_evals=n_evals, eval_batch_size=ac.EVAL_BS, max_new_tokens=mnt,
            ckpt_tokens=getattr(ctx.cfg, "ckpt_tokens", None))
        adapters.load_adapter_state(handle.modules, state)
        t0 = time.perf_counter()
        recs = ctx.records(task, "selection")
        em, texts = heal.exact_match_scores(model, tok, task, recs, max_new_tokens=mnt, batch_size=ac.EVAL_BS)
        tim["score_seconds"] = time.perf_counter() - t0
        cp = (ckpt_fn or (own_ckpt_path if own else ac.ckpt_path))(cfg, task, k, seed, lr)
        ac.save_ckpt(cp, state, {"key": key, "task": task, "k": k, "seed": seed, "lr": lr, "rank": rank,
                                 "n_alive": len(alive), "best_step": tim["best_step"], "own": own})
        row = {"key": key, "task": task, "k": k, "seed": seed, "lr": lr, "steps": steps, "n_alive": len(alive),
               "curve": curve, "best_lrsel": tim["best_lrsel"], "best_step": tim["best_step"], "n": len(recs),
               "sel_accuracy": em, "chance": spec.chance([r["answer"] for r in recs]), "max_new_tokens": mnt,
               "examples": [{"text": t, "answer": heal.answer_text(task, r)} for t, r in zip(texts[:5], recs[:5])],
               "timings": {a: b for a, b in tim.items() if a not in ("best_lrsel", "best_step")},
               "ckpt": str(cp), "ts": ac.now()}
        if own:
            row["removed"] = list(removed)
        if arm is not None:
            row.update(rank=rank, modules=list(arm.modules))
    finally:
        handle.remove()
        del model
        ac.release(ctx)
    print(f"{key}: lrsel exact {row['best_lrsel']:.3f} (step {row['best_step']}) sel exact {em:.3f} "
          f"({tim['step_seconds']:.2f}s/step, train {tim['train_seconds']:.0f}s, score {tim['score_seconds']:.0f}s)",
          flush=True)
    return row


# ---------------------------------------------------------------- stage: sweep

def stage_sweep(ctx, seed: int, shard) -> int:
    """Seed 0 runs both lrs per (task, k) unit; seeds 1 and 2 run the seed-0 winner (tie: 3e-4)."""
    cfg, (i, n) = ctx.cfg, shard
    path = ac.shard_path("runs", cfg.sfx, i, n)
    mine = ac.read_json(path) or {"shard": [i, n], "runs": {}}
    print(f"sweep seed {seed} shard {i}/{n}", flush=True)
    missing = []
    for ti, task in enumerate(cfg.tasks):
        for ki, k in enumerate(cfg.ks):
            if ac.unit_shard(ti, ki, n, "per_k") != i:
                continue
            merged = {**ac.read_merged("runs", cfg.sfx, "runs"), **mine["runs"]}
            lrs = ac.unit_lrs(merged, task, k, seed, "per_k")
            if not lrs:
                missing.append(f"{task}|k{k}")
                print(f"no seed-0 winner for {task} k{k}; skipped", flush=True)
                continue
            for lr in lrs:
                key = ac.run_key(task, k, seed, lr)
                if key in merged:
                    print(f"skip {key}: already done", flush=True)
                    continue
                row = run_whole_heal(ctx, task, k, seed, lr)
                mine["runs"][key] = row
                ac.write(path, mine)
    if missing:
        print(f"sweep incomplete: {len(missing)} units lack a seed-0 winner: {missing}", flush=True)
        return 1
    return 0


# ---------------------------------------------------------------- stage: ownorder

def stage_ownorder(ctx, shard) -> int:
    """Own-order arm: each task at k=cfg.kcut, its own least-damaging blocks cut, lr = the task's seed-0 winner at that k in
    the main sweep, seeds 0, 1, 2. Needs the sweep's seed 0 first. Shards by task."""
    cfg, (i, n) = ctx.cfg, shard
    path = ac.shard_path("ownorder", cfg.sfx, i, n)
    mine = ac.read_json(path) or {"shard": [i, n], "runs": {}}
    sweep = ac.read_merged("runs", cfg.sfx, "runs")
    missing, k = [], cfg.kcut
    for ti, task in enumerate(cfg.own_tasks):
        if ti % n != i:
            continue
        lr = ac.winner_lr(sweep, task, k, "per_k")
        if lr is None:
            missing.append(f"{task}|k{k}")
            print(f"no sweep winner for {task} k{k}; skipped", flush=True)
            continue
        order = load_own_order(task, ctx.n_layers)
        removed = order[:k]
        print(f"ownorder {task}: cut blocks {sorted(removed)} (least damaging first: {removed}), lr {lr:g}", flush=True)
        octx = OwnCtx(ctx, order)
        for seed in SEEDS:
            have = {**ac.read_merged("ownorder", cfg.sfx, "runs"), **mine["runs"]}
            key = ac.run_key(task, k, seed, lr)
            if key in have:
                print(f"skip own|{key}: already done", flush=True)
                continue
            mine["runs"][key] = run_whole_heal(octx, task, k, seed, lr, own=True, removed=removed)
            ac.write(path, mine)
    if missing:
        print(f"ownorder incomplete: {missing}", flush=True)
        return 1
    return 0


# ---------------------------------------------------------------- stage: test

def test_exact(ctx, row: dict, task: str) -> tuple[float, int]:
    model, handle = ac.rebuild_healed(ctx, row)
    adapters.merge_lora(handle)
    recs = ctx.records(task, "test")
    em, _ = heal.exact_match_scores(model, ctx.tokenizer, task, recs, max_new_tokens=max_new_tokens(ctx, task),
                                    batch_size=ac.EVAL_BS)
    del model
    ac.release(ctx)
    return em, len(recs)


def stage_test(ctx) -> int:
    """Test-split exact match, once per model: every (task, k) winner (all seeds) as `healed_k{k}`, every own-order model as
    `ownorder_k{k}`. Written to test_results.json and (not in smoke) the append-only test_ledger.jsonl."""
    cfg = ctx.cfg
    runs = ac.read_merged("runs", cfg.sfx, "runs")
    own = ac.read_merged("ownorder", cfg.sfx, "runs")
    path = ac.OUT / f"test_results{cfg.sfx}.json"
    doc = ac.read_json(path) or {"results": {}}
    ledger = ac.OUT / "test_ledger.jsonl"
    coll = ac.collect(runs, cfg.tasks, cfg.ks, "per_k")

    def record(key, name, task, k, row):
        em, n = test_exact(row["_ctx"], row, task)
        doc["results"][key] = {"task": task, "k": k, "seed": row["seed"], "lr": row["lr"], "names": [name], "n": n,
                               "accuracy": em, "chance": spec.chance([r["answer"] for r in ctx.records(task, "test")]),
                               "ts": ac.now()}
        ac.write(path, doc)
        if cfg.ledger:
            with ledger.open("a") as fh:
                fh.write(json.dumps({"key": key, "names": [name], "task": task, "k": k, "seed": row["seed"], "n": n,
                                     "accuracy": em, "ts": ac.now()}, sort_keys=True) + "\n")
        print(f"test {key} ({name}): {em:.3f}", flush=True)

    for task in cfg.tasks:
        for k, rows in coll.get(task, {}).items():
            for row in rows:
                if row["key"] not in doc["results"]:
                    record(row["key"], f"healed_k{k}", task, k, {**row, "_ctx": ctx})
    for task in cfg.own_tasks:
        octx = None
        for row in sorted((r for r in own.values() if r["task"] == task), key=lambda r: r["seed"]):
            key = f"own|{row['key']}"
            if key in doc["results"]:
                continue
            octx = octx or OwnCtx(ctx, load_own_order(task, ctx.n_layers))
            record(key, f"ownorder_k{row['k']}", task, row["k"], {**row, "_ctx": octx})
    return 0


# ---------------------------------------------------------------- readings (H2-A, H2-B as stage B; H2-E new)

def gate_clean() -> list[str]:
    """Tasks of 22-25 that are clean in the 1.7B gate (BoW - chance <= 0.10)."""
    gate = ac.read_json(MAPS / "gate.json") or {}
    return [r["task"] for r in gate.get("rows", []) if r.get("clean")]


def word_tasks(clean=None) -> tuple[str, ...]:
    clean = gate_clean() if clean is None else clean
    return WORD_BASE + tuple(t for t in WORD_CANDIDATES if t in clean)


def reading_word(r12: dict, tasks) -> dict:
    vals = {t: r12.get(t) for t in tasks}
    if not vals or any(v is None for v in vals.values()):
        return {"tasks": vals, "mean": None, "label": "undetermined"}
    m = float(np.mean(list(vals.values())))
    return {"tasks": vals, "mean": m, "label": "not_explained" if m >= 0.80 else ("rested_on" if m <= 0.50 else "mixed")}


DOING_MIN_GAIN, JUDGING_MAX_GAIN = 0.30, 0.10
H2E_TEXT = "doing tasks need the right blocks kept; judging tasks don't care which"


def h2e(r12_shared: dict, r12_own: dict, doing=OWN_DOING, judging=OWN_JUDGING) -> dict:
    """Pre-stated reading: R12 under own order minus R12 under the shared order, per task. Both doing tasks gain at least
    0.30 and both judging tasks gain under 0.10 -> the doing/judging sentence; otherwise the failed conditions are named."""
    gains = {}
    for t in (*doing, *judging):
        a, b = r12_own.get(t), r12_shared.get(t)
        gains[t] = {"own": a, "shared": b, "gain": None if a is None or b is None else a - b}
    if any(g["gain"] is None for g in gains.values()):
        return {"gains": gains, "label": "undetermined", "failed": [], "text": None}
    failed = [f"doing task {t} gain {gains[t]['gain']:.3f} < {DOING_MIN_GAIN:.2f}" for t in doing
              if gains[t]["gain"] < DOING_MIN_GAIN - 1e-9]
    failed += [f"judging task {t} gain {gains[t]['gain']:.3f} >= {JUDGING_MAX_GAIN:.2f}" for t in judging
               if gains[t]["gain"] >= JUDGING_MAX_GAIN - 1e-9]
    ok = not failed
    return {"gains": gains, "label": "supported" if ok else "not_supported", "failed": failed,
            "text": H2E_TEXT if ok else None}


# ---------------------------------------------------------------- table

def baseline_surface() -> dict:
    out = {}
    for p in BASELINES:
        out.update(((ac.read_json(p) or {}).get("tasks")) or {})
    return out


def task_rows(cfg, coll: dict, base: dict | None = None) -> list[dict]:
    base = baseline_surface() if base is None else base
    grid = tuple(cfg.ks)
    out = []
    for t in sorted(cfg.table_tasks, key=lambda t: spec.TASKS[t].number):
        ts = spec.TASKS[t]
        e = base.get(t)
        row = {"task": t, "number": ts.number, "kind": ts.kind,
               "L_on": (ac.read_json(MAPS / f"map_{t}.json") or {}).get("L_on"),
               "surface": None if not e else {"bow": e["bow"], "chance": e["chance"], "gap": e["bow"] - e["chance"]},
               "r12": None, "dropped": False, "acc": {}, "healed_k0": None, "k_star": None, "chance": None, "acc_k12": None}
        by_k = coll.get(t)
        if by_k:
            acc = {k: [r["sel_accuracy"] for r in rows] for k, rows in by_k.items()}
            chance = next(iter(by_k.values()))[0]["chance"]
            d = ac._describe(acc, chance, None, grid)
            row.update(chance=chance, acc=d["acc"], healed_k0=d["healed_k0"], k_star=d["k_star"],
                       acc_k12=(d["acc"].get(cfg.kcut) or {}).get("mean"), lr={k: rows[0]["lr"] for k, rows in by_k.items()})
            if d["healed_k0"] is not None:
                row["dropped"] = B.is_dropped(d["healed_k0"], chance)
                if not row["dropped"] and row["acc_k12"] is not None:
                    row["r12"] = B.r12_value(row["acc_k12"], d["healed_k0"], chance)
        out.append(row)
    return out


def verdicts(rows: list[dict], n_perm: int, wtasks) -> dict:
    def use(drop=()):
        return [r for r in rows if r["r12"] is not None and r["L_on"] is not None and r["task"] not in drop]

    def a_block(rs):
        kinds, y = [r["kind"] for r in rs], [r["r12"] for r in rs]
        main = B.h2a_test([r["L_on"] for r in rs], y, kinds, n_perm)
        sr = [r for r in rs if r["surface"]]
        rival = B.h2a_test([-r["surface"]["gap"] for r in sr], [r["r12"] for r in sr], [r["kind"] for r in sr], 0)
        return {**main, "rival_rho": rival["rho"], "rival_n": rival["n"], "tasks": [r["task"] for r in rs],
                "label": B.label_a(main["rho"], main["p"], rival["rho"])}

    def b_block(rs):
        return {**B.h2b_test([B.l_on_group(r["L_on"]) for r in rs], [r["r12"] for r in rs], [r["kind"] for r in rs], n_perm),
                "tasks": [r["task"] for r in rs]}

    r12 = {r["task"]: r["r12"] for r in rows}
    leaky = (TASK18, TASK24)
    return {"H2-A": a_block(use()), "H2-A_drop18": a_block(use((TASK18,))), "H2-A_drop_leaky": a_block(use(leaky)),
            "H2-B": b_block(use()), "H2-B_drop18": b_block(use((TASK18,))), "H2-B_drop_leaky": b_block(use(leaky)),
            "dropped": [r["task"] for r in rows if r["dropped"]], "word_shortcut": reading_word(r12, wtasks)}


def own_rows(cfg, own: dict, rows: list[dict]) -> dict:
    """{task: {mean, seeds, lr, removed, r12}}: own-order seed-mean accuracy at kcut and R12 against the shared-order healed
    k=0 accuracy and chance (the same denominator as the shared R12)."""
    by_task = {r["task"]: r for r in rows}
    out = {}
    for t in cfg.own_tasks:
        rs = sorted((r for r in own.values() if r["task"] == t and r["k"] == cfg.kcut), key=lambda r: r["seed"])
        base = by_task.get(t) or {}
        mean = float(np.mean([r["sel_accuracy"] for r in rs])) if rs else None
        r12 = None
        if mean is not None and base.get("healed_k0") is not None and not base.get("dropped"):
            r12 = B.r12_value(mean, base["healed_k0"], base["chance"])
        out[t] = {"mean": mean, "seeds": [r["sel_accuracy"] for r in rs], "lr": rs[0]["lr"] if rs else None,
                  "removed": rs[0].get("removed") if rs else None, "r12": r12}
    return out


def build_summary(cfg, d: dict, n_perm: int | None = None, clean=None) -> dict:
    n_perm = cfg.n_perm if n_perm is None else n_perm
    ks = sorted({r["k"] for r in d["runs"].values()})
    coll = ac.collect(d["runs"], cfg.table_tasks, ks, "per_k")
    rows = task_rows(cfg, coll)
    extra = [t for t in cfg.tasks if t not in cfg.table_tasks]          # arith_long: reported, no L_on
    if extra:
        cfg2 = SimpleNamespace(**{**vars(cfg), "table_tasks": tuple(extra)})
        rows += task_rows(cfg2, ac.collect(d["runs"], extra, ks, "per_k"))
    wt = word_tasks(clean)
    s = {"smoke": cfg.smoke, "ks": list(cfg.ks), "kcut": cfg.kcut, "rows": rows, "verdict": verdicts(rows, n_perm, wt)}
    own = own_rows(cfg, d["own"], rows)
    s["own_order"] = own
    s["verdict"]["H2-E"] = h2e({r["task"]: r["r12"] for r in rows}, {t: v["r12"] for t, v in own.items()})
    tests = {}
    for r in d["test"].values():
        for name in r["names"]:
            tests.setdefault(r["task"], {}).setdefault(name, []).append(r)
    s["test"] = {t: {name: {"k": rs[0]["k"], "accuracy": ac._stat([r["accuracy"] for r in rs]), "chance": rs[0]["chance"]}
                     for name, rs in by.items()} for t, by in tests.items()}
    return s


def load_all(cfg) -> dict:
    return {"runs": ac.read_merged("runs", cfg.sfx, "runs"), "own": ac.read_merged("ownorder", cfg.sfx, "runs"),
            "test": (ac.read_json(ac.OUT / f"test_results{cfg.sfx}.json") or {}).get("results", {})}


def render_tables(s: dict) -> str:
    ks, v, f = s["ks"], s["verdict"], ac._f
    out = [f"# Atlas Phase 2 stage C1 tables (Qwen3-1.7B){' (smoke)' if s['smoke'] else ''}", "",
           f"Every task is trained on the whole answer plus end-of-turn and scored by exact match. R12 = (seed-mean accuracy at "
           f"k={s['kcut']} - chance) / (seed-mean healed k=0 accuracy - chance), clipped 0 to 1. Cells are seed mean (min-max) "
           "of selection exact match.", "", "## All tasks", "",
           "| # | task | kind | L_on | surface (BoW - chance) | chance | healed k0 | " + " | ".join(f"k{k}" for k in ks) +
           " | R12 | k* | note |", "|" + "---|" * (10 + len(ks))]
    for r in s["rows"]:
        note = "dropped (healed k0 under chance + 0.10)" if r["dropped"] else ("" if r["acc"] else "no runs")
        out.append(f"| {r['number']} | {r['task']} | {r['kind']} | {r['L_on']} | {f(r['surface'] and r['surface']['gap'])} | "
                   f"{f(r['chance'])} | {f(r['healed_k0'])} | " + " | ".join(ac._cell(r["acc"].get(k)) for k in ks) +
                   f" | {f(r['r12'])} | {r['k_star']} | {note} |")
    out += ["", "## H2-A (rank correlation of L_on with R12, within kind, weighted by task count)", ""]
    for name in ("H2-A", "H2-A_drop18", "H2-A_drop_leaky"):
        a = v[name]
        out.append(f"- {name}: rho {f(a['rho'])}, p {f(a['p'], 4)}, rival (negated surface) rho {f(a['rival_rho'])}, "
                   f"tasks {len(a['tasks'])} (appraise {a['n']['appraise']}, execute {a['n']['execute']}), "
                   f"by kind {a['by_kind']}: **{a['label']}**")
    out += ["", "## H2-B (appraisal minus execution R12 at equal L_on group)", ""]
    for name in ("H2-B", "H2-B_drop18", "H2-B_drop_leaky"):
        b = v[name]
        out.append(f"- {name}: stat {f(b['stat'])}, p {f(b['p'], 4)}, groups {json.dumps(b['groups'])}: **{b['label']}**")
    e = v["H2-E"]
    out += ["", f"## H2-E (own order against shared order, k={s['kcut']})", "",
            "| task | blocks cut (least damaging first) | lr | own seed mean | R12 own | R12 shared | gain |",
            "|---|---|---|---|---|---|---|"]
    for t, o in s["own_order"].items():
        g = e["gains"].get(t) or {"own": o["r12"], "shared": None, "gain": None}
        out.append(f"| {t} | {o['removed']} | {o['lr']} | {f(o['mean'])} | {f(g['own'])} | {f(g['shared'])} | {f(g['gain'])} |")
    out.append(f"\nReading (doing tasks gain >= {DOING_MIN_GAIN:.2f} and judging tasks gain < {JUDGING_MAX_GAIN:.2f}): "
               f"**{e['label']}**" + (f" - {e['text']}" if e["text"] else "") +
               ("; failed: " + "; ".join(e["failed"]) if e["failed"] else ""))
    w = v["word_shortcut"]
    out += ["", "## Readings", "", f"- Word shortcut: tasks {w['tasks']}, mean {f(w['mean'])}: **{w['label']}**",
            f"- Dropped from H2-A and H2-B: {v['dropped']}", "", "## Test split (once per named model)", ""]
    if s["test"]:
        out += ["| task | model | k | exact match (seeds) | chance |", "|---|---|---|---|---|"]
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


# ---------------------------------------------------------------- stage: probe

def project_c1(step_s: dict, gen_s: dict, tokens: dict, n_layers: int, ks, steps: int, own_k: int = 12, n_own: int = 12,
               n_lrsel: int = N_LRSEL_EVAL, n_sel: int = 500, n_test: int = 500, speedup: float = 1.0) -> dict:
    """Pod hours for the plan: 4 heals per (task, k) (2 lrs at seed 0, then seeds 1 and 2) plus `n_own` own-order heals at
    k=own_k. `step_s` is seconds per training step at k=0 and `gen_s` seconds per generated record, for two probed tasks;
    other tasks interpolate linearly in mean tokens (clamped to half the smaller and twice the larger), as
    `heal.project_hours`. A heal costs steps*step + (4*n_lrsel + n_sel) generations; the block-count factor
    (alive/n_layers) scales both; every finished (task, k) winner is also generated on `n_test` test records (3 seeds)."""
    (ta, _), (tb, _) = sorted(step_s.items(), key=lambda kv: tokens[kv[0]])

    def interp(vals, task):
        sa, sb = vals[ta], vals[tb]
        lo, hi = min(sa, sb), max(sa, sb)
        span = tokens[tb] - tokens[ta]
        s = sa if span == 0 else sa + (tokens[task] - tokens[ta]) * (sb - sa) / span
        return min(max(s, 0.5 * lo), 2.0 * hi)

    f = {k: (n_layers - k) / n_layers for k in set(ks) | {own_k}}

    def heal_s(t, k):
        return f[k] * (steps * interp(step_s, t) + (4 * n_lrsel + n_sel) * interp(gen_s, t))

    def test_s(t, k):
        return 3 * f[k] * n_test * interp(gen_s, t)

    sweep = sum(4 * heal_s(t, k) for t in tokens for k in ks) / 3600.0 / speedup
    test = sum(test_s(t, k) for t in tokens for k in ks) / 3600.0 / speedup
    own_tasks = [t for t in OWN_TASKS if t in tokens]
    own = (n_own * sum(heal_s(t, own_k) for t in own_tasks) / max(1, len(own_tasks))) / 3600.0 / speedup
    own_test = (n_own * sum(test_s(t, own_k) for t in own_tasks) / max(1, len(own_tasks)) / 3) / 3600.0 / speedup
    return {"heals": 4 * len(tokens) * len(list(ks)) + n_own, "sweep": sweep, "ownorder": own, "test": test + own_test,
            "total": sweep + own + test + own_test, "speedup_assumed": speedup}


def peak_gb(ctx) -> float | None:
    return torch.cuda.max_memory_allocated() / 1e9 if ctx.device.startswith("cuda") else None


def probe_task(ctx, task: str, steps: int, n_gen: int = 16) -> dict:
    """Time `steps` whole-answer training steps (k=0), then generate `n_gen` selection records; record peak GPU memory."""
    mnt = max_new_tokens(ctx, task)
    if ctx.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model, alive = ctx.cut(0)
    handle = heal.add_heal(model, len(alive), ac.RANK, 0)
    try:
        _, _, tim = heal.train_whole(model, ctx.tokenizer, task, ctx.records(task, "train"), [], handle.modules, steps,
                                     ac.LRS[0], ac.BATCH, 0, n_evals=0, log=lambda *_: None)
        if ctx.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        recs = ctx.records(task, "selection")[:n_gen]
        heal.exact_match_scores(model, ctx.tokenizer, task, recs, max_new_tokens=mnt, batch_size=ac.EVAL_BS)
        if ctx.device.startswith("cuda"):
            torch.cuda.synchronize()
        gen = (time.perf_counter() - t0) / max(1, len(recs))
    finally:
        handle.remove()
        del model
        ac.release(ctx)
    return {"step_seconds": tim["step_seconds"], "gen_seconds_per_record": gen, "peak_gpu_gb": peak_gb(ctx),
            "max_new_tokens": mnt}


def stage_probe(ctx, speedup: float) -> int:
    cfg = ctx.cfg
    tokens = {t: ac.mean_tokens(ctx, t) for t in cfg.tasks}
    big, small = max(tokens, key=tokens.get), min(tokens, key=tokens.get)
    if big == small:
        big = cfg.tasks[-1]
    probed = {t: probe_task(ctx, t, cfg.probe_steps) for t in (small, big)}
    for t, p in probed.items():
        print(f"probe {t}: {p['step_seconds']:.3f}s/step, {p['gen_seconds_per_record']:.3f}s/generated record, peak GPU "
              f"{p['peak_gpu_gb'] if p['peak_gpu_gb'] is None else round(p['peak_gpu_gb'], 2)} GB, "
              f"{tokens[t]:.0f} mean prompt tokens, max_new_tokens {p['max_new_tokens']}", flush=True)
    mnts = {t: max_new_tokens(ctx, t) for t in cfg.tasks}
    longest = {t: mnts[t] - MNT_MARGIN for t in cfg.tasks}
    print(f"max_new_tokens per task (longest answer + {MNT_MARGIN}): {json.dumps(mnts)}", flush=True)
    step_s = {t: p["step_seconds"] for t, p in probed.items()}
    gen_s = {t: p["gen_seconds_per_record"] for t, p in probed.items()}
    proj = {}
    fb = getattr(cfg, "fallback_grid", KS_FALLBACK)
    grids = (("main", cfg.ks),) + ((("fallback", fb),) if fb else ())
    for name, grid in grids:
        proj[name] = project_c1(step_s, gen_s, tokens, ctx.n_layers, grid, cfg.steps, cfg.kcut,
                                n_own=getattr(cfg, "n_own", 12), speedup=speedup)
        print(f"projected hours, k grid {tuple(grid)}: {json.dumps(proj[name])}", flush=True)
    ac.write(ac.OUT / f"probe{cfg.sfx}.json", {"device": ctx.device, "probed": probed, "mean_tokens": tokens,
                                              "longest_answer_tokens": longest, "max_new_tokens": mnts,
                                              "projection": proj, "ts": ac.now(),
                                              "note": "generation timed on an untrained LoRA (never emits end-of-turn), "
                                                      "so the generation share is an upper bound"})
    return 0


# ---------------------------------------------------------------- H2-E2 follow-up: own order from the healed model

E2_PREFIX = "ownorder2"          # shard files ownorder2{sfx}.shard*, checkpoints ownorder2_*, ledger keys ownorder2|
LESION_N = 500                   # items, as Phase 1's healed rerun (cfg.n_lesion)


def healed_lesion_path(cfg, task: str) -> Path:
    return ac.OUT / f"healed_lesion_{task}{cfg.sfx}.json"


def e2_ckpt_path(cfg, task, k, seed, lr) -> Path:
    return ac.CKPT / f"{E2_PREFIX}_{task}_k{k}_s{seed}_lr{lr:g}{cfg.sfx}.pt"


def load_own_order2(cfg, task: str, n_layers: int = 28) -> list[int]:
    return own_order(json.loads(healed_lesion_path(cfg, task).read_text()), n_layers)


def stage_healedlesion(ctx, task: str) -> int:
    """Single-block lesion sweep (blocks 1..n-2) on the task's seed-0, k=0 healed model at the seed-0 winner lr, rebuilt and
    merged as `atlas_cuts.stage_secondary` does: same `measure.build_store` means over the discovery items, then
    `measure.lesion_sweep` on the first 500 (option first-token accuracy, mean ablation of attn and mlp). Resumable by block;
    written to healed_lesion_<task>.json with `rows` [{block, accuracy, drop, seconds}] for `own_order`."""
    cfg = ctx.cfg
    final = healed_lesion_path(cfg, task)
    blocks = list(range(1, ctx.n_layers - 1))
    old = ac.read_json(final)
    if old and {r["block"] for r in old["rows"]} >= set(blocks):
        print(f"healedlesion {task}: already done", flush=True)
        return 0
    runs = ac.read_merged("runs", cfg.sfx, "runs")
    rows = [r for r in ac.collect(runs, [task], [0], "per_k").get(task, {}).get(0, []) if r["seed"] == 0]
    if not rows:
        print(f"healedlesion {task}: no healed k=0 seed-0 run; skipped", flush=True)
        return 1
    row = rows[0]
    recs = ctx.records(task, "discovery")
    store = ac.ACTS / f"{task}_healed_lesion{cfg.sfx}"
    model, handle = ac.rebuild_healed(ctx, row)
    adapters.merge_lora(handle)
    try:
        measure.build_store(model, ctx.tokenizer, task, recs, store, batch_size=ac.EVAL_BS)
        means = torch.load(store / "means.pt", map_location="cpu", weights_only=True)
        les = measure.lesion_sweep(model, ctx.tokenizer, task, recs[:LESION_N], means, batch_size=ac.EVAL_BS,
                                   path=final, blocks=blocks)
    finally:
        shutil.rmtree(store, ignore_errors=True)
    doc = {k: v for k, v in les.items() if k != "drops"}
    doc.update(lr=row["lr"], ckpt=row["ckpt"], k=0, seed=0, scoring="option first-token accuracy", ts=ac.now())
    ac.write(final, doc)
    del model
    ac.release(ctx)
    worst = sorted(doc["rows"], key=lambda r: -r["drop"])[:3]
    print(f"healedlesion {task}: base {doc['base_accuracy']:.3f} (n {doc['n']}, lr {row['lr']:g}); most damaging "
          f"{[(r['block'], round(r['drop'], 3)) for r in worst]}", flush=True)
    return 0


def stage_ownorder2(ctx, shard) -> int:
    """As `stage_ownorder`, but the cut order is `own_order` of the healed-model lesion sweep; own shard files, checkpoints
    and (in `test2`) ledger keys. Needs the sweep's seed-0 winner at kcut and healed_lesion_<task>.json per task."""
    cfg, (i, n) = ctx.cfg, shard
    path = ac.shard_path(E2_PREFIX, cfg.sfx, i, n)
    mine = ac.read_json(path) or {"shard": [i, n], "runs": {}}
    sweep = ac.read_merged("runs", cfg.sfx, "runs")
    missing, k = [], cfg.kcut
    for ti, task in enumerate(cfg.own_tasks):
        if ti % n != i:
            continue
        lr = ac.winner_lr(sweep, task, k, "per_k")
        if lr is None or not healed_lesion_path(cfg, task).exists():
            missing.append(f"{task}|k{k}")
            print(f"ownorder2: no sweep winner or no healed lesion file for {task} k{k}; skipped", flush=True)
            continue
        order = load_own_order2(cfg, task, ctx.n_layers)
        removed = order[:k]
        print(f"ownorder2 {task}: cut blocks {sorted(removed)} (least damaging first: {removed}), lr {lr:g}", flush=True)
        octx = OwnCtx(ctx, order)
        for seed in SEEDS:
            have = {**ac.read_merged(E2_PREFIX, cfg.sfx, "runs"), **mine["runs"]}
            key = ac.run_key(task, k, seed, lr)
            if key in have:
                print(f"skip {E2_PREFIX}|{key}: already done", flush=True)
                continue
            mine["runs"][key] = run_whole_heal(octx, task, k, seed, lr, own=True, removed=removed, ckpt_fn=e2_ckpt_path)
            ac.write(path, mine)
    if missing:
        print(f"ownorder2 incomplete: {missing}", flush=True)
        return 1
    return 0


def stage_test2(ctx) -> int:
    """Test-split exact match for every ownorder2 model. Results in test_results_e2.json (test_results.json is the main
    run's); ledger lines go to the same append-only test_ledger.jsonl under keys `ownorder2|<run key>` (not in smoke)."""
    cfg = ctx.cfg
    own = ac.read_merged(E2_PREFIX, cfg.sfx, "runs")
    path = ac.OUT / f"test_results_e2{cfg.sfx}.json"
    doc = ac.read_json(path) or {"results": {}}
    ledger = ac.OUT / "test_ledger.jsonl"
    for task in cfg.own_tasks:
        octx = None
        for row in sorted((r for r in own.values() if r["task"] == task), key=lambda r: r["seed"]):
            key = f"{E2_PREFIX}|{row['key']}"
            if key in doc["results"]:
                continue
            octx = octx or OwnCtx(ctx, load_own_order2(cfg, task, ctx.n_layers))
            em, n = test_exact(octx, row, task)
            name = f"{E2_PREFIX}_k{row['k']}"
            doc["results"][key] = {"task": task, "k": row["k"], "seed": row["seed"], "lr": row["lr"], "names": [name], "n": n,
                                   "accuracy": em, "chance": spec.chance([r["answer"] for r in ctx.records(task, "test")]),
                                   "ts": ac.now()}
            ac.write(path, doc)
            if cfg.ledger:
                with ledger.open("a") as fh:
                    fh.write(json.dumps({"key": key, "names": [name], "task": task, "k": row["k"], "seed": row["seed"],
                                         "n": n, "accuracy": em, "ts": ac.now()}, sort_keys=True) + "\n")
            print(f"test2 {key} ({name}): {em:.3f}", flush=True)
    return 0


def build_summary_e2(cfg, runs: dict, own: dict, own2: dict, test2: dict) -> dict:
    """Per own task: R12 shared, R12 own (H2-E, unmodified-model order), R12 healed own (H2-E2), the gains, the blocks each
    order cut, and the `h2e` reading for both (R12 denominators as the main table's)."""
    cfg1 = SimpleNamespace(**{**vars(cfg), "table_tasks": tuple(cfg.own_tasks)})
    ks = sorted({r["k"] for r in runs.values()})
    rows = task_rows(cfg1, ac.collect(runs, cfg.own_tasks, ks, "per_k"))
    shared = {r["task"]: r["r12"] for r in rows}
    e, e2 = own_rows(cfg, own, rows), own_rows(cfg, own2, rows)
    tasks = {}
    for t in cfg.own_tasks:
        g = lambda a, b: None if a is None or b is None else a - b          # noqa: E731
        tasks[t] = {"r12_shared": shared.get(t), "r12_own": e[t]["r12"], "r12_healed_own": e2[t]["r12"],
                    "gain_e": g(e[t]["r12"], shared.get(t)), "gain_e2": g(e2[t]["r12"], shared.get(t)),
                    "mean_own": e[t]["mean"], "mean_healed_own": e2[t]["mean"], "lr": e2[t]["lr"],
                    "removed_own": e[t]["removed"], "removed_healed_own": e2[t]["removed"],
                    "blocks_in_common": (sorted(set(e[t]["removed"]) & set(e2[t]["removed"]))
                                         if e[t]["removed"] and e2[t]["removed"] else None)}
    tests = {}
    for r in test2.values():
        tests.setdefault(r["task"], []).append(r)
    return {"smoke": cfg.smoke, "kcut": cfg.kcut, "tasks": tasks,
            "H2-E": h2e(shared, {t: e[t]["r12"] for t in cfg.own_tasks}),
            "H2-E2": h2e(shared, {t: e2[t]["r12"] for t in cfg.own_tasks}),
            "test_e2": {t: {"k": rs[0]["k"], "accuracy": ac._stat([r["accuracy"] for r in rs]), "chance": rs[0]["chance"]}
                        for t, rs in tests.items()}}


def render_tables_e2(s: dict) -> str:
    f = ac._f
    out = [f"# Atlas Phase 2 stage C1 follow-up H2-E2 (Qwen3-1.7B){' (smoke)' if s['smoke'] else ''}", "",
           f"R12 at k={s['kcut']}. Shared = the 1.7B shared order. Own (H2-E) = the task's order from lesions of the unmodified "
           "model. Healed own (H2-E2) = the order from lesions of the task's seed-0 k=0 healed model. Gain = R12 own - R12 "
           "shared.", "", "| task | R12 shared | R12 own (H2-E) | R12 healed own (H2-E2) | gain E | gain E2 | lr |",
           "|---|---|---|---|---|---|---|"]
    for t, r in s["tasks"].items():
        out.append(f"| {t} | {f(r['r12_shared'])} | {f(r['r12_own'])} | {f(r['r12_healed_own'])} | {f(r['gain_e'])} | "
                   f"{f(r['gain_e2'])} | {r['lr']} |")
    out += ["", "## Blocks cut (least damaging first)", "", "| task | H2-E order | H2-E2 order | in common |", "|---|---|---|---|"]
    for t, r in s["tasks"].items():
        out.append(f"| {t} | {r['removed_own']} | {r['removed_healed_own']} | {r['blocks_in_common']} |")
    for name, e in (("H2-E (as run)", s["H2-E"]), ("H2-E2 (follow-up)", s["H2-E2"])):
        out.append(f"\nReading {name} (doing tasks gain >= {DOING_MIN_GAIN:.2f} and judging tasks gain < "
                   f"{JUDGING_MAX_GAIN:.2f}): **{e['label']}**" + (f" - {e['text']}" if e["text"] else "") +
                   ("; failed: " + "; ".join(e["failed"]) if e["failed"] else ""))
    out += ["", "## Test split, healed-order models (exact match, seeds)", ""]
    if s["test_e2"]:
        out += ["| task | k | exact match (seeds) | chance |", "|---|---|---|---|"]
        out += [f"| {t} | {x['k']} | {ac._cell(x['accuracy'])} | {f(x['chance'])} |" for t, x in s["test_e2"].items()]
    else:
        out.append("Not run.")
    out.append("")
    return "\n".join(out)


def stage_table2(cfg) -> int:
    s = build_summary_e2(cfg, ac.read_merged("runs", cfg.sfx, "runs"), ac.read_merged("ownorder", cfg.sfx, "runs"),
                         ac.read_merged(E2_PREFIX, cfg.sfx, "runs"),
                         (ac.read_json(ac.OUT / f"test_results_e2{cfg.sfx}.json") or {}).get("results", {}))
    ac.write(ac.OUT / f"summary_e2{cfg.sfx}.json", s)
    md = render_tables_e2(s)
    (ac.OUT / f"tables_e2{cfg.sfx}.md").write_text(md + "\n")
    print(md, flush=True)
    return 0


# ---------------------------------------------------------------- step D1: depth measured on the healed model

def map_doc(task: str, sfx: str = "") -> dict:
    """The task's map file in the maps folder: `map_<task><sfx>.json` if present (a smoke map), else `map_<task>.json`."""
    return ac.read_json(MAPS / f"map_{task}{sfx}.json") or ac.read_json(MAPS / f"map_{task}.json") or {}


def healed_probe_path(cfg, task: str) -> Path:
    return ac.OUT / f"healed_probe_{task}{cfg.sfx}.json"


def stage_healedprobe(ctx, task: str) -> int:
    """`L_on_healed` of the task's seed-0, k=0 healed model at the seed-0 winner lr, as `atlas2_cuts.stage_healed` defines it:
    merged LoRA, `measure.build_store` over the discovery records, last-prompt-token probe, accuracy from the store's
    first-token predictions. Written to healed_probe_<task>.json; resumes by skipping when that file exists."""
    import atlas_map
    from atlas2_healed import l_on_healed
    cfg = ctx.cfg
    final = healed_probe_path(cfg, task)
    if final.exists():
        print(f"healedprobe {task}: {final} exists; nothing to do", flush=True)
        return 0
    runs = ac.read_merged("runs", cfg.sfx, "runs")
    rows = [r for r in ac.collect(runs, [task], [0], "per_k").get(task, {}).get(0, []) if r["seed"] == 0]
    if not rows:
        print(f"healedprobe {task}: no healed k=0 seed-0 run; skipped", flush=True)
        return 1
    row = rows[0]
    recs = ctx.records(task, "discovery")
    if cfg.probe_records:
        recs = recs[:cfg.probe_records]
    store = ac.ACTS / f"{task}_healed_probe{cfg.sfx}"
    model, handle = ac.rebuild_healed(ctx, row)
    adapters.merge_lora(handle)
    try:
        meta = measure.build_store(model, ctx.tokenizer, task, recs, store, batch_size=ac.EVAL_BS)
        answers = meta["answers"]
        acc = float(np.mean(np.asarray(meta["pred"]) == np.asarray(answers)))
        X = np.load(store / "resid.npy", mmap_mode="r")
        p = atlas_map._probe(X, answers, meta["template_ids"], np.arange(len(answers)), cfg.boots, seed=0)
        del X
    finally:
        shutil.rmtree(store, ignore_errors=True)
    chance = spec.chance(answers)
    curve = p["curve"] if p else None
    lo, never = l_on_healed(curve, chance, acc) if curve else (None, False)
    res = {"task": task, "smoke": cfg.smoke, "lr": row["lr"], "ckpt": row["ckpt"], "k": 0, "seed": 0, "n": len(answers),
           "chance": chance, "healed_accuracy": acc, "curve": curve, "lo": p["lo"] if p else None,
           "hi": p["hi"] if p else None, "L_on_healed": lo, "never": never,
           "L_on_unmodified": map_doc(task, cfg.sfx).get("L_on"),
           "scoring": "option first-token accuracy", "device": str(ctx.device), "ts": ac.now()}
    ac.write(final, res)
    del model
    ac.release(ctx)
    print(f"healedprobe {task}: healed acc {acc:.3f} chance {chance:.3f} L_on_healed {lo} never {never} "
          f"L_on_unmodified {res['L_on_unmodified']} -> {final}", flush=True)
    return 0


# ---------------------------------------------------------------- step D2: healing, cut-shape and budget arms

C1_OUT = Path("var/results/atlas3/c1")


def d2_prefix(arm: str) -> str:
    return f"d2_{arm}"


def d2_ckpt_path(arm: str, cfg, task, k, seed, lr) -> Path:
    return ac.CKPT / f"d2_{arm}_{task}_k{k}_s{seed}_lr{lr:g}{cfg.sfx}.pt"


def block_params(config, modules, rank: int) -> int:
    """LoRA parameters per block: rank * (in + out) summed over the module set, from a Qwen3 config."""
    h, inter = config.hidden_size, config.intermediate_size
    hd = getattr(config, "head_dim", None) or h // config.num_attention_heads
    q, kv = config.num_attention_heads * hd, config.num_key_value_heads * hd
    dims = {"q_proj": (h, q), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (q, h),
            "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h)}
    return sum(rank * sum(dims[m]) for m in modules)


def check_alllin_params(config, enforce: bool = True) -> tuple[int, int]:
    """Print the all-linear rank-2 count per block against q/v rank 10 (77,824 against 71,680 on Qwen3-1.7B); assert it is
    within `D2_PARAM_TOL` unless `enforce` is off (the smoke model has other dims)."""
    a = block_params(config, D2_ARMS["alllin"].modules, D2_ARMS["alllin"].rank)
    b = block_params(config, D2_ARMS["long"].modules, D2_ARMS["long"].rank)
    print(f"d2 alllin LoRA parameters per block: {a} against {b} for q/v rank {ac.RANK} ({(a - b) / b:+.1%})", flush=True)
    if enforce:
        assert abs(a - b) <= D2_PARAM_TOL * b, f"alllin {a} not within {D2_PARAM_TOL:.0%} of q/v rank-10 {b}"
    return a, b


def c1_winner_lrs(cfg) -> dict:
    """{task: {k: lr}} the C1 sweep's winner lr per (task, k), from the C1 summary.json rows."""
    doc = ac.read_json(C1_OUT / f"summary{cfg.sfx}.json") or {}
    return {r["task"]: {int(k): v for k, v in (r.get("lr") or {}).items()} for r in doc.get("rows", [])}


def d2_contig_order(cfg, n_layers: int) -> list[int]:
    """A cut order that removes `cfg.d2_contig` first (smoke: its first k)."""
    rest = [b for b in range(1, n_layers - 1) if b not in cfg.d2_contig]
    return list(cfg.d2_contig) + rest


def d2_units(cfg, arm: str, seed: int) -> list[tuple[str, int]]:
    """(task, k) heals for this arm and seed: k=cfg.kcut for every seed, plus k=0 at seed 0 for the arms that check it."""
    ks = (cfg.kcut,) + ((0,) if arm in D2_K0_ARMS and seed == 0 else ())
    return [(t, k) for t in (*cfg.d2_deep, *cfg.d2_shallow) for k in ks]


def d2_lrs(arm: str, merged: dict, task: str, k: int, seed: int, c1_lrs: dict) -> list[float]:
    """long: the task's C1 winner lr at k. Two-lr arms: the C1 rule (both lrs at seed 0, then the seed-0 winner)."""
    if arm == "long":
        lr = (c1_lrs.get(task) or {}).get(k)
        return [] if lr is None else [lr]
    return ac.unit_lrs(merged, task, k, seed, "per_k")


def stage_d2sweep(ctx, arm: str, seed: int, shard) -> int:
    """One arm, one seed. Own shard files (`d2_<arm>`), checkpoints `d2_<arm>_*`; resumable by run key."""
    cfg, (i, n), a = ctx.cfg, shard, D2_ARMS[arm]
    path = ac.shard_path(d2_prefix(arm), cfg.sfx, i, n)
    mine = ac.read_json(path) or {"shard": [i, n], "runs": {}}
    steps = cfg.steps * a.mult
    c1_lrs = c1_winner_lrs(cfg)
    if arm == "long" and cfg.smoke and not c1_lrs:
        c1_lrs = {t: {k: heal.TIE_LR for k in (0, cfg.kcut)} for t in (*cfg.d2_deep, *cfg.d2_shallow)}
    if arm == "alllin":
        check_alllin_params(ctx.pristine.config, enforce=not cfg.smoke)
    run_ctx = OwnCtx(ctx, d2_contig_order(cfg, ctx.n_layers)) if arm == "contig" else ctx
    print(f"d2sweep {arm} seed {seed} shard {i}/{n}: {steps} steps, {list(a.modules)} rank {a.rank}", flush=True)
    missing = []
    for ui, (task, k) in enumerate(d2_units(cfg, arm, seed)):
        if ui % n != i:
            continue
        merged = {**ac.read_merged(d2_prefix(arm), cfg.sfx, "runs"), **mine["runs"]}
        lrs = d2_lrs(arm, merged, task, k, seed, c1_lrs)
        if not lrs:
            missing.append(f"{task}|k{k}")
            print(f"d2 {arm}: no seed-0 winner / C1 lr for {task} k{k}; skipped", flush=True)
            continue
        for lr in lrs:
            key = ac.run_key(task, k, seed, lr)
            if key in merged:
                print(f"skip d2_{arm}|{key}: already done", flush=True)
                continue
            removed = run_ctx.order[:k] if arm == "contig" else None
            row = run_whole_heal(run_ctx, task, k, seed, lr, own=arm == "contig", removed=removed, arm=a, steps=steps,
                                 ckpt_fn=lambda c, t, kk, s, l: d2_ckpt_path(arm, c, t, kk, s, l))
            row["arm"] = arm
            mine["runs"][key] = row
            ac.write(path, mine)
    if missing:
        print(f"d2sweep {arm} incomplete: {missing}", flush=True)
        return 1
    return 0


# ---- d2 table: pure arithmetic

def d2_label(diff, deep_mean, shallow_mean) -> str:
    """artifact if lift_deep - lift_shallow >= 0.30; holds if < 0.15 and deep mean R12 < shallow mean R12; else partial."""
    if diff is None or deep_mean is None or shallow_mean is None:
        return "undetermined"
    if diff >= D2_ARTIFACT - 1e-9:
        return "artifact"
    if diff < D2_HOLDS - 1e-9 and deep_mean < shallow_mean:
        return "holds"
    return "partial"


def d2_arm_acc(runs: dict, arm: str, task: str, k: int) -> tuple[float | None, list[float]]:
    """(lr, seed-ordered selection exact matches) of the arm's winner lr at (task, k). The long arm has one lr per cell."""
    if arm == "long":
        rows = sorted((r for r in runs.values() if r["task"] == task and r["k"] == k), key=lambda r: (r["seed"], r["lr"]))
        if not rows:
            return None, []
        return rows[0]["lr"], [r["sel_accuracy"] for r in rows if r["lr"] == rows[0]["lr"]]
    got = ac.collect(runs, [task], [k], "per_k").get(task, {}).get(k)
    return (got[0]["lr"], [r["sel_accuracy"] for r in got]) if got else (None, [])


def _mean(xs):
    xs = list(xs)
    return None if not xs or any(x is None for x in xs) else float(np.mean(xs))


def d2_task_row(runs: dict, arm: str, task: str, c1: dict, kcut: int) -> dict:
    """R12 of the arm against C1's healed k=0 and chance, C1's R12 and the delta. `c1` is a C1 summary row. When the arm's
    own k=0 differs from C1's by more than 0.05, `r12_own_k0` (the arm's own denominator) is also given."""
    chance, c1_k0, c1_r12 = c1.get("chance"), c1.get("healed_k0"), c1.get("r12")
    lr, accs = d2_arm_acc(runs, arm, task, kcut)
    acc12 = float(np.mean(accs)) if accs else None
    r12 = None
    if acc12 is not None and chance is not None and c1_k0 is not None:
        r12 = B.r12_value(acc12, c1_k0, chance)
    lr0, accs0 = d2_arm_acc(runs, arm, task, 0) if arm in D2_K0_ARMS else (None, [])
    k0 = float(np.mean(accs0)) if accs0 else None
    differs = k0 is not None and c1_k0 is not None and abs(k0 - c1_k0) > OWN_K0_TOL + 1e-9
    own = B.r12_value(acc12, k0, chance) if differs and acc12 is not None and k0 - chance > 1e-9 else None
    return {"task": task, "chance": chance, "c1_k0": c1_k0, "arm_k0": k0, "arm_k0_lr": lr0, "k0_differs": bool(differs),
            "lr": lr, "n_seeds": len(accs), "acc12": acc12, "r12": r12, "c1_r12": c1_r12,
            "delta": None if r12 is None or c1_r12 is None else r12 - c1_r12, "r12_own_k0": own}


def d2_arm_summary(runs: dict, arm: str, c1_rows: dict, deep, shallow, kcut: int) -> dict:
    rows = {t: d2_task_row(runs, arm, t, c1_rows.get(t) or {}, kcut) for t in (*deep, *shallow)}
    lift_d = _mean(rows[t]["delta"] for t in deep)
    lift_s = _mean(rows[t]["delta"] for t in shallow)
    diff = None if lift_d is None or lift_s is None else lift_d - lift_s
    dm, sm = _mean(rows[t]["r12"] for t in deep), _mean(rows[t]["r12"] for t in shallow)
    return {"arm": arm, "tasks": rows, "lift_deep": lift_d, "lift_shallow": lift_s, "diff": diff, "deep_mean": dm,
            "shallow_mean": sm, "label": d2_label(diff, dm, sm)}


def build_summary_d2(cfg, runs_by_arm: dict, c1_rows: dict) -> dict:
    return {"smoke": cfg.smoke, "kcut": cfg.kcut, "deep": list(cfg.d2_deep), "shallow": list(cfg.d2_shallow),
            "thresholds": {"artifact": D2_ARTIFACT, "holds": D2_HOLDS},
            "arms": {arm: d2_arm_summary(runs_by_arm.get(arm, {}), arm, c1_rows, cfg.d2_deep, cfg.d2_shallow, cfg.kcut)
                     for arm in D2_ARMS}}


def render_tables_d2(s: dict) -> str:
    f = ac._f
    out = [f"# Atlas Phase 2 step D2 (Qwen3-1.7B){' (smoke)' if s['smoke'] else ''}", "",
           f"R12 at k={s['kcut']} = (arm seed-mean exact match - chance) / (C1 healed k=0 - chance), clipped 0 to 1. Delta = arm "
           f"R12 - C1 R12. Label: artifact if lift_deep - lift_shallow >= {s['thresholds']['artifact']:.2f}; holds if < "
           f"{s['thresholds']['holds']:.2f} and deep mean R12 < shallow mean R12; else partial."]
    for arm, a in s["arms"].items():
        out += ["", f"## Arm {arm}", "", "| task | group | lr | seeds | arm R12 | C1 R12 | delta | arm k0 | C1 k0 | R12 with own k0 |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for t, r in a["tasks"].items():
            note = f"{f(r['r12_own_k0'])} (k0 differs by > {OWN_K0_TOL:.2f})" if r["k0_differs"] else ""
            out.append(f"| {t} | {'deep' if t in s['deep'] else 'shallow'} | {r['lr']} | {r['n_seeds']} | {f(r['r12'])} | "
                       f"{f(r['c1_r12'])} | {f(r['delta'])} | {f(r['arm_k0'])} | {f(r['c1_k0'])} | {note} |")
        out.append(f"\nlift_deep {f(a['lift_deep'])}, lift_shallow {f(a['lift_shallow'])}, difference {f(a['diff'])}; deep mean "
                   f"R12 {f(a['deep_mean'])}, shallow mean R12 {f(a['shallow_mean'])}: **{a['label']}**")
    out.append("")
    return "\n".join(out)


def stage_d2table(cfg) -> int:
    doc = ac.read_json(C1_OUT / f"summary{cfg.sfx}.json") or {}
    c1_rows = {r["task"]: r for r in doc.get("rows", [])}
    runs = {arm: ac.read_merged(d2_prefix(arm), cfg.sfx, "runs") for arm in D2_ARMS}
    s = build_summary_d2(cfg, runs, c1_rows)
    ac.write(ac.OUT / f"summary{cfg.sfx}.json", s)
    md = render_tables_d2(s)
    (ac.OUT / f"tables{cfg.sfx}.md").write_text(md + "\n")
    print(md, flush=True)
    return 0


# ---------------------------------------------------------------- step E4: late-only cuts (docs/ATLAS-robustness.md)
# Pure functions below take plain rows; stage_e4sweep and stage_e4table do the file work.

def e4_prefix(arm: str) -> str:
    return f"e4_{arm}"


def e4_ckpt_path(arm: str, cfg, task, k, seed, lr) -> Path:
    return ac.CKPT / f"e4_{arm}_{task}_k{k}_s{seed}_lr{lr:g}{cfg.sfx}.pt"


def e4_order(cfg, arm: str, n_layers: int) -> list[int]:
    """A cut order that removes the arm's block set first, in order (so `order[:cfg.e4_k]` is exactly that set)."""
    blocks = list(cfg.e4_blocks[arm])
    return blocks + [b for b in range(1, n_layers - 1) if b not in blocks]


def e4_units(cfg, arm: str, seed: int) -> list[tuple[str, int]]:
    """(task, k) heals: every task at k=cfg.e4_k; no k=0 heal (R uses C1's healed k=0)."""
    return [(t, cfg.e4_k) for t in cfg.e4_tasks]


def stage_e4sweep(ctx, arm: str, seed: int, shard) -> int:
    """One arm, one seed; the D2 contig heal on `cfg.e4_blocks[arm]`. Own shard files (`e4_<arm>`), checkpoints `e4_<arm>_*`."""
    cfg, (i, n), a = ctx.cfg, shard, E4_ARMS[arm]
    path = ac.shard_path(e4_prefix(arm), cfg.sfx, i, n)
    mine = ac.read_json(path) or {"shard": [i, n], "runs": {}}
    run_ctx = OwnCtx(ctx, e4_order(cfg, arm, ctx.n_layers))
    print(f"e4sweep {arm} seed {seed} shard {i}/{n}: blocks {list(cfg.e4_blocks[arm])}, {cfg.steps} steps, "
          f"{list(a.modules)} rank {a.rank}", flush=True)
    missing = []
    for ui, (task, k) in enumerate(e4_units(cfg, arm, seed)):
        if ui % n != i:
            continue
        merged = {**ac.read_merged(e4_prefix(arm), cfg.sfx, "runs"), **mine["runs"]}
        lrs = d2_lrs("contig", merged, task, k, seed, {})
        if not lrs:
            missing.append(f"{task}|k{k}")
            print(f"e4 {arm}: no seed-0 winner for {task} k{k}; skipped", flush=True)
            continue
        for lr in lrs:
            key = ac.run_key(task, k, seed, lr)
            if key in merged:
                print(f"skip e4_{arm}|{key}: already done", flush=True)
                continue
            row = run_whole_heal(run_ctx, task, k, seed, lr, own=True, removed=run_ctx.order[:k], arm=a, steps=cfg.steps,
                                 ckpt_fn=lambda c, t, kk, s, l: e4_ckpt_path(arm, c, t, kk, s, l))
            row["arm"] = arm
            mine["runs"][key] = row
            ac.write(path, mine)
    if missing:
        print(f"e4sweep {arm} incomplete: {missing}", flush=True)
        return 1
    return 0


def e4_n_pre(blocks, depth) -> int | None:
    """Cut blocks before the task's read point (index < L_on_healed); None without a depth."""
    return None if depth is None else sum(1 for b in blocks if b < depth)


def e4_verdict(rho, n_low: int) -> str:
    """Bars: rho <= -0.40 holds; -0.40 < rho <= -0.20 weak; else (or undefined) not supported. Fewer than 3 qualifying tasks
    with R < 0.9 (`n_low`): uninformative, whatever rho says."""
    if n_low < E4_MIN_LOW:
        return "uninformative"
    if rho is None or (isinstance(rho, float) and math.isnan(rho)):
        return "not supported"
    if rho <= E4_HOLDS + 1e-9:
        return "holds"
    return "weak" if rho <= E4_WEAK + 1e-9 else "not supported"


def e4_perm_p(depth, r, kinds, n_perm: int = E4_PERM, seed: int = E4_PERM_SEED):
    """One-sided p of the weighted rho: shuffle R `n_perm` times, p = (#null <= obs + 1) / (n_perm + 1). None if rho undefined."""
    obs, _ = B.weighted_rho(list(depth), list(r), list(kinds))
    if obs is None:
        return None
    rng, r, hits = np.random.default_rng(seed), np.asarray(r, float), 0
    for _ in range(n_perm):
        s, _ = B.weighted_rho(list(depth), list(rng.permutation(r)), list(kinds))
        if s is not None and s <= obs + 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def e4_task_row(runs: dict, arm: str, task: str, c1: dict, depth, kind, blocks, k: int) -> dict:
    """One task under one cut. R = (acc - chance) / (C1 healed k=0 - chance), clipped 0 to 1, as D2; `r` is R of the seed-mean
    accuracy (the paper's R), `r_seeds` the per-seed R, `r_seed_mean` their mean."""
    chance, a0 = c1.get("chance"), c1.get("healed_k0")
    lr, accs = d2_arm_acc(runs, arm, task, k)
    usable = chance is not None and a0 is not None and a0 - chance > 1e-9
    rs = [B.r12_value(x, a0, chance) for x in accs] if usable else []
    acc = float(np.mean(accs)) if accs else None
    return {"task": task, "kind": kind, "L_on_healed": depth, "n_pre": e4_n_pre(blocks, depth), "chance": chance, "a0": a0,
            "lr": lr, "n_seeds": len(accs), "acc_seeds": accs, "acc": acc,
            "r_seeds": rs, "r": B.r12_value(acc, a0, chance) if usable and accs else None,
            "r_seed_mean": float(np.mean(rs)) if rs else None}


def e4_arm_summary(rows: dict, blocks, n_perm: int = E4_PERM) -> dict:
    """`rows` {task: e4_task_row}. Qualifying: n_pre == 0 and an R."""
    q = [r for r in rows.values() if r["n_pre"] == 0 and r["r"] is not None]
    dep, rr, kd = [r["L_on_healed"] for r in q], [r["r"] for r in q], [r["kind"] for r in q]
    rho, by_kind = B.weighted_rho(dep, rr, kd) if q else (None, {})
    pooled = heal.spearman(dep, rr) if q else float("nan")
    loo = [B.weighted_rho(dep[:j] + dep[j + 1:], rr[:j] + rr[j + 1:], kd[:j] + kd[j + 1:])[0] for j in range(len(q))]
    loo = [x for x in loo if x is not None]
    n_low = sum(1 for x in rr if x < E4_CEILING - 1e-9)
    return {"blocks": list(blocks), "tasks": rows, "n_qualifying": len(q), "qualifying": [r["task"] for r in q],
            "excluded": [t for t, r in rows.items() if r["n_pre"] != 0 or r["r"] is None],
            "rho": rho, "by_kind": by_kind, "pooled_rho": None if math.isnan(pooled) else pooled,
            "p": e4_perm_p(dep, rr, kd, n_perm) if q else None, "n_below_ceiling": n_low,
            "verdict": e4_verdict(rho, n_low), "loo_min": min(loo) if loo else None, "loo_max": max(loo) if loo else None}


def e4_difference(arms: dict) -> dict:
    """Report only: per task R(late17) - R(late15) against depth, and the Spearman of that difference with depth over the
    tasks that qualify (n_pre == 0 and an R) under both cuts."""
    a, b = arms["late17"]["tasks"], arms["late15"]["tasks"]
    rows = []
    for t in a:
        ra, rb = a[t], b.get(t) or {}
        both = ra["n_pre"] == 0 and rb.get("n_pre") == 0 and ra["r"] is not None and rb.get("r") is not None
        rows.append({"task": t, "L_on_healed": ra["L_on_healed"], "r_late17": ra["r"], "r_late15": rb.get("r"),
                     "diff": ra["r"] - rb["r"] if ra["r"] is not None and rb.get("r") is not None else None, "both": both})
    q = [x for x in rows if x["both"]]
    rho = heal.spearman([x["L_on_healed"] for x in q], [x["diff"] for x in q]) if q else float("nan")
    return {"rows": rows, "n": len(q), "spearman_diff_depth": None if math.isnan(rho) else rho}


def build_summary_e4(cfg, runs_by_arm: dict, c1_rows: dict, e1_tasks: dict, n_perm: int = E4_PERM) -> dict:
    arms = {}
    for arm, blocks in cfg.e4_blocks.items():
        rows = {}
        for t in cfg.e4_tasks:
            e = e1_tasks.get(t) or {}
            rows[t] = e4_task_row(runs_by_arm.get(arm, {}), arm, t, c1_rows.get(t) or {}, e.get("L_on_healed"),
                                  e.get("kind") or spec.TASKS[t].kind, blocks, cfg.e4_k)
        arms[arm] = e4_arm_summary(rows, blocks, n_perm)
    return {"smoke": cfg.smoke, "k": cfg.e4_k, "bars": {"holds": E4_HOLDS, "weak": E4_WEAK, "ceiling_R": E4_CEILING,
                                                         "min_below_ceiling": E4_MIN_LOW, "n_perm": n_perm},
            "arms": arms, "difference": e4_difference(arms) if set(arms) >= set(E4_BLOCKS) else None}


def render_tables_e4(s: dict) -> str:
    f = ac._f
    out = [f"# Atlas robustness E4: late-only cuts (Qwen3-1.7B){' (smoke)' if s['smoke'] else ''}", "",
           "R = (seed-mean exact match - chance) / (C1 healed k=0 - chance), clipped 0 to 1. Depth L_on_healed from D1 (e1.json). "
           "n_pre = cut blocks before the read point; qualifying = n_pre 0. Statistic: task-count-weighted within-kind Spearman "
           f"rho of depth against R (average ranks); one-sided permutation p ({s['bars']['n_perm']} shuffles of R, seed 0). "
           f"holds rho <= {s['bars']['holds']:.2f}; weak <= {s['bars']['weak']:.2f}; else not supported; uninformative if "
           f"fewer than {s['bars']['min_below_ceiling']} qualifying tasks have R < {s['bars']['ceiling_R']}."]
    for arm, a in s["arms"].items():
        out += ["", f"## Cut {arm}: blocks {a['blocks'][0]}-{a['blocks'][-1]}", "",
                f"qualifying {a['n_qualifying']} tasks; weighted rho {f(a['rho'])} (by kind {json.dumps(a['by_kind'])}), "
                f"pooled rho {f(a['pooled_rho'])}, p {f(a['p'], 4)}; tasks with R < {s['bars']['ceiling_R']}: "
                f"{a['n_below_ceiling']}; leave-one-task-out rho {f(a['loo_min'])} to {f(a['loo_max'])}: **{a['verdict']}**", "",
                "| task | kind | depth | n_pre | a0 | lr | acc per seed | acc | R per seed | R |", "|---|---|---|---|---|---|---|---|---|---|"]
        for t, r in a["tasks"].items():
            out.append(f"| {t} | {r['kind']} | {r['L_on_healed']} | {r['n_pre']} | {f(r['a0'])} | {r['lr']} | "
                       f"{', '.join(f(x) for x in r['acc_seeds'])} | {f(r['acc'])} | {', '.join(f(x) for x in r['r_seeds'])} | "
                       f"{f(r['r'])}{'' if r['task'] not in a['excluded'] else ' (excluded)'} |")
    d = s.get("difference")
    if d:
        out += ["", "## R(late17) - R(late15) against depth (report only; late15 sits 2 blocks closer to every read point and "
                "keeps 25-26, so this only partly separates depth from distance to the cut)", "",
                f"Spearman of the difference with depth over {d['n']} tasks qualifying under both: "
                f"{f(d['spearman_diff_depth'])}", "", "| task | depth | R late17 | R late15 | difference |", "|---|---|---|---|---|"]
        out += [f"| {x['task']} | {x['L_on_healed']} | {f(x['r_late17'])} | {f(x['r_late15'])} | {f(x['diff'])} |"
                for x in d["rows"]]
    out.append("")
    return "\n".join(out)


def stage_e4table(cfg) -> int:
    doc = ac.read_json(C1_OUT / f"summary{cfg.sfx}.json") or {}
    c1_rows = {r["task"]: r for r in doc.get("rows", [])}
    e1 = (ac.read_json(E1_PATH) or {}).get("1.7B") or {}
    e1_tasks = {r["task"]: r for r in e1.get("tasks", [])}
    runs = {arm: ac.read_merged(e4_prefix(arm), cfg.sfx, "runs") for arm in cfg.e4_blocks}
    s = build_summary_e4(cfg, runs, c1_rows, e1_tasks, 200 if cfg.smoke else E4_PERM)
    ac.write(ac.OUT / f"summary{cfg.sfx}.json", s)
    md = render_tables_e4(s)
    (ac.OUT / f"summary{cfg.sfx}.md").write_text(md + "\n")
    print(md, flush=True)
    return 0


# ---------------------------------------------------------------- step D3: second family (SmolLM2-1.7B-Instruct)
# The sweep, probe and healedprobe stages are the C1/D1 ones run under `--model smol17`; only the table is new. Bars:
# docs/ATLAS-phase2-stageC.md, "D3 bars". Pure functions below take plain rows; stage_d3table does the file work.

D3_DROP_MARGIN = 0.20        # healed k=0 under chance + 0.20: no room to measure survival, task dropped
D3_REPLICATES, D3_WEAK = -0.40, -0.20
D3_X_HOLDS, D3_X_MIN_N = -0.30, 8
D3_MODEL_LABEL = "SmolLM2-1.7B"


def d3_dropped(healed_k0: float, chance: float) -> bool:
    return healed_k0 < chance + D3_DROP_MARGIN - 1e-9


def _num(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def d3_label_a(rho, p, rival_rho) -> str:
    """replicates: rho <= -0.40, p < 0.05 and |rho| beats the surface rival. does not replicate: rho > -0.20. Otherwise weak
    (-0.40 < rho <= -0.20, or the rival is as strong, or rho <= -0.40 with p >= 0.05)."""
    if not _num(rho):
        return "undetermined"
    if rho > D3_WEAK + 1e-9:
        return "does not replicate"
    if rho <= D3_REPLICATES + 1e-9 and _num(p) and p < 0.05 and abs(rho) > abs(rival_rho if _num(rival_rho) else 0.0):
        return "replicates"
    return "weak"


def d3_label_x(n_zero: int, rho) -> str:
    """Within the tasks with n_pre = 0: holds if rho <= -0.30, else not shown; untestable under 8 such tasks."""
    if n_zero < D3_X_MIN_N:
        return "untestable"
    return "holds" if _num(rho) and rho <= D3_X_HOLDS + 1e-9 else "not shown"


def d3_task_rows(cfg, runs: dict, probes: dict, maps: dict, base: dict, cut) -> list[dict]:
    """One row per table task. `runs` the merged sweep rows, `probes` {task: healed_probe doc or None}, `maps` {task: map doc},
    `base` the BoW baseline {task: {bow, chance}}, `cut` the blocks removed at kcut. R = (mean exact match at kcut - chance) /
    (mean healed k=0 - chance), clipped 0 to 1, over the winner-lr seeds."""
    ks = sorted({r["k"] for r in runs.values()} | set(cfg.ks))
    coll = ac.collect(runs, cfg.table_tasks, ks, "per_k")
    out = []
    for t in sorted(cfg.table_tasks, key=lambda t: spec.TASKS[t].number):
        ts, e, by_k = spec.TASKS[t], base.get(t), coll.get(t) or {}
        row = {"task": t, "number": ts.number, "kind": ts.kind, "chance": None, "healed_k0": None, "acc_cut": None, "r": None,
               "n_seeds": {k: len(v) for k, v in by_k.items()}, "dropped": False, "no_runs": not by_k,
               "L_on": (maps.get(t) or {}).get("L_on"), "L_on_healed": None, "never": False, "healed_accuracy": None,
               "healed_missing": probes.get(t) is None, "n_pre": None,
               "surface": None if not e else {"bow": e["bow"], "chance": e["chance"], "gap": e["bow"] - e["chance"]}}
        if by_k:
            row["chance"] = next(iter(by_k.values()))[0]["chance"]
            mean = lambda k: float(np.mean([r["sel_accuracy"] for r in by_k[k]])) if k in by_k else None   # noqa: E731
            row["healed_k0"], row["acc_cut"] = mean(0), mean(cfg.kcut)
            if row["healed_k0"] is not None:
                row["dropped"] = d3_dropped(row["healed_k0"], row["chance"])
                if not row["dropped"] and row["acc_cut"] is not None:
                    row["r"] = B.r12_value(row["acc_cut"], row["healed_k0"], row["chance"])
        h = probes.get(t)
        if h:
            row.update(L_on_healed=h.get("L_on_healed"), never=bool(h.get("never")), healed_accuracy=h.get("healed_accuracy"))
            if row["L_on_healed"] is not None:
                row["n_pre"] = sum(1 for b in cut if b < row["L_on_healed"])
        out.append(row)
    return out


def build_summary_d3(cfg, rows: list[dict], cut, qwen_r: dict | None = None, n_perm: int | None = None) -> dict:
    """The D3 readings. Reuses atlas3_extra: `d1_size` (D1-A with healed depth, the surface rival, X1's n_pre, partial rhos
    and the judging-minus-doing test) on the kept tasks (not dropped, with R, L_on and a healed probe)."""
    import atlas3_extra as X
    n_perm = cfg.n_perm if n_perm is None else n_perm
    kept = [{**r, "r12": r["r"]} for r in rows if r["r"] is not None and r["L_on"] is not None and not r["dropped"]]
    no_l_on = [r["task"] for r in rows if r["r"] is not None and r["L_on"] is None and not r["dropped"]]
    d = X.d1_size(kept, cut, D3_MODEL_LABEL, n_perm)
    x = d["x1"]
    h = d["healed"]
    a = {"rho": h["rho"], "p": h["p"], "rival_rho": d["rival_rho"], "rival_n": d["rival_n"], "n": d["n_usable"],
         "tasks": d["tasks"], "by_kind": h["by_kind"], "label": d3_label_a(h["rho"], h["p"], d["rival_rho"]),
         "unmodified_same_tasks": d["unmod_same"], "unmodified_full": d["unmod_full"]}
    z = (x or {}).get("npre_zero") or {"n": 0, "rho_Lon": None, "tasks": []}
    xb = {"n_zero": z["n"], "rho": z["rho_Lon"], "tasks": [t[0] for t in z["tasks"]], "label": d3_label_x(z["n"], z["rho_Lon"]),
          "partial_depth_given_npre": (x or {}).get("partial_Lon_given_npre"),
          "partial_npre_given_depth": (x or {}).get("partial_npre_given_Lon"),
          "rho_npre": ((x or {}).get("h2a_npre") or {}).get("rho"), "cut": sorted(cut)}
    by_task = {r["task"]: r for r in rows}
    d2 = {t: {"smol_r": (by_task.get(t) or {}).get("r"), "qwen_r12": (qwen_r or {}).get(t)} for t in (*D2_DEEP, *D2_SHALLOW)}
    return {"smoke": cfg.smoke, "model": D3_MODEL_LABEL, "ks": list(cfg.ks), "kcut": cfg.kcut, "cut": sorted(cut), "rows": rows,
            "dropped": [r["task"] for r in rows if r["dropped"]], "no_L_on": no_l_on,
            "excluded": d["excluded"], "D3-A": a, "D3-X": xb, "judging_minus_doing": d["h2b"], "d2_tasks": d2,
            "bars": {"drop_margin": D3_DROP_MARGIN, "replicates": D3_REPLICATES, "weak": D3_WEAK, "x_holds": D3_X_HOLDS,
                     "x_min_n": D3_X_MIN_N}}


def render_tables_d3(s: dict) -> str:
    f = ac._f
    a, x, b = s["D3-A"], s["D3-X"], s["judging_minus_doing"]
    out = [f"# Atlas Phase 2 step D3 ({s['model']}){' (smoke)' if s['smoke'] else ''}", "",
           f"R = (seed-mean exact match at k={s['kcut']} - chance) / (seed-mean healed k=0 exact match - chance), clipped 0 to 1; "
           f"cut blocks {s['cut']}. Depth = L_on_healed (seed-0, k=0, winner-lr healed model, LoRA merged); L_on is the "
           "unmodified model's, beside it with no bar. n_pre = cut blocks before L_on_healed. Surface = BoW - chance. A task "
           f"whose healed k=0 is under chance + {s['bars']['drop_margin']:.2f} is dropped.", "", "## All tasks", "",
           "| # | task | kind | chance | healed k0 | R | L_on | L_on_healed | n_pre | surface | note |", "|" + "---|" * 11]
    for r in s["rows"]:
        note = ("dropped" if r["dropped"] else "no runs" if r["no_runs"] else "no healed probe" if r["healed_missing"]
                else "never reaches half" if r["never"] else "no healed depth" if r["L_on_healed"] is None else "")
        out.append(f"| {r['number']} | {r['task']} | {r['kind']} | {f(r['chance'])} | {f(r['healed_k0'])} | {f(r['r'])} | "
                   f"{r['L_on']} | {r['L_on_healed']} | {r['n_pre']} | {f(r['surface'] and r['surface']['gap'])} | {note} |")
    out += ["", f"Dropped (healed k0 < chance + {s['bars']['drop_margin']:.2f}): {s['dropped'] or 'none'}",
            f"Left out of the correlations: never reaches half {s['excluded']['never']}, no healed depth {s['excluded']['none']}, "
            f"no healed probe {s['excluded']['missing']}, no unmodified L_on {s['no_L_on']}", "",
            "## D3-A (replication)", "",
            f"- rho(L_on_healed, R) over {a['n']} kept tasks: {f(a['rho'])}, p {f(a['p'], 4)}; surface rival (negated BoW gap) "
            f"rho {f(a['rival_rho'])} (n {a['rival_n']}); by kind {json.dumps(a['by_kind'])}: **{a['label']}**",
            f"- unmodified L_on, same tasks: rho {f(a['unmodified_same_tasks']['rho'])}, p "
            f"{f(a['unmodified_same_tasks']['p'], 4)}; all rows with R: rho {f(a['unmodified_full']['rho'])}, p "
            f"{f(a['unmodified_full']['p'], 4)}", "",
            "## D3-X (beyond the cut count)", "",
            f"- tasks with n_pre = 0: {x['n_zero']} {x['tasks']}; rho(L_on_healed, R) inside: {f(x['rho'])}: **{x['label']}** "
            f"(holds if <= {s['bars']['x_holds']:.2f}; untestable under {s['bars']['x_min_n']} tasks)",
            f"- partial rho of depth holding n_pre fixed: {f(x['partial_depth_given_npre'])}; of n_pre holding depth fixed: "
            f"{f(x['partial_npre_given_depth'])}; rho(n_pre, R): {f(x['rho_npre'])} (no bar)", "",
            "## Judging minus doing at equal depth (no bar)", ""]
    out.append(f"- stat {f(b and b['stat'])}, p {f(b and b['p'], 4)}, groups {json.dumps(b and b['groups'])}"
               if b else "- not computed (under 2 usable tasks)")
    out += ["", "## The six D2 tasks: R here and Qwen3-1.7B C1's R12 (no bar)", "", "| task | R SmolLM2 | R12 Qwen3 (C1) |",
            "|---|---|---|"]
    out += [f"| {t} | {f(v['smol_r'])} | {f(v['qwen_r12'])} |" for t, v in s["d2_tasks"].items()]
    out.append("")
    return "\n".join(out)


def stage_d3table(cfg) -> int:
    runs = ac.read_merged("runs", cfg.sfx, "runs")
    probes = {t: ac.read_json(healed_probe_path(cfg, t)) for t in cfg.table_tasks}
    maps = {t: map_doc(t, cfg.sfx) for t in cfg.table_tasks}
    order = (ac.read_json(ORDER_PATH) or {}).get("shared_order")
    if not order:
        raise SystemExit(f"no shared order at {ORDER_PATH}")
    cut = order[:cfg.kcut]
    c1 = ac.read_json(C1_OUT / "summary.json") or {}
    qwen_r = {r["task"]: r.get("r12") for r in c1.get("rows", [])}
    rows = d3_task_rows(cfg, runs, probes, maps, baseline_surface(), cut)
    s = build_summary_d3(cfg, rows, cut, qwen_r)
    ac.write(ac.OUT / f"summary{cfg.sfx}.json", s)
    md = render_tables_d3(s)
    (ac.OUT / f"tables{cfg.sfx}.md").write_text(md + "\n")
    print(md, flush=True)
    return 0


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=STAGES + D2_STAGES + E4_STAGES + D3_STAGES)
    ap.add_argument("--model", default="qwen17", choices=sorted(PROFILES),
                    help="model profile: qwen17 (C1, D1, D2) or smol17 (step D3; stages probe, sweep, healedprobe, d3table)")
    ap.add_argument("--shard", default=None, help="I/N")
    ap.add_argument("--seed", type=int, default=0, choices=SEEDS)
    ap.add_argument("--arm", default=None, choices=sorted(D2_ARMS) + sorted(E4_ARMS),
                    help="d2sweep: alllin, contig or long; e4sweep: late17 or late15")
    ap.add_argument("--task", default=None,
                    help="healedlesion: one of the own-order tasks; healedprobe: one of the table tasks")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--parallel-speedup", type=float, default=1.0, help="probe: assumed gain from side-by-side shards")
    args = ap.parse_args(argv)
    profile = resolve_profile(args.model, args.smoke)
    allowed = PROFILE_STAGES[args.model]
    if allowed and args.stage not in allowed:
        raise SystemExit(f"--model {args.model} runs only the stages {list(allowed)}")
    if args.model == "qwen17" and args.stage in D3_STAGES:
        raise SystemExit(f"{args.stage} needs --model smol17")
    configure(None if args.model == "qwen17" else profile)
    cfg = make_cfg(args.smoke, profile)
    shard = ac.parse_shard(args.shard)
    if args.stage.startswith("d2"):
        ac.OUT = B.OUT = D2_OUT                  # D2 writes under d2/; C1's files are read through C1_OUT
    if args.stage.startswith("e4"):
        ac.OUT = B.OUT = E4_OUT                  # E4 writes under e4/; C1's files are read through C1_OUT
    ac.OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "e4table":
        return stage_e4table(cfg)
    if args.stage == "e4sweep" and args.arm not in E4_ARMS:
        raise SystemExit("e4sweep needs --arm in " + str(sorted(E4_ARMS)))
    if args.stage == "d2sweep" and args.arm in E4_ARMS:
        raise SystemExit("d2sweep needs --arm in " + str(sorted(D2_ARMS)))
    if args.stage == "d2table":
        return stage_d2table(cfg)
    if args.stage == "d2sweep" and not args.arm:
        raise SystemExit("d2sweep needs --arm in " + str(sorted(D2_ARMS)))
    if args.stage == "d3table":
        return stage_d3table(cfg)
    if args.stage == "table":
        return stage_table(cfg)
    if args.stage == "table2":
        return stage_table2(cfg)
    if args.stage == "healedlesion" and args.task not in cfg.own_tasks:
        raise SystemExit(f"healedlesion needs --task in {list(cfg.own_tasks)}")
    if args.stage == "healedprobe" and args.task not in cfg.table_tasks:
        raise SystemExit(f"healedprobe needs --task in {list(cfg.table_tasks)}")
    ctx = ac.Ctx(cfg, device=os.environ.get("ATLAS3_DEVICE") or None)   # unset: cuda if present, else cpu (as before)
    if args.model != "qwen17":
        check_blocks(ctx.n_layers, profile)
    if args.stage == "d2sweep":
        return stage_d2sweep(ctx, args.arm, args.seed, shard)
    if args.stage == "e4sweep":
        return stage_e4sweep(ctx, args.arm, args.seed, shard)
    if args.stage == "probe":
        return stage_probe(ctx, args.parallel_speedup)
    if args.stage == "sweep":
        return stage_sweep(ctx, args.seed, shard)
    if args.stage == "ownorder":
        return stage_ownorder(ctx, shard)
    if args.stage == "healedlesion":
        return stage_healedlesion(ctx, args.task)
    if args.stage == "healedprobe":
        return stage_healedprobe(ctx, args.task)
    if args.stage == "ownorder2":
        return stage_ownorder2(ctx, shard)
    if args.stage == "test2":
        return stage_test2(ctx)
    return stage_test(ctx)


if __name__ == "__main__":
    sys.exit(main())
