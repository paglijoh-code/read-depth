"""M7 efficiency-frontier sweep (CONTRACT.md M7, steps 2-5): excise k blocks, heal with merged LoRA, then prune KV groups.

  uv run python scripts/frontier_sweep.py --stage {tune,seeds,kvprune,recheck,table,sweep} [--variant V] [--smoke]
  uv run python scripts/frontier_sweep.py --stage test --variant V
  uv run python scripts/frontier_sweep.py --stage sweep [--until HH:MMZ]
Results go to var/results/m7/ (m7_{variant}.json, status.json, report_tables.md) and adapter checkpoints to
var/ckpt/m7/. Every run is resumable (a run key already in the results file is skipped). Each run starts from a
fresh copy of the pristine model (excise, fold and merge mutate it). Selection passes append to
var/results/selection_log.jsonl and test runs to var/results/test_ledger.jsonl (refuses repeats); --smoke
(16 train, 16 held-out, 32 selection, 4 steps, KS 2 and 3, small grids, *_smoke.json) never writes either ledger.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import adapter_sweep as A  # noqa: E402
import surgery_sweep as S  # noqa: E402
from surgery import adapters, stats, structure  # noqa: E402
from surgery.canaries import load_canaries, score_canaries  # noqa: E402
from surgery.eval import _free, load_model  # noqa: E402
from surgery.prompt import build_prompt  # noqa: E402
from surgery.task.generator import LABELS  # noqa: E402

MODEL, SHORT, VARIANTS = A.MODEL, A.SHORT, A.VARIANTS
OUT = Path("var/results/m7")
CKPT = Path("var/ckpt/m7")
M5 = A.M5
N_LAYERS = 28
SEEDS = [0, 1, 2]
KS = [9, 12, 14, 16, 18]
SMOKE_KS = [2, 3]
LRS, STEPS = [3e-4, 1e-3, 3e-3], [100, 300]
SMOKE_LRS = [1e-3]
QV, QKVO = ["q_proj", "v_proj"], ["q_proj", "k_proj", "v_proj", "o_proj"]
KV_STEP, KV_MAX_STEPS, KV_DROP = 8, 8, 0.05
SMOKE_KV_STEP, SMOKE_KV_MAX_STEPS = 2, 2
TARGET_A, TARGET_FLOOR, H7_MIN_K, H7_TEST, H7_FALSIFY = 0.95, 0.90, 12, 0.93, 0.90
RECHECK = [("reft", "10k"), ("reft", "100k"), ("lora", "1m")]
STAGES = ["tune", "seeds", "kvprune", "recheck", "test", "table", "sweep"]

now, write, read_json, append = A.now, A.write, A.read_json, A.append


class Stop(Exception):
    """--until reached."""


def sfx(smoke: bool) -> str:
    return A.smoke_suffix(smoke)


def cfg_of(lr, steps, rank, modules) -> dict:
    return {"lr": lr, "steps": steps, "rank": rank, "modules": list(modules)}


def tag(cfg: dict) -> str:
    return f"lr{cfg['lr']:g}_st{cfg['steps']}_r{cfg['rank']}_{'qkvo' if len(cfg['modules']) == 4 else 'qv'}"


def cfg_text(cfg: dict) -> str:
    return f"lr {cfg['lr']:g}, {cfg['steps']} steps, r{cfg['rank']} {'qkvo' if len(cfg['modules']) == 4 else 'qv'}"


def pick_key(acc, cfg):
    """Best held-out accuracy; ties go to fewer steps, then smaller rank, then smaller lr."""
    return (-acc, cfg["steps"], cfg["rank"], cfg["lr"])


# ---------------------------------------------------------------- context

class FCtx(A.Ctx):
    """adapter_sweep.Ctx with a pristine CPU model and a fresh device copy per run. `a.batch_size` of the inherited
    scoring helpers is the eval batch size; training uses `train_bs`."""

    def __init__(self, a, until=None):
        self.a = argparse.Namespace(**vars(a))
        self.train_bs, self.a.batch_size = a.batch_size, a.eval_batch_size
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.pristine, self.tok = load_model(MODEL, device="cpu")
        self.cfg = copy.deepcopy(self.pristine.config)
        self.model = self.ids = None
        self.until, self.smoke = until, a.smoke
        self._data, self._means, self._dirs, self._base = {}, {}, {}, {}
        self._res, self._seq, self._can, self._order = {}, {}, None, {}

    # model lifecycle
    def release(self):
        m, self.model = self.model, None
        if m is not None:
            del m
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def fresh(self):
        self.release()
        self.model = copy.deepcopy(self.pristine).to(self.device).eval()
        self.ids = adapters.label_ids(self.model, self.tok)
        return self.model

    def baseline_pred(self, variant, split="selection"):  # needs an unmodified model
        if (variant, split) not in self._base:
            self.fresh()
        return super().baseline_pred(variant, split)

    def base_canaries(self):
        if self._can is None:
            if self.smoke:
                self.fresh()
                self._can = score_canaries(self.model, self.tok, load_canaries()[::16], self.a.batch_size)
            else:
                self._can = read_json(A.BASE_CANARIES)["canaries"]
        return self._can

    # results files
    def res(self, variant):
        if variant not in self._res:
            p = OUT / f"m7_{variant}{sfx(self.smoke)}.json"
            self._res[variant] = read_json(p) if p.exists() else {
                "model": MODEL, "variant": variant, "tune": {}, "runs": {}, "recheck": {"grid": {}, "selection": {}}}
        return self._res[variant]

    def save(self, variant):
        write(OUT / f"m7_{variant}{sfx(self.smoke)}.json", self._res[variant])

    def status(self, stage, variant, run, done=False):
        tag_ = self.a.variant or "all"  # one file per process (variants run as parallel processes)
        write(OUT / f"status_{tag_}{sfx(self.smoke)}.json",
              {"stage": stage, "variant": variant, "run": run, "ts": now(), "done": done})
        if not done and self.until is not None and datetime.now(timezone.utc) >= self.until:
            raise Stop

    # excision order
    def removed(self, variant):
        if variant not in self._order:
            kmax = max(self.ks)
            order = read_json(M5 / f"excise_{SHORT}_{variant}.json")["removed_order"]
            if len(order) < kmax:
                sys.exit(f"M5 excision path for {variant} has {len(order)} blocks, need {kmax}: "
                         f"extend var/results/m5/excise_{SHORT}_{variant}.json first")
            ref = OUT / f"excise_m5_{variant}_k10.json"
            if ref.exists():
                assert order[:10] == read_json(ref)["removed_order"], f"{variant}: first 10 removals differ from {ref}"
            else:
                print(f"warning: {ref} missing, first-10 check skipped", flush=True)
            self._order[variant] = order
        return self._order[variant]

    @property
    def ks(self):
        return SMOKE_KS if self.smoke else KS

    def alive(self, variant, k):
        gone = set(self.removed(variant)[:k])
        return [i for i in range(N_LAYERS) if i not in gone]

    def seq_len(self, variant):
        if variant not in self._seq:
            recs = self.records(variant, "selection")
            self._seq[variant] = float(np.mean([len(self.tok(build_prompt(self.tok, r["prompt"]))["input_ids"])
                                                for r in recs]))
        return self._seq[variant]

    def n_boot(self):
        return 2000 if self.smoke else 10_000


# ---------------------------------------------------------------- paths

def ckpt_dir(smoke: bool) -> Path:
    return CKPT / ("smoke" if smoke else "")


def tune_ckpt(ctx, v, k, cfg) -> Path:
    return ckpt_dir(ctx.smoke) / f"{v}_k{k}_tune_{tag(cfg)}.pt"


def point_ckpt(ctx, v, k, ng, seed, cfg) -> Path:
    if ng == 0 and seed == 0:
        return tune_ckpt(ctx, v, k, cfg)
    return ckpt_dir(ctx.smoke) / f"{v}_k{k}_g{ng}_{tag(cfg)}_s{seed}.pt"


def run_key(k, ng, seed) -> str:
    return f"k{k}|g{ng}|s{seed}"


def save_ckpt(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)


# ---------------------------------------------------------------- training pieces

def lora_arm(ctx, n_layers, cfg, seed) -> A.Arm:
    """LoRA on all `n_layers` surviving blocks of ctx.model (in whatever physical numbering it has now)."""
    model, tok, ids = ctx.model, ctx.tok, ctx.ids
    torch.manual_seed(seed)
    h = adapters.add_lora(model, range(n_layers), cfg["modules"], cfg["rank"])
    fwd = lambda recs: adapters.full_logits(model, adapters.encode(model, tok, recs), ids)  # noqa: E731
    return A.Arm(h.modules, fwd, [h], adapters.count_trainable(model))


def fit(ctx, arm, v, seed, cfg):
    """Train for the cfg's steps (4 under smoke); returns (best_state, curve, timings)."""
    heldout, train = ctx.train_heldout(v, seed)
    steps = 4 if ctx.smoke else cfg["steps"]
    t0 = time.perf_counter()
    state, curve = adapters.train_adapter(ctx.model, ctx.tok, train, heldout, arm.params, arm.forward, steps,
                                          lr=cfg["lr"], batch_size=ctx.train_bs, seed=seed, n_evals=4)
    train_s = time.perf_counter() - t0
    step_s = (train_s - curve[-1]["eval_seconds"]) / steps
    _free(ctx.model)
    return state, curve, {"steps": steps, "train_seconds": train_s, "step_seconds": step_s}


def best_heldout(curve) -> float:
    return max(c["heldout_accuracy"] for c in curve)


def build_point(ctx, v, k, cfg, seed, groups, means=None):
    """Fresh model -> excise to k -> (if groups: load and merge this seed's k heal, fold) -> untrained LoRA arm.
    Returns (arm, means). `groups` are [(layer, group)] in the excised numbering."""
    model = ctx.fresh()
    alive = ctx.alive(v, k)
    structure.excise(model, alive)
    if groups:
        base = lora_arm(ctx, len(alive), cfg, seed)
        adapters.load_adapter_state(base.params, torch.load(point_ckpt(ctx, v, k, 0, seed, cfg))["state"])
        adapters.merge_lora(base.handles[0])
        if means is None:
            _, train = ctx.train_heldout(v, seed)
            means = structure.head_means(model, ctx.tok, train, ctx.a.batch_size)
        structure.fold_kv_groups(model, means, groups)
        _free(model)
    return lora_arm(ctx, len(alive), cfg, seed), means


# ---------------------------------------------------------------- tune

def tune_grid(smoke: bool):
    return [(lr, st) for lr in (SMOKE_LRS if smoke else LRS) for st in STEPS]


def run_tune(ctx, v, k, cfg, kind) -> dict:
    d = ctx.res(v)
    key = f"k{k}|{tag(cfg)}"
    if key in d["tune"]:
        print(f"skip {v} tune {key}: already done", flush=True)
        return d["tune"][key]
    ctx.status("tune", v, key)
    print(f"== {v} tune {key} {now()}", flush=True)
    ctx.removed(v)
    arm, _ = build_point(ctx, v, k, cfg, 0, [])
    try:
        state, curve, tm = fit(ctx, arm, v, 0, cfg)
    finally:
        arm.close()
        ctx.release()
    cp = tune_ckpt(ctx, v, k, cfg)
    save_ckpt(cp, {"state": state, "variant": v, "k": k, "cfg": cfg, "seed": 0})
    row = {"k": k, "kind": kind, "cfg": cfg, "heldout_best": best_heldout(curve), "curve": curve,
           "checkpoint": str(cp), "ts": now(), **tm}
    d["tune"][key] = row
    ctx.save(v)
    print(f"{v} tune {key}: heldout {row['heldout_best']:.3f} ({tm['step_seconds']:.2f}s/step, "
          f"train {tm['train_seconds']:.0f}s, sel 0s)", flush=True)
    return row


def tune_rows(d, k):
    return [r for r in d["tune"].values() if r["k"] == k]


def chosen_cfg(d, k):
    """Best held-out config of the 6 grid runs plus the wide run; None until all of them are done."""
    rows = tune_rows(d, k)
    if not rows or not any(r["kind"] == "wide" for r in rows):
        return None
    return min(rows, key=lambda r: pick_key(r["heldout_best"], r["cfg"]))["cfg"]


def stage_tune(ctx, variants):
    for v in variants:
        for k in ctx.ks:
            rows = [run_tune(ctx, v, k, cfg_of(lr, st, 10, QV), "grid") for lr, st in tune_grid(ctx.smoke)]
            best = min(rows, key=lambda r: pick_key(r["heldout_best"], r["cfg"]))["cfg"]
            run_tune(ctx, v, k, cfg_of(best["lr"], best["steps"], 40, QKVO), "wide")


# ---------------------------------------------------------------- scored points

def run_point(ctx, v, k, cfg, seed, groups=(), stage="seeds") -> dict:
    """Heal (or re-heal after a fold) one model for one seed, then score selection (+ canaries for seed 0)."""
    d = ctx.res(v)
    groups = [tuple(g) for g in groups]
    ng = len(groups)
    key = run_key(k, ng, seed)
    if key in d["runs"]:
        print(f"skip {v} {key}: already done", flush=True)
        return d["runs"][key]
    ctx.status(stage, v, key)
    print(f"== {v} {key} {cfg_text(cfg)} {now()}", flush=True)
    sel = ctx.records(v, "selection")
    base_pred = ctx.baseline_pred(v)
    base_can = ctx.base_canaries() if seed == 0 else None  # both need an unmodified model: before building ours
    alive = ctx.alive(v, k)
    arm, means = build_point(ctx, v, k, cfg, seed, groups)
    model = ctx.model
    try:
        cp = point_ckpt(ctx, v, k, ng, seed, cfg)
        if ng == 0 and seed == 0:  # the tune run is the seed-0 heal
            src = torch.load(cp)
            state, tm = src["state"], {"steps": None, "train_seconds": 0.0, "step_seconds": 0.0}
            curve, reused = None, True
        else:
            state, curve, tm = fit(ctx, arm, v, seed, cfg)
            reused = False
            save_ckpt(cp, {"state": state, "variant": v, "k": k, "cfg": cfg, "seed": seed, "groups": groups,
                           "means": means})
        adapters.load_adapter_state(arm.params, state)
        adapters.merge_lora(arm.handles[0])
        _free(model)
        t1 = time.perf_counter()
        pred = A.predict(arm, sel, ctx.a.batch_size)
        sel_s = time.perf_counter() - t1
        can = A.canaries_for(ctx, arm, v, "heal", base_can) if seed == 0 else None
        orig = [(alive[layer], g) for layer, g in groups]
        row = {"k": k, "ng": ng, "seed": seed, "cfg": cfg, "groups": [list(g) for g in groups],
               "groups_original": [list(g) for g in orig], "reused_tune": reused, "curve": curve,
               "train_seconds": tm["train_seconds"], "step_seconds": tm["step_seconds"], "steps": tm["steps"],
               "selection_seconds": sel_s, "checkpoint": str(cp), "ts": now(),
               "selection": A.summarize(sel, pred, base_pred, ctx.n_boot()), "canaries": can,
               "size": structure.model_size(model), "kv_bytes_per_token": structure.kv_cache_bytes_per_token(model),
               "seq_len": ctx.seq_len(v),
               "flops": structure.flops_per_token(ctx.cfg, ctx.seq_len(v), order=alive, pruned_groups=orig),
               "block_params": structure.param_count(ctx.cfg, order=alive, pruned_groups=orig)}
    finally:
        arm.close()
        ctx.release()
    d["runs"][key] = row
    ctx.save(v)
    if not ctx.smoke:
        append(A.SELECTION_LOG, {"ts": now(), "model": MODEL, "dtype": "fp32", "variant": v, "split": "selection",
                                 "seed": seed, "n": len(sel), "result": str(OUT / f"m7_{v}.json"),
                                 "intervention": f"m7-k{k}-g{ng}-{tag(cfg)}-s{seed}",
                                 "config": {"k": k, "kv_groups": ng, **cfg},
                                 "accuracy": row["selection"]["accuracy"]})
    acc = row["selection"]["accuracy"]
    print(f"{v} {key}: selection {acc:.3f} gain {row['selection']['gain']:+.3f} "
          f"(train {tm['train_seconds']:.0f}s, {tm['step_seconds']:.2f}s/step, sel {sel_s:.0f}s)", flush=True)
    return row


def points(d) -> list[dict]:
    """Runs grouped by (k, ng) with the mean over seeds; `complete` once all three seeds exist."""
    groups = {}
    for r in d["runs"].values():
        groups.setdefault((r["k"], r["ng"]), []).append(r)
    out = []
    for (k, ng), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["seed"])
        accs = [r["selection"]["accuracy"] for r in rows]
        cis = [r["selection"]["accuracy_ci"] for r in rows]
        r0 = rows[0]
        out.append({"k": k, "ng": ng, "cfg": r0["cfg"], "rows": rows, "accs": accs, "mean": float(np.mean(accs)),
                    "lo": float(np.mean([c["lo"] for c in cis])), "hi": float(np.mean([c["hi"] for c in cis])),
                    "complete": [r["seed"] for r in rows] == SEEDS, "flops": r0["flops"], "size": r0["size"],
                    "kv_bytes": r0["kv_bytes_per_token"], "block_params": r0["block_params"],
                    "canaries": next((r["canaries"] for r in rows if r["seed"] == 0), None)})
    return out


def point_mean(d, k, ng):
    return next((p["mean"] for p in points(d) if p["k"] == k and p["ng"] == ng and p["complete"]), None)


def stage_seeds(ctx, variants):
    for v in variants:
        d = ctx.res(v)
        for k in ctx.ks:
            cfg = chosen_cfg(d, k)
            if cfg is None:
                print(f"{v} k={k}: tune not finished, skipping", flush=True)
                continue
            for seed in SEEDS:
                run_point(ctx, v, k, cfg, seed)


# ---------------------------------------------------------------- kv pruning

def rank_groups(ctx, v, alive, nkv=8) -> list[tuple[int, int]]:
    """KV groups of the surviving blocks, least damaging first by the M5 kvscan rows, in the excised numbering.
    A group that would be the last one left in its layer is skipped (a layer keeps at least one group)."""
    rows = sorted(read_json(M5 / f"kvscan_{SHORT}_{v}.json")["rows"], key=S.damage_key)
    where = {layer: j for j, layer in enumerate(alive)}
    taken, out = {}, []
    for row in rows:
        (layer, g), = row["spec"]["groups"]
        if layer not in where or taken.get(layer, 0) >= nkv - 1:
            continue
        taken[layer] = taken.get(layer, 0) + 1
        out.append((where[layer], g))
    return out


def base_k(ctx, v):
    pts = [p for p in points(ctx.res(v)) if p["ng"] == 0 and p["complete"]]
    for floor in (TARGET_A, TARGET_FLOOR):
        ok = [p for p in pts if p["mean"] >= floor]
        if ok:  # the most blocks removed that still clears the floor (DECISIONS 2026-11-03: "smallest model", not smallest k)
            return max(ok, key=lambda p: p["k"])
    if pts and ctx.smoke:
        return max(pts, key=lambda p: p["mean"])
    sys.exit(f"{v}: no k has a selection mean >= {TARGET_FLOOR:.0%} "
             f"({', '.join(f'k={p['k']}: {p['mean']:.3f}' for p in pts) or 'no complete points'}); no KV pruning point")


def stage_kvprune(ctx, variants):
    step, max_steps = (SMOKE_KV_STEP, SMOKE_KV_MAX_STEPS) if ctx.smoke else (KV_STEP, KV_MAX_STEPS)
    for v in variants:
        d = ctx.res(v)
        b = base_k(ctx, v)
        k, cfg, ref = b["k"], b["cfg"], b["mean"]
        ranked = rank_groups(ctx, v, ctx.alive(v, k))
        print(f"{v}: kv pruning from k={k} (selection mean {ref:.3f}), {len(ranked)} rankable groups", flush=True)
        for s in range(1, max_steps + 1):
            if step * s > len(ranked):
                break
            groups = ranked[:step * s]
            for seed in SEEDS:
                run_point(ctx, v, k, cfg, seed, groups, "kvprune")
            m = point_mean(d, k, len(groups))
            print(f"{v} k={k} groups {len(groups)}: mean {m:.3f} vs {ref:.3f}", flush=True)
            if ref - m > KV_DROP:
                print(f"{v}: stop, {100 * (ref - m):.1f} points below the unpruned k={k}", flush=True)
                break


# ---------------------------------------------------------------- recheck

def run_recheck(ctx, v, arm_name, budget):
    d = ctx.res(v)["recheck"]
    sel_key = f"{arm_name}|{budget}"
    grid = [(lr, st) for lr, st in tune_grid(ctx.smoke)]
    for lr, st in grid:
        key = f"{sel_key}|lr{lr:g}_st{st}"
        if key in d["grid"]:
            print(f"skip {v} recheck {key}: already done", flush=True)
            continue
        ctx.status("recheck", v, key)
        print(f"== {v} recheck {key} {now()}", flush=True)
        ctx.fresh()
        arm = A.build_arm(ctx, v, arm_name, budget, 0)
        try:
            heldout, train = ctx.train_heldout(v, 0)
            steps = 4 if ctx.smoke else st
            t0 = time.perf_counter()
            state, curve = adapters.train_adapter(ctx.model, ctx.tok, train, heldout, arm.params, arm.forward, steps,
                                                  lr=lr, batch_size=ctx.train_bs, seed=0, n_evals=4)
            train_s = time.perf_counter() - t0
        finally:
            arm.close()
            ctx.release()
        cp = ckpt_dir(ctx.smoke) / f"{v}_recheck_{arm_name}_{budget}_lr{lr:g}_st{st}.pt"
        save_ckpt(cp, {"state": state})
        d["grid"][key] = {"arm": arm_name, "budget": budget, "lr": lr, "steps": st, "heldout_best": best_heldout(curve),
                          "curve": curve, "n_params": arm.n_params, "train_seconds": train_s, "checkpoint": str(cp),
                          "ts": now()}
        ctx.save(v)
        print(f"{v} recheck {key}: heldout {best_heldout(curve):.3f} ({(train_s - curve[-1]['eval_seconds']) / steps:.2f}"
              f"s/step, train {train_s:.0f}s, sel 0s)", flush=True)
    if sel_key in d["selection"]:
        return
    rows = [r for r in d["grid"].values() if r["arm"] == arm_name and r["budget"] == budget]
    best = min(rows, key=lambda r: (-r["heldout_best"], r["steps"], r["lr"]))
    ctx.status("recheck", v, f"{sel_key} selection")
    sel = ctx.records(v, "selection")
    base_pred = ctx.baseline_pred(v)
    ctx.fresh()
    arm = A.build_arm(ctx, v, arm_name, budget, 0)
    try:
        adapters.load_adapter_state(arm.params, torch.load(best["checkpoint"])["state"])
        t1 = time.perf_counter()
        pred = A.predict(arm, sel, ctx.a.batch_size)
        sel_s = time.perf_counter() - t1
    finally:
        arm.close()
        ctx.release()
    res = A.summarize(sel, pred, base_pred, ctx.n_boot())
    d["selection"][sel_key] = {"arm": arm_name, "budget": budget, "lr": best["lr"], "steps": best["steps"],
                               "heldout_best": best["heldout_best"], "n_params": best["n_params"],
                               "selection": res, "selection_seconds": sel_s, "ts": now()}
    ctx.save(v)
    if not ctx.smoke:
        append(A.SELECTION_LOG, {"ts": now(), "model": MODEL, "dtype": "fp32", "variant": v, "split": "selection",
                                 "seed": 0, "n": len(sel), "result": str(OUT / f"m7_{v}.json"),
                                 "intervention": f"m7-recheck-{arm_name}-{budget}-lr{best['lr']:g}-st{best['steps']}",
                                 "config": {"arm": arm_name, "budget": budget, "n_params": best["n_params"]},
                                 "accuracy": res["accuracy"]})
    print(f"{v} recheck {sel_key}: selection {res['accuracy']:.3f} gain {res['gain']:+.3f} "
          f"(train 0s, 0.00s/step, sel {sel_s:.0f}s)", flush=True)


def stage_recheck(ctx, variants):
    for v in variants:
        for arm_name, budget in RECHECK:
            run_recheck(ctx, v, arm_name, budget)


# ---------------------------------------------------------------- test

def named(p) -> str:
    return f"m7-k{p['k']}-g{p['ng']}-s0"


def pick_models(ctx, v):
    """A: fewest FLOPs/token with selection mean >= 0.95; B: fewest with mean >= the unmodified baseline."""
    pts = [p for p in points(ctx.res(v)) if p["complete"]]
    base = stats.accuracy([LABELS.index(r["label"]) for r in ctx.records(v, "selection")], ctx.baseline_pred(v))

    def cheapest(floor):
        ok = [p for p in pts if p["mean"] >= floor]
        if not ok and ctx.smoke and pts:  # smoke accuracies are meaningless; still exercise the path
            ok = [max(pts, key=lambda p: p["mean"])]
        return min(ok, key=lambda p: p["flops"]["total"]) if ok else None
    return cheapest(TARGET_A), cheapest(base), base


def stage_test(ctx, v):
    pa, pb, base_sel = pick_models(ctx, v)
    if pa is None:
        sys.exit(f"{v}: no point has selection mean >= {TARGET_A:.0%}; model A does not exist")
    chosen = {"A": pa, "B": pb}
    names = {}
    for label, p in chosen.items():
        if p is not None:
            names.setdefault(named(p), (p, []))[1].append(label)
    if not ctx.smoke:
        for name in names:
            if A.ledger_has(v, name):
                print(f"refusing: test ledger already has {MODEL} {v} {name}", file=sys.stderr)
                sys.exit(2)
    test = ctx.records(v, "test")
    base_pred = ctx.baseline_pred(v, "test")
    base_can = ctx.base_canaries()
    out = {"model": MODEL, "variant": v, "split": "test", "n": len(test), "selection_baseline": base_sel,
           "baseline_accuracy": stats.accuracy([LABELS.index(r["label"]) for r in test], base_pred),
           "A": named(pa), "B": named(pb) if pb else None, "runs": {}}
    for name, (p, labels) in names.items():
        ctx.status("test", v, name)
        t0 = time.perf_counter()
        groups = [tuple(g) for g in p["rows"][0]["groups"]]
        r0 = next(r for r in p["rows"] if r["seed"] == 0)
        saved = torch.load(r0["checkpoint"])
        arm, _ = build_point(ctx, v, p["k"], p["cfg"], 0, groups, means=saved.get("means"))
        try:
            adapters.load_adapter_state(arm.params, saved["state"])
            adapters.merge_lora(arm.handles[0])
            pred = A.predict(arm, test, ctx.a.batch_size)
            can = A.canaries_for(ctx, arm, v, "heal", base_can)
        finally:
            arm.close()
            ctx.release()
        row = A.summarize(test, pred, base_pred, ctx.n_boot())
        row.update(labels=labels, k=p["k"], ng=p["ng"], cfg=p["cfg"], selection_mean=p["mean"], canaries=can,
                   flops=p["flops"], seconds=time.perf_counter() - t0)
        out["runs"][name] = row
        print(f"test {v} {'/'.join(labels)} {name}: accuracy {row['accuracy']:.3f} (base {out['baseline_accuracy']:.3f}) "
              f"gain {row['gain']:+.3f}", flush=True)
        if not ctx.smoke:
            append(A.TEST_LEDGER, {"ts": now(), "model": MODEL, "dtype": "fp32", "variant": v, "split": "test",
                                   "seed": 0, "intervention": name, "n": len(test), "accuracy": row["accuracy"],
                                   "config": {"k": p["k"], "kv_groups": p["ng"], **p["cfg"]},
                                   "result": str(OUT / f"test_{v}.json")})
        write(OUT / f"test_{v}{sfx(ctx.smoke)}.json", out)  # after each pass, so a crash keeps what the ledger has
    write(OUT / f"test_{v}{sfx(ctx.smoke)}.json", out)


def h7(smoke: bool) -> dict:
    """Resolve H7 from the per-variant points and test files."""
    per, falsified, supported = {}, False, True
    for v in VARIANTS:
        p = OUT / f"m7_{v}{sfx(smoke)}.json"
        pts = [q for q in points(read_json(p)) if q["complete"]] if p.exists() else []
        big = [q for q in pts if q["k"] >= H7_MIN_K]
        f = bool(big) and all(q["mean"] < H7_FALSIFY for q in big)
        tp = OUT / f"test_{v}{sfx(smoke)}.json"
        t = read_json(tp) if tp.exists() else None
        a_row = t["runs"][t["A"]] if t else None
        ok = bool(a_row) and a_row["k"] >= H7_MIN_K and a_row["accuracy"] >= H7_TEST
        per[v] = {"falsified": f, "A_k": a_row and a_row["k"], "A_test": a_row and a_row["accuracy"], "A_ok": ok,
                  "have_test": t is not None, "n_points_k12": len(big)}
        falsified |= f
        supported &= ok
    if falsified:
        verdict = "falsified"
    elif supported:
        verdict = "supported"
    elif not all(per[v]["have_test"] for v in VARIANTS):
        verdict = "pending (needs both variants' test runs)"
    else:
        verdict = "not supported"
    return {"verdict": verdict, "per_variant": per}


# ---------------------------------------------------------------- table

def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def stage_table(smoke: bool):
    lines = ["# M7 efficiency frontier: selection-split tables", "",
             "Accuracy in percent. Selection mean is over seeds 0, 1, 2; the interval is the mean of the per-seed "
             "grouped-bootstrap bounds. FLOPs are analytic prefill FLOPs per token (embedding and LM head excluded), "
             "relative to the unmodified model. Block params are the attention and MLP matrices; total params include "
             "the embedding. KV bytes per token are bf16.", ""]
    for v in VARIANTS:
        p = OUT / f"m7_{v}{sfx(smoke)}.json"
        if not p.exists():
            lines += [f"## {v}", "", "no results yet", ""]
            continue
        d = read_json(p)
        sel = M5 / f"select_{SHORT}_{v}.json"
        base = read_json(sel)["baseline"]["accuracy"] if sel.exists() else None
        lines += [f"## {v}", "", f"Unmodified selection baseline: {pct(base)}", "",
                  "### Frontier", "",
                  "| k | KV groups pruned | config | selection mean (CI) | per seed | canaries | FLOPs rel | block params "
                  "| total params | non-embedding | bf16 MB | KV bytes/token |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for q in points(d):
            can = q["canaries"]
            seeds = ", ".join(f"s{r['seed']} {pct(r['selection']['accuracy'])}" for r in q["rows"])
            lines.append(
                f"| {q['k']} | {q['ng']} | {cfg_text(q['cfg'])} | {pct(q['mean'])} ({pct(q['lo'])}-{pct(q['hi'])})"
                f"{'' if q['complete'] else ' partial'} | {seeds} | {pct(can and can['accuracy'])} | "
                f"{q['flops']['relative']:.3f} | {q['block_params']['total']:,} | {q['size']['total']:,} | "
                f"{q['size']['non_embedding']:,} | {q['size']['bf16_bytes'] / 1e6:.0f} | {q['kv_bytes']:,} |")
        lines += ["", "### Tune grid (held-out best accuracy, seed 0)", "",
                  "| k | config | kind | held-out best |", "|---|---|---|---|"]
        for r in sorted(d["tune"].values(), key=lambda r: (r["k"], r["kind"], r["cfg"]["lr"], r["cfg"]["steps"])):
            chosen = chosen_cfg(d, r["k"]) == r["cfg"]
            lines.append(f"| {r['k']} | {cfg_text(r['cfg'])} | {r['kind']}{' (chosen)' if chosen else ''} | "
                         f"{pct(r['heldout_best'])} |")
        lines += ["", "### Recheck with tuned lr (seed 0, reference rows)", "",
                  "| arm | budget | params | lr | steps | held-out | selection | gain |", "|---|---|---|---|---|---|---|---|"]
        for r in d["recheck"]["selection"].values():
            lines.append(f"| {r['arm']} | {r['budget']} | {r['n_params']:,} | {r['lr']:g} | {r['steps']} | "
                         f"{pct(r['heldout_best'])} | {pct(r['selection']['accuracy'])} | {pct(r['selection']['gain'])} |")
        tp = OUT / f"test_{v}{sfx(smoke)}.json"
        if tp.exists():
            t = read_json(tp)
            lines += ["", f"### Test ({v}), unmodified baseline {pct(t['baseline_accuracy'])}", "",
                      "| model | point | test acc | gain | selection mean | canary acc | canary eligible | FLOPs rel |",
                      "|---|---|---|---|---|---|---|---|"]
            for name, r in t["runs"].items():
                c = r["canaries"]
                lines.append(f"| {'/'.join(r['labels'])} | {name} | {pct(r['accuracy'])} | {pct(r['gain'])} | "
                             f"{pct(r['selection_mean'])} | {pct(c and c['accuracy'])} | "
                             f"{c['eligible'] if c else 'n/a'} | {r['flops']['relative']:.3f} |")
        lines.append("")
    h = h7(smoke)
    lines += ["## H7", "", f"Verdict: **{h['verdict']}**", "",
              "Supported if A has k >= 12 on both variants and A's test accuracy >= 93% on both; falsified if every "
              "k >= 12 point is below 90% on selection on either variant.", "",
              "| variant | A k | A test | A ok | k>=12 points | all < 90 |", "|---|---|---|---|---|---|"]
    for v, r in h["per_variant"].items():
        lines.append(f"| {v} | {r['A_k'] if r['A_k'] is not None else 'n/a'} | {pct(r['A_test'])} | {r['A_ok']} | "
                     f"{r['n_points_k12']} | {r['falsified']} |")
    dst = OUT / f"report_tables{sfx(smoke)}.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n")
    print(f"-> {dst}  (H7: {h['verdict']})")


# ---------------------------------------------------------------- driver

def stage_sweep(ctx, variants):
    try:
        stage_tune(ctx, variants)
        stage_seeds(ctx, variants)
        stage_kvprune(ctx, variants)
        stage_recheck(ctx, variants)
    except Stop:
        print(f"stopping: past {ctx.a.until}", flush=True)
        ctx.status("sweep", None, "stopped", done=False)
        return
    ctx.status("sweep", None, "complete", done=True)
    print("sweep complete", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--until", help="sweep: no new run starts after this UTC time, HH:MMZ")
    ap.add_argument("--batch-size", type=int, default=8, help="training batch size")
    ap.add_argument("--eval-batch-size", type=int, default=8, help="scoring batch size")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.stage == "table":
        return stage_table(a.smoke)
    if a.stage == "test" and a.variant is None:
        sys.exit("--variant is required for the test stage")
    variants = [a.variant] if a.variant else VARIANTS
    ctx = FCtx(a, A.parse_until(a.until) if a.until else None)
    try:
        if a.stage == "test":
            stage_test(ctx, a.variant)
        else:
            globals()[f"stage_{a.stage}"](ctx, variants)
            ctx.status(a.stage, a.variant, "complete", done=True) if a.stage != "sweep" else None
    except Stop:
        print("stopping: past --until", flush=True)


if __name__ == "__main__":
    main()
