"""Capability studies on Qwen3-0.6B (docs/CONTRACT-capability.md). Runs on the pod, from the repo root.

  uv run python scripts/capability_sweep.py --stage {a-tune,a-seeds,a-test,a2,b-calibrate,b-frontier,b-test,table}
      [--variant V] [--kv] [--smoke]

Study A (keep general ability while cutting): a-tune trains the 9-run grid on the M7 A-chain warm start (seed 0),
a-seeds applies the pick rule (no canaries, no perplexity) and runs seeds 1 and 2 of the pick plus selection,
a-test runs the one named model per variant on test, a2 repeats the pick at k = 9 and 6 if its canaries miss the bar.
Study B (harder task): b-calibrate walks the tiers R1.. and freezes the first with healed-unmodified accuracy in
60-90%, b-frontier runs k in {0, 9, 12} (and, with --kv, KV pruning) on the frozen tier, b-test spends the test once.
Results go to var/results/cap/ (a_{v}.json, a2_{v}.json, b_{v}.json, calib_{v}.json, status_{stream}.json,
report_tables.md) and checkpoints to var/ckpt/cap/. Every run is resumable (a run key already in the results file is
skipped). Study A selection passes and tests use the project's selection_log.jsonl / test_ledger.jsonl; Study B uses
var/results/cap/selection_log_b.jsonl and test_ledger_b.jsonl. --smoke (16 task, 32 general rows, 8 steps, k = 2,
tiers on 16 items, *_smoke files) never writes any ledger.

Study B2 (docs/CONTRACT-capability.md): with SURGERY_FAMILY=hard the same b-calibrate / b-frontier / b-test / table stages
run on the harder task family (surgery/task/hard.py, rungs H1-H5 plus the between rungs). Results, ledgers, checkpoints and
tier data go to .../hard subdirectories and models are named cap-b2-*, so Study A and B files are never touched; Study A
stages and variant `sensible` exit. b-calibrate walks the rungs by the contract's rule (see walk_hard). scripts/run_hard.sh.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import adapter_sweep as A  # noqa: E402
import frontier_sweep as F  # noqa: E402
from surgery import adapters, capability as C, stats, structure  # noqa: E402
from surgery.canaries import load_ppl_text, perplexity  # noqa: E402
from surgery.eval import _free, label_logits  # noqa: E402
from surgery.prompt import build_prompt  # noqa: E402
from surgery.task.generator import FAMILY, LABELS, generate, read_jsonl, write_jsonl  # noqa: E402
from surgery.task.hard import BETWEEN, HTIERS, HardKnobs, generate_hard  # noqa: E402
from surgery.task.splits import _make_splits  # noqa: E402

MODEL, VARIANTS, SEEDS = A.MODEL, A.VARIANTS, F.SEEDS
HARD = FAMILY == "hard"   # Study B2: nothing of Study A or B is read-written, names carry b2
OUT = Path("var/results/cap") / ("hard" if HARD else "")
CKPT = Path("var/ckpt/cap") / ("hard" if HARD else "")
GEN = Path("var/data/general")
CAP_DATA = Path("var/data/cap") / ("hard" if HARD else "")
PFX = "cap-b2-" if HARD else "cap-b-"
SELECTION_LOG_B = OUT / "selection_log_b.jsonl"
TEST_LEDGER_B = OUT / "test_ledger_b.jsonl"
STAGES = ["a-tune", "a-seeds", "a-test", "a2", "b-calibrate", "b-frontier", "b-test", "table"]

now, write, read_json, append, sfx = A.now, A.write, A.read_json, A.append, A.smoke_suffix

# Study A
A_STEPS, SMOKE_STEPS = 1500, 8
FULL_LRS, SMOKE_FULL_LRS = [1e-5, 3e-5, 1e-4], [1e-4]
FULL_LAMS, SMOKE_FULL_LAMS = [0.3, 3.0], [0.3]
LORA_LR, LORA_RANK = 1e-3, 40
LORA_LAMS, SMOKE_LORA_LAMS = [0.3, 1.0, 3.0], [1.0]
WARM = {"rule": 48, "sensible": 64}   # KV groups pruned in the M7 A chain at k = 12
WARM_K, SMOKE_WARM = 12, (2, 4)
A2_KS, SMOKE_A2_KS = [9, 6], [2, 1]
DEV_EVAL = 256                         # general dev rows scored at each in-training evaluation

# Study B
CAL_LR, CAL_STEPS, CAL_BAND = 1e-3, 300, (0.60, 0.90)
CAL_SEED, FINAL_SEED = 20, 2
CAL_SIZES, SMOKE_CAL_SIZES = {"discovery": 1000, "calib": 500}, {"discovery": 32, "calib": 16}
FINAL_SIZES = {"discovery": 3000, "selection": 1500, "test": 1500}
SMOKE_FINAL_SIZES = {"discovery": 32, "selection": 16, "test": 16}
B_STEPS, B_LR, B_KS, SMOKE_B_KS = 1200, 1e-3, [0, 9, 12], [0, 2]
B_KV, SMOKE_B_KV = [16, 32], [2, 4]
B_N_TRAIN, B_WITHIN = 2000, 0.05
EPS = 1e-9
OLD_R4_ACC = 0.996   # Study B's R4 calibration accuracy, the "previous rung" of H1


# ---------------------------------------------------------------- context

class CCtx(F.FCtx):
    """FCtx plus a lazily loaded teacher, the general data, tier-data override and this sweep's status file."""

    def __init__(self, a, stage):
        super().__init__(a)
        self.stage, self.n_train, self.tier = stage, 800, False
        self._teacher = self._gen = None

    @property
    def teacher(self):
        """A second, unmodified model on the device: eval, no gradients."""
        if self._teacher is None:
            self._teacher = copy.deepcopy(self.pristine).to(self.device).eval()
            self._teacher.requires_grad_(False)
        return self._teacher

    def general(self):
        if self._gen is None:
            train, dev = read_jsonl(GEN / "train.jsonl"), read_jsonl(GEN / "dev.jsonl")
            self._gen = (train[:32], dev[:32]) if self.smoke else (train, dev)
        return self._gen

    def records(self, variant, split):
        key = (variant, split)
        if self.tier and key not in self._data:
            self._data[key] = read_jsonl(CAP_DATA / f"{variant}_s{FINAL_SEED}_{split}{sfx(self.smoke)}.jsonl")
        return super().records(variant, split)

    def train_heldout(self, variant, seed):
        n_held, n_train = (16, 16) if self.smoke else (200, self.n_train)
        return adapters.split_discovery(self.records(variant, "discovery"), seed, n_held, n_train)

    def baseline_pred(self, variant, split="selection"):
        if not self.tier:
            return super().baseline_pred(variant, split)  # Study A data: the M5 cache applies
        key = (variant, split)
        if key not in self._base:
            recs = self.records(variant, split)
            path = OUT / f"base_{variant}_{split}{sfx(self.smoke)}.json"
            ids = [r["id"] for r in recs]
            if path.exists() and read_json(path)["ids"] == ids:
                self._base[key] = read_json(path)["pred"]
            else:
                self.fresh()
                lg = label_logits(self.model, self.tok, [r["prompt"] for r in recs], self.a.batch_size)
                self._base[key] = lg.argmax(-1).tolist()
                write(path, {"ids": ids, "pred": self._base[key]})
        return self._base[key]

    def status(self, stage, variant, run, done=False):
        write(OUT / f"status_{self.a.stream}{sfx(self.smoke)}.json",
              {"stage": stage, "variant": variant, "run": run, "ts": now(), "done": done})


def ckpt_dir(smoke: bool) -> Path:
    return CKPT / ("smoke" if smoke else "")


def load_doc(path: Path, init: dict) -> dict:
    return read_json(path) if path.exists() else init


def forward_of(ctx):
    model, tok, ids = ctx.model, ctx.tok, ctx.ids
    return lambda recs: adapters.full_logits(model, adapters.encode(model, tok, recs), ids)  # noqa: E731


def predict_model(ctx, recs) -> list[int]:
    return A.predict(A.Arm(None, forward_of(ctx), [], 0), recs, ctx.a.batch_size)


def ppl_of(ctx) -> dict:
    text = load_ppl_text()
    return perplexity(ctx.model, ctx.tok, text[:6000] if ctx.smoke else text)


def structure_row(ctx, v, k, groups) -> dict:
    """size, KV bytes, FLOPs and block params of ctx.model as run_point records them."""
    alive = ctx.alive(v, k)
    orig = [(alive[layer], g) for layer, g in groups]
    return {"groups_original": [list(g) for g in orig], "seq_len": ctx.seq_len(v),
            "size": structure.model_size(ctx.model), "kv_bytes_per_token": structure.kv_cache_bytes_per_token(ctx.model),
            "flops": structure.flops_per_token(ctx.cfg, ctx.seq_len(v), order=alive, pruned_groups=orig),
            "block_params": structure.param_count(ctx.cfg, order=alive, pruned_groups=orig)}


# ---------------------------------------------------------------- Study A: warm start

def m7_row(ctx, v, k, ng, seed):
    p = F.OUT / f"m7_{v}{sfx(ctx.smoke)}.json"
    return read_json(p)["runs"].get(F.run_key(k, ng, seed)) if p.exists() else None


def warm_start(ctx, v, k, ng, seed, cold_ok=False):
    """Rebuild the M7 point (k, ng, seed) on ctx.model with its heal merged and verify the structure against the M7 row.
    Returns that row. With cold_ok and no M7 row: excise only, returns None."""
    row = m7_row(ctx, v, k, ng, seed)
    if row is None:
        if not cold_ok:
            sys.exit(f"{v}: no M7 row {F.run_key(k, ng, seed)} in {F.OUT / f'm7_{v}{sfx(ctx.smoke)}.json'}")
        print(f"{v} k={k} s{seed}: no M7 checkpoint, excise only (cold start)", flush=True)
        ctx.fresh()
        structure.excise(ctx.model, ctx.alive(v, k))
        return None
    groups = [tuple(g) for g in row["groups"]]
    saved = torch.load(row["checkpoint"], map_location="cpu")
    arm, _ = F.build_point(ctx, v, k, row["cfg"], seed, groups, means=saved.get("means"))
    adapters.load_adapter_state(arm.params, saved["state"])
    adapters.merge_lora(arm.handles[0])
    arm.close()
    _free(ctx.model)
    now_ = structure_row(ctx, v, k, groups)
    ok = (now_["size"] == row["size"] and now_["kv_bytes_per_token"] == row["kv_bytes_per_token"]
          and math.isclose(structure.flops_per_token(ctx.cfg, row["seq_len"], order=ctx.alive(v, k),
                                                     pruned_groups=now_["groups_original"])["total"],
                           row["flops"]["total"], rel_tol=1e-9))
    if not ok:
        sys.exit(f"{v} k={k} g{ng} s{seed}: rebuilt structure differs from the M7 row (size/KV/FLOPs)")
    return row


# ---------------------------------------------------------------- Study A: one trained run

def spec_name(spec) -> str:
    return f"{'ceiling' if spec['ceiling'] else spec['arm']}_lr{spec['lr']:g}_lam{spec['lam']:g}"


def a_key(k, ng, spec, seed) -> str:
    return f"k{k}|g{ng}|{spec_name(spec)}|s{seed}"


def mk_spec(arm, lr, lam, ceiling=False) -> dict:
    return {"arm": arm, "lr": lr, "lam": lam, "ceiling": ceiling}


def run_a(ctx, v, doc, path, prefix, k, ng, spec, seed, cold_ok=False):
    """Warm start (or cold excise), train the combined loss, keep the best state, round it through bf16, score
    held-out accuracy, general dev KL, canaries and perplexity on the model as saved. Selection is not scored here."""
    key = a_key(k, ng, spec, seed)
    if key in doc["runs"]:
        print(f"skip {v} {key}: already done", flush=True)
        return doc["runs"][key]
    ctx.status(prefix, v, key)
    print(f"== {v} {prefix} {key} {now()}", flush=True)
    base_can = ctx.base_canaries()  # needs an unmodified model: before building ours
    steps = SMOKE_STEPS if ctx.smoke else A_STEPS
    gen_train, gen_dev = ctx.general()
    dev_eval = gen_dev[::max(1, len(gen_dev) // DEV_EVAL)][:DEV_EVAL]  # dev is sorted by source: an even sample of all three
    heldout, train = ctx.train_heldout(v, seed)
    m7 = warm_start(ctx, v, k, ng, seed, cold_ok)
    model, tok, teacher = ctx.model, ctx.tok, ctx.teacher
    lora = spec["arm"] == "lora"
    cfg = F.cfg_of(spec["lr"], steps, LORA_RANK, F.QKVO) if lora else {"lr": spec["lr"], "steps": steps,
                                                                       "rank": 0, "modules": []}
    arm = F.lora_arm(ctx, len(ctx.alive(v, k)), cfg, seed) if lora else None
    try:
        params = ([p for p in arm.params.parameters() if p.requires_grad] if lora
                  else [p for _, p in C.full_trainable(model)])
        fwd = forward_of(ctx)
        t0 = time.perf_counter()
        state, curve = C.train_combined(model, teacher, tok, train, heldout, gen_train, dev_eval, params, fwd,
                                        steps, spec["lr"], spec["lam"], batch_size=ctx.train_bs, seed=seed, n_evals=4,
                                        task_weight=0.0 if spec["ceiling"] else 1.0)
        train_s = time.perf_counter() - t0
        cp = ckpt_dir(ctx.smoke) / f"{v}_{prefix}_{key.replace('|', '_')}.pt"
        C.save_trainables(cp, state, {"variant": v, "key": key, "spec": spec, "cfg": cfg, "seed": seed})
        C.load_trainables(model, cp)  # score the model exactly as the checkpoint holds it
        if lora:
            adapters.merge_lora(arm.handles[0])
        _free(model)
        t1 = time.perf_counter()
        held = adapters.heldout_accuracy(fwd, heldout, ctx.a.batch_size)
        dev_kl = C.general_dev_kl(model, teacher, tok, gen_dev, ctx.a.batch_size)
        can = A.canaries_for(ctx, None, v, "heal", base_can)
        ppl = ppl_of(ctx)
        score_s = time.perf_counter() - t1
        row = {"k": k, "ng": ng, "seed": seed, **spec, "cold": m7 is None, "cfg": cfg, "steps": steps, "curve": curve,
               "heldout": held, "dev_kl": dev_kl, "canaries": can, "ppl": ppl, "train_seconds": train_s,
               "step_seconds": (train_s - curve[-1]["eval_seconds"]) / steps, "score_seconds": score_s,
               "checkpoint": str(cp), "warm_checkpoint": m7 and m7["checkpoint"], "ts": now(),
               **structure_row(ctx, v, k, [tuple(g) for g in m7["groups"]] if m7 else [])}
    finally:
        if arm is not None:
            arm.close()
        ctx.release()
    doc["runs"][key] = row
    write(path, doc)
    print(f"{v} {key}: heldout {row['heldout']:.3f} dev_kl {row['dev_kl']:.4f} canaries {can['accuracy']:.3f} "
          f"ppl {ppl['ppl']:.2f} ({row['step_seconds']:.2f}s/step, train {train_s:.0f}s)", flush=True)
    return row


def rebuild_a(ctx, v, row):
    """The model of a saved Study A row: warm start (or cold excise) plus its checkpoint, adapters merged."""
    warm_start(ctx, v, row["k"], row["ng"], row["seed"], cold_ok=row["cold"])
    arm = F.lora_arm(ctx, len(ctx.alive(v, row["k"])), row["cfg"], row["seed"]) if row["arm"] == "lora" else None
    C.load_trainables(ctx.model, row["checkpoint"])
    if arm is not None:
        adapters.merge_lora(arm.handles[0])
        arm.close()
    _free(ctx.model)
    return ctx.model


def a_doc(ctx, v, prefix="a") -> tuple[dict, Path]:
    path = OUT / f"{prefix}_{v}{sfx(ctx.smoke)}.json"
    return load_doc(path, {"model": MODEL, "variant": v, "runs": {}, "selection": {}}), path


def warm_point(ctx, v) -> tuple[int, int]:
    return SMOKE_WARM if ctx.smoke else (WARM_K, WARM[v])


def grid_specs(smoke: bool, best_lr=None):
    """(full lr sweep, then best-lr lambda runs, then LoRA runs); the ceiling is added by the caller."""
    lrs = SMOKE_FULL_LRS if smoke else FULL_LRS
    if best_lr is None:
        return [mk_spec("full", lr, 1.0) for lr in lrs]
    return [mk_spec("full", best_lr, lam) for lam in (SMOKE_FULL_LAMS if smoke else FULL_LAMS)]


def lora_specs(smoke: bool):
    return [mk_spec("lora", LORA_LR, lam) for lam in (SMOKE_LORA_LAMS if smoke else LORA_LAMS)]


def row_rank(r):
    return C.rank_key(r["heldout"], r["dev_kl"])


def stage_a_tune(ctx, variants):
    for v in variants:
        doc, path = a_doc(ctx, v)
        k, ng = warm_point(ctx, v)
        run = lambda spec: run_a(ctx, v, doc, path, "a", k, ng, spec, 0)  # noqa: E731
        sweep = [run(s) for s in grid_specs(ctx.smoke)]
        best = max(sweep, key=row_rank)
        print(f"{v}: best full-weight lr {best['lr']:g} (heldout {best['heldout']:.3f}, dev_kl {best['dev_kl']:.4f})",
              flush=True)
        for s in grid_specs(ctx.smoke, best["lr"]):
            run(s)
        for s in lora_specs(ctx.smoke):
            run(s)
        run(mk_spec("full", best["lr"], 1.0, ceiling=True))
        ctx.status("a-tune", v, "complete", done=True)


def seed0_rows(doc):
    return [r for r in doc["runs"].values() if r["seed"] == 0]


def expected_grid(smoke: bool) -> int:
    return 4 if smoke else 9


def sel_log(ctx, v, row, name, res, path, stage="cap-a"):
    if ctx.smoke:
        return
    append(A.SELECTION_LOG, {"ts": now(), "model": MODEL, "dtype": "fp32", "variant": v, "split": "selection",
                             "seed": row["seed"], "n": res["n"], "result": str(path),
                             "intervention": f"{stage}-{v}-{name}-s{row['seed']}",
                             "config": {"k": row["k"], "kv_groups": row["ng"], "arm": row["arm"], "lr": row["lr"],
                                        "lam": row["lam"]},
                             "accuracy": res["accuracy"]})


def stage_a_seeds(ctx, variants):
    for v in variants:
        doc, path = a_doc(ctx, v)
        rows = seed0_rows(doc)
        if len(rows) < expected_grid(ctx.smoke):
            print(f"{v}: seed-0 grid has {len(rows)} of {expected_grid(ctx.smoke)} runs, run a-tune first", flush=True)
            continue
        pick, fallback = C.pick_a(rows)  # held-out and dev KL only
        if pick is None:
            doc["pick"] = {"none": True, "fallback": fallback and a_key(fallback["k"], fallback["ng"], fallback, 0),
                           "fallback_heldout": fallback and fallback["heldout"],
                           "note": "no run reached 99% held-out: H-A not supported"}
            write(path, doc)
            print(f"{v}: no run reached {C.GATE:.0%} held-out; best {fallback and fallback['heldout']:.3f}", flush=True)
            continue
        spec = mk_spec(pick["arm"], pick["lr"], pick["lam"])
        doc["pick"] = {"none": False, "name": spec_name(spec), "spec": spec, "k": pick["k"], "ng": pick["ng"],
                       "heldout": pick["heldout"], "dev_kl": pick["dev_kl"]}
        write(path, doc)
        print(f"{v}: pick {spec_name(spec)} (heldout {pick['heldout']:.3f}, dev_kl {pick['dev_kl']:.4f})", flush=True)
        for seed in SEEDS[1:]:
            run_a(ctx, v, doc, path, "a", pick["k"], pick["ng"], spec, seed)
        score_selection(ctx, v, doc, path, [(a_key(pick["k"], pick["ng"], spec, s), f"s{s}") for s in SEEDS],
                        spec_name(spec), "cap-a")
        accs = [doc["selection"][f"s{s}"]["selection"]["accuracy"] for s in SEEDS]
        doc["selection_mean"] = float(np.mean(accs))
        write(path, doc)
        print(f"{v}: selection per seed {accs}, mean {doc['selection_mean']:.3f}", flush=True)
        ctx.status("a-seeds", v, "complete", done=True)


def score_selection(ctx, v, doc, path, items, name, stage):
    """Selection accuracy (A.summarize) of saved Study A rows: items are [(run key, label)]."""
    sel = ctx.records(v, "selection")
    base_pred = ctx.baseline_pred(v)
    for key, label in items:
        if label in doc["selection"]:
            continue
        row = doc["runs"][key]
        ctx.status(stage, v, f"{key} selection")
        rebuild_a(ctx, v, row)
        try:
            t0 = time.perf_counter()
            res = A.summarize(sel, predict_model(ctx, sel), base_pred, ctx.n_boot())
        finally:
            ctx.release()
        doc["selection"][label] = {"run": key, "selection": res, "selection_seconds": time.perf_counter() - t0,
                                   "canaries": row["canaries"], "ppl": row["ppl"], "heldout": row["heldout"],
                                   "dev_kl": row["dev_kl"], "ts": now()}
        write(path, doc)
        sel_log(ctx, v, row, name if stage == "cap-a" else f"k{row['k']}-{name}", res, path, stage)
        print(f"{v} {key}: selection {res['accuracy']:.3f} gain {res['gain']:+.3f}", flush=True)


def ledger_has_b(v, name) -> bool:
    if not TEST_LEDGER_B.exists():
        return False
    import json
    for line in TEST_LEDGER_B.read_text().splitlines():
        r = json.loads(line) if line.strip() else {}
        if r.get("model") == MODEL and r.get("variant") == v and r.get("intervention") == name:
            return True
    return False


def stage_a_test(ctx, variants):
    for v in variants:
        doc, path = a_doc(ctx, v)
        pick = doc.get("pick")
        if not pick or pick.get("none") or len(doc["selection"]) < len(SEEDS):
            print(f"{v}: no pick with selection on all seeds (run a-seeds); no test", flush=True)
            continue
        spec = pick["spec"]
        name = f"cap-a-{v}-{pick['name']}"
        if not ctx.smoke and A.ledger_has(v, name):
            print(f"refusing: test ledger already has {MODEL} {v} {name}", file=sys.stderr)
            sys.exit(2)
        row = doc["runs"][a_key(pick["k"], pick["ng"], spec, 0)]
        test = ctx.records(v, "test")
        base_pred = ctx.baseline_pred(v, "test")
        base_can = ctx.base_canaries()
        ctx.status("a-test", v, name)
        t0 = time.perf_counter()
        rebuild_a(ctx, v, row)
        try:
            res = A.summarize(test, predict_model(ctx, test), base_pred, ctx.n_boot())
            can = A.canaries_for(ctx, None, v, "heal", base_can)
            ppl = ppl_of(ctx)
        finally:
            ctx.release()
        verdict = C.verdict_a(res["accuracy"], can["accuracy"])
        doc["test"] = {"name": name, "split": "test", "n": len(test), "test": res, "canaries": can, "ppl": ppl,
                       "baseline_accuracy": stats.accuracy([LABELS.index(r["label"]) for r in test], base_pred),
                       "verdict": verdict, "seconds": time.perf_counter() - t0, "ts": now()}
        if not ctx.smoke:
            append(A.TEST_LEDGER, {"ts": now(), "model": MODEL, "dtype": "fp32", "variant": v, "split": "test",
                                   "seed": 0, "intervention": name, "n": len(test), "accuracy": res["accuracy"],
                                   "config": {"k": pick["k"], "kv_groups": pick["ng"], **spec},
                                   "result": str(path)})
        write(path, doc)
        print(f"test {v} {name}: accuracy {res['accuracy']:.3f}, canaries {can['accuracy']:.3f} -> {verdict}", flush=True)
        ctx.status("a-test", v, "complete", done=True)


def stage_a2(ctx, variants):
    for v in variants:
        doc, _ = a_doc(ctx, v)
        t = doc.get("test")
        if not t or doc["pick"].get("none"):
            print(f"{v}: a-test has not run; a2 needs its canaries", flush=True)
            continue
        if t["canaries"]["accuracy"] >= C.CANARY_SUPPORTED - EPS and not ctx.smoke:
            print(f"{v}: canaries {t['canaries']['accuracy']:.3f} meet the bar, a2 not needed", flush=True)
            continue
        spec = doc["pick"]["spec"]
        d2, p2 = a_doc(ctx, v, "a2")
        for k in (SMOKE_A2_KS if ctx.smoke else A2_KS):
            for seed in SEEDS:
                run_a(ctx, v, d2, p2, "a2", k, 0, spec, seed, cold_ok=True)
            score_selection(ctx, v, d2, p2, [(a_key(k, 0, spec, s), f"k{k}|s{s}") for s in SEEDS], spec_name(spec),
                            "cap-a2")
            accs = [d2["selection"][f"k{k}|s{s}"]["selection"]["accuracy"] for s in SEEDS]
            d2.setdefault("selection_mean", {})[f"k{k}"] = float(np.mean(accs))
            write(p2, d2)
        ctx.status("a2", v, "complete", done=True)


# ---------------------------------------------------------------- Study B: tiers and data

def build_splits(v, seed, sizes, knobs, smoke):
    """Template-disjoint splits; under smoke a plain slice of one small generated pool (too few templates otherwise)."""
    if not smoke:
        return _make_splits(v, seed, sizes, knobs=knobs)
    gen = generate_hard if isinstance(knobs, HardKnobs) else generate
    recs, out, i = gen(sum(sizes.values()), seed, knobs, v), {}, 0
    for name, n in sizes.items():
        out[name], i = recs[i:i + n], i + n
    return out, {"variant": v, "seed": seed, "sizes": dict(sizes), "smoke": True}


def mean_tokens(ctx, recs) -> float:
    return float(np.mean([len(ctx.tok(build_prompt(ctx.tok, r["prompt"]))["input_ids"]) for r in recs]))


def in_band(acc: float) -> bool:
    return CAL_BAND[0] - EPS <= acc <= CAL_BAND[1] + EPS


def walk_hard(rungs: dict, run_rung, order=None, prev_acc: float = OLD_R4_ACC):
    """B2 calibration walk (contract): rungs in order, freeze the first in the band. A rung below the band whose previous
    rung was above it (H1's previous is the old R4) runs its between rung once; freeze that if in band, else none. All
    rungs above the band (H5 included): none. `rungs` caches finished rungs; `run_rung(name)` scores one. Returns the
    frozen rung name or None."""
    def acc_of(name):
        if name not in rungs:
            rungs[name] = run_rung(name)
        return rungs[name]["accuracy"]

    prev = prev_acc
    for name in order or list(HTIERS):
        acc = acc_of(name)
        if in_band(acc):
            return name
        if acc > CAL_BAND[1] + EPS:
            prev = acc
            continue
        if prev > CAL_BAND[1] + EPS:
            between = BETWEEN[name][0]
            return between if in_band(acc_of(between)) else None
        return None
    return None


def calibrate_rung(ctx, v, tier) -> dict:
    sizes = SMOKE_CAL_SIZES if ctx.smoke else CAL_SIZES
    knobs = C.knobs_of(tier)
    splits, _ = build_splits(v, CAL_SEED, sizes, knobs, ctx.smoke)
    ctx._data[(v, "discovery")], ctx._data[(v, "calib")] = splits["discovery"], splits["calib"]
    calib = splits["calib"]
    layers, modules, rank = A.LORA["1m"]
    cfg = F.cfg_of(CAL_LR, CAL_STEPS, rank, modules)
    heldout, train = ctx.train_heldout(v, 0)
    ctx.fresh()
    arm = F.lora_arm(ctx, len(layers), cfg, 0)
    try:
        state, curve, tm = F.fit(ctx, arm, v, 0, cfg)
        adapters.load_adapter_state(arm.params, state)
        adapters.merge_lora(arm.handles[0])
        _free(ctx.model)
        pred = A.predict(arm, calib, ctx.a.batch_size)
    finally:
        arm.close()
        ctx.release()
    acc = stats.accuracy([LABELS.index(r["label"]) for r in calib], pred)
    return {"tier": tier, "knobs": vars(knobs) | {}, "accuracy": acc, "n_calib": len(calib),
            "n_train": len(train), "n_heldout": len(heldout), "heldout_best": F.best_heldout(curve),
            "label_counts": {l: sum(r["label"] == l for r in calib) for l in LABELS},
            "pred_counts": dict(Counter(LABELS[p] for p in pred)), "mean_prompt_tokens": mean_tokens(ctx, calib),
            "step_seconds": tm["step_seconds"], "train_seconds": tm["train_seconds"], "steps": tm["steps"],
            "in_band": in_band(acc), "ts": now(), **({"hops": knobs.hops} if HARD else {})}


def write_tier_data(v, tier, smoke) -> dict:
    """Seed-2 splits on the frozen tier, written once under var/data/cap with a manifest."""
    import hashlib
    import json
    sizes = SMOKE_FINAL_SIZES if smoke else FINAL_SIZES
    splits, manifest = build_splits(v, FINAL_SEED, sizes, C.knobs_of(tier), smoke)
    manifest = dict(manifest, tier=tier, knobs=vars(C.knobs_of(tier)) | {}, files={})
    for name, recs in splits.items():
        p = CAP_DATA / f"{v}_s{FINAL_SEED}_{name}{sfx(smoke)}.jsonl"
        write_jsonl(recs, p)
        manifest["files"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    mp = CAP_DATA / f"{v}_s{FINAL_SEED}_manifest{sfx(smoke)}.json"
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def stage_b_calibrate(ctx, variants):
    ctx.tier = True
    for v in variants:
        path = OUT / f"calib_{v}{sfx(ctx.smoke)}.json"
        doc = load_doc(path, {"model": MODEL, "variant": v, "rungs": {}, "frozen": None})
        doc["frozen"] = None

        def run_rung(tier):
            ctx.status("b-calibrate", v, tier)
            print(f"== {v} calibrate {tier} {now()}", flush=True)
            r = calibrate_rung(ctx, v, tier)
            doc["rungs"][tier] = r
            write(path, doc)
            print(f"{v} {tier}: calibration accuracy {r['accuracy']:.3f}, labels {r['label_counts']}, "
                  f"{r['mean_prompt_tokens']:.0f} prompt tokens, {r['step_seconds']:.2f}s/step", flush=True)
            return r

        if HARD:
            doc["frozen"] = walk_hard(doc["rungs"], run_rung)
        for tier in () if HARD else C.TIERS:
            if tier not in doc["rungs"]:
                ctx.status("b-calibrate", v, tier)
                print(f"== {v} calibrate {tier} {now()}", flush=True)
                doc["rungs"][tier] = calibrate_rung(ctx, v, tier)
                write(path, doc)
            r = doc["rungs"][tier]
            print(f"{v} {tier}: calibration accuracy {r['accuracy']:.3f}, labels {r['label_counts']}, "
                  f"{r['mean_prompt_tokens']:.0f} prompt tokens, {r['step_seconds']:.2f}s/step", flush=True)
            if r["in_band"]:
                doc["frozen"] = tier
                break
        if doc["frozen"] is None and ctx.smoke:
            doc["frozen"], doc["smoke_forced"] = next(iter(C.TIERS)), True
        write(path, doc)
        if doc["frozen"] is None:
            print(f"{v}: no tier has healed-unmodified accuracy in [{CAL_BAND[0]:.0%}, {CAL_BAND[1]:.0%}]; "
                  f"nothing frozen, Study {'B2' if HARD else 'B'} stops here", flush=True)
            ctx.status("b-calibrate", v, "none frozen", done=True)
            continue
        man = write_tier_data(v, doc["frozen"], ctx.smoke)
        doc["data_manifest"] = man["files"]
        write(path, doc)
        print(f"{v}: froze {doc['frozen']}; data in {CAP_DATA}", flush=True)
        ctx.status("b-calibrate", v, f"frozen {doc['frozen']}", done=True)


# ---------------------------------------------------------------- Study B: frontier

def frozen_tier(ctx, v) -> str:
    p = OUT / f"calib_{v}{sfx(ctx.smoke)}.json"
    tier = read_json(p)["frozen"] if p.exists() else None
    if tier is None:
        sys.exit(f"{v}: no frozen tier (run b-calibrate first, or calibration found none)")
    return tier


def b_cfg(k) -> dict:
    steps = B_STEPS
    rank = A.LORA["1m"][2] if k == 0 else A.HEAL["1m"][2]
    return F.cfg_of(B_LR, steps, rank, A.QV)


def b_doc(ctx, v, tier) -> tuple[dict, Path]:
    path = OUT / f"b_{v}{sfx(ctx.smoke)}.json"
    return load_doc(path, {"model": MODEL, "variant": v, "tier": tier, "runs": {}}), path


def run_b(ctx, v, doc, path, k, groups, seed):
    """Heal (or re-heal after a fold) one point for one seed; selection scored per seed, canaries on seed 0."""
    groups = [tuple(g) for g in groups]
    ng = len(groups)
    key = F.run_key(k, ng, seed)
    if key in doc["runs"]:
        print(f"skip {v} {key}: already done", flush=True)
        return doc["runs"][key]
    cfg = b_cfg(k)
    ctx.status("b-frontier", v, key)
    print(f"== {v} b {key} {F.cfg_text(cfg)} {now()}", flush=True)
    sel = ctx.records(v, "selection")
    base_pred = ctx.baseline_pred(v)
    base_can = ctx.base_canaries() if seed == 0 else None
    alive = ctx.alive(v, k)
    arm, means = F.build_point(ctx, v, k, cfg, seed, groups)
    model = ctx.model
    try:
        state, curve, tm = F.fit(ctx, arm, v, seed, cfg)
        cp = F.point_ckpt(ctx, v, k, ng, seed, cfg)
        F.save_ckpt(cp, {"state": state, "variant": v, "k": k, "cfg": cfg, "seed": seed, "groups": groups,
                         "means": means})
        adapters.load_adapter_state(arm.params, state)
        adapters.merge_lora(arm.handles[0])
        _free(model)
        t1 = time.perf_counter()
        pred = A.predict(arm, sel, ctx.a.batch_size)
        sel_s = time.perf_counter() - t1
        can = A.canaries_for(ctx, arm, v, "heal", base_can) if seed == 0 else None
        row = {"k": k, "ng": ng, "seed": seed, "cfg": cfg, "groups": [list(g) for g in groups], "curve": curve,
               "train_seconds": tm["train_seconds"], "step_seconds": tm["step_seconds"], "steps": tm["steps"],
               "selection_seconds": sel_s, "checkpoint": str(cp), "ts": now(),
               "selection": A.summarize(sel, pred, base_pred, ctx.n_boot()), "canaries": can,
               **structure_row(ctx, v, k, groups)}
    finally:
        arm.close()
        ctx.release()
    doc["runs"][key] = row
    write(path, doc)
    if not ctx.smoke:
        append(SELECTION_LOG_B, {"ts": now(), "model": MODEL, "dtype": "fp32", "variant": v, "split": "selection",
                                 "seed": seed, "n": len(sel), "result": str(path), "tier": doc["tier"],
                                 "intervention": f"{PFX}k{k}-g{ng}-{F.tag(cfg)}-s{seed}",
                                 "config": {"k": k, "kv_groups": ng, **cfg}, "accuracy": row["selection"]["accuracy"]})
    curve_txt = ", ".join(f"{c['step']}: {c['heldout_accuracy']:.3f}" for c in curve)
    print(f"{v} {key}: selection {row['selection']['accuracy']:.3f} (heldout {curve_txt}; "
          f"{tm['step_seconds']:.2f}s/step, sel {sel_s:.0f}s)", flush=True)
    return row


def complete_points(doc):
    return [p for p in F.points(doc) if p["complete"]]


def within_points(doc):
    """(k = 0 point, [k > 0 points with ng = 0 within 5 points of it], largest-k point or None)."""
    pts = complete_points(doc)
    k0 = next((p for p in pts if p["k"] == 0 and p["ng"] == 0), None)
    if k0 is None:
        return None, [], None
    ok = [p for p in pts if p["k"] > 0 and p["ng"] == 0 and k0["mean"] - p["mean"] <= B_WITHIN + EPS]
    return k0, ok, (max(ok, key=lambda p: p["k"]) if ok else None)


def stage_b_frontier(ctx, variants, kv):
    ctx.tier, ctx.n_train = True, B_N_TRAIN
    for v in variants:
        tier = frozen_tier(ctx, v)
        doc, path = b_doc(ctx, v, tier)
        for k in (SMOKE_B_KS if ctx.smoke else B_KS):
            for seed in SEEDS:
                run_b(ctx, v, doc, path, k, [], seed)
        k0, ok, chosen = within_points(doc)
        print(f"{v}: k=0 mean {k0['mean']:.3f}; within 5 points: {[p['k'] for p in ok]}", flush=True)
        if kv:
            if chosen is None:
                print(f"{v}: no k > 0 within 5 points of k=0, no KV pruning", flush=True)
            else:
                ranked = F.rank_groups(ctx, v, ctx.alive(v, chosen["k"]))
                for ng in (SMOKE_B_KV if ctx.smoke else B_KV):
                    for seed in SEEDS:
                        run_b(ctx, v, doc, path, chosen["k"], ranked[:ng], seed)
        ctx.status("b-frontier", v, "complete", done=True)


def stage_b_test(ctx, variants):
    ctx.tier, ctx.n_train = True, B_N_TRAIN
    for v in variants:
        tier = frozen_tier(ctx, v)
        doc, path = b_doc(ctx, v, tier)
        k0, ok, chosen = within_points(doc)
        if k0 is None:
            print(f"{v}: k=0 frontier incomplete, run b-frontier first", flush=True)
            continue
        models = [(k0, "k0")]
        if chosen is not None:
            models.append((chosen, "chosen"))
            kvp = [p for p in complete_points(doc) if p["k"] == chosen["k"] and p["ng"] > 0
                   and k0["mean"] - p["mean"] <= B_WITHIN + EPS]
            if kvp:
                models.append((max(kvp, key=lambda p: p["ng"]), "smallest_kv"))
        names = {f"{PFX}k{p['k']}-g{p['ng']}-s0": (p, lab) for p, lab in models}
        if not ctx.smoke:
            for n in names:
                if ledger_has_b(v, n):
                    print(f"refusing: test ledger b already has {MODEL} {v} {n}", file=sys.stderr)
                    sys.exit(2)
        test = ctx.records(v, "test")
        base_pred = ctx.baseline_pred(v, "test")
        out = {"model": MODEL, "variant": v, "tier": tier, "split": "test", "n": len(test), "runs": {},
               "baseline_accuracy": stats.accuracy([LABELS.index(r["label"]) for r in test], base_pred),
               "verdict_capacity": C.verdict_b_capacity(k0["mean"]), "k0_selection_mean": k0["mean"]}
        if chosen is not None:
            out["verdict_cut"] = C.verdict_b_cut(k0["mean"], chosen["mean"])
        else:
            best = max((p for p in complete_points(doc) if p["k"] > 0 and p["ng"] == 0), key=lambda p: p["mean"],
                       default=None)
            out["verdict_cut"] = best and C.verdict_b_cut(k0["mean"], best["mean"])
            out["note"] = "no k > 0 within 5 points of k=0; verdict_cut uses the best k > 0 selection mean"
        for name, (p, label) in names.items():
            ctx.status("b-test", v, name)
            t0 = time.perf_counter()
            r0 = next(r for r in p["rows"] if r["seed"] == 0)
            saved = torch.load(r0["checkpoint"], map_location="cpu")
            groups = [tuple(g) for g in r0["groups"]]
            arm, _ = F.build_point(ctx, v, p["k"], p["cfg"], 0, groups, means=saved.get("means"))
            try:
                adapters.load_adapter_state(arm.params, saved["state"])
                adapters.merge_lora(arm.handles[0])
                pred = A.predict(arm, test, ctx.a.batch_size)
            finally:
                arm.close()
                ctx.release()
            row = A.summarize(test, pred, base_pred, ctx.n_boot())
            row.update(label=label, k=p["k"], ng=p["ng"], cfg=p["cfg"], selection_mean=p["mean"], flops=p["flops"],
                       kv_bytes_per_token=p["kv_bytes"], seconds=time.perf_counter() - t0)
            out["runs"][name] = row
            if not ctx.smoke:
                append(TEST_LEDGER_B, {"ts": now(), "model": MODEL, "dtype": "fp32", "variant": v, "split": "test",
                                       "seed": 0, "intervention": name, "n": len(test), "accuracy": row["accuracy"],
                                       "tier": tier, "config": {"k": p["k"], "kv_groups": p["ng"], **p["cfg"]},
                                       "result": str(OUT / f"b_test_{v}.json")})
            write(OUT / f"b_test_{v}{sfx(ctx.smoke)}.json", out)  # after each pass, so a crash keeps what the ledger has
            print(f"test {v} {name}: accuracy {row['accuracy']:.3f}", flush=True)
        print(f"{v}: capacity: {out['verdict_capacity']}; cut: {out['verdict_cut']}", flush=True)
        ctx.status("b-test", v, "complete", done=True)


# ---------------------------------------------------------------- table

def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def stage_table(smoke: bool):
    s = sfx(smoke)
    lines = [f"# Capability studies: tables{' (Study B2, hard task family)' if HARD else ''}", "",
             "Accuracy in percent. KL is T=2, T^2-scaled, per token.", ""]
    for v in () if HARD else VARIANTS:   # Study A has no hard-family results
        p = OUT / f"a_{v}{s}.json"
        lines += [f"## Study A: {v}", ""]
        if not p.exists():
            lines += ["no results yet", ""]
            continue
        d = read_json(p)
        lines += ["| run | held-out | dev KL | canaries | ppl | s/step |", "|---|---|---|---|---|---|"]
        for r in sorted(seed0_rows(d), key=lambda r: (r["ceiling"], r["arm"], r["lr"], r["lam"])):
            lines.append(f"| {spec_name(r)}{' (ceiling)' if r['ceiling'] else ''} | {pct(r['heldout'])} | "
                         f"{r['dev_kl']:.4f} | {pct(r['canaries']['accuracy'])} | {r['ppl']['ppl']:.2f} | "
                         f"{r['step_seconds']:.2f} |")
        pk = d.get("pick")
        if pk:
            lines += ["", f"Pick: {pk.get('name') or 'none (' + str(pk.get('note')) + ')'}"]
        if d.get("selection"):
            accs = [(k, x["selection"]["accuracy"], x["canaries"]["accuracy"]) for k, x in d["selection"].items()]
            lines += ["", "| seed | selection | canaries |", "|---|---|---|"]
            lines += [f"| {k} | {pct(a)} | {pct(c)} |" for k, a, c in accs]
        t = d.get("test")
        if t:
            lines += ["", f"Test {t['name']}: accuracy {pct(t['test']['accuracy'])} (unmodified "
                      f"{pct(t['baseline_accuracy'])}), canaries {pct(t['canaries']['accuracy'])}, "
                      f"ppl {t['ppl']['ppl']:.2f}. **H-A: {t['verdict']}**"]
        p2 = OUT / f"a2_{v}{s}.json"
        if p2.exists():
            d2 = read_json(p2)
            lines += ["", "A2 (report only):", "", "| point | selection | canaries |", "|---|---|---|"]
            lines += [f"| {k.replace('|', ' ')} | {pct(x['selection']['accuracy'])} | {pct(x['canaries']['accuracy'])} |"
                      for k, x in d2["selection"].items()]
        lines.append("")
    study = "B2" if HARD else "B"
    for v in (["rule"] if HARD else VARIANTS):
        pc = OUT / f"calib_{v}{s}.json"
        if not pc.exists():
            continue
        c = read_json(pc)
        lines += [f"## Study {study} calibration: {v}", "", f"Frozen tier: {c['frozen']}", "",
                  "| tier | calib acc | label counts | prompt tokens | s/step |", "|---|---|---|---|---|"]
        lines += [f"| {t} | {pct(r['accuracy'])} | {r['label_counts']} | {r['mean_prompt_tokens']:.0f} | "
                  f"{r['step_seconds']:.2f} |" for t, r in c["rungs"].items()]
        pb = OUT / f"b_{v}{s}.json"
        if pb.exists():
            d = read_json(pb)
            lines += ["", f"### Frontier ({d['tier']})", "",
                      "| k | KV groups pruned | selection mean (CI) | per seed | held-out at 300 / 1200 (seed 0) | canaries "
                      "| FLOPs rel | block params | KV bytes/token |", "|---|---|---|---|---|---|---|---|---|"]
            for q in F.points(d):
                cv = {c_["step"]: c_["heldout_accuracy"] for c_ in q["rows"][0]["curve"] or []}
                can = q["canaries"]
                lines.append(
                    f"| {q['k']} | {q['ng']} | {pct(q['mean'])} ({pct(q['lo'])}-{pct(q['hi'])})"
                    f"{'' if q['complete'] else ' partial'} | "
                    f"{', '.join(f's{r['seed']} {pct(r['selection']['accuracy'])}' for r in q['rows'])} | "
                    f"{pct(cv.get(300))} / {pct(cv.get(1200))} | {pct(can and can['accuracy'])} | "
                    f"{q['flops']['relative']:.3f} | {q['block_params']['total']:,} | {q['kv_bytes']:,} |")
        pt = OUT / f"b_test_{v}{s}.json"
        if pt.exists():
            t = read_json(pt)
            lines += ["", f"### Test (unmodified {pct(t['baseline_accuracy'])})", "",
                      "| model | test acc | selection mean | FLOPs rel |", "|---|---|---|---|"]
            lines += [f"| {n} | {pct(r['accuracy'])} | {pct(r['selection_mean'])} | {r['flops']['relative']:.3f} |"
                      for n, r in t["runs"].items()]
            lines += ["", f"**H-{study} capacity: {t['verdict_capacity']}. H-{study} cut: {t['verdict_cut']}.**"]
        lines.append("")
    dst = OUT / f"report_tables{s}.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n")
    print(f"-> {dst}")


# ---------------------------------------------------------------- driver

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--variant", choices=VARIANTS, help="default: both variants for a-*, rule for b-*")
    ap.add_argument("--kv", action="store_true", help="b-frontier: also run KV pruning at the largest k within 5 points")
    ap.add_argument("--stream", help="status file name (default: stage and variant)")
    ap.add_argument("--batch-size", type=int, default=8, help="training batch size")
    ap.add_argument("--eval-batch-size", type=int, default=8, help="scoring batch size")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if HARD and (not a.stage.startswith("b-") and a.stage != "table"):
        sys.exit(f"SURGERY_FAMILY=hard: stage {a.stage} is Study A and does not run on the hard task family")
    if HARD and a.variant not in (None, "rule"):
        sys.exit(f"SURGERY_FAMILY=hard: only variant rule exists, got {a.variant}")
    if a.stage == "table":
        return stage_table(a.smoke)
    # B2 takes this stream's GPU slot first (A2 gives way, contract). Held before any model loads, so it holds no memory.
    if a.stage == "a2" and a.variant and Path(f"var/.hold_a2_{a.variant}").exists():
        print(f"{a.variant}: a2 held until Study B2 ends", flush=True)
        while not (Path("var/results/cap/hard/DONE_hard").exists() or Path("var/results/cap/hard/FAILED_hard").exists()):
            time.sleep(60)
    variants = [a.variant] if a.variant else (["rule"] if a.stage.startswith("b-") else VARIANTS)
    a.stream = a.stream or f"{a.stage}_{a.variant or 'all'}"
    if a.stage.startswith("b-"):
        F.CKPT = CKPT  # F.point_ckpt / ckpt_dir read this module global
    ctx = CCtx(a, a.stage)
    if a.stage == "b-frontier":
        stage_b_frontier(ctx, variants, a.kv)
    elif a.stage == "b-calibrate":
        stage_b_calibrate(ctx, variants)
    else:
        globals()[f"stage_{a.stage.replace('-', '_')}"](ctx, variants)


if __name__ == "__main__":
    main()
