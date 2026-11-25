#!/usr/bin/env python
"""P0 per-task LoRA heals on the uncut base model, plain P0 prompt, loss = cross-entropy over the 4 option logits.

  .venv/bin/python scripts/p0_heal.py --model Qwen/Qwen2.5-1.5B --task fact_color --lr 1e-3 --seed 0 --out var/p0/heal
  .venv/bin/python scripts/p0_heal.py --model Qwen/Qwen2.5-1.5B --plan --out var/results/cap/p0heal   # the 12-heal pilot

LoRA as `heal.add_heal` (q and v of every block, rank 10, scale 2.0), 300 steps, batch 8, AdamW with 10% warm-up and linear
decay (`heal._schedule`). Data: the first 2,000 `lora` items train, the last 300 are dev, the P0 test split is the test.
Writes <out>/<task>_lr<lr>_s<seed>/{adapter.safetensors,heal.json}; --plan also heal_log.tsv and plan_done.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgery.atlas import p0_tasks as T  # noqa: E402

LRS = (1e-3, 3e-4)
TIE_LR = 3e-4
PLAN_TASKS = ("fact_continent", "fact_color", "arith_add", "hop_fact")
N_LORA, N_TRAIN = 2300, 2000
RANK, SCALE, BATCH, STEPS = 10, 2.0, 8, 300


def pick_device(device=None) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- data and forward ----

def task_data(tok, task: str, limit_train=None):
    """(train, dev, test) as (token id lists, answers). Plain prompt: raw text, no special tokens (as `Adapter.encode`)."""
    lora = T.generate_lora(task, seed=0, n=N_LORA)
    tr = lora[:N_TRAIN] if limit_train is None else lora[:limit_train]
    dev = lora[N_TRAIN:]
    _, te = T.split_items(T.generate(task, seed=0))
    enc = lambda xs: ([tok(x["prompt"], add_special_tokens=False).input_ids for x in xs], [x["answer"] for x in xs])
    return enc(tr), enc(dev), enc(te)


def option_logits(model, ids_list, opt_ids, device):
    """[B, 4] float32 logits over the options at the last real token (right padded; only those rows go through the head)."""
    n = max(len(x) for x in ids_list)
    x = torch.zeros(len(ids_list), n, dtype=torch.long)
    m = torch.zeros(len(ids_list), n, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        x[i, :len(ids)], m[i, :len(ids)] = torch.tensor(ids), 1
    x, m = x.to(device), m.to(device)
    h = model.model(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state
    last = h[torch.arange(len(ids_list), device=device), m.sum(1) - 1]
    w = model.get_output_embeddings().weight[torch.tensor(opt_ids, device=device)]
    return (last @ w.T).float()


@torch.no_grad()
def accuracy(model, ids, y, opt_ids, device, bs=32) -> float:
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    pred = np.zeros(len(ids), dtype=int)
    for s in range(0, len(order), bs):
        sel = order[s:s + bs]
        pred[sel] = option_logits(model, [ids[i] for i in sel], opt_ids, device).argmax(1).cpu().numpy()
    return float((pred == np.asarray(y)).mean())


# ---- adapter io ----

def save_adapter(handle, path) -> None:
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in handle.modules.state_dict().items() if "base_layer." not in k}, str(path))


def load_adapter(handle, path) -> None:
    from safetensors.torch import load_file
    st = load_file(str(path))
    missing, unexpected = handle.modules.load_state_dict(st, strict=False)
    if unexpected or any("base_layer." not in k for k in missing):
        raise ValueError(f"adapter does not match the model: missing={missing} unexpected={unexpected}")


def attach_lora(model, seed=0):
    from surgery.atlas.heal import add_heal
    return add_heal(model, len(model.model.layers), rank=RANK, seed=seed, scale=SCALE)


# ---- one heal ----

def heal_task(model, tok, task, lr, seed, out, device, steps=STEPS, limit_train=None, base_cache=None, log=print):
    """One heal on the clean `model` (LoRA attached for the run and removed again). Returns the heal.json dict."""
    from surgery.atlas.heal import _schedule
    opt_ids = T.check_options(tok, task)
    (x_tr, y_tr), (x_dev, y_dev), (x_te, y_te) = task_data(tok, task, limit_train)
    cache = base_cache if base_cache is not None else {}
    model.eval()
    if task not in cache:
        cache[task] = accuracy(model, x_te, y_te, opt_ids, device)
    handle = attach_lora(model, seed)
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
                lg = option_logits(model, [x_tr[i] for i in idx], opt_ids, device)
            loss = torch.nn.functional.cross_entropy(lg, y_all[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
            if step % 50 == 0:
                log(f"  {task} lr{lr:g} s{seed} step {step}/{steps} loss {np.mean(losses[-50:]):.3f}")
        if device == "cuda":
            torch.cuda.synchronize()
        train_s = time.perf_counter() - t0
        with amp:
            dev_acc = accuracy(model, x_dev, y_dev, opt_ids, device)
            test_acc = accuracy(model, x_te, y_te, opt_ids, device)
        wall = time.perf_counter() - t0
        d = Path(out) / f"{task}_lr{lr:g}_s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        save_adapter(handle, d / "adapter.safetensors")
    finally:
        handle.remove()
    res = {"task": task, "wall_seconds": wall, "train_seconds": train_s, "step_seconds": train_s / max(steps, 1),
           "steps": steps, "lr": lr, "seed": seed, "dev_acc": dev_acc, "test_acc": test_acc,
           "base_test_acc": cache[task], "device": device, "n_train": len(x_tr), "n_dev": len(x_dev),
           "n_test": len(x_te), "final_train_loss": float(np.mean(losses[-20:]))}
    (d / "heal.json").write_text(json.dumps(res, indent=1))
    return res


# ---- plan ----

def pick_lr(rows) -> float:
    """rows: dicts with lr and dev_acc (seed 0, every task). Highest mean dev accuracy over the tasks; a tie goes to 3e-4."""
    by = {}
    for r in rows:
        by.setdefault(r["lr"], []).append(r["dev_acc"])
    mean = {lr: float(np.mean(v)) for lr, v in by.items()}
    top = max(mean.values())
    tied = [lr for lr, m in mean.items() if m >= top - 1e-9]
    return TIE_LR if any(abs(lr - TIE_LR) < 1e-12 for lr in tied) else min(tied)


def run_plan(heal_one, out, tasks=PLAN_TASKS, lrs=LRS, log=print) -> dict:
    """heal_one(task, lr, seed) -> heal.json dict. 8 heals at seed 0, then the chosen lr at seed 1 (12 total)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t_plan, results = time.time(), []

    def one(task, lr, seed):
        a = _now()
        r = heal_one(task, lr, seed)
        with open(out / "heal_log.tsv", "a") as f:
            f.write("\t".join(map(str, [a, _now(), task, lr, seed, f"{r['wall_seconds']:.1f}", f"{r['dev_acc']:.4f}",
                                         f"{r['test_acc']:.4f}"])) + "\n")
        log(f"heal {task} lr{lr:g} s{seed}: wall {r['wall_seconds']:.1f}s dev {r['dev_acc']:.3f} test {r['test_acc']:.3f}")
        results.append(r)
        return r

    for lr in lrs:
        for task in tasks:
            one(task, lr, 0)
    chosen = pick_lr(results)
    log(f"chosen lr {chosen:g}")
    for task in tasks:
        one(task, chosen, 1)
    done = {"n_heals": len(results), "chosen_lr": chosen, "total_wall_seconds": time.time() - t_plan,
            "heal_seconds_sum": sum(r["wall_seconds"] for r in results),
            "heals_per_hour": 3600 * len(results) / (time.time() - t_plan), "finished": _now()}
    (out / "plan_done.json").write_text(json.dumps(done, indent=1))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    if not a.plan and not a.task:
        ap.error("--task is required without --plan")
    device = pick_device(a.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(device).eval()
    print(f"loaded {a.model} on {device} in {time.time() - t0:.1f}s", flush=True)
    cache = {}
    one = lambda task, lr, seed: heal_task(model, tok, task, lr, seed, a.out, device, a.steps, a.limit_train, cache,
                                           log=lambda *x: print(*x, flush=True))
    if a.plan:
        print(json.dumps(run_plan(one, a.out, log=lambda *x: print(*x, flush=True))))
    else:
        r = one(a.task, a.lr, a.seed)
        print(json.dumps(r))


if __name__ == "__main__":
    main()
