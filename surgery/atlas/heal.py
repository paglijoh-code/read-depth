"""M9 atlas Phase 1: cut, heal and score (docs/ATLAS-phase1.md). Library only, no paths.

A heal is a LoRA (r10, q and v of every surviving block) trained on one task's answer index over the option first-token
logits, after `structure.excise` has physically dropped the first k blocks of the shared order. The ceiling control is the
capability study's general-loss-only recipe (`train_ceiling`). The rest is pure bookkeeping: lr tie rule, k* bootstrap,
the pod-hours projection.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from surgery import adapters, capability, structure
from surgery.atlas import measure, metrics, spec
from surgery.eval import _free

LRS = (1e-3, 3e-4)
TIE_LR = 3e-4


# ---------------------------------------------------------------- cut and heal

def alive_blocks(order, k: int, n_layers: int) -> list[int]:
    """All blocks but the first k of `order`, in original order. Block 0 and the last block may never be cut."""
    order = list(order)
    if not 0 <= k <= len(order):
        raise ValueError(f"k={k} outside 0..{len(order)}")
    cut = order[:k]
    if len(set(cut)) != len(cut) or any(not 0 < i < n_layers - 1 for i in cut):
        raise ValueError("cut blocks must be distinct and exclude block 0 and the last block")
    gone = set(cut)
    return [i for i in range(n_layers) if i not in gone]


def cut_model(model, alive) -> None:
    """Excise in place (permanent): the model keeps only `alive`, renumbered."""
    structure.excise(model, alive)


def add_heal(model, n_layers: int, rank: int = 10, seed: int = 0, scale: float = 2.0):
    """LoRA on q_proj and v_proj of every block of the (already cut) model. Freezes the rest. The handle has `.modules`."""
    torch.manual_seed(seed)
    return adapters.add_lora(model, list(range(n_layers)), modules=("q_proj", "v_proj"), r=rank, scale=scale)


def option_forward(model, tok, task: str):
    """fwd(records) -> (B, n_options) float32 logits over the task's option first tokens, with grad (left padding)."""
    ids = torch.tensor(spec.option_token_ids(tok, task), device=model.device)

    def fwd(records):
        enc = measure._encode(tok, task, records, model.device)
        return model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1, :][:, ids].float()
    return fwd


def accuracy_and_preds(model, tok, task: str, records, batch_size: int = 16):
    """(accuracy, predicted option indices) with no grad."""
    if not records:
        return float("nan"), []
    logits = measure.option_logits(model, tok, task, records, batch_size)
    preds = logits.argmax(-1)
    answers = np.array([r["answer"] for r in records])
    return float((preds == answers).mean()), preds.tolist()


def answer_ce(model, tok, task: str, records, batch_size: int = 16) -> float:
    """Mean cross-entropy of the right option under the softmax over the task's options."""
    logits = torch.tensor(measure.option_logits(model, tok, task, records, batch_size))
    y = torch.tensor([r["answer"] for r in records])
    return float(nn.functional.cross_entropy(logits, y))


def _schedule(opt, steps: int):
    warm = max(1, round(0.1 * steps))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))


def _eval_points(steps: int, n_evals: int) -> set[int]:
    return {max(1, round(steps * (i + 1) / n_evals)) for i in range(n_evals)} if n_evals else set()


def train_heal(model, tok, task: str, train_recs, eval_recs, params: nn.Module, steps: int, lr: float,
               batch_size: int = 8, seed: int = 0, n_evals: int = 4, eval_batch_size: int = 16,
               record_steps: bool = False, log=print):
    """AdamW (10% warm-up, linear decay) on cross-entropy over the task's option logits, answer = record["answer"].
    `n_evals` evenly spaced evaluations on `eval_recs`; the best accuracy is kept, ties to the earlier step. With
    n_evals=0 the final state is returned. Returns (best_state, curve, timings)."""
    g = torch.Generator().manual_seed(seed)
    trainable = [p for p in params.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    sched = _schedule(opt, steps)
    eval_at = sorted(_eval_points(steps, n_evals))
    fwd = option_forward(model, tok, task)
    y_all = torch.tensor([r["answer"] for r in train_recs])
    order, best, curve, eval_s, step_times = [], None, [], 0.0, []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        ts = time.perf_counter()
        if len(order) < batch_size:
            order = torch.randperm(len(train_recs), generator=g).tolist() + order
        idx, order = order[:batch_size], order[batch_size:]
        loss = nn.functional.cross_entropy(fwd([train_recs[i] for i in idx]), y_all[idx].to(model.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        loss_v = float(loss.detach())
        step_times.append(time.perf_counter() - ts)
        if step in eval_at:
            te = time.perf_counter()
            acc, _ = accuracy_and_preds(model, tok, task, eval_recs, eval_batch_size)
            eval_s += time.perf_counter() - te
            curve.append({"step": step, "lrsel_accuracy": acc, "train_loss": loss_v,
                          "seconds": time.perf_counter() - t0, "eval_seconds": eval_s})
            if best is None or acc > best[0]:
                best = (acc, adapters.adapter_state(params), step)
            _free(model)
            log(f"  step {step}/{steps} loss {loss_v:.3f} lrsel {acc:.3f}")
    total = time.perf_counter() - t0
    if best is None:
        best = (float("nan"), adapters.adapter_state(params), steps)
    timings = {"train_seconds": total - eval_s, "eval_seconds": eval_s, "total_seconds": total,
               "step_seconds": (total - eval_s) / max(steps, 1)}
    if record_steps:
        timings["step_times"] = step_times
    return best[1], curve, {**timings, "best_lrsel": best[0], "best_step": best[2]}


# ---------------------------------------------------------------- generation

@torch.no_grad()
def generate_answers(model, tok, task: str, records, max_new_tokens: int = 8, batch_size: int = 16) -> list[str]:
    """Greedy continuation of the task prompt (thinking off), decoded without special tokens."""
    out = []
    for b in range(0, len(records), batch_size):
        enc = measure._encode(tok, task, records[b:b + batch_size], model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id)
        out.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        del enc, gen
        _free(model)
    return out


def exact_match(text: str, option: str) -> bool:
    """Whitespace stripped, one trailing full stop removed, then equal to the option."""
    t = text.strip()
    if t.endswith("."):
        t = t[:-1].rstrip()
    return t == option


# ---------------------------------------------------------------- ceiling control

def ceiling_step(model, teacher, enc, opt, sched=None, T: float = 2.0) -> float:
    """One general-loss-only step: T^2 KL(teacher || student) on a right-padded batch. Returns the loss."""
    with torch.no_grad():
        t_logits = teacher(**enc, use_cache=False).logits
    kl = capability.distill_kl(model(**enc, use_cache=False).logits, t_logits, enc["attention_mask"], T)
    opt.zero_grad(set_to_none=True)
    kl.backward()
    opt.step()
    if sched is not None:
        sched.step()
    return float(kl.detach())


def train_ceiling(model, teacher, tok, gen_train, gen_dev, steps: int, lr: float, batch_size: int = 8, seed: int = 0,
                  n_evals: int = 4, T: float = 2.0, log=print):
    """Decoder blocks trainable (embeddings, norm and head frozen), general loss only (lambda 1), AdamW with 10% warm-up
    and linear decay. `n_evals` evenly spaced evaluations by `general_dev_kl`; the lowest is kept (ties: earlier).
    Returns (best state {param name: cpu tensor}, curve, timings)."""
    teacher.eval().requires_grad_(False)
    trainable = [p for _, p in capability.full_trainable(model)]
    names = {id(p): n for n, p in model.named_parameters()}
    gg = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.AdamW(trainable, lr=lr)
    sched = _schedule(opt, steps)
    eval_at = sorted(_eval_points(steps, n_evals))
    gorder, best, curve, eval_s = [], None, [], 0.0
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        if len(gorder) < batch_size:
            gorder = torch.randperm(len(gen_train), generator=gg).tolist() + gorder
        gidx, gorder = gorder[:batch_size], gorder[batch_size:]
        enc = capability.encode_general(tok, [gen_train[i]["text"] for i in gidx], model.device)
        kl = ceiling_step(model, teacher, enc, opt, sched, T)
        del enc
        if step in eval_at:
            te = time.perf_counter()
            dev = capability.general_dev_kl(model, teacher, tok, gen_dev, batch_size, T)
            eval_s += time.perf_counter() - te
            curve.append({"step": step, "general_dev_kl": dev, "train_kl": kl,
                          "seconds": time.perf_counter() - t0, "eval_seconds": eval_s})
            if best is None or dev < best[0]:
                best = (dev, {names[id(p)]: p.detach().cpu().clone() for p in trainable}, step)
            _free(model)
            log(f"  step {step}/{steps} kl {kl:.3f} dev_kl {dev:.4f}")
    total = time.perf_counter() - t0
    if best is None:
        best = (float("nan"), {names[id(p)]: p.detach().cpu().clone() for p in trainable}, steps)
    return best[1], curve, {"train_seconds": total - eval_s, "eval_seconds": eval_s,
                            "step_seconds": (total - eval_s) / max(steps, 1),
                            "best_dev_kl": best[0], "best_step": best[2]}


# ---------------------------------------------------------------- choices 3 and 12

def pick_lr(rows) -> float:
    """rows: dicts with "lr" and "best_lrsel" (seed 0, one per lr). Highest best lrsel wins; a tie goes to 3e-4."""
    rows = list(rows)
    top = max(r["best_lrsel"] for r in rows)
    tied = [r["lr"] for r in rows if r["best_lrsel"] >= top - 1e-9]
    if any(math.isclose(lr, TIE_LR) for lr in tied):
        return TIE_LR
    return min(tied)


def _ranks(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Spearman rho with average ranks for ties; nan if either side is constant or there are under 3 points."""
    if len(x) < 3:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def kstar_by_task(acc: dict, grid, tol: float = 0.05) -> dict:
    """{task: k*} from {task: {k: [per-seed accuracy]}}: seed means against healed k=0 (`metrics.k_star`)."""
    out = {}
    for t, by_k in acc.items():
        if 0 in by_k:
            out[t] = metrics.k_star(by_k, float(np.mean(by_k[0])), grid, tol)
    return out


def bootstrap_rho(acc: dict, predictor: dict, grid, tol: float = 0.05, n_boot: int = 2000, seed: int = 9) -> dict:
    """Spearman rho between a per-task predictor and k*. acc = {task: {k: [per-seed accuracy]}}. The interval resamples
    the tasks with replacement and, for each drawn task, one seed at random per k (k=0 included, its draw is the
    baseline), then recomputes k* and rho. Interval = 2.5 and 97.5 percentiles of the finite draws."""
    grid = tuple(grid)
    tasks = sorted(t for t in acc if predictor.get(t) is not None and 0 in acc[t])
    ks = kstar_by_task({t: acc[t] for t in tasks}, grid, tol)
    tasks = [t for t in tasks if ks.get(t) is not None]
    out = {"n_tasks": len(tasks), "rho": None, "lo": None, "hi": None, "n_valid": 0,
           "kstar": {t: ks[t] for t in tasks}}
    if len(tasks) < 3:
        return out
    point = spearman([predictor[t] for t in tasks], [ks[t] for t in tasks])
    out["rho"] = None if math.isnan(point) else point
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pred, kst = [], []
        for t in (tasks[i] for i in rng.integers(len(tasks), size=len(tasks))):
            one = {k: v[int(rng.integers(len(v)))] for k, v in acc[t].items()}
            kk = metrics.k_star(one, one[0], grid, tol)
            if kk is not None:
                pred.append(predictor[t])
                kst.append(kk)
        r = spearman(pred, kst)
        if not math.isnan(r):
            draws.append(r)
    out["n_valid"] = len(draws)
    if draws:
        out["lo"], out["hi"] = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    return out


# ---------------------------------------------------------------- pod-hours projection (probe stage)

def project_hours(step_s: dict, tokens: dict, ceiling_step_s, n_layers: int = 28, ks=metrics.K_GRID, steps: int = 300,
                  batch: int = 8, n_lrsel: int = 200, n_sel: int = 500, n_transfer: int = 13 * 200,
                  n_discovery: int = 1000, n_lesion: int = 500, ceiling_steps: int = 3 * 1500,
                  speedup: float = 1.0) -> dict:
    """Pod hours under choice 3's main rule and the fallback. `step_s` is {task: seconds per training step at k=0} for the
    two probed tasks (route_work, arith); `tokens` is {task: mean prompt tokens} for all tasks. The other tasks interpolate
    linearly in tokens. A block-count factor (alive/n_layers) scales every k. Forward-only costs a third of a training
    record. `speedup` is the assumed gain from side-by-side shards (1.0 = none)."""
    (ta, sa), (tb, sb) = sorted(step_s.items(), key=lambda kv: tokens[kv[0]])
    lo, hi = min(sa, sb), max(sa, sb)

    def rec_cost(task):  # seconds per training record at k=0
        span = tokens[tb] - tokens[ta]
        s = sa if span == 0 else sa + (tokens[task] - tokens[ta]) * (sb - sa) / span
        return min(max(s, 0.5 * lo), 2.0 * hi) / batch

    cost = {t: rec_cost(t) for t in tokens}
    mean_c = sum(cost.values()) / len(cost)
    f = {k: (n_layers - k) / n_layers for k in ks}

    def run_s(task, k):
        c = cost[task]
        return f[k] * (steps * batch * c + (4 * n_lrsel + n_sel) * c / 3 + n_transfer * mean_c / 3)

    def rule_hours(runs_at):  # runs_at(k) = number of heals per (task, k)
        return sum(runs_at(k) * run_s(t, k) for t in tokens for k in ks) / 3600.0 / speedup

    cut = sum(f[k] * n_sel * cost[t] / 3 for t in tokens for k in ks) / 3600.0 / speedup
    sec = sum((n_discovery + (n_layers + 1) * n_lesion) * cost[t] / 3 for t in tokens) / 3600.0 / speedup
    ceil = None if ceiling_step_s is None else ceiling_steps * ceiling_step_s / 3600.0
    others = cut + sec + (ceil or 0.0)
    main = rule_hours(lambda k: 4)
    fb = rule_hours(lambda k: 4 if k == 0 else 3)
    return {"main": {"sweep": main, "cutonly": cut, "secondary": sec, "ceiling": ceil, "total": main + others},
            "fallback": {"sweep": fb, "cutonly": cut, "secondary": sec, "ceiling": ceil, "total": fb + others},
            "heals": {"main": 4 * len(tokens) * len(ks), "fallback": len(tokens) * (4 + 3 * (len(ks) - 1))},
            "speedup_assumed": speedup}


# ---------------------------------------------------------------- phase 2 stage B: whole-answer heals, permutations

EOT = "<|im_end|>"


def eot_id(tok) -> int:
    """End-of-turn token id, looked up through the tokenizer (never hard-coded)."""
    i = tok.convert_tokens_to_ids(EOT)
    if i is None or i == getattr(tok, "unk_token_id", None):
        raise ValueError(f"tokenizer has no {EOT}")
    return int(i)


def answer_text(task: str, rec: dict) -> str:
    """The option string, or str(meta["value"]) for arith_long (whose option index is only the first digit)."""
    if task == "arith_long":
        return str(rec["meta"]["value"])
    return spec.TASKS[task].options[rec["answer"]]


def whole_target(tok, task: str, rec: dict) -> tuple[list[int], list[int]]:
    """(prompt ids, target ids): the target is the answer text followed by the end-of-turn token."""
    prompt = tok.encode(spec.build_prompt(tok, task, rec["prompt"]), add_special_tokens=False)
    return prompt, tok.encode(answer_text(task, rec), add_special_tokens=False) + [eot_id(tok)]


def whole_batch(tok, task: str, records, device):
    """Teacher-forced batch, left padded so every target ends at the last position: (input_ids, mask, labels, n_target).
    Labels are -100 everywhere but the target tokens (prompt and padding are masked)."""
    pairs = [whole_target(tok, task, r) for r in records]
    n, tmax = max(len(p) + len(t) for p, t in pairs), max(len(t) for _, t in pairs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(pairs), n), pad, dtype=torch.long)
    mask = torch.zeros(len(pairs), n, dtype=torch.long)
    labels = torch.full((len(pairs), n), -100, dtype=torch.long)
    for i, (p, t) in enumerate(pairs):
        s = len(p) + len(t)
        ids[i, n - s:] = torch.tensor(p + t)
        mask[i, n - s:] = 1
        labels[i, n - len(t):] = torch.tensor(t)
    return ids.to(device), mask.to(device), labels.to(device), tmax


def whole_loss(model, enc) -> torch.Tensor:
    """Mean token cross-entropy over the target tokens only (logits kept for the target window, not the whole prompt)."""
    ids, mask, labels, tmax = enc
    logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=tmax + 1, use_cache=False).logits[:, :-1].float()
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, -tmax:].reshape(-1),
                                       ignore_index=-100)


@torch.no_grad()
def exact_match_scores(model, tok, task: str, records, max_new_tokens: int = 8, batch_size: int = 16):
    """(exact-match rate, texts): greedy, thinking off, each generation cut at the end-of-turn token, matched against
    `answer_text` by `exact_match`."""
    eot = eot_id(tok)
    texts = []
    was_training = model.training
    model.eval()
    for b in range(0, len(records), batch_size):
        enc = measure._encode(tok, task, records[b:b + batch_size], model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id, eos_token_id=eot)
        for row in gen[:, enc["input_ids"].shape[1]:].tolist():
            texts.extend(tok.batch_decode([row[:row.index(eot)] if eot in row else row], skip_special_tokens=True))
        del enc, gen
        _free(model)
    model.train(was_training)
    if not records:
        return float("nan"), texts
    return float(np.mean([exact_match(t, answer_text(task, r)) for t, r in zip(texts, records)])), texts


def _checkpointing(model, on: bool) -> None:
    """Gradient checkpointing on (train mode, which HF needs for it; the models here have no dropout) or off (eval mode)."""
    if on:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        if getattr(model, "is_gradient_checkpointing", False):
            model.gradient_checkpointing_disable()
        model.eval()


def train_whole(model, tok, task: str, train_recs, eval_recs, params: nn.Module, steps: int, lr: float,
                batch_size: int = 8, seed: int = 0, n_evals: int = 4, eval_batch_size: int = 16,
                max_new_tokens: int = 8, eval_n: int = 200, log=print, ckpt_tokens: int | None = None):
    """`train_heal`'s loop and schedule, but the loss is token cross-entropy on the whole answer plus end-of-turn. The
    `n_evals` evenly spaced evaluations are greedy exact match on the first `eval_n` of `eval_recs`; the best is kept,
    ties to the earlier one. Returns (best_state, curve, timings) with timings["best_lrsel"] the best exact match."""
    g = torch.Generator().manual_seed(seed)
    trainable = [p for p in params.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    sched = _schedule(opt, steps)
    eval_at = sorted(_eval_points(steps, n_evals))
    eval_recs = list(eval_recs)[:eval_n]
    order, best, curve, eval_s = [], None, [], 0.0
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        if len(order) < batch_size:
            order = torch.randperm(len(train_recs), generator=g).tolist() + order
        idx, order = order[:batch_size], order[batch_size:]
        enc = whole_batch(tok, task, [train_recs[i] for i in idx], model.device)
        if ckpt_tokens is not None:  # recompute activations on big batches only: same loss and gradients, less memory
            _checkpointing(model, enc[0].numel() > ckpt_tokens)
        loss = whole_loss(model, enc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        loss_v = float(loss.detach())
        if step in eval_at:
            te = time.perf_counter()
            em, _ = exact_match_scores(model, tok, task, eval_recs, max_new_tokens, eval_batch_size)
            eval_s += time.perf_counter() - te
            curve.append({"step": step, "lrsel_exact": em, "train_loss": loss_v,
                          "seconds": time.perf_counter() - t0, "eval_seconds": eval_s})
            if best is None or em > best[0]:
                best = (em, adapters.adapter_state(params), step)
            _free(model)
            log(f"  step {step}/{steps} loss {loss_v:.3f} lrsel exact {em:.3f}")
    if ckpt_tokens is not None:
        _checkpointing(model, False)
    total = time.perf_counter() - t0
    if best is None:
        best = (float("nan"), adapters.adapter_state(params), steps)
    return best[1], curve, {"train_seconds": total - eval_s, "eval_seconds": eval_s, "total_seconds": total,
                            "step_seconds": (total - eval_s) / max(steps, 1),
                            "best_lrsel": best[0], "best_step": best[2]}


def perm_within(values, groups, rng) -> np.ndarray:
    """`values` shuffled among the positions that share a group label (each group permuted on its own)."""
    values, groups = np.asarray(values), np.asarray(groups)
    out = values.copy()
    for gname in dict.fromkeys(groups.tolist()):
        pos = np.flatnonzero(groups == gname)
        out[pos] = values[pos][rng.permutation(len(pos))]
    return out
