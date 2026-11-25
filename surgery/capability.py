"""Capability studies (docs/CONTRACT-capability.md): distillation on general text while training the task, the
general-data filter, harder task tiers, the pick rule and the verdict bars.

Loss per step = task_weight * CE(label logits) + lam * T^2 * KL(teacher || student) per token on one general batch.
The teacher is the unmodified model; the student is the cut model. Only `model.model.layers` weights train in the
full-weight arm (embeddings, final norm and the tied LM head stay frozen).
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import torch
from torch import nn

from surgery.adapters import heldout_accuracy
from surgery.eval import _free
from surgery.task.generator import FAMILY, LABELS, Knobs

# (n_rules, cond_size, n_distractor_files = n_distractor_tests, history_extra, conflict_rate)
# The contract's R4 has 8 rules, but `rule` draws distinct actions and there are only 6 labels, so the generator cannot
# produce more than 6 active rules (random.sample raises). R4 is therefore capped at len(LABELS) = 6 rules.
_TIER_TUPLES = {"R1": (3, 2, 3, 3, 0.4), "R2": (4, 3, 4, 4, 0.5), "R3": (6, 3, 5, 5, 0.6),
                "R4": (min(8, len(LABELS)), 4, 6, 6, 0.7)}
TIERS = {name: Knobs(n_rules=r, cond_size=c, n_distractor_files=d, n_distractor_tests=d, history_extra=h,
                     conflict_rate=cr) for name, (r, c, d, h, cr) in _TIER_TUPLES.items()}

if FAMILY == "hard":   # Study B2: the harder task family's rungs (surgery/task/hard.py)
    from surgery.task.hard import ALL_KNOBS, HTIERS
    TIERS = HTIERS


def knobs_of(name: str):
    """Knobs of a rung by name; under the hard family this also resolves the between rungs (H0, H1b, ...)."""
    return ALL_KNOBS[name] if FAMILY == "hard" else TIERS[name]


GATE = 0.99  # held-out task accuracy a candidate must reach


# ---------------------------------------------------------------- loss and data

def distill_kl(student_logits, teacher_logits, attention_mask, T: float = 2.0):
    """T^2 * mean over positions with mask 1 of KL(teacher || student) per token (same positions, no shift)."""
    s = torch.log_softmax(student_logits.float() / T, dim=-1)
    t = torch.log_softmax(teacher_logits.float() / T, dim=-1)
    kl = (t.exp() * (t - s)).sum(-1)
    m = attention_mask.to(kl.dtype)
    return T * T * (kl * m).sum() / m.sum().clamp(min=1.0)


def encode_general(tok, texts, device, max_len: int = 128):
    """Right-padded, truncated batch for the general loss (positions line up between student and teacher)."""
    side, tok.padding_side = tok.padding_side, "right"
    try:
        return tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    finally:
        tok.padding_side = side


@torch.no_grad()
def general_dev_kl(model, teacher, tok, rows, batch_size: int = 8, T: float = 2.0) -> float:
    """Token-weighted mean T^2 KL(teacher || model) over the rows' text."""
    total, count = 0.0, 0.0
    for i in range(0, len(rows), batch_size):
        enc = encode_general(tok, [r["text"] for r in rows[i:i + batch_size]], model.device)
        kl = distill_kl(model(**enc, use_cache=False).logits, teacher(**enc, use_cache=False).logits,
                       enc["attention_mask"], T)
        n = float(enc["attention_mask"].sum())
        total, count = total + float(kl) * n, count + n
        del enc, kl
        _free(model)
    return total / max(count, 1.0)


# ---------------------------------------------------------------- selection of the kept state

def rank_key(heldout: float, dev_kl: float, task_weight: float = 1.0) -> tuple:
    """Higher is better: runs at or above the gate beat runs below; at the gate the lowest dev KL wins, below it
    the highest held-out accuracy. With no task loss (ceiling control) only the dev KL counts."""
    if task_weight == 0:
        return (1, -dev_kl)
    return (1, -dev_kl) if heldout >= GATE else (0, heldout)


# ---------------------------------------------------------------- training

def full_trainable(model) -> list[tuple[str, nn.Parameter]]:
    """Freeze everything, then unfreeze the decoder blocks only. Returns the (name, parameter) pairs."""
    model.requires_grad_(False)
    out = []
    for name, p in model.model.layers.named_parameters():
        p.requires_grad_(True)
        out.append((f"model.layers.{name}", p))
    return out


def _snapshot(names: dict, trainable) -> dict:
    return {names[id(p)]: p.detach().cpu().clone() for p in trainable}


def train_combined(model, teacher, tok, task_train, task_heldout, gen_train, gen_dev, trainable: list, forward_fn,
                   steps: int, lr: float, lam: float, batch_size: int = 8, seed: int = 0, n_evals: int = 4,
                   task_weight: float = 1.0, T: float = 2.0):
    """AdamW (10% warm-up, linear decay) on task_weight * CE + lam * distill_kl, one task batch and one general
    batch per step. Returns (best trainable state {name: cpu tensor}, curve); the kept state follows `rank_key`."""
    teacher.eval().requires_grad_(False)
    names = {id(p): n for n, p in model.named_parameters()}
    missing = [p for p in trainable if id(p) not in names]
    if missing:
        raise ValueError("trainable parameters must belong to the model")
    g = torch.Generator().manual_seed(seed)
    gg = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.AdamW(trainable, lr=lr)
    warm = max(1, round(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    eval_at = sorted({max(1, round(steps * (i + 1) / n_evals)) for i in range(n_evals)})
    y_all = torch.tensor([LABELS.index(r["label"]) for r in task_train])
    order, gorder, best, curve, eval_s = [], [], None, [], 0.0
    ce_v = kl_v = 0.0
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        if len(order) < batch_size:
            order = torch.randperm(len(task_train), generator=g).tolist() + order
        if len(gorder) < batch_size:
            gorder = torch.randperm(len(gen_train), generator=gg).tolist() + gorder
        idx, order = order[:batch_size], order[batch_size:]
        gidx, gorder = gorder[:batch_size], gorder[batch_size:]
        opt.zero_grad(set_to_none=True)
        if task_weight:
            ce = task_weight * nn.functional.cross_entropy(forward_fn([task_train[i] for i in idx]),
                                                           y_all[idx].to(model.device))
            ce.backward()
            ce_v = float(ce.detach())
        if lam:
            enc = encode_general(tok, [gen_train[i]["text"] for i in gidx], model.device)
            with torch.no_grad():
                t_logits = teacher(**enc, use_cache=False).logits
            kl = distill_kl(model(**enc, use_cache=False).logits, t_logits, enc["attention_mask"], T)
            (lam * kl).backward()
            kl_v = float(kl.detach())
            del enc, t_logits, kl
        opt.step()
        sched.step()
        if step in eval_at:
            te = time.perf_counter()
            acc = heldout_accuracy(forward_fn, task_heldout, batch_size)
            dev = general_dev_kl(model, teacher, tok, gen_dev, batch_size, T)
            eval_s += time.perf_counter() - te
            curve.append({"step": step, "heldout_accuracy": acc, "general_dev_kl": dev, "task_loss": ce_v,
                          "train_kl": kl_v, "seconds": time.perf_counter() - t0, "eval_seconds": eval_s,
                          "kept": False})
            key = rank_key(acc, dev, task_weight)
            if best is None or key > best[0]:
                best = (key, _snapshot(names, trainable), len(curve) - 1)
            _free(model)
            print(f"  step {step}/{steps} ce {ce_v:.3f} kl {kl_v:.3f} heldout {acc:.3f} dev_kl {dev:.4f}", flush=True)
    curve[best[2]]["kept"] = True
    return best[1], curve


# ---------------------------------------------------------------- checkpoints

def save_trainables(path, state: dict, meta: dict | None = None) -> None:
    """bf16 CPU tensors keyed by parameter name (atomic write)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"state": {k: v.detach().cpu().to(torch.bfloat16) for k, v in state.items()}, "meta": meta or {}}, tmp)
    tmp.replace(path)


def load_trainables(model, src) -> int:
    """Copy a saved (or in-memory) {name: tensor} into the model's parameters by name, cast to the parameter dtype.
    Every saved name must exist in the model. Returns the number of tensors loaded."""
    state = src if isinstance(src, dict) else torch.load(src, map_location="cpu")["state"]
    params = dict(model.named_parameters())
    unknown = [k for k in state if k not in params]
    if unknown:
        raise KeyError(f"{len(unknown)} saved tensors not in the model, e.g. {unknown[0]}")
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v.to(device=params[k].device, dtype=params[k].dtype))
    return len(state)


# ---------------------------------------------------------------- general-data filter

def norm_text(text: str) -> str:
    return " ".join(text.lower().split())


def word_list(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def shingles(text: str, n: int = 12) -> set:
    w = word_list(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def filter_sets(canary_prompts: list[str], ppl_text: str):
    """(normalised canary prompts, canary word sets, ppl 12-word shingles) for `drop_reason`."""
    return ([norm_text(p) for p in canary_prompts], [set(word_list(p)) for p in canary_prompts],
            shingles(ppl_text))


def drop_reason(text, question, canary_prompts_norm, canary_word_sets, ppl_shingles):
    """Why a general-data row must go, or None. `question` (Q&A rows only) is compared with word-set Jaccard."""
    nt = norm_text(text)
    if any(p in nt for p in canary_prompts_norm if p):
        return "canary_prompt"
    if question is not None:
        q = set(word_list(question))
        if q and any(len(q & c) / len(q | c) >= 0.6 for c in canary_word_sets if c):
            return "canary_jaccard"
    if ppl_shingles and not ppl_shingles.isdisjoint(shingles(text)):
        return "ppl_overlap"
    return None


# ---------------------------------------------------------------- pick rule and verdicts

def pick_a(rows):
    """(pick, fallback). Ceiling rows never count and canaries/ppl are never read. Among rows with held-out >= 0.99
    the lowest general dev KL; if none, pick is None and fallback is the row with the highest held-out accuracy."""
    cand = [r for r in rows if not r.get("ceiling")]
    ok = [r for r in cand if r["heldout"] >= GATE]
    if ok:
        return min(ok, key=lambda r: r["dev_kl"]), None
    if not cand:
        return None, None
    return None, max(cand, key=lambda r: (r["heldout"], -r["dev_kl"]))


TASK_BAR, CANARY_SUPPORTED, CANARY_PARTIAL = 0.97, 0.58, 0.48
CAPACITY_BAR, CUT_BAR = 0.90, 0.05
_EPS = 1e-9


def verdict_a(task_test: float, canaries: float) -> str:
    if task_test < TASK_BAR - _EPS:
        return "not supported"
    if canaries >= CANARY_SUPPORTED - _EPS:
        return "supported"
    if canaries >= CANARY_PARTIAL - _EPS:
        return "partial"
    return "not supported"


def verdict_b_capacity(k0_mean: float) -> str:
    return "300 steps was the limit" if k0_mean >= CAPACITY_BAR - _EPS else "out of capacity"


def verdict_b_cut(k0_mean: float, k_mean: float) -> str:
    return "cutting stays free" if k0_mean - k_mean <= CUT_BAR + _EPS else "cutting costs capability"
