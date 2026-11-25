"""Shared framework for the depth-atlas task families (spec docs/ATLAS-families.md).

A `Task` couples a name/family/type label/4 options with a `Spec` (templates, sampling, twins, oracle, near-miss rules).
`generate` is P0's generator plus an `aux` balancing variable; the rivals follow shared rule 8.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from surgery.atlas.p0_tasks import rival_bow, split_items  # noqa: F401  (re-exported)

TYPE_LABELS = ("surface form", "fact lookup", "binding/reference", "arithmetic", "code tracing", "rule application",
               "multi-hop composition")

REGISTRY: dict = {}


@dataclass
class Task:
    name: str
    family: str
    type_label: str
    options: tuple
    spec: object
    copy: bool = False
    notes: str = ""


def register(task: Task) -> Task:
    assert task.type_label in TYPE_LABELS, task.type_label
    assert task.name not in REGISTRY, task.name
    assert len(task.options) == 4 and len(set(task.options)) == 4, task.name
    REGISTRY[task.name] = task
    return task


def get_task(name: str) -> Task:
    return REGISTRY[name]


def _split_words(pool, split: str) -> list:
    """P0's held-out split: every third word (index % 3 == 2) is a test word."""
    return [w for i, w in enumerate(pool) if (i % 3 == 2) == (split == "test")]


def words_of(text: str) -> list:
    return re.findall(r"[A-Za-z0-9]+", text)


class Spec:
    """Base class. Subclasses set `templates` (>= 12 format strings starting with a fixed frame word), `near_miss`
    ({rule name: fn(slots) -> class}), optionally `class_pools` (list of 4 word lists) and `cls_key` (the slot that holds
    the class word), and implement `sample`, `ans_twin`, `surf_twin` and `solve`.

    `sample(rng, tpl, cls, split, aux)`: `aux` is an int 0..3 balanced within each class; copy tasks use it to fix the
    mention rank of the answer so the position rival stays at chance. Twin methods return the new slots (and class).
    """
    templates: list
    near_miss: dict = {}
    class_pools = None
    cls_key = None

    def pick_templates(self, rng, n_test_tpl):
        t = list(self.templates)
        rng.shuffle(t)
        return t[:-n_test_tpl], t[-n_test_tpl:]

    def render(self, tpl, s):
        return tpl.format(**s)

    def sample(self, rng, tpl, cls, split, aux=0):
        raise NotImplementedError

    def ans_twin(self, rng, tpl, s, cls, split):
        raise NotImplementedError

    def surf_twin(self, rng, tpl, s, cls, split):
        raise NotImplementedError

    def solve(self, prompt: str) -> int:
        raise NotImplementedError


class WordSpec(Spec):
    """Answer = class of one slot word (`cls_key`, drawn from held-out class pools); the surface slot `surf_key` is a
    word of another kind. Extra slots (`extra`: key -> pool) are filled but never decide the answer. `solve` finds the
    one class-pool word in the prompt. Template frame words are fixed text, never slots."""

    def __init__(self, templates, cls_key, class_pools, surf_key, surf_pool, extra=None, near_miss=None):
        self.templates, self.cls_key, self.class_pools = templates, cls_key, class_pools
        self.surf_key, self.surf_pool, self.extra = surf_key, surf_pool, extra or {}
        self.near_miss = near_miss or {}
        self._cls_of = {w: c for c, p in enumerate(class_pools) for w in p}
        assert len(self._cls_of) == sum(map(len, class_pools)), "a word is in two classes"
        assert not set(surf_pool) & set(self._cls_of), "surface pool overlaps class pools"
        assert len(templates) >= 12 and len(surf_pool) >= 6

    def _pool(self, c, split):
        return _split_words(self.class_pools[c], split)

    def sample(self, rng, tpl, cls, split, aux=0):
        s = {k: str(rng.choice(v)) for k, v in self.extra.items()}
        s[self.cls_key] = str(rng.choice(self._pool(cls, split)))
        s[self.surf_key] = str(rng.choice(self.surf_pool))
        return s

    def ans_twin(self, rng, tpl, s, cls, split):
        c2 = int(rng.choice([c for c in range(4) if c != cls]))
        s2 = dict(s)
        s2[self.cls_key] = str(rng.choice(self._pool(c2, split)))
        return s2, c2

    def surf_twin(self, rng, tpl, s, cls, split):
        s2 = dict(s)
        s2[self.surf_key] = str(rng.choice([w for w in self.surf_pool if w != s[self.surf_key]]))
        return s2

    def solve(self, prompt):
        hit = [self._cls_of[w] for w in words_of(prompt) if w in self._cls_of]
        assert len(hit) == 1, prompt
        return hit[0]


def generate(name: str, seed: int = 0, n_train: int = 400, n_test: int = 200) -> list:
    task = get_task(name)
    spec = task.spec
    rng = np.random.default_rng(seed * 1000003 + zlib.crc32(name.encode()))
    tr_t, te_t = spec.pick_templates(rng, 4)
    items = []
    for split, n, tpls in (("train", n_train, tr_t), ("test", n_test, te_t)):
        cls = np.tile(np.arange(4), n // 4 + 1)[:n]
        rng.shuffle(cls)
        aux = np.zeros(n, dtype=int)
        for c in range(4):  # aux balanced within each class
            idx = np.flatnonzero(cls == c)
            aux[idx] = rng.permutation(np.tile(np.arange(4), len(idx) // 4 + 1)[:len(idx)])
        tp = np.tile(np.arange(len(tpls)), n // len(tpls) + 1)[:n]
        rng.shuffle(tp)
        for i in range(n):
            tpl, c = tpls[int(tp[i])], int(cls[i])
            s = spec.sample(rng, tpl, c, split, int(aux[i]))
            s_a, c_a = spec.ans_twin(rng, tpl, s, c, split)
            s_s = spec.surf_twin(rng, tpl, s, c, split)
            items.append({
                "id": f"{name}-{split}-{i}", "prompt": spec.render(tpl, s), "answer": c,
                "template": tpl, "split": split, "slots": s,
                "ans_twin": {"prompt": spec.render(tpl, s_a), "answer": c_a},
                "surf_twin": {"prompt": spec.render(tpl, s_s), "answer": c},
            })
    return items


# ---- rivals (accuracies on test, trained on train) ----

def rival_position(items, task) -> float | None:
    """Copy tasks: max over mention ranks k of the accuracy of answering the k-th mentioned option word."""
    task = get_task(task) if isinstance(task, str) else task
    if not task.copy:
        return None
    _, te = split_items(items)
    words = [o.strip() for o in task.options]
    hits = np.zeros(len(words))
    for x in te:
        order = []
        for t in words_of(x["prompt"]):
            if t in words and words.index(t) not in order:
                order.append(words.index(t))
        for k, c in enumerate(order):
            hits[k] += c == x["answer"]
    return float(hits.max() / len(te))


def rival_near_miss(items, name) -> dict:
    """{rule name: test accuracy} for the task's hand-written shallow rules (slots -> class)."""
    _, te = split_items(items)
    rules = get_task(name).spec.near_miss
    return {r: float(np.mean([int(f(x["slots"])) == x["answer"] for x in te])) for r, f in rules.items()}


def rivals(items, name) -> dict:
    return {"bow": rival_bow(items), "position": rival_position(items, name),
            "near_miss": rival_near_miss(items, name)}


def check_options(tok, task, n_prompts: int = 6) -> list:
    """Token id of each option, in context on a few of the task's prompts. Raises ValueError if an option is not
    exactly one token after any of them, or if two options share an id."""
    from surgery.atlas.families.tokcheck import single_token_in_context
    task = get_task(task) if isinstance(task, str) else task
    items = generate(task.name, 0, 40, 20)
    step = max(1, len(items) // n_prompts)
    ids = []
    for x in items[::step][:n_prompts]:
        ids = []
        for o in task.options:
            if not single_token_in_context(tok, x["prompt"], o):
                raise ValueError(f"{task.name}: option {o!r} is not a single token after {x['prompt'][-30:]!r}")
            ids.append(list(tok(x["prompt"] + o, add_special_tokens=False).input_ids)[-1])
        if len(set(ids)) != 4:
            raise ValueError(f"{task.name}: options share a token")
    return ids
