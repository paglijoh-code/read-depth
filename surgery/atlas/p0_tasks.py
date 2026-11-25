"""P0 pilot tasks (depth atlas, proposal 4.2): 4 families x 2 tasks, 4 options, plain completion prompts.

Each item: {id, prompt, answer, template, split, ans_twin:{prompt,answer}, surf_twin:{prompt,answer}, slots}.
Answer twin: exactly one content word (slot) changes and the answer changes. Surface twin: one word of the same kind
at the same position changes and the answer does not. Neither touches the final word, so the last token is shared.
Train/test are split by template; for word->class tasks the class-defining words are also split (no word leakage).
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

import numpy as np

CONT = (" Europe", " Asia", " Africa", " America")
NAMES = "Anna Peter Sarah David James Daniel Henry Paul Mark Frank Simon Ben Rose Tom Jack Mary John Carl".split()
EXW = "table chair river stone door lamp book wall road cloud house paper window glass bridge".split()
SENT = "Today Earlier Recently Meanwhile Later Sometimes".split()  # single token at the start of a text too
FILL2 = "Compute Solve Add Find Calculate Work Simple Quick Basic".split()
COUNTRIES = [
    "France Spain Italy Germany Poland Norway Sweden Greece Portugal Ireland Austria Belgium Finland Denmark "
    "Hungary Romania Serbia Croatia Ukraine Switzerland Iceland Bulgaria".split(),
    "China Japan India Korea Thailand Vietnam Nepal Iran Iraq Turkey Indonesia Malaysia Pakistan Cambodia Israel "
    "Syria Jordan Bangladesh Philippines Singapore".split(),
    "Egypt Kenya Nigeria Ghana Morocco Ethiopia Tanzania Uganda Sudan Libya Zimbabwe Niger Somalia".split(),
    "Canada Mexico Brazil Peru Chile Cuba Argentina Colombia Venezuela Ecuador Panama Guatemala Jamaica Haiti".split(),
]
CITIES = ("Paris Tokyo Cairo Oslo Rome Delhi Dallas Perth Milan Dublin Vienna Berlin Madrid Seoul Boston Denver "
          "Austin Geneva Prague Athens Kyoto").split()
COLOR_OPTS = (" red", " yellow", " green", " white")
COLOR_OBJ = [
    "tomato cherry rose brick blood fire beet rust lava".split(),
    "banana lemon corn butter honey mustard egg straw taxi bee".split(),
    "grass frog lettuce leaf spinach lime moss olive parsley mint celery basil".split(),
    "snow milk salt cotton sugar cloud paper chalk rice flour ice bone tooth ghost".split(),
]
FIRST_WORDS = [
    "marble mirror market magnet mouse mountain monkey".split(),
    "sugar sister silver sock salt snow spoon sand soap".split(),
    "tiger tomato tooth tunnel ticket tower tent table turtle".split(),
    "pencil pepper plate pillow people planet pocket".split(),
]


def _split_words(pool: list[str], split: str) -> list[str]:
    return [w for i, w in enumerate(pool) if (i % 3 == 2) == (split == "test")]


class _Base:
    def pick_templates(self, rng, n_test_tpl):
        t = list(self.templates)
        rng.shuffle(t)
        return t[:-n_test_tpl], t[-n_test_tpl:]

    def render(self, tpl, s):
        return tpl.format(**s)


class WordTask(_Base):
    """Answer = class of one slot word (cls_key); surface twin changes surf_key (a word of another kind)."""

    def __init__(self, templates, cls_key, class_pools, surf_key, surf_pool, extra=None, split_words=True):
        self.templates, self.cls_key, self.class_pools = templates, cls_key, class_pools
        self.surf_key, self.surf_pool, self.extra, self.split_words = surf_key, surf_pool, extra or {}, split_words

    def _pool(self, c, split):
        p = self.class_pools[c]
        return _split_words(p, split) if self.split_words else p

    def sample(self, rng, tpl, cls, split):
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


def _copy_templates():
    pairs = [("Word", "Repeat"), ("Input", "Output"), ("Text", "Copy"), ("Item", "Answer"),
             ("Source", "Result"), ("Query", "Echo"), ("Say", "Echo"), ("Word", "Copy")]
    demos = [("sun", "cat"), ("tree", "moon"), ("fish", "bird"), ("hand", "star")]
    out = []
    for i, (a, b) in enumerate(pairs):
        for d in (demos[i % 4][0], demos[i % 4][1]):
            out.append(f"{a}: {d}. {b}: {d}. Other: {{ex}}. {a}: {{tgt}}. {b}:")
    return out


class ArithAdd(_Base):
    nums = range(1, 10)

    def __init__(self):
        pre = ["Q:", "Math:", "Exercise", "Problem:", "Test:"]
        self.templates = [f"{p} {{f}}{c}{{a}}+{{b}}=" for p in pre for c in (" ", ": ", ". ")]
        self.lo = 5

    def value(self, s):
        return s["a"] + s["b"] - self.lo

    def triples(self):
        return [(a, b) for a in self.nums for b in self.nums if 0 <= a + b - self.lo < 4]

    def _slots(self, tup):
        return {"a": tup[0], "b": tup[1]}

    def _tup(self, s):
        return (s["a"], s["b"])

    def sample(self, rng, tpl, cls, split):
        c = [t for t in self.triples() if self.value(self._slots(t)) == cls]
        s = self._slots(c[rng.integers(len(c))])
        s["f"] = str(rng.choice(FILL2))
        return s

    def ans_twin(self, rng, tpl, s, cls, split):
        keys = list(self._slots(self.triples()[0]))
        opts = []
        for i, k in enumerate(keys):
            for v in self.nums:
                t = list(self._tup(s))
                t[i] = v
                s2 = self._slots(t)
                if v != s[k] and 0 <= self.value(s2) < 4 and self.value(s2) != cls:
                    opts.append(s2)
        s2 = opts[rng.integers(len(opts))]
        s2["f"] = s["f"]
        return s2, self.value(s2)

    def surf_twin(self, rng, tpl, s, cls, split):
        s2 = dict(s)
        s2["f"] = str(rng.choice([w for w in FILL2 if w != s["f"]]))
        return s2


class ArithTwo(ArithAdd):
    def __init__(self):
        super().__init__()
        self.templates = [t.replace("{b}=", "{b}-{c}=") for t in self.templates]
        self.lo = 3

    def value(self, s):
        return s["a"] + s["b"] - s["c"] - self.lo

    def triples(self):
        return [(a, b, c) for a in self.nums for b in self.nums for c in self.nums
                if 0 <= a + b - c - self.lo < 4]

    def _slots(self, tup):
        return {"a": tup[0], "b": tup[1], "c": tup[2]}

    def _tup(self, s):
        return (s["a"], s["b"], s["c"])


class HopCtx(_Base):
    verbs = ["lives in", "stays in", "works in", "is based in"]
    rels = ["is in", "lies in", "is part of", "belongs to"]

    def __init__(self):
        self.templates = [f"{v}|{r}" for v in self.verbs for r in self.rels]

    def render(self, tpl, s):
        v, r = tpl.split("|")
        ppl, cit, order, cont = s["ppl"], s["cit"], s["order"], s["cont"]
        stmt = [f"{p} {v} {c}." for p, c in zip(ppl, cit)]
        rel = [f"{cit[j]} {r} {CONT[cont[j]].strip()}." for j in order]
        return f"{s['p']}, " + " ".join(stmt + rel) + f" {s['q']} lives on the continent of"

    def sample(self, rng, tpl, cls, split):
        ppl = [str(x) for x in rng.choice(NAMES, 4, replace=False)]
        cit = [str(x) for x in rng.choice(CITIES, 4, replace=False)]
        cont = [int(x) for x in rng.permutation(4)]
        qi = int(rng.integers(4))
        j = cont.index(cls)
        cont[qi], cont[j] = cont[j], cont[qi]
        return {"ppl": ppl, "cit": cit, "cont": cont, "order": [int(x) for x in rng.permutation(4)],
                "q": ppl[qi], "p": str(rng.choice(SENT))}

    def ans_twin(self, rng, tpl, s, cls, split):
        qi = s["ppl"].index(s["q"])
        j = int(rng.choice([k for k in range(4) if k != qi]))
        s2 = dict(s)
        s2["q"] = s["ppl"][j]
        return s2, s["cont"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        s2 = dict(s)
        s2["p"] = str(rng.choice([w for w in SENT if w != s["p"]]))
        return s2


def _fact_tpls():
    verbs = ["visited", "toured", "explored", "reached", "travelled to", "flew to", "moved to"]
    ends = [", which is in", ", a country in", ", a nation located in"]
    return [f"{{n}} {v} {{c}}{e}" for v in verbs for e in ends]


def _color_tpls():
    verbs = ["says", "thinks", "knows", "reports", "noted", "wrote that"]
    ends = ["the colour of the {o} is", "the usual colour of the {o} is", "the {o} has a typical colour, which is"]
    return [f"{{n}} {v} {e}" for v in verbs for e in ends]


def _first_tpls():
    a = ["{n} wrote the word {w} on the board.", "{n} said the word {w} out loud.",
         "{n} spelled the word {w} slowly.", "{n} found the word {w} in a book."]
    b = ["The word begins with the letter", "Its first letter is", "The first letter of the word is",
         "The initial letter is"]
    return [f"{x} {y}" for x in a for y in b]


def _hopfact_tpls():
    a = ["{n} is from {c}.", "{n} was born in {c}.", "{n} comes from {c}.", "{n} grew up in {c}."]
    b = ["{n} lives on the continent of", "{n} now lives on the continent of",
         "{n} works on the continent of", "{n} currently lives on the continent of"]
    return [f"{{p}}, {x} {y}" for x in a for y in b]


@dataclass
class Task:
    name: str
    family: str
    type_label: str
    options: tuple
    spec: object
    copy: bool = False
    notes: str = ""


_SPECS = {
    "surf_copy": Task("surf_copy", "surface", "surface form", (" red", " blue", " green", " pink"),
                      WordTask(_copy_templates(), "tgt", [["red"], ["blue"], ["green"], ["pink"]], "ex", EXW,
                               split_words=False), copy=True),
    "surf_first": Task("surf_first", "surface", "surface form", (" m", " s", " t", " p"),
                       WordTask(_first_tpls(), "w", FIRST_WORDS, "n", NAMES)),
    "fact_continent": Task("fact_continent", "fact", "fact lookup", CONT,
                           WordTask(_fact_tpls(), "c", COUNTRIES, "n", NAMES)),
    "fact_color": Task("fact_color", "fact", "fact lookup", COLOR_OPTS,
                       WordTask(_color_tpls(), "o", COLOR_OBJ, "n", NAMES)),
    "arith_add": Task("arith_add", "arithmetic", "arithmetic", ("5", "6", "7", "8"), ArithAdd()),
    "arith_two": Task("arith_two", "arithmetic", "arithmetic", ("3", "4", "5", "6"), ArithTwo()),
    "hop_fact": Task("hop_fact", "multihop", "multi-hop composition", CONT,
                     WordTask(_hopfact_tpls(), "c", COUNTRIES, "p", SENT, extra={"n": NAMES})),
    "hop_ctx": Task("hop_ctx", "multihop", "multi-hop composition", CONT, HopCtx()),
}
TASK_NAMES = list(_SPECS)


def get_task(name: str) -> Task:
    return _SPECS[name]


def check_options(tok, task) -> list[int]:
    """Token id of each option; raises ValueError if an option is not exactly one token."""
    task = get_task(task) if isinstance(task, str) else task
    ids = []
    for o in task.options:
        t = tok(o, add_special_tokens=False).input_ids
        if len(t) != 1:
            raise ValueError(f"{task.name}: option {o!r} is {len(t)} tokens")
        ids.append(t[0])
    if len(set(ids)) != 4:
        raise ValueError(f"{task.name}: options share a token")
    return ids


def generate(name: str, seed: int = 0, n_train: int = 400, n_test: int = 200) -> list[dict]:
    task = get_task(name)
    spec = task.spec
    rng = np.random.default_rng(seed * 1000003 + zlib.crc32(name.encode()))
    tr_t, te_t = spec.pick_templates(rng, 4)
    items = []
    for split, n, tpls in (("train", n_train, tr_t), ("test", n_test, te_t)):
        cls = np.tile(np.arange(4), n // 4 + 1)[:n]
        rng.shuffle(cls)
        tp = np.tile(np.arange(len(tpls)), n // len(tpls) + 1)[:n]
        rng.shuffle(tp)
        for i in range(n):
            tpl, c = tpls[int(tp[i])], int(cls[i])
            s = spec.sample(rng, tpl, c, split)
            s_a, c_a = spec.ans_twin(rng, tpl, s, c, split)
            s_s = spec.surf_twin(rng, tpl, s, c, split)
            items.append({
                "id": f"{name}-{split}-{i}", "prompt": spec.render(tpl, s), "answer": c,
                "template": tpl, "split": split, "slots": s,
                "ans_twin": {"prompt": spec.render(tpl, s_a), "answer": c_a},
                "surf_twin": {"prompt": spec.render(tpl, s_s), "answer": c},
            })
    return items


def split_items(items):
    return [x for x in items if x["split"] == "train"], [x for x in items if x["split"] == "test"]


# ---- rivals (accuracies on test, trained on train) ----

def rival_bow(items) -> float:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    tr, te = split_items(items)
    cv = CountVectorizer(token_pattern=r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", lowercase=False)
    X = cv.fit_transform([x["prompt"] for x in tr])
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X, [x["answer"] for x in tr])
    return float(np.mean(m.predict(cv.transform([x["prompt"] for x in te])) == [x["answer"] for x in te]))


def rival_position(items, task) -> float | None:
    """Copy tasks: predict the class of the last option word in the prompt."""
    task = get_task(task) if isinstance(task, str) else task
    if not task.copy:
        return None
    _, te = split_items(items)
    words = [o.strip() for o in task.options]
    hit = 0
    for x in te:
        toks = re.findall(r"[A-Za-z]+", x["prompt"])
        pred = 0
        for t in reversed(toks):
            if t in words:
                pred = words.index(t)
                break
        hit += pred == x["answer"]
    return hit / len(te)


def _near_miss(name, x):
    s = x["slots"]
    if name == "surf_copy":
        return 0  # constant guess
    if name == "surf_first":
        return len(s["w"]) % 4
    if name in ("fact_continent", "hop_fact"):
        return len(s["c"]) % 4
    if name == "fact_color":
        return len(s["o"]) % 4
    if name == "arith_add":
        return s["a"] % 4
    if name == "arith_two":
        return int(min(max(s["a"] + s["b"] - 3, 0), 3))  # ignores the subtrahend
    if name == "hop_ctx":
        return s["cont"][s["order"][0]]  # first continent mentioned
    raise KeyError(name)


def rival_near_miss(items, name) -> float:
    """One hand-written near-miss rule per task (descriptive)."""
    _, te = split_items(items)
    return float(np.mean([_near_miss(name, x) == x["answer"] for x in te]))


def rivals(items, name) -> dict:
    return {"bow": rival_bow(items), "position": rival_position(items, name), "near_miss": rival_near_miss(items, name)}


# ---- lora split: per-task heal data (train templates and train-side class words only) ----

def generate_lora(name: str, seed: int = 0, n: int = 2300) -> list[dict]:
    """Items for LoRA heals. Same train templates as `generate(name, seed)` (the test templates stay unseen), train-side
    class words, its own rng stream, ids `<name>-lora-<i>` (never a probe id), split "lora". Class-balanced and shuffled.
    Duplicate prompts are possible (small word pools); see `unique_prompts`."""
    spec = get_task(name).spec
    tr_t, _ = spec.pick_templates(np.random.default_rng(seed * 1000003 + zlib.crc32(name.encode())), 4)
    rng = np.random.default_rng([seed, zlib.crc32(name.encode()), 77])
    cls = np.tile(np.arange(4), n // 4 + 1)[:n]
    rng.shuffle(cls)
    tp = np.tile(np.arange(len(tr_t)), n // len(tr_t) + 1)[:n]
    rng.shuffle(tp)
    items = []
    for i in range(n):
        tpl, c = tr_t[int(tp[i])], int(cls[i])
        s = spec.sample(rng, tpl, c, "train")
        items.append({"id": f"{name}-lora-{i}", "prompt": spec.render(tpl, s), "answer": c, "template": tpl,
                      "split": "lora", "slots": s})
    return items


def unique_prompts(items) -> int:
    return len({x["prompt"] for x in items})
