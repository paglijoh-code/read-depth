"""Batch 1 (families 1-5): surf_initial, fact_continent, bind_owner, arith_addsub, code_alias (two tasks each).

Pools below are literal lists, vetted with `tokcheck.vet_words` against all 17 tokenizers (7 distinct vocabularies):
every word is one token with its leading space everywhere. Class pools are ordered so that P0's split (every third word is
a test word) puts a balanced sample of word lengths and initials in the test half (the near-miss rules stay at chance).
"""
from __future__ import annotations

import re

import numpy as np

from .core import Spec, Task, WordSpec, register, words_of

NAMES = ("Anna Peter Sarah David James Daniel Henry Paul Mark Frank Simon Ben Rose Tom Jack Mary John Carl Dan Emma "
         "Lucy Kate Alice Bob Fred George Helen Jane Kevin Laura").split()
LETTERS = "a b c d e f g h k m n p q r s t u v w x y z".split()


def _check_templates(templates, forbidden, min_n=12):
    """Template text (outside {slots}) must not contain a class/object/name word, or `solve` would be fooled."""
    assert len(templates) >= min_n, len(templates)
    for t in templates:
        bad = set(words_of(re.sub(r"\{[a-z0-9]+\}", " ", t))) & set(forbidden)
        assert not bad, (t, bad)


# ---------------------------------------------------------------- 1. surf_initial
_SI_A = [[w for w in "music moon mountain mother mouth metal meat mirror match mouse mask map".split()],
         "ship sand storm seed stone silver street sun sky salt snow soap".split(),
         "tip tile tub tunnel tea tent thread tool ticket tail tree town".split(),
         "park port pilot people planet piano plant plate price pot pocket pen".split()]
_SI_B = ["boot bell bus box bread ball basket boat brick bridge bank bed".split(),
         "cow comb car cup clock crown chair cave cap cart coin cook".split(),
         "dock date drink doll dust doctor door dog drum dinner dozen dance".split(),
         "film food fly flag flower father field forest face fork finger fruit".split()]


def _si_templates(a, b):
    return [f"{x} {y}" for x in a for y in b]


_SI_TPL_A = _si_templates(
    ["Then {n} wrote the word {w} on the board.", "Later {n} said the word {w} out loud.",
     "Soon {n} spelled the word {w} slowly.", "Today {n} found the word {w} in a text."],
    ["The word begins with the letter", "Its first letter is", "The first letter of the word is",
     "The initial letter is"])
_SI_TPL_B = _si_templates(
    ["Next {n} copied the word {w} onto a card.", "Once {n} heard the word {w} on the radio.",
     "Now {n} read the word {w} twice.", "Here {n} typed the word {w} into a file."],
    ["It starts with the letter", "The word starts with the letter", "Its opening letter is",
     "The first letter is"])


def _si_near(options):
    first = [o.strip() for o in options]
    return {"len_mod4": lambda s: len(s["w"]) % 4,
            "name_initial": lambda s: first.index(s["n"][0].lower()) if s["n"][0].lower() in first else 0}


for _s, _tpl, _pools, _opts in (("a", _SI_TPL_A, _SI_A, (" m", " s", " t", " p")),
                                ("b", _SI_TPL_B, _SI_B, (" b", " c", " d", " f"))):
    _check_templates(_tpl, [w for p in _pools for w in p] + NAMES)
    register(Task(f"surf_initial_{_s}", "surf_initial", "surface form", _opts,
                  WordSpec(_tpl, "w", _pools, "n", NAMES, near_miss=_si_near(_opts))))

# ---------------------------------------------------------------- 2. fact_continent
# TinyLlama's vocabulary has no single-token African country or Asian city, so the sketch's Africa class and the
# city pools cannot be built. a: place (country, or US state for North America) -> continent, with North and South America
# as separate classes ("is in North America"); b: US place (state or city) -> region of the country.
_FC_A = ["Russia Germany Sweden Spain Ukraine Portugal Switzerland Finland Greece Norway Ireland Austria France "
         "Netherlands Italy".split(),
         "Korea Japan Jordan Turkey Philippines Singapore Israel Vietnam Pakistan Iran China Taiwan India".split(),
         "Texas Alabama Mexico Colorado Cuba Illinois Maine Ohio California Kansas Michigan Florida Arizona Canada "
         "Oregon".split(),
         "Chile Peru Brazil Argentina Colombia Venezuela".split()]
_FC_B = ["Indiana Missouri Detroit Ohio Illinois Iowa Minnesota Cleveland Wisconsin Chicago Michigan Kansas".split(),
         "Maine Connecticut Boston Jersey Pennsylvania Philadelphia Massachusetts".split(),
         "Virginia Austin Miami Mississippi Tennessee Dallas Alabama Oklahoma Louisiana Houston Florida Kentucky".split(),
         "California Phoenix Seattle Arizona Oregon Colorado".split()]


def _fc_templates(ends):
    starts = ["Last year", "Then", "Later", "Once"]
    verbs = ["visited", "toured", "explored", "reached"]
    return [f"{s} {{n}} {v} {{c}}{e}" for s in starts for v, e in zip(verbs[1:] + verbs[:1], ends)] + \
           [f"{s} {{n}} {v} {{c}}{e}" for s in starts for v, e in zip(verbs, ends)]


_FC_TPL_A = sorted(set(_fc_templates([", which is in", ", a place in", ", which lies in", ", a land in"])))
_FC_TPL_B = sorted(set(_fc_templates([", which is in the", ", a place in the", ", which lies in the", ", a spot in the"])))
_FC_NEAR = {"len_mod4": lambda s: len(s["c"]) % 4, "initial_bucket": lambda s: (ord(s["c"][0]) - 65) % 4}
for _s, _tpl, _pools, _opts in (("a", _FC_TPL_A, _FC_A, (" Europe", " Asia", " North", " South")),
                                ("b", _FC_TPL_B, _FC_B, (" north", " east", " south", " west"))):
    _check_templates(_tpl, [w for p in _pools for w in p] + NAMES)
    register(Task(f"fact_continent_{_s}", "fact_continent", "fact lookup", _opts,
                  WordSpec(_tpl, "c", _pools, "n", NAMES, near_miss=_FC_NEAR),
                  notes="a: place -> continent (North/South America split); b: US place -> region (see module note)"))


# ---------------------------------------------------------------- 3. bind_owner
class BindSpec(Spec):
    """k names each own one of the 4 option words (a permutation); ask for one name's object. Copy task: the answer
    word is mentioned once, at the queried name's list position (`aux`, balanced), so the position rival is chance."""

    def __init__(self, templates, options, names, near_miss):
        self.templates, self.opts, self.names, self.near_miss = templates, [o.strip() for o in options], names, near_miss
        _check_templates(templates, self.opts + names)

    def sample(self, rng, tpl, cls, split, aux=0):
        n = [str(x) for x in rng.choice(self.names, 4, replace=False)]
        rest = [i for i in range(4) if i != cls]
        order = [None] * 4  # order[i] = class of the object named n[i]
        order[aux] = cls
        for i, c in zip([i for i in range(4) if i != aux], rng.permutation(rest)):
            order[i] = int(c)
        s = {f"n{i}": n[i] for i in range(4)} | {f"o{i}": self.opts[order[i]] for i in range(4)}
        s["q"] = n[aux]
        s["order"] = order
        return s

    def _names(self, s):
        return [s[f"n{i}"] for i in range(4)]

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if s[f"n{i}"] != s["q"]]))
        return dict(s, q=s[f"n{j}"]), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if s[f"n{i}"] != s["q"]]))
        return dict(s, **{f"n{j}": str(rng.choice([w for w in self.names if w not in self._names(s)]))})

    def solve(self, prompt):
        ws = words_of(prompt)
        names = [w for w in ws if w in self.names]
        objs = [w for w in ws if w in self.opts]
        assert len(names) == 5 and len(objs) == 4, prompt
        return self.opts.index(objs[names[:4].index(names[4])])


def _bo_templates(verbs, clause, last):
    intros = ["So,", "Well,", "Now,", "Look,", "Okay,"]
    return [f"{x} " + clause(v, "{n0}", "{o0}") + ", " + clause(v, "{n1}", "{o1}") + ", " +
            clause(v, "{n2}", "{o2}") + " and " + clause(v, "{n3}", "{o3}") + ". " + last(v)
            for x in intros for v in verbs]


_BO_NEAR = {"first_object": lambda s: s["order"][0], "last_object": lambda s: s["order"][3]}
_BO_OPTS_A, _BO_OPTS_B = (" cat", " dog", " fish", " bird"), (" red", " blue", " green", " black")
_BO_TPL_A = _bo_templates(["has", "owns", "keeps"], lambda v, n, o: f"{n} {v} a {o}", lambda v: "{q} " + v + " a")
_BO_TPL_B = _bo_templates(["car", "bag", "hat"], lambda v, n, o: f"{n}'s {v} is {o}", lambda v: "{q}'s " + v + " is")
for _s, _tpl, _opts in (("a", _BO_TPL_A, _BO_OPTS_A), ("b", _BO_TPL_B, _BO_OPTS_B)):
    register(Task(f"bind_owner_{_s}", "bind_owner", "binding/reference", _opts,
                  BindSpec(_tpl, _opts, NAMES, _BO_NEAR), copy=True))


# ---------------------------------------------------------------- 4. arith_addsub
class ArithSpec(Spec):
    """`{a}+{b}-{c}=` (form "ab-c") or `{a}-{b}+{c}=` (form "a-bc"): the answer is the value, an option digit.
    The surface slot is the item number k (1..9), which the answer ignores. Operands are 1..9."""
    nums = range(1, 10)

    def __init__(self, templates, form, lo, near_miss):
        self.templates, self.form, self.lo, self.near_miss = templates, form, lo, near_miss
        _check_templates(templates, [])

    def value(self, a, b, c):
        return a + b - c if self.form == "ab-c" else a - b + c

    def _trip(self, c):
        return [(a, b, d) for a in self.nums for b in self.nums for d in self.nums
                if self.value(a, b, d) - self.lo == c]

    def sample(self, rng, tpl, cls, split, aux=0):
        t = self._trip(cls)
        a, b, c = t[rng.integers(len(t))]
        return {"a": a, "b": b, "c": c, "k": int(rng.choice(list(self.nums)))}

    def ans_twin(self, rng, tpl, s, cls, split):
        opts = []
        for key in "abc":
            for v in self.nums:
                s2 = dict(s, **{key: v})
                c2 = self.value(s2["a"], s2["b"], s2["c"]) - self.lo
                if v != s[key] and 0 <= c2 < 4 and c2 != cls:
                    opts.append((s2, c2))
        return opts[rng.integers(len(opts))]

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, k=int(rng.choice([k for k in self.nums if k != s["k"]])))

    def solve(self, prompt):
        m = re.search(r"(\d)([+-])(\d)([+-])(\d)=$", prompt)
        a, o1, b, o2, c = int(m[1]), m[2], int(m[3]), m[4], int(m[5])
        return a + (b if o1 == "+" else -b) + (c if o2 == "+" else -c) - self.lo


def _as_templates(expr):
    pre = ["Problem", "Exercise", "Question", "Task"]
    return [f"{p} {{k}}{sep}" + expr for p in pre for sep in (":\n", ".\n", "\n")]


_AS_NEAR_A = {"ignore_sub": lambda s: min(max(s["a"] + s["b"] - 3, 0), 3), "last_operand": lambda s: s["c"] % 4}
_AS_NEAR_B = {"ignore_sub": lambda s: min(max(s["a"] + s["c"] - 2, 0), 3), "last_operand": lambda s: s["c"] % 4}
register(Task("arith_addsub_a", "arith_addsub", "arithmetic", ("3", "4", "5", "6"),
              ArithSpec(_as_templates("{a}+{b}-{c}="), "ab-c", 3, _AS_NEAR_A)))
register(Task("arith_addsub_b", "arith_addsub", "arithmetic", ("2", "3", "4", "5"),
              ArithSpec(_as_templates("{a}-{b}+{c}="), "a-bc", 2, _AS_NEAR_B)))


# ---------------------------------------------------------------- 5. code_alias
class CodeAliasSpec(Spec):
    """Four variables hold the 4 option digits (permuted). `alias`: `t = s` binds a new name t, then `print(t)`;
    `overwrite`: `t = s` rebinds one of the four names, then `print(t)`. The answer is the value of s, at mention
    rank `aux` among the digits (balanced). Surface slot: the name of a variable that is neither t nor s."""

    def __init__(self, templates, options, mode, near_miss):
        self.templates, self.opts, self.mode, self.near_miss = templates, [o.strip() for o in options], mode, near_miss
        _check_templates(templates, self.opts)

    def sample(self, rng, tpl, cls, split, aux=0):
        names = [str(x) for x in rng.choice(LETTERS, 5, replace=False)]
        v, t = names[:4], names[4]
        order = [None] * 4
        order[aux] = cls
        for i, c in zip([i for i in range(4) if i != aux], rng.permutation([c for c in range(4) if c != cls])):
            order[i] = int(c)
        if self.mode == "overwrite":
            t = v[int(rng.choice([i for i in range(4) if i != aux]))]
        s = {f"v{i}": v[i] for i in range(4)} | {f"d{i}": self.opts[order[i]] for i in range(4)}
        return s | {"t": t, "s": v[aux], "order": order, "t_idx": v.index(t) if t in v else -1}

    def _v(self, s):
        return [s[f"v{i}"] for i in range(4)]

    def ans_twin(self, rng, tpl, s, cls, split):
        v = self._v(s)
        j = int(rng.choice([i for i in range(4) if v[i] not in (s["s"], s["t"])]))
        return dict(s, s=v[j]), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        v = self._v(s)
        j = int(rng.choice([i for i in range(4) if v[i] not in (s["s"], s["t"])]))
        free = [w for w in LETTERS if w not in v and w != s["t"]]
        return dict(s, **{f"v{j}": str(rng.choice(free))})

    def solve(self, prompt):
        env, shown = {}, None
        for line in prompt.split("\n"):
            line = re.sub(r"^(>>> )", "", line.strip())
            if m := re.fullmatch(r"(\w+) ?= ?(\w+)", line):
                env[m[1]] = m[2] if m[2].isdigit() else env[m[2]]
            elif m := re.fullmatch(r"print\((\w+)\)", line):
                shown = env[m[1]]
        return self.opts.index(shown)


def _ca_templates():
    out = []
    for p, hdrs in ((">>> ", ["", "# demo\n", "# trace\n", "# test\n"]), ("", ["# demo\n", "# trace\n", "# test\n"])):
        for h in hdrs:
            for eq in (" = ", "="):
                body = "".join(f"{p}{{v{i}}}{eq}{{d{i}}}\n" for i in range(4))
                out.append(h + body + f"{p}{{t}}{eq}{{s}}\n{p}print({{t}})\n")
    return out


_CA_NEAR_A = {"first_value": lambda s: s["order"][0], "last_value": lambda s: s["order"][3]}
# No "stale value" rule: t's old value is never the answer (accuracy 0.0 by construction, not a shortcut); eliminating
# it and guessing among the other three scores 1/3, within the 0.10 band.
_CA_NEAR_B = {"first_value": lambda s: s["order"][0], "last_value": lambda s: s["order"][3]}
_CA_OPTS_A, _CA_OPTS_B = ("3", "5", "7", "9"), ("2", "4", "6", "8")
register(Task("code_alias_a", "code_alias", "code tracing", _CA_OPTS_A,
              CodeAliasSpec(_ca_templates(), _CA_OPTS_A, "alias", _CA_NEAR_A), copy=True,
              notes="copy=True: the answer digit is one of the four assigned digits shown in the prompt"))
register(Task("code_alias_b", "code_alias", "code tracing", _CA_OPTS_B,
              CodeAliasSpec(_ca_templates(), _CA_OPTS_B, "overwrite", _CA_NEAR_B), copy=True,
              notes="copy=True: the answer digit is one of the four assigned digits shown in the prompt"))
