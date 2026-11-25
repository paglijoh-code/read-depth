"""Batch 9 (families 41-45): code_var, code_overwrite, rule_means, rule_exclude, hop_lang (two tasks each).

Pools below are literal lists, vetted with `tokcheck.vet_words` against all 17 tokenizers (7 distinct vocabularies): every
word is one token with its leading space everywhere (code values: one token after "# prints", checked by the option test).
"""
from __future__ import annotations

import re

from .b1 import LETTERS, NAMES, _check_templates
from .b5 import _FL_A, _FL_B, _FL_OPTS_A, _FL_OPTS_B
from .b7 import MONTHS, _perm_with
from .core import Spec, Task, _split_words, register, words_of

_ORD = "first second third fourth".split()


# ---------------------------------------------------------------- 41. code_var
# Four variables are each assigned one string value (the 4 option words, permuted); `print(q)` prints one of them. The answer is
# shown once, at the printed variable's line, whose rank among the values is `aux` (balanced). The prompt ends "print(q)\n# prints"
# and the answer is the next word (" red"), a single token everywhere. Surface slot: the name of an unprinted variable.
class CodeVarSpec(Spec):
    def __init__(self, templates, options, names, near_miss):
        self.templates, self.opts, self.names, self.near_miss = templates, [o.strip() for o in options], names, near_miss
        _check_templates(templates, self.opts)

    def sample(self, rng, tpl, cls, split, aux=0):
        v = [str(x) for x in rng.choice(self.names, 4, replace=False)]
        order = _perm_with(rng, cls, aux)
        s = {f"v{i}": v[i] for i in range(4)} | {f"d{i}": self.opts[order[i]] for i in range(4)}
        return s | {"q": v[aux], "pos": aux, "order": order}

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["pos"]]))
        return dict(s, q=s[f"v{j}"], pos=j), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["pos"]]))
        used = [s[f"v{i}"] for i in range(4)]
        return dict(s, **{f"v{j}": str(rng.choice([w for w in self.names if w not in used]))})

    def solve(self, prompt):
        env, shown = {}, None
        for line in prompt.split("\n"):
            if m := re.fullmatch(r"(\w+) = (['\"])(\w+)\2", line.strip()):
                env[m[1]] = m[3]
            elif m := re.fullmatch(r"print\((\w+)\)", line.strip()):
                shown = env[m[1]]
        return self.opts.index(shown)


def _cv_templates(heads, body):
    return [f"# {h}\n" + body.replace("'", q) + "print({q})\n# prints" for h in heads for q in ("'", '"')]


_CV_HEADS = "demo trace test code run vars check play".split()
_CV_BODY = "".join(f"{{v{i}}} = '{{d{i}}}'\n" for i in range(4))
_CV_NAMES_A, _CV_NAMES_B = "a b c d e f g h k m n".split(), "p q r s t u v w x y z".split()
_CV_OPTS_A, _CV_OPTS_B = (" red", " blue", " green", " white"), (" cat", " dog", " cow", " hen")
_CV_NEAR = {"first_value": lambda s: s["order"][0], "second_value": lambda s: s["order"][1],
            "last_value": lambda s: s["order"][3]}
register(Task("code_var_a", "code_var", "code tracing", _CV_OPTS_A,
              CodeVarSpec(_cv_templates(_CV_HEADS, _CV_BODY), _CV_OPTS_A, _CV_NAMES_A, _CV_NEAR), copy=True,
              notes="four string variables, print one; ends 'print(q)\\n# prints'; answer rank balanced (aux)"))
register(Task("code_var_b", "code_var", "code tracing", _CV_OPTS_B,
              CodeVarSpec(_cv_templates(_CV_HEADS[::-1], _CV_BODY), _CV_OPTS_B, _CV_NAMES_B, _CV_NEAR), copy=True,
              notes="as code_var_a with other variable names and animal values"))


# ---------------------------------------------------------------- 42. code_overwrite
# Four assignment lines with 4 distinct string values (the 4 option words). The printed variable `p` is assigned on several lines
# (a: up to two, b: up to three) and its last assignment is line `aux` (balanced), so the answer is that line's value and every
# line after it belongs to another variable; the lines of other variables each use a fresh name. The prompt ends "print(p)\n#
# prints". Near-misses: "the variable's first value" is only right when p is assigned once (aux 0), so it sits at 0.25; "the last value on the page" is right when aux = 3. Surface slot: the name of another variable's line.
class CodeOverwriteSpec(Spec):
    def __init__(self, templates, options, names, max_p, near_miss):
        self.templates, self.opts, self.names, self.max_p = templates, [o.strip() for o in options], names, max_p
        self.near_miss = near_miss
        _check_templates(templates, self.opts)

    def sample(self, rng, tpl, cls, split, aux=0):
        nm = [str(x) for x in rng.choice(self.names, 4, replace=False)]
        p, rest = nm[0], nm[1:]
        k = min(self.max_p, aux + 1)
        pl = sorted([int(i) for i in rng.choice(aux, k - 1, replace=False)] + [aux]) if aux else [0]
        order = _perm_with(rng, cls, aux)
        v = []
        for i in range(4):
            v.append(p if i in pl else rest.pop())
        s = {f"v{i}": v[i] for i in range(4)} | {f"d{i}": self.opts[order[i]] for i in range(4)}
        return s | {"q": p, "p": p, "pl": pl, "pos": aux, "order": order}

    def _others(self, s):
        return [i for i in range(4) if s[f"v{i}"] != s["p"]]

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice(self._others(s)))
        return dict(s, q=s[f"v{j}"]), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice(self._others(s)))
        used = [s[f"v{i}"] for i in range(4)]
        return dict(s, **{f"v{j}": str(rng.choice([w for w in self.names if w not in used]))})

    def solve(self, prompt):
        env, shown = {}, None
        for line in prompt.split("\n"):
            if m := re.fullmatch(r"(\w+) = (['\"])(\w+)\2", line.strip()):
                env[m[1]] = m[3]
            elif m := re.fullmatch(r"print\((\w+)\)", line.strip()):
                shown = env[m[1]]
        return self.opts.index(shown)


_CO_HEADS_A, _CO_HEADS_B = "demo trace test code run edit set vars".split(), "check play step case mix list data base".split()
_CO_OPTS_A, _CO_OPTS_B = (" red", " blue", " green", " black"), (" cat", " dog", " horse", " bear")
_CO_NEAR = {"first_of_var": lambda s: s["order"][s["pl"][0]], "first_on_page": lambda s: s["order"][0],
            "last_on_page": lambda s: s["order"][3]}
register(Task("code_overwrite_a", "code_overwrite", "code tracing", _CO_OPTS_A,
              CodeOverwriteSpec(_cv_templates(_CO_HEADS_A, _CV_BODY), _CO_OPTS_A, _CV_NAMES_A, 2, _CO_NEAR), copy=True,
              notes="the printed variable is assigned up to twice among four lines; answer is its last value"))
register(Task("code_overwrite_b", "code_overwrite", "code tracing", _CO_OPTS_B,
              CodeOverwriteSpec(_cv_templates(_CO_HEADS_B, _CV_BODY), _CO_OPTS_B, _CV_NAMES_B, 3, _CO_NEAR), copy=True,
              notes="the printed variable is assigned up to three times among four lines; answer is its last value"))


# ---------------------------------------------------------------- 43. rule_means
# "{k0} means {a0}, ... and {k3} means {a3}": four keys (colours in a, sounds in b) and the 4 option actions, with the actions
# permuted over the rules every item; the query names the key of rule `aux` (balanced), so the answer's mention rank is balanced.
# Each action is shown once. Surface slot: the person's name.
class MeansSpec(Spec):
    def __init__(self, templates, options, keys, near_miss):
        self.templates, self.opts, self.keys, self.near_miss = templates, [o.strip() for o in options], keys, near_miss
        _check_templates(templates, self.opts + keys + NAMES)

    def sample(self, rng, tpl, cls, split, aux=0):
        k = [str(x) for x in rng.choice(self.keys, 4, replace=False)]
        order = _perm_with(rng, cls, aux)
        s = {f"k{i}": k[i] for i in range(4)} | {f"a{i}": self.opts[order[i]] for i in range(4)}
        return s | {"q": k[aux], "n": str(rng.choice(NAMES)), "pos": aux, "order": order}

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["pos"]]))
        return dict(s, q=s[f"k{j}"], pos=j), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, n=str(rng.choice([w for w in NAMES if w != s["n"]])))

    def solve(self, prompt):
        rule = {m[1]: m[2] for m in re.finditer(r"(\w+) means (\w+)", prompt)}
        q = [w for w in words_of(prompt) if w in self.keys][-1]
        return self.opts.index(rule[q])


def _mn_templates(art, tails):
    rules = ", ".join(f"{art}{{k{i}}} means {{a{i}}}" for i in range(3)) + f" and {art}{{k3}} means {{a3}}"
    return [f"{f} {{n}} {v} {x}: {rules}. {t}" for f, v, x in
            (("Today", "learned", "a game"), ("Once", "played", "a game"), ("Later", "learned", "the rules"),
             ("Then", "played", "the rules")) for t in tails]


_MN_A = _mn_templates("", ["The light is {q}, so you", "The signal is {q}, so you", "The card is {q}, so you"])
_MN_B = _mn_templates("a ", ["You hear a {q}, so you", "Then comes a {q}, so you", "The sound is a {q}, so you"])
_MN_NEAR = {"first_rule": lambda s: s["order"][0], "second_rule": lambda s: s["order"][1],
            "last_rule": lambda s: s["order"][3]}
_MN_OPTS_A, _MN_OPTS_B = (" stop", " go", " wait", " turn"), (" eat", " sleep", " run", " jump")
register(Task("rule_means_a", "rule_means", "rule application", _MN_OPTS_A,
              MeansSpec(_MN_A, _MN_OPTS_A, "red green blue white black yellow".split(), _MN_NEAR), copy=True,
              notes="colour means action, mapping permuted every item; answer's rule rank balanced (aux)"))
register(Task("rule_means_b", "rule_means", "rule application", _MN_OPTS_B,
              MeansSpec(_MN_B, _MN_OPTS_B, "bell drum horn flag lamp".split(), _MN_NEAR), copy=True,
              notes="sound means action, mapping permuted every item; answer's rule rank balanced (aux)"))


# ---------------------------------------------------------------- 44. rule_exclude
# Four options are listed (permuted, the answer at list position `aux`, balanced); three of them are taken or leave, each named by
# its list ordinal ("took the first"), and the answer is the one left. Deviation from the sketch ("Tom took red, Anna took blue"):
# naming the taken colours again made the answer the only option word mentioned once, which a bag of words solves; with ordinals
# every option word is shown exactly once. a: Of {colours}, {name} took the {ordinal} ...; b: four people enter a {room}, three
# leave by ordinal, "Still inside is". Surface slot: a name (a), the room word (b).
class ExcludeSpec(Spec):
    def __init__(self, templates, options, mode, rooms, near_miss):
        self.templates, self.opts, self.mode, self.rooms = templates, [o.strip() for o in options], mode, rooms
        self.near_miss = near_miss
        _check_templates(templates, self.opts + _ORD + NAMES + rooms)

    def sample(self, rng, tpl, cls, split, aux=0):
        order = _perm_with(rng, cls, aux)
        t = [int(i) for i in rng.permutation([i for i in range(4) if i != aux])]
        s = {f"c{i}": self.opts[order[i]] for i in range(4)} | {f"t{i}": _ORD[t[i]] for i in range(3)}
        if self.mode == "a":
            s |= {f"n{i}": str(x) for i, x in enumerate(rng.choice(NAMES, 3, replace=False))}
        else:
            s["r"] = str(rng.choice(self.rooms))
        return s | {"pos": aux, "order": order, "t": t}

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.integers(3))
        t = list(s["t"])
        left, t[j] = s["pos"], s["pos"]
        return dict(s, **{f"t{j}": _ORD[left]}, t=t, pos=s["t"][j]), s["order"][s["t"][j]]

    def surf_twin(self, rng, tpl, s, cls, split):
        if self.mode == "a":
            j = int(rng.integers(3))
            free = [w for w in NAMES if w not in (s["n0"], s["n1"], s["n2"])]
            return dict(s, **{f"n{j}": str(rng.choice(free))})
        return dict(s, r=str(rng.choice([w for w in self.rooms if w != s["r"]])))

    def solve(self, prompt):
        ws = words_of(prompt)
        listed = [w for w in ws if w in self.opts]
        taken = [_ORD.index(w) for w in ws if w in _ORD]
        assert len(listed) == 4 and len(taken) == 3, prompt
        return self.opts.index(listed[[i for i in range(4) if i not in taken][0]])


_EX_A = [f"{s} {{c0}}, {{c1}}, {{c2}} and {{c3}}, {{n0}} {v} the {{t0}}, {{n1}} {v} the {{t1}} and {{n2}} {v} the {{t2}}. "
         f"The one left was" for s in ("Of", "Among", "From", "Out of") for v in ("took", "chose", "picked")]
_EX_B = [f"{s} {{c0}}, {{c1}}, {{c2}} and {{c3}} {w} the {{r}}. Then the {{t0}}, the {{t1}} and the {{t2}} {v}. Still inside is"
         for s, w in (("Today", "went into"), ("Then", "walked into"), ("Once", "entered"), ("Later", "came into"))
         for v in ("left", "went out", "walked out")]
_EX_NEAR = {"first_listed": lambda s: s["order"][0], "second_listed": lambda s: s["order"][1],
            "last_listed": lambda s: s["order"][3]}
_EX_OPTS_A, _EX_OPTS_B = (" red", " blue", " green", " white"), (" Tom", " Anna", " Ben", " Mary")
_EX_ROOMS = "kitchen garden hall office room shop house school".split()
register(Task("rule_exclude_a", "rule_exclude", "rule application", _EX_OPTS_A,
              ExcludeSpec(_EX_A, _EX_OPTS_A, "a", _EX_ROOMS, _EX_NEAR), copy=True,
              notes="four colours listed, three taken by list ordinal; answer is the colour left; listed rank balanced (aux)"))
register(Task("rule_exclude_b", "rule_exclude", "rule application", _EX_OPTS_B,
              ExcludeSpec(_EX_B, _EX_OPTS_B, "b", _EX_ROOMS, _EX_NEAR), copy=True,
              notes="four people enter a room, three leave by list ordinal; answer is the one still inside"))


# ---------------------------------------------------------------- 45. hop_lang
# Four people are born in (a) or live in (b) four places of four different language classes (reusing fact_language's vetted pools
# and options; a: fact_language_b's, b: fact_language_a's), then the query asks the language of one person: name -> place -> language.
# The queried person is at rank `aux` (balanced); the three other places are the distractors, so both hops are needed. Held-out
# places per class (P0 split). Deviation from the sketch ("a {job}"): no job word, because with only two people the first-place
# rule scored 0.5; the surface slot is the month in the frame, as in hop_twohop.
class HopLangSpec(Spec):
    def __init__(self, templates, class_pools, near_miss):
        self.templates, self.class_pools, self.near_miss, self.cls_key = templates, class_pools, near_miss, "c"
        self._cls_of = {w: c for c, p in enumerate(class_pools) for w in p}
        _check_templates(templates, list(self._cls_of) + NAMES + MONTHS)

    def sample(self, rng, tpl, cls, split, aux=0):
        order = _perm_with(rng, cls, aux)  # order[i] = language class of person i's place
        n = [str(x) for x in rng.choice(NAMES, 4, replace=False)]
        c = [str(rng.choice(_split_words(self.class_pools[order[i]], split))) for i in range(4)]
        s = {f"n{i}": n[i] for i in range(4)} | {f"c{i}": c[i] for i in range(4)}
        return s | {"q": n[aux], "c": c[aux], "mo": str(rng.choice(MONTHS)), "pos": aux, "order": order}

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["pos"]]))
        return dict(s, q=s[f"n{j}"], c=s[f"c{j}"], pos=j), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, mo=str(rng.choice([m for m in MONTHS if m != s["mo"]])))

    def solve(self, prompt):
        ws = words_of(prompt)
        n, c = [w for w in ws if w in NAMES], [w for w in ws if w in self._cls_of]
        assert len(n) == 5 and len(c) == 4, prompt
        return self._cls_of[c[n[:4].index(n[4])]]


def _hl_templates(intros, link, tails):
    body = f"{{n0}} {link} {{c0}}, {{n1}} in {{c1}}, {{n2}} in {{c2}} and {{n3}} in {{c3}}"
    return [f"{i} {body}. {t}" for i in intros for t in tails]


_HL_A = _hl_templates(["In {mo} we asked four people about home.", "By {mo} we had met four people.",
                       "Last {mo} we met four people.", "Next {mo} we will meet four people."], "was born in",
                      ["The language {q} grew up speaking is", "The language {q} spoke as a child is",
                       "The first language of {q} is"])
_HL_B = _hl_templates(["In {mo} we visited four people.", "By {mo} we had called four people.",
                       "Last {mo} we wrote to four people.", "Next {mo} we will see four people."], "lives in",
                      ["The language {q} speaks at home is", "The language {q} hears every day is",
                       "The local language where {q} lives is"])
_HL_NEAR = {"first_place": lambda s: s["order"][0], "second_place": lambda s: s["order"][1],
            "last_place": lambda s: s["order"][3]}
register(Task("hop_lang_a", "hop_lang", "multi-hop composition", _FL_OPTS_B,
              HopLangSpec(_HL_A, _FL_B, _HL_NEAR),
              notes="name -> birthplace -> language; four people, four language classes; fact_language_b's pools"))
register(Task("hop_lang_b", "hop_lang", "multi-hop composition", _FL_OPTS_A,
              HopLangSpec(_HL_B, _FL_A, _HL_NEAR),
              notes="name -> home place -> language; four people, four language classes; fact_language_a's pools"))
