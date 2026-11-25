"""Batch 2 (families 6-10): rule_ifmap, hop_fact, fact_antonym, bind_listpos, arith_extreme (two tasks each).

Pools below are literal lists, vetted with `tokcheck.vet_words` against all 17 tokenizers (7 distinct vocabularies): every
word is one token with its leading space everywhere. Class pools are ordered so that P0's split (every third word is a
test word) leaves the length and initial-letter near-miss rules at chance on the held-out half.
"""
from __future__ import annotations

import re

import numpy as np

from .b1 import _BO_NEAR, _FC_A, _FC_B, _FC_NEAR, NAMES, BindSpec, _check_templates
from .core import Spec, Task, WordSpec, _split_words, register, words_of


# ---------------------------------------------------------------- 6. rule_ifmap
# Four "if condition, output" rules (conditions drawn from a pool of 10, outputs a permutation of the 4 options), then a
# query condition. Reuses b1's BindSpec: condition = `name`, output = `object`. The answer word is in the prompt, so it is a
# copy task (position rival reported); the answer's rule rank is balanced by `aux`.
_RI_SHAPES = "circle square triangle star heart cross ring cube cone moon".split()
_RI_WEATHER = "rain snow wind fog storm sun ice heat mist cloud".split()
_RI_OPTS_A, _RI_OPTS_B = (" red", " blue", " green", " white"), (" bus", " car", " train", " boat")


def _ri_templates(heads, verbs, clause, last):
    out = []
    for h in heads:
        for v in verbs:
            body = "; ".join(clause(i, v, h if i == 0 else None) for i in range(4))
            out.append(body + ". " + last(v))
    return out


_RI_TPL_A = _ri_templates(
    ["If", "Rule: if", "Note: if", "Remember, if"], ["paint it", "colour it", "make it", "dye it"],
    lambda i, v, h: (f"{h} the shape is a " if i == 0 else "if a ") + "{n%d}, %s {o%d}" % (i, v, i),
    lambda v: "The shape is a {q}, so " + v)
_RI_TPL_B = _ri_templates(
    ["If", "Rule: if", "Note: if", "Remember, if"], ["go by", "travel by", "get there by", "come by"],
    lambda i, v, h: (f"{h} there is " if i == 0 else "if ") + "{n%d}, %s {o%d}" % (i, v, i),
    lambda v: "Today there is {q}, so " + v)

for _s, _tpl, _pool, _opts in (("a", _RI_TPL_A, _RI_SHAPES, _RI_OPTS_A), ("b", _RI_TPL_B, _RI_WEATHER, _RI_OPTS_B)):
    register(Task(f"rule_ifmap_{_s}", "rule_ifmap", "rule application", _opts,
                  BindSpec(_tpl, _opts, _pool, _BO_NEAR), copy=True,
                  notes="the output words are all in the prompt; the answer's rule rank is balanced (aux)"))


# ---------------------------------------------------------------- 7. hop_fact
# a: one person with a job (surface slot) born in a place; answer = class of the place (continent, as batch 1's
#    fact_continent_a pools). b: four people, each from a place of a different class; ask about one (deviation from the
#    sketch's two people: with two people "always answer the first/last place" scores >= 0.5, so no chance-level rival
#    rule can exist; with four distinct-class places it is 0.25). Places are held out (b1's pools and class labels).
class HopSpec(Spec):
    """Four people n0..n3 from places c0..c3 of four different classes (a permutation), query person q = n[aux]; the
    answer is the class of q's place. `c` is q's place (the class-word slot). Surface slot: a non-queried name."""

    def __init__(self, templates, class_pools, names, near_miss):
        self.templates, self.class_pools, self.names, self.near_miss = templates, class_pools, names, near_miss
        self.cls_key = "c"
        self._cls_of = {w: c for c, p in enumerate(class_pools) for w in p}
        assert len(self._cls_of) == sum(map(len, class_pools)), "a word is in two classes"
        _check_templates(templates, list(self._cls_of) + names)

    def sample(self, rng, tpl, cls, split, aux=0):
        n = [str(x) for x in rng.choice(self.names, 4, replace=False)]
        order = [None] * 4  # order[i] = class of the place of n[i]
        order[aux] = cls
        for i, c in zip([i for i in range(4) if i != aux], rng.permutation([c for c in range(4) if c != cls])):
            order[i] = int(c)
        pl = [str(rng.choice(_split_words(self.class_pools[order[i]], split))) for i in range(4)]
        s = {f"n{i}": n[i] for i in range(4)} | {f"c{i}": pl[i] for i in range(4)}
        return s | {"q": n[aux], "c": pl[aux], "order": order}

    def _names(self, s):
        return [s[f"n{i}"] for i in range(4)]

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if s[f"n{i}"] != s["q"]]))
        return dict(s, q=s[f"n{j}"], c=s[f"c{j}"]), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if s[f"n{i}"] != s["q"]]))
        return dict(s, **{f"n{j}": str(rng.choice([w for w in self.names if w not in self._names(s)]))})

    def solve(self, prompt):
        ws = words_of(prompt)
        names = [w for w in ws if w in self.names]
        places = [w for w in ws if w in self._cls_of]
        assert len(names) == 5 and len(places) == 4, prompt
        return self._cls_of[places[names[:4].index(names[4])]]


_HF_JOBS = "doctor teacher pilot lawyer driver painter singer writer cook soldier judge guard".split()


def _hf_templates_a(places):
    heads = ["Meet {n}, a {j} born in {c}.", "Here is {n}, a {j} who was born in {c}.", "Consider {n}, a {j} from {c}.",
             "Take {n}, a {j} who grew up in {c}."]
    tails = ["So {n} comes from a place in", "The home of {n} lies in a land in", "Thus {n} is from a place in",
             "That means {n} lives in a land in"]
    return [f"{h} {t}" + places for h in heads for t in tails]


def _hf_templates_b(places):
    out = []
    for x in ["So,", "Well,", "Now,", "Look,", "Okay,"]:
        for v, tail in (("is from", "{q} is from a place in"), ("was born in", "{q} was born in a place in"),
                        ("grew up in", "{q} grew up in a land in")):
            cl = [f"{{n{i}}} {v} {{c{i}}}" for i in range(4)]
            out.append(f"{x} " + ", ".join(cl[:3]) + " and " + cl[3] + ". " + tail + places)
    return out


_HF_NEAR_B = dict(_FC_NEAR, first_place=lambda s: s["order"][0], last_place=lambda s: s["order"][3])
_HF_OPTS_A, _HF_OPTS_B = (" Europe", " Asia", " North", " South"), (" north", " east", " south", " west")
_HF_TPL_A, _HF_TPL_B = _hf_templates_a(""), _hf_templates_b(" the")
_check_templates(_HF_TPL_A, [w for p in _FC_A for w in p] + NAMES + _HF_JOBS)
register(Task("hop_fact_a", "hop_fact", "multi-hop composition", _HF_OPTS_A,
              WordSpec(_HF_TPL_A, "c", _FC_A, "j", _HF_JOBS, extra={"n": NAMES}, near_miss=_FC_NEAR),
              notes="name -> place (stated) -> continent (recalled); places held out"))
register(Task("hop_fact_b", "hop_fact", "multi-hop composition", _HF_OPTS_B,
              HopSpec(_HF_TPL_B, _FC_B, NAMES, _HF_NEAR_B),
              notes="four people from four places (US regions), ask about one; places held out"))


# ---------------------------------------------------------------- 8. fact_antonym
# Cue word -> its opposite. Cues are held out (every third word of a class pool is a test word). Surface slot: the asker.
_FA_A = ["giant immense massive large big wide vast broad huge great".split(),  # -> small
         "rapid fleet prompt quick swift hurried racing fast".split(),  # -> slow
         "powerful solid dense strong bold firm mighty".split(),  # -> weak
         "good super pleasant decent splendid fine excellent grand perfect wonderful nice".split()]  # -> bad
_FA_B = ["accurate precise sure true certain exact correct".split(),  # -> false
         "silent soft calm quiet still gentle".split(),  # -> loud
         "light clear lit pale brilliant bright".split(),  # -> dark
         "recent modern young current new latest fresh".split()]  # -> old
_FA_OPTS_A, _FA_OPTS_B = (" small", " slow", " weak", " bad"), (" false", " loud", " dark", " old")


def _fa_templates(intros, tails):
    return [f"{i} {t}" for i in intros for t in tails]


_FA_TPL_A = _fa_templates(
    ["Then {n} asked for the opposite of the word {w}.", "Later {n} looked up the opposite of {w}.",
     "Today {n} wondered about the opposite of {w}.", "Once {n} was asked for the opposite of {w}."],
    ["The opposite is", "The opposite word is", "Its opposite is", "The antonym is"])
_FA_TPL_B = _fa_templates(
    ["Next {n} wrote down the opposite of the word {w}.", "Soon {n} was told to find the opposite of {w}.",
     "Now {n} tried to recall the opposite of {w}.", "Here {n} searched for the opposite of {w}."],
    ["The answer is the word", "The opposite is the word", "Its antonym is", "The opposing word is"])
_FA_NEAR = {"len_mod4": lambda s: len(s["w"]) % 4, "initial_bucket": lambda s: (ord(s["w"][0]) - 97) % 4}
for _s, _tpl, _pools, _opts in (("a", _FA_TPL_A, _FA_A, _FA_OPTS_A), ("b", _FA_TPL_B, _FA_B, _FA_OPTS_B)):
    _check_templates(_tpl, [w for p in _pools for w in p] + [o.strip() for o in _opts] + NAMES)
    register(Task(f"fact_antonym_{_s}", "fact_antonym", "fact lookup", _opts,
                  WordSpec(_tpl, "w", _pools, "n", NAMES, near_miss=_FA_NEAR)))


# ---------------------------------------------------------------- 9. bind_listpos
ORDINALS = "first second third fourth".split()


class ListPosSpec(Spec):
    """The 4 option words appear in a list (a permutation); the prompt asks for the k-th. The answer sits at list
    position `aux` (balanced), so the answer's mention rank is balanced and the position rival is chance."""

    def __init__(self, templates, options, surf_key, surf_pool, near_miss):
        self.templates, self.opts, self.near_miss = templates, [o.strip() for o in options], near_miss
        self.surf_key, self.surf_pool = surf_key, surf_pool
        _check_templates(templates, self.opts + ORDINALS + surf_pool)

    def sample(self, rng, tpl, cls, split, aux=0):
        order = [None] * 4  # order[i] = class of the i-th listed word
        order[aux] = cls
        for i, c in zip([i for i in range(4) if i != aux], rng.permutation([c for c in range(4) if c != cls])):
            order[i] = int(c)
        s = {f"o{i}": self.opts[order[i]] for i in range(4)}
        return s | {"k": ORDINALS[aux], "pos": aux, "order": order, self.surf_key: str(rng.choice(self.surf_pool))}

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["pos"]]))
        return dict(s, k=ORDINALS[j], pos=j), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, **{self.surf_key: str(rng.choice([w for w in self.surf_pool if w != s[self.surf_key]]))})

    def solve(self, prompt):
        ws = words_of(prompt)
        objs = [w for w in ws if w in self.opts]
        ks = [w for w in ws if w in ORDINALS]
        assert len(objs) == 4 and len(ks) == 1, prompt
        return self.opts.index(objs[ORDINALS.index(ks[0])])


def _lp_templates_a():
    out = []
    for h in ("At the {u}, the %s was ", "Outside the {u}, the %s was "):
        for noun in ("queue", "line"):
            for lst in ("{o0}, {o1}, {o2} and {o3}", "{o0}, {o1}, {o2}, {o3}"):
                for t in ("The {k} in the %s was", "Who was {k} in the %s? It was", "Looking at the %s, the {k} person was"):
                    out.append((h + lst + ". " + t) % (noun, noun))
    return out


def _lp_templates_b():
    return [f"{x} {{u}} {v} four flags in a row: {{o0}}, {{o1}}, {{o2}} and {{o3}}. {t}"
            for x in ("Then", "Later", "Once", "Today") for v in ("saw", "spotted", "counted", "noticed")
            for t in ("The {k} flag was",)] + \
           [f"{x} {{u}} {v} four flags in a row: {{o0}}, {{o1}}, {{o2}} and {{o3}}. {t}"
            for x in ("Then", "Later") for v in ("saw", "spotted") for t in ("Of the four, the {k} flag was",)]


_LP_PLACES = ("bank shop park school station market cinema museum library hotel church farm office store garden theatre "
              "bridge").split()
_LP_NEAR = {"first_listed": lambda s: s["order"][0], "last_listed": lambda s: s["order"][3]}
_LP_OPTS_A, _LP_OPTS_B = (" Tom", " Anna", " Ben", " Dan"), (" red", " blue", " green", " white")
register(Task("bind_listpos_a", "bind_listpos", "binding/reference", _LP_OPTS_A,
              ListPosSpec(_lp_templates_a(), _LP_OPTS_A, "u", _LP_PLACES, _LP_NEAR), copy=True))
register(Task("bind_listpos_b", "bind_listpos", "binding/reference", _LP_OPTS_B,
              ListPosSpec(_lp_templates_b(), _LP_OPTS_B, "u", NAMES, _LP_NEAR), copy=True))


# ---------------------------------------------------------------- 10. arith_extreme
class ExtremeSpec(Spec):
    """Four names (the options, a permutation) with distinct digit scores 1..9; the answer is the name with the highest
    (`mode="max"`) or lowest (`"min"`) score, at list position `aux`. Surface slot: the round number k. Answer twin: one
    score changes so that another name becomes the extreme."""
    nums = range(1, 10)

    def __init__(self, templates, options, mode, near_miss):
        self.templates, self.opts, self.mode, self.near_miss = templates, [o.strip() for o in options], mode, near_miss
        self.pick = max if mode == "max" else min
        _check_templates(templates, self.opts)

    def _extreme(self, d):
        return d.index(self.pick(d))

    def _changes(self, d, pos):
        """Single-score changes (i, v) that keep the digits distinct and move the extreme off `pos`."""
        out = []
        for i in range(4):
            for v in self.nums:
                d2 = list(d)
                d2[i] = v
                if v != d[i] and len(set(d2)) == 4 and self._extreme(d2) != pos:
                    out.append((i, v))
        return out

    def sample(self, rng, tpl, cls, split, aux=0):
        order = [None] * 4  # order[i] = class (option index) of the i-th listed name
        order[aux] = cls
        for i, c in zip([i for i in range(4) if i != aux], rng.permutation([c for c in range(4) if c != cls])):
            order[i] = int(c)
        while True:
            vals = sorted(int(v) for v in rng.choice(list(self.nums), 4, replace=False))
            top = vals.pop(-1 if self.mode == "max" else 0)
            rest = [int(v) for v in rng.permutation(vals)]
            d = rest[:aux] + [top] + rest[aux:]
            if self._changes(d, aux):
                break
        s = {f"n{i}": self.opts[order[i]] for i in range(4)} | {f"d{i}": d[i] for i in range(4)}
        return s | {"k": int(rng.choice(list(self.nums))), "pos": aux, "order": order}

    def ans_twin(self, rng, tpl, s, cls, split):
        d = [s[f"d{i}"] for i in range(4)]
        ch = self._changes(d, s["pos"])
        i, v = ch[rng.integers(len(ch))]
        d[i] = v
        return dict(s, **{f"d{i}": v}), s["order"][self._extreme(d)]

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, k=int(rng.choice([k for k in self.nums if k != s["k"]])))

    def solve(self, prompt):
        pairs = [(m[1], int(m[2])) for m in re.finditer(r"([A-Za-z]+)[:= ]+(\d)\b", prompt) if m[1] in self.opts]
        assert len(pairs) == 4, prompt
        return self.opts.index(self.pick(pairs, key=lambda p: p[1])[0])


def _ae_templates(heads, tails):
    styles = ("{n0} {d0}, {n1} {d1}, {n2} {d2} and {n3} {d3}", "{n0}: {d0}, {n1}: {d1}, {n2}: {d2}, {n3}: {d3}",
              "{n0}={d0}; {n1}={d1}; {n2}={d2}; {n3}={d3}")
    return [f"{h} {styles[(i + j) % 3]}. {t}" for i, h in enumerate(heads) for j, t in enumerate(tails)]


_AE_HEADS = ["Round {k} scores:", "Game {k} points:", "Match {k} results:", "Quiz {k} marks:"]
_AE_TPL_A = _ae_templates(_AE_HEADS, ["The highest score was by", "The top score belonged to", "The best score was by",
                                      "The most points went to"])
_AE_TPL_B = _ae_templates(_AE_HEADS, ["The lowest score was by", "The worst score was by", "The smallest score was by",
                                      "The fewest points went to"])
_AE_OPTS_A, _AE_OPTS_B = (" Tom", " Anna", " Ben", " Dan"), (" Mary", " John", " Kate", " Fred")
register(Task("arith_extreme_a", "arith_extreme", "arithmetic", _AE_OPTS_A,
              ExtremeSpec(_AE_TPL_A, _AE_OPTS_A, "max", _LP_NEAR), copy=True))
register(Task("arith_extreme_b", "arith_extreme", "arithmetic", _AE_OPTS_B,
              ExtremeSpec(_AE_TPL_B, _AE_OPTS_B, "min", _LP_NEAR), copy=True))
