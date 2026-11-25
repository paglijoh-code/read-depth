"""Batch 6 (families 26-30): rule_cycle, hop_transitive, fact_season, bind_swap, arith_story (two tasks each).

Pools below are literal lists, vetted with `tokcheck.vet_words` against all 17 tokenizers (7 distinct vocabularies): every
slot word is one token with its leading space everywhere. Class pools are ordered so that P0's split (every third word is a
test word) leaves the length / initial-letter near-miss rules at chance on the held-out half.
"""
from __future__ import annotations

import re

import numpy as np

from .b1 import NAMES, _check_templates
from .b4 import _fill
from .core import Spec, Task, WordSpec, register, words_of


def _first_mentions(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


# ---------------------------------------------------------------- 26. rule_cycle
# A rule moves four colours around a cycle ("red to blue, blue to green, ..."; the four clauses are listed in a random order,
# so the cycle has to be reassembled). A pawn starts at one colour; a: after one step, b: after two steps. The answer colour
# is mentioned in the rule; the clause listing is drawn so that its first-mention rank is `aux` (balanced), which keeps the
# position rival at chance. Answer twin: the start colour. Surface slot: the pawn's name `n` (it appears once).
class CycleSpec(Spec):
    def __init__(self, templates, options, steps, near_miss):
        self.templates, self.opts, self.steps, self.near_miss = templates, [o.strip() for o in options], steps, near_miss
        _check_templates(templates, self.opts + NAMES)

    def _advance(self, nxt, start):
        for _ in range(self.steps):
            start = nxt[start]
        return start

    def sample(self, rng, tpl, cls, split, aux=0):
        p = [int(x) for x in rng.permutation(4)]  # the cycle p[0] -> p[1] -> p[2] -> p[3] -> p[0]
        start = p[(p.index(cls) - self.steps) % 4]
        cl = [(p[j], p[(j + 1) % 4]) for j in range(4)]
        while True:
            perm = [int(x) for x in rng.permutation(4)]
            seen = _first_mentions([c for j in perm for c in cl[j]])
            if seen.index(cls) == aux:
                break
        cl = [cl[j] for j in perm]
        s = {f"a{i}": self.opts[cl[i][0]] for i in range(4)} | {f"b{i}": self.opts[cl[i][1]] for i in range(4)}
        return s | {"s": self.opts[start], "n": str(rng.choice(NAMES)), "cl": cl, "start": start}

    def _nxt(self, s):
        return dict(s["cl"])

    def ans_twin(self, rng, tpl, s, cls, split):
        s2 = int(rng.choice([c for c in range(4) if c != s["start"]]))
        return dict(s, s=self.opts[s2], start=s2), self._advance(self._nxt(s), s2)

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, n=str(rng.choice([w for w in NAMES if w != s["n"]])))

    def solve(self, prompt):
        nxt = {}
        for a, b in re.findall(r"(\w+) to (\w+)", prompt):
            if a in self.opts and b in self.opts:
                nxt[self.opts.index(a)] = self.opts.index(b)
        start = self.opts.index(re.search(r"(?:starts|begins|waits) at (\w+)\.", prompt)[1])
        assert len(nxt) == 4, prompt
        for _ in range(1 if "one step" in prompt else 2):
            start = nxt[start]
        return start


def _rcy_templates(steps_phrase):
    out = []
    for v in ("moves", "shifts", "turns"):
        rule = f"Rule: each step {v} {{a0}} to {{b0}}, {{a1}} to {{b1}}, {{a2}} to {{b2}} and {{a3}} to {{b3}}."
        for st in ("{n} starts at {s}.", "{n} begins at {s}.", "{n} waits at {s}."):
            for tl in ("it is at", "it ends at", "it sits at"):
                out.append(f"{rule} {st} After {steps_phrase} {tl}")
    return out


_RCY_NEAR = {"first_clause_dst": lambda s: s["cl"][0][1], "last_clause_dst": lambda s: s["cl"][3][1],
             "first_clause_src": lambda s: s["cl"][0][0]}
_RCY_OPTS_A, _RCY_OPTS_B = (" red", " blue", " green", " white"), (" black", " gold", " brown", " orange")
register(Task("rule_cycle_a", "rule_cycle", "rule application", _RCY_OPTS_A,
              CycleSpec(_rcy_templates("one step"), _RCY_OPTS_A, 1, _RCY_NEAR), copy=True,
              notes="four-colour cycle given as shuffled clauses; one step from the start colour; the answer colour is in the rule"))
register(Task("rule_cycle_b", "rule_cycle", "rule application", _RCY_OPTS_B,
              CycleSpec(_rcy_templates("two steps"), _RCY_OPTS_B, 2, _RCY_NEAR), copy=True,
              notes="as a, two steps (other colour set)"))


# ---------------------------------------------------------------- 27. hop_transitive
# Four fixed people (the 4 options) form a chain; four comparisons (the three adjacent ones and top vs bottom, so every name is
# compared exactly twice and a bag of words cannot tell the two ends from the middle) are listed in a random order and in either
# direction ("X is bigger than Y" / "Y is smaller than X"; "X beat Y" / "Y lost to X"). The question asks for one end of the
# chain (a: biggest / smallest, b: first / last to finish), so the chain has to be followed through the comparisons. The answer
# twin flips the question word; the answer's first-mention rank is `aux` (balanced, by rejection over order and directions).
# Surface slot: the place `p`. The places are not names; the 4 option names appear only in the comparisons.
class TransSpec(Spec):
    def __init__(self, templates, options, forms, qwords, places, near_miss):
        self.templates, self.opts, self.forms, self.qwords, self.places = templates, [o.strip() for o in options], forms, qwords, places
        self.near_miss = near_miss
        _check_templates(templates, self.opts + places + list(qwords))

    def _stmt(self, g, l, form):
        hi, lo = self.forms
        return f"{self.opts[g]} {hi} {self.opts[l]}." if form == 0 else f"{self.opts[l]} {lo} {self.opts[g]}."

    def _end(self, order, qk):
        return order[0] if qk == 0 else order[3]

    def sample(self, rng, tpl, cls, split, aux=0):
        qk = int(rng.integers(2))
        order = _fill(rng, 4, 0 if qk == 0 else 3, cls)  # top .. bottom of the chain
        pairs = [(order[i], order[i + 1]) for i in range(3)] + [(order[0], order[3])]
        while True:
            perm = [int(x) for x in rng.permutation(4)]
            forms = [int(x) for x in rng.integers(0, 2, 4)]
            sub = [(g, l) if f == 0 else (l, g) for (g, l), f in zip([pairs[j] for j in perm], forms)]
            men = _first_mentions([c for pr in sub for c in pr])
            if men.index(cls) == aux:
                break
        pl = [pairs[j] for j in perm]
        s = {f"st{i}": self._stmt(pl[i][0], pl[i][1], forms[i]) for i in range(4)}
        return s | {"p": str(rng.choice(self.places)), "q": self.qwords[qk], "qk": qk, "order": order, "pairs": pl,
                    "forms": forms, "mention": men, "subject": [pr[0] for pr in sub]}

    def ans_twin(self, rng, tpl, s, cls, split):
        qk = 1 - s["qk"]
        return dict(s, q=self.qwords[qk], qk=qk), self._end(s["order"], qk)

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, p=str(rng.choice([w for w in self.places if w != s["p"]])))

    def solve(self, prompt):
        hi, lo = self.forms
        gl = [(m[1], m[2]) for m in re.finditer(rf"(\w+) {hi} (\w+)\.", prompt)]
        gl += [(m[2], m[1]) for m in re.finditer(rf"(\w+) {lo} (\w+)\.", prompt)]
        assert len(gl) == 4, prompt
        top = [w for w in self.opts if w not in {l for _, l in gl}]
        bot = [w for w in self.opts if w not in {g for g, _ in gl}]
        assert len(top) == 1 and len(bot) == 1, prompt
        return self.opts.index(top[0] if self.qwords[0] in words_of(prompt) else bot[0])


def _ht_templates(intros, tails):
    return [f"{i} {{st0}} {{st1}} {{st2}} {{st3}} {t}" for i in intros for t in tails]


_HT_PLACES_A = "school park shop market farm garden beach library station office bank hotel".split()
_HT_PLACES_B = "church lake museum beach garden park farm station school hotel market shop".split()
_HT_TPL_A = _ht_templates(["At the {p},", "In the {p},", "Near the {p},", "By the {p},"],
                          ["So the {q} of the four is", "Thus the {q} of the four is", "Hence the {q} of the four is",
                           "Overall the {q} of the four is"])
_HT_TPL_B = _ht_templates(["At the {p} race,", "In the {p} race,", "Near the {p} race,", "By the {p} race,"],
                          ["So the {q} to finish was", "Thus the {q} to finish was", "Hence the {q} to finish was",
                           "Overall the {q} to finish was"])
_HT_NEAR = {"first_mentioned": lambda s: s["mention"][0], "last_mentioned": lambda s: s["mention"][3],
            "first_subject": lambda s: s["subject"][0]}
_HT_OPTS_A, _HT_OPTS_B = (" Tom", " Anna", " Ben", " Dan"), (" Mary", " John", " Kate", " Fred")
register(Task("hop_transitive_a", "hop_transitive", "multi-hop composition", _HT_OPTS_A,
              TransSpec(_HT_TPL_A, _HT_OPTS_A, (("is bigger than", "is smaller than")), ("biggest", "smallest"), _HT_PLACES_A,
                        _HT_NEAR), copy=True,
              notes="four comparisons on a chain (shuffled, either direction); biggest or smallest of the four "
                    "(tallest/shortest are not single tokens everywhere)"))
register(Task("hop_transitive_b", "hop_transitive", "multi-hop composition", _HT_OPTS_B,
              TransSpec(_HT_TPL_B, _HT_OPTS_B, (("beat", "lost to")), ("first", "last"), _HT_PLACES_B, _HT_NEAR), copy=True,
              notes="four race results on a chain (shuffled, either direction); first or last to finish"))


# ---------------------------------------------------------------- 28. fact_season
# a: month (name or abbreviation) -> season; b: cue word -> season. Held-out words. Only "Apr" and the lower-case month names
# are not single tokens everywhere, and "autumn" is not either, so the fourth season is `fall` and the spring pool uses
# `march` and `may` (lower case) as extra forms. Surface slot: the traveller's name.
_FS_A = ["December January Dec Jan Feb February".split(),  # winter
         "March April Mar march May may".split(),  # spring
         "June July Jun Jul Aug August".split(),  # summer
         "October Sep September Oct Nov November Sept".split()]  # fall
_FS_B = ["leaf hay leaves school apple fog brown nut corn".split(),  # fall
         "sand fan sun lake tan heat pool sea beach".split(),  # summer
         "ice cold cap coal coat fire ski snow soup".split(),  # winter
         "mud seed bud garden plant rain egg flower grass nest".split()]  # spring
_FS_STARTS = ["Then", "Later", "Once", "Soon"]
_FS_TPL_A = [f"{s} {{n}} {v} in {{w}}{e}" for s in _FS_STARTS for v, e in
             zip(["travelled", "left", "arrived", "stayed"],
                 [", so the season was", ", when the season was", ", and the season was", ", in the season of"])]
_FS_TPL_B = [f"{s} {{n}} {v} about the {{w}}{e}" for s in _FS_STARTS for v, e in
             zip(["talked", "wrote", "read", "spoke"],
                 [". That is the season of", ". It is the season of", ". This sounds like the season of",
                  ". It makes you think of the season of"])]
# No month name begins with a letter whose bucket matches its season, so a uses the last letter's bucket instead of the initial's.
_FS_NEAR_A = {"len_mod4": lambda s: len(s["w"]) % 4, "last_bucket": lambda s: (ord(s["w"][-1].lower()) - 97) % 4}
_FS_NEAR_B = {"len_mod4": lambda s: len(s["w"]) % 4, "initial_bucket": lambda s: (ord(s["w"][0]) - 97) % 4}
_FS_OPTS_A, _FS_OPTS_B = (" winter", " spring", " summer", " fall"), (" fall", " summer", " winter", " spring")
for _s, _tpl, _pools, _opts, _near in (("a", _FS_TPL_A, _FS_A, _FS_OPTS_A, _FS_NEAR_A),
                                       ("b", _FS_TPL_B, _FS_B, _FS_OPTS_B, _FS_NEAR_B)):
    _check_templates(_tpl, [w for p in _pools for w in p] + NAMES + [o.strip() for o in _opts])
    register(Task(f"fact_season_{_s}", "fact_season", "fact lookup", _opts,
                  WordSpec(_tpl, "w", _pools, "n", NAMES, near_miss=_near),
                  notes="a: month (name or abbreviation) -> season; b: cue word -> season; `fall` for autumn (see module note)"))


# ---------------------------------------------------------------- 29. bind_swap
# a: four people (random names) hold the 4 option objects (a permutation); two swap; ask what one holds now. The queried
# person is a swapper (answer: the partner's object) or, one time in four, a non-swapper (answer: their own object). The answer
# object is listed once, at list position `aux` (balanced), so the position rival is chance. Answer twin: the queried person.
# Surface slot: the name of a person who is neither swapping nor queried (it appears once).
class SwapSpec(Spec):
    def __init__(self, templates, options, near_miss):
        self.templates, self.opts, self.near_miss = templates, [o.strip() for o in options], near_miss
        _check_templates(templates, self.opts + NAMES)

    @staticmethod
    def _final(pair, j):
        return pair[1] if j == pair[0] else pair[0] if j == pair[1] else j

    def sample(self, rng, tpl, cls, split, aux=0):
        names = [str(x) for x in rng.choice(NAMES, 4, replace=False)]
        order = _fill(rng, 4, aux, cls)  # order[i] = class of the object listed for person i
        others = [i for i in range(4) if i != aux]
        if rng.random() < 0.25:  # the queried person does not swap
            qi, pair = aux, [int(x) for x in rng.choice(others, 2, replace=False)]
        else:
            qi = int(rng.choice(others))
            pair = [qi, aux] if rng.random() < 0.5 else [aux, qi]
        s = {f"n{i}": names[i] for i in range(4)} | {f"o{i}": self.opts[order[i]] for i in range(4)}
        return s | {"x": names[pair[0]], "y": names[pair[1]], "q": names[qi], "qi": qi, "pair": pair, "order": order}

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["qi"]]))
        return dict(s, q=s[f"n{j}"], qi=j), s["order"][self._final(s["pair"], j)]

    def surf_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["qi"] and i not in s["pair"]]))
        names = [s[f"n{i}"] for i in range(4)]
        return dict(s, **{f"n{j}": str(rng.choice([w for w in NAMES if w not in names]))})

    def solve(self, prompt):
        ws = words_of(prompt)
        names = [w for w in ws if w in NAMES]
        objs = [w for w in ws if w in self.opts]
        assert len(names) == 7 and len(objs) == 4, prompt
        listed, (x, y), q = names[:4], names[4:6], names[6]
        i = listed.index(q)
        return self.opts.index(objs[self._final([listed.index(x), listed.index(y)], i)])


def _bs_templates():
    out = []
    for x in ["So,", "Well,", "Now,", "Look,", "Okay,"]:
        for v in ("has", "owns", "keeps"):
            cl = [f"{{n{i}}} {v} a {{o{i}}}" for i in range(4)]
            for w in ("swap", "trade"):
                out.append(f"{x} " + ", ".join(cl[:3]) + " and " + cl[3] + f". Then {{x}} and {{y}} {w}. Now {{q}} {v} a")
    return out


_BS_NEAR = {"own_object": lambda s: s["order"][s["qi"]], "first_object": lambda s: s["order"][0],
            "last_object": lambda s: s["order"][3]}
_BS_OPTS_A = (" ball", " pen", " cup", " book")
register(Task("bind_swap_a", "bind_swap", "binding/reference", _BS_OPTS_A,
              SwapSpec(_bs_templates(), _BS_OPTS_A, _BS_NEAR), copy=True,
              notes="four people hold four objects, two swap, ask one; the answer object is in the prompt"))


# b: the 4 option names (a permutation, listed in order) each hold one object (drawn from a pool of 15). One person gives their
# object to another (one way), then the question names an object chosen uniformly: the given one or the receiver's own (answer:
# the receiver, half the time) or an uninvolved person's (answer: that person). The answer name's list position is `aux`
# (balanced); the giver and receiver are the two names mentioned twice, so a bag of words cannot tell which is the answer. Answer twin: the queried object (to one
# whose holder differs). Surface slot: an object that is not given, not queried and not held by the receiver.
class GiveSpec(Spec):
    def __init__(self, templates, options, objs, near_miss):
        self.templates, self.opts, self.objs, self.near_miss = templates, [o.strip() for o in options], objs, near_miss
        _check_templates(templates, self.opts + objs)

    def sample(self, rng, tpl, cls, split, aux=0):
        order = _fill(rng, 4, aux, cls)  # order[i] = class (option name) of the i-th listed person
        ob = [str(x) for x in rng.choice(self.objs, 4, replace=False)]
        others = [i for i in range(4) if i != aux]
        kind = int(rng.integers(4))  # queried object: 0 the given one, 1 the receiver's own, 2-3 an uninvolved person's
        if kind < 2:
            g, r = int(rng.choice(others)), aux
            qi = g if kind == 0 else r
        else:
            g, r = (int(x) for x in rng.choice(others, 2, replace=False))
            qi = aux
        s = {f"n{i}": self.opts[order[i]] for i in range(4)} | {f"o{i}": ob[i] for i in range(4)}
        return s | {"g": self.opts[order[g]], "r": self.opts[order[r]], "x": ob[g], "q": ob[qi], "qi": qi, "gi": g, "ri": r,
                    "order": order}

    @staticmethod
    def _holder(s, j):
        return s["ri"] if j == s["gi"] else j

    def ans_twin(self, rng, tpl, s, cls, split):
        here = self._holder(s, s["qi"])
        j = int(rng.choice([i for i in range(4) if i != s["qi"] and self._holder(s, i) != here]))
        return dict(s, q=s[f"o{j}"], qi=j), s["order"][self._holder(s, j)]

    def surf_twin(self, rng, tpl, s, cls, split):
        cand = [i for i in range(4) if i not in (s["gi"], s["ri"], s["qi"])] or [s["ri"]]
        j = int(rng.choice(cand))
        used = [s[f"o{i}"] for i in range(4)]
        return dict(s, **{f"o{j}": str(rng.choice([w for w in self.objs if w not in used]))})

    def solve(self, prompt):
        ws = words_of(prompt)
        names = [w for w in ws if w in self.opts]
        objs = [w for w in ws if w in self.objs]
        assert len(names) == 6 and len(objs) == 6, prompt
        listed, g, r = names[:4], names[4], names[5]
        ob, x, q = objs[:4], objs[4], objs[5]
        owner = listed[ob.index(q)]
        return self.opts.index(r if q == x else owner)


def _bg_templates():
    out = []
    for x in ["So,", "Well,", "Now,", "Look,", "Okay,"]:
        for v in ("has", "owns", "keeps"):
            cl = [f"{{n{i}}} {v} a {{o{i}}}" for i in range(4)]
            for gv, tail in (("gives", "Now the {q} is with"), ("hands", "So the {q} is now with"),
                             ("passes", "Thus the {q} belongs to")):
                out.append(f"{x} " + ", ".join(cl[:3]) + " and " + cl[3] + f". Then {{g}} {gv} the {{x}} to {{r}}. {tail}")
    return out


_BG_OBJS = "key coin hat ring lamp card bag box map watch doll ball pen cup book".split()
_BG_NEAR = {"first_name": lambda s: s["order"][0], "second_name": lambda s: s["order"][1],
            "last_name": lambda s: s["order"][3]}
_BG_OPTS_B = (" Tom", " Anna", " Ben", " Dan")
register(Task("bind_swap_b", "bind_swap", "binding/reference", _BG_OPTS_B,
              GiveSpec(_bg_templates(), _BG_OPTS_B, _BG_OBJS, _BG_NEAR), copy=True,
              notes="one-way give: four fixed names hold four objects, one gives to another; who has the queried object"))


# ---------------------------------------------------------------- 30. arith_story
# "Then Tom had seven coins. He lost three and found one. Now he has" -> number word (6..9). Operands are number words after
# spaces (never digits). Copy task: option words (one..four) can appear as operands, so the draw is steered by `aux` (balanced
# within each class): aux 0 and 3 use triples where no operand equals the answer, aux 1 and 2 triples where the answer word is
# the first / second distinct option word mentioned, which keeps the position rival at chance. a: lose then find, b: find then lose
# (same value, other order of operations). Answer twin: one
# operand (start, lost or found) changes so the result changes. Surface slot: the person's name (it appears once).
_NUMW = "one two three four five six seven eight nine".split()


class StorySpec(Spec):
    def __init__(self, templates, options, order, near_miss):
        self.templates, self.opts, self.order, self.near_miss = templates, [o.strip() for o in options], order, near_miss
        _check_templates(templates, _NUMW + NAMES)

    @staticmethod
    def _val(t):
        return t[0] - t[1] + t[2]

    def _ok(self, t):  # nobody loses more than they hold at that moment (b: the finds come first)
        return t[1] <= (t[0] if self.order == "a" else t[0] + t[2])

    def _triples(self, r):
        return [(a, l, f) for a in range(1, 10) for l in range(1, 10) for f in range(1, 10)
                if self._val((a, l, f)) == r and self._ok((a, l, f))]

    def _rank(self, t, r):
        """Index of the answer word among the distinct option words (1..4) mentioned, in prompt order; None if absent."""
        seq = (t[0], t[1], t[2]) if self.order == "a" else (t[0], t[2], t[1])
        seen = [v for v in dict.fromkeys(seq) if v <= 4]
        return seen.index(r) if r in seen else None

    def _slots(self, s, t):
        return dict(s, s=_NUMW[t[0] - 1], l=_NUMW[t[1] - 1], f=_NUMW[t[2] - 1], t=t)

    def sample(self, rng, tpl, cls, split, aux=0):
        want = {0: None, 1: 0, 2: 1, 3: None}[aux]
        ts = [x for x in self._triples(cls + 1) if self._rank(x, cls + 1) == want]
        ts = ts or [x for x in self._triples(cls + 1) if self._rank(x, cls + 1) == (aux > 0)]  # a, answer one: the found word is always `one`; split the fallback between ranks
        return self._slots({"n": str(rng.choice(NAMES))}, ts[int(rng.integers(len(ts)))])

    def ans_twin(self, rng, tpl, s, cls, split):
        cand = []
        for k in range(3):
            for v in range(1, 10):
                t = list(s["t"])
                t[k] = v
                if self._ok(t) and 1 <= self._val(t) <= 4 and self._val(t) != cls + 1:
                    cand.append(tuple(t))
        t = cand[int(rng.integers(len(cand)))]
        return self._slots(s, t), self._val(t) - 1

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, n=str(rng.choice([w for w in NAMES if w != s["n"]])))

    def solve(self, prompt):
        ws = words_of(prompt)
        start = _NUMW.index([w for w in ws if w in _NUMW][0]) + 1
        tot = start
        for v, w in re.findall(r"\b(lost|spent|away|found|earned|got) (\w+)", prompt):
            tot += (_NUMW.index(w) + 1) * (-1 if v in ("lost", "spent", "away") else 1)
        return tot - 1


def _as_templates(order):
    out = []
    for intro in ("Then", "Today", "Once", "Later"):
        for down, up in (("lost", "found"), ("spent", "earned"), ("gave away", "got")):
            body = f"He {down} {{l}} and {up} {{f}}." if order == "a" else f"He {up} {{f}} and then {down} {{l}}."
            for tail in ("Now he has", "So now he has"):
                out.append(f"{intro} {{n}} had {{s}} coins. {body} {tail}")
    return out


_AS_OPTS = (" one", " two", " three", " four")
_AS_NEAR = {"ignore_found": lambda s: min(max(s["t"][0] - s["t"][1] - 1, 0), 3),
            "ignore_lost": lambda s: min(max(s["t"][0] + s["t"][2] - 1, 0), 3),
            "start_operand": lambda s: min(s["t"][0] - 1, 3)}
register(Task("arith_story_a", "arith_story", "arithmetic", _AS_OPTS, StorySpec(_as_templates("a"), _AS_OPTS, "a", _AS_NEAR), copy=True,
              notes="start - lost + found, told in that order; number words throughout"))
register(Task("arith_story_b", "arith_story", "arithmetic", _AS_OPTS, StorySpec(_as_templates("b"), _AS_OPTS, "b", _AS_NEAR), copy=True,
              notes="start + found - lost, told in that order; number words throughout"))
