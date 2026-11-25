"""Batch 7 (families 31-35): code_func, rule_table, hop_twohop, surf_length, fact_animal (two tasks each).

Pools below are literal lists, vetted with `tokcheck.vet_words` against all 17 tokenizers (7 distinct vocabularies): every
word is one token with its leading space everywhere. Class pools are ordered so that P0's split (every third word is a test
word) leaves the near-miss rules at chance on the held-out half.
"""
from __future__ import annotations

import re

from .b1 import LETTERS, NAMES, _check_templates
from .core import Spec, Task, WordSpec, register, words_of

MONTHS = ("January February March April May June July August September October November December").split()


def _perm_with(rng, cls, pos, k=4):
    """A permutation of range(k) (as a list) with value `cls` at index `pos`, the rest shuffled."""
    out = [None] * k
    out[pos] = cls
    for i, c in zip([i for i in range(k) if i != pos], rng.permutation([c for c in range(k) if c != cls])):
        out[i] = int(c)
    return out


# ---------------------------------------------------------------- 31. code_func
# Four one-line functions return the 4 option digits (permuted); `print(t())` calls one of them (a), or a wrapper function
# `w` that returns `t()` (b: indirection, a fifth one-line function). The answer digit is shown once, at the def of the
# called function, whose rank among the digits is `aux` (balanced). Surface slot: the name of an uncalled function.
class CodeFuncSpec(Spec):
    def __init__(self, templates, options, mode, near_miss):
        self.templates, self.opts, self.mode, self.near_miss = templates, [o.strip() for o in options], mode, near_miss
        _check_templates(templates, self.opts)

    def sample(self, rng, tpl, cls, split, aux=0):
        names = [str(x) for x in rng.choice(LETTERS, 5, replace=False)]
        order = _perm_with(rng, cls, aux)
        s = {f"v{i}": names[i] for i in range(4)} | {f"d{i}": self.opts[order[i]] for i in range(4)}
        return s | {"t": names[aux], "w": names[4], "order": order}

    def _v(self, s):
        return [s[f"v{i}"] for i in range(4)]

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if self._v(s)[i] != s["t"]]))
        return dict(s, t=self._v(s)[j]), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        v = self._v(s)
        j = int(rng.choice([i for i in range(4) if v[i] != s["t"]]))
        free = [w for w in LETTERS if w not in v and w != s["w"]]
        return dict(s, **{f"v{j}": str(rng.choice(free))})

    def solve(self, prompt):
        env, shown = {}, None
        for line in prompt.split("\n"):
            if m := re.fullmatch(r"def (\w+)\(\): return (\w+)(\(\))?", line.strip()):
                env[m[1]] = (m[2], bool(m[3]))
            elif m := re.fullmatch(r"print\((\w+)\(\)\)", line.strip()):
                shown = m[1]
        val, call = env[shown]
        while call:
            val, call = env[val]
        return self.opts.index(val)


def _cf_templates(call):
    out = []
    for hdr in ("", "# demo\n", "# trace\n", "# test\n", "# calls\n", "# code\n"):
        for gap in ("", "\n"):
            out.append(hdr + "".join(f"def {{v{i}}}(): return {{d{i}}}\n" for i in range(4)) + "{wrap}" + gap + call)
    return out


def _cf_fix(tpls, wrap):
    return [t.replace("{wrap}", wrap) for t in tpls]


_CF_NEAR = {"first_def": lambda s: s["order"][0], "last_def": lambda s: s["order"][3]}
_CF_OPTS_A, _CF_OPTS_B = ("3", "5", "7", "9"), ("2", "4", "6", "8")
register(Task("code_func_a", "code_func", "code tracing", _CF_OPTS_A,
              CodeFuncSpec(_cf_fix(_cf_templates("print({t}())\n"), ""), _CF_OPTS_A, "direct", _CF_NEAR), copy=True,
              notes="copy=True: the answer digit is one of the four returned digits shown in the prompt"))
register(Task("code_func_b", "code_func", "code tracing", _CF_OPTS_B,
              CodeFuncSpec(_cf_fix(_cf_templates("print({w}())\n"), "def {w}(): return {t}()\n"), _CF_OPTS_B, "wrapper",
                           _CF_NEAR), copy=True,
              notes="copy=True: the answer digit is one of the four returned digits shown in the prompt"))


# ---------------------------------------------------------------- 32. rule_table
# A 2 x 2 table written as two day lines (the day word is one of two month words): rows = two names (a) or two animals (b),
# columns = two months; the 4 cells hold the 4 option words (permuted). The query names one cell. Copy task: each option word
# is shown once, in cell order (month-major); the queried cell is `aux` (balanced), so the position rival is at chance.
# Answer twin: the query name or the query month changes (one token). Surface slot: the month of the column that is NOT queried
# (it appears once in the table; the answer ignores which month it is).
class TableSpec(Spec):
    def __init__(self, templates, options, keys, near_miss):
        self.templates, self.opts, self.keys, self.near_miss = templates, [o.strip() for o in options], keys, near_miss
        _check_templates(templates, self.opts + keys + MONTHS)

    def sample(self, rng, tpl, cls, split, aux=0):
        p = [str(x) for x in rng.choice(self.keys, 2, replace=False)]
        d = [str(x) for x in rng.choice(MONTHS, 2, replace=False)]
        order = _perm_with(rng, cls, aux)  # order[cell] = class; cell = 2 * month index + key index
        s = {"p0": p[0], "p1": p[1], "d0": d[0], "d1": d[1]} | {f"c{i}": self.opts[order[i]] for i in range(4)}
        qd, qp = divmod(aux, 2)
        return s | {"qday": d[qd], "qname": p[qp], "qd": qd, "qp": qp, "order": order}

    def ans_twin(self, rng, tpl, s, cls, split):
        if rng.integers(2):
            qp = 1 - s["qp"]
            return dict(s, qp=qp, qname=s[f"p{qp}"]), s["order"][2 * s["qd"] + qp]
        qd = 1 - s["qd"]
        return dict(s, qd=qd, qday=s[f"d{qd}"]), s["order"][2 * qd + s["qp"]]

    def surf_twin(self, rng, tpl, s, cls, split):
        k = 1 - s["qd"]
        return dict(s, **{f"d{k}": str(rng.choice([m for m in MONTHS if m not in (s["d0"], s["d1"])]))})

    def solve(self, prompt):
        ws = words_of(prompt)
        days = [w for w in ws if w in MONTHS]
        keys = [w for w in ws if w in self.keys]
        cols = [w for w in ws if w in self.opts]
        assert len(days) == 3 and len(keys) == 5 and len(cols) == 4, prompt
        return self.opts.index(cols[2 * days[:2].index(days[2]) + keys[:2].index(keys[4])])


def _rt_templates_a():
    return [f"{f} in {{d0}}, {{p0}} {v} {{c0}} and {{p1}} {v} {{c1}}. In {{d1}}, {{p0}} {v} {{c2}} and {{p1}} {v} {{c3}}. "
            f"In {{qday}}, {{qname}} {v}" for f in ("So,", "Well,", "Now,", "Look,") for v in ("wears", "picks", "likes")]


def _rt_templates_b():
    return [f"{f} in {{d0}} the {{p0}} {v} {{c0}} and the {{p1}} {v} {{c1}}. In {{d1}} the {{p0}} {v} {{c2}} and the {{p1}} "
            f"{v} {{c3}}. In {{qday}} the {{qname}} {v}" for f in ("So,", "Well,", "Now,", "Look,") for v in ("eats", "takes", "has")]


_RT_ANIMALS = "cat dog cow horse sheep bear mouse".split()
# first / last cell of the table. No rule that answers from the person's other-month cell: it is never right by construction.
_RT_NEAR = {"first_cell": lambda s: s["order"][0], "last_cell": lambda s: s["order"][3]}
_RT_OPTS_A, _RT_OPTS_B = (" red", " blue", " green", " white"), (" rice", " soup", " corn", " pie")
register(Task("rule_table_a", "rule_table", "rule application", _RT_OPTS_A,
              TableSpec(_rt_templates_a(), _RT_OPTS_A, NAMES, _RT_NEAR), copy=True,
              notes="2x2 table name x month -> colour; month words (weekdays are mostly not single tokens)"))
register(Task("rule_table_b", "rule_table", "rule application", _RT_OPTS_B,
              TableSpec(_rt_templates_b(), _RT_OPTS_B, _RT_ANIMALS, _RT_NEAR), copy=True,
              notes="2x2 table animal x month -> food"))


# ---------------------------------------------------------------- 33. hop_twohop
# Four people own 4 objects (list 1) and live in 4 cities (list 2, people in another order). a: the owner of an object ->
# the city (object -> person -> city); b: the person living in a city -> the object (city -> person -> object). Both hops are
# needed: the query word sits next to a name in one list, and the other list is in a different order, with three distractors.
# Copy: the answer words (cities in a, objects in b) are each shown once. The target person sits at rank `aux` among the answer
# words (a: in list 2; b: in list 1), balanced. Surface slot: the month (in the frame).
class TwoHopSpec(Spec):
    def __init__(self, templates, objs, cities, mode, near_miss):
        self.templates, self.objs, self.cities, self.mode, self.near_miss = templates, objs, cities, mode, near_miss
        self.opts = objs if mode == "b" else cities
        _check_templates(templates, objs + cities + NAMES + MONTHS)

    def sample(self, rng, tpl, cls, split, aux=0):
        n = [str(x) for x in rng.choice(NAMES, 4, replace=False)]
        ocls, ccls = [int(x) for x in rng.permutation(4)], [int(x) for x in rng.permutation(4)]  # per person (list 1 order)
        if self.mode == "a":
            i = int(rng.integers(4))
            ccls = _perm_with(rng, cls, i)  # the target's city has class cls
            pi = _perm_with(rng, i, aux)  # list 2 order: pi[j] = person at list 2 position j; target at rank aux
            q = self.objs[ocls[i]]
        else:
            i = aux
            ocls = _perm_with(rng, cls, i)
            pi = [int(x) for x in rng.permutation(4)]
            q = self.cities[ccls[i]]
        s = {f"n{k}": n[k] for k in range(4)} | {f"o{k}": self.objs[ocls[k]] for k in range(4)}
        s |= {f"m{j}": n[pi[j]] for j in range(4)} | {f"c{j}": self.cities[ccls[pi[j]]] for j in range(4)}
        return s | {"q": q, "mo": str(rng.choice(MONTHS)), "i": i, "pi": pi, "ocls": ocls, "ccls": ccls}

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([k for k in range(4) if k != s["i"]]))
        if self.mode == "a":
            return dict(s, q=self.objs[s["ocls"][j]], i=j), s["ccls"][j]
        return dict(s, q=self.cities[s["ccls"][j]], i=j), s["ocls"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, mo=str(rng.choice([m for m in MONTHS if m != s["mo"]])))

    def solve(self, prompt):
        ws = words_of(prompt)
        n, o, c = [w for w in ws if w in NAMES], [w for w in ws if w in self.objs], [w for w in ws if w in self.cities]
        if self.mode == "a":
            assert len(n) == 8 and len(o) == 5 and len(c) == 4, prompt
            return self.cities.index(c[n[4:].index(n[o[:4].index(o[4])])])
        assert len(n) == 8 and len(o) == 4 and len(c) == 5, prompt
        return self.objs.index(o[n[:4].index(n[4:][c[:4].index(c[4])])])


def _th_templates(mode):
    out = []
    for f in ("In {mo},", "By {mo},", "Last {mo},", "Next {mo},"):
        for v, r in (("has", "lives in"), ("owns", "stays in"), ("keeps", "works in")):
            l1 = f"{{n0}} {v} a {{o0}}, {{n1}} {v} a {{o1}}, {{n2}} {v} a {{o2}} and {{n3}} {v} a {{o3}}."
            l2 = f"{{m0}} {r} {{c0}}, {{m1}} {r} {{c1}}, {{m2}} {r} {{c2}} and {{m3}} {r} {{c3}}."
            q = f"The owner of the {{q}} {r}" if mode == "a" else f"The one who {r} {{q}} {v} a"
            out.append(f"{f} {l1} {l2} {q}")
    return out


_TH_OBJS = "cat dog fish bird".split()
_TH_CITIES = "Paris London Rome Berlin".split()
# aligned: the answer for the person who sits at the same ordinal position in the other list (a one-hop positional shortcut)
_TH_NEAR_A = {"first_city": lambda s: s["ccls"][s["pi"][0]], "last_city": lambda s: s["ccls"][s["pi"][3]],
              "aligned": lambda s: s["ccls"][s["pi"][s["i"]]]}
_TH_NEAR_B = {"first_object": lambda s: s["ocls"][0], "last_object": lambda s: s["ocls"][3],
              "aligned": lambda s: s["ocls"][s["pi"].index(s["i"])]}
register(Task("hop_twohop_a", "hop_twohop", "multi-hop composition", tuple(" " + c for c in _TH_CITIES),
              TwoHopSpec(_th_templates("a"), _TH_OBJS, _TH_CITIES, "a", _TH_NEAR_A), copy=True,
              notes="object -> owner -> city; month words in the frame (weekdays are mostly not single tokens)"))
register(Task("hop_twohop_b", "hop_twohop", "multi-hop composition", tuple(" " + o for o in _TH_OBJS),
              TwoHopSpec(_th_templates("b"), _TH_OBJS, _TH_CITIES, "b", _TH_NEAR_B), copy=True,
              notes="city -> resident -> object"))


# ---------------------------------------------------------------- 34. surf_length
# The class is the number of letters of the cue word (held out); every word is one token everywhere, so the count is not
# visible in the tokens. a: lengths 3-6, b: lengths 4-7. Surface slot: the name.
_SL_A = ["rat nap pen pod map cup bus dot hat lip nut bag sun".split(),
         "rain spot hill name mint plot jump shop chef rose hall hook deep".split(),
         "table trust start teach trick magic label adult motor point peace grand truth".split(),
         "weight finish shield planet leader launch minute danger player bottom cinema center camera".split()]
_SL_B = ["hall spot plot mint hook name deep rose jump hill shop chef rain".split(),
         "magic trick peace adult label point motor teach start trust table truth grand".split(),
         "leader cinema center bottom danger launch player weight finish camera minute shield planet".split(),
         "silence printer thought natural cluster machine fashion passion present history kingdom support diagram".split()]
_SL_TPL_A = [f"{i} {t}" for i in
             ["Then {n} wrote the word {w} on the board.", "Later {n} said the word {w} out loud.",
              "Soon {n} spelled the word {w} slowly.", "Today {n} found the word {w} in a text."]
             for t in ["The word has", "It has", "The word contains", "Its length is"]]
_SL_TPL_B = [f"{i} {t}" for i in
             ["Next {n} copied the word {w} onto a card.", "Once {n} heard the word {w} on the radio.",
              "Now {n} read the word {w} twice.", "Here {n} typed the word {w} into a file."]
             for t in ["The word is made of", "It is made of", "It consists of", "The word consists of"]]
_SL_NEAR = {"initial_bucket": lambda s: (ord(s["w"][0]) - 97) % 4, "last_bucket": lambda s: (ord(s["w"][-1]) - 97) % 4}
_SL_OPTS_A, _SL_OPTS_B = (" three", " four", " five", " six"), (" four", " five", " six", " seven")
for _s, _tpl, _pools, _opts in (("a", _SL_TPL_A, _SL_A, _SL_OPTS_A), ("b", _SL_TPL_B, _SL_B, _SL_OPTS_B)):
    _check_templates(_tpl, [w for p in _pools for w in p] + [o.strip() for o in _opts] + NAMES)
    register(Task(f"surf_length_{_s}", "surf_length", "surface form", _opts,
                  WordSpec(_tpl, "w", _pools, "n", NAMES, near_miss=_SL_NEAR)))


# ---------------------------------------------------------------- 35. fact_animal
# Only about 30 animal words are single tokens in all 7 vocabularies (no eagle, shark, bee, lion, pig, fox ...), too few for
# four kinds (bird, fish, insect, mammal) of 6+ words each. Both tasks therefore use the same four pools of single-token
# animals, grouped by where they live (water, sky, ground, field), with different frames, option order and pool order
# (so different held-out animals): a: spends its time in the ...; b: usually found in the ...
_FA_A = ["ray bass ide gar cod sole".split(),  # water
         "bat swift moth dove crow fly".split(),  # sky
         "ant rat mouse tick bug mole".split(),  # ground
         "cow cattle sow hen ox sheep horse ram".split()]  # field
_FA_B = ["hen cow ox sheep cattle ram sow horse".split(),  # field
         "mole tick ant mouse rat bug".split(),  # ground
         "gar ray cod bass sole ide".split(),  # water
         "crow dove bat moth swift fly".split()]  # sky
_FA_TPL_A = [f"{i} {t}" for i in
             ["Then {n} saw the {w} at the zoo.", "Later {n} drew the {w} on a page.",
              "Soon {n} read about the {w} in a book.", "Today {n} talked about the {w} in class."]
             for t in ["The animal spends most of its time in the", "It spends most of its time in the",
                       "The creature spends most of its days in the", "This animal spends its days in the"]]
_FA_TPL_B = [f"{i} {t}" for i in
             ["Next {n} watched the {w} for an hour.", "Once {n} took a photo of the {w}.",
              "Now {n} wrote a note about the {w}.", "Here {n} pointed at the {w} and smiled."]
             for t in ["You can usually find it in the", "It usually lives in the", "The animal is usually found in the",
                       "This animal usually lives in the"]]
_FA_NEAR = {"len_mod4": lambda s: len(s["w"]) % 4, "initial_bucket": lambda s: (ord(s["w"][0]) - 97) % 4}
_FA_OPTS_A, _FA_OPTS_B = (" water", " sky", " ground", " field"), (" field", " ground", " water", " sky")
for _s, _tpl, _pools, _opts in (("a", _FA_TPL_A, _FA_A, _FA_OPTS_A), ("b", _FA_TPL_B, _FA_B, _FA_OPTS_B)):
    _check_templates(_tpl, [w for p in _pools for w in p] + [o.strip() for o in _opts] + NAMES)
    register(Task(f"fact_animal_{_s}", "fact_animal", "fact lookup", _opts,
                  WordSpec(_tpl, "w", _pools, "n", NAMES, near_miss=_FA_NEAR),
                  notes="animal -> habitat; both tasks share four animal pools (see module note)"))
