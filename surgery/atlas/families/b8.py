"""Batch 8 (families 36-40): bind_where, arith_price, code_update, surf_final, fact_place (two tasks each).

Pools below are literal lists, vetted with `tokcheck.vet_words` against all 17 tokenizers (7 distinct vocabularies): every
word is one token with its leading space everywhere. Class pools are ordered so that P0's split (every third word is a test
word) leaves the near-miss rules at chance on the held-out half.
"""
from __future__ import annotations

import re

from .b1 import NAMES, _check_templates
from .b7 import _perm_with
from .core import Spec, Task, WordSpec, register, words_of


# ---------------------------------------------------------------- 36. bind_where
# Four option words (names in a, chair colours in b) are each placed in a different room, listed in order (class order
# permuted); ask which one is in a given room. Copy task: the answer word is at list position `aux` (balanced), so the
# position rival is at chance. Answer twin: the queried room changes to another listed room. Surface slot: a room that is not
# queried (it appears once) becomes a room that is not in the prompt.
class WhereSpec(Spec):
    def __init__(self, templates, options, rooms, near_miss):
        self.templates, self.opts, self.rooms, self.near_miss = templates, [o.strip() for o in options], rooms, near_miss
        _check_templates(templates, self.opts + rooms)

    def sample(self, rng, tpl, cls, split, aux=0):
        r = [str(x) for x in rng.choice(self.rooms, 4, replace=False)]
        order = _perm_with(rng, cls, aux)  # order[j] = class of the option at list position j
        s = {f"r{j}": r[j] for j in range(4)} | {f"o{j}": self.opts[order[j]] for j in range(4)}
        return s | {"q": r[aux], "order": order}

    def _r(self, s):
        return [s[f"r{j}"] for j in range(4)]

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if self._r(s)[i] != s["q"]]))
        return dict(s, q=self._r(s)[j]), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if self._r(s)[i] != s["q"]]))
        free = [w for w in self.rooms if w not in self._r(s)]
        return dict(s, **{f"r{j}": str(rng.choice(free))})

    def solve(self, prompt):
        ws = words_of(prompt)
        rooms, objs = [w for w in ws if w in self.rooms], [w for w in ws if w in self.opts]
        assert len(rooms) == 5 and len(objs) == 4, prompt
        return self.opts.index(objs[rooms[:4].index(rooms[4])])


def _bw_templates(clause, last, verbs):
    out = []
    for x in ("So,", "Well,", "Now,", "Look,", "Okay,"):
        for v in verbs:
            out.append(f"{x} " + clause(v, "{o0}", "{r0}") + ", " + clause(v, "{o1}", "{r1}") + ", " +
                       clause(v, "{o2}", "{r2}") + " and " + clause(v, "{o3}", "{r3}") + ". " + last(v))
    return out


_BW_ROOMS_A = "kitchen garden hall office yard park shop school store".split()
_BW_ROOMS_B = "kitchen garden hall yard room park shop church lab farm".split()
_BW_NEAR = {"first_object": lambda s: s["order"][0], "last_object": lambda s: s["order"][3]}
_BW_OPTS_A, _BW_OPTS_B = (" Tom", " Anna", " Ben", " Jack"), (" red", " blue", " green", " white")
_BW_TPL_A = _bw_templates(lambda v, o, r: f"{o} {v} in the {r}", lambda v: "The one in the {q} is",
                          ["is", "sits", "waits"])
_BW_TPL_B = _bw_templates(lambda v, o, r: f"the {o} chair {v} in the {r}", lambda v: "The chair in the {q} is",
                          ["is", "stands", "sits"])
register(Task("bind_where_a", "bind_where", "binding/reference", _BW_OPTS_A,
              WhereSpec(_BW_TPL_A, _BW_OPTS_A, _BW_ROOMS_A, _BW_NEAR), copy=True,
              notes="4 names in 4 rooms, ask who is in a room"))
register(Task("bind_where_b", "bind_where", "binding/reference", _BW_OPTS_B,
              WhereSpec(_BW_TPL_B, _BW_OPTS_B, _BW_ROOMS_B, _BW_NEAR), copy=True,
              notes="4 coloured chairs in 4 rooms, ask the colour of the chair in a room"))


# ---------------------------------------------------------------- 37. arith_price
# Five items with prices in dollars (digits 1-9); a: the two bought items' total (6-9 = the options); b: how much more the
# first of two compared items costs than the second (1-4 = the options). The option digits can occur as prices, so the answer
# digit may be shown in the prompt: copy task. With few prices the digit counts alone reveal the class (a bag of words reached
# 0.4 with four prices from a narrow range), so there are five prices over the full digit range. Surface slot: the price of an
# item that the query does not mention. Digits follow `$` everywhere.
_PR_ITEMS = "pen cup hat bag book ball box key coin pot lamp map card ring bell fork plate".split()
_PR_K = 5


class PriceSpec(Spec):
    def __init__(self, templates, mode, near_miss):
        self.templates, self.mode, self.near_miss = templates, mode, near_miss
        self.base = 6 if mode == "sum" else 1  # first option digit
        _check_templates(templates, _PR_ITEMS)

    def _val(self, px, py):
        return px + py if self.mode == "sum" else px - py

    def sample(self, rng, tpl, cls, split, aux=0):
        it = [str(x) for x in rng.choice(_PR_ITEMS, _PR_K, replace=False)]
        # rejection sampling: draw all prices uniformly, keep the draw if some ordered pair has the class (keeps the price
        # digits, which a bag of words sees, nearly independent of the class)
        while True:
            p = [int(rng.integers(1, 10)) for _ in range(_PR_K)]
            ok = [(a, b) for a in range(_PR_K) for b in range(_PR_K)
                  if a != b and self._val(p[a], p[b]) - self.base == cls]
            if ok:
                break
        bi = list(ok[int(rng.integers(len(ok)))])  # bought / compared item indices (x, then y)
        s = {f"i{j}": it[j] for j in range(_PR_K)} | {f"p{j}": p[j] for j in range(_PR_K)}
        return s | {"x": it[bi[0]], "y": it[bi[1]], "bi": bi, "n": str(rng.choice(NAMES))}

    def _p(self, s):
        return [s[f"p{j}"] for j in range(_PR_K)]

    def ans_twin(self, rng, tpl, s, cls, split):
        p, opts = self._p(s), []
        for k in (0, 1):
            for v in range(1, 10):
                p2 = list(p)
                p2[s["bi"][k]] = v
                c2 = self._val(p2[s["bi"][0]], p2[s["bi"][1]]) - self.base
                if v != p[s["bi"][k]] and 0 <= c2 < 4 and c2 != cls:
                    opts.append((dict(s, **{f"p{s['bi'][k]}": v}), c2))
        return opts[int(rng.integers(len(opts)))]

    def surf_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(_PR_K) if i not in s["bi"]]))
        return dict(s, **{f"p{j}": int(rng.choice([v for v in range(1, 10) if v != s[f"p{j}"]]))})

    def solve(self, prompt):
        ws = [w for w in words_of(prompt) if w in _PR_ITEMS]
        price = dict(zip(ws[:_PR_K], map(int, re.findall(r"\$(\d)", prompt))))
        return self._val(price[ws[_PR_K]], price[ws[_PR_K + 1]]) - self.base


def _pr_list(v1, v2):
    a = ", ".join(f"a {{i{j}}} {v1} ${{p{j}}}" for j in range(_PR_K - 1)) + f" and a {{i{_PR_K - 1}}} {v1} ${{p{_PR_K - 1}}}."
    b = ", ".join(f"the {{i{j}}} {v2} ${{p{j}}}" for j in range(_PR_K - 1)) + \
        f" and the {{i{_PR_K - 1}}} {v2} ${{p{_PR_K - 1}}}."
    return "A " + a[2:], "The " + b[4:]


_PR_LIST = [*_pr_list("costs", "costs"), *_pr_list("sells for", "sells for"), *_pr_list("is", "is")]
_PR_TPL_A = [c + " " + q for c in _PR_LIST for q in ("{n} buys a {x} and a {y} for $", "Then {n} buys a {x} and a {y} for $")]
_PR_TPL_B = [c + " " + q for c in _PR_LIST for q in ("{n} compares the {x} with the {y}. The {x} costs more, by $",
                                                      "Then {n} compares the {x} with the {y}. The {x} costs more, by $")]
# partial computations: the first bought / first compared price alone, and sums or gaps of fixed listed positions
_PR_NEAR_A = {"first_bought": lambda s: min(max(s[f"p{s['bi'][0]}"] - 1, 0), 3),
              "first_two_listed": lambda s: min(max(s["p0"] + s["p1"] - 6, 0), 3),
              "last_two_listed": lambda s: min(max(s["p3"] + s["p4"] - 6, 0), 3)}
_PR_NEAR_B = {"first_listed_gap": lambda s: min(max(s["p0"] - s["p1"] - 1, 0), 3),
              "x_price": lambda s: min(max(s[f"p{s['bi'][0]}"] - 5, 0), 3),
              "last_listed_gap": lambda s: min(max(s["p4"] - s["p3"] - 1, 0), 3)}
register(Task("arith_price_a", "arith_price", "arithmetic", ("6", "7", "8", "9"),
              PriceSpec(_PR_TPL_A, "sum", _PR_NEAR_A), copy=True,
              notes="total of two bought items out of five; copy=True: an option digit can be a listed price"))
register(Task("arith_price_b", "arith_price", "arithmetic", ("1", "2", "3", "4"),
              PriceSpec(_PR_TPL_B, "diff", _PR_NEAR_B), copy=True,
              notes="price gap of two items out of five; copy=True: an option digit can be a listed price"))


# ---------------------------------------------------------------- 38. code_update
# Three variables are set, each updated once (a: `v = v + d`, final 6-9 = the options; b: `v = v - d`, final 1-4);
# `print(q)` asks for one of them. Digits 1-9 are used everywhere, so the option digits can occur in the prompt (copy task); with
# two variables the digit counts alone revealed the class to a bag of words (0.44), with three they reveal much less. Answer
# twin: the printed variable's start or update changes, or another variable is printed. Surface slot: the update of a variable
# that is not printed.
_CU_K = 3


class UpdateSpec(Spec):
    def __init__(self, templates, mode, near_miss):
        self.templates, self.mode, self.near_miss = templates, mode, near_miss
        self.base = 6 if mode == "add" else 1
        _check_templates(templates, [])

    def _val(self, i, d):
        return i + d if self.mode == "add" else i - d

    def sample(self, rng, tpl, cls, split, aux=0):
        v = [str(x) for x in rng.choice("x y a b m n p q r s".split(), _CU_K, replace=False)]
        while True:  # rejection sampling, as in PriceSpec.sample
            i = [int(rng.integers(1, 10)) for _ in range(_CU_K)]
            d = [int(rng.integers(1, 10)) for _ in range(_CU_K)]
            ok = [k for k in range(_CU_K) if self._val(i[k], d[k]) - self.base == cls]
            if ok:
                break
        q = ok[int(rng.integers(len(ok)))]
        return {f"v{k}": v[k] for k in range(_CU_K)} | {f"i{k}": i[k] for k in range(_CU_K)} | \
               {f"d{k}": d[k] for k in range(_CU_K)} | {"q": v[q], "qi": q}

    def _cls(self, s, q):
        return self._val(s[f"i{q}"], s[f"d{q}"]) - self.base

    def ans_twin(self, rng, tpl, s, cls, split):
        q, opts = s["qi"], []
        for key in (f"i{q}", f"d{q}"):
            for v in range(1, 10):
                s2 = dict(s, **{key: v})
                c2 = self._cls(s2, q)
                if v != s[key] and 0 <= c2 < 4 and c2 != cls:
                    opts.append((s2, c2))
        for o in range(_CU_K):
            if o != q and 0 <= self._cls(s, o) < 4 and self._cls(s, o) != cls:
                opts.append((dict(s, q=s[f"v{o}"], qi=o), self._cls(s, o)))
        return opts[int(rng.integers(len(opts)))]

    def surf_twin(self, rng, tpl, s, cls, split):
        o = int(rng.choice([k for k in range(_CU_K) if k != s["qi"]]))
        return dict(s, **{f"d{o}": int(rng.choice([v for v in range(1, 10) if v != s[f"d{o}"]]))})

    def solve(self, prompt):
        env, shown = {}, None
        for line in prompt.split("\n"):
            line = re.sub(r"^(>>> )", "", line.strip())
            if m := re.fullmatch(r"(\w+) = (\d)", line):
                env[m[1]] = int(m[2])
            elif m := re.fullmatch(r"(\w+) = (\w+) ([+-]) (\d)", line):
                env[m[1]] = env[m[2]] + (int(m[4]) if m[3] == "+" else -int(m[4]))
            elif m := re.fullmatch(r"print\((\w+)\)", line):
                shown = env[m[1]]
        return shown - self.base


def _cu_templates(sign):
    out = []
    for p in ("", ">>> "):
        for h in ("", "# demo\n", "# trace\n", "# test\n"):
            sets = "".join(f"{p}{{v{k}}} = {{i{k}}}\n" for k in range(_CU_K))
            ups = "".join(f"{p}{{v{k}}} = {{v{k}}} {sign} {{d{k}}}\n" for k in range(_CU_K))
            mix = "".join(f"{p}{{v{k}}} = {{i{k}}}\n{p}{{v{k}}} = {{v{k}}} {sign} {{d{k}}}\n" for k in range(_CU_K))
            for body in (sets + ups, mix):
                out.append(h + body + f"{p}print({{q}})\n")
    return out


# the next variable's own result, and the start value or the update alone of the printed variable
_CU_NEAR_A = {"next_var": lambda s: min(max(s[f"i{(s['qi'] + 1) % 3}"] + s[f"d{(s['qi'] + 1) % 3}"] - 6, 0), 3),
              "start_only": lambda s: min(max(s[f"i{s['qi']}"] - 1, 0), 3),
              "update_only": lambda s: min(max(s[f"d{s['qi']}"] - 1, 0), 3)}
_CU_NEAR_B = {"next_var": lambda s: min(max(s[f"i{(s['qi'] + 1) % 3}"] - s[f"d{(s['qi'] + 1) % 3}"] - 1, 0), 3),
              "start_only": lambda s: min(max(s[f"i{s['qi']}"] - 5, 0), 3),
              "update_only": lambda s: min(max(9 - s[f"d{s['qi']}"] - 3, 0), 3)}
register(Task("code_update_a", "code_update", "code tracing", ("6", "7", "8", "9"),
              UpdateSpec(_cu_templates("+"), "add", _CU_NEAR_A), copy=True,
              notes="v = v + d on three variables; copy=True: an option digit can occur in the prompt"))
register(Task("code_update_b", "code_update", "code tracing", ("1", "2", "3", "4"),
              UpdateSpec(_cu_templates("-"), "sub", _CU_NEAR_B), copy=True,
              notes="v = v - d on three variables; copy=True: an option digit can occur in the prompt"))


# ---------------------------------------------------------------- 39. surf_final
# The class is the last letter of the cue word (held out); every word is one token everywhere, so the last letter is not
# visible in the tokens. a: d t n k, b: l r s e. Surface slot: the name.
_SF_A = ["wood sound road food field hand bread bird wind cloud gold friend".split(),  # d
         "plant heart night coat test root foot point sort paint boat post".split(),  # t
         "town bean brain chain pain rain coin crown moon corn clean train".split(),  # n
         "check lock bank brick book clock deck cook dock black back hook".split()]  # k
_SF_B = ["hill rail soil coal ball pool tall bell steel sail wall doll".split(),  # l
         "power four star jar motor car floor door chair hair sister hour".split(),  # r
         "miss gas loss bus kiss glass class dress bass pass grass cross".split(),  # s
         "gate page hope name fire nose cave face game home horse line".split()]  # e
_SF_TPL_A = [f"{i} {t}" for i in
             ["Then {n} wrote the word {w} on the board.", "Later {n} said the word {w} out loud.",
              "Soon {n} spelled the word {w} slowly.", "Today {n} found the word {w} in a text."]
             for t in ["The word ends with the letter", "Its last letter is", "The last letter of the word is",
                       "The final letter is"]]
_SF_TPL_B = [f"{i} {t}" for i in
             ["Next {n} copied the word {w} onto a card.", "Once {n} heard the word {w} on the radio.",
              "Now {n} read the word {w} twice.", "Here {n} typed the word {w} into a file."]
             for t in ["It ends with the letter", "The word finishes with the letter", "Its closing letter is",
                       "The last letter is"]]
_SF_NEAR = {"len_mod4": lambda s: len(s["w"]) % 4, "initial_bucket": lambda s: (ord(s["w"][0]) - 97) % 4}
_SF_OPTS_A, _SF_OPTS_B = (" d", " t", " n", " k"), (" l", " r", " s", " e")
for _s, _tpl, _pools, _opts in (("a", _SF_TPL_A, _SF_A, _SF_OPTS_A), ("b", _SF_TPL_B, _SF_B, _SF_OPTS_B)):
    _check_templates(_tpl, [w for p in _pools for w in p] + NAMES)
    register(Task(f"surf_final_{_s}", "surf_final", "surface form", _opts,
                  WordSpec(_tpl, "w", _pools, "n", NAMES, near_miss=_SF_NEAR)))


# ---------------------------------------------------------------- 40. fact_place
# a: job -> workplace; b: object -> room. Only a few job words are single tokens in all 7 vocabularies (no doctor-nurse-cook
# set big enough for a hospital or a kitchen class), so a uses school (teaching words), castle (court words), office and
# studio. Held-out words. Surface slot: the name.
_FP_A = ["coach principal professor teacher fellow student".split(),  # school
         "lord servant king lady prince queen".split(),  # castle
         "journalist editor engineer programmer manager secretary designer lawyer".split(),  # office
         "model actor artist poet singer painter director composer".split()]  # studio
_FP_B = ["pie plate bread sink milk soup cup pot pan rice egg fork".split(),  # kitchen
         "leaf bush tree soil flower seed plant grass".split(),  # garden
         "folder computer phone keyboard paper laptop printer file".split(),  # office
         "coat rug hook boot mat door".split()]  # hall
_FP_TPL_A = [f"{i} {t}" for i in
             ["Then {n} met the {w} in town.", "Later {n} spoke to the {w} on the phone.",
              "Soon {n} waved at the {w} in the street.", "Today {n} read a story about the {w}."]
             for t in ["That person usually spends the day at the", "This person spends most days at the",
                       "The person works every day at the", "That person is usually found at the"]]
_FP_TPL_B = [f"{i} {t}" for i in
             ["Next {n} looked for the {w} for an hour.", "Once {n} took a photo of the {w}.",
              "Now {n} wrote a note about the {w}.", "Here {n} pointed at the {w} and smiled."]
             for t in ["You can usually find it in the", "It is usually kept in the",
                       "The thing is normally in the", "People keep it in the"]]
_FP_NEAR = {"len_mod4": lambda s: len(s["w"]) % 4, "initial_bucket": lambda s: (ord(s["w"][0]) - 97) % 4}
_FP_OPTS_A, _FP_OPTS_B = (" school", " castle", " office", " studio"), (" kitchen", " garden", " office", " hall")
for _s, _tpl, _pools, _opts in (("a", _FP_TPL_A, _FP_A, _FP_OPTS_A), ("b", _FP_TPL_B, _FP_B, _FP_OPTS_B)):
    _check_templates(_tpl, [w for p in _pools for w in p] + [o.strip() for o in _opts] + NAMES)
    register(Task(f"fact_place_{_s}", "fact_place", "fact lookup", _opts,
                  WordSpec(_tpl, "w", _pools, "n", NAMES, near_miss=_FP_NEAR),
                  notes="a: job -> workplace; b: object -> room (see module note)"))
