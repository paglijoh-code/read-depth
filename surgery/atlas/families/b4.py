"""Batch 4 (families 16-20): bind_transfer, arith_boundvar, code_listpos, rule_range, hop_kin (two tasks each).

Pools below are literal lists, vetted with `tokcheck.vet_words` against all 17 tokenizers (7 distinct vocabularies): every
word is one token with its leading space everywhere. None of these tasks has a held-out class word pool: the answer is
computed from the prompt (a name or label chosen by a role, a sum or difference, a list position), so there is no word to
hold out; the class assignment is permuted per item instead.
"""
from __future__ import annotations

import re
from itertools import permutations

import numpy as np

from .b1 import NAMES, _check_templates
from .b2 import _LP_PLACES
from .core import Spec, Task, register, words_of


def _fill(rng, n, pos, cls):
    """A list of n classes (a permutation of 0..n-1) with `cls` at index `pos`, the rest shuffled."""
    out = [None] * n
    out[pos] = cls
    for i, c in zip([i for i in range(n) if i != pos], rng.permutation([c for c in range(n) if c != cls])):
        out[i] = int(c)
    return out


# ---------------------------------------------------------------- 16. bind_transfer
# Two transfers of two objects among four people (the 4 option names). a: ask who received an object; b: ask who gave it.
# Each clause is either giver-first ("X gave the O to Y") or recipient-first ("Y got the O from X"); the form is fixed by the
# template and the four people are slots n0..n3 in mention order, so the mention rank of the answer depends on the template
# form and on which of the two transfers is queried (`aux` % 2). Templates are picked so that every form pair appears
# among the test templates (and so ranks 1-4 are equally likely), which keeps the position rival at chance.
# Surface slot: the object of the transfer the query does not ask about.
class TransferSpec(Spec):
    def __init__(self, templates, meta, options, objs, mode, near_miss):
        self.templates, self.meta, self.opts, self.objs, self.mode = templates, meta, [o.strip() for o in options], objs, mode
        self.near_miss = near_miss
        _check_templates(templates, self.opts + objs)

    def pick_templates(self, rng, n_test_tpl):
        by = {}
        for t in self.templates:
            by.setdefault(self.meta[t], []).append(t)
        assert len(by) == n_test_tpl
        test = [g[int(rng.integers(len(g)))] for g in by.values()]
        train = [t for t in self.templates if t not in test]
        rng.shuffle(train)
        return train, test

    def _pos(self, tpl, qs):
        """Mention index of the answer person when transfer `qs` is queried."""
        giver_first = self.meta[tpl][qs] == 0
        recipient = (self.mode == "recv")
        return 2 * qs + (int(giver_first) if recipient else int(not giver_first))

    def sample(self, rng, tpl, cls, split, aux=0):
        qs = aux % 2
        order = _fill(rng, 4, self._pos(tpl, qs), cls)
        x = [str(w) for w in rng.choice(self.objs, 2, replace=False)]
        s = {f"n{i}": self.opts[order[i]] for i in range(4)} | {"x0": x[0], "x1": x[1], "q": x[qs]}
        return s | {"qs": qs, "order": order}

    def ans_twin(self, rng, tpl, s, cls, split):
        qs = 1 - s["qs"]
        return dict(s, q=s[f"x{qs}"], qs=qs), s["order"][self._pos(tpl, qs)]

    def surf_twin(self, rng, tpl, s, cls, split):
        free = [w for w in self.objs if w not in (s["x0"], s["x1"])]
        return dict(s, **{f"x{1 - s['qs']}": str(rng.choice(free))})

    def solve(self, prompt):
        who = {}
        for m in re.finditer(r"(\w+) (?:gave|handed) the (\w+) to (\w+)", prompt):
            who[m[2]] = (m[1], m[3])  # object -> (giver, recipient)
        for m in re.finditer(r"(\w+) (?:got|received) the (\w+) from (\w+)", prompt):
            who[m[2]] = (m[3], m[1])
        q = re.findall(r"\b[Tt]he (\w+)", prompt)[-1]
        assert len(who) == 2 and q in who, prompt
        return self.opts.index(who[q][1 if self.mode == "recv" else 0])


def _bt_templates(tails):
    forms = {0: ("{a} {v} the {o} to {b}", ("gave", "handed")), 1: ("{b} {w} the {o} from {a}", ("got", "received"))}
    out, meta = [], {}
    for intro in ("So,", "Well,", "Now,", "Look,"):
        for f0 in (0, 1):
            for f1 in (0, 1):
                for vi in (0, 1):
                    for sep in (", and ", "; "):
                        cl = []
                        for k, f in enumerate((f0, f1)):
                            tpl, verbs = forms[f]
                            first, second = (f"{{n{2 * k}}}", f"{{n{2 * k + 1}}}")
                            # form 0: first = giver; form 1: first = recipient
                            cl.append(tpl.format(a=first if f == 0 else second, b=second if f == 0 else first,
                                                 v=verbs[vi], w=verbs[vi], o=f"{{x{k}}}"))
                        for tail in tails:
                            t = f"{intro} {cl[0]}{sep}{cl[1]}. {tail}"
                            out.append(t)
                            meta[t] = (f0, f1)
    return out, meta


_BT_OBJS = "ball pen cup book key hat coin ring lamp card bag box".split()
_BT_OPTS_A, _BT_OPTS_B = (" Tom", " Anna", " Ben", " Dan"), (" Mary", " John", " Kate", " Fred")
_BT_NEAR = {"first_mentioned": lambda s: s["order"][0], "second_mentioned": lambda s: s["order"][1],
            "last_mentioned": lambda s: s["order"][3]}
_BT_TPL_A, _BT_META_A = _bt_templates(["The {q} went to", "Who got the {q}? It was"])
_BT_TPL_B, _BT_META_B = _bt_templates(["The {q} came from", "Who gave the {q}? It was"])
register(Task("bind_transfer_a", "bind_transfer", "binding/reference", _BT_OPTS_A,
              TransferSpec(_BT_TPL_A, _BT_META_A, _BT_OPTS_A, _BT_OBJS, "recv", _BT_NEAR), copy=True,
              notes="two transfers among four people; ask the recipient (the answer name is in the prompt)"))
register(Task("bind_transfer_b", "bind_transfer", "binding/reference", _BT_OPTS_B,
              TransferSpec(_BT_TPL_B, _BT_META_B, _BT_OPTS_B, _BT_OBJS, "give", _BT_NEAR), copy=True,
              notes="two transfers among four people; ask the giver (the answer name is in the prompt)"))


# ---------------------------------------------------------------- 17. arith_boundvar
# Six variables bound to digits ("x=3, y=5, ..."; `=` then digit, so no digit follows a space); the query combines two of
# them (four are ignored). a: the sum (answers 5..8); b: the difference (1..4, first operand larger). Surface slot: the value of a variable the
# query ignores. Not a copy task: the answer is computed (it may coincide with a shown digit by chance only).
class BoundVarSpec(Spec):
    nums = range(1, 10)

    def __init__(self, templates, options, mode, near_miss):
        self.templates, self.opts, self.mode, self.near_miss = templates, [o.strip() for o in options], mode, near_miss
        self.lo = int(self.opts[0])
        _check_templates(templates, [])

    def val(self, u, v):
        return u + v if self.mode == "sum" else u - v

    def _names(self, tpl):
        return [m[0] for m in re.findall(r"(\w)=\{v(\d)\}", tpl)]

    def _pairs(self, c):
        return [(u, v) for u in self.nums for v in self.nums if self.val(u, v) - self.lo == c]

    def sample(self, rng, tpl, cls, split, aux=0):
        names = self._names(tpl)
        ip, ir = (int(i) for i in rng.choice(6, 2, replace=False))
        pr = self._pairs(cls)
        u, v = pr[rng.integers(len(pr))]
        vals = [int(rng.choice(list(self.nums))) for _ in range(6)]
        vals[ip], vals[ir] = u, v
        s = {f"v{i}": vals[i] for i in range(6)} | {"p": names[ip], "r": names[ir]}
        return s | {"ip": ip, "ir": ir}

    def ans_twin(self, rng, tpl, s, cls, split):
        out = []
        for key in ("ip", "ir"):
            for v in self.nums:
                s2 = dict(s, **{f"v{s[key]}": v})
                c2 = self.val(s2[f"v{s['ip']}"], s2[f"v{s['ir']}"]) - self.lo
                if v != s[f"v{s[key]}"] and 0 <= c2 < 4 and c2 != cls:
                    out.append((s2, c2))
        return out[rng.integers(len(out))]

    def surf_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(6) if i not in (s["ip"], s["ir"])]))
        return dict(s, **{f"v{j}": int(rng.choice([v for v in self.nums if v != s[f"v{j}"]]))})

    def solve(self, prompt):
        vals = dict(re.findall(r"(\w)=(\d)", prompt))
        m = re.search(r"(\w) ([+-]) (\w) =$", prompt)
        u, v = int(vals[m[1]]), int(vals[m[3]])
        return (u + v if m[2] == "+" else u - v) - self.lo


def _bv_templates(op):
    out = []
    for intro in ("Let", "Suppose", "Say", "Take"):
        for names in ("xyzwuv", "abcdef"):
            for sep in ("sep", "and"):
                cl = [f"{n}={{v{i}}}" for i, n in enumerate(names)]
                body = ", ".join(cl[:5]) + " and " + cl[5] if sep == "and" else "; ".join(cl)
                for tail in ("Then", "So", "Now"):
                    out.append(f"{intro} {body}. {tail} {{p}} {op} {{r}} =")
    return out


_BV_OPTS_A, _BV_OPTS_B = ("5", "6", "7", "8"), ("1", "2", "3", "4")
_BV_NEAR_A = {"first_pair": lambda s: min(max(s["v0"] + s["v1"] - 5, 0), 3),
              "total_mod4": lambda s: sum(s[f"v{i}"] for i in range(6)) % 4}
_BV_NEAR_B = {"first_pair": lambda s: min(max(abs(s["v0"] - s["v1"]) - 1, 0), 3),
              "total_mod4": lambda s: sum(s[f"v{i}"] for i in range(6)) % 4}
register(Task("arith_boundvar_a", "arith_boundvar", "arithmetic", _BV_OPTS_A,
              BoundVarSpec(_bv_templates("+"), _BV_OPTS_A, "sum", _BV_NEAR_A)))
register(Task("arith_boundvar_b", "arith_boundvar", "arithmetic", _BV_OPTS_B,
              BoundVarSpec(_bv_templates("-"), _BV_OPTS_B, "diff", _BV_NEAR_B)))


# ---------------------------------------------------------------- 18. code_listpos
# a: `x = [6, 8, 5, 9]` holds the 4 option digits (a permutation); `print(x[2])` reads index 2, so the answer is the digit at
#    the queried index, whose rank (= index) is `aux`, balanced. Surface slot: the case word in the header comment.
# b: `x = [7, 9, 4, 6]`, `print(x.index(9))` -> the index "0".."3". List values are drawn from 4..9, so no option digit is in
#    the prompt (not a copy task). Surface slot: the value of a list element the query does not ask about (changed to a digit
#    that is not in the list, so the query value stays unique).
_CL_CASES = "one two three four five six seven eight".split()


class ListAtSpec(Spec):
    def __init__(self, templates, options, near_miss):
        self.templates, self.opts, self.near_miss = templates, [o.strip() for o in options], near_miss
        _check_templates(templates, self.opts + _CL_CASES)

    def sample(self, rng, tpl, cls, split, aux=0):
        order = _fill(rng, 4, aux, cls)
        s = {f"l{i}": self.opts[order[i]] for i in range(4)}
        return s | {"q": str(aux), "pos": aux, "order": order, "k": str(rng.choice(_CL_CASES))}

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["pos"]]))
        return dict(s, q=str(j), pos=j), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, k=str(rng.choice([w for w in _CL_CASES if w != s["k"]])))

    def solve(self, prompt):
        lst = re.findall(r"\d", re.search(r"\[([\d, ]+)\]\s*\n", prompt)[1])
        idx = int(re.search(r"\[(\d)\]\)\s*$", prompt)[1])
        return self.opts.index(lst[idx])


class ListIndexSpec(Spec):
    pool = list(range(4, 10))

    def __init__(self, templates, near_miss):
        self.templates, self.near_miss = templates, near_miss
        _check_templates(templates, [])

    def sample(self, rng, tpl, cls, split, aux=0):
        vals = [int(v) for v in rng.choice(self.pool, 4, replace=False)]
        return {f"l{i}": vals[i] for i in range(4)} | {"q": vals[cls], "pos": cls}

    def _vals(self, s):
        return [s[f"l{i}"] for i in range(4)]

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["pos"]]))
        return dict(s, q=s[f"l{j}"], pos=j), j

    def surf_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["pos"]]))
        free = [v for v in self.pool if v not in self._vals(s)]
        return dict(s, **{f"l{j}": int(rng.choice(free))})

    def solve(self, prompt):
        lst = [int(d) for d in re.findall(r"\d", re.search(r"\[([\d, ]+)\]\s*\n", prompt)[1])]
        q = int(re.search(r"\.index\((\d)\)", prompt)[1])
        return lst.index(q)


def _cl_templates(heads, body_fn):
    out = []
    for h in heads:
        for var in ("x", "t"):
            for lst in ("[{l0}, {l1}, {l2}, {l3}]", "[{l0},{l1},{l2},{l3}]"):
                out.append(h + body_fn(var, lst))
    return out


_CL_TPL_A = _cl_templates(["# case {k}\n", "# test {k}\n", "# run {k}\n", "# step {k}\n"],
                          lambda v, lst: f"{v} = {lst}\nprint({v}[{{q}}])\n")
_CL_TPL_B = _cl_templates(["", "# demo\n", "# test\n", "# trace\n"],
                          lambda v, lst: f"{v} = {lst}\nprint({v}.index({{q}}))\n")
_CL_OPTS_A, _CL_OPTS_B = ("3", "5", "7", "9"), ("0", "1", "2", "3")
_CL_NEAR_A = {"first_element": lambda s: s["order"][0], "last_element": lambda s: s["order"][3]}
_CL_NEAR_B = {"value_mod4": lambda s: (s["q"] - 4) % 4,
              "max_position": lambda s: max(range(4), key=lambda i: s[f"l{i}"]),
              "min_position": lambda s: min(range(4), key=lambda i: s[f"l{i}"])}
register(Task("code_listpos_a", "code_listpos", "code tracing", _CL_OPTS_A,
              ListAtSpec(_CL_TPL_A, _CL_OPTS_A, _CL_NEAR_A), copy=True,
              notes="copy=True: the answer digit is one of the four list digits shown in the prompt"))
register(Task("code_listpos_b", "code_listpos", "code tracing", _CL_OPTS_B,
              ListIndexSpec(_CL_TPL_B, _CL_NEAR_B),
              notes="x.index(v) -> position digit; list values 4..9 so no option digit is in the prompt"))


# ---------------------------------------------------------------- 19. rule_range
# "Scores 1-2 get red, 3-4 get blue, 5-6 get green and 7-8 get white. {n} scored 4, so gets": four ranges of width 2 (a: 1-8,
# b: 0-7) mapped to the 4 option labels (a permutation, listed in range order). The score falls in range `aux` (balanced),
# so the answer's mention rank is balanced. Digits follow spaces in the score slot only; the range digits are fixed text.
# Surface slot: the person's name.
class RangeSpec(Spec):
    def __init__(self, templates, options, lo, near_miss):
        self.templates, self.opts, self.lo, self.near_miss = templates, [o.strip() for o in options], lo, near_miss
        _check_templates(templates, self.opts + NAMES)

    def sample(self, rng, tpl, cls, split, aux=0):
        order = _fill(rng, 4, aux, cls)
        s = {f"o{i}": self.opts[order[i]] for i in range(4)}
        return s | {"n": str(rng.choice(NAMES)), "s": self.lo + 2 * aux + int(rng.integers(2)), "rng": aux, "order": order}

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["rng"]]))
        return dict(s, s=self.lo + 2 * j + int(rng.integers(2)), rng=j), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, n=str(rng.choice([w for w in NAMES if w != s["n"]])))

    def solve(self, prompt):
        labels = [w for w in words_of(prompt) if w in self.opts]
        score = int(re.findall(r"\d+", prompt)[-1])
        assert len(labels) == 4, prompt
        return self.opts.index(labels[(score - self.lo) // 2])


def _rr_templates(lo, heads, verbs, tails):
    out = []
    for h in heads:
        for v, third in verbs:
            cl = [f"{lo + 2 * i}-{lo + 2 * i + 1} {v} {{o{i}}}" for i in range(4)]
            body = ", ".join(cl[:3]) + " and " + cl[3]
            out += [f"{h} {body}. {t} {third}" for t in tails]
    return out


_RR_OPTS_A, _RR_OPTS_B = (" red", " blue", " green", " white"), (" bus", " car", " train", " boat")
_RR_NEAR = {"first_label": lambda s: s["order"][0], "last_label": lambda s: s["order"][3],
            "score_mod4": lambda s: s["order"][s["s"] % 4]}
_RR_TPL_A = _rr_templates(1, ["Scores", "Marks", "Points", "Grades"],
                          [("get", "gets"), ("earn", "earns"), ("win", "wins")],
                          ["{n} scored {s}, so", "{n} got {s}, so", "{n} reached {s}, so"])
_RR_TPL_B = _rr_templates(0, ["Ages", "People aged", "Those aged", "Kids aged"],
                          [("go by", "goes by"), ("travel by", "travels by"), ("come by", "comes by")],
                          ["{n} is aged {s}, so", "{n} is {s}, so", "{n} turned {s}, so"])
register(Task("rule_range_a", "rule_range", "rule application", _RR_OPTS_A,
              RangeSpec(_RR_TPL_A, _RR_OPTS_A, 1, _RR_NEAR), copy=True,
              notes="score ranges 1-2, 3-4, 5-6, 7-8 -> labels (permuted); answer's range rank balanced (aux)"))
register(Task("rule_range_b", "rule_range", "rule application", _RR_OPTS_B,
              RangeSpec(_RR_TPL_B, _RR_OPTS_B, 0, _RR_NEAR), copy=True,
              notes="age ranges 0-1, 2-3, 4-5, 6-7 -> transport (permuted); answer's range rank balanced (aux)"))


# ---------------------------------------------------------------- 20. hop_kin
# A chain p0 -> p1 -> p2 -> p3 of four people (the 4 option names, permuted) stated as three links in a shuffled order, each
# written senior-first or junior-first as the template says (a: "X is the parent of Y" / "X is the child of Y"; b: "X manages
# Y" / "X reports to Y"). The query asks two levels above p2 or p3 (answer p0 or p1), so both links are needed and the other
# statements are distractors. The answer's first-mention rank is balanced (`aux`) by rejection sampling over the link order
# and the queried person. Surface slot: a place named in a sentence between the statements and the question.
class KinSpec(Spec):
    def __init__(self, templates, meta, options, senior_first, junior_first, near_miss):
        self.templates, self.meta, self.opts, self.near_miss = templates, meta, [o.strip() for o in options], near_miss
        self.re_sf, self.re_jf = senior_first, junior_first
        _check_templates(templates, self.opts + _LP_PLACES)

    def sample(self, rng, tpl, cls, split, aux=0):
        forms = self.meta[tpl]  # per statement: 0 senior-first, 1 junior-first
        for _ in range(400):
            pi = [int(i) for i in rng.permutation(3)]  # statement i states link pi[i]
            k = int(rng.choice([2, 3]))
            order = []
            for i in range(3):
                j = pi[i]
                pair = (j, j + 1) if forms[i] == 0 else (j + 1, j)
                for p in pair:
                    if p not in order:
                        order.append(p)
            if order.index(k - 2) == aux:
                break
        else:
            raise AssertionError((tpl, aux))
        cl = _fill(rng, 4, k - 2, cls)  # cl[p] = class of person p
        nm = [self.opts[c] for c in cl]
        s = {}
        for i in range(3):
            j = pi[i]
            a, b = (nm[j], nm[j + 1]) if forms[i] == 0 else (nm[j + 1], nm[j])
            s |= {f"x{i}": a, f"y{i}": b}
        return s | {"q": nm[k], "u": str(rng.choice(_LP_PLACES)), "k": k, "nm": nm,
                    "order": [cl[p] for p in order]}

    def ans_twin(self, rng, tpl, s, cls, split):
        k = 5 - s["k"]
        return dict(s, q=s["nm"][k], k=k), self.opts.index(s["nm"][k - 2])

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, u=str(rng.choice([w for w in _LP_PLACES if w != s["u"]])))

    def solve(self, prompt):
        up = {}  # person -> the person one level above
        for m in self.re_sf.finditer(prompt):
            up[m[2]] = m[1]
        for m in self.re_jf.finditer(prompt):
            up[m[1]] = m[2]
        q = [w for w in words_of(prompt) if w in self.opts][-1]
        return self.opts.index(up[up[q]])


def _kin_templates(sf, jf, ends):
    forms = (sf, jf)
    out, meta = [], {}
    for intro in ("So,", "Well,"):
        for combo in range(8):
            fs = [(combo >> i) & 1 for i in range(3)]
            st = [forms[f].format(x=f"{{x{i}}}", y=f"{{y{i}}}") for i, f in enumerate(fs)]
            for end in ends:
                t = f"{intro} " + ". ".join(st) + ". They all met at the {u}. " + end
                out.append(t)
                meta[t] = fs
    return out, meta


_KIN_NEAR = {"first_mentioned": lambda s: s["order"][0], "second_mentioned": lambda s: s["order"][1],
             "last_mentioned": lambda s: s["order"][3]}
_KA_SF, _KA_JF = "{x} is the parent of {y}", "{x} is the child of {y}"
_KA_TPL, _KA_META = _kin_templates(_KA_SF, _KA_JF, ["The grandparent of {q} is", "Who is the grandparent of {q}? It is"])
_KB_SF, _KB_JF = "{x} manages {y}", "{x} reports to {y}"
_KB_TPL, _KB_META = _kin_templates(_KB_SF, _KB_JF, ["The boss of the boss of {q} is", "Who is the boss of the boss of {q}? It is"])
_HK_OPTS_A, _HK_OPTS_B = (" Tom", " Anna", " Ben", " Dan"), (" Mary", " John", " Kate", " Fred")
_RX = lambda sf: re.compile(re.sub(r"\\\{(\w)\\\}", r"(\\w+)", re.escape(sf)))  # noqa: E731
register(Task("hop_kin_a", "hop_kin", "multi-hop composition", _HK_OPTS_A,
              KinSpec(_KA_TPL, _KA_META, _HK_OPTS_A, _RX(_KA_SF), _RX(_KA_JF), _KIN_NEAR), copy=True,
              notes="parent chain over four people, three links in shuffled order; ask a grandparent"))
register(Task("hop_kin_b", "hop_kin", "multi-hop composition", _HK_OPTS_B,
              KinSpec(_KB_TPL, _KB_META, _HK_OPTS_B, _RX(_KB_SF), _RX(_KB_JF), _KIN_NEAR), copy=True,
              notes="manager chain over four people, three links in shuffled order; ask the boss's boss"))
