"""Batch 10 (families 46-50): hop_seat, arith_sum3, code_last, bind_color, fact_kind (two tasks each).

Pools below are literal lists, vetted with `tokcheck.vet_words` against all 17 tokenizers (7 distinct vocabularies): every
word is one token with its leading space everywhere (code values: one token after "# prints", checked by the option test).
"""
from __future__ import annotations

import re

from .b1 import NAMES, _check_templates
from .b7 import _perm_with
from .core import Spec, Task, WordSpec, register, words_of

_PLACES = "park school shop farm house garden market lake beach hotel yard camp".split()
_COLS = "red blue green white black yellow brown gray orange".split()


# ---------------------------------------------------------------- 46. hop_seat
# a: "{n0} sits next to {n1} and {n2} sits next to {n3}. {the four people, permuted} wears a {colour} hat ... The person next
#    to {q} wears a" -> the colour of q's neighbour (hop 1: q -> neighbour, hop 2: neighbour -> colour); the four colours are the
#    4 options, so every person has a colour and the queried person's own colour is a distractor.
# b: "{n_i} has a {pet_i} ... the {pet} sleeps in the {container}, ... {q}'s pet sleeps in the" (hop 1: q -> pet, hop 2: pet ->
#    container); each of the four pets has a container (the 4 options).
# The answer's container/colour is shown once, at statement rank `aux` (balanced), so the position rival is chance.
# Surface slot: the place word at the start of the prompt (ignored by the answer).
class SeatSpec(Spec):
    def __init__(self, templates, options, mode, near_miss, pets=()):
        self.templates, self.opts, self.mode, self.pets = templates, [o.strip() for o in options], mode, list(pets)
        self.near_miss = near_miss
        _check_templates(templates, self.opts + NAMES + _PLACES + self.pets)

    def sample(self, rng, tpl, cls, split, aux=0):
        n = [str(x) for x in rng.choice(NAMES, 4, replace=False)]
        j = int(rng.integers(4))  # queried person
        p = j ^ 1 if self.mode == "a" else j  # the person whose colour / container is the answer
        cc = _perm_with(rng, cls, p)  # cc[person] = class of that person's colour / container
        ord_ = _perm_with(rng, p, aux)  # statement order: ord_[r] = person; the answer person is at rank aux
        s = {f"n{i}": n[i] for i in range(4)}
        if self.mode == "b":
            pt = [str(x) for x in rng.choice(self.pets, 4, replace=False)]
            s |= {f"p{i}": pt[i] for i in range(4)} | {f"sp{r}": pt[ord_[r]] for r in range(4)}
        else:
            s |= {f"sn{r}": n[ord_[r]] for r in range(4)}
        s |= {f"sc{r}": self.opts[cc[ord_[r]]] for r in range(4)}
        return s | {"q": n[j], "pl": str(rng.choice(_PLACES)), "j": j, "cc": cc, "ord": ord_}

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["j"]]))
        p = j ^ 1 if self.mode == "a" else j
        return dict(s, q=s[f"n{j}"], j=j), s["cc"][p]

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, pl=str(rng.choice([w for w in _PLACES if w != s["pl"]])))

    def solve(self, prompt):
        ws = words_of(prompt)
        names, cols = [w for w in ws if w in NAMES], [w for w in ws if w in self.opts]
        assert len(cols) == 4, prompt
        if self.mode == "a":
            assert len(names) == 9, prompt
            return self.opts.index(cols[names[4:8].index(names[:4][names[:4].index(names[8]) ^ 1])])
        pets = [w for w in ws if w in self.pets]
        assert len(names) == 5 and len(pets) == 8, prompt
        return self.opts.index(cols[pets[4:].index(pets[names[:4].index(names[4])])])


def _hs_templates(body, tails):
    intros = ["At the {pl},", "In the {pl},", "Near the {pl},", "By the {pl},"]
    return [f"{i} {body} {t}" for i in intros for t in tails]


_HS_A = [t.replace("sits", v) for t in _hs_templates(
    "{n0} sits next to {n1} and {n2} sits next to {n3}. {sn0} wears a {sc0} hat, {sn1} wears a {sc1} hat, "
    "{sn2} wears a {sc2} hat and {sn3} wears a {sc3} hat.",
    ["The person next to {q} wears a", "The one beside {q} wears a", "Whoever is next to {q} wears a"])
    for v in ("sits", "stands")]
_HS_B = _hs_templates(
    "{n0} has a {p0}, {n1} has a {p1}, {n2} has a {p2} and {n3} has a {p3}. The {sp0} sleeps in the {sc0}, the {sp1} in the "
    "{sc1}, the {sp2} in the {sc2} and the {sp3} in the {sc3}.",
    ["{q}'s pet sleeps in the", "The pet owned by {q} sleeps in the", "The pet that {q} owns sleeps in the"])
_HS_OPTS_A, _HS_OPTS_B = (" red", " blue", " green", " white"), (" box", " bag", " bed", " jar")
_HS_PETS = "cat dog hen cow horse mouse bird fish sheep rat bear".split()
_HS_NEAR_A = {"first_named": lambda s: s["cc"][0], "first_colour": lambda s: s["cc"][s["ord"][0]],
              "last_colour": lambda s: s["cc"][s["ord"][3]]}
_HS_NEAR_B = {"first_owner": lambda s: s["cc"][0], "first_container": lambda s: s["cc"][s["ord"][0]],
              "last_container": lambda s: s["cc"][s["ord"][3]]}
register(Task("hop_seat_a", "hop_seat", "multi-hop composition", _HS_OPTS_A,
              SeatSpec(_HS_A, _HS_OPTS_A, "a", _HS_NEAR_A), copy=True,
              notes="two seat pairs; colour of the person next to q; both hops needed (all four people have a colour)"))
register(Task("hop_seat_b", "hop_seat", "multi-hop composition", _HS_OPTS_B,
              SeatSpec(_HS_B, _HS_OPTS_B, "b", _HS_NEAR_B, _HS_PETS), copy=True,
              notes="name -> pet -> container; four pets, four containers; both hops needed"))


# ---------------------------------------------------------------- 47. arith_sum3
# a: `{a}+{b}+{c}=` -> 6..9 (the options); b: `{a}-{b}-{c}=` -> 1..4. Compact (no spaces around operators) as in arith_addsub, so
# the answer digit follows `=`. The header carries an item number `k.m.x.y` of four random digits (deviation: the sketch has one
# item number): the digit counts of a prompt add up to the answer, and with 0-2 extra digits a bag of words solved the sum (0.41-0.46);
# with four it cannot. Option digits can occur as operands or in the item number: copy task. Surface slot: the first digit k
# of the item number (a digit after a space; the tokenizer test checks the twin alignment).
class Sum3Spec(Spec):
    nums = range(1, 10)

    def __init__(self, templates, mode, lo, near_miss, hi=9):
        self.templates, self.mode, self.lo, self.near_miss, self.hi = templates, mode, lo, near_miss, hi
        _check_templates(templates, [])

    def value(self, a, b, c):
        return a + b + c if self.mode == "add" else a - b - c

    def _triples(self, c):
        r = range(1, self.hi + 1)
        return [(a, b, d) for a in r for b in r for d in r if self.value(a, b, d) - self.lo == c]

    def sample(self, rng, tpl, cls, split, aux=0):
        t = self._triples(cls)
        a, b, c = t[int(rng.integers(len(t)))]
        return {"a": a, "b": b, "c": c} | {key: int(rng.integers(1, 10)) for key in "kmxy"}

    def ans_twin(self, rng, tpl, s, cls, split):
        opts = []
        for key in "abc":
            for v in range(1, self.hi + 1):
                s2 = dict(s, **{key: v})
                c2 = self.value(s2["a"], s2["b"], s2["c"]) - self.lo
                if v != s[key] and 0 <= c2 < 4 and c2 != cls:
                    opts.append((s2, c2))
        return opts[int(rng.integers(len(opts)))]

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, k=int(rng.choice([k for k in self.nums if k != s["k"]])))

    def solve(self, prompt):
        m = re.search(r"(\d)([+-])(\d)([+-])(\d)=$", prompt)
        a, b, c = int(m[1]), int(m[3]), int(m[5])
        assert m[2] == m[4] == ("+" if self.mode == "add" else "-"), prompt
        return self.value(a, b, c) - self.lo


def _s3_templates(op):
    q = "{a}" + op + "{b}" + op + "{c}="
    return [f"{p} {{k}}.{{m}}.{{x}}.{{y}}{sep}" + q for p in ("Problem", "Exercise", "Question", "Task")
            for sep in (":\n", ".\n", "\n")]


def _clamp(v):
    return min(max(v, 0), 3)


_S3_NEAR_A = {"first_two_plus_five": lambda s: _clamp(s["a"] + s["b"] + 5 - 6), "last_operand": lambda s: s["c"] % 4}
_S3_NEAR_B = {"first_two_minus_five": lambda s: _clamp(s["a"] - s["b"] - 5), "first_operand": lambda s: s["a"] % 4}
register(Task("arith_sum3_a", "arith_sum3", "arithmetic", ("6", "7", "8", "9"),
              Sum3Spec(_s3_templates("+"), "add", 6, _S3_NEAR_A), copy=True,
              notes="three-operand sum 6..9 ; copy=True: an option digit can be an operand or the item number"))
register(Task("arith_sum3_b", "arith_sum3", "arithmetic", ("1", "2", "3", "4"),
              Sum3Spec(_s3_templates("-"), "sub", 1, _S3_NEAR_B), copy=True,
              notes="a-b-c = 1..4; copy=True: an option digit can be an operand or the item number"))


_CL_HEADS = "one two three four five six seven eight".split()  # the case number words
_CL_FRAMES = "case test run step demo trace".split()


# ---------------------------------------------------------------- 48. code_last
# xs = []; four `xs.append('v')` lines (the 4 option words, permuted); b adds a fifth append of a filler word and `xs.pop()`
# (removes the filler). `print(xs[-j])` with j = aux + 1 (balanced) prints the element at rank 3 - aux of the options, so the answer's
# mention rank is balanced (deviation from the sketch's fixed `xs[-1]`, whose answer would always be the last word mentioned).
# The prompt ends "print(xs[-j])\n# prints" as in code_var (b9) and the answer is the next word. Surface slot: the case number
# word in the header comment (`# case three`).
class LastSpec(Spec):
    def __init__(self, templates, options, fillers, near_miss, pop):
        self.templates, self.opts, self.fillers, self.near_miss, self.pop = templates, [o.strip() for o in options], fillers, near_miss, pop
        _check_templates(templates, self.opts + fillers + _CL_HEADS)

    def sample(self, rng, tpl, cls, split, aux=0):
        order = _perm_with(rng, cls, 3 - aux)
        s = {f"d{i}": self.opts[order[i]] for i in range(4)}
        if self.pop:
            s["d4"] = str(rng.choice(self.fillers))
        return s | {"j": aux + 1, "h": str(rng.choice(_CL_HEADS)), "order": order}

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([k for k in range(1, 5) if k != s["j"]]))
        return dict(s, j=j), s["order"][4 - j]

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, h=str(rng.choice([w for w in _CL_HEADS if w != s["h"]])))

    def solve(self, prompt):
        lst = []
        for line in prompt.split("\n"):
            if m := re.fullmatch(r"\w+\.append\((['\"])(\w+)\1\)", line.strip()):
                lst.append(m[2])
            elif re.fullmatch(r"\w+\.pop\(\)", line.strip()):
                lst.pop()
            elif m := re.fullmatch(r"print\(\w+\[-(\d)\]\)", line.strip()):
                return self.opts.index(lst[-int(m[1])])
        raise AssertionError(prompt)


def _cl_templates(var, n_app, pop):
    out = []
    for h in _CL_FRAMES:
        for q in ("'", '"'):
            body = f"{var} = []\n" + "".join(f"{var}.append({q}{{d{i}}}{q})\n" for i in range(n_app))
            out.append(f"# {h} {{h}}\n" + body + (f"{var}.pop()\n" if pop else "") + f"print({var}[-{{j}}])\n# prints")
    return out


_CL_OPTS_A, _CL_OPTS_B = (" red", " blue", " green", " white"), (" cat", " dog", " cow", " hen")
_CL_FILL_A, _CL_FILL_B = "black brown gray orange yellow".split(), "horse sheep mouse bird fish".split()
_CL_NEAR_A = {"first_appended": lambda s: s["order"][0], "last_appended": lambda s: s["order"][3]}
_CL_NEAR_B = {"first_appended": lambda s: s["order"][0], "second_to_last": lambda s: s["order"][3]}
register(Task("code_last_a", "code_last", "code tracing", _CL_OPTS_A,
              LastSpec(_cl_templates("xs", 4, False) + _cl_templates("ys", 4, False), _CL_OPTS_A, _CL_FILL_A, _CL_NEAR_A, False),
              copy=True, notes="four appends, print(xs[-j]); answer rank balanced through j (deviation: not always xs[-1])"))
register(Task("code_last_b", "code_last", "code tracing", _CL_OPTS_B,
              LastSpec(_cl_templates("xs", 5, True) + _cl_templates("ys", 5, True), _CL_OPTS_B, _CL_FILL_B, _CL_NEAR_B, True),
              copy=True, notes="five appends (the last is a filler word), xs.pop(), print(xs[-j]); answer rank balanced through j"))


# ---------------------------------------------------------------- 49. bind_color
# a: four people each have one coloured item (the 4 colours permuted); "{q}'s hat is" -> colour. b: reverse: four fixed names
# (the options) each have an item whose colour is drawn from a pool of 6; "The one with the {colour} hat is" -> the name. The
# answer word is mentioned once, at list position `aux` (balanced). Surface slot: the place word at the start.
class ColorSpec(Spec):
    def __init__(self, templates, options, mode, near_miss):
        self.templates, self.opts, self.mode, self.near_miss = templates, [o.strip() for o in options], mode, near_miss
        _check_templates(templates, self.opts + NAMES + _PLACES + _COLS)

    def sample(self, rng, tpl, cls, split, aux=0):
        order = _perm_with(rng, cls, aux)  # order[i] = class of person i
        if self.mode == "a":
            n = [str(x) for x in rng.choice(NAMES, 4, replace=False)]
            s = {f"n{i}": n[i] for i in range(4)} | {f"c{i}": self.opts[order[i]] for i in range(4)} | {"q": n[aux]}
        else:
            c = [str(x) for x in rng.choice(_COLS, 4, replace=False)]
            s = {f"n{i}": self.opts[order[i]] for i in range(4)} | {f"c{i}": c[i] for i in range(4)} | {"q": c[aux]}
        return s | {"pl": str(rng.choice(_PLACES)), "pos": aux, "order": order}

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["pos"]]))
        return dict(s, q=s[f"n{j}"] if self.mode == "a" else s[f"c{j}"], pos=j), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, pl=str(rng.choice([w for w in _PLACES if w != s["pl"]])))

    def solve(self, prompt):
        ws = words_of(prompt)
        if self.mode == "a":
            names, cols = [w for w in ws if w in NAMES], [w for w in ws if w in self.opts]
            assert len(names) == 5 and len(cols) == 4, prompt
            return self.opts.index(cols[names[:4].index(names[4])])
        names, cols = [w for w in ws if w in self.opts], [w for w in ws if w in _COLS]
        assert len(names) == 4 and len(cols) == 5, prompt
        return self.opts.index(names[cols[:4].index(cols[4])])


def _bc_templates(verbs, tail):
    intros = ["At the {pl},", "In the {pl},", "Near the {pl},", "By the {pl},"]
    out = []
    for i in intros:
        for x in ("hat", "bag", "coat"):
            for v in verbs:
                body = ", ".join(f"{{n{k}}} {v} a {{c{k}}} {x}" for k in range(3)) + f" and {{n3}} {v} a {{c3}} {x}."
                out.append(f"{i} {body} {tail(x)}")
    return out


_BC_OPTS_A, _BC_OPTS_B = (" red", " blue", " green", " white"), (" Tom", " Anna", " Ben", " Mary")
_BC_TPL_A = _bc_templates(["has"], lambda x: "{q}'s " + x + " is")
_BC_TPL_A += _bc_templates(["owns"], lambda x: "The " + x + " of {q} is")
_BC_TPL_B = _bc_templates(["has"], lambda x: "The one with the {q} " + x + " is")
_BC_TPL_B += _bc_templates(["owns"], lambda x: "The person with the {q} " + x + " is")
_BC_NEAR_A = {"first_colour": lambda s: s["order"][0], "last_colour": lambda s: s["order"][3]}
_BC_NEAR_B = {"first_name": lambda s: s["order"][0], "last_name": lambda s: s["order"][3]}
register(Task("bind_color_a", "bind_color", "binding/reference", _BC_OPTS_A,
              ColorSpec(_BC_TPL_A, _BC_OPTS_A, "a", _BC_NEAR_A), copy=True,
              notes="person -> colour of their hat/bag/coat; answer's list position balanced (aux)"))
register(Task("bind_color_b", "bind_color", "binding/reference", _BC_OPTS_B,
              ColorSpec(_BC_TPL_B, _BC_OPTS_B, "b", _BC_NEAR_B), copy=True,
              notes="colour -> which of four fixed names has it; answer's list position balanced (aux)"))


# ---------------------------------------------------------------- 50. fact_kind
# word -> category. a: vehicle / building / animal / food; b: tool / relative / clothes / weather (the two sets are disjoint).
# Only about 10 words per category are single tokens in all 7 vocabularies, so the pools below are what was left after
# vetting (no furniture, garment, instrument or fruit label words/lists were usable). Held-out words (P0 split). Surface slot:
# the name.
_FK_A = ["boat cart jet carriage ship coach train van cab bus tram plane tank car".split(),  # vehicle
         "house museum school church hotel temple tent palace castle tower prison shop bank cabin inn mill".split(),  # building
         "mouse swift bear dog sole bass dove fish cod bat ray horse sheep rat hen cow crow cat".split(),  # animal
         "bread sugar egg soup corn ham rice bean pie meat salt".split()]  # food
_FK_B = ["wire chain hook pin key fork jack file lock pen saw pick".split(),  # tool
         "daughter uncle cousin son mother husband wife father brother sister".split(),  # relative
         "uniform cap hat tie vest suit dress boot coat".split(),  # clothes
         "cloud rain storm ice fog mist heat wind snow".split()]  # weather
_FK_TPL_A = [f"{i} {t}" for i in
             ["Then {n} pointed at the {w},", "Later {n} talked about the {w},", "Soon {n} drew the {w},",
              "Today {n} read about the {w},"]
             for t in ["which is a kind of", "and it is a kind of", "since it is a kind of"]]
_FK_TPL_B = [f"{i} {t}" for i in
             ["Next {n} looked at the {w},", "Once {n} wrote down the word {w},", "Now {n} thought about the {w},",
              "Here {n} spoke about the {w},"]
             for t in ["which is a type of", "and it is a type of", "because it is a type of"]]
_FK_NEAR = {"len_mod4": lambda s: len(s["w"]) % 4, "initial_bucket": lambda s: (ord(s["w"][0]) - 97) % 4}
_FK_OPTS_A, _FK_OPTS_B = (" vehicle", " building", " animal", " food"), (" tool", " relative", " clothes", " weather")
for _s, _tpl, _pools, _opts in (("a", _FK_TPL_A, _FK_A, _FK_OPTS_A), ("b", _FK_TPL_B, _FK_B, _FK_OPTS_B)):
    _check_templates(_tpl, [w for p in _pools for w in p] + [o.strip() for o in _opts] + NAMES)
    register(Task(f"fact_kind_{_s}", "fact_kind", "fact lookup", _opts,
                  WordSpec(_tpl, "w", _pools, "n", NAMES, near_miss=_FK_NEAR),
                  notes="a: vehicle/building/animal/food; b: tool/relative/clothes/weather (see module note)"))
