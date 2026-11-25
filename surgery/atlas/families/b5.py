"""Batch 5 (families 21-25): surf_rhyme, fact_language, bind_box, arith_count, code_strindex (two tasks each).

Pools below are literal lists, vetted with `tokcheck.vet_words` against all 17 tokenizers (7 distinct vocabularies): every
word is one token with its leading space everywhere (the code words: one token between quotes, checked by the alignment test).
Class pools are ordered so that P0's split (every third word is a test word) leaves the length / initial-letter near-miss
rules at chance on the held-out half.
"""
from __future__ import annotations

import re

import numpy as np

from .b1 import LETTERS, NAMES, BindSpec, _bo_templates, _check_templates
from .core import Spec, Task, WordSpec, _split_words, register, words_of

# ---------------------------------------------------------------- 21. surf_rhyme
# A cue word (held out) rhymes with exactly one of the 4 option words, which never appear in the prompt. Surface slot: name.
_SR_A = ["pat fat rat chat mat hat that sat bat flat".split(),  # cat
         "dead fed head spread thread wed shed red instead ahead said bread".split(),  # bed
         "sing string ring bring spring swing wing thing".split(),  # king
         "bay play way hay pay say stay ray they gray pray may".split()]  # day
_SR_B = ["sea key we tea see three she agree me free he degree".split(),  # tree
         "write right might quite height sight tight flight night bright fright fight".split(),  # light
         "gun ton fun won run none done son one".split(),  # sun
         "core four bore shore pour more floor store".split()]  # door
_SR_INTROS_A = ["Then {n} heard the word {w}.", "Later {n} spoke the word {w} out loud.",
                "Soon {n} read the word {w} in a book.", "Today {n} found the word {w} in a text."]
_SR_INTROS_B = ["Next {n} wrote the word {w} on a card.", "Once {n} sang the word {w} in a song.",
                "Now {n} copied the word {w} twice.", "Here {n} typed the word {w} into a file."]
_SR_TPL_A = [f"{i} {t}" for i in _SR_INTROS_A for t in
             ["It rhymes with", "The word rhymes with", "It sounds like", "It has the same ending as"]]
_SR_TPL_B = [f"{i} {t}" for i in _SR_INTROS_B for t in
             ["It rhymes with", "Its rhyme is", "The word sounds like", "It ends the same way as"]]
_SR_NEAR = {"len_mod4": lambda s: len(s["w"]) % 4, "initial_bucket": lambda s: (ord(s["w"][0]) - 97) % 4}
_SR_OPTS_A, _SR_OPTS_B = (" cat", " bed", " king", " day"), (" tree", " light", " sun", " door")
for _s, _tpl, _pools, _opts in (("a", _SR_TPL_A, _SR_A, _SR_OPTS_A), ("b", _SR_TPL_B, _SR_B, _SR_OPTS_B)):
    _check_templates(_tpl, [w for p in _pools for w in p] + [o.strip() for o in _opts] + NAMES)
    register(Task(f"surf_rhyme_{_s}", "surf_rhyme", "surface form", _opts,
                  WordSpec(_tpl, "w", _pools, "n", NAMES, near_miss=_SR_NEAR)))

# ---------------------------------------------------------------- 22. fact_language
# Too few countries and cities are single tokens in every tokenizer to build four language classes from countries (a) or
# cities (b) alone (French has 6 places in all, German 7). Both tasks therefore draw on the same four class pools of
# single-token places (countries, regions and cities: Spanish, English, German, French), with different frames (a: a visit,
# b: a move), different option order and different pool order (so different held-out places).
_FL_A = ["Cuba Chile Madrid Argentina Colombia Mexico Peru Venezuela Spain".split(),  # Spanish
         "England Sydney Dublin Scotland America Britain Ireland Australia London".split(),  # English
         "Hamburg Frankfurt Vienna Austria Germany Berlin".split(),  # German
         "Lyon Paris France Nice Montreal Quebec".split()]  # French
_FL_B = ["France Nice Paris Lyon Montreal Quebec".split(),  # French
         "Madrid Peru Chile Mexico Santiago Cuba Barcelona Spain".split(),  # Spanish
         "Dublin Melbourne Boston Houston London Seattle Chicago Dallas Sydney".split(),  # English
         "Berlin Hamburg Vienna Austria Germany Wien Frankfurt".split()]  # German
_FL_STARTS = ["Last year", "Then", "Later", "Once"]
_FL_TPL_A = [f"{s} {{n}} {v} {{c}}{e}" for s in _FL_STARTS for v, e in
             zip(["visited", "toured", "explored", "reached"],
                 [", where people speak", ", where the main language is", ", where most people speak",
                  ", where the language is"])]
_FL_TPL_B = [f"{s} {{n}} {v} {{c}}{e}" for s in _FL_STARTS for v, e in
             zip(["moved to", "flew to", "drove to", "went to"],
                 [". People there speak", ". The local language is", ". Most locals speak",
                  ". There the main language is"])]
_FL_NEAR = {"len_mod4": lambda s: len(s["c"]) % 4, "initial_bucket": lambda s: (ord(s["c"][0]) - 65) % 4}
_FL_OPTS_A, _FL_OPTS_B = (" Spanish", " English", " German", " French"), (" French", " Spanish", " English", " German")
for _s, _tpl, _pools, _opts in (("a", _FL_TPL_A, _FL_A, _FL_OPTS_A), ("b", _FL_TPL_B, _FL_B, _FL_OPTS_B)):
    _check_templates(_tpl, [w for p in _pools for w in p] + NAMES)
    register(Task(f"fact_language_{_s}", "fact_language", "fact lookup", _opts,
                  WordSpec(_tpl, "c", _pools, "n", NAMES, near_miss=_FL_NEAR),
                  notes="place -> language; both tasks share four place pools (see module note); b has other frames and order"))

# ---------------------------------------------------------------- 23. bind_box
# Four boxes (named by colour in a, by the thing inside in b) hold four things (the 4 option objects in a) or are of four
# colours (the 4 option colours in b); a is colour -> object, b is object -> colour. Same mechanism and balancing as bind_owner:
# the answer word is mentioned once, at the queried key's list position `aux` (balanced). Surface slot: a non-queried key word.
_BB_COLOURS = "red blue green black white yellow brown gray orange gold silver".split()
_BB_THINGS = "key coin pen cup ring ball lamp watch card stone shell doll map hat book bell bag".split()
_BB_OPTS_A, _BB_OPTS_B = (" key", " coin", " pen", " cup"), (" red", " blue", " green", " black")
_BB_POOL_A = [c for c in _BB_COLOURS]
_BB_POOL_B = [t for t in _BB_THINGS if t not in ("key", "coin", "pen", "cup")] + ["key", "coin", "pen", "cup"]
_BB_TPL_A = _bo_templates(["holds", "keeps", "has"], lambda v, n, o: f"the {n} box {v} a {o}",
                          lambda v: "The {q} box " + v + " a")
_BB_TPL_B = _bo_templates(["is", "sits", "stays"], lambda v, n, o: f"the {n} {v} in the {o} box",
                          lambda v: "The {q} " + v + " in the")
_BB_NEAR = {"first_answer_word": lambda s: s["order"][0], "last_answer_word": lambda s: s["order"][3]}
for _s, _tpl, _opts, _pool in (("a", _BB_TPL_A, _BB_OPTS_A, _BB_POOL_A), ("b", _BB_TPL_B, _BB_OPTS_B, _BB_POOL_B)):
    register(Task(f"bind_box_{_s}", "bind_box", "binding/reference", _opts,
                  BindSpec(_tpl, _opts, _pool, _BB_NEAR), copy=True,
                  notes="a: colour box -> object; b: object -> box colour; the answer word is shown once, at rank aux"))


# ---------------------------------------------------------------- 24. arith_count
# Five words (comma list, fixed length 5); the answer is how many belong to a category, 1..4 (options `one`..`four`). Member and
# non-member words are held out (P0 split). Answer twin: one member <-> one non-member at the same position (count +-1).
# Surface slot: a non-member word -> another non-member (the count and the category words are untouched).
class CountSpec(Spec):
    def __init__(self, templates, options, members, others, near_miss):
        self.templates, self.opts, self.members, self.others = templates, [o.strip() for o in options], members, others
        self.near_miss = near_miss
        assert not set(members) & set(others)
        _check_templates(templates, members + others + self.opts + NAMES)

    def _pick(self, rng, flags, split):
        mem, oth = _split_words(self.members, split), _split_words(self.others, split)
        wm = [str(x) for x in rng.choice(mem, sum(flags), replace=False)]
        wo = [str(x) for x in rng.choice(oth, 5 - sum(flags), replace=False)]
        return [wm.pop() if f else wo.pop() for f in flags]

    def sample(self, rng, tpl, cls, split, aux=0):
        pos = rng.choice(5, cls + 1, replace=False)
        flags = [bool(i in pos) for i in range(5)]
        ws = self._pick(rng, flags, split)
        return {f"w{i}": ws[i] for i in range(5)} | {"m": flags}

    def _words(self, s):
        return [s[f"w{i}"] for i in range(5)]

    def ans_twin(self, rng, tpl, s, cls, split):
        ok = [j for j in range(5) if (s["m"][j] and cls > 0) or (not s["m"][j] and cls < 3)]
        j = int(rng.choice(ok))
        pool = self.others if s["m"][j] else self.members
        free = [w for w in _split_words(pool, split) if w not in self._words(s)]
        m2 = list(s["m"])
        m2[j] = not m2[j]
        return dict(s, **{f"w{j}": str(rng.choice(free)), "m": m2}), cls + (-1 if s["m"][j] else 1)

    def surf_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([j for j in range(5) if not s["m"][j]]))
        free = [w for w in _split_words(self.others, split) if w not in self._words(s)]
        return dict(s, **{f"w{j}": str(rng.choice(free))})

    def solve(self, prompt):
        ws = [w for w in words_of(prompt) if w in self.members or w in self.others]
        assert len(ws) == 5, prompt
        return sum(w in self.members for w in ws) - 1


_AC_ANIMALS = ("fish rat mouse sheep crow cattle ox ant bat horse cow bird dog hen cat bear".split())
_AC_FOODS = ("jam bean soup oil bread apple sugar ham egg nut salt milk meat coffee pie fruit rice orange corn wine tea".split())
_AC_THINGS_A = ("house box road ring cloud cup door boat clock tree hill street bus train bridge window chair wall engine coat "
                "table wheel floor ship paper stone lamp key hat pen river phone plate glass bag car book".split())
_AC_THINGS_B = ("bus door river table lamp key tree road chair coat street floor cloud house paper bag car hat ring plate phone "
                "clock glass cup boat train wall box hill ship pen window wheel engine bridge stone book".split())


def _ac_templates(kind):
    lists = ["Of the words {w0}, {w1}, {w2}, {w3} and {w4},", "Among the words {w0}, {w1}, {w2}, {w3} and {w4},",
             "In the list {w0}, {w1}, {w2}, {w3} and {w4},", "Within the words {w0}, {w1}, {w2}, {w3} and {w4},"]
    tails = [f"the number of {kind} is", f"the count of {kind} is", f"the total number of {kind} is",
             f"the quantity of {kind} is"]
    return [f"{a} {b}" for a in lists for b in tails]


_AC_NEAR = {"first_word_member": lambda s: (0 if s["m"][0] else 3) if s["m"][0] != s["m"][4] else 1,
            "last_word_soft": lambda s: 1 if s["m"][4] else 2, "len_mod4": lambda s: len(s["w0"]) % 4}


_AC_OPTS = (" one", " two", " three", " four")
register(Task("arith_count_a", "arith_count", "arithmetic", _AC_OPTS,
              CountSpec(_ac_templates("animals"), _AC_OPTS, _AC_ANIMALS, _AC_THINGS_A, _AC_NEAR)))
register(Task("arith_count_b", "arith_count", "arithmetic", _AC_OPTS,
              CountSpec(_ac_templates("foods"), _AC_OPTS, _AC_FOODS, _AC_THINGS_B, _AC_NEAR)))


# ---------------------------------------------------------------- 25. code_strindex
# Two string variables; print one letter of the one that is printed. a: `x[0]` (first letter), b: `x[-1]` (last letter). The
# printed word is held out (class = its letter); the other variable's word is the surface slot. The oracle runs the code.
class StrIndexSpec(WordSpec):
    def __init__(self, templates, class_pools, surf_pool, options, near_miss):
        super().__init__(templates, "w", class_pools, "u", surf_pool, near_miss=near_miss)
        self.opts = [o.strip() for o in options]
        _check_templates(templates, self.opts)

    def solve(self, prompt):
        env, shown = {}, None
        for line in prompt.split("\n"):
            if m := re.fullmatch(r"(\w+) = (['\"])(\w+)\2", line.strip()):
                env[m[1]] = m[3]
            elif m := re.fullmatch(r"print\((\w+)\[(-?\d)\]\)", line.strip()):
                shown = env[m[1]][int(m[2])]
        return self.opts.index(shown)


def _si_templates(idx):
    out = []
    for hdr in ("", "# demo\n", "# trace\n", "# test\n"):
        for q in ("'", '"'):
            for order in (0, 1):
                a, b = ("{w}", "{u}") if order == 0 else ("{u}", "{w}")
                pv = "x" if order == 0 else "y"
                out.append(f"{hdr}x = {q}{a}{q}\ny = {q}{b}{q}\nprint({pv}[{idx}])\n")
    return out


_SI_FIRST = ["dog deep door dark date day".split(),  # d
             "cook cat cap car card cup clock cart".split(),  # c
             "mile mask man mark mine map mouse math".split(),  # m
             "path pole pool plan pen pet park pie pan port pot pair".split()]  # p
_SI_LAST = ["bat test pot net cat nut hat last art east cut hot hit".split(),  # t
            "man pen town rain ten sun run fan burn pan can win".split(),  # n
            "flag ring bag leg dog song big tag rug bug long king".split(),  # g
            "bird wind red hand card god road bad bed".split()]  # d
_SI_SURF = "tree ship lamp bell rain wall road hill star moon bag lake book coin farm wind cloud snow".split()
_SI_SURF = ("tree ship bell rain wall hill star book cloud fire fish rock wood leaf bone sock glass house water paper "
            "stone".split())
_SI_SURF = [w for w in _SI_SURF if w not in {x for p in _SI_FIRST + _SI_LAST for x in p}]
_SI_NEAR_A = {"len_mod4": lambda s: len(s["w"]) % 4, "last_bucket": lambda s: (ord(s["w"][-1]) - 97) % 4}
_SI_NEAR_B = {"len_mod4": lambda s: len(s["w"]) % 4, "initial_bucket": lambda s: (ord(s["w"][0]) - 97) % 4}
_SI_OPTS_A, _SI_OPTS_B = ("d", "c", "m", "p"), ("t", "n", "g", "d")
register(Task("code_strindex_a", "code_strindex", "code tracing", _SI_OPTS_A,
              StrIndexSpec(_si_templates(0), _SI_FIRST, _SI_SURF, _SI_OPTS_A, _SI_NEAR_A),
              notes="first letter of the printed word; the answer letter follows a newline"))
register(Task("code_strindex_b", "code_strindex", "code tracing", _SI_OPTS_B,
              StrIndexSpec(_si_templates(-1), _SI_LAST, _SI_SURF, _SI_OPTS_B, _SI_NEAR_B),
              notes="last letter of the printed word; the answer letter follows a newline"))
