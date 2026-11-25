"""Batch 3 (families 11-15): code_dict, rule_category, hop_chain, surf_spell, fact_property (two tasks each).

Pools below are literal lists, vetted with `tokcheck.vet_words` against all 17 tokenizers (7 distinct vocabularies): every
word is one token with its leading space everywhere. Class pools are ordered so that P0's split (every third word is a
test word) leaves the length / initial-letter near-miss rules at chance on the held-out half.
"""
from __future__ import annotations

import re

import numpy as np

from .b1 import LETTERS, NAMES, _check_templates
from .core import Spec, Task, WordSpec, _split_words, register, words_of


# ---------------------------------------------------------------- 11. code_dict
class CodeDictSpec(Spec):
    """A dict maps 4 letter keys to the 4 option digits (a permutation); print one value. `direct`: `print(d['b'])`;
    `indirect`: `key = 'b'` then `print(d[key])`. Copy task: the answer digit is shown once, at the queried key's position
    `aux` (balanced), so the position rival is chance. Surface slot: the letter of a key the query never touches."""

    def __init__(self, templates, options, near_miss):
        self.templates, self.opts, self.near_miss = templates, [o.strip() for o in options], near_miss
        _check_templates(templates, self.opts)

    def sample(self, rng, tpl, cls, split, aux=0):
        keys = [str(x) for x in rng.choice(LETTERS, 4, replace=False)]
        order = [None] * 4  # order[i] = class (option index) of the value stored under keys[i]
        order[aux] = cls
        for i, c in zip([i for i in range(4) if i != aux], rng.permutation([c for c in range(4) if c != cls])):
            order[i] = int(c)
        s = {f"k{i}": keys[i] for i in range(4)} | {f"d{i}": self.opts[order[i]] for i in range(4)}
        return s | {"q": keys[aux], "pos": aux, "order": order}

    def _keys(self, s):
        return [s[f"k{i}"] for i in range(4)]

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["pos"]]))
        return dict(s, q=s[f"k{j}"], pos=j), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["pos"]]))
        free = [w for w in LETTERS if w not in self._keys(s)]
        return dict(s, **{f"k{j}": str(rng.choice(free))})

    def solve(self, prompt):
        vals = dict(re.findall(r"'(\w)':\s?(\d)", prompt))
        m = re.search(r"print\(\w+\[(?:'(\w)'|(\w+))\]\)", prompt)
        q = m[1] or re.search(rf"\b{m[2]} = '(\w)'", prompt)[1]
        return self.opts.index(vals[q])


def _cd_templates(indirect):
    out = []
    for hdr in ("", "# demo\n", "# trace\n", "# test\n"):
        for var in ("d", "t"):
            for kv, sep in ((": ", ", "), (":", ",")):
                body = sep.join(f"'{{k{i}}}'{kv}{{d{i}}}" for i in range(4))
                if indirect:
                    out.append(f"{hdr}{var} = {{{{{body}}}}}\nkey = '{{q}}'\nprint({var}[key])\n")
                else:
                    out.append(f"{hdr}{var} = {{{{{body}}}}}\nprint({var}['{{q}}'])\n")
    return out


_CD_NEAR = {"first_value": lambda s: s["order"][0], "last_value": lambda s: s["order"][3],
            "key_letter": lambda s: (ord(s["q"]) - 97) % 4}
_CD_OPTS_A, _CD_OPTS_B = ("3", "5", "7", "9"), ("2", "4", "6", "8")
register(Task("code_dict_a", "code_dict", "code tracing", _CD_OPTS_A,
              CodeDictSpec(_cd_templates(False), _CD_OPTS_A, _CD_NEAR), copy=True,
              notes="copy=True: the answer digit is one of the four dict values shown in the prompt"))
register(Task("code_dict_b", "code_dict", "code tracing", _CD_OPTS_B,
              CodeDictSpec(_cd_templates(True), _CD_OPTS_B, _CD_NEAR), copy=True,
              notes="copy=True: the answer digit is one of the four dict values shown in the prompt; key via a variable"))


# ---------------------------------------------------------------- 12. rule_category
# A rule maps 4 categories (fixed clause order) to the 4 option words (a permutation); a held-out word of one category is
# read out; the answer is that category's option. The queried category is clause `aux` (balanced), so the answer's mention
# rank is balanced and the position rival is chance. Surface slot: the reader's name.
class CategorySpec(Spec):
    def __init__(self, templates, options, class_pools, near_miss):
        self.templates, self.opts, self.class_pools, self.near_miss = templates, [o.strip() for o in options], class_pools, near_miss
        self.cls_key = "w"
        self._cat_of = {w: c for c, p in enumerate(class_pools) for w in p}
        assert len(self._cat_of) == sum(map(len, class_pools)), "a word is in two categories"
        _check_templates(templates, list(self._cat_of) + self.opts + NAMES)

    def sample(self, rng, tpl, cls, split, aux=0):
        order = [None] * 4  # order[i] = class (option index) given to the i-th category
        order[aux] = cls
        for i, c in zip([i for i in range(4) if i != aux], rng.permutation([c for c in range(4) if c != cls])):
            order[i] = int(c)
        s = {f"o{i}": self.opts[order[i]] for i in range(4)}
        return s | {"w": str(rng.choice(_split_words(self.class_pools[aux], split))), "n": str(rng.choice(NAMES)),
                    "cat": aux, "order": order}

    def ans_twin(self, rng, tpl, s, cls, split):
        j = int(rng.choice([i for i in range(4) if i != s["cat"]]))
        return dict(s, w=str(rng.choice(_split_words(self.class_pools[j], split))), cat=j), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        return dict(s, n=str(rng.choice([w for w in NAMES if w != s["n"]])))

    def solve(self, prompt):
        ws = words_of(prompt)
        cats = [self._cat_of[w] for w in ws if w in self._cat_of]
        objs = [w for w in ws if w in self.opts]
        assert len(cats) == 1 and len(objs) == 4, prompt
        return self.opts.index(objs[cats[0]])


def _rc_templates(cats, verbs, readers):
    out = []
    for v, vcap in verbs:
        cl = [f"{{o{i}}} for {art} {noun}" for i, (art, noun) in enumerate(cats)]
        rule = f"Rule: {v} " + ", ".join(cl[:3]) + " and " + cl[3] + "."
        out += [f"{rule} {r} {vcap}" for r in readers]
    return out


_RC_READERS = ["{n} reads the word {w}.", "{n} hears the word {w}.", "{n} sees the word {w}.", "{n} spots the word {w}."]
_RC_A = ["horse dog cat cow sheep bear mouse fish bird hen rat crow".split(),  # animal
         "apple orange bean rice bread egg meat soup nut pie corn cookie".split(),  # food
         "saw fork hook pipe wire chain comb bucket stick club lock pan".split(),  # tool
         "car bus train boat ship van plane jet tram cab cart tank".split()]  # vehicle
_RC_B = ["arm leg hand foot head nose ear eye mouth neck finger hair".split(),  # body part
         "city town farm park beach school church hotel bank shop store forest".split(),  # place
         "tree flower grass leaf rose bush seed root stem".split(),  # plant
         "rain snow wind storm fog cloud sun ice heat mist".split()]  # weather
_RC_OPTS_A, _RC_OPTS_B = (" red", " blue", " green", " white"), (" one", " two", " three", " four")
_RC_TPL_A = _rc_templates([("an", "animal"), ("a", "food"), ("a", "tool"), ("a", "vehicle")],
                          [("say", "Say"), ("write", "Write"), ("reply", "Reply"), ("use", "Use")], _RC_READERS)
_RC_TPL_B = _rc_templates([("a", "body part"), ("a", "place"), ("a", "plant"), ("a", "kind of weather")],
                          [("give", "Give"), ("award", "Award"), ("assign", "Assign"), ("score", "Score")], _RC_READERS)
_RC_NEAR = {"len_mod4": lambda s: len(s["w"]) % 4, "first_option": lambda s: s["order"][0],
            "last_option": lambda s: s["order"][3]}
for _s, _tpl, _pools, _opts in (("a", _RC_TPL_A, _RC_A, _RC_OPTS_A), ("b", _RC_TPL_B, _RC_B, _RC_OPTS_B)):
    register(Task(f"rule_category_{_s}", "rule_category", "rule application", _opts,
                  CategorySpec(_tpl, _opts, _pools, _RC_NEAR), copy=True,
                  notes="the option words are all in the rule text; the queried category's clause rank is balanced (aux)"))


# ---------------------------------------------------------------- 13. hop_chain
# Four things (objects / people) sit in four containers (teams), and the containers sit in four rooms (have four colours).
# Statement 1 pairs thing -> container (in the thing's list order), statement 2 pairs container -> room (listed in a
# permuted order `sigma`); the question names one thing. Both hops are needed: the other three things and containers are
# distractors, and statement 2 is listed in a different order than statement 1. Copy task: the answer room is listed once in
# statement 2, at rank `aux` (balanced).
class ChainSpec(Spec):
    def __init__(self, templates, options, thing_pool, cont_pool, near_miss):
        self.templates, self.opts, self.things, self.conts = templates, [o.strip() for o in options], thing_pool, cont_pool
        self.near_miss = near_miss
        assert not set(thing_pool) & set(cont_pool) and len(thing_pool) >= 6 and len(cont_pool) >= 6
        _check_templates(templates, self.opts + thing_pool + cont_pool)

    def sample(self, rng, tpl, cls, split, aux=0):
        ts = [str(x) for x in rng.choice(self.things, 4, replace=False)]
        cs = [str(x) for x in rng.choice(self.conts, 4, replace=False)]
        qi = int(rng.integers(4))
        order = [None] * 4  # order[j] = class (option index) of the j-th listed room
        order[aux] = cls
        for j, c in zip([j for j in range(4) if j != aux], rng.permutation([c for c in range(4) if c != cls])):
            order[j] = int(c)
        sigma = [None] * 4  # sigma[j] = index of the thing whose container is the j-th listed in statement 2
        sigma[aux] = qi
        for j, i in zip([j for j in range(4) if j != aux], rng.permutation([i for i in range(4) if i != qi])):
            sigma[j] = int(i)
        s = {f"o{i}": ts[i] for i in range(4)} | {f"k{i}": cs[i] for i in range(4)}
        s |= {f"m{j}": cs[sigma[j]] for j in range(4)} | {f"r{j}": self.opts[order[j]] for j in range(4)}
        return s | {"q": ts[qi], "qi": qi, "pos": aux, "order": order, "sigma": sigma}

    def _things(self, s):
        return [s[f"o{i}"] for i in range(4)]

    def ans_twin(self, rng, tpl, s, cls, split):
        i = int(rng.choice([i for i in range(4) if i != s["qi"]]))
        j = s["sigma"].index(i)
        return dict(s, q=s[f"o{i}"], qi=i, pos=j), s["order"][j]

    def surf_twin(self, rng, tpl, s, cls, split):
        i = int(rng.choice([i for i in range(4) if i != s["qi"]]))
        free = [w for w in self.things if w not in self._things(s)]
        return dict(s, **{f"o{i}": str(rng.choice(free))})

    def solve(self, prompt):
        ws = words_of(prompt)
        ts = [w for w in ws if w in self.things]
        cs = [w for w in ws if w in self.conts]
        rooms = [w for w in ws if w in self.opts]
        assert len(ts) == 5 and len(cs) == 8 and len(rooms) == 4, prompt
        return self.opts.index(rooms[cs[4:].index(cs[ts[:4].index(ts[4])])])


def _hca_templates():
    out = []
    for intro in ("First,", "Here,", "Today,", "Note:"):
        for v in ("is", "sits", "stays"):
            s1 = (f"{intro} the {{o0}} {v} in the {{k0}}, the {{o1}} in the {{k1}}, the {{o2}} in the {{k2}} "
                  f"and the {{o3}} in the {{k3}}.")
            s2 = (f"The {{m0}} {v} in the {{r0}}, the {{m1}} in the {{r1}}, the {{m2}} in the {{r2}} and the {{m3}} in the {{r3}}.")
            for q in ("Which room is the {q} in? It is in the", "Which room has the {q}? It is in the"):
                out.append(f"{s1} {s2} {q}")
    return out


def _hcb_templates():
    out = []
    for intro in ("First,", "Here,", "Today,", "Note:"):
        for v, rest in (("is on", "on"), ("plays on", "on"), ("plays for", "for")):
            for w in ("wears", "uses"):
                s1 = (f"{intro} {{o0}} {v} the {{k0}} team, {{o1}} {rest} the {{k1}} team, {{o2}} {rest} the {{k2}} team "
                      f"and {{o3}} {rest} the {{k3}} team.")
                s2 = (f"The {{m0}} team {w} {{r0}}, the {{m1}} team {w} {{r1}}, the {{m2}} team {w} {{r2}} "
                      f"and the {{m3}} team {w} {{r3}}.")
                out.append(f"{s1} {s2} What colour does {{q}} wear? It is")
    return out


_HC_OBJS = "key coin pen cup book ring ball lamp watch card stone shell doll map hat".split()
_HC_CONTS = "box bag jar tin basket case pot bucket".split()
_HC_TEAMS = "horse cow sheep bear mouse fish bird dog".split()
_HC_OPTS_A, _HC_OPTS_B = (" kitchen", " office", " hall", " library"), (" red", " blue", " green", " white")
_HC_NEAR = {"first_room": lambda s: s["order"][0], "last_room": lambda s: s["order"][3],
            "room_at_thing_rank": lambda s: s["order"][s["qi"]]}
register(Task("hop_chain_a", "hop_chain", "multi-hop composition", _HC_OPTS_A,
              ChainSpec(_hca_templates(), _HC_OPTS_A, _HC_OBJS, _HC_CONTS, _HC_NEAR), copy=True,
              notes="object -> container -> room; four containers, four rooms listed in a permuted order"))
register(Task("hop_chain_b", "hop_chain", "multi-hop composition", _HC_OPTS_B,
              ChainSpec(_hcb_templates(), _HC_OPTS_B, NAMES, _HC_TEAMS, _HC_NEAR), copy=True,
              notes="person -> team -> team colour; four teams, four colours listed in a permuted order"))


# ---------------------------------------------------------------- 14. surf_spell
# Three-letter words; the prompt spells out the leading letters joined by an en dash and the answer is the next letter (a: the
# second letter, after `w0–`; b: the third, after `w0–w1–`), appended with no space. Words are held out. The answer twin
# prefers a word of another class that shares the longest leading-letter prefix, so it changes as few tokens as possible.
class SpellSpec(WordSpec):
    def __init__(self, templates, class_pools, surf_pool, near_miss):
        super().__init__(templates, "w", class_pools, "n", surf_pool, near_miss=near_miss)

    def _lead(self, s):
        return dict(s, w0=s["w"][0], w1=s["w"][1])

    def sample(self, rng, tpl, cls, split, aux=0):
        return self._lead(super().sample(rng, tpl, cls, split, aux))

    def ans_twin(self, rng, tpl, s, cls, split):
        c2 = int(rng.choice([c for c in range(4) if c != cls]))
        pool = self._pool(c2, split)
        share = [max(k for k in range(4) if w[:k] == s["w"][:k]) for w in pool]
        best = [w for w, k in zip(pool, share) if k == max(share)]
        return self._lead(dict(s, w=str(rng.choice(best)))), c2


def _ss_templates(tails):
    intros = ["Then {n} wrote the word {w} on the board.", "Later {n} said the word {w} out loud.",
              "Soon {n} read the word {w} in a book.", "Today {n} found the word {w} in a text."]
    return [f"{i} {t}" for i in intros for t in tails]


_SS_A = ["lad tan can sat cap gap tap hat van map man tab".split(),  # second letter a
         "den net wet yes set pen beg hen men leg led jet".split(),  # e
         "rip sit hit bit rib zip tip fig tin fit big did".split(),  # i
         "pop pot hop dot jog fog hot nod pod job box log".split()]  # o
_SS_B = ["fat net sat cat met bit pet fit rat hat not hit".split(),  # third letter t
         "gig jug leg beg dig big jog bug fig dog fog log".split(),  # g
         "sun fin can run fun fan bin pin pen gun den ten".split(),  # n
         "tap gap zip hip map rip lip tip nap cop hop pop".split()]  # p
_SS_TPL_A = _ss_templates(["It is spelled {w0}–", "The spelling begins {w0}–", "Spelled out, it starts {w0}–",
                           "Letter by letter, it is {w0}–"])
_SS_TPL_B = _ss_templates(["It is spelled {w0}–{w1}–", "The spelling begins {w0}–{w1}–", "Spelled out, it starts {w0}–{w1}–",
                           "Letter by letter, it is {w0}–{w1}–"])
_SS_NEAR_A = {"initial_bucket": lambda s: (ord(s["w"][0]) - 97) % 4, "last_bucket": lambda s: (ord(s["w"][2]) - 97) % 4}
_SS_NEAR_B = {"initial_bucket": lambda s: (ord(s["w"][0]) - 97) % 4, "middle_bucket": lambda s: (ord(s["w"][1]) - 97) % 4}
for _s, _tpl, _pools, _opts, _near in (("a", _SS_TPL_A, _SS_A, ("a", "e", "i", "o"), _SS_NEAR_A),
                                       ("b", _SS_TPL_B, _SS_B, ("t", "g", "n", "p"), _SS_NEAR_B)):
    _check_templates(_tpl, [w for p in _pools for w in p] + NAMES)
    register(Task(f"surf_spell_{_s}", "surf_spell", "surface form", _opts,
                  SpellSpec(_tpl, _pools, NAMES, _near),
                  notes="options are glued after an en dash (no space; an ASCII hyphen merges with the next letter in Qwen/OLMo-2 vocabularies); a: second letter, b: third letter of a 3-letter word"))


# ---------------------------------------------------------------- 15. fact_property
_FP_A = ["blood rose brick fire heart apple wine rust".split(),  # red
         "salt rice egg snow paper sugar milk cloud".split(),  # white
         "wave water river lake pool sea sky ocean".split(),  # blue
         "crow tar oil night coal shadow".split()]  # black
_FP_B = ["board log branch stick door table chair tree".split(),  # wood
         "ring can chain fork coin sword bell wire pin key".split(),  # metal
         "ticket page note book menu newspaper poster map letter card".split(),  # paper
         "coat hat dress flag sail tent rug sheet".split()]  # cloth
_FP_OPTS_A, _FP_OPTS_B = (" red", " white", " blue", " black"), (" wood", " metal", " paper", " cloth")


def _fp_templates(intros, tails):
    return [f"{i} {t}" for i in intros for t in tails]


_FP_TPL_A = _fp_templates(
    ["Then {n} saw the {w}.", "Later {n} looked at the {w}.", "Today {n} drew the {w}.", "Once {n} painted the {w}."],
    ["Its usual colour is", "Its colour is usually", "The typical colour of it is", "Most of the time its colour is"])
_FP_TPL_B = _fp_templates(
    ["Next {n} picked up the {w}.", "Here {n} looked at the {w}.", "Now {n} touched the {w}.", "Once {n} bought the {w}."],
    ["It is usually made of", "It is mostly made of", "It is typically made of", "Its main material is"])
_FP_NEAR = {"len_mod4": lambda s: len(s["w"]) % 4, "initial_bucket": lambda s: (ord(s["w"][0]) - 97) % 4}
for _s, _tpl, _pools, _opts in (("a", _FP_TPL_A, _FP_A, _FP_OPTS_A), ("b", _FP_TPL_B, _FP_B, _FP_OPTS_B)):
    _check_templates(_tpl, [w for p in _pools for w in p] + [o.strip() for o in _opts] + NAMES)
    register(Task(f"fact_property_{_s}", "fact_property", "fact lookup", _opts,
                  WordSpec(_tpl, "w", _pools, "n", NAMES, near_miss=_FP_NEAR)))
