"""Study B2 task family (docs/CONTRACT-capability.md, "Study B2: a harder task family").

Same mock coding session as `generator` (same state sampling, same twelve derived facts), but the answer takes a chain
of steps: named flags are defined over facts and earlier flags (AND / OR / NOT), and an ordered list of action rules
over facts and flags picks the label. The prompt prints the files, tests and history, then (depending on the rung) a
`Status:` section with all, half or none of the derived facts, a `Flags:` section and the rules.

Decision procedure (`solve`, a pure function of the state dict):
  1. Facts come from `generator._facts` (derivable from the files, tests and history the prompt prints).
  2. Each flag is True when its definition holds; a definition may use facts and flags of lower depth. Depth 1 uses
     facts only; depth d uses at least one flag of depth d - 1. Flags are printed shuffled when `hops >= 2`.
  3. Rules are in priority order; the first whose condition holds decides. The last rule is an unconditional
     "otherwise", so exactly one rule decides every item. Actions are distinct across rules; the disabled rules use the
     remaining labels (never fire), so every prompt lists each label once.

Expressions are nested lists: ["fact", key, "yes"|"no"], ["flag", name], ["not", e], ["and", [e, ..]], ["or", [e, ..]].
Nested compounds are always printed with parentheses.

`generate_hard` rejection-samples on (label, conflict, depth): labels are dealt in cycles of all twelve (every label
once per cycle), `conflict_rate` of items have two or more conditional rules firing, the catch-all decides few items, and
on rungs with hops > 0 most items are decided by a rule that depends on a flag of full depth whose flip changes the answer,
and most differ from the answer a reader gets by treating every flag as false.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from surgery.task.generator import FACT_KEYS, HARD_LABELS, _facts, _sample_state, _yn

FLAG_WORDS = ["blocked", "risky", "stale", "ready", "cautious", "urgent", "fragile", "settled", "pending", "clean",
              "noisy", "stuck", "safe", "tense", "loose", "strict"]
FLAT_SHARE = 0.65    # share of items on rungs with flags whose answer differs from reading every flag as false
DEEP_SHARE = 0.8     # share of items on rungs with hops > 0 that must be decided through a full-depth, flip-sensitive flag
CATCH_KEEP = 0.4     # an item decided by the final catch-all is kept with this probability
MAX_TRIES = 20000
PROGRAM_REUSE = 8    # states tried on one sampled program before drawing another


@dataclass
class HardKnobs:
    n_rules: int = 8             # action rules including the final "otherwise"
    n_flags: int = 0
    hops: int = 0                # longest flag chain; at least one flag has exactly this depth
    hide_status: bool = False    # True: no Status section, facts must be derived from the event log
    or_not: bool = True          # False: conditions are ANDs of facts only (flags positive)
    status_share: float = 1.0    # share of facts listed under Status when not hidden
    history_extra: int = 5       # as the old R3
    n_distractor_files: int = 5
    n_distractor_tests: int = 5
    conflict_rate: float = 0.6
    cond_size: int = 3


# ---------------------------------------------------------------- expressions

def render_expr(e) -> str:
    t = e[0]
    if t == "fact":
        return f"{e[1]}={e[2]}"
    if t == "flag":
        return e[1]
    if t == "not":
        inner = render_expr(e[1])
        return f"not ({inner})" if e[1][0] in ("and", "or") else f"not {inner}"
    parts = [f"({render_expr(c)})" if c[0] in ("and", "or") else render_expr(c) for c in e[1]]
    return f" {t} ".join(parts)


def eval_expr(e, facts: dict, flag) -> bool:
    t = e[0]
    if t == "fact":
        return facts[e[1]] == e[2]
    if t == "flag":
        return flag(e[1])
    if t == "not":
        return not eval_expr(e[1], facts, flag)
    if t == "and":
        return all(eval_expr(c, facts, flag) for c in e[1])
    return any(eval_expr(c, facts, flag) for c in e[1])


def _refs(e, kind: str) -> set:
    t = e[0]
    if t == kind:
        return {e[1]}
    if t == "not":
        return _refs(e[1], kind)
    if t in ("and", "or"):
        return set().union(*(_refs(c, kind) for c in e[1]))
    return set()


def _combine(rng: random.Random, atoms: list, or_not: bool):
    if len(atoms) == 1:
        return atoms[0]
    if not or_not:
        return ["and", atoms]
    shape = rng.choices(["and", "or", "mix", "not"], [0.4, 0.2, 0.25, 0.15])[0]
    if shape == "and" or shape == "or":
        return [shape, atoms]
    if shape == "not":
        return ["not", [rng.choice(["and", "or"]), atoms]]
    if len(atoms) == 2:
        return [rng.choice(["and", "or"]), atoms]
    top, inner = rng.choice([("and", "or"), ("or", "and")])
    return [top, [[inner, atoms[:2]], atoms[2]]]


def _flag_atom(rng: random.Random, name: str, or_not: bool):
    a = ["flag", name]
    return ["not", a] if or_not and rng.random() < 0.35 else a


def _fact_atoms(rng: random.Random, n: int) -> list:
    return [["fact", key, rng.choice(["yes", "no"])] for key in rng.sample(FACT_KEYS, n)]


# ---------------------------------------------------------------- program (flags, rules) sampling

def _sample_flags(rng: random.Random, k: HardKnobs) -> list[dict]:
    """Flag definitions in printed order (shuffled when hops >= 2): [{"name", "depth", "cond"}]."""
    if not k.n_flags:
        return []
    names = rng.sample(FLAG_WORDS, k.n_flags)
    depths = sorted(list(range(1, k.hops + 1)) + [rng.randint(1, k.hops) for _ in range(k.n_flags - k.hops)])
    flags: list[dict] = []
    for name, d in zip(names, depths):
        lower = [f for f in flags if f["depth"] < d]
        atoms = []
        if d > 1:
            prior = rng.choice([f for f in flags if f["depth"] == d - 1])
            atoms.append(_flag_atom(rng, prior["name"], k.or_not))
        else:
            prior = None
        size = rng.randint(max(2, len(atoms)), max(2, k.cond_size))
        extra_flags = [f for f in lower if f is not prior]
        n_fact = size - len(atoms)
        if extra_flags and n_fact > 1 and rng.random() < 0.35:
            atoms.append(_flag_atom(rng, rng.choice(extra_flags)["name"], k.or_not))
            n_fact -= 1
        atoms += _fact_atoms(rng, n_fact)
        rng.shuffle(atoms)
        flags.append({"name": name, "depth": d, "cond": _combine(rng, atoms, k.or_not)})
    if k.hops >= 2:
        rng.shuffle(flags)
    return flags


def _sample_rules(rng: random.Random, k: HardKnobs, flags: list[dict]) -> tuple[list[dict], list[dict]]:
    """(active rules, the last with cond None; disabled rules on the unused labels)."""
    acts = rng.sample(HARD_LABELS, k.n_rules)
    weights = [f["depth"] ** 2 for f in flags]
    rules = []
    for a in acts[:-1]:
        size = rng.choices([1, 2, 3], [0.2, 0.5, 0.3])[0] if k.cond_size >= 3 else rng.randint(1, k.cond_size)
        atoms: list = []
        if flags and rng.random() < 0.65:
            for name in dict.fromkeys(rng.choices([f["name"] for f in flags], weights, k=1 + (rng.random() < 0.25))):
                atoms.append(_flag_atom(rng, name, k.or_not))
        atoms = atoms[:size]
        atoms += _fact_atoms(rng, max(0, size - len(atoms)))
        rng.shuffle(atoms)
        rules.append({"cond": _combine(rng, atoms, k.or_not), "action": a})
    rules.append({"cond": None, "action": acts[-1]})
    off = [a for a in HARD_LABELS if a not in acts]
    rng.shuffle(off)
    disabled = [{"cond": _combine(rng, _fact_atoms(rng, rng.randint(1, k.cond_size)), k.or_not), "action": a}
                for a in off]
    return rules, disabled


# ---------------------------------------------------------------- solving

def flag_values(s: dict, facts: dict, force: tuple | None = None) -> dict:
    defs = {f["name"]: f for f in s["flags"]}
    memo: dict = {}

    def val(name):
        if name not in memo:
            memo[name] = force[1] if force and name == force[0] else eval_expr(defs[name]["cond"], facts, val)
        return memo[name]

    for n in defs:
        val(n)
    return memo


def _decide(s: dict, facts: dict, fv: dict) -> int:
    for i, r in enumerate(s["rules"]):
        if r["cond"] is None or eval_expr(r["cond"], facts, fv.__getitem__):
            return i
    raise ValueError("no rule fires for this state")


def _deps(s: dict, rule: dict) -> set:
    """Flags the rule depends on, through the flag definitions."""
    if rule["cond"] is None:
        return set()
    defs = {f["name"]: f for f in s["flags"]}
    todo, seen = list(_refs(rule["cond"], "flag")), set()
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            todo += _refs(defs[n]["cond"], "flag")
    return seen


def solve(s: dict, flip: bool = True) -> dict:
    """{"label", "index" (deciding rule), "conflict", "hops_used", "flip_sensitive"} (flip_sensitive only if `flip`)."""
    facts = _facts(s)
    fv = flag_values(s, facts)
    i = _decide(s, facts, fv)
    label = s["rules"][i]["action"]
    fired = sum(1 for r in s["rules"] if r["cond"] is not None and eval_expr(r["cond"], facts, fv.__getitem__))
    depth = {f["name"]: f["depth"] for f in s["flags"]}
    deps = _deps(s, s["rules"][i])
    used = max((depth[n] for n in deps), default=0)
    flat = s["rules"][_decide(s, facts, dict.fromkeys(fv, False))]["action"] if fv else label
    sensitive = False
    for n in sorted(n for n in deps if depth[n] == used and used) if flip else []:
        alt = flag_values(s, facts, (n, not fv[n]))
        if s["rules"][_decide(s, facts, alt)]["action"] != label:
            sensitive = True
            break
    return {"label": label, "index": i, "conflict": fired >= 2, "hops_used": used, "flip_sensitive": sensitive,
            "flat_differs": flat != label}


def wrong_label_hard(s: dict, label: str) -> str:
    """Action of the highest-priority firing rule other than the label's (the answer if the deciding rule were skipped)."""
    facts = _facts(s)
    fv = flag_values(s, facts)
    for r in s["rules"]:
        if r["action"] != label and (r["cond"] is None or eval_expr(r["cond"], facts, fv.__getitem__)):
            return r["action"]
    return HARD_LABELS[(HARD_LABELS.index(label) + 1) % len(HARD_LABELS)]


# ---------------------------------------------------------------- prompt

def render_prompt_hard(s: dict) -> str:
    t = s["task"]
    out = [f"Task: {t['description']}.", f"Target file: {t['target']}",
           f"Task is ambiguous: {_yn(t['ambiguous'])}", f"Target path is known: {_yn(t['path_known'])}",
           "", "Files:"]
    out += [f"- {f['name']}: read={_yn(f['read'])} edited={_yn(f['edited'])}" for f in s["files"]]
    out += ["", "Tests:"] + [f"- {x['name']}: {x['status']}" for x in s["tests"]]
    out += ["", "History (oldest first):"]
    if not s["history"]:
        out.append("(empty)")
    for i, h in enumerate(s["history"], 1):
        out.append(f"{i}. {h['action']} {h['arg']}" + (f" -> {h['result']}" if "result" in h else ""))
    if s["listed"]:
        facts = _facts(s)
        partial = len(s["listed"]) < len(FACT_KEYS)
        out += ["", "Status (only these facts are listed; work out the others from the files, tests and history):"
                if partial else "Status:"]
        out += [f"{k}={facts[k]}" for k in s["listed"]]
    if s["flags"]:
        out += ["", "Flags:", "A name in a condition that has no '=' is a flag. It holds when its definition here holds."]
        out += [f"{f['name']} holds if {render_expr(f['cond'])} [end]" for f in s["flags"]]
    out += ["", "Disabled rules (never apply these):"]
    out += [f"Disabled {i}: if {render_expr(r['cond'])}, then {r['action']} [end]"
            for i, r in enumerate(s["disabled"], 1)] or ["(none)"]
    out += ["", "Rules (highest priority first; the first rule whose condition holds decides):"]
    for i, r in enumerate(s["rules"], 1):
        cond = "otherwise" if r["cond"] is None else f"if {render_expr(r['cond'])}"
        out.append(f"Rule {i}: {cond}, then {r['action']} [end]")
    return "\n".join(out)


def template_id(s: dict) -> str:
    """Skeleton shared by items, as the base task's set of facts: the facts the flag definitions and the rules read.
    Whole skeletons go to one split, so a held-out program never reuses a training skeleton (flag names, polarity,
    operators, order, actions and the state vary inside one)."""
    exprs = [f["cond"] for f in s["flags"]] + [r["cond"] for r in s["rules"] if r["cond"] is not None]
    return "+".join(sorted(set().union(set(), *(_refs(e, "fact") for e in exprs))))


def _sample_program(rng: random.Random, k: HardKnobs) -> dict:
    flags = _sample_flags(rng, k)
    rules, disabled = _sample_rules(rng, k, flags)
    return {"flags": flags, "rules": rules, "disabled": disabled}


def _sample_item(rng: random.Random, k: HardKnobs, program: dict | None = None) -> dict:
    s = _sample_state(rng, k)
    s.update(program or _sample_program(rng, k))
    if k.hide_status:
        s["listed"] = []
    elif k.status_share < 1:
        keep = set(rng.sample(FACT_KEYS, round(k.status_share * len(FACT_KEYS))))
        s["listed"] = [f for f in FACT_KEYS if f in keep]
    else:
        s["listed"] = list(FACT_KEYS)
    return s


def generate_hard(n: int, seed: int, knobs: HardKnobs | None = None, variant: str = "rule") -> list[dict]:
    if variant != "rule":
        raise ValueError(f"the hard family has only the 'rule' variant, got {variant!r}")
    k = knobs or HardKnobs()
    if k.n_rules > len(HARD_LABELS) or k.n_flags < k.hops or (k.hops and not k.n_flags):
        raise ValueError(f"inconsistent knobs: {k}")
    rng = random.Random(seed)
    remaining: list[str] = []
    recs = []
    for i in range(n):
        if not remaining:
            remaining = HARD_LABELS[:]
        want_conf = rng.random() < k.conflict_rate
        deep = k.hops > 0 and rng.random() < DEEP_SHARE
        flat = k.hops > 0 and rng.random() < FLAT_SHARE
        tries = 0
        while True:
            program = _sample_program(rng, k)   # one program is tried on several states before a new one is drawn
            for _ in range(PROGRAM_REUSE):
                tries += 1
                if tries > MAX_TRIES:
                    raise RuntimeError(f"could not generate item {i} with {k}")
                s = _sample_item(rng, k, program)
                r = solve(s, flip=False)
                if r["label"] not in remaining:
                    continue
                if r["index"] == len(s["rules"]) - 1 and rng.random() > CATCH_KEEP:
                    continue
                if tries < 400 and r["conflict"] != want_conf:
                    continue
                if (deep and r["hops_used"] != k.hops) or (flat and not r["flat_differs"]):
                    continue
                r = solve(s)
                if deep and not r["flip_sensitive"]:
                    continue
                break
            else:
                continue
            break
        remaining.remove(r["label"])
        recs.append({"id": f"{seed}-{i}", "prompt": render_prompt_hard(s), "label": r["label"],
                     "wrong_label": wrong_label_hard(s, r["label"]), "knobs": asdict(k), "state": s,
                     "variant": variant, "template_id": template_id(s), "hops_used": r["hops_used"],
                     "flip_sensitive": r["flip_sensitive"], "flat_differs": r["flat_differs"], "decided_by": r["index"] + 1})
    return recs


# ---------------------------------------------------------------- ladder

def _k(**kw) -> HardKnobs:
    return HardKnobs(**kw)


# Study B2 rungs, in order (all twelve labels; distractors and history as the old R3).
HTIERS = {
    "H1": _k(n_rules=8, n_flags=0, hops=0),
    "H2": _k(n_rules=8, n_flags=2, hops=1),
    "H3": _k(n_rules=8, n_flags=4, hops=2),
    "H4": _k(n_rules=8, n_flags=4, hops=2, hide_status=True),
    "H5": _k(n_rules=12, n_flags=6, hops=3, hide_status=True),
}
# Fixed before any run: the rung tried when a rung drops from above 90 to below 60, keyed by the upper rung.
BETWEEN = {
    "H1": ("H0", _k(n_rules=6, n_flags=0, hops=0, or_not=False)),
    "H2": ("H1b", _k(n_rules=8, n_flags=1, hops=1)),
    "H3": ("H2b", _k(n_rules=8, n_flags=3, hops=2)),
    "H4": ("H3b", _k(n_rules=8, n_flags=4, hops=2, status_share=0.5)),
    "H5": ("H4b", _k(n_rules=10, n_flags=5, hops=3, hide_status=True)),
}
BETWEEN_KNOBS = {name: kn for name, kn in BETWEEN.values()}
ALL_KNOBS = {**HTIERS, **BETWEEN_KNOBS}
