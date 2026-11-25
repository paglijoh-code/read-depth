"""M9 atlas: generators for the 14 tagged tasks and the template-disjoint splits (docs/ATLAS-phase0.md, Brief A).

    generate(task, n, seed)             -> n records of one task (schema in spec.py)
    generate_pair(appraise_task, n, seed) -> (appraise records, execute records) over the SAME base items
    make_splits(family, seed, sizes)    -> {split: [items]}; an item holds the record of each task of its family

A "family" is one task, or the two tasks of a matched pair, built from the same base items (same ids, same user text,
same template). Splits are dealt by whole template, so no template is in two splits of a task or of its pair; every
split is then filled to its size by "cell" quotas (the answer, or the pair of answers) so the answers stay balanced.
Everything is seeded and uses no set or hash order, so two runs give identical bytes.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import random
import re
import sys
import zlib
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from surgery.atlas import spec
from surgery.task.generator import HARD_LABELS, _cond_holds, _facts, firing_rules
from surgery.task.generator import generate as generate_rule
from surgery.task.hard import HTIERS, generate_hard

ROOT = Path(__file__).resolve().parents[2]
ROUTING_DIR = ROOT / "var" / "routing" / "v5"      # the only routing folder this module may read
ROUTING_FILES = ("generated", "wild_train", "wild_test")

FAMILY_TASKS = {
    "code": ("code_shape", "code_output"), "fact": ("fact_topic", "fact_check"), "hard": ("hard_hops", "hard"),
    "arith": ("arith_shape", "arith"), "syntax": ("syntax_depth", "syntax_valid"),
    "rule": ("rule",), "sensible": ("sensible",),
    "route_judgement": ("route_judgement",), "route_work": ("route_work",),
    # Phase 2 stage A: one task per family, after the nine Phase 1 families (dict order matters to the build's default)
    "arith_order": ("arith_order",), "syntax_first_close": ("syntax_first_close",), "code_uses_first": ("code_uses_first",),
    "hard_target_read": ("hard_target_read",), "arith_id": ("arith_id",), "code_literal": ("code_literal",),
    "arith_long": ("arith_long",),
    # Phase 2 stage C (docs/ATLAS-phase2-stageC.md)
    "code_traces_first": ("code_traces_first",), "syntax_outer_pair": ("syntax_outer_pair",),
    "target_read_bare": ("target_read_bare",), "bind_compare": ("bind_compare",),
    # Robustness E2 (docs/ATLAS-robustness.md): check tasks over the do-tasks' items, after the stage C families
    "rule_check": ("rule_check",), "hard_check": ("hard_check",), "arith_check": ("arith_check",),
    "code_output_check": ("code_output_check",), "code_literal_check": ("code_literal_check",),
    # E2 revision R1: binary doing tasks ("is the answer one of A?"), same items as the doing tasks
    "rule_bin": ("rule_bin",), "hard_bin": ("hard_bin",), "arith_bin": ("arith_bin",),
    "code_output_bin": ("code_output_bin",), "code_literal_bin": ("code_literal_bin",),
}
PHASE1_FAMILIES = tuple(list(FAMILY_TASKS)[:9])
NEW_FAMILIES = tuple(list(FAMILY_TASKS)[9:16])      # stage A (manifest2.json)
C_FAMILIES = tuple(list(FAMILY_TASKS)[16:20])       # stage C (manifest3.json)
E2_FAMILIES = tuple(list(FAMILY_TASKS)[20:25])      # robustness E2 check tasks (manifest4.json, written by scripts/atlas_e2_gate.py --build)
BIN_FAMILIES = tuple(list(FAMILY_TASKS)[25:])       # E2 revision R1 binary doing tasks (same manifest)
CHECK_BASE = {"rule_check": "rule", "hard_check": "hard", "arith_check": "arith",
              "code_output_check": "code", "code_literal_check": "code_literal"}   # check family -> the do-task's family
CHECK_DO = {"rule_check": "rule", "hard_check": "hard", "arith_check": "arith",
            "code_output_check": "code_output", "code_literal_check": "code_literal"}   # check task -> the do task
BIN_BASE = {"rule_bin": "rule", "hard_bin": "hard", "arith_bin": "arith",
            "code_output_bin": "code", "code_literal_bin": "code_literal"}   # bin family -> the do-task's family
BIN_DO = {"rule_bin": "rule", "hard_bin": "hard", "arith_bin": "arith",
          "code_output_bin": "code_output", "code_literal_bin": "code_literal"}   # bin task -> the do task
BIN_OF_CHECK = {"rule_check": "rule_bin", "hard_check": "hard_bin", "arith_check": "arith_bin",
                "code_output_check": "code_output_bin", "code_literal_check": "code_literal_bin"}
SIZED = ("rule", "sensible") + NEW_FAMILIES + C_FAMILIES + E2_FAMILIES + BIN_FAMILIES     # families whose items are drawn n at a time (the others are fixed pools)
FAMILY_OF = {t: f for f, ts in FAMILY_TASKS.items() for t in ts}
STEPS_TASKS = ("steps_copy", "steps_ops", "steps_describe",   # built by surgery/atlas/steps.py, not here
               "steps3_swap", "steps3_total", "steps3_order", "steps3_mix")   # built by surgery/atlas/steps3.py
assert sorted(FAMILY_OF) == sorted(set(spec.TASKS) - set(STEPS_TASKS)), set(FAMILY_OF) ^ (set(spec.TASKS) - set(STEPS_TASKS))
NATURAL = ("route_judgement", "route_work", "arith_shape", "arith_long")   # answer balance is whatever the design gives
POOL_FACTORS = (2.0, 3.0, 4.5, 7.0, 10.0)


def _rng(*parts) -> random.Random:
    return random.Random("-".join(str(p) for p in parts))


def _record(task: str, base_id: str, prompt: str, answer: int, template: str, rung, meta: dict) -> dict:
    assert 0 <= answer < len(spec.TASKS[task].options), (task, answer)
    return {"id": base_id, "task": task, "prompt": prompt, "answer": int(answer), "template_id": template,
            "rung": rung, "meta": meta}


def _spread(keys: list) -> dict:
    """Rank of each key so that any prefix of the order is balanced on both parts of a pair key (diagonals of the grid)."""
    if not all(isinstance(k, tuple) and len(k) == 2 for k in keys):
        return {k: i for i, k in enumerate(sorted(keys, key=repr))}
    a, b = sorted({k[0] for k in keys}, key=repr), sorted({k[1] for k in keys}, key=repr)
    m = max(len(a), len(b))
    order = sorted(keys, key=lambda k: ((a.index(k[0]) + b.index(k[1])) % m, a.index(k[0]), b.index(k[1])))
    return {k: i for i, k in enumerate(order)}


def _quotas(total: int, weights: dict) -> dict:
    """Split `total` over keys by weight (largest remainder; ties go to the diagonal order of `_spread`)."""
    keys = sorted(weights, key=repr)
    rank = _spread(keys)
    s = sum(weights.values())
    raw = {k: total * weights[k] / s for k in keys}
    out = {k: int(math.floor(raw[k])) for k in keys}
    for k in sorted(keys, key=lambda k: (-round(raw[k] - out[k], 9), rank[k]))[:total - sum(out.values())]:
        out[k] += 1
    return out


# ================================================================ code (tasks 3 and 12)

_NAMES = ["x", "y", "z", "a", "b", "c", "n", "m", "s", "t", "total", "count", "acc", "val", "res", "num", "p", "q",
          "r", "w", "u", "v", "h", "d", "score", "size", "step", "base"]
_LOOPVARS = ["i", "j", "k", "idx", "e"]
_KW = {"for", "while", "if", "else", "in", "range", "print", "abs", "max", "min"}
SHAPES = ("loop", "branch", "neither")


def run_py(code: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {})
    return buf.getvalue().strip()


def _atom(rng, vs, lo=0, hi=9):
    return rng.choice(vs) if vs and rng.random() < 0.65 else str(rng.randint(lo, hi))


def _expr(rng, vs):
    kind = rng.random()
    a = _atom(rng, vs)
    if kind < 0.7:
        return f"{a} {rng.choice('+-*')} {_atom(rng, vs, 1, 5)}"
    if kind < 0.85:
        return f"{a} // {rng.randint(2, 4)}"
    if kind < 0.95:
        return f"{a} % {rng.randint(2, 6)}"
    return a


def _print(rng, vs):
    v = vs[-1]
    w = rng.choice(vs)
    form = rng.choice(["{v}", "{v}", "{v} % {k}", "{v} % 10", "{v} // {k}", "abs({v})", "{v} + {w}", "{v} - {w}",
                       "{v} * {w}", "max({v}, {w})", "min({v}, {w})"])
    return "print(" + form.format(v=v, w=w, k=rng.randint(2, 5)) + ")"


def _update(rng, target, vs):
    if rng.random() < 0.3:
        return f"{target} {rng.choice(['+=', '-=', '*='])} {_atom(rng, [v for v in vs if v != target], 1, 4)}"
    return f"{target} = {_expr(rng, vs)}"


def _program(rng, shape: str) -> str:
    names = rng.sample(_NAMES, 4)
    loopv = rng.choice(_LOOPVARS)
    lines: list[str] = []
    if shape == "neither":
        n_stmt = rng.randint(2, 5)
        vs = [names[0]]
        lines.append(f"{names[0]} = {rng.randint(0, 9)}")
        for i in range(n_stmt - 2):
            if rng.random() < 0.6 and len(vs) < 3:
                nv = names[len(vs)]
                lines.append(f"{nv} = {_expr(rng, vs)}")
                vs.append(nv)
            else:
                lines.append(_update(rng, rng.choice(vs), vs))
        lines.append(_print(rng, vs))
    elif shape == "loop":
        acc, other = names[0], names[1]
        vs = [acc]
        lines.append(f"{acc} = {rng.randint(0, 5)}")
        if rng.random() < 0.4:
            lines.append(f"{other} = {rng.randint(1, 4)}")
            vs.append(other)
        body_vars = vs + [loopv]
        if rng.random() < 0.6:
            n = rng.randint(2, 5)
            head = f"for {loopv} in range({n}):" if rng.random() < 0.6 else f"for {loopv} in range({rng.randint(1, 3)}, {rng.randint(4, 6)}):"
            body = []
        else:
            n = rng.randint(2, 5)
            lines.append(f"{loopv} = 0")
            head = f"while {loopv} < {n}:"
            body = [f"{loopv} += 1" if rng.random() < 0.5 else f"{loopv} = {loopv} + 1"]
        for _ in range(rng.randint(1, 2)):
            tgt = acc if rng.random() < 0.7 else rng.choice(vs)
            if rng.random() < 0.5:
                body.append(f"{tgt} {rng.choice(['+=', '-=', '+='])} {_atom(rng, body_vars, 1, 4)}")
            else:
                body.append(f"{tgt} = {_atom(rng, body_vars)} {rng.choice('+-*')} {_atom(rng, body_vars, 1, 3)}")
        if head.startswith("while") and rng.random() < 0.5:
            body = body[1:] + body[:1]            # counter step last
        lines.append(head)
        lines += ["    " + b for b in body]
        lines.append(_print(rng, vs))
    else:
        r = names[3]
        vs = [names[0]]
        lines.append(f"{names[0]} = {rng.randint(0, 9)}")
        if rng.random() < 0.6:
            lines.append(f"{names[1]} = {rng.randint(0, 9)}")
            vs.append(names[1])
        v = rng.choice(vs)
        cond = rng.choice([f"{v} > {rng.randint(1, 8)}", f"{v} < {rng.randint(2, 8)}", f"{v} == {rng.randint(0, 9)}",
                           f"{v} % 2 == 0", f"{v} != {rng.randint(0, 9)}", f"{v} >= {_atom(rng, vs)}"])
        lines.append(f"if {cond}:")
        lines.append(f"    {r} = {_expr(rng, vs)}")
        lines.append("else:")
        lines.append(f"    {r} = {_expr(rng, vs)}")
        lines.append(_print(rng, vs + [r]))
    return "\n".join(lines)


def skeleton(code: str) -> str:
    """The program with variable names numbered by first appearance and numbers replaced by N."""
    seen: dict[str, str] = {}

    def sub(m):
        t = m.group(0)
        if t[0].isdigit():
            return "N"
        if t in _KW:
            return t
        return seen.setdefault(t, f"V{len(seen)}")
    return re.sub(r"[A-Za-z_]\w*|\d+", sub, code)


def code_shape_of(code: str) -> str:
    has_loop = bool(re.search(r"^\s*(for|while)\b", code, re.M))
    has_if = bool(re.search(r"^\s*if\b", code, re.M))
    assert not (has_loop and has_if), code
    return "loop" if has_loop else "branch" if has_if else "neither"


def _code_item(rng, shape, digit, tries=4000):
    for _ in range(tries):
        code = _program(rng, shape)
        try:
            out = run_py(code)
        except Exception:
            continue
        if re.fullmatch(r"\d", out) and int(out) == digit:
            assert code_shape_of(code) == shape
            return code
    return None


# ================================================================ syntax (tasks 11 and 14)

_PAIRS = ["()", "[]", "{}"]
_JKEYS = ["name", "id", "age", "city", "tags", "price", "active", "count", "color", "size", "year", "score"]
_JSCAL = ['"red"', '"Paris"', '"north"', "42", "7", "3.5", "0.25", "true", "false", "null"]


def _balanced(s: str) -> bool:
    st, pairs = [], {")": "(", "]": "[", "}": "{"}
    for c in s:
        if c in "([{":
            st.append(c)
        elif c in pairs:
            if not st or st.pop() != pairs[c]:
                return False
    return not st


def bracket_depth(s: str) -> int:
    """Most brackets open at once, reading left to right (a stray closer never goes below zero)."""
    cur = best = 0
    for c in s:
        if c in "([{":
            cur += 1
            best = max(best, cur)
        elif c in ")]}":
            cur = max(0, cur - 1)
    return best


def _bseq(rng, d: int) -> str:
    depths = [d] + [rng.randint(1, d) for _ in range(rng.randint(0, 3 if d == 1 else 2))]
    rng.shuffle(depths)
    return "".join(_bnode(rng, e) for e in depths)


def _bnode(rng, e: int) -> str:
    o, c = rng.choice(_PAIRS)
    return o + (_bseq(rng, e - 1) if e > 1 else "") + c


def _jval(rng, d: int, top: bool = False) -> str:
    """JSON value with bracket depth exactly d >= 1."""
    as_obj = top or rng.random() < 0.55
    n = rng.randint(2, 4) if as_obj else rng.randint(1, 3)
    kids = [rng.choice(_JSCAL) for _ in range(n)]
    if d > 1:
        kids[rng.randrange(n)] = _jval(rng, d - 1)
        for i in range(n):
            if rng.random() < 0.2 and d > 2:
                kids[i] = _jval(rng, rng.randint(1, d - 1))
    if as_obj:
        return "{" + ", ".join(f'"{k}": {v}' for k, v in zip(rng.sample(_JKEYS, n), kids)) + "}"
    return "[" + ", ".join(kids) + "]"


def _json_valid(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


def syntax_skeleton(s: str) -> str:
    if s[:1] in "{[" and ('"' in s or "'" in s or ":" in s):
        t = re.sub(r'"[^"]*"|\'[^\']*\'', lambda m: "Q" if m.group(0)[0] == '"' else "q", s)
        t = re.sub(r"-?\d+\.\d+", "D", t)
        t = re.sub(r"-?\d+", "N", t)
        t = re.sub(r"\b[A-Za-z_]+\b(?<![QqDN])", "W", t)
        return "J" + t
    return "B" + "".join("o" if c in "([{" else "c" for c in s)


def _syntax_item(rng, depth: int, valid: int, tries: int = 400):
    """valid: 0 = well-formed (yes), 1 = broken (no). Returns the string or None."""
    _scripts()
    from build_canaries import break_brackets, break_json   # scripts/build_canaries.py
    for _ in range(tries):
        if rng.random() < 0.5:
            s = _bseq(rng, depth)
            assert _balanced(s)
            if valid:
                s = break_brackets(rng, s)
                assert not _balanced(s)
        else:
            s = _jval(rng, depth, top=True)
            assert _json_valid(s)
            if valid:
                s = break_json(rng, s)
                assert not _json_valid(s)
        if bracket_depth(s) == depth:
            return s
    return None


# ================================================================ arithmetic (tasks 9 and 13)

_OPSYM = {"plus": "+", "minus": "-", "times": "*"}
_PREFIX = ["", "Compute: ", "What is ", "Evaluate ", "Calculate: ", "Find the value of ", "Work out ", "Simplify ",
           "Solve: ", "Answer: "]
_WHAT_IS = 2
_APPLY = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}


def _arith_text(a, b, c, op1, op2, prefix, tight, full, wrap=False):
    sp = "" if tight else " "
    first = f"{a}{sp}{op1}{sp}{b}"
    if op2 is None:
        body = first
    else:
        need = op1 in "+-" and op2 == "*"
        body = (f"({first})" if (need or full) else first) + f"{sp}{op2}{sp}{c}"
    if wrap:
        body = f"({body})"
    text = _PREFIX[prefix] + body + ("?" if prefix == _WHAT_IS else "")
    return text, body


def _arith_item(rng, opclass: str, digit: int, tries: int = 20000):
    for _ in range(tries):
        a, b, c = rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 9)
        if opclass == "mixed":
            op1, op2 = rng.sample(["+", "-", "*"], 2)
            v1 = _APPLY[op1](a, b)
            if v1 < 0:
                continue
            val = _APPLY[op2](v1, c)
        else:
            op1, op2 = _OPSYM[opclass], None
            val = _APPLY[op1](a, b)
        if val != digit:
            continue
        prefix, tight, full, wrap = rng.randrange(len(_PREFIX)), rng.random() < 0.5, rng.random() < 0.5, rng.random() < 0.3
        text, body = _arith_text(a, b, c, op1, op2, prefix, tight, full, wrap)
        assert eval(body) == val and 0 <= val <= 9, (body, val)   # python's precedence agrees with left to right
        if op2 is None:
            bucket = "lt" if a < b else "eq" if a == b else "gt"
            pattern = op1
        else:
            vals = sorted({a, b, c})
            bucket = "".join(str(vals.index(x)) for x in (a, b, c))
            pattern = op1 + op2
        fmt = f"{prefix}{'t' if tight else 's'}{'f' if (full and op2) else 'm'}{'w' if wrap else ''}"
        return text, f"{pattern}|{bucket}|{fmt}", {"a": a, "b": b, "c": c if op2 else None, "ops": pattern}
    return None


# ================================================================ facts (tasks 4 and 10)

def _lines(block: str) -> list[list[str]]:
    return [[f.strip() for f in ln.split(">")] for ln in block.strip().splitlines() if ln.strip()]


def _art(w: str) -> str:
    return ("an " if w[0].lower() in "aeiou" else "a ") + w


# (topic, relation, phrasings with {s} {o}, data lines "subject>object"[">tag"]). Facts the author is sure of; doubtful ones dropped.
_BANK = []


def _rel(topic, name, phrasings, block):
    _BANK.append((topic, name, phrasings, _lines(block)))


_rel("geography", "capital", [
    "Is {o} the capital of {s}?", "Is the capital of {s} {o}?", "Does {s} have {o} as its capital?",
    "Is it correct that {o} is the capital of {s}?", "Would you say {o} is the capital city of {s}?"], """
France>Paris
Japan>Tokyo
Italy>Rome
Spain>Madrid
Germany>Berlin
Canada>Ottawa
Egypt>Cairo
Russia>Moscow
China>Beijing
India>New Delhi
Australia>Canberra
Brazil>Brasília
Argentina>Buenos Aires
Kenya>Nairobi
Greece>Athens
Portugal>Lisbon
Norway>Oslo
Sweden>Stockholm
Poland>Warsaw
Turkey>Ankara
Thailand>Bangkok
Peru>Lima
Cuba>Havana
Ireland>Dublin
Austria>Vienna
Hungary>Budapest
South Korea>Seoul
Chile>Santiago
Vietnam>Hanoi
Morocco>Rabat
Finland>Helsinki
Denmark>Copenhagen
Netherlands>Amsterdam
Belgium>Brussels
Switzerland>Bern
Czech Republic>Prague
Romania>Bucharest
Bulgaria>Sofia
Ukraine>Kyiv
Iran>Tehran
Iraq>Baghdad
Saudi Arabia>Riyadh
Pakistan>Islamabad
Bangladesh>Dhaka
Indonesia>Jakarta
Philippines>Manila
Malaysia>Kuala Lumpur
Mexico>Mexico City
Colombia>Bogotá
Venezuela>Caracas
Ecuador>Quito
Uruguay>Montevideo
Paraguay>Asunción
Nigeria>Abuja
Ghana>Accra
Ethiopia>Addis Ababa
Algeria>Algiers
Tunisia>Tunis
Libya>Tripoli
Zimbabwe>Harare
Zambia>Lusaka
Uganda>Kampala
Angola>Luanda
Senegal>Dakar
Mongolia>Ulaanbaatar
Nepal>Kathmandu
Afghanistan>Kabul
Cambodia>Phnom Penh
New Zealand>Wellington
Croatia>Zagreb
Serbia>Belgrade
Slovakia>Bratislava
Slovenia>Ljubljana
Iceland>Reykjavik
Estonia>Tallinn
Latvia>Riga
Lithuania>Vilnius
Jordan>Amman
Lebanon>Beirut
Syria>Damascus
North Korea>Pyongyang
Jamaica>Kingston
Panama>Panama City
Costa Rica>San José
Mozambique>Maputo
Madagascar>Antananarivo
Namibia>Windhoek
Cameroon>Yaoundé
Qatar>Doha
Kuwait>Kuwait City
Oman>Muscat
Belarus>Minsk
Armenia>Yerevan
Azerbaijan>Baku
Uzbekistan>Tashkent
Laos>Vientiane
Malta>Valletta
Cyprus>Nicosia
Fiji>Suva
Haiti>Port-au-Prince
Albania>Tirana
Sudan>Khartoum
Rwanda>Kigali
Bhutan>Thimphu
""")
_rel("geography", "continent", [
    "Is {s} in {o}?", "Is {s} a country in {o}?", "Does {s} lie in {o}?",
    "Is it correct that {s} is located in {o}?", "Can {s} be found in {o}?"], """
France>Europe
Germany>Europe
Italy>Europe
Spain>Europe
Poland>Europe
Sweden>Europe
Norway>Europe
Finland>Europe
Portugal>Europe
Greece>Europe
Ireland>Europe
Hungary>Europe
Austria>Europe
Netherlands>Europe
Belgium>Europe
Denmark>Europe
Japan>Asia
China>Asia
India>Asia
Thailand>Asia
Vietnam>Asia
South Korea>Asia
Mongolia>Asia
Nepal>Asia
Pakistan>Asia
Bangladesh>Asia
Philippines>Asia
Iran>Asia
Iraq>Asia
Cambodia>Asia
Kenya>Africa
Nigeria>Africa
Ghana>Africa
Ethiopia>Africa
Morocco>Africa
Algeria>Africa
Uganda>Africa
Zimbabwe>Africa
Madagascar>Africa
Senegal>Africa
Brazil>South America
Argentina>South America
Chile>South America
Peru>South America
Colombia>South America
Venezuela>South America
Uruguay>South America
Mexico>North America
Canada>North America
Cuba>North America
Jamaica>North America
Australia>Oceania
New Zealand>Oceania
Fiji>Oceania
""")
_rel("geography", "river", [
    "Is {s} a river in {o}?", "Does {s} flow through {o}?", "Is it correct that {s} is a river of {o}?",
    "Would you find {s} in {o}?", "Is {s} one of the rivers of {o}?"], """
the Nile>Africa
the Congo>Africa
the Niger>Africa
the Zambezi>Africa
the Limpopo>Africa
the Orange River>Africa
the Amazon>South America
the Orinoco>South America
the Paraná>South America
the Magdalena>South America
the Mississippi>North America
the Missouri>North America
the Colorado River>North America
the Rio Grande>North America
the Yukon>North America
the Columbia River>North America
the Hudson>North America
the Yangtze>Asia
the Yellow River>Asia
the Mekong>Asia
the Ganges>Asia
the Indus>Asia
the Brahmaputra>Asia
the Tigris>Asia
the Euphrates>Asia
the Irrawaddy>Asia
the Amur>Asia
the Lena>Asia
the Yenisei>Asia
the Danube>Europe
the Rhine>Europe
the Volga>Europe
the Seine>Europe
the Thames>Europe
the Loire>Europe
the Elbe>Europe
the Po>Europe
the Tagus>Europe
the Ebro>Europe
the Dnieper>Europe
the Vistula>Europe
the Oder>Europe
the Tiber>Europe
the Shannon>Europe
the Murray>Oceania
the Darling>Oceania
""")

_rel("history", "event", [
    "Did {s} take place in {o}?", "Was {o} the year of {s}?", "Is it correct that {s} happened in {o}?",
    "Did {s} occur in {o}?", "Would {o} be the right year for {s}?"], """
the signing of Magna Carta>1215
the Battle of Hastings>1066
the fall of Constantinople>1453
Columbus's first voyage to the Americas>1492
the posting of Luther's Ninety-five Theses>1517
the defeat of the Spanish Armada>1588
the founding of Jamestown>1607
the voyage of the Mayflower>1620
the Great Fire of London>1666
the Boston Tea Party>1773
the signing of the American Declaration of Independence>1776
the storming of the Bastille>1789
the Louisiana Purchase>1803
Napoleon's coronation as emperor>1804
the Battle of Trafalgar>1805
the Battle of Waterloo>1815
the Meiji Restoration>1868
the opening of the Suez Canal>1869
the Great Chicago Fire>1871
the eruption of Krakatoa>1883
the dedication of the Statue of Liberty>1886
the Wright brothers' first powered flight>1903
the sinking of the Titanic>1912
the start of the First World War>1914
the opening of the Panama Canal>1914
the Russian Revolution>1917
the signing of the Treaty of Versailles>1919
the Wall Street Crash>1929
the discovery of penicillin>1928
the start of the Second World War>1939
the attack on Pearl Harbor>1941
the D-Day landings>1944
the end of the Second World War>1945
the independence of India>1947
the founding of the People's Republic of China>1949
the start of the Korean War>1950
the launch of Sputnik>1957
the first human spaceflight by Yuri Gagarin>1961
the building of the Berlin Wall>1961
the Cuban Missile Crisis>1962
the assassination of John F. Kennedy>1963
the first Moon landing>1969
the fall of Saigon>1975
the Chernobyl disaster>1986
the fall of the Berlin Wall>1989
the release of Nelson Mandela from prison>1990
the dissolution of the Soviet Union>1991
the September 11 attacks>2001
the Battle of Gettysburg>1863
the start of the American Civil War>1861
the publication of On the Origin of Species>1859
the Battle of Agincourt>1415
the fall of the Western Roman Empire>476
the coronation of Charlemagne as emperor>800
the Xinhai Revolution>1911
the Easter Rising>1916
the start of the Spanish Civil War>1936
the Suez Crisis>1956
the Prague Spring>1968
the assassination of Martin Luther King Jr.>1968
the Iranian Revolution>1979
the Falklands War>1982
the Rwandan genocide>1994
the handover of Hong Kong to China>1997
the start of the Irish Potato Famine>1845
the start of the Crimean War>1853
the start of the Franco-Prussian War>1870
the assassination of Archduke Franz Ferdinand>1914
the launch of the Hubble Space Telescope>1990
the abolition of slavery by the Thirteenth Amendment>1865
""")
_rel("history", "leader", [
    "Was {s} a leader of {o}?", "Did {s} lead {o}?", "Is {s} known as a leader of {o}?",
    "Is it correct that {s} led {o}?", "Was {o} once led by {s}?"], """
Napoleon Bonaparte>France>E
Charles de Gaulle>France>E
Louis XIV>France>E
Napoleon III>France>E
Winston Churchill>the United Kingdom>E
Margaret Thatcher>the United Kingdom>E
Queen Victoria>the United Kingdom>E
Tony Blair>the United Kingdom>E
Clement Attlee>the United Kingdom>E
Henry VIII>England>E
Elizabeth I>England>E
Oliver Cromwell>England>E
Abraham Lincoln>the United States>N
George Washington>the United States>N
Franklin D. Roosevelt>the United States>N
Theodore Roosevelt>the United States>N
John F. Kennedy>the United States>N
Ronald Reagan>the United States>N
Thomas Jefferson>the United States>N
Dwight D. Eisenhower>the United States>N
Barack Obama>the United States>N
Richard Nixon>the United States>N
Jimmy Carter>the United States>N
Lyndon B. Johnson>the United States>N
Vladimir Lenin>the Soviet Union>E
Joseph Stalin>the Soviet Union>E
Mikhail Gorbachev>the Soviet Union>E
Nikita Khrushchev>the Soviet Union>E
Leonid Brezhnev>the Soviet Union>E
Peter the Great>Russia>E
Catherine the Great>Russia>E
Ivan the Terrible>Russia>E
Vladimir Putin>Russia>E
Boris Yeltsin>Russia>E
Benito Mussolini>Italy>E
Adolf Hitler>Germany>E
Otto von Bismarck>Germany>E
Angela Merkel>Germany>E
Konrad Adenauer>Germany>E
Helmut Kohl>Germany>E
Willy Brandt>Germany>E
Julius Caesar>Rome>E
Augustus>Rome>E
Francisco Franco>Spain>E
Cleopatra>Egypt>F
Ramesses II>Egypt>F
Gamal Abdel Nasser>Egypt>F
Anwar Sadat>Egypt>F
Mao Zedong>China>A
Deng Xiaoping>China>A
Sun Yat-sen>China>A
Xi Jinping>China>A
Genghis Khan>the Mongol Empire>A
Kublai Khan>the Mongol Empire>A
Jawaharlal Nehru>India>A
Indira Gandhi>India>A
Narendra Modi>India>A
Ho Chi Minh>Vietnam>A
Kim Il-sung>North Korea>A
Sukarno>Indonesia>A
Lee Kuan Yew>Singapore>A
Golda Meir>Israel>A
David Ben-Gurion>Israel>A
Mustafa Kemal Atatürk>Turkey>A
Suleiman the Magnificent>the Ottoman Empire>A
Saddam Hussein>Iraq>A
Emperor Hirohito>Japan>A
Pol Pot>Cambodia>A
Fidel Castro>Cuba>N
Pierre Trudeau>Canada>N
Justin Trudeau>Canada>N
John A. Macdonald>Canada>N
Juan Perón>Argentina>S
Salvador Allende>Chile>S
Augusto Pinochet>Chile>S
Hugo Chávez>Venezuela>S
Getúlio Vargas>Brazil>S
Nelson Mandela>South Africa>F
Kwame Nkrumah>Ghana>F
Jomo Kenyatta>Kenya>F
Haile Selassie>Ethiopia>F
Muammar Gaddafi>Libya>F
Idi Amin>Uganda>F
Julius Nyerere>Tanzania>F
Robert Menzies>Australia>O
Helen Clark>New Zealand>O
Éamon de Valera>Ireland>E
António de Oliveira Salazar>Portugal>E
Josip Broz Tito>Yugoslavia>E
Lech Wałęsa>Poland>E
""")

_rel("science", "element", [
    "Is {o} the chemical symbol for {s}?", "Is the chemical symbol of {s} {o}?",
    "Does {o} stand for {s} in the periodic table?", "Is it correct that {s} has the symbol {o}?",
    "Would you write {s} as {o} in a chemical formula?"], """
hydrogen>H
helium>He
lithium>Li
beryllium>Be
boron>B
carbon>C
nitrogen>N
oxygen>O
fluorine>F
neon>Ne
sodium>Na
magnesium>Mg
aluminum>Al
silicon>Si
phosphorus>P
sulfur>S
chlorine>Cl
argon>Ar
potassium>K
calcium>Ca
scandium>Sc
titanium>Ti
vanadium>V
chromium>Cr
manganese>Mn
iron>Fe
cobalt>Co
nickel>Ni
copper>Cu
zinc>Zn
gallium>Ga
germanium>Ge
arsenic>As
selenium>Se
bromine>Br
krypton>Kr
rubidium>Rb
strontium>Sr
zirconium>Zr
molybdenum>Mo
rhodium>Rh
palladium>Pd
silver>Ag
cadmium>Cd
tin>Sn
antimony>Sb
tellurium>Te
iodine>I
xenon>Xe
cesium>Cs
barium>Ba
lanthanum>La
neodymium>Nd
tantalum>Ta
tungsten>W
osmium>Os
iridium>Ir
platinum>Pt
gold>Au
mercury>Hg
lead>Pb
bismuth>Bi
radon>Rn
radium>Ra
thorium>Th
uranium>U
plutonium>Pu
""")
_rel("science", "planet_order", [
    "Is {s} the {o} planet from the Sun?", "Counting outward from the Sun, is {s} the {o} planet?",
    "Is it correct that {s} is the {o} planet from the Sun?", "Would {s} be called the {o} planet from the Sun?",
    "Is the {o} planet from the Sun {s}?"], """
Mercury>first
Venus>second
Earth>third
Mars>fourth
Jupiter>fifth
Saturn>sixth
Uranus>seventh
Neptune>eighth
""")
_rel("science", "moon", [
    "Is {s} a moon of {o}?", "Does {o} have a moon called {s}?", "Is it correct that {s} orbits {o}?",
    "Is {s} one of the natural satellites of {o}?", "Does {s} orbit the planet {o}?"], """
Phobos>Mars
Deimos>Mars
Io>Jupiter
Europa>Jupiter
Ganymede>Jupiter
Callisto>Jupiter
Titan>Saturn
Enceladus>Saturn
Mimas>Saturn
Rhea>Saturn
Iapetus>Saturn
Tethys>Saturn
Dione>Saturn
Titania>Uranus
Oberon>Uranus
Miranda>Uranus
Ariel>Uranus
Umbriel>Uranus
Triton>Neptune
Nereid>Neptune
Proteus>Neptune
""")
_rel("science", "unit", [
    "Is the {o} the SI unit of {s}?", "Is the SI unit of {s} the {o}?", "Is it correct that the SI unit of {s} is the {o}?",
    "Does the SI system measure {s} with the {o}?", "Is {s} measured in the SI system using the {o}?"], """
force>newton
energy>joule
power>watt
pressure>pascal
frequency>hertz
electric current>ampere
electrical resistance>ohm
electric potential difference>volt
capacitance>farad
magnetic flux density>tesla
inductance>henry
temperature>kelvin
mass>kilogram
length>meter
time>second
amount of substance>mole
luminous intensity>candela
electric charge>coulomb
magnetic flux>weber
electrical conductance>siemens
radioactivity>becquerel
absorbed radiation dose>gray
luminous flux>lumen
illuminance>lux
""")
_rel("science", "animal_class", [
    "Is {sa} {oa}?", "Would {sa} count as {oa}?", "Is it correct that {sa} is {oa}?",
    "Is {sa} classified as {oa}?", "Can {sa} be called {oa}?"], """
whale>mammal
dolphin>mammal
bat>mammal
elephant>mammal
kangaroo>mammal
horse>mammal
tiger>mammal
seal>mammal
otter>mammal
platypus>mammal
eagle>bird
owl>bird
ostrich>bird
penguin>bird
sparrow>bird
parrot>bird
hawk>bird
pigeon>bird
duck>bird
snake>reptile
lizard>reptile
crocodile>reptile
gecko>reptile
iguana>reptile
alligator>reptile
turtle>reptile
frog>amphibian
toad>amphibian
newt>amphibian
salamander>amphibian
shark>fish
salmon>fish
trout>fish
tuna>fish
goldfish>fish
eel>fish
seahorse>fish
ant>insect
bee>insect
wasp>insect
butterfly>insect
beetle>insect
mosquito>insect
dragonfly>insect
grasshopper>insect
ladybug>insect
spider>arachnid
scorpion>arachnid
tick>arachnid
crab>crustacean
lobster>crustacean
shrimp>crustacean
""")
_rel("science", "organ", [
    "Is the {s} part of the {o} system?", "Does the {o} system include the {s}?",
    "Is it correct that the {s} belongs to the {o} system?", "Would the {s} be counted as part of the {o} system?",
    "Is the {s} a structure of the {o} system?"], """
heart>circulatory
arteries>circulatory
veins>circulatory
lungs>respiratory
trachea>respiratory
brain>nervous
spinal cord>nervous
stomach>digestive
small intestine>digestive
esophagus>digestive
femur>skeletal
skull>skeletal
tibia>skeletal
humerus>skeletal
biceps>muscular
triceps>muscular
kidneys>urinary
bladder>urinary
thyroid gland>endocrine
pituitary gland>endocrine
""")

_rel("other", "book", [
    "Did {o} write {s}?", "Is {o} the author of {s}?", "Is {s} a work by {o}?",
    "Is it correct that {s} was written by {o}?", "Was {s} written by {o}?"], """
Pride and Prejudice>Jane Austen
Emma>Jane Austen
Sense and Sensibility>Jane Austen
1984>George Orwell
Animal Farm>George Orwell
Moby-Dick>Herman Melville
Great Expectations>Charles Dickens
Oliver Twist>Charles Dickens
A Tale of Two Cities>Charles Dickens
Hamlet>William Shakespeare
Macbeth>William Shakespeare
Othello>William Shakespeare
King Lear>William Shakespeare
Romeo and Juliet>William Shakespeare
The Odyssey>Homer
The Iliad>Homer
Don Quixote>Miguel de Cervantes
War and Peace>Leo Tolstoy
Anna Karenina>Leo Tolstoy
Crime and Punishment>Fyodor Dostoevsky
The Brothers Karamazov>Fyodor Dostoevsky
Les Misérables>Victor Hugo
The Hunchback of Notre-Dame>Victor Hugo
The Great Gatsby>F. Scott Fitzgerald
To Kill a Mockingbird>Harper Lee
The Catcher in the Rye>J. D. Salinger
Of Mice and Men>John Steinbeck
The Grapes of Wrath>John Steinbeck
Frankenstein>Mary Shelley
Dracula>Bram Stoker
Jane Eyre>Charlotte Brontë
Wuthering Heights>Emily Brontë
The Hobbit>J. R. R. Tolkien
The Lord of the Rings>J. R. R. Tolkien
The Lion, the Witch and the Wardrobe>C. S. Lewis
Alice's Adventures in Wonderland>Lewis Carroll
Treasure Island>Robert Louis Stevenson
The Old Man and the Sea>Ernest Hemingway
A Farewell to Arms>Ernest Hemingway
Brave New World>Aldous Huxley
Fahrenheit 451>Ray Bradbury
The Metamorphosis>Franz Kafka
The Trial>Franz Kafka
The Stranger>Albert Camus
The Plague>Albert Camus
Ulysses>James Joyce
Dubliners>James Joyce
Heart of Darkness>Joseph Conrad
Lord of the Flies>William Golding
The Divine Comedy>Dante Alighieri
Paradise Lost>John Milton
Candide>Voltaire
Faust>Johann Wolfgang von Goethe
Gulliver's Travels>Jonathan Swift
Robinson Crusoe>Daniel Defoe
Little Women>Louisa May Alcott
Adventures of Huckleberry Finn>Mark Twain
The Adventures of Tom Sawyer>Mark Twain
The Call of the Wild>Jack London
The Scarlet Letter>Nathaniel Hawthorne
Middlemarch>George Eliot
The Picture of Dorian Gray>Oscar Wilde
The Jungle Book>Rudyard Kipling
Harry Potter and the Philosopher's Stone>J. K. Rowling
The Little Prince>Antoine de Saint-Exupéry
One Hundred Years of Solitude>Gabriel García Márquez
The Name of the Rose>Umberto Eco
Things Fall Apart>Chinua Achebe
Beloved>Toni Morrison
Slaughterhouse-Five>Kurt Vonnegut
Catch-22>Joseph Heller
Dune>Frank Herbert
The Handmaid's Tale>Margaret Atwood
The Road>Cormac McCarthy
Matilda>Roald Dahl
Charlie and the Chocolate Factory>Roald Dahl
The Hound of the Baskervilles>Arthur Conan Doyle
Murder on the Orient Express>Agatha Christie
The Time Machine>H. G. Wells
The War of the Worlds>H. G. Wells
Twenty Thousand Leagues Under the Sea>Jules Verne
Around the World in Eighty Days>Jules Verne
The Three Musketeers>Alexandre Dumas
The Count of Monte Cristo>Alexandre Dumas
Madame Bovary>Gustave Flaubert
""")
_rel("other", "painting", [
    "Did {o} paint {s}?", "Is {s} a painting by {o}?", "Is {o} the artist who made {s}?",
    "Is it correct that {s} was painted by {o}?", "Was {s} painted by {o}?"], """
the Mona Lisa>Leonardo da Vinci
The Last Supper>Leonardo da Vinci
The Starry Night>Vincent van Gogh
Sunflowers>Vincent van Gogh
Café Terrace at Night>Vincent van Gogh
Guernica>Pablo Picasso
Les Demoiselles d'Avignon>Pablo Picasso
The Persistence of Memory>Salvador Dalí
The Scream>Edvard Munch
The Kiss>Gustav Klimt
Girl with a Pearl Earring>Johannes Vermeer
The Night Watch>Rembrandt
The Birth of Venus>Sandro Botticelli
Primavera>Sandro Botticelli
The Creation of Adam>Michelangelo
The School of Athens>Raphael
Water Lilies>Claude Monet
Impression, Sunrise>Claude Monet
Las Meninas>Diego Velázquez
The Garden of Earthly Delights>Hieronymus Bosch
American Gothic>Grant Wood
Nighthawks>Edward Hopper
Liberty Leading the People>Eugène Delacroix
The Third of May 1808>Francisco Goya
Composition VIII>Wassily Kandinsky
The Hay Wain>John Constable
Luncheon of the Boating Party>Pierre-Auguste Renoir
A Sunday Afternoon on the Island of La Grande Jatte>Georges Seurat
Campbell's Soup Cans>Andy Warhol
The Arnolfini Portrait>Jan van Eyck
The Raft of the Medusa>Théodore Géricault
The Card Players>Paul Cézanne
Olympia>Édouard Manet
The Death of Marat>Jacques-Louis David
Wanderer above the Sea of Fog>Caspar David Friedrich
The Sleeping Gypsy>Henri Rousseau
The Son of Man>René Magritte
Christina's World>Andrew Wyeth
The Swing>Jean-Honoré Fragonard
The Gleaners>Jean-François Millet
""")
_rel("other", "composition", [
    "Did {o} compose {s}?", "Is {o} the composer of {s}?", "Is {s} a work by {o}?",
    "Is it correct that {s} was composed by {o}?", "Was {s} composed by {o}?"], """
The Four Seasons>Antonio Vivaldi
Eine kleine Nachtmusik>Wolfgang Amadeus Mozart
The Magic Flute>Wolfgang Amadeus Mozart
Don Giovanni>Wolfgang Amadeus Mozart
The Marriage of Figaro>Wolfgang Amadeus Mozart
the Moonlight Sonata>Ludwig van Beethoven
Fidelio>Ludwig van Beethoven
Für Elise>Ludwig van Beethoven
the Brandenburg Concertos>Johann Sebastian Bach
The Well-Tempered Clavier>Johann Sebastian Bach
the Goldberg Variations>Johann Sebastian Bach
the St Matthew Passion>Johann Sebastian Bach
Messiah>George Frideric Handel
Water Music>George Frideric Handel
Swan Lake>Pyotr Ilyich Tchaikovsky
The Nutcracker>Pyotr Ilyich Tchaikovsky
the 1812 Overture>Pyotr Ilyich Tchaikovsky
Eugene Onegin>Pyotr Ilyich Tchaikovsky
The Rite of Spring>Igor Stravinsky
The Firebird>Igor Stravinsky
Petrushka>Igor Stravinsky
Boléro>Maurice Ravel
Clair de lune>Claude Debussy
La Mer>Claude Debussy
Carmen>Georges Bizet
La Traviata>Giuseppe Verdi
Aida>Giuseppe Verdi
Rigoletto>Giuseppe Verdi
Otello>Giuseppe Verdi
La Bohème>Giacomo Puccini
Tosca>Giacomo Puccini
Madama Butterfly>Giacomo Puccini
Turandot>Giacomo Puccini
The Barber of Seville>Gioachino Rossini
the William Tell Overture>Gioachino Rossini
Rhapsody in Blue>George Gershwin
Porgy and Bess>George Gershwin
The Planets>Gustav Holst
Peer Gynt>Edvard Grieg
Finlandia>Jean Sibelius
The Blue Danube>Johann Strauss II
Also sprach Zarathustra>Richard Strauss
Pictures at an Exhibition>Modest Mussorgsky
Scheherazade>Nikolai Rimsky-Korsakov
The Carnival of the Animals>Camille Saint-Saëns
the Symphonie fantastique>Hector Berlioz
the Hungarian Rhapsodies>Franz Liszt
Peter and the Wolf>Sergei Prokofiev
the Gymnopédies>Erik Satie
Lohengrin>Richard Wagner
Tristan und Isolde>Richard Wagner
The Flying Dutchman>Richard Wagner
Appalachian Spring>Aaron Copland
Carmina Burana>Carl Orff
Canon in D>Johann Pachelbel
The Creation>Joseph Haydn
""")
_rel("other", "language", [
    "Is {o} the main language spoken in {s}?", "Do most people in {s} speak {o}?", "Is the main language of {s} {o}?",
    "Is it correct that {o} is the main language of {s}?", "Would {o} be the most widely spoken language in {s}?"], """
Brazil>Portuguese
Portugal>Portuguese
Spain>Spanish
Argentina>Spanish
Chile>Spanish
Colombia>Spanish
Cuba>Spanish
Mexico>Spanish
Venezuela>Spanish
France>French
Germany>German
Austria>German
Italy>Italian
Netherlands>Dutch
Sweden>Swedish
Norway>Norwegian
Denmark>Danish
Poland>Polish
Russia>Russian
Greece>Greek
Turkey>Turkish
Egypt>Arabic
Saudi Arabia>Arabic
Iran>Persian
Japan>Japanese
South Korea>Korean
China>Mandarin Chinese
Vietnam>Vietnamese
Thailand>Thai
Hungary>Hungarian
Czech Republic>Czech
Romania>Romanian
Indonesia>Indonesian
Israel>Hebrew
""")

FACT_TOPICS = tuple(spec.TASKS["fact_topic"].options)
_CONT = "EAFNSO"


@lru_cache(maxsize=None)
def fact_bank() -> dict:
    """{"relation:subject": {s, topic, relation, true, cands (>= 3 false objects), phr}}; checked on load."""
    bank: dict[str, dict] = {}
    for topic, name, phr, rows in _BANK:
        objs = sorted({r[1] for r in rows})
        cont_of = {r[1]: r[2] for r in rows if len(r) > 2}
        for row in rows:
            s, o = row[0], row[1]
            assert f"{name}:{s}" not in bank, ("duplicate subject", s)
            if len(row) > 2:     # leaders: false countries only from other continents (a leader may have ruled nearby lands)
                cands = [x for x in objs if cont_of[x] != row[2]]
            else:
                cands = [x for x in objs if x != o]
            assert len(cands) >= 3 and o not in cands
            bank[f"{name}:{s}"] = {"s": s, "topic": topic, "relation": name, "true": o, "cands": cands, "phr": phr}
    return bank


def _fact_pool(seed: int) -> list[dict]:
    """Every (subject, phrasing) once true and once false (a false object cycled from three drawn per subject)."""
    rng = _rng("fact", seed)
    items = []
    for key in sorted(fact_bank()):
        f = fact_bank()[key]
        s = f["s"]
        falses = rng.sample(f["cands"], 3)
        for p, phrase in enumerate(f["phr"]):
            for truth in (1, 0):
                o = f["true"] if truth else falses[p % 3]
                assert truth == (o == f["true"])
                text = phrase.format(s=s, o=o, sa=_art(s), oa=_art(o))
                items.append({"s": s, "o": o, "truth": truth, "text": text, "topic": f["topic"],
                              "relation": f["relation"], "phrasing": p})
    return items


# ================================================================ routing (tasks 1 and 2)

_TOK = None


def _scripts():
    path = str(ROOT / "scripts")
    if path not in sys.path:
        sys.path.insert(0, path)


def routing_tokenizer():
    global _TOK
    if _TOK is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return _TOK


@lru_cache(maxsize=None)
def _routing_rows() -> tuple:
    """(group, id, source, kind, judgement, work, text) for the three v5 files only; text cut to 512 tokens from the oldest end."""
    _scripts()
    import routing_four
    import routing_rivals
    tok = routing_tokenizer()
    rows = []
    for name in ROUTING_FILES:
        with open(ROUTING_DIR / f"{name}.jsonl", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                text = routing_four.cut_oldest(tok, routing_rivals.full_text(r), spec.ROUTING_MAX_TOKENS)
                rows.append((r["group"], r["id"], r["source"], r["kind"], r["judgement"], r["work"], text))
    return tuple(rows)


def _route_items(task: str) -> list[dict]:
    col = 4 if task == "route_judgement" else 5
    words = {"route_judgement": {"yes": 0, "no": 1}, "route_work": {"light": 0, "heavy": 1}}[task]
    items = []
    for row in _routing_rows():
        if row[col] not in words:      # drops "unclear"
            continue
        rec = _record(task, row[1], row[6], words[row[col]], row[0], None, {"source": row[2], "kind": row[3]})
        items.append({"template_id": row[0], "cell": None, "records": {task: rec}})
    assert len({it["records"][task]["id"] for it in items}) == len(items)
    return items


# ================================================================ families -> items

def _hard_items(n: int, seed: int) -> list[dict]:
    items = []
    per = _quotas(n, {t: 1 for t in ("H1", "H2", "H3")})
    for k, tier in enumerate(("H1", "H2", "H3")):
        recs = generate_hard(per[tier], seed * 10 + k, HTIERS[tier])
        for i, r in enumerate(recs):
            bid = f"hard-{seed}-{tier}-{i}"
            meta = {"label": r["label"], "tier": tier, "hops_used": r["hops_used"], "decided_by": r["decided_by"]}
            items.append({"template_id": r["template_id"], "cell": (tier, r["label"]), "aux": {"state": r["state"]}, "records": {
                "hard_hops": _record("hard_hops", bid, r["prompt"], k, r["template_id"], tier, meta),
                "hard": _record("hard", bid, r["prompt"], HARD_LABELS.index(r["label"]), r["template_id"], tier, meta)}})
    return items


def _rule_items(task: str, n: int, seed: int) -> list[dict]:
    variant = {"rule": "rule", "sensible": "sensible"}[task]
    labels = list(spec.TASKS[task].options)
    items = []
    for r in generate_rule(n, seed, variant=variant):
        rec = _record(task, f"{task}-{seed}-{r['id']}", r["prompt"], labels.index(r["label"]), r["template_id"], None,
                      {"label": r["label"], "wrong_label": r["wrong_label"]})
        items.append({"template_id": r["template_id"], "cell": r["label"], "aux": {"state": r["state"]}, "records": {task: rec}})
    return items


def _cells_for(n, weights, rng):
    q = _quotas(n, weights)
    seq = [c for c in sorted(q, key=repr) for _ in range(q[c])]
    rng.shuffle(seq)
    return seq


def _code_items(n: int, seed: int) -> list[dict]:
    rng = _rng("code", seed)
    cells = _cells_for(n, {(s, d): 1 for s in SHAPES for d in range(10)}, rng)
    items = []
    for i, (shape, digit) in enumerate(cells):
        code = _code_item(rng, shape, digit)
        assert code is not None, (shape, digit)
        tpl = skeleton(code)
        bid = f"code-{seed}-{i}"
        meta = {"shape": shape, "digit": digit}
        items.append({"template_id": tpl, "cell": (shape, digit), "records": {
            "code_shape": _record("code_shape", bid, code, SHAPES.index(shape), tpl, None, meta),
            "code_output": _record("code_output", bid, code, digit, tpl, None, meta)}})
    return items


def _syntax_items(n: int, seed: int) -> list[dict]:
    rng = _rng("syntax", seed)
    cells = _cells_for(n, {(d, v): 1 for d in (1, 2, 3) for v in (0, 1)}, rng)
    items = []
    for i, (depth, bad) in enumerate(cells):
        s = _syntax_item(rng, depth, bad)
        assert s is not None, (depth, bad)
        tpl = syntax_skeleton(s)
        is_json = bool(re.search(r"[\"':]", s))
        ok = _json_valid(s) if is_json else _balanced(s)
        assert ok == (not bad) and bracket_depth(s) == depth, s
        bid = f"syntax-{seed}-{i}"
        meta = {"depth": depth, "well_formed": not bad}
        items.append({"template_id": tpl, "cell": (depth, bad), "records": {
            "syntax_depth": _record("syntax_depth", bid, s, depth - 1, tpl, None, meta),
            "syntax_valid": _record("syntax_valid", bid, s, bad, tpl, f"depth{depth}", meta)}})
    return items


def _arith_weight(cell):
    return 3 if cell[0] == "mixed" else 1


def _arith_items(n: int, seed: int) -> list[dict]:
    rng = _rng("arith", seed)
    cells = _cells_for(n, {(c, d): _arith_weight((c, d)) for c in ("plus", "minus", "times", "mixed") for d in range(10)}, rng)
    items = []
    for i, (opclass, digit) in enumerate(cells):
        got = _arith_item(rng, opclass, digit)
        assert got is not None, (opclass, digit)
        text, tpl, meta = got
        bid = f"arith-{seed}-{i}"
        meta = dict(meta, value=digit)
        items.append({"template_id": tpl, "cell": (opclass, digit), "records": {
            "arith_shape": _record("arith_shape", bid, text, ("plus", "minus", "times", "mixed").index(opclass), tpl, None, meta),
            "arith": _record("arith", bid, text, digit, tpl, "ops2" if opclass == "mixed" else "ops1", meta)}})
    return items


def _fact_items(n: int | None, seed: int) -> list[dict]:
    pool = _fact_pool(seed)
    items = []
    for i, f in enumerate(pool):
        bid = f"fact-{seed}-{i}"
        meta = {k: f[k] for k in ("s", "o", "truth", "relation", "phrasing")}
        topic = FACT_TOPICS.index(f["topic"])
        items.append({"template_id": f["s"], "cell": (f["topic"], f["truth"]), "records": {
            "fact_topic": _record("fact_topic", bid, f["text"], topic, f["s"], None, meta),
            "fact_check": _record("fact_check", bid, f["text"], 0 if f["truth"] else 1, f["s"], None, meta)}})
    return items


# ================================================================ Phase 2 stage A (docs/ATLAS-phase2-stageA.md)

_LEADS = ("", "{w}:", "Here it is:", "Look at this:")      # label-free lead-in line; {w} is Expression / String
_ID_FORMS = ("x+0", "0+x", "x-0", "x*1", "1*x")
_LONG_NAMES = [n for n in _NAMES if len(n) >= 2]            # 10 names of 2+ letters (tasks 17)
_CLOSER = {"r": ")", "s": "]", "c": "}"}
_OPEN = {"r": "(", "s": "[", "c": "{"}


def _lead_text(rng, word: str) -> tuple[int, str]:
    k = rng.randrange(len(_LEADS))
    return k, _LEADS[k].format(w=word)


def _with_lead(lead: str, body: str) -> str:
    return f"{lead}\n{body}" if lead else body


def _rank_pattern(t) -> tuple:
    v = sorted(set(t))
    return tuple(v.index(x) for x in t)


def _order_items(n: int, seed: int) -> list[dict]:
    """Task 15: is the first number larger than the last? Pair {first, last} and middle drawn, order by a fair coin."""
    rng = _rng("arith_order", seed)
    cells = _cells_for(n, {(k, y): 1 for k in (1, 2) for y in (0, 1)}, rng)
    items = []
    for i, (nops, label) in enumerate(cells):
        hi, lo = sorted(rng.sample(range(10), 2), reverse=True)
        first, last = (hi, lo) if label == 0 else (lo, hi)      # label 0 = yes (first larger); cells come from a shuffled balanced list, which is the fair coin
        ops = [rng.choice("+-*") for _ in range(nops)]
        mid = rng.randint(0, 9)
        toks = [str(first)] + ([ops[0], str(mid), ops[1]] if nops == 2 else [ops[0]]) + [str(last)]
        pat = ops[0] if nops == 1 else ops[0] + ops[1]
        mirror = (_rank_pattern((first, mid, last)), _rank_pattern((last, mid, first))) if nops == 2 else ((0, 1), (0, 1))
        k, lead = _lead_text(rng, "Expression")
        tpl = f"{k}|{pat}|{''.join(map(str, min(mirror)))}"
        meta = {"first": first, "last": last, "ops": pat, "mid": mid if nops == 2 else None}
        items.append({"template_id": tpl, "cell": (nops, label), "records": {"arith_order": _record(
            "arith_order", f"arith_order-{seed}-{i}", _with_lead(lead, " ".join(toks)), label, tpl, None, meta)}})
    return items


@lru_cache(maxsize=None)
def _bracket_shapes() -> tuple:
    """Every ordered forest of six pairs with depth 1-3, as a string of 'o' and 'c', in a fixed order."""
    @lru_cache(maxsize=None)
    def forests(n, d):
        if n == 0:
            return ("",)
        if d == 0:
            return ()
        return tuple("o" + inner + "c" + rest for k in range(1, n + 1)
                     for inner in forests(k - 1, d - 1) for rest in forests(n - k, d))
    return tuple(sorted(forests(6, 3)))


def _close_items(n: int, seed: int) -> list[dict]:
    """Task 16: which kind closes first. A shape, then kinds to its pairs by a random permutation of rrsscc."""
    rng = _rng("syntax_first_close", seed)
    shapes = _bracket_shapes()
    cells = _cells_for(n, {y: 1 for y in range(3)}, rng)
    items = []
    for i, label in enumerate(cells):
        shape = rng.choice(shapes)
        while True:
            kinds = list("rrsscc")
            rng.shuffle(kinds)
            stack, pair_of, nxt, first_closed = [], [], 0, None
            for ch in shape:
                if ch == "o":
                    stack.append(nxt)
                    nxt += 1
                elif first_closed is None:
                    first_closed = stack.pop()
                else:
                    stack.pop()
            if "rsc".index(kinds[first_closed]) == label:
                break
        stack, out, k_i = [], [], 0
        for ch in shape:
            if ch == "o":
                stack.append(kinds[k_i])
                k_i += 1
                out.append(_OPEN[stack[-1]])
            else:
                out.append(_CLOSER[stack.pop()])
        k, lead = _lead_text(rng, "String")
        tpl = f"{k}|{shape}"
        text = " ".join(out)
        assert bracket_depth(text.replace(" ", "")) <= 3 and _balanced(text.replace(" ", ""))
        meta = {"first_closed": "rsc"[label], "depth": bracket_depth(text.replace(" ", ""))}
        items.append({"template_id": tpl, "cell": label, "records": {"syntax_first_close": _record(
            "syntax_first_close", f"syntax_first_close-{seed}-{i}", _with_lead(lead, text), label, tpl, None, meta)}})
    return items


def _uses_first_items(n: int, seed: int) -> list[dict]:
    """Task 17: does the print line use the name assigned on line 1. Template: the unordered name set and the print form."""
    rng = _rng("code_uses_first", seed)
    cells = _cells_for(n, {y: 1 for y in (0, 1)}, rng)
    items = []
    for i, label in enumerate(cells):
        names = rng.sample(_LONG_NAMES, 3)               # uniformly random line order
        printed = names[0] if label == 0 else rng.choice(names[1:])
        form = rng.choice(["", "+", "-", "*"])
        arg = printed if not form else f"{printed} {form} {rng.randint(0, 9)}"
        code = "\n".join([f"{nm} = {rng.randint(0, 9)}" for nm in names] + [f"print({arg})"])
        tpl = "+".join(sorted(names)) + f"|{form or 'bare'}"
        meta = {"printed_line": names.index(printed) + 1, "form": form or "bare"}
        items.append({"template_id": tpl, "cell": label, "records": {"code_uses_first": _record(
            "code_uses_first", f"code_uses_first-{seed}-{i}", code, label, tpl, None, meta)}})
    return items


_READ_RE = re.compile(r"^- (\S+): read=(yes|no) edited=", re.M)


_STATUS_READ = re.compile(r"^target_read=(yes|no)\n", re.M)


def hard_read_info(prompt: str) -> tuple[int, int]:
    """(target read flag as 0 = yes / 1 = no, number of files with read=yes), parsed from a `hard` prompt."""
    m = re.search(r"^Target file: (\S+)$", prompt, re.M)
    assert m, prompt
    files = _READ_RE.findall(prompt.split("\nTests:")[0])
    flags = [f for f in files if f[0] == m.group(1)]
    assert len(flags) == 1, prompt
    return (0 if flags[0][1] == "yes" else 1), sum(f[1] == "yes" for f in files)


def _read_items(task: str, n: int, seed: int, strip) -> list[dict]:
    """H1 hard prompts, label = the target's read flag, `strip` applied to the prompt. Balanced yes/no within each count
    of read=yes files (1-5: at 0 and 6 the count alone gives the label, so those are dropped)."""
    cells = {(c, y): 1 for c in range(1, 6) for y in (0, 1)}
    quota = _quotas(n, cells)
    got = {c: [] for c in cells}
    batch = 0
    while any(len(got[c]) < quota[c] for c in cells):
        assert batch < 400, f"{task}: cell quotas not reachable"
        for r in generate_hard(500, 7000 + seed * 1000 + batch, HTIERS["H1"]):
            y, cnt = hard_read_info(r["prompt"])
            r = dict(r, prompt=strip(r["prompt"]))
            if 1 <= cnt <= 5 and len(got[(cnt, y)]) < quota[(cnt, y)]:
                got[(cnt, y)].append(r)
        batch += 1
    rng = _rng("target_read", seed)
    rows = [(c, r) for c in sorted(cells) for r in got[c]]
    rng.shuffle(rows)
    items = []
    for i, ((cnt, y), r) in enumerate(rows):
        tpl = r["template_id"]
        meta = {"read_count": cnt, "target_read": "yes" if y == 0 else "no"}
        items.append({"template_id": tpl, "cell": (cnt, y), "records": {task: _record(
            task, f"{task}-{seed}-{i}", r["prompt"], y, tpl, None, meta)}})
    return items


def _target_read_items(n: int, seed: int) -> list[dict]:
    """Task 18: the Status value line `target_read=` is removed (it would hand over the answer)."""
    return _read_items("hard_target_read", n, seed, lambda p: _STATUS_READ.sub("", p))


def _id_items(n: int, seed: int) -> list[dict]:
    """Task 19: one operation that keeps an operand (x+0, 0+x, x-0, x*1, 1*x); answer x."""
    rng = _rng("arith_id", seed)
    cells = _cells_for(n, {d: 1 for d in range(10)}, rng)
    items = []
    for i, x in enumerate(cells):
        form = rng.choice(_ID_FORMS)
        a, op, b = {"x+0": (x, "+", 0), "0+x": (0, "+", x), "x-0": (x, "-", 0), "x*1": (x, "*", 1), "1*x": (1, "*", x)}[form]
        prefix, tight, wrap = rng.randrange(len(_PREFIX)), rng.random() < 0.5, rng.random() < 0.3
        text, body = _arith_text(a, b, 0, op, None, prefix, tight, False, wrap)
        assert eval(body) == x
        tpl = f"{form}|{prefix}{'t' if tight else 's'}{'w' if wrap else ''}"
        items.append({"template_id": tpl, "cell": x, "records": {"arith_id": _record(
            "arith_id", f"arith_id-{seed}-{i}", text, x, tpl, None, {"form": form, "x": x})}})
    return items


def _literal_items(n: int, seed: int) -> list[dict]:
    """Task 20: two or three constant assignments, then print of one of the names. Template: the unordered name set."""
    rng = _rng("code_literal", seed)
    cells = _cells_for(n, {d: 1 for d in range(10)}, rng)
    items = []
    for i, digit in enumerate(cells):
        k = rng.choice((2, 3))
        names = rng.sample(_NAMES, k)
        pick = rng.randrange(k)
        vals = [rng.randint(0, 9) for _ in range(k)]
        vals[pick] = digit
        code = "\n".join([f"{nm} = {v}" for nm, v in zip(names, vals)] + [f"print({names[pick]})"])
        assert run_py(code) == str(digit)
        tpl = "+".join(sorted(names))
        items.append({"template_id": tpl, "cell": digit, "records": {"code_literal": _record(
            "code_literal", f"code_literal-{seed}-{i}", code, digit, tpl, None, {"n_assign": k})}})
    return items


def _long_items(n: int, seed: int) -> list[dict]:
    """Task 21: `_arith_text` style expressions with values 10-999; answer = index of the first digit of the value."""
    rng = _rng("arith_long", seed)
    items = []
    while len(items) < n:
        a, b, c = rng.randint(0, 99), rng.randint(0, 99), rng.randint(0, 9)
        if rng.random() < 0.5:
            op1, op2 = rng.choice("+-*"), None
            val = _APPLY[op1](a, b)
        else:
            op1, op2 = rng.sample(["+", "-", "*"], 2)
            v1 = _APPLY[op1](a, b)
            if v1 < 0:
                continue
            val = _APPLY[op2](v1, c)
        if not 10 <= val <= 999:
            continue
        prefix, tight, full, wrap = rng.randrange(len(_PREFIX)), rng.random() < 0.5, rng.random() < 0.5, rng.random() < 0.3
        text, body = _arith_text(a, b, c, op1, op2, prefix, tight, full, wrap)
        assert eval(body) == val
        i = len(items)
        pattern = op1 + (op2 or "")
        vals = sorted({a, b, c})
        bucket = "".join(str(vals.index(x)) for x in ((a, b, c) if op2 else (a, b)))
        tpl = f"{pattern}|{bucket}|{prefix}{'t' if tight else 's'}{'f' if (full and op2) else 'm'}{'w' if wrap else ''}"
        meta = {"value": val, "a": a, "b": b, "c": c if op2 else None, "ops": pattern}
        items.append({"template_id": tpl, "cell": None, "records": {"arith_long": _record(
            "arith_long", f"arith_long-{seed}-{i}", text, int(str(val)[0]), tpl, None, meta)}})
    return items


_NEW_ITEMS = {"arith_order": _order_items, "syntax_first_close": _close_items, "code_uses_first": _uses_first_items,
              "hard_target_read": _target_read_items, "arith_id": _id_items, "code_literal": _literal_items,
              "arith_long": _long_items}


# ================================================================ Phase 2 stage C (docs/ATLAS-phase2-stageC.md)

_STATUS_BLOCK = re.compile(r"\n\nStatus[^\n]*(?:\n[^\n]+)+(?=\n\n)")
_ASSIGN_RE = re.compile(r"^(\w+) = (\d|\w+)$")


def strip_status(prompt: str) -> str:
    """Task 24: the whole Status block (its heading, every fact line and the blank line before it) removed."""
    out, k = _STATUS_BLOCK.subn("", prompt)
    assert k == 1, prompt
    return out


def _target_read_bare_items(n: int, seed: int) -> list[dict]:
    return _read_items("target_read_bare", n, seed, strip_status)


def trace_info(code: str) -> tuple[int, int]:
    """(copy hops, line number of the constant the printed name ends at) for a task 22 program."""
    lines = code.split("\n")
    line_of, src = {}, {}
    for i, ln in enumerate(lines[:-1], 1):
        m = _ASSIGN_RE.match(ln)
        assert m, ln
        line_of[m.group(1)] = i
        src[m.group(1)] = None if m.group(2).isdigit() else m.group(2)
    name = re.fullmatch(r"print\((\w+)\)", lines[-1]).group(1)
    hops = 0
    while src[name] is not None:
        name, hops = src[name], hops + 1
    return hops, line_of[name]


# The brief's templates (shape without names, plus a lead-in) give only 64 shapes for task 22 and 72 for task 25, under the
# 100 that five splits of 20 need. True adds the unordered name set (as stage A did for tasks 17 and 20); it is off until
# a decision is needed, so the strict build stops at the template check.
NAMES_IN_TEMPLATE = True  # decided 2026-11-11, before any build: as stage A tasks 17 and 20


def _name_part(names) -> str:
    return "|" + "+".join(sorted(names)) if NAMES_IN_TEMPLATE else ""


# (hops, assignment lines, constant lines): a "no" program needs a constant besides line 1, so both labels are held to
# the same line and constant counts (else those counts would give the label). Hop 3 would need a fifth assignment line
# for its "no" side, which the 4-5 line limit forbids: see the report.
_TRACE_CELLS = {(1, 3, 2): 1, (1, 4, 2): 1, (1, 4, 3): 1, (2, 4, 2): 2}


def _traces_items(n: int, seed: int) -> list[dict]:
    """Task 22: does the printed name trace back to line 1's constant. Template: line types, print line and lead-in."""
    rng = _rng("code_traces_first", seed)
    cells = _cells_for(n, {(h, m, c, y): w for (h, m, c), w in _TRACE_CELLS.items() for y in (0, 1)}, rng)
    items = []
    for i, (hops, m, nconst, label) in enumerate(cells):
        while True:
            names = rng.sample(_LONG_NAMES, m)
            kinds = ["c"] + [rng.choice("cp") for _ in range(m - 1)]
            if kinds.count("c") != nconst:
                continue
            srcs = [None] + [rng.randrange(j) if kinds[j] == "p" else None for j in range(1, m)]
            pr = rng.randrange(m)
            lines = [f"{names[j]} = {rng.randint(0, 9) if kinds[j] == 'c' else names[srcs[j]]}" for j in range(m)]
            code = "\n".join(lines + [f"print({names[pr]})"])
            h, root = trace_info(code)
            if (h, root == 1) == (hops, label == 0):
                break
        k, lead = _lead_text(rng, "Program")
        tpl = f"{k}|{''.join(kinds)}|{pr + 1}{_name_part(names)}"
        meta = {"hops": hops, "root_line": root, "n_assign": m}
        items.append({"template_id": tpl, "cell": (hops, m, nconst, label), "records": {"code_traces_first": _record(
            "code_traces_first", f"code_traces_first-{seed}-{i}", _with_lead(lead, code), label, tpl, None, meta)}})
    return items


@lru_cache(maxsize=None)
def _outer_shapes() -> tuple:
    """Every ordered forest of six pairs with depth 1-4, as a string of 'o' and 'c', in a fixed order."""
    @lru_cache(maxsize=None)
    def forests(n, d):
        if n == 0:
            return ("",)
        if d == 0:
            return ()
        return tuple("o" + inner + "c" + rest for k in range(1, n + 1)
                     for inner in forests(k - 1, d - 1) for rest in forests(n - k, d))
    return tuple(sorted(forests(6, 4)))


def _outer_items(n: int, seed: int) -> list[dict]:
    """Task 23: does the first bracket close at the very end. Cells: kind of the first bracket x label."""
    rng = _rng("syntax_outer_pair", seed)
    shapes = _outer_shapes()
    by_label = {0: [s for s in shapes if _wraps(s)],
                1: [s for s in shapes if not _wraps(s)]}
    cells = _cells_for(n, {(kd, y): 1 for kd in range(3) for y in (0, 1)}, rng)
    items = []
    for i, (kd, label) in enumerate(cells):
        shape = rng.choice(by_label[label])
        while True:
            kinds = list("rrsscc")
            rng.shuffle(kinds)
            if "rsc".index(kinds[0]) == kd:
                break
        stack, out, k_i = [], [], 0
        for ch in shape:
            if ch == "o":
                stack.append(kinds[k_i])
                k_i += 1
                out.append(_OPEN[stack[-1]])
            else:
                out.append(_CLOSER[stack.pop()])
        k, lead = _lead_text(rng, "String")
        tpl = f"{k}|{shape}"
        text = " ".join(out)
        flat = text.replace(" ", "")
        assert _balanced(flat) and 1 <= bracket_depth(flat) <= 4
        meta = {"first_kind": "rsc"[kd], "depth": bracket_depth(flat)}
        items.append({"template_id": tpl, "cell": (kd, label), "records": {"syntax_outer_pair": _record(
            "syntax_outer_pair", f"syntax_outer_pair-{seed}-{i}", _with_lead(lead, text), label, tpl, None, meta)}})
    return items


def _wraps(shape: str) -> bool:
    """True if the first bracket's partner is the last symbol (depth of the running count returns to 0 only at the end)."""
    d = 0
    for j, ch in enumerate(shape):
        d += 1 if ch == "o" else -1
        if d == 0:
            return j == len(shape) - 1
    return False


def _bind_items(n: int, seed: int) -> list[dict]:
    """Task 25: is the first variable in the `compare` line larger than the second. Template: size, the two positions, lead-in."""
    rng = _rng("bind_compare", seed)
    cells = _cells_for(n, {(k, y): 1 for k in (3, 4) for y in (0, 1)}, rng)
    items = []
    for i, (k_vars, label) in enumerate(cells):
        names = rng.sample(_LONG_NAMES, k_vars)
        p1, p2 = rng.sample(range(k_vars), 2)
        drawn = rng.sample(range(10), k_vars)
        a, b = sorted(drawn[:2], reverse=True)              # label 0 = yes (first larger)
        vals = iter(drawn[2:])
        vals = [(a if label == 0 else b) if j == p1 else (b if label == 0 else a) if j == p2 else next(vals)
                for j in range(k_vars)]
        k, lead = _lead_text(rng, "Values")
        code = "\n".join([f"{nm} = {v}" for nm, v in zip(names, vals)] + [f"compare {names[p1]} {names[p2]}"])
        tpl = f"{k}|{k_vars}|{p1 + 1}{p2 + 1}{_name_part(names)}"
        meta = {"first_value": vals[p1], "second_value": vals[p2], "n_assign": k_vars}
        items.append({"template_id": tpl, "cell": (k_vars, label), "records": {"bind_compare": _record(
            "bind_compare", f"bind_compare-{seed}-{i}", _with_lead(lead, code), label, tpl, None, meta)}})
    return items


_NEW_ITEMS.update({"code_traces_first": _traces_items, "syntax_outer_pair": _outer_items,
                   "target_read_bare": _target_read_bare_items, "bind_compare": _bind_items})


def cell_weight(family: str, cell) -> int:
    return _arith_weight(cell) if family == "arith" else 1


def family_items(family: str, n: int | None, seed: int) -> list[dict]:
    """n base items (None: the whole fixed pool for facts and routing)."""
    if family == "code":
        return _code_items(n, seed)
    if family == "syntax":
        return _syntax_items(n, seed)
    if family == "arith":
        return _arith_items(n, seed)
    if family == "hard":
        return _hard_items(n, seed)
    if family in ("rule", "sensible"):
        return _rule_items(family, n, seed)
    if family in _NEW_ITEMS:
        return _NEW_ITEMS[family](n, seed)
    if family == "fact":
        return _fact_items(n, seed)
    return _route_items(family)


def _take(items: list[dict], n: int, seed: int, family: str) -> list[dict]:
    """First-class subsample of a fixed pool: n items with balanced cells (no template cap)."""
    if n is None or n == len(items):
        return items
    assert n <= len(items), (family, n, len(items))
    got, _ = select(items, n, _rng("take", family, seed), lambda c: cell_weight(family, c), cap=None)
    return got


def _check_records(task: str, recs: list[dict]) -> None:
    k = len(spec.TASKS[task].options)
    assert len({r["id"] for r in recs}) == len(recs), f"{task}: ids not unique"
    assert all(0 <= r["answer"] < k and r["task"] == task for r in recs)


def balance_dev(answers: list[int], k: int) -> float:
    """Largest relative deviation of a class frequency from 1/k."""
    c = Counter(answers)
    return max(abs(c.get(i, 0) / len(answers) * k - 1) for i in range(k))


def generate_pair(appraise_task: str, n: int, seed: int) -> tuple[list[dict], list[dict]]:
    fam = FAMILY_OF[appraise_task]
    a, b = FAMILY_TASKS[fam]
    assert a == appraise_task and len(FAMILY_TASKS[fam]) == 2, appraise_task
    items = _take(family_items(fam, n if fam != "fact" else None, seed), n, seed, fam)
    ra, rb = [it["records"][a] for it in items], [it["records"][b] for it in items]
    for t, recs in ((a, ra), (b, rb)):
        _check_records(t, recs)
        if n >= 200 and t not in NATURAL:
            assert balance_dev([r["answer"] for r in recs], len(spec.TASKS[t].options)) <= 0.10 + len(spec.TASKS[t].options) / n, t
    return ra, rb


def generate(task: str, n: int, seed: int) -> list[dict]:
    fam = FAMILY_OF[task]
    if fam in CHECK_BASE:
        return _generate_check(task, n, seed)
    if fam in BIN_BASE:
        return _generate_bin(task, n, seed)
    if len(FAMILY_TASKS[fam]) == 2:
        a, b = FAMILY_TASKS[fam]
        ra, rb = generate_pair(a, n, seed)
        return ra if task == a else rb
    items = family_items(fam, n if fam in SIZED else None, seed)
    items = _take(items, n, seed, fam)
    recs = [it["records"][task] for it in items]
    _check_records(task, recs)
    if n >= 200 and task not in NATURAL:
        assert balance_dev([r["answer"] for r in recs], len(spec.TASKS[task].options)) <= 0.10 + len(spec.TASKS[task].options) / n, task
    return recs


# ================================================================ Robustness E2 (docs/ATLAS-robustness.md)
# A check task is its do-task's item (same generator, seed, pool, template deal and split selection, so record i of the
# check task shares its problem and id with record i of the do task) plus one proposed answer. Half the records show the
# correct answer, half a near miss. The proposals are dealt per split after the items are chosen, so each split is 50/50
# and the proposed answers of the correct and wrong halves have the same distribution as far as the near misses allow.

CHECK_PROPOSAL = "\n\nProposed answer: "


def _trace_values(code: str) -> list[int]:
    """Every int that a global variable holds at any line event of the program (the value 'from the wrong line')."""
    vals: list[int] = []

    def local(frame, event, arg):
        if event in ("line", "return"):
            vals.extend(v for k, v in frame.f_globals.items() if k != "__builtins__" and type(v) is int)
        return local

    def glob(frame, event, arg):
        return local if frame.f_code.co_filename == "<prog>" else None

    old = sys.gettrace()
    sys.settrace(glob)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "<prog>", "exec"), {})
    finally:
        sys.settrace(old)
    return vals


_LOOP_BOUND = re.compile(r"(range\((?:\d+, )?|while \w+ < )(\d+)(\)?:)")


def _loop_step_outputs(code: str) -> list[int]:
    """What the program prints with its one loop run a step shorter or longer."""
    out = []
    for d in (-1, 1):
        mod = _LOOP_BOUND.sub(lambda m: f"{m.group(1)}{int(m.group(2)) + d}{m.group(3)}", code, count=1)
        if mod == code:
            continue
        try:
            o = run_py(mod)
        except Exception:
            continue
        if re.fullmatch(r"\d", o):
            out.append(int(o))
    return out


def _code_output_near(code: str, answer: int, shape: str) -> list[int]:
    cands = (_loop_step_outputs(code) if shape == "loop" else []) + [v for v in _trace_values(code) if 0 <= v <= 9]
    return sorted({c for c in cands if c != answer})


def _literal_near(code: str, answer: int) -> list[int]:
    """The other constants of the program: a variable's value from the wrong line."""
    vals = [int(m.group(1)) for m in re.finditer(r"^\w+ = (\d)$", code, re.M)]
    return sorted({v for v in vals if v != answer})


def _arith_near(meta: dict, answer: int) -> list[int]:
    """R1: within 2 of the answer and of the same parity, so only answer +- 2 (kept if it is a legal digit). The off-by-one-operation
    errors of the first design (carried through the second operation) are kept as candidates when they land there."""
    a, b, c, ops = meta["a"], meta["b"], meta["c"], meta["ops"]
    out = set()
    for d in [k * s for k in range(1, 11) for s in (-1, 1)]:
        out.add(answer + d)                                        # the last operation's result
        if len(ops) == 2:
            out.add(_APPLY[ops[1]](_APPLY[ops[0]](a, b) + d, c))   # the first operation's result, carried through
    return sorted(v for v in out if 0 <= v <= 9 and v != answer and abs(v - answer) == 2)


def _assign_events(code: str) -> list[tuple[int, int]]:
    """(line, value) for every change of a global int variable, in execution order; the line is the one that assigned it."""
    events: list[tuple[int, int]] = []
    seen: dict = {}
    prev: list = [None]

    def local(frame, event, arg):
        if event in ("line", "return"):
            now = {k: v for k, v in frame.f_globals.items() if k != "__builtins__" and type(v) is int}
            if prev[0] is not None:
                events.extend((prev[0], v) for k, v in now.items() if k not in seen or seen[k] != v)
            seen.clear()
            seen.update(now)
            prev[0] = frame.f_lineno
        return local

    def glob(frame, event, arg):
        return local if frame.f_code.co_filename == "<prog>" else None

    old = sys.gettrace()
    sys.settrace(glob)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "<prog>", "exec"), {})
    finally:
        sys.settrace(old)
    return events


def _near_misses(task: str, item: dict) -> list[int]:
    """Indices (into the do-task's options) of the plausible wrong answers of one item; empty if there is none (R1)."""
    do = CHECK_DO[task]
    rec = item["records"][do]
    opts = list(spec.TASKS[do].options)
    if task == "rule_check":
        # the action of another ACTIVE rule (never a disabled rule's action unless an active rule has it too)
        acts = {r["action"] for r in item["aux"]["state"]["rules"]}
        return sorted(opts.index(a) for a in acts if a != rec["meta"]["label"])
    if task == "hard_check":
        rules = item["aux"]["state"]["rules"]
        i = rec["meta"]["decided_by"] - 1
        near = [rules[j]["action"] for j in (i - 1, i + 1) if 0 <= j < len(rules)]   # one hop short, one hop past
        return sorted({opts.index(a) for a in near if a != rec["meta"]["label"]})
    if task == "arith_check":
        return _arith_near(rec["meta"], rec["answer"])
    if task == "code_output_check":
        return _code_output_near(rec["prompt"], rec["answer"], item["cell"][0])
    return _literal_near(rec["prompt"], rec["answer"])


_LIT = re.compile(r"(?<![\w.])\d+(?![\w.])")
PAIRS = False


def _pos_features(task: str, item: dict):
    """f(members) -> tuple of discrete position features of a proposed value (check task) or a listed set (bin task) in the item,
    the one R1 names first: the rule index where it appears (rule, hard), the number-token index (arith), the last-assignment line (code)."""
    base = task if task in CHECK_DO else next(k for k, v in BIN_OF_CHECK.items() if v == task)
    do = CHECK_DO[base]
    rec = item["records"][do]
    opts = list(spec.TASKS[do].options)
    if base in ("rule_check", "hard_check"):
        st = item["aux"]["state"]
        act, dis = [r["action"] for r in st["rules"]], [r["action"] for r in st["disabled"]]

        def f(members):
            names = {opts[e] for e in members}
            first = next((i + 1 for i, a in enumerate(act) if a in names), 0)
            return (first,)
        return f
    if base == "arith_check":
        lits = _LIT.findall(rec["prompt"])

        def f(members):
            names = {str(e) for e in members}
            first = next((i + 1 for i, x in enumerate(lits) if x in names), 0)
            return (first, min(sum(x in names for x in lits), 3))
        return f
    code = rec["prompt"]
    ev = _assign_events(code)
    lines = code.split("\n")
    final = ev[-1][1] if ev else -1

    def f(members):
        has = [j + 1 for j, ln in enumerate(lines) if any(int(x) in members for x in _LIT.findall(ln))]
        return (max((ln for ln, v in ev if v in members), default=0), int(final in members), has[0] if has else 0,
                min(sum(int(x) in members for x in _LIT.findall(code)), 4))
    return f


CHECK_DROP_COST = {"code_output_check": 2.0}   # round 2: its joint keys would otherwise cost too many items (the test split needs 200)
JOINT_TASKS = ("arith_check", "code_output_check", "arith_bin", "code_literal_bin")   # E2 round 2: the four tasks that leaked


def _joint_keys(task: str, item: dict, members) -> list[tuple]:
    """Round 2 (E2 gate, last round): each position feature of the shown value or set crossed with a descriptor of the item (operation
    count and length for arith; trace length, loops and ifs for code), so a boosted tree cannot read yes/no off feature x size
    interactions. Used only by JOINT_TASKS; the other tasks' keys are unchanged."""
    base = task if task in CHECK_DO else next(k for k, v in BIN_OF_CHECK.items() if v == task)
    rec = item["records"][CHECK_DO[base]]
    if base == "arith_check":
        lits = _LIT.findall(rec["prompt"])
        names = {str(e) for e in members}
        pos = [next((i + 1 for i, x in enumerate(lits) if x in names), 0), max((i + 1 for i, x in enumerate(lits) if x in names), default=0),
               min(sum(x in names for x in lits), 3), min(members) if len(members) == 1 else -1]
        d = [len(rec["meta"]["ops"]), len(rec["prompt"]) // 6]
        return [("xj", j, p, k, dv) for j, p in enumerate(pos) for k, dv in enumerate(d)]
    code = rec["prompt"]
    ev, lines = _assign_events(code), code.split("\n")
    has = [j + 1 for j, ln in enumerate(lines) if any(int(x) in members for x in _LIT.findall(ln))]
    asg = [j + 1 for j, ln in enumerate(lines) if "=" in ln and any(int(x) in members for x in _LIT.findall(ln))]
    pos = [max((ln for ln, v in ev if v in members), default=0), int(bool(ev) and ev[-1][1] in members), has[0] if has else 0,
           max(asg, default=0), min(sum(int(x) in members for x in _LIT.findall(code)), 4), sum(v in members for _, v in ev)]
    d = (min(len(ev), 8), sum(bool(re.match(r"\s*(for|while)\b", ln)) for ln in lines), sum(bool(re.match(r"\s*if\b", ln)) for ln in lines))
    first = min((ln for ln, v in ev if v in members), default=0)
    return [("xj", j, p, d) for j, p in enumerate(pos)] + [("xj", "c", min(pos[3], 4), min(first, 4), min(pos[5], 3))]


def _feature_keys(feats: tuple, tag: str = "") -> list[tuple]:
    """The marginals (each feature on its own) and, if PAIRS, the pairs of neighbouring features, as balancing keys."""
    keys = [(tag, j, x) for j, x in enumerate(feats)]
    if PAIRS:
        keys += [(tag + "pair", j, feats[j], feats[j + 1]) for j in range(len(feats) - 1)]
    return keys


def _balance(items: list[list[tuple[int, list[tuple]]]], rng, drop_cost: float = 0.15, iters: int | None = None) -> list:
    """items[i] = [(side, keys), ...]: the states item i can take (side +1 = shown as correct / answer yes, -1 = the other). Returns
    the chosen state index per item, or None for an item dropped. Hill-climbing on: the L1 gap, over the keys, between the +1 and -1
    halves; plus 2 x the gap in how many items each half has; plus `drop_cost` per dropped item. Moves keep the halves' sizes
    equal, so neither the shown value, nor where it sits, nor the item's own answer says which half an item is in."""
    n = len(items)
    gap: Counter = Counter()
    obj = [drop_cost * n]
    state: list = [None] * n
    lists: dict = {1: [], -1: [], 0: list(range(n))}      # kept with side +1, kept with side -1, dropped
    where = {i: i for i in range(n)}                       # index of the item in its list

    def add(i: int, st, sign: int) -> None:
        if st is None:
            obj[0] += drop_cost * sign
            return
        side, keys = items[i][st]
        for k in (*keys, "n"):
            w = 3 if k == "n" or k[0] in ("x", "xpair") else 1       # sizes and positions count more than values
            obj[0] -= w * abs(gap[k])
            gap[k] += sign * side
            obj[0] += w * abs(gap[k])

    def put(i: int, st) -> None:
        old = state[i]
        lst = lists[0 if old is None else items[i][old][0]]
        ix = where[i]
        lst[ix] = lst[-1]
        where[lst[ix]] = ix
        lst.pop()
        add(i, old, -1)
        state[i] = st
        add(i, st, 1)
        lst = lists[0 if st is None else items[i][st][0]]
        where[i] = len(lst)
        lst.append(i)

    def pick(i: int, side: int):
        ok = [st for st, (sd, _) in enumerate(items[i]) if sd == side]
        return rng.choice(ok) if ok else None

    order = list(range(n))
    rng.shuffle(order)
    for r, i in enumerate(order):          # start: every item kept, the sides alternating where the item allows
        want = 1 if r % 2 == 0 else -1
        st = pick(i, want)
        put(i, st if st is not None else pick(i, -want))
    for _ in range(iters or 150 * n):
        before = obj[0]
        m = rng.random()
        undo: list = []

        def do(i, st):
            undo.append((i, state[i]))
            put(i, st)
        if m < 0.3 and (lists[1] or lists[-1]):                    # another state of the same side
            side = rng.choice([sd for sd in (1, -1) if lists[sd]])
            i = rng.choice(lists[side])
            do(i, pick(i, side))
        elif m < 0.6 and lists[1] and lists[-1]:                   # two kept items trade sides
            i, j = rng.choice(lists[1]), rng.choice(lists[-1])
            a, b = pick(i, -1), pick(j, 1)
            if a is None or b is None:
                continue
            do(i, a)
            do(j, b)
        elif m < 0.75 and lists[1] and lists[-1]:                  # drop one of each side
            i, j = rng.choice(lists[1]), rng.choice(lists[-1])
            do(i, None)
            do(j, None)
        elif m < 0.9 and len(lists[0]) > 1:                        # bring back one of each side
            i, j = rng.choice(lists[0]), rng.choice(lists[0])
            a, b = (pick(i, 1), pick(j, -1)) if i != j else (None, None)
            if a is None or b is None:
                continue
            do(i, a)
            do(j, b)
        elif lists[0] and (lists[1] or lists[-1]):                 # swap a kept item for a dropped one on the same side
            k = rng.choice(lists[0])
            side = rng.choice([sd for sd in (1, -1) if lists[sd]])
            st = pick(k, side)
            if st is None:
                continue
            do(rng.choice(lists[side]), None)
            do(k, st)
        else:
            continue
        if obj[0] > before + 1e-9:
            for i, st in reversed(undo):
                put(i, st)
    while len(lists[1]) != len(lists[-1]):                         # equal halves, exactly: drop from the larger the item that costs least
        side = 1 if len(lists[1]) > len(lists[-1]) else -1
        best, best_obj = None, None
        for i in rng.sample(lists[side], min(40, len(lists[side]))):
            old = state[i]
            put(i, None)
            if best_obj is None or obj[0] < best_obj:
                best, best_obj = i, obj[0]
            put(i, old)
        put(best, None)
    return state


def _attach_checks(task: str, items: list[dict], rng) -> list[dict]:
    """One check record per item that has a near miss and survives the balancing (items without a near miss are dropped, not shown
    as correct; items are also dropped, at most as many as it takes, so the halves match in shown value, position and true answer)."""
    do = CHECK_DO[task]
    opts = spec.TASKS[do].options
    near = [_near_misses(task, it) for it in items]
    items, near = [it for it, c in zip(items, near) if c], [c for c in near if c]   # no near miss: the item is dropped
    states = []
    for it, c in zip(items, near):
        f, a = _pos_features(task, it), it["records"][do]["answer"]
        extra = (lambda e: _joint_keys(task, it, {e})) if task in JOINT_TASKS else (lambda e: [])
        key = lambda e, side: _feature_keys(f({e}), "x") + [("v", e), ("a", a)] + extra(e)  # noqa: E731
        states.append([(1, key(a, 1))] + [(-1, key(e, -1)) for e in c])
    chosen = _balance(states, rng, drop_cost=CHECK_DROP_COST.get(task, 0.15))
    out = []
    for it, c, st in zip(items, near, chosen):
        if st is None:
            continue
        d = it["records"][do]
        e, ok = (d["answer"], True) if st == 0 else (c[st - 1], False)
        meta = {"do_answer": d["answer"], "proposed": e, "correct": ok, "near_misses": len(c)}
        rec = _record(task, d["id"], d["prompt"] + CHECK_PROPOSAL + opts[e], 0 if ok else 1, d["template_id"], d["rung"], meta)
        out.append({"template_id": it["template_id"], "cell": it["cell"], "records": {task: rec}})
    return out


def _generate_check(task: str, n: int, seed: int) -> list[dict]:
    items = family_items(CHECK_BASE[task], n, seed)
    recs = [it["records"][task] for it in _attach_checks(task, items, _rng("check", task, seed))]
    _check_records(task, recs)
    if n >= 200:
        assert balance_dev([r["answer"] for r in recs], 2) <= 0.10 + 2 / n, task
    return recs


def _check_splits(family: str, seed: int, sizes: dict, pool_n: int | None) -> tuple[dict, dict]:
    """The do-task's splits (same items), with the check records dealt in."""
    task = FAMILY_TASKS[family][0]
    splits, info = make_splits(CHECK_BASE[family], seed, sizes, pool_n)
    out = {s: _attach_checks(task, items, _rng("check", task, seed, s)) for s, items in splits.items()}
    info = {**info, "dropped_items": {s: len(splits[s]) - len(out[s]) for s in splits}}
    check_splits(family, out, {s: max(1, int(spec.TEMPLATE_CAP * n)) for s, n in sizes.items()})
    return out, info


# ---------------------------------------------------------------- E2 revision R1: binary doing tasks

BIN_QUESTION = "\n\nIs the answer one of: "


# Round 2: arith_bin's first A ({0, 6, 7, 8, 9}, the seed-0 draw) tracked the expression's surface digits (BoW 0.621 on test). The new
# A was fixed by this rule, written before the test split was read and computed on the real train + lrsel arith items only: among the
# 252 halves of 0-9 whose in-A rate is in [0.45, 0.55], take the least max(|AUC - 0.5|) of a bag-of-words rival and of the structural
# boosted tree (5-fold CV, folds grouped by expression), ties by the seed-0 shuffled order of the halves. It is the best of the 252.
BIN_SET_FIXED = {"arith_bin": (1, 5, 6, 8, 9)}


def bin_set(task: str) -> tuple[int, ...]:
    """A: a fixed half of the do-task's options (indices, ascending), drawn with seed 0 (arith_bin: BIN_SET_FIXED); the same for every
    record of the task."""
    if task in BIN_SET_FIXED:
        return BIN_SET_FIXED[task]
    k = len(spec.TASKS[BIN_DO[task]].options)
    return tuple(sorted(_rng("bin", task, 0).sample(range(k), k // 2)))


def bin_suffix(task: str) -> str:
    opts = spec.TASKS[BIN_DO[task]].options
    return BIN_QUESTION + ", ".join(opts[i] for i in bin_set(task)) + "?"


def _bin_keys(task: str, item: dict) -> list[tuple]:
    """Balancing keys of a bin item: the position features of the set A, plus (rule, hard) whether each rule's action is in A, or
    (arith, code) which digits the program or expression contains."""
    do = BIN_DO[task]
    members = set(bin_set(task))
    keys = _feature_keys(_pos_features(task, item)(members), "x")
    rec = item["records"][do]
    if do in ("rule", "hard"):
        st, opts = item["aux"]["state"], spec.TASKS[do].options
        names = {opts[e] for e in members}
        keys.append(("rj", tuple(r["action"] in names for r in st["rules"][:3])))
    if task in JOINT_TASKS:
        keys += _joint_keys(task, item, members)
    return keys


def _attach_bins(task: str, items: list[dict], rng) -> list[dict]:
    """The do-task's item plus "Is the answer one of: A?" (yes = answer in A, index 0). Items are dropped until yes and no are
    equal and their position features are balanced across the two (the first state of the greedy balancing, `_balance`)."""
    do = BIN_DO[task]
    members, suffix = set(bin_set(task)), bin_suffix(task)
    states = [[(1 if it["records"][do]["answer"] in members else -1, _bin_keys(task, it))] for it in items]
    chosen = _balance(states, rng)
    out = []
    for it, st in zip(items, chosen):
        if st is None:
            continue
        d = it["records"][do]
        ok = d["answer"] in members
        meta = {"do_answer": d["answer"], "in_set": ok}
        rec = _record(task, d["id"], d["prompt"] + suffix, 0 if ok else 1, d["template_id"], d["rung"], meta)
        out.append({"template_id": it["template_id"], "cell": it["cell"], "records": {task: rec}})
    return out


def _generate_bin(task: str, n: int, seed: int) -> list[dict]:
    recs = [it["records"][task] for it in _attach_bins(task, family_items(BIN_BASE[task], n, seed), _rng("bin", task, seed))]
    _check_records(task, recs)
    return recs


def _arith_core(item: dict) -> tuple:
    m = item["records"]["arith"]["meta"]
    return (m["a"], m["b"], m["c"], m["ops"])


def _disjoint_cores(splits: dict) -> dict:
    """Round 2, arith_bin: a bag-of-words rival memorises "this exact expression -> its answer" when an expression (same numbers and
    operators, any phrasing) sits in both train/lrsel and test, and the answer fixes the label. Every expression in both is given to
    one side by the parity of its crc32 (label-blind): the other side's items are dropped, so train + lrsel and test share none."""
    fit = {_arith_core(it) for s in ("train", "lrsel") for it in splits[s]}
    both = {_arith_core(it) for it in splits["test"]} & fit
    to_test = {c for c in both if zlib.crc32(repr(c).encode()) & 1}
    out = dict(splits)
    out["test"] = [it for it in splits["test"] if _arith_core(it) not in both or _arith_core(it) in to_test]
    for sp in ("train", "lrsel"):
        out[sp] = [it for it in splits[sp] if _arith_core(it) not in to_test]
    return out


def _bin_splits(family: str, seed: int, sizes: dict, pool_n: int | None) -> tuple[dict, dict]:
    task = FAMILY_TASKS[family][0]
    splits, info = make_splits(BIN_BASE[family], seed, sizes, pool_n)
    full = {s: len(v) for s, v in splits.items()}
    if task == "arith_bin":
        splits = _disjoint_cores(splits)
    out = {s: _attach_bins(task, items, _rng("bin", task, seed, s)) for s, items in splits.items()}
    info = {**info, "dropped_items": {s: full[s] - len(out[s]) for s in splits}}
    check_splits(family, out, {s: max(1, int(spec.TEMPLATE_CAP * n)) for s, n in sizes.items()})
    return out, info


# ================================================================ splits

def _interleave(rng, group: list[dict]) -> list[dict]:
    """Items ordered so consecutive picks come from different templates (a cap then bites late)."""
    g = defaultdict(list)
    for it in group:
        g[it["template_id"]].append(it)
    ts = sorted(g)
    rng.shuffle(ts)
    for t in ts:
        rng.shuffle(g[t])
    out, r = [], 0
    while len(out) < len(group):
        for t in ts:
            if len(g[t]) > r:
                out.append(g[t][r])
        r += 1
    return out


def select(items: list[dict], size: int, rng, weight, cap: int | None, cells: list | None = None):
    """size items with cell counts proportional to weight(cell), at most `cap` per template. Returns (items, shortfall)."""
    by_cell = defaultdict(list)
    for it in items:
        by_cell[it["cell"]].append(it)
    keys = sorted(set(by_cell) | set(cells or []), key=repr)
    quota = _quotas(size, {c: weight(c) for c in keys})
    used, chosen, rest = Counter(), [], []
    for c in keys:
        taken = 0
        for it in _interleave(rng, by_cell.get(c, [])):
            if taken < quota[c] and (cap is None or used[it["template_id"]] < cap):
                chosen.append(it)
                used[it["template_id"]] += 1
                taken += 1
            else:
                rest.append(it)
    short = size - len(chosen)
    if short > 0:
        rng.shuffle(rest)
        for it in rest:
            if len(chosen) == size:
                break
            if cap is None or used[it["template_id"]] < cap:
                chosen.append(it)
                used[it["template_id"]] += 1
    rng.shuffle(chosen)
    return chosen, short


def deal(items: list[dict], rng, sizes: dict, caps: dict) -> dict:
    """Whole templates to splits, always to the split furthest below its (scaled) size; a template contributes at most the split's cap."""
    groups = defaultdict(list)
    for it in items:
        groups[it["template_id"]].append(it)
    templates = sorted(groups)
    rng.shuffle(templates)
    low = min(caps.values())
    scale = max(1.0, sum(min(len(g), low) for g in groups.values()) / sum(sizes.values()))
    out = {s: [] for s in sizes}
    for t in templates:
        s = max(sizes, key=lambda n: (1 - len(out[n]) / (sizes[n] * scale), sizes[n]))
        g = groups[t][:]
        rng.shuffle(g)
        out[s].extend(g[:caps[s]])
    return out


def make_splits(family: str, seed: int = spec.SPLIT_SEED, sizes: dict | None = None,
                pool_n: int | None = None) -> tuple[dict, dict]:
    """({split: items}, info). The pool grows (x1.5) until every split meets its cell quotas; info says if it did not."""
    sizes = sizes or spec.SPLITS
    if family in CHECK_BASE:
        return _check_splits(family, seed, sizes, pool_n)
    if family in BIN_BASE:
        return _bin_splits(family, seed, sizes, pool_n)
    caps = {s: max(1, int(spec.TEMPLATE_CAP * n)) for s, n in sizes.items()}
    total = sum(sizes.values())
    fixed = family in ("fact", "route_judgement", "route_work")
    factors = (1.0,) if fixed else POOL_FACTORS
    best = None
    for factor in factors:
        n = pool_n if pool_n else int(total * factor)
        pool = family_items(family, None if fixed else n, seed)
        dealt = deal(pool, _rng("deal", family, seed), sizes, caps)
        splits, short = {}, {}
        weights = (lambda c, f=family: cell_weight(f, c))
        cells = sorted({it["cell"] for it in pool}, key=repr)
        for s in sizes:
            splits[s], short[s] = select(dealt[s], sizes[s], _rng("select", family, seed, s), weights, caps[s], cells)
        best = (splits, {"pool_items": len(pool), "pool_factor": factor, "quota_shortfall": short})
        if not any(short.values()) and all(len(v) == sizes[s] for s, v in splits.items()):
            break
    splits, info = best
    check_splits(family, splits, caps)
    return splits, info


def check_splits(family: str, splits: dict, caps: dict | None = None) -> None:
    """Template-disjoint across splits, per-template cap, enough templates per split."""
    owner: dict[str, str] = {}
    for s, items in splits.items():
        tc = Counter(it["template_id"] for it in items)
        for t in tc:
            assert owner.setdefault(t, s) == s, f"{family}: template {t!r} in {owner[t]} and {s}"
        assert len(tc) >= spec.MIN_TEMPLATES_PER_SPLIT, f"{family}/{s}: {len(tc)} templates"
        if caps:
            assert max(tc.values()) <= caps[s], f"{family}/{s}: template over cap"
        for t in FAMILY_TASKS[family]:
            ids = [it["records"][t]["id"] for it in items]
            assert len(set(ids)) == len(ids)
