"""Steps study v3 (docs/ATLAS-steps3.md, Stage 2): generators for steps3_swap (G1), steps3_total (G3a),
steps3_order (G3b) and steps3_mix (G4).

Generation never touches a tokenizer. Variable names in G4 come from steps.NAMES (single tokens in both tokenizers); G1
and G3 use letters and x/y/z. Line shapes are fixed per family so token counts match across cells: G1 always has 4 swap
lines, G3 always 3 constant lines and N_BODY body lines, G4 always MIX_L lines with a filler mix chosen so the Qwen token
total is MIX_T in every cell (line costs in MIX_COST: constant 4, copy 3, operation 6).

Every item is checked by `verify` (from the prompt and meta only): the stored answer is recomputed, and every step
that is supposed to matter is removed (or, for a copy hop, re-pointed at the decoy chain) and the answer must change.
Items that a gate rule (RULES) would answer correctly are rejected, except the rules in `expected` (at m = 1 the
last-operation rule is the real solution). Splits are dealt by whole template (the line-role layout).

steps3_total is balanced a second way (rebuild 2026-11-18): per cell and split, answer x last operand on x, answer x first
operand on x and answer x start value are kept as even as greedy choice among BALANCE_PICKS candidates allows.

Known limits:
- Template ids are near-unique per item (they carry slot numbers, signs and filler layout), so "whole template" splitting
  is weak: it keeps train and test disjoint but does not group items by structure.
- G3 (total, order) has N_BODY = 6 body lines; at 4, m = 4 had too few layouts for five template-disjoint splits.
- Rule shortcuts score 0.00 by construction: the generator rejects every item a gate rule (RULES) gets right, except the
  rules in `expected`, so a rule's digit is never the answer. The gate numbers for rules therefore carry no information
  beyond that; bag-of-words is the live check.
"""
from __future__ import annotations

import itertools
import random
import re
from collections import Counter

from surgery.atlas import spec, steps
from surgery.atlas.steps import NAMES, SPLIT_ORDER, _balanced_cycle

TASKS3 = ("steps3_swap", "steps3_total", "steps3_order", "steps3_mix")
LETTERS = spec.STEPS3_LETTERS
MAX_TRIES = 20000
N_BODY = 6       # G3 body lines (m on x, the rest on y and z); enough layouts for 5 template-disjoint splits at m = 4

# G4 shape: MIX_L lines before the print, Qwen token cost per line type (name, =, value): constant 4, copy 3, operation 6.
MIX_L, MIX_T = 21, 96
MIX_COST = {"const": 4, "copy": 3, "op": 6}


def _rng(*parts) -> random.Random:
    return random.Random("steps3-" + "-".join(str(p) for p in parts))


def cell_sizes(task: str) -> dict[str, dict[str, int]]:
    """{split: {cell: records}}: v2's steps_ops sizes for every cell."""
    per = {s: steps.cell_sizes("steps_ops")[s]["s1"] for s in SPLIT_ORDER}
    return {s: {c: per[s] for c in spec.TASKS[task].rungs} for s in SPLIT_ORDER}


# ================================================================ program IR (G3, G4)
# ("const", v, val)  ("copy", v, src)  ("sum", v, src, sign, b)  ("mod", v, sign, b)   [mod: v = (v sign b) % 10]

def render_line(l: tuple) -> str:
    k = l[0]
    if k in ("const", "copy"):
        return f"{l[1]} = {l[2]}"
    if k == "sum":
        return f"{l[1]} = {l[2]} {l[3]} {l[4]}"
    return f"{l[1]} = ({l[1]} {l[2]} {l[3]}) % 10"


def render_program(lines: list[tuple], pv: str) -> str:
    return "\n".join([render_line(l) for l in lines] + [f"print({pv})"])


_RE_CONST = re.compile(r"(\w+) = (\d)")
_RE_SUM = re.compile(r"(\w+) = (\w+) ([+-]) (\d)")
_RE_MOD = re.compile(r"(\w+) = \((\w+) ([+\-*]) (\d)\) % 10")
_RE_COPY = re.compile(r"(\w+) = (\w+)")
_RE_PRINT = re.compile(r"print\((\w+)\)")


def parse_program(text: str) -> tuple[list[tuple], str]:
    lines = text.split("\n")
    m = _RE_PRINT.fullmatch(lines[-1])
    out = []
    for s in lines[:-1]:
        if m2 := _RE_CONST.fullmatch(s):
            out.append(("const", m2[1], int(m2[2])))
        elif m2 := _RE_SUM.fullmatch(s):
            out.append(("sum", m2[1], m2[2], m2[3], int(m2[4])))
        elif m2 := _RE_MOD.fullmatch(s):
            assert m2[1] == m2[2], s
            out.append(("mod", m2[1], m2[3], int(m2[4])))
        elif m2 := _RE_COPY.fullmatch(s):
            out.append(("copy", m2[1], m2[2]))
        else:
            raise ValueError(f"cannot parse line {s!r}")
    return out, m[1]


def _apply(v: int, sign: str, b: int) -> int:
    return v + b if sign == "+" else v - b if sign == "-" else v * b


def run_lines(lines: list[tuple], strict: bool = False) -> dict[str, int]:
    """Execute; strict raises ValueError if a sum line leaves 0-9."""
    env: dict[str, int] = {}
    for l in lines:
        k = l[0]
        if k == "const":
            env[l[1]] = l[2]
        elif k == "copy":
            env[l[1]] = env[l[2]]
        elif k == "sum":
            v = _apply(env[l[2]], l[3], l[4])
            if strict and not 0 <= v <= 9:
                raise ValueError(l)
            env[l[1]] = v
        else:
            env[l[1]] = _apply(env[l[1]], l[2], l[3]) % 10
    return env


def solve_program(prompt: str) -> int:
    lines, pv = parse_program(prompt)
    return run_lines(lines, strict=False)[pv]


def literal_digits(prompt: str) -> list[int]:
    """Literal digits of the program text in order (the print line and the modulus 10 are not literals)."""
    body = "\n".join(s for s in prompt.split("\n") if not s.startswith("print(")).replace("% 10", "")
    return [int(c) for c in re.findall(r"\d", body)]


# ================================================================ G1: steps3_swap

def simulate_swaps(row: list[str], swaps: list[tuple[int, int]]) -> list[str]:
    row = list(row)
    for i, j in swaps:
        row[i], row[j] = row[j], row[i]
    return row


_RE_ROW = re.compile(r"(\d) ([A-J])")
_RE_SWAP = re.compile(r"swap slot (\d) and slot (\d)")
_RE_ASK = re.compile(r"Which letter is in slot (\d)\?")


def parse_swap(prompt: str) -> tuple[list[str], list[tuple[int, int]], int]:
    ls = prompt.split("\n")
    row = [""] * 10
    for i, c in _RE_ROW.findall(ls[0]):
        row[int(i)] = c
    swaps = [(int(a), int(b)) for a, b in (_RE_SWAP.fullmatch(s).groups() for s in ls[1:-1])]
    return row, swaps, int(_RE_ASK.fullmatch(ls[-1])[1])


def solve_swap(prompt: str) -> int:
    row, swaps, p = parse_swap(prompt)
    return LETTERS.index(simulate_swaps(row, swaps)[p])


def _swap_item(rng: random.Random, cell: str, tg: int) -> dict | None:
    m = int(cell[1])
    p = rng.randrange(10)
    path, back = [p], []
    for _ in range(m):
        d = rng.choice([s for s in range(10) if s not in path])
        back.append((path[-1], d) if rng.random() < 0.5 else (d, path[-1]))
        path.append(d)
    on = back[::-1]                                   # forward order; the last one touches slot p
    free = [s for s in range(10) if s not in path]
    fil = []
    for _ in range(4 - m):
        a, b = rng.sample(free, 2)
        fil.append((a, b) if rng.random() < 0.5 else (b, a))
    fpos = set(rng.sample(range(4), 4 - m))
    seq, oi, fi = [], iter(on), iter(fil)
    for k in range(4):
        seq.append((*next(fi), False) if k in fpos else (*next(oi), True))
    src = path[-1]
    others = [c for c in LETTERS if c != LETTERS[tg]]
    rng.shuffle(others)
    row = [LETTERS[tg] if s == src else others.pop() for s in range(10)]
    prompt = "\n".join(["Slots: " + ", ".join(f"{i} {row[i]}" for i in range(10))]
                       + [f"swap slot {i} and slot {j}" for i, j, _ in seq] + [f"Which letter is in slot {p}?"])
    tid = f"{cell}|p{p}|" + ",".join(f"{i}{j}{'*' if o else ''}" for i, j, o in seq)
    meta = {"cell": cell, "m": m, "row": row, "p": p, "swaps": [{"i": i, "j": j, "on_path": o} for i, j, o in seq],
            "source_slot": src, "answer_letter": LETTERS[tg]}
    return {"template_id": tid, "prompt": prompt, "meta": meta, "answer": tg}


def _verify_swap(rec: dict) -> None:
    meta = rec["meta"]
    row, p = meta["row"], meta["p"]
    sw = [(s["i"], s["j"], s["on_path"]) for s in meta["swaps"]]
    ans = LETTERS.index(simulate_swaps(row, [(i, j) for i, j, _ in sw])[p])
    assert ans == rec["answer"] == solve_swap(rec["prompt"]), rec["id"]
    on = [(i, j) for i, j, o in sw if o]
    assert len(on) == meta["m"] and LETTERS.index(simulate_swaps(row, on)[p]) == ans, "filler swaps changed the answer"
    for k, (i, j, o) in enumerate(sw):
        rest = [(a, b) for kk, (a, b, _) in enumerate(sw) if kk != k]
        same = LETTERS.index(simulate_swaps(row, rest)[p]) == ans
        assert same != o, f"{rec['id']}: swap {k} on_path={o} but dropping it {'keeps' if same else 'changes'} the answer"


def _swap_rule_initial(rec: dict) -> int:
    return LETTERS.index(rec["meta"]["row"][rec["meta"]["p"]])


def _swap_rule_only(rec: dict, which: int) -> int:
    meta = rec["meta"]
    on = [(s["i"], s["j"]) for s in meta["swaps"] if s["on_path"]]
    return LETTERS.index(simulate_swaps(meta["row"], [on[which]])[meta["p"]])


# ================================================================ G3: steps3_total / steps3_order

def _tot_item(rng: random.Random, cell: str, tg: int, order: bool) -> dict | None:
    m = int(cell[1])
    pos = set(rng.sample(range(N_BODY), m))
    kinds = ("+", "-", "*2", "*3") if order else ("+", "-")
    roles = [("x" if k in pos else rng.choice("yz"), rng.choice(kinds)) for k in range(N_BODY)]
    header = rng.sample("xyz", 3)
    cur = {v: rng.randint(0, 9) for v in "xyz"}
    start = cur["x"]
    lines = [("const", v, cur[v]) for v in header]
    real = []
    for var, kind in roles:
        sg = kind[0]
        if order:
            b = int(kind[1]) if sg == "*" else rng.randint(1, 9)
            lines.append(("mod", var, sg, b))
            cur[var] = _apply(cur[var], sg, b) % 10
        else:
            bs = [b for b in range(1, 10) if 0 <= _apply(cur[var], sg, b) <= 9]
            if not bs:
                return None
            b = rng.choice(bs)
            lines.append(("sum", var, var, sg, b))
            cur[var] = _apply(cur[var], sg, b)
        if var == "x":
            real.append((sg, b))
    if cur["x"] != tg:
        return None
    prompt = render_program(lines, "x")
    tid = f"{cell}|{''.join(header)}|" + ",".join(v + k for v, k in roles)
    meta = {"cell": cell, "m": m, "start": start, "steps": [{"sign": s, "b": b} for s, b in real],
            "roles": [v + k for v, k in roles], "header": header, "mod10": order}
    return {"template_id": tid, "prompt": prompt, "meta": meta, "answer": tg}


def _verify_tot(rec: dict) -> None:
    lines, pv = parse_program(rec["prompt"])
    assert pv == "x" and run_lines(lines, strict=not rec["meta"]["mod10"])[pv] == rec["answer"], rec["id"]
    xs = [k for k, l in enumerate(lines) if l[1] == "x" and l[0] in ("sum", "mod")]
    assert len(xs) == rec["meta"]["m"]
    for k in xs:
        assert run_lines(lines[:k] + lines[k + 1:])["x"] != rec["answer"], f"{rec['id']}: dropping line {k} keeps the answer"
    if rec["meta"]["mod10"] and rec["meta"]["m"] >= 2:
        st = [(lines[k][2], lines[k][3]) if lines[k][0] == "mod" else None for k in xs]
        a = rec["meta"]["start"]

        def go(order):
            v = a
            for sg, b in order:
                v = _apply(v, sg, b) % 10
            return v
        assert any(go(p) != rec["answer"] for p in itertools.permutations(st)), f"{rec['id']}: order does not matter"


def _tot_start(rec: dict) -> int:
    return rec["meta"]["start"]


def _tot_last_op(rec: dict) -> int:
    sg, b = rec["meta"]["steps"][-1]["sign"], rec["meta"]["steps"][-1]["b"]
    return _apply(rec["meta"]["start"], sg, b) % 10


def _first_lit(rec: dict) -> int:
    return literal_digits(rec["prompt"])[0]


def _last_lit(rec: dict) -> int:
    return literal_digits(rec["prompt"])[-1]


# ================================================================ G4: steps3_mix

def _mix_fills() -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    """(k_copy, k_const, k_op) filler mixes (k_const counts the first filler line) that give MIX_L lines and MIX_T tokens."""
    out = {}
    for h in (0, 2, 4):
        for m in (1, 2, 3):
            f = MIX_L - 2 - 2 * h - 2 * m
            tok = MIX_T - 2 * MIX_COST["const"] - 2 * h * MIX_COST["copy"] - 2 * m * MIX_COST["op"]
            out[(h, m)] = [(k3, k4, f - k3 - k4) for k3 in range(f + 1) for k4 in range(1, f + 1 - k3)
                           if MIX_COST["copy"] * k3 + MIX_COST["const"] * k4 + MIX_COST["op"] * (f - k3 - k4) == tok]
            assert out[(h, m)], (h, m)
    return out


MIX_FILL = _mix_fills()


def _sum_step(rng: random.Random, cur: int) -> tuple[str, int]:
    """A random (sign, b) with b in 1-9 that keeps cur +/- b inside 0-9."""
    return rng.choice([(sg, b) for sg in "+-" for b in range(1, 10) if 0 <= _apply(cur, sg, b) <= 9])


def _chain(rng: random.Random, h: int, m: int, names: list[str], end_op: bool):
    """([(line, kind)], final value, name list used) for one chain: constant, then h copies and m in-place +/- operations."""
    mid = ["H"] * h + ["O"] * (m - 1 if end_op and h else m)
    rng.shuffle(mid)
    seq = mid + (["O"] if end_op and h else [])
    var = names[0]
    cur = rng.randint(0, 9)
    ev = [(("const", var, cur), "const")]
    nxt = 1
    for t in seq:
        if t == "H":
            ev.append((("copy", names[nxt], var), "hop"))
            var, nxt = names[nxt], nxt + 1
        else:
            sg, b = _sum_step(rng, cur)
            ev.append((("sum", var, var, sg, b), "op"))
            cur = _apply(cur, sg, b)
    return ev, cur


def _merge(rng: random.Random, lists: list[list]) -> list:
    lists = [list(l) for l in lists]
    out = []
    while any(lists):
        w = [len(l) for l in lists]
        k = rng.choices(range(len(lists)), weights=w)[0]
        out.append(lists[k].pop(0))
    return out


def _mix_item(rng: random.Random, cell: str, tg: int) -> dict | None:
    h, m = int(cell[1]), int(cell[4])
    k3, k4, k6 = rng.choice(MIX_FILL[(h, m)])
    names = rng.sample(NAMES, 2 * (h + 1) + k4 + k3)
    cn, dn, fn = names[:h + 1], names[h + 1:2 * h + 2], names[2 * h + 2:]
    for _ in range(80):
        cev, cval = _chain(rng, h, m, cn, True)
        if cval == tg:
            break
    else:
        return None
    for _ in range(80):
        dev, dval = _chain(rng, h, m, dn, True)
        if dval != tg:
            break
    else:
        return None
    # fillers: a first constant line, then copies, constants and operations on filler variables only
    fvals = {fn[0]: rng.randint(0, 9)}
    first = (("const", fn[0], fvals[fn[0]]), "fconst")
    types = ["c"] * (k4 - 1) + ["h"] * k3 + ["o"] * k6
    rng.shuffle(types)
    fev, nf = [], 1
    for t in types:
        if t == "c":
            fvals[fn[nf]] = rng.randint(0, 9)
            fev.append((("const", fn[nf], fvals[fn[nf]]), "fconst"))
            nf += 1
        elif t == "h":
            src = rng.choice(list(fvals))
            fvals[fn[nf]] = fvals[src]
            fev.append((("copy", fn[nf], src), "fhop"))
            nf += 1
        else:
            v = rng.choice(list(fvals))
            sg, b = _sum_step(rng, fvals[v])
            fvals[v] = _apply(fvals[v], sg, b)
            fev.append((("sum", v, v, sg, b), "fop"))
    tagged = [(l, "C" + k) for l, k in cev]
    dtag = [(l, "D" + k) for l, k in dev]
    merged = [(first[0], "Fconst")] + _merge(rng, [tagged, dtag, [(l, "F" + k[1:]) for l, k in fev]])
    lines = [l for l, _ in merged]
    tags = [t for _, t in merged]
    # the decoy must have started before the chain's first copy (the hop check re-points a copy at it)
    first_hop = next((i for i, t in enumerate(tags) if t == "Chop"), None)
    if first_hop is not None and not any(t == "Dconst" for t in tags[:first_hop]):
        return None
    pv = cn[-1]
    env, dcur, ccur, hop_alts, last_hop_src = {}, None, None, [], None
    for i, (l, t) in enumerate(merged):
        if t == "Chop":
            if env[dcur] == env[l[2]]:
                return None
            hop_alts.append({"line": i + 1, "alt_var": dcur, "alt_value": env[dcur], "chain_value": env[l[2]]})
            last_hop_src = env[l[2]]
        if l[0] == "const":
            env[l[1]] = l[2]
        elif l[0] == "copy":
            env[l[1]] = env[l[2]]
        else:
            env[l[1]] = _apply(env[l[2]], l[3], l[4])
        if t.startswith("D"):
            dcur = l[1]
    assert env[pv] == tg
    dval = env[dn[-1]]
    prompt = render_program(lines, pv)
    chain_ops = [i + 1 for i, t in enumerate(tags) if t == "Cop"]
    hop_lines = [i + 1 for i, t in enumerate(tags) if t == "Chop"]
    tid = f"{cell}|" + ",".join(t + (l[3] if l[0] == "sum" else "") for l, t in merged)
    meta = {"cell": cell, "h": h, "m": m, "print_var": pv, "line_roles": tags, "chain_const_line": tags.index("Cconst") + 1,
            "hop_lines": hop_lines, "op_lines": chain_ops, "decoy_const_line": tags.index("Dconst") + 1,
            "decoy_hop_lines": [i + 1 for i, t in enumerate(tags) if t == "Dhop"],
            "decoy_op_lines": [i + 1 for i, t in enumerate(tags) if t == "Dop"],
            "chain_vars": cn, "decoy_vars": dn, "decoy_print_var": dn[-1], "decoy_value": dval,
            "value_before_last_hop": last_hop_src if hop_lines else tg, "hop_alts": hop_alts,
            "fill": {"copies": k3, "consts": k4, "ops": k6}}
    return {"template_id": tid, "prompt": prompt, "meta": meta, "answer": tg}


def _verify_mix(rec: dict) -> None:
    meta, ans = rec["meta"], rec["answer"]
    lines, pv = parse_program(rec["prompt"])
    assert pv == meta["print_var"] and run_lines(lines, strict=True)[pv] == ans, rec["id"]
    assert len(lines) == MIX_L
    env = run_lines(lines, strict=True)
    assert env[meta["decoy_print_var"]] == meta["decoy_value"] != ans
    for ln in meta["op_lines"]:
        k = ln - 1
        assert run_lines(lines[:k] + lines[k + 1:])[pv] != ans, f"{rec['id']}: dropping op line {ln} keeps the answer"
    assert len(meta["op_lines"]) == meta["m"] and len(meta["hop_lines"]) == meta["h"]
    assert len(meta["decoy_op_lines"]) == meta["m"] and len(meta["decoy_hop_lines"]) == meta["h"]
    for a in meta["hop_alts"]:
        k = a["line"] - 1
        l = lines[k]
        alt = lines[:k] + [("copy", l[1], a["alt_var"])] + lines[k + 1:]
        assert run_lines(alt)[pv] != ans, f"{rec['id']}: re-pointing hop line {a['line']} keeps the answer"
    assert len(meta["hop_alts"]) == meta["h"]


def _sum_lines(prompt: str) -> list[tuple]:
    return [l for l in parse_program(prompt)[0] if l[0] == "sum"]


def _mix_sum_all(rec: dict) -> int:
    v = literal_digits(rec["prompt"])[0]
    for l in _sum_lines(rec["prompt"]):
        v = _apply(v, l[3], l[4])
    return v % 10


def _mix_own_ops(rec: dict) -> int:
    v = literal_digits(rec["prompt"])[0]
    pv = rec["meta"]["print_var"]
    for l in _sum_lines(rec["prompt"]):
        if l[1] == pv:
            v = _apply(v, l[3], l[4])
    return v % 10


def _mix_early(rec: dict) -> int:
    return rec["meta"]["value_before_last_hop"]


def _mix_decoy(rec: dict) -> int:
    return rec["meta"]["decoy_value"]


def _mix_nearest(rec: dict) -> int:
    for line in reversed(rec["prompt"].split("\n")[:-1]):
        if m := re.fullmatch(r"\w+ = (\d)", line):
            return int(m[1])
    return -1


# ================================================================ rules (shortcut gates) and expectations

RULES = {
    "steps3_swap": {"initial_letter_p": _swap_rule_initial, "last_swap_only": lambda r: _swap_rule_only(r, -1),
                    "first_swap_only": lambda r: _swap_rule_only(r, 0)},
    "steps3_total": {"start_only": _tot_start, "last_op_on_start": _tot_last_op, "first_literal": _first_lit,
                     "last_literal": _last_lit},
    "steps3_order": {"start_only": _tot_start, "last_op_on_start": _tot_last_op, "first_literal": _first_lit,
                     "last_literal": _last_lit},
    "steps3_mix": {"sum_all_operands": _mix_sum_all, "own_ops_first_literal": _mix_own_ops,
                   "stop_one_hop_early": _mix_early, "decoy_value": _mix_decoy, "nearest_binding": _mix_nearest},
}


def expected(task: str, cell: str) -> set[str]:
    """Rules that are the real solution in this cell (excluded from pass/fail and from item rejection)."""
    if task == "steps3_swap" and cell == "m1":
        return {"last_swap_only", "first_swap_only"}
    if task in ("steps3_total", "steps3_order") and cell == "m1":
        return {"last_op_on_start"}
    if task == "steps3_mix" and cell.startswith("h0"):
        return {"stop_one_hop_early"}
    return set()


SOLVERS = {"steps3_swap": solve_swap, "steps3_total": solve_program, "steps3_order": solve_program,
           "steps3_mix": solve_program}


def verify(rec: dict) -> None:
    """Recompute the answer from the prompt and check necessity (and order dependence) from the prompt and meta."""
    assert SOLVERS[rec["task"]](rec["prompt"]) == rec["answer"], rec["id"]
    {"steps3_swap": _verify_swap, "steps3_total": _verify_tot, "steps3_order": _verify_tot,
     "steps3_mix": _verify_mix}[rec["task"]](rec)


# ================================================================ dealing

def _item(task: str, rng: random.Random, cell: str, tg: int) -> dict | None:
    if task == "steps3_swap":
        return _swap_item(rng, cell, tg)
    if task == "steps3_mix":
        return _mix_item(rng, cell, tg)
    return _tot_item(rng, cell, tg, task == "steps3_order")


def _balance_keys(rec: dict) -> list[tuple]:
    """steps3_total: the pairs kept even per cell and split (rebuild 2026-11-18): answer x last and first operand on x, answer x start."""
    st, a = rec["meta"]["steps"], rec["answer"]
    return [("last", a, st[-1]["b"]), ("first", a, st[0]["b"]), ("start", a, rec["meta"]["start"])]


BALANCE_PICKS = 8      # candidates per item among which the least-used (answer, operand, start) pairs win


def _cell_split(task: str, cell: str, split: str, n: int, seed: int, owner: dict) -> list[dict]:
    rng = _rng(task, seed, cell, split)
    skip = expected(task, cell)
    picks = BALANCE_PICKS if task == "steps3_total" else 1
    used: Counter = Counter()
    recs = []
    for i, tg in enumerate(_balanced_cycle(rng, n)):
        cands = []
        for _ in range(MAX_TRIES):
            it = _item(task, rng, cell, tg)
            if it is None or owner.get(it["template_id"], split) != split:
                continue
            rec = {"id": f"{cell}-{split}-{i:04d}", "task": task, "prompt": it["prompt"], "answer": it["answer"],
                   "template_id": it["template_id"], "rung": cell, "meta": it["meta"]}
            if any(fn(rec) == rec["answer"] for name, fn in RULES[task].items() if name not in skip):
                continue
            try:
                verify(rec)
            except AssertionError:
                if task == "steps3_swap":
                    raise
                continue
            cands.append(rec)
            if len(cands) == picks:
                break
        if not cands:
            raise RuntimeError(f"{task}/{cell}/{split}: no item for answer {tg}")
        rec = cands[0] if picks == 1 else min(cands, key=lambda r: (sum(used[k] for k in _balance_keys(r)), rng.random()))
        if picks > 1:
            used.update(_balance_keys(rec))
        owner[rec["template_id"]] = split
        recs.append(rec)
    return recs


def generate(task: str, seed: int = spec.SPLIT_SEED) -> dict[str, list[dict]]:
    """{split: records} for one steps3 task. Deterministic. A template claimed by an earlier split is never used by a later one."""
    sizes = cell_sizes(task)
    out: dict[str, list[dict]] = {s: [] for s in SPLIT_ORDER}
    owner: dict = {}
    for s in SPLIT_ORDER:
        for cell, n in sizes[s].items():
            out[s] += _cell_split(task, cell, s, n, seed, owner)
    return out
