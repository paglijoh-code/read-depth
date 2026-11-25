"""Steps study v2 (docs/ATLAS-steps.md): generators for steps_copy (F1), steps_ops (F2) and steps_describe (F4).

Generation never touches a tokenizer. Variable names are drawn from NAMES, 4-letter words that are one token (with and
without a leading space) under both Qwen3 and SmolLM2, so re-pointing a link never changes a token count. F4 clue
phrasings carry their Qwen token counts in steps_concepts.json, and items are accepted only inside a token window.
Token-level meta (F1 distances in tokens) is added by `annotate_f1_tokens`, called by scripts/steps_build.py.

Splits are dealt by whole template (F1: the line-role layout, shared by a twin group; F2: label/order/sign patterns;
F4: concept plus the set of shown features). Sizes per cell are in `cell_sizes`, not in spec.SPLITS.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from surgery.atlas import spec

STEPS_TASKS = ("steps_copy", "steps_ops", "steps_describe")
SPLIT_ORDER = ("discovery", "train", "lrsel", "selection", "test")

NAMES = ["aged", "anti", "aval", "avid", "away", "axes", "bank", "belt", "blue", "boat", "boot", "born", "bury", "cast",
         "ceil", "chin", "cook", "crop", "disk", "dogs", "door", "echo", "empt", "exam", "fact", "fail", "fair", "fall",
         "fast", "feel", "felt", "firm", "five", "flat", "flex", "flip", "fold", "foot", "fort", "four", "gang", "gene",
         "gent", "girl", "give", "gold", "gone", "grab", "hack", "hair", "hall", "have", "heat", "heed", "hell", "hide",
         "hind", "hole", "hood", "hook", "inch", "iris", "iron", "itch", "jack", "kick", "kill", "late", "less", "lift",
         "live", "loan", "look", "lord", "male", "mant", "many", "mass", "mate", "mere", "mind", "mine", "more", "most",
         "much", "must", "near", "none", "noun", "odor", "once", "oval", "oven", "over", "pace", "pain", "park", "perm",
         "phon", "phot", "plug", "plus", "pose", "poss", "prof", "punk", "quad", "race", "raid", "rain", "rent", "reve",
         "rice", "ring", "risk", "road", "rock", "rose", "rush", "rust", "sale", "same", "seat", "sett", "shed", "skin",
         "slow", "snap", "soon", "star", "stay", "stud", "tail", "tank", "teen", "tele", "then", "they", "thin", "thro",
         "tick", "tone", "town", "trip", "tube", "upon", "very", "vine", "viol", "wake", "wall", "warm", "wash", "wave",
         "ways", "weak", "when", "wide", "wine", "your"]
COMMENT_WORDS = ["baked", "beetle", "belief", "border", "buoy", "bush", "chant", "chest", "chew", "chol", "commit",
                 "cooper", "dairy", "dinner", "divide", "drink", "drip", "drop", "eighth", "enable", "enough", "ensure",
                 "entail", "fare", "fiber", "gains", "gamma", "glad", "glue", "ground", "heir", "hem", "hive", "ill",
                 "intent", "juicy", "kan", "lance", "lonely", "maiden", "main", "mango", "meat", "medial", "mig",
                 "mosque", "oppose", "optic", "peek", "pillow", "pine", "prick", "quite", "quot", "relic", "rogue",
                 "secret", "shore", "song", "streak", "strict", "stripe", "super", "sweet", "symbol", "tandem", "tape",
                 "there", "this", "tile", "toss", "tragic", "trivia", "vera", "wages", "waving", "weird", "wood",
                 "yummy", "zig"]


def _rng(*parts) -> random.Random:
    return random.Random("steps-" + "-".join(str(p) for p in parts))


def _balanced_cycle(rng: random.Random, n: int, k: int = 10, offset: int = 0) -> list[int]:
    """n values from range(k), as even as possible, in random order; the extra values start at `offset`."""
    out = [(offset + i) % k for i in range(n)]
    rng.shuffle(out)
    return out


# ================================================================ F1: steps_copy

F1_POS = ("far", "middle", "near")
F1_MAIN = [f"L{L}_{p}" for L in (1, 2, 3, 4) for p in F1_POS]
F1_UNIT = [f"L{L}_{p}_unit" for L in (1, 4) for p in F1_POS]
F1_PAD_A, F1_PAD_B = "L1_far_pad", "L1_far_pad_keep"
F1_CELLS = F1_MAIN + F1_UNIT + [F1_PAD_B, F1_PAD_A]     # all but F1_PAD_A are dealt in triplets
N_LINES = 24
N_CONST, N_COPY = 11, 12          # 23 assignment lines: 11 constants (2 sources + 9 fillers) and 12 copies, every program
TSRC = {"far": (3,), "middle": (11,), "near": (17,)}   # rebuild 2026-11-17: print on line 24


def cell_sizes(task: str) -> dict[str, dict[str, int]]:
    """{split: {cell: records}}. F1 triplet cells hold a multiple of 3; train 842 and lrsel 180 here (19 triplet cells + 1 single cell), spread evenly."""
    if task == "steps_copy":
        trip = [c for c in F1_CELLS if c != F1_PAD_A]
        per = {"discovery": (252, 250), "train": (42, 44), "lrsel": (9, 9), "selection": (51, 50), "test": (102, 100)}
        return {s: {**{c: per[s][0] for c in trip}, F1_PAD_A: per[s][1]} for s in SPLIT_ORDER}
    if task == "steps_ops":
        per = {"discovery": 250, "train": 200, "lrsel": 50, "selection": 50, "test": 100}
        return {s: {f"s{k}": per[s] for k in (1, 2, 3, 4)} for s in SPLIT_ORDER}
    per = {"discovery": 250, "train": 100, "lrsel": 25, "selection": 50, "test": 100}
    return {s: {f"n{n}_{p}": per[s] for n in (1, 2, 3, 4) for p in ("far", "near")} for s in SPLIT_ORDER}


def _src_lines(rng, tpos: str, dpos: str) -> tuple[int, int]:
    t = rng.choice(TSRC[tpos])
    if dpos == "far":
        d = 2                       # lines 1-2 hold the decoy source or fillers; line 1 is always a filler constant
    elif dpos == "middle":
        d = 11 if t != 11 else rng.choice((10, 12))
    else:
        d = 17 if t != 17 else 16
    return t, d


def _layout(rng, L: int, Ld: int, tpos: str, dpos: str, unit: bool) -> dict | None:
    """Line roles of one program (names, digits and filler pointers come later). None if the combination does not fit."""
    for attempt in range(400):
        t, d = _src_lines(rng, tpos, dpos)
        if unit:
            tl = list(range(t + 1, t + L + 1))
            if tl[-1] > N_LINES - 1 or d in tl or d == t:
                continue
        free = [i for i in range(1, N_LINES) if i not in (t, d)]
        if not unit:
            pool = [i for i in free if i > t]
            if len(pool) < L:
                continue
            tl = sorted(rng.sample(pool, L))
        dpool = [i for i in free if i > d and i not in tl]
        if len(dpool) < Ld:
            continue
        dl = sorted(rng.sample(dpool, Ld))
        if not any(x > d for x in tl):      # a flip twin needs a target link after the decoy source
            continue
        rest = [i for i in free if i not in tl and i not in dl]
        fconst = [rest[0]] + rng.sample(rest[1:], N_CONST - 2 - 1)   # the first filler line is a constant
        fcopy = [i for i in rest if i not in fconst]
        # a filler constant follows the last target copy (and both sources): the nearest binding to the query is never the target
        if max(fconst) < max(t, d, tl[-1]):
            continue
        return {"t": t, "d": d, "tl": tl, "dl": dl, "fconst": sorted(fconst), "fcopy": fcopy}
    return None


def _roles(lay: dict) -> dict[int, str]:
    r = {lay["t"]: "T0", lay["d"]: "D0"}
    for k, l in enumerate(lay["tl"], 1):
        r[l] = f"T{k}"
    for k, l in enumerate(lay["dl"], 1):
        r[l] = f"D{k}"
    for l in lay["fconst"]:
        r[l] = "Fc"
    for l in lay["fcopy"]:
        r[l] = "Fk"
    r[N_LINES] = "P"
    return r


def _template_key(cell: str, lay: dict) -> str:
    r = _roles(lay)
    return cell + ":" + ",".join(r[i] for i in range(1, N_LINES + 1))


def _program(rng, lay: dict, tdig: int, ddig: int) -> dict:
    """Lines 1..23 as {line: {"kind": "const"|"copy", "name", "val": digit | pointee line}}, plus names, print var."""
    for _ in range(100):
        nm = rng.sample(NAMES, N_LINES - 1)
        if nm != sorted(nm):
            break
    prog: dict[int, dict] = {}
    chain_t = [lay["t"]] + lay["tl"]
    chain_d = [lay["d"]] + lay["dl"]
    prog[lay["t"]] = {"kind": "const", "val": tdig}
    prog[lay["d"]] = {"kind": "const", "val": ddig}
    for ch in (chain_t, chain_d):
        for a, b in zip(ch, ch[1:]):
            prog[b] = {"kind": "copy", "val": a}
    for l in lay["fconst"]:
        prog[l] = {"kind": "const", "val": rng.randint(0, 9)}
    fil = sorted(lay["fconst"] + lay["fcopy"])
    for l in lay["fcopy"]:
        prog[l] = {"kind": "copy", "val": rng.choice([x for x in fil if x < l])}
    for l, p in prog.items():
        p["name"] = nm[l - 1]
    return prog


def _text(prog: dict, lay: dict, comments: dict[int, str]) -> str:
    out = []
    for l in range(1, N_LINES):
        if l in comments:
            out.append(comments[l])
            continue
        p = prog[l]
        rhs = str(p["val"]) if p["kind"] == "const" else prog[p["val"]]["name"]
        out.append(f"{p['name']} = {rhs}")
    out.append(f"print({prog[lay['tl'][-1]]['name']})")
    return "\n".join(out)


def _comments(rng, prog: dict, lines: list[int]) -> dict[int, str]:
    """Comment lines with the token count of the line they replace (constant 4 tokens, copy 3, in both tokenizers)."""
    return {l: "# " + " ".join(rng.sample(COMMENT_WORDS, 3 if prog[l]["kind"] == "const" else 2)) for l in lines}


def _flip_options(lay: dict) -> list[int]:
    return [i for i, l in enumerate(lay["tl"]) if l > lay["d"]]


def _f1_group(rng, cell: str, lay: dict, tdig: int, ddig: int, pad: str | None, tag: str):
    """(records' raw parts) for one group: base, and flip and control twins unless pad == 'A'."""
    prog = _program(rng, lay, tdig, ddig)
    comments: dict[int, str] = {}
    if pad:
        needed = {lay["t"], lay["tl"][0]}
        after = [l for l in range(lay["t"] + 1, N_LINES) if l in prog and l not in needed]
        if pad == "B":      # keep the decoy chain, replace only filler lines after the target source
            after = [l for l in after if l in lay["fconst"] or l in lay["fcopy"]]
        comments = _comments(rng, prog, after)
    live = [l for l in range(1, N_LINES) if l not in comments]
    out = [("base", prog, tdig)]
    if pad == "A":
        return out, comments
    # flip: re-point one target link after the decoy source to a decoy-chain variable defined before it
    i = rng.choice(_flip_options(lay))
    line = lay["tl"][i]
    decoy_vars = [x for x in [lay["d"]] + lay["dl"] if x < line]
    flip = {l: dict(p) for l, p in prog.items()}
    flip[line]["val"] = rng.choice(decoy_vars)
    # control: re-point one non-target-chain link to another live earlier variable
    tchain = {lay["t"], *lay["tl"]}
    cands = []
    for l in live:
        if prog[l]["kind"] == "copy" and l not in tchain:
            alts = [x for x in live if x < l and x != prog[l]["val"]]
            if alts:
                cands.append((l, alts))
    if not cands:
        return None, None
    cl, alts = rng.choice(cands)
    ctrl = {l: dict(p) for l, p in prog.items()}
    ctrl[cl]["val"] = rng.choice(alts)
    out += [("flip", flip, ddig), ("control", ctrl, tdig)]
    for role, pr, _ in out[1:]:
        pr["_link"] = line if role == "flip" else cl
    return out, comments


def _f1_cell_split(task: str, cell: str, split: str, n: int, seed: int, owner: dict, cell_index: int) -> list[dict]:
    pad = {F1_PAD_A: "A", F1_PAD_B: "B"}.get(cell)
    unit = cell.endswith("_unit")
    base = cell.replace("_unit", "").replace("_pad_keep", "").replace("_pad", "")
    if pad:
        L, tpos = 1, "far"
    else:
        L, tpos = int(base[1]), base.split("_")[1]
    ngrp = n if pad == "A" else n // 3
    assert pad == "A" or n % 3 == 0, (cell, n)
    rng = _rng(task, seed, cell, split)
    ntpl = max(3, math.ceil(ngrp / (9 if pad == "A" else 3)))
    combos = [(dp, ld) for dp in F1_POS for ld in (1, 2, 3, 4)]
    rng.shuffle(combos)
    layouts, ci = [], 0
    while len(layouts) < ntpl:
        dp, ld = combos[ci % len(combos)]
        ci += 1
        if ci > 4000:
            raise RuntimeError(f"{cell}/{split}: cannot find {ntpl} layouts")
        lay = _layout(rng, L, ld, tpos, dp, unit)
        if lay is None:
            continue
        key = _template_key(cell, lay)
        if owner.get(key, split) != split or any(_template_key(cell, x) == key for x in layouts):
            continue
        owner[key] = split
        lay["dpos"], lay["Ld"], lay["key"] = dp, ld, key
        layouts.append(lay)
    tdigs = _balanced_cycle(rng, ngrp, offset=cell_index)
    offs = _balanced_cycle(rng, ngrp, 9)
    recs = []
    for g in range(ngrp):
        lay = layouts[g % len(layouts)]
        tdig, ddig = tdigs[g], (tdigs[g] + 1 + offs[g]) % 10
        for _ in range(50):
            out, comments = _f1_group(rng, cell, lay, tdig, ddig, pad, split)
            if out:
                break
        else:
            raise RuntimeError("no control twin")
        gid = f"{cell}-{split}-{g:04d}"
        for role, prog, ans in out:
            text = _text(prog, lay, comments)
            chain = [lay["t"]] + lay["tl"]
            meta = {"cell": cell, "L": L, "tpos": tpos, "dpos": lay["dpos"], "Ld": lay["Ld"], "pad": pad, "unit": unit,
                    "twin_group": gid, "twin_role": role, "answer_digit": ans, "target_digit": tdig,
                    "decoy_digit": ddig, "chain_lines": chain, "decoy_src_line": lay["d"],
                    "decoy_chain_lines": lay["dl"], "print_line": N_LINES,
                    "flip_link_line": None if pad == "A" else prog.get("_link") if role == "flip" else None,
                    "control_link_line": prog.get("_link") if role == "control" else None,
                    "src_to_query_lines": N_LINES - chain[0], "final_to_query_lines": N_LINES - chain[-1],
                    "gaps_lines": [b - a for a, b in zip(chain, chain[1:])],
                    "comment_lines": sorted(comments)}
            recs.append({"id": f"{gid}-{role}", "task": task, "prompt": text, "answer": int(ans),
                         "template_id": lay["key"], "rung": cell, "meta": meta})
    # the line of the twins' differing link is the same for the whole group (needed by the token meta of every member)
    if pad != "A":
        byg: dict[str, int] = {}
        for r in recs:
            if r["meta"]["flip_link_line"]:
                byg[r["meta"]["twin_group"]] = r["meta"]["flip_link_line"]
        for r in recs:
            r["meta"]["flip_link_line"] = byg[r["meta"]["twin_group"]]
    return recs


# ================================================================ F2: steps_ops

OPS_TOTAL = 5      # operations in both expressions together, so token counts match across s = 1..4


def _walk(rng, signs: str, target: int | None = None, tries: int = 5000):
    for _ in range(tries):
        v = rng.randint(0, 9)
        ops = [v]
        for s in signs:
            xs = [x for x in range(10) if 0 <= (v + x if s == "+" else v - x) <= 9]
            x = rng.choice(xs)
            v = v + x if s == "+" else v - x
            ops.append(x)
        if target is None or (v == target and ops[0] != v and ops[-1] != v):   # no operand rule can read the answer
            return ops, v
    return None


def _walks(rng, signs: str, target: int, k: int) -> list:
    out = []
    for _ in range(4000):
        w = _walk(rng, signs, target, tries=1)
        if w:
            out.append(w)
            if len(out) == k:
                break
    return out


def _expr_text(ops: list[int], signs: str) -> str:
    s = str(ops[0])
    for sg, x in zip(signs, ops[1:]):
        s += f" {sg} {x}"
    return s


def _f2_templates(s: int) -> list[tuple]:
    m = OPS_TOTAL - s
    qs = [format(i, f"0{s}b").replace("0", "+").replace("1", "-") for i in range(2 ** s)]
    os_ = [format(i, f"0{m}b").replace("0", "+").replace("1", "-") for i in range(2 ** m)]
    return [(lab, qf, a, b) for lab in "AB" for qf in (True, False) for a in qs for b in os_]


def _f2_cell_split(task: str, cell: str, split: str, n: int, seed: int, tpl_share: float, owner: dict) -> list[dict]:
    s = int(cell[1])
    m = OPS_TOTAL - s
    rng = _rng(task, seed, cell, split)
    tpls = _f2_templates(s)
    rng.shuffle(tpls)
    tpls = [t for t in tpls if owner.get(cell + str(t), split) == split]
    ntpl = max(10, round(len(_f2_templates(s)) * tpl_share))
    mine = tpls[:ntpl]
    for t in mine:
        owner[cell + str(t)] = split
    targets = _balanced_cycle(rng, n)
    recs = []
    # rebuild 2026-11-17: answer x sign pattern, answer x first operand and answer x last operand are kept even by greedy
    # rejection sampling (counts per cell and split); the rule "first/last operand is never the answer" is kept as well
    pat, first_c, last_c = Counter(), Counter(), Counter()
    dead: set = set()                # (sign pattern, answer) pairs with no walk under the operand rule
    for i, tg in enumerate(targets):
        cands = None
        for _ in range(len(mine)):
            live = [t for t in mine if (t[2], tg) not in dead]
            picks = [live[rng.randrange(len(live))] for _ in range(8)]
            lab, qfirst, qs, os_ = min(picks, key=lambda t: (pat[(t[2], tg)], rng.random()))
            cands = _walks(rng, qs, tg, 24)
            if cands:
                break
            dead.add((qs, tg))
        if not cands:
            raise RuntimeError("no walk")
        qops, qv = min(cands, key=lambda w: (first_c[(tg, w[0][0])] + last_c[(tg, w[0][-1])], rng.random()))
        pat[(qs, tg)] += 1
        first_c[(tg, qops[0])] += 1
        last_c[(tg, qops[-1])] += 1
        oops, ov = _walk(rng, os_)
        other = "B" if lab == "A" else "A"
        qline, oline = f"{lab} = {_expr_text(qops, qs)}", f"{other} = {_expr_text(oops, os_)}"
        first, second = (qline, oline) if qfirst else (oline, qline)
        # the first/second line order and the label are separate factors: A may sit on either line
        text = f"{first}\n{second}\nWhat is the value of {lab}?"
        tid = f"{cell}|{lab}|{'q1' if qfirst else 'q2'}|{qs}|{os_}"
        meta = {"cell": cell, "s": s, "m": m, "qlabel": lab, "qline": 1 if qfirst else 2, "qsigns": qs, "osigns": os_,
                "qops": qops, "oops": oops, "other_value": ov}
        recs.append({"id": f"{cell}-{split}-{i:04d}", "task": task, "prompt": text, "answer": qv, "template_id": tid,
                     "rung": cell, "meta": meta})
    return recs


# ================================================================ F4: steps_describe

_DATA = json.loads((Path(__file__).with_name("steps_concepts.json")).read_text(encoding="utf-8"))
CONCEPTS: tuple[str, ...] = tuple(_DATA["concepts"])
FEATURES: list[str] = list(_DATA["features"])
FEATURE_TRUE = {f: set(v["true"]) for f, v in _DATA["features"].items()}
PHRASES = {f: {True: v["pos"], False: v["neg"]} for f, v in _DATA["features"].items()}
PHRASE_TOKENS = _DATA["phrase_tokens_qwen"]
N_CLUES = 6
CLUE_TOKEN_TARGET, CLUE_TOKEN_SLACK = 50, 1     # summed Qwen tokens of the six clue sentences
TRAIN_VARIANTS = (0, 1)                         # phrasing 2 appears only in discovery, selection and test


def truth(concept: int, feature: str) -> bool:
    return CONCEPTS[concept] in FEATURE_TRUE[feature]


def settles(answer: int, clues: list[tuple[str, bool]]) -> bool:
    """True if the clues (feature, polarity), all read as true of `answer`, leave only `answer`."""
    left = {c for c in range(len(CONCEPTS)) if all(truth(c, f) == p for f, p in clues)}
    return left == {answer}


def n_needed(answer: int, clues: list[tuple[str, bool]]) -> int:
    """Size of the smallest subset of the clues that leaves exactly one concept (checking every subset); 0 if none."""
    for size in range(1, len(clues) + 1):
        for i in range(1 << len(clues)):
            if bin(i).count("1") == size and settles(answer, [c for j, c in enumerate(clues) if i >> j & 1]):
                return size
    return 0


@lru_cache(maxsize=None)
def _combos() -> np.ndarray:
    import itertools
    return np.array(list(itertools.combinations(range(len(FEATURES)), N_CLUES)), dtype=np.int16)


@lru_cache(maxsize=None)
def _candidates(answer: int) -> dict:
    """All 6-feature sets whose clues (true of `answer`) settle it and have an essential clue, with n_needed,
    the essential positions (bit mask over the sorted feature order) and the size of the strongest single clue's remaining set."""
    nc = len(CONCEPTS)
    ex = np.array([sum(1 << c for c in range(nc) if truth(c, f) != truth(answer, f)) for f in FEATURES], dtype=np.int32)
    comb = _combos()
    M = ex[comb]                                    # (N, 6) concepts each clue excludes
    full = ((1 << nc) - 1) & ~(1 << answer)
    cov = [np.zeros(len(comb), dtype=np.int32) for _ in range(64)]
    for sub in range(1, 64):
        low = (sub & -sub).bit_length() - 1
        cov[sub] = cov[sub & (sub - 1)] | M[:, low]
    settle = [c == full for c in cov]
    nn = np.zeros(len(comb), dtype=np.int8)
    for sub in sorted(range(1, 64), key=lambda x: -bin(x).count("1")):
        nn = np.where(settle[sub], bin(sub).count("1"), nn)
    ess = np.zeros(len(comb), dtype=np.int16)
    for i in range(N_CLUES):
        ess |= ((settle[63] & ~settle[63 ^ (1 << i)]).astype(np.int16) << i)
    cnt = np.zeros_like(M)
    for c in range(nc):
        cnt += ((M >> c) & 1).astype(M.dtype)
    rem = nc - cnt.max(1)
    ok = settle[63] & (ess > 0)
    idx = np.nonzero(ok)[0]
    by_n: dict[int, np.ndarray] = {}
    for n in (1, 2, 3, 4):
        by_n[n] = idx[nn[idx] == n]
    return {"comb": comb, "n": nn, "ess": ess, "rem": rem, "by_n": by_n}


def _polarity(answer: int, f: str) -> bool:
    return truth(answer, f)


def _f4_cell_split(task: str, cell: str, split: str, n: int, seed: int, cell_index: int, owner: dict) -> list[dict]:
    lvl, pos = int(cell[1]), cell.split("_")[1]
    rng = _rng(task, seed, cell, split)
    variants = TRAIN_VARIANTS if split in ("train", "lrsel") else (0, 1, 2)
    concepts = [(i + (5 * cell_index if n % 10 else 0)) % 10 for i in range(n)]
    rng.shuffle(concepts)
    recs = []
    for i, a in enumerate(concepts):
        cand = _candidates(a)
        pool = cand["by_n"][lvl]
        found: list = []
        tries = 0
        while len(found) < 8:        # best of 8: prefer the set whose strongest single clue leaves the most concepts
            tries += 1
            if tries > 20000:
                if found:
                    break
                raise RuntimeError(f"{cell}/{split}: no candidate for {CONCEPTS[a]}")
            row = int(pool[rng.randrange(len(pool))])
            feats = tuple(FEATURES[j] for j in cand["comb"][row])
            key = (a, feats)
            if owner.get(key, split) != split or any(key == f[3] for f in found):
                continue
            var = [rng.choice(variants) for _ in feats]
            tok = sum(PHRASE_TOKENS[PHRASES[f][_polarity(a, f)][v]] for f, v in zip(feats, var))
            if abs(tok - CLUE_TOKEN_TARGET) > CLUE_TOKEN_SLACK:
                continue
            found.append((int(cand["rem"][row]), rng.random(), row, key, var, tok))
        rem, _, row, key, var, tok = max(found)
        owner[key] = split
        feats = list(key[1])
        ess_mask = int(cand["ess"][row])
        ess = [j for j in range(N_CLUES) if ess_mask >> j & 1]
        placed = rng.choice(ess)
        others_e = [j for j in ess if j != placed]
        non = [j for j in range(N_CLUES) if j != placed and j not in others_e]
        rng.shuffle(non)
        slots = [None] * N_CLUES
        end = N_CLUES - 1 if pos == "near" else 0
        slots[end] = placed
        mid = [1, 2, 3, 4]
        rng.shuffle(mid)
        for j in others_e:
            slots[mid.pop()] = j
        free = [k for k in range(N_CLUES) if slots[k] is None]
        for k, j in zip(free, non):
            slots[k] = j
        clues = [(feats[j], _polarity(a, feats[j]), var[j]) for j in slots]
        texts = [PHRASES[f][p][v] for f, p, v in clues]
        prefix = next(k for k in range(1, N_CLUES + 1) if settles(a, [(f, p) for f, p, _ in clues[:k]]))
        nn = int(cand["n"][row])
        assert n_needed(a, [(f, p) for f, p, _ in clues]) == nn
        meta = {"cell": cell, "n_needed": nn, "pos": pos, "concept": CONCEPTS[a],
                "features": [c[0] for c in clues], "polarities": [c[1] for c in clues], "variants": [c[2] for c in clues],
                "essential_positions": sorted(slots.index(j) for j in ess), "placed_essential_position": end,
                "settle_prefix": prefix, "strongest_remaining": rem, "clue_tokens_qwen": tok}
        recs.append({"id": f"{cell}-{split}-{i:04d}", "task": task, "prompt": "\n".join(texts), "answer": a,
                     "template_id": f"{CONCEPTS[a]}|{','.join(sorted(feats))}", "rung": cell, "meta": meta})
    return recs


# ================================================================ dealing

def generate(task: str, seed: int = spec.SPLIT_SEED) -> dict[str, list[dict]]:
    """{split: records} for one steps task. Deterministic. Splits are filled in SPLIT_ORDER; a template claimed by an
    earlier split is never used by a later one."""
    sizes = cell_sizes(task)
    out: dict[str, list[dict]] = {s: [] for s in SPLIT_ORDER}
    owner: dict = {}
    total_rec = {c: sum(sizes[s][c] for s in SPLIT_ORDER) for c in sizes["train"]}
    for s in SPLIT_ORDER:
        for ci, (cell, n) in enumerate(sizes[s].items()):
            if task == "steps_copy":
                out[s] += _f1_cell_split(task, cell, s, n, seed, owner, ci)
            elif task == "steps_ops":
                out[s] += _f2_cell_split(task, cell, s, n, seed, n / total_rec[cell], owner)
            else:
                out[s] += _f4_cell_split(task, cell, s, n, seed, ci, owner)
    return out


# ================================================================ token meta (needs a tokenizer; used by the build)

def line_spans(tok, task: str, prompt: str) -> list[tuple[int, int]]:
    """For each line of `prompt` (0-based), (first token index, index after its last token) in the full chat prompt."""
    full = spec.build_prompt(tok, task, prompt)
    base = full.index(prompt)
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    starts = [o[0] for o in enc["offset_mapping"]]
    spans, pos = [], base
    for line in prompt.split("\n"):
        a, b = pos, pos + len(line)
        i0 = next(i for i, s in enumerate(starts) if s >= a)
        i1 = next((i for i, s in enumerate(starts) if s >= b), len(starts))
        spans.append((i0, i1))
        pos = b + 1
    return spans


def annotate_f1_tokens(tok, rec: dict) -> None:
    """Add Qwen-token distances to meta: query index is the first token of the print line."""
    m = rec["meta"]
    sp = line_spans(tok, rec["task"], rec["prompt"])
    q = sp[N_LINES - 1][0]
    chain = m["chain_lines"]
    first = [sp[l - 1][0] for l in chain]
    m["query_tok"] = q
    m["src_to_query_tok"] = q - first[0]
    m["final_to_query_tok"] = q - first[-1]
    m["gaps_tok"] = [b - a for a, b in zip(first, first[1:])]
    m["chain_tok_index"] = first
    m["final_binding_tok"] = sp[chain[-1] - 1][1] - 1          # last token of the printed variable's final assignment
    fl = m["flip_link_line"]
    m["flip_after_tok"] = sp[fl - 1][1] if fl else None        # first token after the first link that differs in the flip twin
    m["prompt_tokens"] = len(tok(spec.build_prompt(tok, rec["task"], rec["prompt"]), add_special_tokens=False)["input_ids"])
