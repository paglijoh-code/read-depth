#!/usr/bin/env python3
"""POST HOC shortcut check (docs/ATLAS-shortcut.md, bars fixed 2026-11-20 before scoring).

Reads saved first-token predictions on the steps_copy test split; runs no model. Writes
var/results/steps3/shortcut/shortcut.json and tables.md.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from surgery.atlas import spec  # noqa: E402

RES = ROOT / "var/results/steps3/stage1"
OUT = ROOT / "var/results/steps3/shortcut"
RULES = ("W1", "W2", "ES", "DV", "TS", "LL")
SHARE, GAP, MIN_N, SHUFFLES, SEED = 0.60, 0.25, 20, 1000, 20261120
_CONST = re.compile(r"\s*\w+\s*=\s*(\d)\s*")


def rule_digits(rec: dict) -> dict:
    lines = rec["prompt"].split("\n")
    consts = {}
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("print("):
            break
        if m := _CONST.fullmatch(line):
            consts[i] = int(m.group(1))
    m = rec["meta"]
    t, d = m["chain_lines"][0], m["decoy_src_line"]
    return {"W1": consts.get(1), "W2": consts.get(2),
            "ES": m["target_digit"] if t < d else m["decoy_digit"],
            "DV": m["decoy_digit"], "TS": m["target_digit"],
            "LL": consts[max(consts)] if consts else None}


def score_rule(rows, rule, unique, rng):
    """rows: (cell, pred, answer, digits, decoy_on_line2). Wrong answers only."""
    kept = []
    for c, p, a, dg, l2 in rows:
        d = dg[rule]
        if d is None or d == a:
            continue
        if unique and any(dg[o] == d for o in RULES if o != rule):
            continue
        kept.append((c, p, d))
    if not kept:
        return {"n": 0, "share": None, "null": None, "follows": None}
    share = float(np.mean([p == d for _, p, d in kept]))
    hits = np.zeros(SHUFFLES)
    for c in sorted({k[0] for k in kept}):
        p = np.asarray([x for cc, x, _ in kept if cc == c])
        d = np.asarray([x for cc, _, x in kept if cc == c])
        perm = np.argsort(rng.random((SHUFFLES, len(d))), axis=1)
        hits += (d[perm] == p[None, :]).sum(axis=1)
    null = float(hits.mean() / len(kept))
    ok = len(kept) >= MIN_N and share >= SHARE - 1e-9 and share - null >= GAP - 1e-9
    return {"n": len(kept), "share": share, "null": null, "follows": bool(ok) if len(kept) >= MIN_N else None}


def twin_same(recs, items):
    g = defaultdict(dict)
    for r in recs:
        if r["meta"]["twin_role"] in ("base", "flip"):
            g[r["meta"]["twin_group"]][r["meta"]["twin_role"]] = r["id"]
    pairs = [(v["base"], v["flip"]) for v in g.values() if "base" in v and "flip" in v]
    return float(np.mean([items[a]["pred"] == items[b]["pred"] for a, b in pairs]))


def classify(acc, ts):
    if acc <= 0.60 and ts >= 0.80:
        return "stuck"
    if acc >= 0.85 and ts <= 0.20:
        return "learned"
    return "between"


def sources():
    out = []
    for w in ("early", "mid"):
        for s in (0, 2, 3):
            out.append((f"ctt300 {w}", s, RES / f"ctt_steps_copy_{w}_s{s}.json", "score"))
            out.append((f"ctt600 {w}", s, RES / f"ctt600_steps_copy_{w}_s{s}.json", "score"))
            out.append((f"tc {w}", s, RES / f"tc_steps_copy_{w}_s{s}.json", "after_repair"))
    for arm, w in (("A", "early"), ("B", "early"), ("C", "mid"), ("D", "early"), ("F", "pre")):
        for s in (0, 2, 3, 4, 5):
            p = RES / f"cf_{arm}_steps_copy_{w}_s{s}.json"
            if not p.exists() and arm == "D":
                p = RES / f"tc_steps_copy_early_s{s}.json"   # D seeds 0, 2, 3 reuse Stage 1 tc early
                out.append((f"conf {arm} {w}", s, p, "after_repair"))
                continue
            out.append((f"conf {arm} {w}", s, p, "score"))
    for s in (0, 2, 3):
        out.append(("uncut", s, RES / f"uncut_steps_copy_s{s}.json", "score"))
    for s in (4, 5):
        out.append(("uncut", s, RES / f"cf_uncut_steps_copy_s{s}.json", "score"))
    return out


def main():
    recs = spec.read_records("steps_copy", "test")
    byid = {r["id"]: r for r in recs}
    digs = {r["id"]: rule_digits(r) for r in recs}
    models = []
    for name, seed, path, key in sources():
        if not path.exists():
            continue
        items = json.loads(path.read_text())[key]["items"]
        acc = float(np.mean([items[i]["ok"] for i in items]))
        ts = twin_same(recs, items)
        rows = [(byid[i]["rung"], int(v["pred"]), byid[i]["answer"], digs[i], byid[i]["meta"]["decoy_src_line"] == 2)
                for i, v in items.items() if not v["ok"]]
        rng = np.random.default_rng(SEED)
        res = {"raw": {r: score_rule(rows, r, False, rng) for r in RULES},
               "unique": {r: score_rule(rows, r, True, rng) for r in RULES},
               "DV_line2": score_rule([x for x in rows if x[4]], "DV", False, rng),
               "DV_not_line2": score_rule([x for x in rows if not x[4]], "DV", False, rng)}
        roles = {}
        for role in ("base", "flip", "control"):
            ids = [i for i in items if byid[i]["meta"]["twin_role"] == role]
            p = [items[i]["pred"] for i in ids]
            roles[role] = {"n": len(ids),
                           "target": float(np.mean([x == byid[i]["meta"]["target_digit"] for x, i in zip(p, ids)])),
                           "decoy": float(np.mean([x == byid[i]["meta"]["decoy_digit"] for x, i in zip(p, ids)]))}
        models.append({"model": name, "seed": seed, "file": str(path.relative_to(ROOT)), "acc": acc, "twin_same": ts,
                       "class": classify(acc, ts), "n_wrong": len(rows), "rules": res, "roles": roles})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shortcut.json").write_text(json.dumps({"post_hoc": True, "doc": "docs/ATLAS-shortcut.md", "models": models}, indent=1))
    f = lambda x: "—" if x is None else f"{x:.2f}"
    L = ["| model | seed | acc | twin-same | class | wrong | " + " | ".join(f"{r} share/null (n)" for r in RULES) + " |",
         "|" + "---|" * (6 + len(RULES))]
    for kind in ("raw", "unique"):
        L2 = [f"\n### {kind}\n"] + L[:]
        for m in models:
            cells = [f"{f(m['rules'][kind][r]['share'])}/{f(m['rules'][kind][r]['null'])} ({m['rules'][kind][r]['n']})"
                     + (" ✓" if m["rules"][kind][r]["follows"] else "") for r in RULES]
            L2.append(f"| {m['model']} | {m['seed']} | {m['acc']:.3f} | {m['twin_same']:.3f} | {m['class']} | {m['n_wrong']} | "
                      + " | ".join(cells) + " |")
        L2 and (OUT / f"tables_{kind}.md").write_text("\n".join(L2) + "\n")
    T = ["| model | seed | class | DV on decoy-line-2 items | DV elsewhere | base: tgt/decoy | flip: tgt/decoy | control: tgt/decoy |",
         "|---|---|---|---|---|---|---|---|"]
    for m in models:
        a, b, ro = m["rules"]["DV_line2"], m["rules"]["DV_not_line2"], m["roles"]
        T.append(f"| {m['model']} | {m['seed']} | {m['class']} | {f(a['share'])}/{f(a['null'])} ({a['n']}) | "
                 f"{f(b['share'])}/{f(b['null'])} ({b['n']}) | "
                 + " | ".join(f"{ro[k]['target']:.2f}/{ro[k]['decoy']:.2f}" for k in ("base", "flip", "control")) + " |")
    (OUT / "tables_roles.md").write_text("\n".join(T) + "\n")
    explore(models, byid)
    print((OUT / "tables_unique.md").read_text()); print("\n".join(T))
    print((OUT / "tables_explore.md").read_text())


def explore(models, byid):
    """Exploratory, no bar (added after scoring): which source a stuck model picks, and flip accuracy by link position."""
    E = ["Exploratory (no bar). Twin items whose answer is one of the two source digits: share that is the target digit, "
         "split by whether the target source comes before the decoy source. Then flip items: share right (decoy digit), "
         "by the number of target links after the re-pointed one (0 = the printed variable's own line).", "",
         "| model | seed | P(target) target earlier | P(target) target later | answer is neither source | flip right by links after: 0 / 1 / 2 / 3 |",
         "|---|---|---|---|---|---|"]
    for m in models:
        if m["class"] != "stuck":
            continue
        doc = json.loads((ROOT / m["file"]).read_text())
        items = doc["after_repair" if "after_repair" in doc else "score"]["items"]
        early, late, other, tw, flip = [], [], 0, 0, defaultdict(list)
        for i, v in items.items():
            me = byid[i]["meta"]
            if me["twin_role"] not in ("base", "flip", "control"):
                continue
            tw += 1
            p = v["pred"]
            if me["twin_role"] == "flip":
                k = len(me["chain_lines"]) - 1 - me["chain_lines"].index(me["flip_link_line"])
                flip[k].append(p == me["decoy_digit"])
            if p not in (me["target_digit"], me["decoy_digit"]):
                other += 1
                continue
            (early if me["chain_lines"][0] < me["decoy_src_line"] else late).append(p == me["target_digit"])
        E.append(f"| {m['model']} | {m['seed']} | {np.mean(early):.2f} ({len(early)}) | {np.mean(late):.2f} ({len(late)}) | "
                 f"{other / tw:.2f} | " + " / ".join(f"{np.mean(flip[k]):.2f}" for k in sorted(flip)) + " |")
    (OUT / "tables_explore.md").write_text("\n".join(E) + "\n")


if __name__ == "__main__":
    main()
