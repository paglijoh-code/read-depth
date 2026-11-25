#!/usr/bin/env python3
"""Robustness E2 revision R1, gate round 2 (the last round): build the five check tasks (26-30) and five binary doing tasks (31-35) and run the Mac gate
(docs/ATLAS-robustness.md, "E2 revision R1").

  uv run python scripts/atlas_e2_gate.py --build            # write the 10 x 5 split files and manifest4.json (never touches tasks 1-25)
  uv run python scripts/atlas_e2_gate.py                    # the gate: three rivals per task on the test split (+ heuristics)
  uv run python scripts/atlas_e2_gate.py --probe [--smoke]  # unmodified Qwen3-1.7B probe L_on per task (reported, not gating)

Rivals, on the test split (fit on train + lrsel, C / depth chosen on lrsel):
  1. BoW surface rival on the whole prompt (the baseline's TF-IDF + logistic regression);
  2. answer-only rival (check tasks: logistic regression on the proposed answer's tokens alone) or set-only rival (bin tasks:
     the same on the listed set A, which is one constant string per task, so it can only guess the commoner answer);
  3. structural rival: a gradient-boosted tree on position features (the line or rule index where the proposed value or a set
     member appears, its last-assignment position, counts of loops, rules, flags).
A task is clean if all three are <= chance + 0.05; a pair is clean if its check and bin tasks both are. At least 4 clean pairs ->
"gate passes", else "stop" (no pod). Also reported, no bar: a hand-written heuristic per near-miss type for each check task
(including the first design's leak), and the unmodified-1.7B probe `L_on`.
Writes var/results/atlas3/e2_gate_r2.json and e2_gate_r2.md (e2_gate.* and e2_gate_r1.* are kept). Round 2 rebuilt arith_check,
code_output_check, arith_bin and code_literal_bin; rule_* and hard_* rows are copied from r1 (same bytes, marked), the other tasks are
re-gated (their files are unchanged, so their rows come out the same) and keep their r1 L_on.
--probe adds `L_on` per task to the json (--smoke: first 64 discovery records of the first --tasks entry, e2_gate_smoke.json only).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from surgery.atlas import spec, tasks  # noqa: E402

CHECK_TASKS = tuple(t for f in tasks.E2_FAMILIES for t in tasks.FAMILY_TASKS[f])
BIN_TASKS = tuple(tasks.BIN_OF_CHECK[t] for t in CHECK_TASKS)
ALL_TASKS = CHECK_TASKS + BIN_TASKS
OUT = Path("var/results/atlas3")
NAME = "e2_gate_r2"
R1 = "e2_gate_r1"
COPIED = ("rule_check", "hard_check", "rule_bin", "hard_bin")      # data byte-identical to r1: rows copied, not recomputed
REBUILT = ("arith_check", "code_output_check", "arith_bin", "code_literal_bin")   # round 2 data: their r1 activation stores are stale
MARGIN, MIN_CLEAN = 0.05, 4
CS = [0.1, 1.0, 10.0, 100.0]
DEPTHS = [2, 3, 4]
PROBE_MODEL = "Qwen/Qwen3-1.7B"
PROPOSAL = tasks.CHECK_PROPOSAL


def store_suffix(task: str) -> str:
    """The first design's activation stores hold other prompts; round 2's rebuilt tasks get fresh ones (the others reuse r1's)."""
    return "_r2" if task in REBUILT else "_r1"


def is_bin(task: str) -> bool:
    return task in tasks.BIN_DO


def pair_of(task: str) -> tuple[str, str]:
    c = task if task in tasks.CHECK_DO else next(k for k, v in tasks.BIN_OF_CHECK.items() if v == task)
    return c, tasks.BIN_OF_CHECK[c]


def proposed_text(rec: dict) -> str:
    """The proposed answer as shown in the prompt (the text after the last 'Proposed answer: ')."""
    return rec["prompt"].rsplit(PROPOSAL, 1)[1]


def set_text(rec: dict) -> str:
    """The listed set of a bin record (the text after the last 'Is the answer one of: ')."""
    return rec["prompt"].rsplit(tasks.BIN_QUESTION, 1)[1]


def do_part(rec: dict) -> str:
    """The doing task's own input (what comes before the proposal or the set)."""
    return rec["prompt"].rsplit(tasks.BIN_QUESTION if "bin" in rec["task"] else PROPOSAL, 1)[0]


# ---------------------------------------------------------------- structural features

_RULE = re.compile(r"^Rule (\d+): .*then (\w+) \[end\]$", re.M)
_DIS = re.compile(r"^Disabled (\d+): .*then (\w+) \[end\]$", re.M)


def rule_lists(text: str) -> tuple[list[str], list[str], int]:
    """(actions of the active rules in order, actions of the disabled rules in order, number of flag definitions)."""
    return ([m.group(2) for m in _RULE.finditer(text)], [m.group(2) for m in _DIS.finditer(text)],
            len(re.findall(r"^\w+ holds if", text, re.M)))


def _first(xs, pred) -> int:
    return next((i + 1 for i, x in enumerate(xs) if pred(x)), 0)


def _last(xs, pred) -> int:
    return max((i + 1 for i, x in enumerate(xs) if pred(x)), default=0)


def _members(rec: dict) -> set[str]:
    """The values in play: the proposal (check) or the listed set (bin), as option strings."""
    return {x.strip() for x in set_text(rec).rstrip("?").split(",")} if is_bin(rec["task"]) else {proposed_text(rec)}


def rule_features(rec: dict) -> list[float]:
    text = do_part(rec)
    act, dis, nflags = rule_lists(text)
    m = _members(rec)
    in_a = [1 if a in m else 0 for a in act] + [-1] * (12 - len(act))
    return [_first(act, lambda a: a in m), _last(act, lambda a: a in m), sum(a in m for a in act),
            _first(dis, lambda a: a in m), sum(a in m for a in dis), len(act), len(dis), nflags, *in_a[:12]]


_NUM = re.compile(r"(?<![\w.])\d+(?![\w.])")


def arith_features(rec: dict) -> list[float]:
    text = do_part(rec)
    nums = _NUM.findall(text)
    m = _members(rec)
    ops = re.findall(r"[-+*]", text.split(":")[-1] if ":" in text else text)
    in_a = [1 if x in m else 0 for x in nums[:3]] + [-1] * (3 - len(nums[:3]))
    vals = sorted(int(x) for x in m)
    return [_first(nums, lambda x: x in m), _last(nums, lambda x: x in m), sum(x in m for x in nums), len(nums), len(ops),
            len(text), *in_a, vals[0] if len(vals) == 1 else sum(vals) / len(vals), vals[0] % 2 if len(vals) == 1 else -1]


def code_features(rec: dict) -> list[float]:
    code = do_part(rec)
    lines = code.split("\n")
    m = _members(rec)
    ints = {int(x) for x in m if x.isdigit()}
    has = [1 if any(t in m for t in re.findall(r"(?<![\w.])\d+(?![\w.])", ln)) else 0 for ln in lines]
    ev = tasks._assign_events(code)
    last_ln = max((ln for ln, v in ev if v in ints), default=0)
    first_ln = min((ln for ln, v in ev if v in ints), default=0)
    n_ev = sum(1 for _, v in ev if v in ints)
    final = ev[-1][1] if ev else -1
    loops = sum(1 for ln in lines if re.match(r"\s*(for|while)\b", ln))
    ifs = sum(1 for ln in lines if re.match(r"\s*if\b", ln))
    asg = [1 if has[i] and re.search(r"=", ln) else 0 for i, ln in enumerate(lines)]
    return [_first(has, lambda x: x), _last(has, lambda x: x), sum(has), _last(asg, lambda x: x), last_ln, first_ln, n_ev,
            int(final in ints), len(ev), len(lines), loops, ifs, *(has + [-1] * 10)[:10]]


def features(rec: dict) -> list[float]:
    return {"rule": rule_features, "hard": rule_features, "arith": arith_features, "code_output": code_features,
            "code_literal": code_features}[(tasks.BIN_DO | tasks.CHECK_DO)[rec["task"]]](rec)


# ---------------------------------------------------------------- heuristics (reported, no bar)

def _eval(expr: str) -> int | None:
    body = re.sub(r"^[^(\d-]*", "", expr.rstrip("?")).strip()
    if not re.fullmatch(r"[\d+\-*() ]+", body):
        return None
    try:
        return int(eval(body, {"__builtins__": {}}))   # noqa: S307 - digits, + - * and parentheses only
    except Exception:
        return None


def heuristics(task: str, recs: list[dict]) -> dict[str, float]:
    """Accuracy of hand-written yes/no rules on `recs` (answer 0 = yes). Never fitted."""
    def acc(pred) -> float:
        return float(np.mean([(0 if pred(r) else 1) == r["answer"] for r in recs]))

    out: dict[str, float] = {}
    if task == "rule_check":
        out["proposal appears only under Disabled rules -> no"] = acc(lambda r: not (
            proposed_text(r) in rule_lists(do_part(r))[1] and proposed_text(r) not in rule_lists(do_part(r))[0]))
        out["proposal equals Rule 1's action -> yes"] = acc(lambda r: (rule_lists(do_part(r))[0] or [""])[0] == proposed_text(r))
        out["proposal appears in the active rules -> yes"] = acc(lambda r: proposed_text(r) in rule_lists(do_part(r))[0])
    elif task == "hard_check":
        out["proposal equals Rule 1's action -> yes"] = acc(lambda r: (rule_lists(do_part(r))[0] or [""])[0] == proposed_text(r))
        out["proposal appears only under Disabled rules -> no"] = acc(lambda r: not (
            proposed_text(r) in rule_lists(do_part(r))[1] and proposed_text(r) not in rule_lists(do_part(r))[0]))
    elif task == "arith_check":
        out["proposal has the value's parity -> yes"] = acc(lambda r: _eval(do_part(r)) is not None and
                                                           int(proposed_text(r)) % 2 == _eval(do_part(r)) % 2)
        out["proposal is a number in the expression -> yes"] = acc(lambda r: proposed_text(r) in _NUM.findall(do_part(r)))
    else:   # code_output_check, code_literal_check
        def last_assigned(r) -> bool:
            ev = tasks._assign_events(do_part(r))
            return bool(ev) and str(ev[-1][1]) == proposed_text(r)

        out["last-assigned value -> yes"] = acc(last_assigned)
        out["proposal is a constant of the program -> yes"] = acc(
            lambda r: proposed_text(r) in re.findall(r"^\w+ = (\d)$", do_part(r), re.M))
    return out


# ---------------------------------------------------------------- scoring

def _answer_model(C: float):
    return make_pipeline(TfidfVectorizer(token_pattern=r"\S+", lowercase=False), LogisticRegression(C=C, max_iter=2000))


def _tree(depth: float):
    return GradientBoostingClassifier(n_estimators=100, max_depth=int(depth), random_state=0)


def _score(make, X_of, task: str, data: Path, grid=CS) -> dict:
    """Fit on train with each hyperparameter, pick it on lrsel (ties: smaller), refit on train+lrsel, score on test."""
    tr, lr, te = (spec.read_records(task, s, data) for s in ("train", "lrsel", "test"))
    y = lambda rs: np.array([r["answer"] for r in rs])  # noqa: E731
    scores = {C: float(np.mean(make(C).fit(X_of(tr), y(tr)).predict(X_of(lr)) == y(lr))) for C in grid}
    best = max(grid, key=lambda c: (scores[c], -c))
    m = make(best).fit(X_of(tr + lr), y(tr + lr))
    return {"C": best, "acc": float(np.mean(m.predict(X_of(te)) == y(te)))}


def gate_task(task: str, data: Path) -> dict:
    import atlas_baseline
    whole = _score(atlas_baseline.model, lambda rs: [r["prompt"] for r in rs], task, data)
    side = set_text if is_bin(task) else proposed_text
    only = _score(_answer_model, lambda rs: [side(r) for r in rs], task, data)
    struct = _score(_tree, lambda rs: np.array([features(r) for r in rs], dtype=float), task, data, DEPTHS)
    te = spec.read_records(task, "test", data)
    chance = spec.chance([r["answer"] for r in te])
    limit = chance + MARGIN
    row = {"task": task, "number": spec.TASKS[task].number, "kind": "bin" if is_bin(task) else "check", "n_test": len(te),
           "chance": chance, "bow_whole": whole["acc"], "bow_C": whole["C"], "side_only": only["acc"], "side_C": only["C"],
           "structural": struct["acc"], "structural_depth": struct["C"], "items": kept_dropped(task, data),
           "clean": all(x <= limit + 1e-9 for x in (whole["acc"], only["acc"], struct["acc"]))}
    if not is_bin(task):
        row["heuristics"] = heuristics(task, te)
    return row


def verdict(rows: list[dict]) -> dict:
    """Pairs: check task + bin task; a pair is clean if both rows are."""
    by = {r["task"]: r for r in rows}
    pairs = []
    for c in CHECK_TASKS:
        b = tasks.BIN_OF_CHECK[c]
        if c in by and b in by:
            pairs.append({"check": c, "bin": b, "clean": bool(by[c]["clean"] and by[b]["clean"])})
    n = sum(p["clean"] for p in pairs)
    return {"n_clean": n, "n_pairs": len(pairs), "pairs": pairs, "verdict": "gate passes" if n >= MIN_CLEAN else "stop",
            "rule": f"clean task = BoW (whole prompt), answer-only (check) / set-only (bin) and the structural boosted tree all <= "
                    f"chance + {MARGIN} on the test split; clean pair = its check and bin tasks both clean; at least {MIN_CLEAN} of "
                    f"{len(pairs)} clean pairs -> gate passes, else stop"}


def kept_dropped(task: str, data: Path) -> dict:
    """Records kept per split, and items dropped (no near miss for check tasks; balancing for bin tasks), from manifest4.json."""
    e = json.loads((data / "manifest4.json").read_text())["tasks"][task]
    return {"kept": {s: v["n"] for s, v in e["splits"].items()}, "dropped": e["pool"].get("dropped_items", {})}


def render(res: dict) -> str:
    lines = ["# E2 gate, revision R1, round 2 (last round): five check + five binary tasks, Mac rivals ($0)", "",
             f"Test split, n = {min(r['n_test'] for r in res['rows'])} to {max(r['n_test'] for r in res['rows'])} per task. {res['rule']}.", "",
             "| task | chance | BoW | answer-only / set-only | structural (boosted tree) | clean | L_on (1.7B, unmodified) |",
             "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        lon = r["L_on"] if r.get("L_on") is not None else "not run"
        mark = " (copied from r1)" if r.get("copied_from_r1") else ""
        lines.append(f"| {r['number']} {r['task']}{mark} | {r['chance']:.3f} | {r['bow_whole']:.3f} | {r['side_only']:.3f} | "
                     f"{r['structural']:.3f} | {'yes' if r['clean'] else 'NO'} | {lon} |")
    lines += ["", "Hand-written heuristics on the check tasks (test accuracy; reported, no bar):", "", "| task | heuristic | accuracy |",
              "|---|---|---|"]
    for r in res["rows"]:
        for h, a in (r.get("heuristics") or {}).items():
            lines.append(f"| {r['task']} | {h} | {a:.3f} |")
    lines += ["", "Pairs:", "", "| check | bin | clean |", "|---|---|---|"]
    lines += [f"| {p['check']} | {p['bin']} | {'yes' if p['clean'] else 'NO'} |" for p in res.get("pairs", [])]
    lines += ["", "Records kept / items dropped per split (check: no legal near miss; bin: dropped to balance yes/no):", "",
              "| task | " + " | ".join(spec.SPLITS) + " |", "|---|" + "---|" * len(spec.SPLITS)]
    for r in res["rows"]:
        kd = r.get("items", {})
        lines.append(f"| {r['task']} | " + " | ".join(f"{kd.get('kept', {}).get(s, '?')} / {kd.get('dropped', {}).get(s, '?')}"
                                                     for s in spec.SPLITS) + " |")
    lines += ["", f"**Verdict: {res['verdict']}** ({res['n_clean']} of {res['n_pairs']} pairs clean)."]
    if res["verdict"] == "stop":
        lines.append("Fewer than 4 clean pairs: no pod run. The finding is that a deep check task is hard to build without a shortcut.")
    return "\n".join(lines) + "\n"


def write(res: dict, out: Path, name: str = NAME) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / f"{name}.md").write_text(render(res), encoding="utf-8")


def build(data: Path) -> None:
    import hashlib

    import atlas_build
    man = atlas_build.build(data, list(tasks.E2_FAMILIES) + list(tasks.BIN_FAMILIES))
    (data / "manifest4.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for task, e in sorted(man["tasks"].items(), key=lambda kv: kv[1]["number"]):
        print(f"{e['number']:>2} {task:<20} n={sum(v['n'] for v in e['splits'].values())} "
              f"max_dev={max(v['max_dev_from_uniform'] for v in e['splits'].values())} pool={e['pool']['pool_items']} "
              f"short={sum(e['pool']['quota_shortfall'].values())}")
    assert all(hashlib.sha256((data / f).read_bytes()).hexdigest() == h for f, h in man["files"].items())
    print("wrote", data / "manifest4.json")


def probe(task: str, data: Path, smoke: bool) -> dict:
    """Unmodified-model probe curve on the discovery split, as scripts/atlas_map.py (same store and `_probe`)."""
    from atlas_map import ACTS, _probe

    from surgery.atlas import measure
    from surgery.eval import load_model
    records = spec.read_records(task, "discovery", data)
    if smoke:
        records = records[:64]
    store = ACTS / PROBE_MODEL.split("/")[-1] / f"{task}_discovery{store_suffix(task)}{'_smoke' if smoke else ''}"
    if not (store / "meta.json").exists():
        model, tok = load_model(PROBE_MODEL, dtype="fp32")
        measure.build_store(model, tok, task, records, store, batch_size=8)
        del model
    meta = json.loads((store / "meta.json").read_text())
    if meta["ids"] != [r["id"] for r in records]:
        raise ValueError(f"store at {store} does not match the discovery records")
    n = len(meta["answers"])
    p = _probe(np.load(store / "resid.npy", mmap_mode="r"), meta["answers"], meta["template_ids"], np.arange(n),
               20 if smoke else 500, seed=0)
    return {"L_on": p["L_on"] if p else None, "peak": p["peak"] if p else None,
            "peak_layer": p["peak_layer"] if p else None, "n": n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=spec.DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--build", action="store_true", help="build the check- and bin-task data files and manifest4.json, then stop")
    ap.add_argument("--probe", action="store_true", help="add the unmodified 1.7B probe L_on (reported, not gating)")
    ap.add_argument("--smoke", action="store_true", help="with --probe: 64 records, one task, separate output")
    ap.add_argument("--tasks", nargs="*", default=list(ALL_TASKS), choices=ALL_TASKS)
    a = ap.parse_args(argv)
    if a.build:
        build(a.data)
        return 0
    path = a.out / f"{NAME}.json"
    if a.probe:
        if a.smoke:
            t = a.tasks[0]
            res = {"task": t, "smoke": True, **probe(t, a.data, True)}
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / "e2_gate_smoke.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(res)
            return 0
        res = json.loads(path.read_text())
        by = {r["task"]: r for r in res["rows"]}
        for t in a.tasks:
            by[t].update(probe(t, a.data, False))
            write(res, a.out)
            print(t, by[t]["L_on"], flush=True)
        return 0
    r1 = {r["task"]: r for r in json.loads((a.out / f"{R1}.json").read_text())["rows"]}
    rows = []
    for t in ALL_TASKS:
        if t in COPIED:
            rows.append({**r1[t], "copied_from_r1": True})
        else:
            rows.append(gate_task(t, a.data))
            if t not in REBUILT:                       # same data as r1, so the same probe
                rows[-1].update({k: r1[t][k] for k in ("L_on", "peak", "peak_layer") if k in r1[t]})
        r = rows[-1]
        print(f"{r['number']} {t:<20} chance={r['chance']:.3f} bow={r['bow_whole']:.3f} side_only={r['side_only']:.3f} "
              f"structural={r['structural']:.3f} {'clean' if r['clean'] else 'LEAKY'} {r.get('heuristics', '')}", flush=True)
    res = {"rows": rows, **verdict(rows)}
    if path.exists():                                  # keep probe numbers from an earlier --probe run
        old = {r["task"]: r for r in json.loads(path.read_text())["rows"]}
        for r in rows:
            for k in ("L_on", "peak", "peak_layer"):
                if k in old.get(r["task"], {}):
                    r[k] = old[r["task"]][k]
    write(res, a.out)
    print(res["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
