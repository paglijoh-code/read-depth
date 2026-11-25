"""Steps study v3, Stage 1 (docs/ATLAS-steps3.md; EXPLORATORY, 3 seeds): what a cut-and-repair can and cannot rebuild.

Qwen3-1.7B, the v2 tasks steps_copy and steps_ops. Results go to var/results/steps3/stage1/, checkpoints to
var/ckpt/steps/steps3_<arm>_<window>/. Every stage resumes (a finished result file or heal row is skipped) and every
JSON carries "exploratory": true.

Training objective: every repair and every uncut model is trained on the WHOLE ANSWER plus end-of-turn (token
cross-entropy, checkpoint chosen by lrsel exact match), the v2 steps recipe (`atlas3_cuts.run_whole_heal` /
`heal.train_whole`). Only the `ftonly` control trains on the first-token objective (`atlas_cuts.run_heal`). R is
reported on first-token accuracy (primary) and on exact match (beside it).

    ATLAS3_DEVICE=mps uv run python scripts/steps3_stage1.py --stage STAGE [--task T] [--window W] [--seed S] [--arm A]

Stages: probe (fill in probes skipped under STEPS3_NOPROBE=1), uncut (k=0 task-trained row + score), ctt (1a), tc (1b, windows plus --window none), arms (1c, ops, needs
--window), ftonly (1d, copy, window late, first-token-only objective), v2score (score the v2 k12 copy models), errors (1e, no model), floor (1f, no
model), table (tables.md and verdicts.json, no model). Windows are k=4 cuts of original blocks; "v2k12" is the first 12
of the shared cut order. The stages that load a model (uncut, ctt, tc, arms, ftonly, v2score) are not exercised by the tests.
"""
from __future__ import annotations

import argparse
import fcntl
import fnmatch
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import atlas3_cuts as C  # noqa: E402
import atlas_cuts as ac  # noqa: E402
import steps_cutprobe as CP  # noqa: E402
import steps_run as S  # noqa: E402
from surgery import adapters  # noqa: E402
from surgery.atlas import heal, measure, spec  # noqa: E402
from surgery.eval import _free  # noqa: E402

OUT = Path("var/results/steps3/stage1")
ORDER_JSON = Path("var/results/atlas3/1.7B/order.json")

WINDOWS = {"early": (7, 8, 9, 10), "mid": (11, 12, 13, 14), "late": (16, 17, 18, 19), "last": (22, 23, 24, 25),
           "spread": (8, 13, 18, 23)}
K = 4
V2K = "v2k12"
NONE = "none"
SEEDS = {S.COPY: (0, 2, 3), S.OPS: (0, 1, 2)}
USABLE_ACC = 0.90
GEN_TOKENS = 4
STD_STEPS = 300
QV = ("q_proj", "v_proj")
FETCH_PATTERN = "L1_*_pad*"                      # pure-fetch (single-candidate) copy cells: L1_far_pad, L1_far_pad_keep
OPS_CELLS = ("s2", "s3", "s4")
ALLLIN = C.D2_ARMS["alllin"]
ARMS = {                                         # name: (modules, rank, step multiple of 300)
    "mlp": (ALLLIN.modules, ALLLIN.rank, 1),
    "long": (C.D2_ARMS["long"].modules, C.D2_ARMS["long"].rank, C.D2_ARMS["long"].mult),
    "both": (ALLLIN.modules, ALLLIN.rank, C.D2_ARMS["long"].mult),
}
HI, LO = 0.80, 0.50                              # R bars
RULES = ("first_source", "last_assignment", "decoy_end")
GROUPS = ("far", "middle", "near")
SHUFFLES, ERR_SEED = 1000, 20261118
E1_SHARE, E1_GAP = 0.50, 0.25
EPS = 1e-9
STAGES = ["uncut", "ctt", "tc", "arms", "ftonly", "v2score", "probe", "errors", "floor", "table", "ctt600"]
MATCHED_STEPS = 2 * STD_STEPS                     # ctt600 (post hoc, 2026-11-19): cut-then-train at the step count train-then-cut gets (300 train + 300 repair)
NOPROBE = os.environ.get("STEPS3_NOPROBE") == "1"   # pod: heal and score only; --stage probe fills the probes in later
SHORT = {"copy": S.COPY, "ops": S.OPS}
CTT600_CELLS = {S.COPY: ("early", "mid"), S.OPS: ("early", "late", "spread")}  # where 1a and 1b differ by > 0.2


# ---------------------------------------------------------------- windows, seeds, learning rate (pure)

def window_k(window: str) -> int:
    if window in WINDOWS:
        return K
    return {V2K: S.KCUT, NONE: 0}[window]


def window_order(shared, window: str, *, windows=None) -> list[int]:
    """The window's blocks first, then the rest of the shared order; v2k12 and none use the shared order itself."""
    shared = list(shared)
    windows = WINDOWS if windows is None else windows
    if window in windows:
        blocks = list(windows[window])
        if not set(blocks) <= set(shared):
            raise ValueError(f"window {window} has blocks outside the shared order")
        return blocks + [b for b in shared if b not in blocks]
    if window in (V2K, NONE):
        return shared
    raise ValueError(f"unknown window {window}")


def removed_blocks(shared, window: str) -> list[int]:
    return window_order(shared, window)[:window_k(window)]


def alive_for(shared, window: str, n_layers: int) -> list[int]:
    return heal.alive_blocks(window_order(shared, window), window_k(window), n_layers)


def lr_plan(seed0_rows, seed: int) -> list[float]:
    """Seed 0 tries both learning rates; any other seed uses `heal.pick_lr` of seed 0's rows (which must exist)."""
    if seed == 0:
        return list(ac.LRS)
    if len({round(r["lr"], 12) for r in seed0_rows}) < len(ac.LRS):
        raise SystemExit("run seed 0 first (both learning rates) for this task, window and arm")
    return [heal.pick_lr(seed0_rows)]


def usable(acc) -> bool:
    return acc is not None and acc >= USABLE_ACC - EPS


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------- scoring: classification and assembly (pure)

def classify_generation(text: str, option: str) -> dict:
    """first_ok: the generation starts (leading space dropped) with the answer; exact: `heal.exact_match` on the whole
    text (the stop); repeat: the answer twice at the start."""
    t = text.lstrip()
    return {"first_ok": t.startswith(option), "exact": heal.exact_match(text, option),
            "repeat": t.startswith(option + option)}


def _rates(rows: list[dict]) -> dict:
    n = len(rows)
    first = [r for r in rows if r["first_ok"]]
    return {"n": n, "acc": float(np.mean([r["ok"] for r in rows])), "top1_full": float(np.mean([r["full"] for r in rows])),
            "n_first_ok": len(first),
            "stop_given_correct": float(np.mean([r["exact"] for r in first])) if first else None,
            "exact": float(np.mean([r["exact"] for r in rows])), "repeat": float(np.mean([r["repeat"] for r in rows]))}


def assemble_score(recs, options, preds, full_hits, texts) -> dict:
    """Per item and per cell scores from the model's raw outputs (all lists aligned with `recs`)."""
    rows, items = [], {}
    for r, p, f, t in zip(recs, preds, full_hits, texts):
        g = classify_generation(t, options[r["answer"]])
        row = {"ok": int(p == r["answer"]), "full": int(f), **{k: int(v) for k, v in g.items()}}
        rows.append(row)
        items[r["id"]] = {"pred": int(p), "ok": row["ok"], "full": row["full"], "gen": t}
    cells = {}
    for c, idx in S.cell_index([r["rung"] for r in recs]).items():
        cells[c] = {**_rates([rows[i] for i in idx]), "chance": spec.chance([recs[i]["answer"] for i in idx])}
    return {**_rates(rows), "chance": spec.chance([r["answer"] for r in recs]), "cells": cells, "items": items}


@torch.no_grad()
def full_vocab_hits(model, tok, task: str, recs, batch_size: int = 16) -> list[int]:
    """1 where the argmax over the whole vocabulary at the answer position is the answer's option token."""
    ids = spec.option_token_ids(tok, task)
    out = []
    for b in range(0, len(recs), batch_size):
        chunk = recs[b:b + batch_size]
        enc = measure._encode(tok, task, chunk, model.device)
        top = model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1, :].argmax(-1).cpu().tolist()
        out.extend(int(t == ids[r["answer"]]) for t, r in zip(top, chunk))
        del enc
    _free(model)
    return out


def score_all(ctx, model, task: str, recs_test: list[dict]) -> dict:
    """10-option first-token accuracy and predictions, full-vocabulary top-1, greedy 4-token generation (stop given a
    correct first token, whole-answer exact, repeat), per cell and per item id."""
    tok = ctx.tokenizer
    _, preds = heal.accuracy_and_preds(model, tok, task, recs_test, ac.EVAL_BS)
    full = full_vocab_hits(model, tok, task, recs_test, ac.EVAL_BS)
    texts = heal.generate_answers(model, tok, task, recs_test, max_new_tokens=GEN_TOKENS, batch_size=ac.EVAL_BS)
    return assemble_score(recs_test, spec.TASKS[task].options, preds, full, texts)


# ---------------------------------------------------------------- R and the bars (pure)

def r_groups(task: str, score: dict, uncut: dict, metric: str = "acc") -> dict:
    """R per cell against the same seed's uncut score (chance-adjusted, `S.r_value`), and the group means the bars use:
    copy: fetch (cells matching L1_*_pad*) and select (the rest); ops: ops24 (s2-s4). metric "acc" is first-token
    accuracy (primary), "exact" is whole-answer exact match (same cell chance)."""
    by_cell = {}
    for c, u in uncut["cells"].items():
        a = score["cells"].get(c)
        by_cell[c] = None if a is None else S.r_value(a[metric], u[metric], u["chance"])
    cells = list(by_cell)
    if task == S.COPY:
        fetch = [c for c in cells if fnmatch.fnmatch(c, FETCH_PATTERN)]
        return {"fetch": mean_or_none(by_cell[c] for c in fetch),
                "select": mean_or_none(by_cell[c] for c in cells if c not in fetch), "by_cell": by_cell}
    return {"ops24": mean_or_none(by_cell.get(c) for c in OPS_CELLS), "by_cell": by_cell}


def quantify(values: dict, keys, ok) -> str:
    """"All keys" bar: fails as soon as one known value breaks it; holds only if every key is present and passes."""
    known = [values.get(k) for k in keys]
    if any(v is not None and not ok(v) for v in known):
        return "fails"
    return "holds" if keys and all(v is not None for v in known) else "not evaluable"


def _hi(v): return v >= HI - EPS
def _lo(v): return v <= LO + EPS


def verdict_w1(fetch: dict, seeds) -> dict:
    """fetch: {window: {seed: R}} (cut-then-train). R >= 0.80 under each of the five cuts, all seeds."""
    vals = {(w, s): fetch.get(w, {}).get(s) for w in WINDOWS for s in seeds}
    return {"label": quantify(vals, list(vals), _hi), "rule": f"fetch R >= {HI} under all five windows, all seeds",
            "values": {f"{w}|s{s}": v for (w, s), v in vals.items()}}


def verdict_w2(ops: dict, seeds) -> dict:
    """ops: {window: {seed: R}}. Mean R over s2-s4 <= 0.50 under W-mid or W-late (named), all seeds. W-last is
    reported but cannot satisfy W2 alone."""
    per = {w: quantify({s: ops.get(w, {}).get(s) for s in seeds}, list(seeds), _lo) for w in ("mid", "late")}
    named = [w for w, v in per.items() if v == "holds"]
    label = "holds" if named else ("fails" if all(v == "fails" for v in per.values()) else "not evaluable")
    last = quantify({s: ops.get("last", {}).get(s) for s in seeds}, list(seeds), _lo)
    note = "W-last reported only; it cannot satisfy W2 alone"
    if not named and last == "holds":
        note += " (W-last holds here but mid and late do not)"
    return {"label": label, "named": named, "per_window": per, "last": last, "note": note,
            "rule": f"ops mean R (s2-s4) <= {LO} under W-mid or W-late, all seeds",
            "values": {f"{w}|s{s}": ops.get(w, {}).get(s) for w in ("mid", "late", "last") for s in seeds}}


def verdict_wcontrol(ops_spread: dict, fetch_spread: dict, seeds) -> dict:
    """Under W-spread, ops R <= 0.50 in all seeds means W2 is "general damage, not location". Label: holds = the control
    does not show general damage (some seed above 0.50), fails = it does, not evaluable = seeds missing and none above."""
    general = quantify({s: ops_spread.get(s) for s in seeds}, list(seeds), _lo)
    label = {"holds": "fails", "fails": "holds"}.get(general, "not evaluable")
    return {"label": label, "general_damage": {"holds": True, "fails": False}.get(general),
            "rule": "under W-spread, ops R <= 0.50 in all seeds => W2 reads as general damage, not location "
                    "(label holds = the control does not show general damage)",
            "ops_spread": {f"s{s}": ops_spread.get(s) for s in seeds},
            "fetch_spread": {f"s{s}": fetch_spread.get(s) for s in seeds}}


def verdict_v1(cut_fetch: dict, seeds) -> dict:
    """cut_fetch: {seed: R} of copy pure-fetch right after W-late is removed (train-then-cut)."""
    vals = {s: cut_fetch.get(s) for s in seeds}
    redundant = quantify(vals, list(seeds), _hi) == "holds"
    return {"label": quantify(vals, list(seeds), _lo), "rule": f"fetch R <= {LO} right after W-late, all seeds",
            "values": {f"s{s}": v for s, v in vals.items()},
            "reading": "redundant across blocks (stays >= 0.80), nothing needed to move" if redundant else None}


def verdict_v2(rep_fetch: dict, seeds) -> dict:
    vals = {s: rep_fetch.get(s) for s in seeds}
    return {"label": quantify(vals, list(seeds), _hi), "rule": f"fetch R >= {HI} after repair under W-late, all seeds",
            "values": {f"s{s}": v for s, v in vals.items()}}


def verdict_v3(rep_ops: dict, named: list, seeds) -> dict:
    """rep_ops: {window: {seed: R}} after repair; `named` = the windows W2 named."""
    if not named:
        return {"label": "not evaluable", "rule": f"ops R <= {LO} after repair under the window W2 names, all seeds",
                "note": "W2 names no window", "values": {}}
    per = {w: quantify({s: rep_ops.get(w, {}).get(s) for s in seeds}, list(seeds), _lo) for w in named}
    label = "fails" if "fails" in per.values() else ("holds" if all(v == "holds" for v in per.values()) else "not evaluable")
    return {"label": label, "per_window": per, "rule": f"ops R <= {LO} after repair under the window(s) W2 names ({named}), all seeds",
            "values": {f"{w}|s{s}": rep_ops.get(w, {}).get(s) for w in named for s in seeds}}


def b1_class(values: dict, seeds) -> str:
    """rebuilt: R >= 0.80 in all seeds; not rebuilt: <= 0.50 in all seeds; else mixed (not evaluable while seeds are missing
    and the known ones do not already rule both out)."""
    known = [values.get(s) for s in seeds if values.get(s) is not None]
    can_up, can_down = all(_hi(v) for v in known), all(_lo(v) for v in known)
    if len(known) == len(seeds) and seeds:
        return "rebuilt" if can_up else ("not rebuilt" if can_down else "mixed")
    return "mixed" if not can_up and not can_down else "not evaluable"


def verdict_b1(values: dict, seeds) -> dict:
    """values: {(cut, arm): {seed: R}} (ops mean R over s2-s4). Label: fails if any arm/cut is mixed, not evaluable if any
    is missing, else holds (every arm and cut classified)."""
    cuts = sorted({cut for cut, _ in values})
    values = {(cut, arm): values.get((cut, arm), {}) for cut in cuts for arm in ARMS}
    classes = {f"{cut}|{arm}": b1_class(v, seeds) for (cut, arm), v in values.items()}
    label = ("fails" if "mixed" in classes.values() else
             "not evaluable" if not classes or "not evaluable" in classes.values() else "holds")
    return {"label": label, "classes": classes,
            "rule": f"per arm and cut: rebuilt if ops R >= {HI} in all seeds, not rebuilt if <= {LO} in all seeds, else mixed",
            "values": {f"{cut}|{arm}|s{s}": v.get(s) for (cut, arm), v in values.items() for s in seeds}}


def verdict_e1(errors: dict) -> dict:
    """errors: {source: {seed: {group: {rule: {"passes": bool|None, ...}}}}}. A rule explains the errors in a position group
    for a source if it passes in all of that source's seeds. Holds if any (source, rule, group) does."""
    status = {}
    for src, by_seed in errors.items():
        for g in GROUPS:
            for rule in RULES:
                ps = [by_seed[s].get(g, {}).get(rule, {}).get("passes") for s in by_seed]
                if not ps or any(p is None for p in ps):
                    st = "no data"
                else:
                    st = "passes" if all(ps) else "fails"
                status[f"{src}|{g}|{rule}"] = st
    passing = [k for k, v in status.items() if v == "passes"]
    label = "holds" if passing else ("fails" if "fails" in status.values() else "not evaluable")
    return {"label": label, "passing": passing, "status": status,
            "rule": f"share >= {E1_SHARE} and share - null >= {E1_GAP} in all seeds of a source"}


# ---------------------------------------------------------------- copy errors (pure)

_ASSIGN = re.compile(r"\s*\w+\s*=\s*(\d)\s*")


def rule_digits(rec: dict) -> dict:
    """The digit each rule predicts for a copy program. first-source: the digit on the first line of the form `name = digit`;
    last-assignment: the same for the last such line before `print`; decoy-end: meta["decoy_digit"]. None if absent."""
    lit = []
    for line in rec["prompt"].split("\n"):
        if line.lstrip().startswith("print("):
            break
        m = _ASSIGN.fullmatch(line)
        if m:
            lit.append(int(m.group(1)))
    return {"first_source": lit[0] if lit else None, "last_assignment": lit[-1] if lit else None,
            "decoy_end": rec["meta"].get("decoy_digit")}


def position_group(cell: str):
    parts = cell.split("_")
    return parts[1] if len(parts) > 1 and parts[1] in GROUPS else None


def keep_for_rule(rule: str, answer: int, digits: dict) -> bool:
    """A wrong item counts for `rule` unless its digit is missing, equals the right answer or equals another rule's digit."""
    d = digits[rule]
    if d is None or d == answer:
        return False
    return all(digits[o] != d for o in RULES if o != rule and digits[o] is not None)


def error_analysis(recs, preds: dict, n_shuffle: int = SHUFFLES, seed: int = ERR_SEED) -> dict:
    """{group: {rule: {n, n_wrong, share, null, passes}}} over wrong first tokens only. `preds` = {id: option index}; options
    are the digits, so an index is a digit. The null shuffles the rule's digits among the kept wrong items of each cell."""
    wrong = {g: [] for g in GROUPS}
    for r in recs:
        g, p = position_group(r["rung"]), preds.get(r["id"])
        if g is None or p is None or p == r["answer"]:
            continue
        wrong[g].append((r["rung"], int(p), r["answer"], rule_digits(r)))
    out = {}
    for gi, g in enumerate(GROUPS):
        out[g] = {}
        for ri, rule in enumerate(RULES):
            kept = [(c, p, d[rule]) for c, p, a, d in wrong[g] if keep_for_rule(rule, a, d)]
            if not kept:
                out[g][rule] = {"n": 0, "n_wrong": len(wrong[g]), "share": None, "null": None, "passes": None}
                continue
            share = float(np.mean([p == d for _, p, d in kept]))
            rng = np.random.default_rng([seed, gi, ri])
            hits = np.zeros(n_shuffle)
            for c in sorted({c for c, _, _ in kept}):
                p = np.asarray([x for cc, x, _ in kept if cc == c])
                d = np.asarray([x for cc, _, x in kept if cc == c])
                perm = np.argsort(rng.random((n_shuffle, len(d))), axis=1)
                hits += (d[perm] == p[None, :]).sum(axis=1)
            null = float(hits.mean() / len(kept))
            out[g][rule] = {"n": len(kept), "n_wrong": len(wrong[g]), "share": share, "null": null,
                            "passes": bool(share >= E1_SHARE - EPS and share - null >= E1_GAP - EPS)}
    return out


# ---------------------------------------------------------------- words-only floor (pure given records)

def words_only(train_recs, test_recs) -> dict:
    """Bag-of-words (token counts) multinomial logistic regression, trained on discovery, scored on test, per cell."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    out = {}
    for c in dict.fromkeys(r["rung"] for r in test_recs):
        tr, te = [r for r in train_recs if r["rung"] == c], [r for r in test_recs if r["rung"] == c]
        y_tr, y_te = np.asarray([r["answer"] for r in tr]), np.asarray([r["answer"] for r in te])
        if not len(tr):
            continue
        if len(set(y_tr.tolist())) < 2:
            pred = np.full(len(te), y_tr[0])
        else:
            vec = CountVectorizer(token_pattern=r"[A-Za-z0-9_]+")
            clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform([r["prompt"] for r in tr]), y_tr)
            pred = clf.predict(vec.transform([r["prompt"] for r in te]))
        out[c] = {"acc": float((pred == y_te).mean()), "chance": spec.chance(y_te.tolist()),
                  "n_train": len(tr), "n_test": len(te)}
    return out


# ---------------------------------------------------------------- probe helpers

def half_peak_layer(curve, chance: float):
    """First layer whose probe accuracy reaches chance + half of (peak - chance). Layer 0 = embeddings."""
    peak = max(curve)
    thr = chance + 0.5 * (peak - chance)
    return next((i for i, v in enumerate(curve) if v >= thr - EPS), None)


def layer_to_block(layer, alive):
    """Original block behind a probe layer of a (cut) model: layer i >= 1 is the output of alive[i - 1]; 0 is the embeddings."""
    if layer is None:
        return None
    return "embed" if layer == 0 else int(alive[layer - 1])


def store_resid(ctx, model, task: str, tag: str):
    sd = ac.ACTS / tag
    meta = measure.build_store(model, ctx.tokenizer, task, ctx.records(task, "discovery"), sd, batch_size=ac.EVAL_BS)
    return meta, sd


def fit_probe(meta, sd: Path, seed: int, jobs: int, n_alive: int) -> dict:
    """The per-layer, per-cell linear probe of `steps_cutprobe.probe_healed`, from an existing store (removed afterwards)."""
    import shutil
    try:
        X = np.load(sd / "resid.npy", mmap_mode="r")
        if X.shape[1] != n_alive + 1:
            raise SystemExit(f"store has {X.shape[1]} layers, expected {n_alive + 1}")
        a_all, g_all = np.asarray(meta["answers"]), np.asarray(meta["template_ids"])
        cells = {}
        for c, idx in S.cell_index(meta["rungs"]).items():
            a = a_all[idx]
            curve, _ = S.linear_curve(np.asarray(X[idx]), a, g_all[idx], seed, jobs)
            cells[c] = {"n": int(len(idx)), "chance": spec.chance(a.tolist()), "curve": curve, "peak": max(curve)}
        del X
    finally:
        shutil.rmtree(sd, ignore_errors=True)
    return cells


# ---------------------------------------------------------------- run store, environment

class Runs:
    """Heal rows of every arm in one file (runs.json), keyed `arm|window|task|s<seed>|lr<lr>`."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.runs = (ac.read_json(self.path) or {}).get("runs", {})

    def rows(self, **match) -> list[dict]:
        return [r for r in self.runs.values() if all(r.get(k) == v for k, v in match.items())]

    def put(self, key: str, row: dict) -> None:
        # two task streams may run side by side: merge with the file under a lock so neither drops the other's rows
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            disk = (ac.read_json(self.path) or {}).get("runs", {})
            self.runs = {**disk, **self.runs, key: row}
            ac.write(self.path, {"exploratory": True, "runs": self.runs})


def row_key(arm, window, task, seed, lr) -> str:
    return f"{arm}|{window}|{task}|s{seed}|lr{lr:g}"


def rpath(root: Path, *parts) -> Path:
    return Path(root) / ("_".join(str(p) for p in parts) + ".json")


def setup(root: Path = OUT) -> SimpleNamespace:
    d = S.configure("qwen17", False)
    cfg = S.make_cfg("qwen17", False)
    cfg.transfer = False
    root.mkdir(parents=True, exist_ok=True)
    ctx = S.make_ctx(cfg)
    return SimpleNamespace(root=root, ckpt_root=d.ckpt, cfg=cfg, ctx=ctx, runs=Runs(root / "runs.json"), wctxs={},
                           shared=list(ctx.order))


def window_ctx(env, window: str):
    """A context sharing the pretrained model and records but cutting `window` first."""
    if window not in env.wctxs:
        env.wctxs[window] = C.OwnCtx(env.ctx, window_order(env.shared, window, windows=getattr(env, "windows", None)))
    return env.wctxs[window]


def ckpt_dir(env, arm: str, window: str) -> Path:
    return env.ckpt_root / f"steps3_{arm}_{window}"


def ckpt_fn_for(env, arm: str, window: str):
    return lambda cfg, t, k, s, lr: ckpt_dir(env, arm, window) / f"{t}_k{k}_s{s}_lr{lr:g}.pt"


# ---------------------------------------------------------------- models: build, train, score

def prepare_model(env, spec_: dict):
    """The model before any repair adapter. spec_: kind ctt (pretrained, window cut), tc (rebuilt task-trained model, merged,
    then the window removed) or none (merged, uncut), plus window, k and, for tc and none, base (the k=0 row)."""
    wctx = window_ctx(env, spec_["window"])
    if spec_["kind"] == "ctt":
        return wctx.cut(spec_["k"])
    model, handle = ac.rebuild_healed(wctx, spec_["base"])
    adapters.merge_lora(handle)
    alive = list(range(model.config.num_hidden_layers))
    if spec_["kind"] == "tc":
        alive = alive_for(env.shared, spec_["window"], len(alive))
        heal.cut_model(model, alive)             # structure.excise: also renumbers config and layer_idx, as Ctx.cut does
    return model, alive


def spec_of(row: dict) -> dict:
    return {k: row[k] for k in ("kind", "window", "k", "base") if k in row}


def build_scored(env, row: dict, merge: bool = True):
    """The repaired model of a heal row (adapter loaded, merged unless asked otherwise)."""
    model, alive = prepare_model(env, spec_of(row))
    torch.manual_seed(row["seed"])
    handle = adapters.add_lora(model, list(range(len(alive))), modules=tuple(row["modules"]), r=row["rank"], scale=2.0)
    adapters.load_adapter_state(handle.modules, torch.load(row["ckpt"], map_location="cpu")["state"])
    if merge:
        adapters.merge_lora(handle)
    return model, alive


def train_built(env, spec_: dict, task: str, seed: int, lr: float, steps: int, modules, rank: int, arm: str,
                *, n_evals: int = 4) -> dict:
    """`atlas3_cuts.run_whole_heal` for a model built by `prepare_model` (train-then-cut and its control): the same
    `heal.train_whole` arguments and row fields (selection exact match in `sel_accuracy`)."""
    wctx = window_ctx(env, spec_["window"])
    cfg, tok = env.cfg, wctx.tokenizer
    key = ac.run_key(task, spec_["k"], seed, lr)
    mnt = C.max_new_tokens(wctx, task)
    print(f"== {arm}|{spec_['window']}|{key} {ac.now()} ({steps} steps, {list(modules)} r{rank}, max_new_tokens {mnt})", flush=True)
    model, alive = prepare_model(env, spec_)
    torch.manual_seed(seed)
    handle = adapters.add_lora(model, list(range(len(alive))), modules=tuple(modules), r=rank, scale=2.0)
    try:
        state, curve, tim = heal.train_whole(
            model, tok, task, wctx.records(task, "train"), wctx.records(task, "lrsel"), handle.modules, steps, lr,
            ac.BATCH, seed, n_evals=n_evals, eval_batch_size=ac.EVAL_BS, max_new_tokens=mnt,
            ckpt_tokens=getattr(cfg, "ckpt_tokens", None))
        adapters.load_adapter_state(handle.modules, state)
        t0 = time.perf_counter()
        recs = wctx.records(task, "selection")
        em, texts = heal.exact_match_scores(model, tok, task, recs, max_new_tokens=mnt, batch_size=ac.EVAL_BS)
        tim["score_seconds"] = time.perf_counter() - t0
        cp = ckpt_fn_for(env, arm, spec_["window"])(cfg, task, spec_["k"], seed, lr)
        ac.save_ckpt(cp, state, {"key": key, "task": task, "k": spec_["k"], "seed": seed, "lr": lr, "rank": rank,
                                 "n_alive": len(alive), "best_step": tim["best_step"], "arm": arm})
        row = {"key": key, "task": task, "k": spec_["k"], "seed": seed, "lr": lr, "steps": steps, "n_alive": len(alive),
               "curve": curve, "best_lrsel": tim["best_lrsel"], "best_step": tim["best_step"], "n": len(recs),
               "sel_accuracy": em, "chance": spec.chance([r["answer"] for r in recs]), "max_new_tokens": mnt,
               "examples": [{"text": t, "answer": heal.answer_text(task, r)} for t, r in zip(texts[:5], recs[:5])],
               "timings": {k_: v for k_, v in tim.items() if k_ not in ("best_lrsel", "best_step")},
               "ckpt": str(cp), "ts": ac.now()}
    finally:
        handle.remove()
        del model
        ac.release(wctx)
    print(f"{key}: lrsel exact {row['best_lrsel']:.3f} (step {row['best_step']}) sel exact {row['sel_accuracy']:.3f}", flush=True)
    return row


def ensure_rows(env, arm: str, window: str, task: str, seed: int, train_fn, extra: dict) -> dict:
    """Train what is missing under the learning-rate rule and return the winner-lr row of (arm, window, task, seed)."""
    base = dict(arm=arm, window=window, task=task)
    for lr in lr_plan(env.runs.rows(**base, seed=0), seed):
        key = row_key(arm, window, task, seed, lr)
        if key not in env.runs.runs:
            row = train_fn(lr)
            row.update(arm=arm, window=window, exploratory=True, **extra)
            env.runs.put(key, row)
    lr = heal.pick_lr(env.runs.rows(**base, seed=0))
    return env.runs.runs[row_key(arm, window, task, seed, lr)]


def run_score(env, row: dict, task: str) -> dict:
    wctx = env.ctx
    model, _ = build_scored(env, row)
    try:
        return score_all(wctx, model, task, wctx.records(task, "test"))
    finally:
        del model
        ac.release(wctx)


def get_k0_row(env, task: str, seed: int, train: bool):
    """The uncut task-trained row: the v2 winner when present, else this study's own (trained here for copy seed 3)."""
    v2 = ac.read_merged("runs", "", "runs")
    row = S.winner_row(v2, task, 0, seed)
    if row is not None:
        return row
    own = env.runs.rows(arm="uncut", task=task, seed=seed)
    if own:
        return own[0]
    if not train:
        raise SystemExit(f"no k=0 row for {task} seed {seed}; run --stage uncut first")
    lr = ac.winner_lr(v2, task, 0, "per_k")       # seed 0's pick in v2
    if lr is None:
        raise SystemExit(f"no v2 seed-0 winner learning rate for {task} k=0")
    row = C.run_whole_heal(env.ctx, task, 0, seed, lr, ckpt_fn=ckpt_fn_for(env, "uncut", NONE))
    row.update(arm="uncut", window=NONE, exploratory=True)
    env.runs.put(row_key("uncut", NONE, task, seed, lr), row)
    return row


def load_uncut(env, task: str, seed: int) -> dict:
    """The seed's uncut score; refuses (clear message) unless the seed is usable."""
    p = rpath(env.root, "uncut", task, f"s{seed}")
    doc = ac.read_json(p)
    if doc is None:
        raise SystemExit(f"{task} seed {seed}: no uncut score; run --stage uncut first")
    if not usable(doc["score"]["acc"]):
        raise SystemExit(f"{task} seed {seed} is not usable: uncut test accuracy {doc['score']['acc']:.3f} < {USABLE_ACC}")
    return doc["score"]


def base_spec(env, task: str, seed: int) -> dict:
    r = get_k0_row(env, task, seed, train=False)
    return {"ckpt": r["ckpt"], "seed": r["seed"], "k": 0}


# ---------------------------------------------------------------- stages

def seeds_of(task: str, seed) -> tuple:
    if seed is None:
        return SEEDS[task]
    if seed not in SEEDS[task]:
        raise SystemExit(f"{task}: seeds are {SEEDS[task]}, not {seed}")
    return (seed,)


def stage_uncut(env, tasks, seed) -> None:
    for task in tasks:
        for sd in seeds_of(task, seed):
            out = rpath(env.root, "uncut", task, f"s{sd}")
            if out.exists():
                print(f"{out} exists", flush=True)
                continue
            row = get_k0_row(env, task, sd, train=True)
            model, handle = ac.rebuild_healed(env.ctx, row)
            adapters.merge_lora(handle)
            try:
                score = score_all(env.ctx, model, task, env.ctx.records(task, "test"))
            finally:
                del model
                ac.release(env.ctx)
            ok = usable(score["acc"])
            ac.write(out, {"exploratory": True, "task": task, "seed": sd, "row_key": row["key"], "lr": row["lr"],
                           "ckpt": row["ckpt"], "usable": ok, "score": score, "ts": ac.now()})
            print(f"uncut {task} s{sd}: first-token {score['acc']:.3f} -> {'usable' if ok else 'NOT USABLE (< 0.90)'}", flush=True)


def stage_ctt(env, tasks, windows, seed, jobs, arm: str = "ctt", steps: int | None = None) -> None:
    """arm "ctt": the 1a cells. arm "ctt600" (post hoc budget control): the same with `steps` = MATCHED_STEPS, no probe."""
    noprobe = NOPROBE or arm != "ctt"
    for task in tasks:
        for window in windows:
            for sd in seeds_of(task, seed):
                uncut = load_uncut(env, task, sd)
                out = rpath(env.root, arm, task, window, f"s{sd}")
                if out.exists():
                    print(f"{out} exists", flush=True)
                    continue
                wctx = window_ctx(env, window)
                alive = alive_for(env.shared, window, env.cfg.n_layers)
                extra = {"kind": "ctt", "k": K, "modules": list(QV), "rank": ac.RANK, "alive_blocks": alive,
                         "removed": list(WINDOWS[window])}

                if steps is not None:
                    extra["steps"] = steps

                def train(lr, wctx=wctx, task=task, sd=sd, window=window):
                    # order-based ctx: wctx.cut(4) removes exactly the window's blocks from the pretrained model
                    return C.run_whole_heal(wctx, task, K, sd, lr, own=True, removed=list(WINDOWS[window]),
                                            ckpt_fn=ckpt_fn_for(env, arm, window), steps=steps)
                row = ensure_rows(env, arm, window, task, sd, train, extra)
                t0 = time.perf_counter()
                score = run_score(env, row, task)
                probe = None
                if not noprobe:
                    cells, _ = CP.probe_healed(wctx, row, task, sd, jobs, len(alive), f"steps3_ctt_{task}_{window}_s{sd}")
                    probe = {"cells": cells, "alive_blocks": alive}
                ac.write(out, {"exploratory": True, "task": task, "window": window, "seed": sd, "lr": row["lr"],
                               "ckpt": row["ckpt"], "alive_blocks": alive, "uncut_acc": uncut["acc"], "score": score,
                               "probe": probe, "seconds": time.perf_counter() - t0, "arm": arm,
                               "steps": steps or STD_STEPS,
                               "ts": ac.now()})
                print(f"{arm} {task} {window} s{sd}: first-token {score['acc']:.3f}", flush=True)


def stage_tc(env, tasks, windows, seed, jobs) -> None:
    for task in tasks:
        for window in windows:
            for sd in seeds_of(task, seed):
                load_uncut(env, task, sd)
                out = rpath(env.root, "tc", task, window, f"s{sd}")
                if out.exists():
                    print(f"{out} exists", flush=True)
                    continue
                base = base_spec(env, task, sd)
                kind = NONE if window == NONE else "tc"
                spec_ = {"kind": kind, "window": window, "k": window_k(window), "base": base}
                t0 = time.perf_counter()
                after_cut = None
                if kind == "tc":
                    model, _ = prepare_model(env, spec_)
                    try:
                        after_cut = score_all(env.ctx, model, task, env.ctx.records(task, "test"))
                    finally:
                        del model
                        ac.release(env.ctx)
                    print(f"tc {task} {window} s{sd}: after cut first-token {after_cut['acc']:.3f}", flush=True)
                alive = alive_for(env.shared, window, env.cfg.n_layers)
                extra = {"kind": kind, "k": window_k(window), "base": base, "modules": list(QV), "rank": ac.RANK,
                         "alive_blocks": alive}

                def train(lr, spec_=spec_, task=task, sd=sd):
                    return train_built(env, spec_, task, sd, lr, STD_STEPS, QV, ac.RANK, "tc")
                row = ensure_rows(env, "tc", window, task, sd, train, extra)
                model, _ = build_scored(env, row)
                wctx = window_ctx(env, window)
                probe = None
                try:
                    after_repair = score_all(env.ctx, model, task, env.ctx.records(task, "test"))
                    if kind == "tc" and not NOPROBE:
                        tag = f"steps3_tc_{task}_{window}_s{sd}"
                        meta, sd_dir = store_resid(wctx, model, task, tag)
                finally:
                    del model
                    ac.release(env.ctx)
                if kind == "tc" and not NOPROBE:
                    probe = {"cells": fit_probe(meta, sd_dir, sd, jobs, len(alive)), "alive_blocks": alive}
                ac.write(out, {"exploratory": True, "task": task, "window": window, "seed": sd, "lr": row["lr"],
                               "ckpt": row["ckpt"], "base_ckpt": base["ckpt"], "alive_blocks": alive,
                               "after_cut": after_cut, "after_repair": after_repair, "probe": probe,
                               "seconds": time.perf_counter() - t0, "ts": ac.now()})
                print(f"tc {task} {window} s{sd}: after repair first-token {after_repair['acc']:.3f}", flush=True)


def stage_arms(env, window, arms, seed) -> None:
    if window is None or window == NONE:
        raise SystemExit("--stage arms needs --window (the window W2 names, or v2k12)")
    C.check_alllin_params(env.ctx.pristine.config)
    task = S.OPS
    for cut in dict.fromkeys([window, V2K]):
        for arm in arms:
            modules, rank, mult = ARMS[arm]
            steps = STD_STEPS * mult
            for sd in seeds_of(task, seed):
                load_uncut(env, task, sd)
                out = rpath(env.root, "arms", cut, arm, f"s{sd}")
                if out.exists():
                    print(f"{out} exists", flush=True)
                    continue
                extra = {"kind": "ctt", "k": window_k(cut), "modules": list(modules), "rank": rank}

                def train(lr, cut=cut, sd=sd, steps=steps, modules=modules, rank=rank, arm=arm):
                    # exactly as D2: run_whole_heal with an arm object and a step count (4 evaluations)
                    return C.run_whole_heal(window_ctx(env, cut), task, window_k(cut), sd, lr, own=True,
                                            removed=removed_blocks(env.shared, cut),
                                            arm=SimpleNamespace(modules=modules, rank=rank), steps=steps,
                                            ckpt_fn=ckpt_fn_for(env, arm, cut))
                row = ensure_rows(env, arm, cut, task, sd, train, extra)
                score = run_score(env, row, task)
                ac.write(out, {"exploratory": True, "task": task, "cut": cut, "arm": arm, "seed": sd, "lr": row["lr"],
                               "steps": steps, "curve": row["curve"], "ckpt": row["ckpt"], "score": score, "ts": ac.now()})
                print(f"arms {cut} {arm} s{sd}: first-token {score['acc']:.3f}", flush=True)


def stage_probe(env, jobs) -> None:
    """Fill in the probe of every finished ctt / tc result that was written without one (STEPS3_NOPROBE=1 runs)."""
    for out in sorted(env.root.glob("ctt_*.json")) + sorted(env.root.glob("tc_*.json")):
        if not out.name.startswith(os.environ.get("STEPS3_PROBE_ONLY", "")):  # optional file-name prefix, to order a partial pass
            continue
        d = ac.read_json(out)
        if d.get("probe") or d["window"] == NONE:
            continue
        arm = out.name.split("_")[0]
        task, window, sd, alive = d["task"], d["window"], d["seed"], d["alive_blocks"]
        row = env.runs.runs[row_key(arm, window, task, sd, d["lr"])]
        if row["ckpt"] != d["ckpt"]:
            raise SystemExit(f"{out}: checkpoint {d['ckpt']} is not the run row's {row['ckpt']}")
        wctx = window_ctx(env, window)
        tag = f"steps3_{arm}_{task}_{window}_s{sd}"
        if arm == "ctt":
            cells, _ = CP.probe_healed(wctx, row, task, sd, jobs, len(alive), tag)
        else:
            model, _ = build_scored(env, row)
            try:
                meta, sd_dir = store_resid(wctx, model, task, tag)
            finally:
                del model
                ac.release(env.ctx)
            cells = fit_probe(meta, sd_dir, sd, jobs, len(alive))
        d["probe"] = {"cells": cells, "alive_blocks": alive}
        d["probe_ts"] = ac.now()
        ac.write(out, d)
        print(f"probe {out.name}: done", flush=True)


def stage_ftonly(env, seed) -> None:
    """1d control: copy, W-late, repaired on the first-token objective only (`ac.run_heal`), to show what the objective alone does."""
    task, window = S.COPY, "late"
    wctx = window_ctx(env, window)
    for sd in seeds_of(task, seed):
        load_uncut(env, task, sd)
        out = rpath(env.root, "ftonly", f"s{sd}")
        if out.exists():
            print(f"{out} exists", flush=True)
            continue
        ac.CKPT = ckpt_dir(env, "ftonly", window)
        alive = alive_for(env.shared, window, env.cfg.n_layers)
        extra = {"kind": "ctt", "k": K, "modules": list(QV), "rank": ac.RANK, "alive_blocks": alive,
                 "objective": "first-token"}

        def train(lr, sd=sd):
            return ac.run_heal(wctx, task, K, sd, lr)
        row = ensure_rows(env, "ftonly", window, task, sd, train, extra)
        score = run_score(env, row, task)
        ac.write(out, {"exploratory": True, "task": task, "window": window, "seed": sd, "lr": row["lr"],
                       "ckpt": row["ckpt"], "objective": "first-token", "score": score, "ts": ac.now()})
        print(f"ftonly {window} s{sd}: first-token {score['acc']:.3f}, stop|correct {score['stop_given_correct']}", flush=True)


def stage_v2score(env, tasks, seed) -> None:
    v2 = ac.read_merged("runs", "", "runs")
    for task in tasks:
        for sd in seeds_of(task, seed):
            row = S.winner_row(v2, task, S.KCUT, sd)
            if row is None:
                print(f"v2score {task} s{sd}: no v2 k12 winner; skipped", flush=True)
                continue
            load_uncut(env, task, sd)
            out = rpath(env.root, "v2score", task, f"s{sd}")
            if out.exists():
                print(f"{out} exists", flush=True)
                continue
            model, handle = ac.rebuild_healed(env.ctx, row)
            adapters.merge_lora(handle)
            try:
                score = score_all(env.ctx, model, task, env.ctx.records(task, "test"))
            finally:
                del model
                ac.release(env.ctx)
            ac.write(out, {"exploratory": True, "task": task, "seed": sd, "k": S.KCUT, "lr": row["lr"],
                           "ckpt": row["ckpt"], "score": score, "ts": ac.now()})
            print(f"v2score {task} s{sd}: first-token {score['acc']:.3f}", flush=True)


def error_sources(root: Path, seeds) -> dict:
    """{source: {seed: path}} of the saved copy test scores that exist."""
    out = {}
    for src, name in [(V2K, lambda s: rpath(root, "v2score", S.COPY, f"s{s}"))] + \
            [(f"ctt:{w}", (lambda s, w=w: rpath(root, "ctt", S.COPY, w, f"s{s}"))) for w in WINDOWS]:
        found = {s: name(s) for s in seeds if name(s).exists()}
        if found:
            out[src] = found
    return out


def stage_errors(root: Path, recs=None, n_shuffle: int = SHUFFLES) -> dict:
    recs = recs if recs is not None else spec.read_records(S.COPY, "test")
    seeds = usable_seeds(root, S.COPY)
    result = {"exploratory": True, "seeds": list(seeds), "shuffles": n_shuffle, "sources": {}}
    for src, files in error_sources(root, seeds).items():
        result["sources"][src] = {}
        for sd, p in files.items():
            items = json.loads(Path(p).read_text())["score"]["items"]
            result["sources"][src][str(sd)] = error_analysis(recs, {i: v["pred"] for i, v in items.items()}, n_shuffle)
    ac.write(Path(root) / "errors.json", result)
    return result


def stage_floor(root: Path, tasks=None) -> dict:
    out = {"exploratory": True, "method": "bag-of-words counts, multinomial logistic regression; discovery -> test, per cell",
           "tasks": {}}
    for task in tasks or S.TASKS:
        out["tasks"][task] = words_only(spec.read_records(task, "discovery"), spec.read_records(task, "test"))
    ac.write(Path(root) / "floor.json", out)
    return out


# ---------------------------------------------------------------- table stage

def usable_seeds(root: Path, task: str) -> list[int]:
    out = []
    for s in SEEDS[task]:
        doc = ac.read_json(rpath(root, "uncut", task, f"s{s}"))
        if doc and usable(doc["score"]["acc"]):
            out.append(s)
    return out


def load_results(root: Path) -> dict:
    """Every result file that exists, under `uncut`, `ctt`, `tc`, `arms`, `ftonly`, `v2score`, `errors`, `floor`."""
    rd = lambda p: ac.read_json(p)  # noqa: E731
    D = {"uncut": {}, "ctt": {}, "tc": {}, "arms": {}, "ftonly": {}, "v2score": {}}
    for task in SEEDS:
        for s in SEEDS[task]:
            for name in ("uncut", "v2score"):
                doc = rd(rpath(root, name, task, f"s{s}"))
                if doc:
                    D[name].setdefault(task, {})[s] = doc
            for w in [*WINDOWS, NONE]:
                for name in ("ctt", "tc"):
                    doc = rd(rpath(root, name, task, w, f"s{s}"))
                    if doc:
                        D[name].setdefault(task, {}).setdefault(w, {})[s] = doc
    for s in SEEDS[S.OPS]:
        for cut in [*WINDOWS, V2K]:
            for arm in ARMS:
                doc = rd(rpath(root, "arms", cut, arm, f"s{s}"))
                if doc:
                    D["arms"].setdefault(cut, {}).setdefault(arm, {})[s] = doc
    for s in SEEDS[S.COPY]:
        doc = rd(rpath(root, "ftonly", f"s{s}"))
        if doc:
            D["ftonly"][s] = doc
    D["errors"], D["floor"] = rd(Path(root) / "errors.json"), rd(Path(root) / "floor.json")
    return D


def _r(D, task, doc_score, seed, group):
    unc = D["uncut"].get(task, {}).get(seed)
    if doc_score is None or unc is None:
        return None
    return r_groups(task, doc_score, unc["score"])[group]


def collect_r(D, seeds_by_task) -> dict:
    """The R dictionaries the bar functions read."""
    out = {"ctt_fetch": {}, "ctt_ops": {}, "tc_cut_fetch": {}, "tc_rep_fetch": {}, "tc_cut_ops": {}, "tc_rep_ops": {},
           "none_fetch": {}, "none_ops": {}, "arms": {}}
    for task, gk, tag in ((S.COPY, "fetch", "fetch"), (S.OPS, "ops24", "ops")):
        for s in seeds_by_task[task]:
            for w, by_s in D["ctt"].get(task, {}).items():
                if s in by_s:
                    out[f"ctt_{tag}"].setdefault(w, {})[s] = _r(D, task, by_s[s]["score"], s, gk)
            for w, by_s in D["tc"].get(task, {}).items():
                if s in by_s:
                    d = by_s[s]
                    if w == NONE:
                        out[f"none_{tag}"][s] = _r(D, task, d["after_repair"], s, gk)
                        continue
                    out[f"tc_cut_{tag}"].setdefault(w, {})[s] = _r(D, task, d.get("after_cut"), s, gk)
                    out[f"tc_rep_{tag}"].setdefault(w, {})[s] = _r(D, task, d["after_repair"], s, gk)
    for s in seeds_by_task[S.OPS]:
        for cut, by_arm in D["arms"].items():
            for arm, by_s in by_arm.items():
                if s in by_s:
                    out["arms"].setdefault((cut, arm), {})[s] = _r(D, S.OPS, by_s[s]["score"], s, "ops24")
    return out


def build_verdicts(D, seeds_by_task, errors) -> dict:
    cs, os_ = seeds_by_task[S.COPY], seeds_by_task[S.OPS]
    R = collect_r(D, seeds_by_task)
    w2 = verdict_w2(R["ctt_ops"], os_)
    return {"exploratory": True, "seeds": {"steps_copy": list(cs), "steps_ops": list(os_)},
            "bars": {"W1": verdict_w1(R["ctt_fetch"], cs), "W2": w2,
                     "W-control": verdict_wcontrol(R["ctt_ops"].get("spread", {}), R["ctt_fetch"].get("spread", {}), os_),
                     "V1": verdict_v1(R["tc_cut_fetch"].get("late", {}), cs),
                     "V2": verdict_v2(R["tc_rep_fetch"].get("late", {}), cs),
                     "V3": verdict_v3(R["tc_rep_ops"], w2["named"], os_),
                     "B1": verdict_b1(R["arms"], os_),
                     "E1": verdict_e1((errors or {}).get("sources", {}))}}


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def damage_table(D, shared_ppl: dict | None) -> list[str]:
    """Unrepaired damage per window: each task's score right after the cut (tc after_cut) and the single-block perplexity rise."""
    lines = ["| window | blocks | copy acc | copy R fetch | copy R select | ops acc | ops R s2-s4 | single-block ppl rise (sum; mean) |",
             "|---|---|---|---|---|---|---|---|"]
    for w, blocks in WINDOWS.items():
        cells = []
        for task, gks in ((S.COPY, ("fetch", "select")), (S.OPS, ("ops24",))):
            docs = D["tc"].get(task, {}).get(w, {})
            accs = [d["after_cut"]["acc"] for d in docs.values() if d.get("after_cut")]
            rs = {g: mean_or_none(_r(D, task, d.get("after_cut"), s, g) for s, d in docs.items()) for g in gks}
            cells += [_f(mean_or_none(accs))] + [_f(rs[g]) for g in gks]
        ppl = "n/a"
        if shared_ppl:
            v = [shared_ppl[str(b)] for b in blocks if str(b) in shared_ppl]
            if len(v) == len(blocks):
                ppl = f"{sum(v):.1f}; {sum(v) / len(v):.1f}"
        lines.append(f"| {w} | {','.join(map(str, blocks))} | " + " | ".join(cells) + f" | {ppl} |")
    return lines


def ppl_rise(order_path: Path | None) -> dict | None:
    doc = ac.read_json(order_path) if order_path else None
    if not doc or "ppl" not in doc:
        return None
    base = doc.get("base_ppl")
    return {b: v - base for b, v in doc["ppl"].items()} if base is not None else None


def score_row(label, task, seed, score, unc) -> str:
    g = r_groups(task, score, unc["score"])
    rs = (f"fetch {_f(g['fetch'])}, select {_f(g['select'])}" if task == S.COPY else
          f"s2-s4 {_f(g['ops24'])}; " + ", ".join(f"{c} {_f(g['by_cell'].get(c))}" for c in ("s1", "s2", "s3", "s4")))
    ge = r_groups(task, score, unc["score"], "exact")
    rs += (f" // exact-match R: fetch {_f(ge['fetch'])}, select {_f(ge['select'])}" if task == S.COPY
           else f" // exact-match R s2-s4 {_f(ge['ops24'])}")
    return (f"| {label} | {seed} | {_f(score['acc'])} | {_f(score['top1_full'])} | {_f(score['stop_given_correct'])} | "
            f"{_f(score['exact'])} | {_f(score['repeat'])} | {rs} |")


def render_tables(D, verdicts, shared_ppl, floor) -> str:
    L = ["# Steps v3, Stage 1 results (EXPLORATORY: 3 seeds, nothing here is a confirmed result)", "",
         "Columns: first-token = 10-option accuracy; top-1 = full-vocabulary argmax equals the answer token; stop|c = share "
         "exactly the answer among items whose generated first token is right; repeat = answer token twice. R is "
         "chance-adjusted against the same seed's uncut score; first-token R is primary, exact-match R is given after '//'. "
         "Training objective: all models and repairs are trained on the whole answer plus end-of-turn (v2 recipe, checkpoint by "
         "lrsel exact match); only the 1d control is first-token only.", ""]
    L += ["## Uncut task-trained models", "", "| task | seed | first-token | top-1 | stop|c | exact | usable (>= 0.90) |", "|---|---|---|---|---|---|---|"]
    for task in SEEDS:
        for s, d in sorted(D["uncut"].get(task, {}).items()):
            sc = d["score"]
            L.append(f"| {task} | {s} | {_f(sc['acc'])} | {_f(sc['top1_full'])} | {_f(sc['stop_given_correct'])} | "
                     f"{_f(sc['exact'])} | {'yes' if d['usable'] else 'NO'} |")
    head = ["| model | seed | first-token | top-1 | stop|c | exact | repeat | R |", "|---|---|---|---|---|---|---|---|"]
    for title, key, getter in (("1a Cut-then-train", "ctt", lambda d: [("", d["score"])]),
                               ("1b Train-then-cut", "tc", lambda d: [("after cut", d.get("after_cut")),
                                                                      ("after repair", d["after_repair"])])):
        for task in SEEDS:
            if not D[key].get(task):
                continue
            L += ["", f"## {title}: {task}", ""] + head
            for w, by_s in D[key][task].items():
                for s, d in sorted(by_s.items()):
                    unc = D["uncut"].get(task, {}).get(s)
                    for tag, sc in getter(d):
                        if sc is not None and unc is not None:
                            label = w if key == "ctt" else (f"control (+{STD_STEPS} steps)" if w == NONE else f"{w} {tag}")
                            L.append(score_row(label, task, s, sc, unc))
    if D["ctt"].get(S.OPS):
        ops = {w: mean_or_none(_r(D, S.OPS, d["score"], s, "ops24") for s, d in by_s.items())
               for w, by_s in D["ctt"][S.OPS].items()}
        best = max((v, w) for w, v in ops.items() if v is not None)[1] if any(v is not None for v in ops.values()) else None
        L += ["", f"### Report only: ops R by operation count, cut-then-train (ops-best window: {best})", "",
              "| window | s1 | s2 | s3 | s4 | s1-best | s2-best | s3-best | s4-best |", "|---|---|---|---|---|---|---|---|---|"]
        mean_c = lambda w, c: mean_or_none(  # noqa: E731
            r_groups(S.OPS, d["score"], D["uncut"][S.OPS][s]["score"])["by_cell"].get(c)
            for s, d in D["ctt"][S.OPS].get(w, {}).items() if s in D["uncut"].get(S.OPS, {}))
        for w in D["ctt"][S.OPS]:
            vals = [mean_c(w, c) for c in ("s1", "s2", "s3", "s4")]
            ref = [mean_c(best, c) for c in ("s1", "s2", "s3", "s4")] if best else [None] * 4
            diff = [None if a is None or b is None else a - b for a, b in zip(vals, ref)]
            L.append(f"| {w} | " + " | ".join(_f(v) for v in vals + diff) + " |")
    if D["arms"]:
        L += ["", "## 1c Arms (ops, cut-then-train, whole-answer objective as D2; mean R over s2-s4; curve = lrsel exact match at the 4 evaluations)", "",
              "| cut | arm | seed | steps | first-token | R s2-s4 | curve |", "|---|---|---|---|---|---|---|"]
        for cut, by_arm in D["arms"].items():
            for arm, by_s in by_arm.items():
                for s, d in sorted(by_s.items()):
                    curve = " ".join(_f(c.get("lrsel_exact"), 2) for c in d["curve"])
                    L.append(f"| {cut} | {arm} | {s} | {d['steps']} | {_f(d['score']['acc'])} | "
                             f"{_f(_r(D, S.OPS, d['score'], s, 'ops24'))} | {curve} |")
    if D["ftonly"]:
        L += ["", "## 1d First-token-only control (copy, W-late; standard = whole-answer repair, first-token-only = repair on the "
                  "first-token objective alone)", "",
              "| arm | seed | first-token | stop|c | exact | repeat |", "|---|---|---|---|---|---|"]
        for s, d in sorted(D["ftonly"].items()):
            std = D["ctt"].get(S.COPY, {}).get("late", {}).get(s)
            for lab, sc in (("standard (whole-answer)", std and std["score"]), ("first-token only", d["score"])):
                if sc:
                    L.append(f"| {lab} | {s} | {_f(sc['acc'])} | {_f(sc['stop_given_correct'])} | {_f(sc['exact'])} | {_f(sc['repeat'])} |")
    L += ["", "## Unrepaired damage per window (score right after the cut; perplexity from the cut-order file)", ""]
    L += damage_table(D, shared_ppl)
    if shared_ppl is None:
        L += ["", "Perplexity rise per block is not available (order file missing or without per-block values)."]
    else:
        L += ["", "Perplexity: rise of the general-text perplexity when that single block alone is removed "
                  "(order.json `ppl` minus `base_ppl`); the window's joint rise was not measured."]
    L += ["", "## Probe: half-peak block before and after (pure-fetch cells; 'before' = v2 probe of the uncut task-trained model, "
          "linear curve, same half-peak rule), and probe peak against the words-only floor", "",
          "| arm | window | seed | cell | block before | block after | probe peak | words-only floor | inside the model |", "|---|---|---|---|---|---|---|---|---|"]
    fl = (floor or {}).get("tasks", {}).get(S.COPY, {})
    for key in ("ctt", "tc"):
        for w, by_s in D[key].get(S.COPY, {}).items():
            for s, d in sorted(by_s.items()):
                probe = d.get("probe")
                if not probe:
                    continue
                v2p = ac.read_json(S.ROOT_OUT / f"probe_{S.COPY}_s{s}.json")
                for c, pc in probe["cells"].items():
                    if not fnmatch.fnmatch(c, FETCH_PATTERN):
                        continue
                    before = None
                    if v2p and c in v2p.get("cells", {}):
                        v = v2p["cells"][c]
                        before = layer_to_block(half_peak_layer(v["curve"], v["chance"]), list(range(len(v["curve"]) - 1)))
                    after = layer_to_block(half_peak_layer(pc["curve"], pc["chance"]), probe["alive_blocks"])
                    f_acc = fl.get(c, {}).get("acc")
                    gap = None if f_acc is None else pc["peak"] - f_acc
                    L.append(f"| {key} | {w} | {s} | {c} | {before if before is not None else 'n/a'} | {after} | "
                             f"{_f(pc['peak'])} | {_f(f_acc)} | {_f(gap)} |")
    L += ["", "## 1e Copy errors (wrong first tokens only)", ""]
    err = D.get("errors")
    if err:
        L += ["| source | seed | group | rule | n | share | null | passes |", "|---|---|---|---|---|---|---|---|"]
        for src, by_s in err["sources"].items():
            for s, groups in by_s.items():
                for g in GROUPS:
                    for rule in RULES:
                        e = groups.get(g, {}).get(rule, {})
                        L.append(f"| {src} | {s} | {g} | {rule} | {e.get('n')} | {_f(e.get('share'))} | {_f(e.get('null'))} | {e.get('passes')} |")
    else:
        L.append("not run (needs the errors stage)")
    L += ["", "## Bars", "", "| bar | label | rule | numbers |", "|---|---|---|---|"]
    for name, v in verdicts["bars"].items():
        nums = v.get("values") or v.get("classes") or v.get("passing") or v.get("ops_spread")
        extra = " ".join(str(v[k]) for k in ("note", "reading", "named") if v.get(k))
        L.append(f"| {name} | **{v['label']}** | {v['rule']} | {json.dumps(nums, default=str)} {extra} |")
    return "\n".join(L) + "\n"


def stage_table(root: Path, order_path: Path | None = ORDER_JSON) -> dict:
    root = Path(root)
    D = load_results(root)
    seeds_by_task = {t: usable_seeds(root, t) for t in SEEDS}
    verdicts = build_verdicts(D, seeds_by_task, D["errors"])
    md = render_tables(D, verdicts, ppl_rise(order_path), D["floor"])
    (root / "tables.md").write_text(md)
    ac.write(root / "verdicts.json", verdicts)
    return verdicts


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--task", choices=[*SHORT, *SHORT.values()], default=None)
    ap.add_argument("--window", choices=[*WINDOWS, NONE, V2K], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--arm", choices=sorted(ARMS), default=None)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args(argv)
    task = SHORT.get(args.task, args.task)
    if args.stage == "table":
        stage_table(OUT)
        print(f"wrote {OUT / 'tables.md'} and {OUT / 'verdicts.json'}")
        return 0
    if args.stage == "floor":
        stage_floor(OUT)
        return 0
    if args.stage == "errors":
        stage_errors(OUT)
        return 0
    env = setup()
    both = (task,) if task else (S.COPY, S.OPS)
    if args.stage == "uncut":
        stage_uncut(env, both, args.seed)
    elif args.stage == "ctt":
        if args.window in (NONE, V2K):
            raise SystemExit("ctt windows are early, mid, late, last, spread")
        stage_ctt(env, both, [args.window] if args.window else list(WINDOWS), args.seed, args.jobs)
    elif args.stage == "ctt600":
        if args.window in (NONE, V2K):
            raise SystemExit("ctt600 windows are early, mid, late, last, spread")
        for t in both:
            ws = [args.window] if args.window else list(CTT600_CELLS[t])
            stage_ctt(env, [t], ws, args.seed, args.jobs, arm="ctt600", steps=MATCHED_STEPS)
    elif args.stage == "tc":
        if args.window == V2K:
            raise SystemExit("tc windows are early, mid, late, last, spread, none")
        stage_tc(env, both, [args.window] if args.window else [*WINDOWS, NONE], args.seed, args.jobs)
    elif args.stage == "arms":
        if task not in (None, S.OPS):
            raise SystemExit("arms is an ops-only stage")
        stage_arms(env, args.window, [args.arm] if args.arm else list(ARMS), args.seed)
    elif args.stage == "ftonly":
        if task not in (None, S.COPY) or args.window not in (None, "late"):
            raise SystemExit("ftonly is copy, window late")
        stage_ftonly(env, args.seed)
    elif args.stage == "probe":
        stage_probe(env, args.jobs)
    elif args.stage == "v2score":
        stage_v2score(env, (task,) if task else (S.COPY,), args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
