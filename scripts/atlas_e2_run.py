#!/usr/bin/env python3
"""Robustness E2 pod runner (docs/ATLAS-robustness.md, "E2 revision R1" and "E2 revision R2", where R2 wins): the check tasks 26-30
and binary doing tasks 31-35 of the clean pairs healed like their doing twins, then the verdicts E2-A, E2-B, E2-F, E2-D, E2-C, E2-L and
the claim-6 reading. R2: every R and dR of a pair is computed on the pair's common test ids (per-record hits are saved by `test`
and `doexact`); arith pairs on unseen-expression records; E2-L annotates check - do only; dR gets a record bootstrap interval; E2-A
uses eligible pairs; E2-D is descriptive; a staged pod (stage 1 at 1.7B seed 0, then `eligibility`, then the full plan).

  --stage sweep        --size 1.7B|0.6B --seed S [--shard I/N]   heal check and bin tasks, k in {0, 12}, lrs 3e-4/1e-3 at seed 0,
                                                                  seeds 1-2 at the seed-0 winner (C1's rule; tie 3e-4)
  --stage test         --size ...                                 whole-answer exact match on the test split (C1's test_exact),
                                                                  per-record hits to test_hits.jsonl (id, task, k, seed, lr, hit)
  --stage healedprobe  --size ... --task T                        L_on_healed of the seed-0, k=0 model (D1's measure) with a 1,000-resample
                                                                  bootstrap interval; check tasks also probe meta.do_answer
  --stage bytype       --size ...                                 check accuracy by near-miss type and distance (seed-0 winner, k 0 and 12)
  --stage ftdo         --size ...                                 E2-L and the kept-subset report: the doing tasks' saved heal checkpoints
                                                                  (read-only), restricted-option argmax on the test split, per record
  --stage doexact      --size ... [--seed S]                      the doing side's saved shared-order checkpoints (C1's lr and
                                                                  seeds; 0.6B: stage B's whole arm) re-scored by exact match on the
                                                                  test split, per record to doexact_hits.jsonl
  --stage eligibility  no model [--force-eligible N]              stage 1's gate: eligibility on common ids at seed 0 ->
                                                                  e2chk/stage1.md, stage1.json, marker STAGE1_GO (>= 3 E2-B-eligible
                                                                  pairs) or STAGE1_STOP; the gate file needs >= 4 clean pairs (R3)
  --stage table        no model                                   per-pair tables, verdicts, summary.json and tables.md
  --list-tasks                                                    the tasks to run (clean pairs: check then bin), one line (for the shell script)
  --smoke   2 tasks, k in 0 and 2, 4 steps, 16 records; every folder gets `_smoke`. R3: the smoke doing side is the smoke arith_bin
            heals relabelled `arith` (k 0 and 2, `smoke_do_rows`), so doexact and the eligibility stage compare real common ids.

Shared cut order and healing are the doing side's: 1.7B uses C1's order (var/results/atlas3/1.7B/order.json), 0.6B stage B's
(var/results/atlas/order.json); both heal q/v LoRA rank 10, 300 steps, whole answer plus end-of-turn (`atlas3_cuts.run_whole_heal`).
Everything lands under var/results/atlas3/e2chk/<size>/ and var/ckpt/atlas3/e2chk/<size>/; nothing existing is written.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import atlas2_cuts as B  # noqa: E402
import atlas3_cuts as C  # noqa: E402
import atlas_cuts as ac  # noqa: E402
import atlas_e2_gate as G  # noqa: E402
import atlas_robust as R  # noqa: E402
import numpy as np  # noqa: E402

from surgery import adapters  # noqa: E402
from surgery.atlas import heal, spec, tasks  # noqa: E402

STAGES = ["sweep", "test", "healedprobe", "bytype", "ftdo", "doexact", "eligibility", "table"]
CHECK_DO = dict(tasks.CHECK_DO)                  # check task -> doing task
BIN_DO = dict(tasks.BIN_DO)                      # bin task -> doing task
BIN_OF = dict(tasks.BIN_OF_CHECK)                # check task -> its bin task
DO_OF = {**CHECK_DO, **BIN_DO}                   # any task of this run -> its doing task
CHECK_ORDER = tuple(CHECK_DO)                    # rule_check, hard_check, arith_check, code_output_check, code_literal_check
DO_KS = (0, 12)                                  # the doing side's saved checkpoints
SMOKE_TASKS = ("arith_check", "arith_bin")
GATE = Path("var/results/atlas3/e2_gate_r2.json")

SIZES = {
    "1.7B": SimpleNamespace(model="Qwen/Qwen3-1.7B", maps=Path("var/results/atlas3/1.7B"), n_layers=28),
    "0.6B": SimpleNamespace(model="Qwen/Qwen3-0.6B", maps=Path("var/results/atlas"), n_layers=28),
}
# ---------------------------------------------------------------- paths and configuration

def sfx_dir(smoke: bool) -> str:
    return "e2chk_smoke" if smoke else "e2chk"


def e2_root(smoke: bool) -> Path:
    return Path("var/results/atlas3") / sfx_dir(smoke)


def size_out(size: str, smoke: bool) -> Path:
    return e2_root(smoke) / size


def size_ckpt(size: str, smoke: bool) -> Path:
    return Path("var/ckpt/atlas3") / sfx_dir(smoke) / size


def size_profile(size: str, smoke: bool) -> SimpleNamespace:
    """A C1-style profile (name qwen17: no gradient checkpointing, C1's recipe) pointed at this size's model, order and folders."""
    s = SIZES[size]
    return SimpleNamespace(name="qwen17", model=s.model, maps=s.maps, out=size_out(size, smoke), ckpt=size_ckpt(size, smoke),
                           acts=Path("var/acts/atlas3") / sfx_dir(smoke) / size, n_layers=s.n_layers, kcut=12, ks=DO_KS)


def check_tasks(smoke: bool, gate_path: Path = GATE, override=None) -> tuple[str, ...]:
    """The tasks to run: the check and bin task of every clean pair of the gate file, pair by pair in registry order (smoke: two
    fixed ones). `override` (--tasks a,b,c or $E2_TASKS) wins and may name check or bin tasks. Fewer than GATE_NEED clean pairs: the
    gate stopped E2."""
    if override:
        bad = [t for t in override if t not in DO_OF]
        if bad:
            raise SystemExit(f"not check or bin tasks: {bad}")
        return tuple(override)
    if smoke:
        return SMOKE_TASKS
    gate = ac.read_json(gate_path) or {}
    clean = [p["check"] for p in gate.get("pairs", []) if p.get("clean")]
    if len(clean) < GATE_NEED:
        raise SystemExit(f"only {len(clean)} clean pairs in {gate_path}; the gate stops E2")
    return tuple(t for c in CHECK_ORDER if c in clean for t in (c, BIN_OF[c]))


def pairs_of(tasks_run) -> list[tuple[str, str | None]]:
    """(check, bin) per check task of the run; bin is None when the run holds no bin task for it."""
    return [(c, BIN_OF[c] if BIN_OF[c] in tasks_run else None) for c in CHECK_ORDER if c in tasks_run]


def make_cfg(size: str, smoke: bool, chk: tuple[str, ...]) -> SimpleNamespace:
    cfg = C.make_cfg(smoke, size_profile(size, smoke))
    ks = (0, 2) if smoke else DO_KS
    return SimpleNamespace(**{**vars(cfg), "tasks": chk, "table_tasks": chk, "ks": ks, "own_tasks": (), "kcut": ks[1],
                              "n_own": 0, "fallback_grid": None, "ledger": not smoke})


# ---------------------------------------------------------------- pure pieces (verdicts, tie, keys)

def select_lr(rows) -> float:
    """The doing side's rule: higher best lrsel among the seed-0 runs; a tie goes to 3e-4 (`heal.pick_lr`)."""
    return heal.pick_lr(rows)


def run_key(task, k, seed, lr) -> str:
    """The resumability key of a heal (and of an ftdo row): the doing side's own."""
    return ac.run_key(task, k, seed, lr)


# R1 eligibility (docs/ATLAS-robustness.md): a task's R counts only if its healed k=0 accuracy is at least chance + 0.15; a pair
# enters a sign count only if the comparison side's R is neither at the ceiling nor at the floor.
ELIG_MARGIN, R_CEIL, R_FLOOR = 0.15, 0.95, 0.05


def r_value(acc_k, acc_0, chance):
    """R as the doing side defines it (mean accuracy at the cut over mean healed k=0, chance-corrected, clipped 0..1);
    None when a number is missing or healed k=0 is under chance + 0.15 (the task does not count)."""
    if acc_k is None or acc_0 is None or chance is None or acc_0 < chance + ELIG_MARGIN - 1e-9:
        return None
    return B.r12_value(acc_k, acc_0, chance)


def interior(r) -> bool:
    """True when R is a number strictly between the floor and the ceiling (so a difference to it can have either sign)."""
    return r is not None and R_FLOOR < r < R_CEIL


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _diff(a, b):
    return None if a is None or b is None else a - b


def _depth(x):
    return None if x.get("depth") is None or x.get("never") else x["depth"]


def _r(x, kcut):
    return r_value(_mean(x["acc"].get(kcut, [])), _mean(x["acc"].get(0, [])), x.get("chance"))


def hit_runs(rows) -> dict:
    """{task: {(k, seed): {id: hit}}} from jsonl rows (id, task, k, seed, lr, hit); a repeated (task, k, seed, id) keeps the last."""
    out: dict = {}
    for r in rows:
        out.setdefault(r["task"], {}).setdefault((r["k"], r["seed"]), {})[r["id"]] = int(r["hit"])
    return out


def _hit_lines(path) -> list[dict]:
    """The parsed rows of a hits file. A malformed final line (an interrupted write) is skipped with a printed warning; a malformed
    line elsewhere is an error."""
    p = Path(path)
    if not p.exists():
        return []
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    rows = []
    for n, ln in enumerate(lines):
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            if n != len(lines) - 1:
                raise
            print(f"WARNING: {p}: ignoring a malformed final line (interrupted write)", flush=True)
    return rows


def read_hits(path) -> dict:
    return hit_runs(_hit_lines(path))


def append_hits(path, task: str, k: int, seed: int, lr: float, recs, hits) -> None:
    """One jsonl line per test record: id, task, k, seed, lr, hit (exact match 0/1). Atomic (R3): the existing rows, minus any of
    this (task, k, seed), plus the new ones go to a temp file in the same folder, then `os.replace`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keep = [r for r in _hit_lines(p) if (r["task"], r["k"], r["seed"]) != (task, k, seed)]
    new = [{"id": r["id"], "task": task, "k": k, "seed": seed, "lr": lr, "hit": int(h)} for r, h in zip(recs, hits)]
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in keep + new))
    os.replace(tmp, p)


def run_hits_complete(path, task: str, k: int, seed: int, ids) -> bool:
    """True when the hits file holds a hit for every id of this (task, k, seed) run."""
    got = read_hits(path).get(task, {}).get((k, seed), {})
    return all(i in got for i in ids)


def ftdo_hit_runs(doc: dict) -> dict:
    """{do task: {(k, seed): {id: hit}}} from ftdo.json (restricted-option argmax hits per record, ids per task)."""
    out: dict = {}
    for v in (doc or {}).get("runs", {}).values():
        ids = (doc.get("ids") or {}).get(v["task"])
        if ids and v.get("ft_correct") and len(ids) == len(v["ft_correct"]):
            out.setdefault(v["task"], {})[(v["k"], v["seed"])] = dict(zip(ids, v["ft_correct"]))
    return out


def common_ids(side_runs, ks) -> list:
    """Test ids scored in every run (k in ks) of every side; [] when a side has no run at some k in ks. `side_runs`: list of
    {(k, seed): {id: hit}}."""
    sets = []
    for runs in side_runs:
        for k in ks:
            ofk = [set(h) for (kk, _), h in runs.items() if kk == k]
            if not ofk:
                return []
            sets += ofk
    return sorted(set.intersection(*sets)) if sets else []


def side_from_hits(runs: dict, ids, chance, seeds=None) -> dict:
    """acc {k: [per-seed accuracy on ids]}, arr {k: [per-seed 0/1 array aligned to ids]} (for the bootstrap), chance."""
    arr: dict = {}
    for (k, s), h in sorted(runs.items()):
        if (seeds is None or s in seeds) and ids and all(i in h for i in ids):
            arr.setdefault(k, []).append(np.array([h[i] for i in ids], float))
    return {"arr": arr, "acc": {k: [float(a.mean()) for a in v] for k, v in arr.items()}, "chance": chance}


def arith_unseen_ids(task_names, read=None) -> set:
    """Test ids of the arith tasks in `task_names` whose expression core (`tasks._arith_core`, the function behind the arith_bin
    `_disjoint_cores`) is in no task's heal train or lrsel, for every task of the list (a pair's check, bin and doing task). Cores
    come from the arith task's records (the check and bin records share its ids); an id with no known core is left out."""
    read = read or spec.read_records
    core = {}
    for sp in ("train", "lrsel", "selection", "discovery", "test"):
        for r in read("arith", sp):
            core[r["id"]] = tasks._arith_core({"records": {"arith": r}})
    keep = None
    for t in task_names:
        fit = {core[r["id"]] for sp in ("train", "lrsel") for r in read(t, sp) if r["id"] in core}
        ids = {r["id"] for r in read(t, "test") if r["id"] in core and core[r["id"]] not in fit}
        keep = ids if keep is None else keep & ids
    return keep or set()


def _r_idx(x: dict, idx, kcut: int, arr="arr", chance="chance"):
    def m(k):
        a = (x.get(arr) or {}).get(k)
        return float(np.mean([s[idx].mean() for s in a])) if a else None
    return r_value(m(kcut), m(0), x.get(chance))


def boot_dr(c: dict, b: dict | None, d: dict, kcut: int, n_boot: int | None = None, seed: int | None = None) -> dict:
    """Record-bootstrap 95% interval (2.5 / 97.5 percentiles over resamples where both R's are defined) of each pair difference:
    resample the common test ids, recompute the seed-mean accuracies and R's. Needs the sides' `arr` (and the doing side's `ft_arr`).
    Reported without a bar."""
    n_boot, seed = N_BOOT if n_boot is None else n_boot, SEED if seed is None else seed
    first = next((a[0] for a in (c.get("arr") or {}).values() if a), None)
    if first is None or not len(first):
        return {}
    n = len(first)
    idxs = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    vals: dict = {"dR_kind": [], "dR_fmt": [], "dR_cd": [], "dR_ft": []}
    has_b = bool(b and b.get("arr"))
    for idx in idxs:
        rc, rd = _r_idx(c, idx, kcut), _r_idx(d, idx, kcut)
        rb = _r_idx(b, idx, kcut) if has_b else None
        rf = _r_idx(d, idx, kcut, "ft_arr", "ft_chance") if d.get("ft_arr") else None
        for name, v in (("dR_kind", _diff(rc, rb)), ("dR_fmt", _diff(rb, rd)), ("dR_cd", _diff(rc, rd)), ("dR_ft", _diff(rc, rf))):
            if v is not None:
                vals[name].append(v)
    return {name: ({"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)), "n_defined": len(v)} if v else None)
            for name, v in vals.items()} | {"n_boot": n_boot, "seed": seed, "n_records": n}


def build_pair(check: str, bin_, do: str, c: dict, b: dict | None, d: dict, kcut: int = 12, boot: bool = True) -> dict:
    """One pair, every number on the same (common) test ids. `c` (check) and `b` (bin, or None): depth, never, acc {k: [per seed]},
    chance (and arr, the per-record hits, for the bootstrap). `d` (doing): depth, never, acc (exact), ft {k: [...]} (restricted-option
    argmax), ft_chance, chance (and arr / ft_arr).
    dR_kind = R(check) - R(bin), dR_fmt = R(bin) - R(do), dR_cd = R(check) - R(do), dR_ft = R(check) - R_ft(do). The ok_* flags say
    whether the pair enters a sign count: both R's defined and the subtracted side interior. dR_boot holds each difference's 95%
    record-bootstrap interval."""
    b = b or {"acc": {}, "chance": None}
    rc, rb, rd = _r(c, kcut), _r(b, kcut), _r(d, kcut)
    rf = r_value(_mean(d.get("ft", {}).get(kcut, [])), _mean(d.get("ft", {}).get(0, [])), d.get("ft_chance"))
    dc, db, dd = _depth(c), _depth(b), _depth(d)
    out = {"check": check, "bin": bin_, "do": do, "depth_check": dc, "depth_bin": db, "depth_do": dd,
           "dd_cb": _diff(dc, db), "dd_bd": _diff(db, dd), "dd_cd": _diff(dc, dd),
           "R_check": rc, "R_bin": rb, "R_do": rd, "R_do_ft": rf,
           "dR_kind": _diff(rc, rb), "dR_fmt": _diff(rb, rd), "dR_cd": _diff(rc, rd), "dR_ft": _diff(rc, rf),
           "ok_kind": rc is not None and interior(rb), "ok_fmt": rb is not None and interior(rd),
           "ok_cd": rc is not None and interior(rd),
           "R_do_selection": d.get("r_selection"), "acc_check": {k: _mean(v) for k, v in c["acc"].items()},
           "acc_bin": {k: _mean(v) for k, v in b["acc"].items()}, "acc_do": {k: _mean(v) for k, v in d["acc"].items()},
           "acc_do_ft": {k: _mean(v) for k, v in d.get("ft", {}).items()},
           "acc_do_full": d.get("acc_full"), "chance_do_full": d.get("chance_full"),
           "chance_check": c.get("chance"), "chance_bin": b.get("chance"), "chance_do": d.get("chance"),
           "chance_do_ft": d.get("ft_chance"), "depth_boot": {"check": c.get("boot"), "bin": b.get("boot")}}
    out["dR_boot"] = boot_dr(c, b, d, kcut) if boot else {}
    return out


def pair_from_hits(check: str, bin_, do: str, rh: dict, dh: dict, fh: dict, answers: dict, depths: dict, kcut: int = 12,
                   seeds=None, unseen=None, boot: bool = True, do_extra=None) -> dict:
    """A pair from per-record hits: `rh` check and bin runs, `dh` doing runs (doexact), `fh` doing restricted-option runs (ftdo),
    each {task: {(k, seed): {id: hit}}}; `answers` {task: {id: answer}} (for chance); `depths` {task: depth/never/boot dict}. The common
    ids are those scored in every run of the check, bin and doing side at k 0 and kcut (`seeds` limits the seeds). `unseen` (arith)
    narrows the scored ids to those in it; the all-record numbers go beside them as `all_records`."""
    ks = (0, kcut)
    pick = lambda h, t: {key: v for key, v in h.get(t, {}).items() if key[0] in ks and (seeds is None or key[1] in seeds)}  # noqa: E731
    cr, br, dr, fr = pick(rh, check), (pick(rh, bin_) if bin_ else None), pick(dh, do), pick(fh, do)
    ids_all = common_ids([cr, dr] + ([br] if bin_ else []), ks)

    def build(ids, bt):
        ch = lambda t: spec.chance([answers[t][i] for i in ids]) if ids else None  # noqa: E731
        c = {**side_from_hits(cr, ids, ch(check), seeds), **depths.get(check, {})}
        b = {**side_from_hits(br, ids, ch(bin_), seeds), **depths.get(bin_, {})} if bin_ else None
        d = {**side_from_hits(dr, ids, ch(do), seeds), **depths.get(do, {}), **(do_extra or {})}
        ft = side_from_hits(fr, ids, ch(do), seeds) if fr else {"acc": {}, "arr": {}}
        d.update(ft=ft["acc"], ft_arr=ft["arr"], ft_chance=ch(do) if ft["acc"] else None)
        return build_pair(check, bin_, do, c, b, d, kcut, boot=bt)

    ids = ids_all if unseen is None else [i for i in ids_all if i in unseen]
    out = build(ids, boot)
    out.update(n_common=len(ids_all), n_scored=len(ids), scored_on="unseen-expression records" if unseen is not None else "common ids")
    if unseen is not None:
        a = build(ids_all, False)
        out["all_records"] = {k: a[k] for k in ("R_check", "R_bin", "R_do", "R_do_ft", "dR_kind", "dR_fmt", "dR_cd", "dR_ft",
                                               "ok_kind", "ok_fmt", "ok_cd")} | {"n": len(ids_all)}
    return out


# ================================================================ VERDICTS AND BARS (one swappable section)
# Everything that encodes a bar or a reading lives here: constants, E2-A, E2-B, E2-F, E2-D, E2-L, claim 6, E2-C. Pairs come in as
# dicts from build_pair (keys dd_*, dR_*, ok_*); nothing below touches files or models.

N_PERM, SEED, N_BOOT = 10_000, 0, 1_000
# the bars (docs/ATLAS-robustness.md, "E2 revision R1", unchanged by R2); fixed before any E2 heal
A_SAME, A_EARLY = -2, -5                         # median delta depth (check - bin)
B_EFFECT, B_NONE, B_SPREAD, B_NEED = 0.20, 0.10, 0.25, 3
GATE_NEED = 4                                    # clean pairs the gate file must hold (R1); B_NEED is for E2-B / eligibility
L_MIN = 0.10                                     # mean delta R_ft under this: "mostly answer length"
EPS = 1e-9

WORDING = {
    "same_none": "Checking a proposed answer forms as late as deciding a property of the answer, and survives the cut no better. "
                 "In these five problems, checking did not escape the depth cost.",
    "early_effect": "Checking a proposed answer can be done from earlier layers than deciding a property of the answer, and survives "
                    "better.",
    "same_effect": "At similar read depth, checking survived better than membership, so something beyond depth helps checking.",
    "early_none": "Checking reads earlier than membership but survives no better, so within these pairs read depth does not track "
                  "survival. (This qualifies claim 1, and the paper says so.)",
    "other": "We could not separate task kind from depth; most of the judging advantage in our earlier tasks runs through depth.",
}
FORMAT_NOTE = "Two-answer versions of the same problem survive better than the full answer."
FORMAT_NOTE_NONE = "Two-answer versions of the same problem survive no better than the full answer."
BARS_NOTE = ("Bars are descriptive labels on few pairs; a 'no effect' label is not evidence of absence. Check vs bin compares "
             "checking a proposed answer with deciding set membership, not with full doing.")


def e2a(ddepths) -> dict:
    """Median depth gap, check - bin: same or later if >= -2; check reads early if <= -5; otherwise partly earlier."""
    vals = [d for d in ddepths if d is not None]
    if not vals:
        return {"median": None, "n": 0, "label": "undetermined"}
    med = float(statistics.median(vals))
    label = "same or later" if med >= A_SAME - EPS else ("check reads early" if med <= A_EARLY + EPS else "partly earlier")
    return {"median": med, "n": len(vals), "label": label}


def effect_bars(pairs: list[dict], dkey: str, okkey: str, effect: str, none: str) -> dict:
    """The E2-B bars on pairs[dkey] over the eligible pairs (okkey and the difference defined). Effect: at least 3 eligible pairs,
    mean >= +0.20 and the difference > 0 in all but at most one. None: at least 3, |mean| < 0.10 and every |difference| < 0.25.
    Fewer than 3 eligible pairs: not testable. Otherwise unclear."""
    vals = [p[dkey] for p in pairs if p.get(okkey) and p.get(dkey) is not None]
    n = len(vals)
    mean = _mean(vals)
    pos = sum(1 for x in vals if x > 0)
    parts = {"n_eligible": n, "n_positive": pos, "max_abs": max((abs(x) for x in vals), default=None),
             "mean_ge_0.20": mean is not None and mean >= B_EFFECT - EPS, "positive_ok": n >= B_NEED and pos >= n - 1,
             "mean_lt_0.10": mean is not None and abs(mean) < B_NONE - EPS,
             "spread_ok": bool(vals) and all(abs(x) < B_SPREAD - EPS for x in vals)}
    if n < B_NEED:
        label = "not testable"
    elif parts["mean_ge_0.20"] and parts["positive_ok"]:
        label = effect
    elif parts["mean_lt_0.10"] and parts["spread_ok"]:
        label = none
    else:
        label = "unclear"
    return {"mean": mean, "n_pairs": len(pairs), "n_eligible": n, "parts": parts, "values": vals, "label": label}


def e2b(pairs: list[dict]) -> dict:
    """Kind at matched format: R12(check) - R12(bin) over eligible pairs."""
    return effect_bars(pairs, "dR_kind", "ok_kind", "kind has its own effect", "no kind effect")


def e2f(pairs: list[dict]) -> dict:
    """Format, reported with the same bars: R12(bin) - R12(do). 'Format helps' = mean >= +0.20 and > 0 in all but one."""
    return effect_bars(pairs, "dR_fmt", "ok_fmt", "format helps", "no format effect")


def e2d(pairs: list[dict]) -> dict:
    """Descriptive only (R2): Spearman rho of the depth gap against the survival gap over every check/bin/do pair difference
    (check - bin, bin - do, check - do) where both are defined. No p: the three differences of a pair are dependent."""
    pts = [(p[a], p[b]) for p in pairs for a, b in (("dd_cb", "dR_kind"), ("dd_bd", "dR_fmt"), ("dd_cd", "dR_cd"))
           if p.get(a) is not None and p.get(b) is not None]
    obs = R.spearman_plain([a for a, _ in pts], [b for _, b in pts]) if len(pts) >= 4 else None
    return {"n": len(pts), "rho": obs, "p": None, "descriptive": True}


def e2l(pairs: list[dict]) -> dict:
    """E2-L (R2), an annotation on the check - do comparison only; it never touches E2-B's label. Where the mean dR_cd shows a
    check advantage (>= +0.20) and the restricted-option difference dR_ft is under +0.10, the advantage is 'mostly answer length'.
    'not run' when no pair has a dR_ft (ftdo missing)."""
    use = [p for p in pairs if p.get("dR_ft") is not None and p.get("ok_cd")]
    if not any(p.get("dR_ft") is not None for p in pairs):
        return {"run": False, "n": 0, "mean_dR_cd": None, "mean_dR_ft": None, "label": "not run (ftdo missing)"}
    cd, ft = _mean([p["dR_cd"] for p in use]), _mean([p["dR_ft"] for p in use])
    if cd is None:
        label = "no eligible check - do pair"
    elif cd < B_EFFECT - EPS:
        label = f"not triggered (no check - do advantage: mean dR_cd {cd:.2f})"
    elif ft < L_MIN - EPS:
        label = "the check - do advantage is mostly answer length"
    else:
        label = "length control passed (the check - do advantage survives)"
    return {"run": True, "n": len(use), "mean_dR_cd": cd, "mean_dR_ft": ft, "label": label}


def claim6(a_label: str, b_label: str, f_label: str = "") -> dict:
    """The wording table on E2-A and E2-B (E2-L plays no part, R2; R3 narrowed the texts); 'format helps' and 'no format effect' add their sentence."""
    effect = b_label == "kind has its own effect"
    late = a_label == "same or later"
    early = a_label == "check reads early"
    if late and b_label == "no kind effect":
        key = "same_none"
    elif early and effect:
        key = "early_effect"
    elif late and effect:
        key = "same_effect"
    elif early and b_label == "no kind effect":
        key = "early_none"
    else:
        key = "other"
    note = FORMAT_NOTE if f_label == "format helps" else (FORMAT_NOTE_NONE if f_label == "no format effect" else "")
    return {"row": key, "text": WORDING[key] + (" " + note if note else ""), "format_note": bool(note)}


def e2c(rows24: list[dict], rows_new: list[dict], n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """The 24 tasks, then the 24 plus the clean check and bin tasks: D1-A's weighted within-kind rho of L_on_healed against R and
    E1-c's partial rho(kind, R | depth), both with a permutation p. Rows: kind, depth, r."""
    def block(rows):
        depth, r, kinds = [x["depth"] for x in rows], [x["r"] for x in rows], [x["kind"] for x in rows]
        n_sel = sum(1 for x in rows if x.get("r_source", "test") == "selection")
        if len(rows) < 4:
            return {"n": len(rows), "rho": None, "p": None, "partial": None, "partial_p": None, "partial_verdict": "undetermined",
                    "descriptive": bool(n_sel)}
        a = B.h2a_test(depth, r, kinds, n_perm, seed)
        c = R.e1c(depth, r, kinds, n_perm, seed)
        # R3: a block that mixes in selection-split R gets no verdict; its numbers stay, marked descriptive
        verdict = f"incomplete ({n_sel} of {len(rows)} rows on the selection split)" if n_sel else c["verdict"]
        return {"n": len(rows), "rho": a["rho"], "p": a["p"], "by_kind": a["by_kind"], "rho_kind_depth": c["rho_kind_depth"],
                "rho_kind_R": c["rho_kind_r12"], "partial": c["partial"], "partial_p": c["p"], "partial_verdict": verdict,
                "descriptive": bool(n_sel), "n_selection_rows": n_sel}
    return {"tasks24": block(rows24), "with_new": block(rows24 + rows_new)}


def verdicts(pairs: list[dict]) -> dict:
    """E2-A on the pairs eligible for E2-B (check vs do: eligible for check - do)."""
    a = e2a([p["dd_cb"] for p in pairs if p["ok_kind"]])
    a_cd = e2a([p["dd_cd"] for p in pairs if p["ok_cd"]])
    b = e2b(pairs)
    f = e2f(pairs)
    return {"E2-A": a, "E2-A_check_vs_do": a_cd, "E2-B": b, "E2-F": f, "E2-D": e2d(pairs), "E2-L": e2l(pairs),
            "claim6": claim6(a["label"], b["label"], f["label"])}


def boot_l_on(correct, answers, preds, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Bootstrap of L_on_healed over records. correct: (layers, N) 0/1 out-of-fold probe hits; answers, preds: the healed model's
    labels and first-token predictions per record. Each resample recomputes the probe curve, the chance and the healed accuracy,
    then `l_on_healed`. Reports the 2.5 / 97.5 percentiles of the defined values (a 'never' counts as the curve length) and how many
    resamples were defined."""
    from atlas2_healed import l_on_healed
    correct, answers, preds = np.asarray(correct, float), np.asarray(answers), np.asarray(preds)
    n = correct.shape[1]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        lo, _ = l_on_healed(correct[:, idx].mean(1).tolist(), spec.chance(answers[idx].tolist()), float((preds[idx] == answers[idx]).mean()))
        if lo is not None:
            vals.append(lo)
    if not vals:
        return {"n_boot": n_boot, "seed": seed, "n_defined": 0, "lo": None, "hi": None}
    return {"n_boot": n_boot, "seed": seed, "n_defined": len(vals), "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5))}


def near_type(task: str, rec: dict, do_rec: dict) -> str:
    """The near-miss type and distance of one check record (free report): 'correct', or by task: rule index of the proposed rule,
    hops short / past, signed difference to the answer, digit distance."""
    if rec["meta"]["correct"]:
        return "correct"
    p, a = rec["meta"]["proposed"], rec["meta"]["do_answer"]
    if task == "rule_check":
        act = G.rule_lists(G.do_part(rec))[0]
        return f"rule {act.index(G.proposed_text(rec)) + 1}" if G.proposed_text(rec) in act else "other"
    if task == "hard_check":
        act = G.rule_lists(G.do_part(rec))[0]
        i = act.index(G.proposed_text(rec)) + 1 if G.proposed_text(rec) in act else 0
        return "one hop short" if i < do_rec["meta"]["decided_by"] else "one hop past"
    if task == "arith_check":
        return f"off by {p - a:+d}"
    dist = abs(p - a)
    return f"distance {dist}" if dist < 3 else "distance 3+"

# ================================================================ end of verdicts and bars


# ---------------------------------------------------------------- stages that need the model

def stage_healedprobe(ctx, task: str) -> int:
    """`L_on_healed` of the task's seed-0, k=0 healed model at the seed-0 winner lr, as `atlas3_cuts.stage_healedprobe` defines it
    (merged LoRA, `measure.build_store` over the discovery records, last-prompt-token probe), plus (R1) a bootstrap interval of
    L_on_healed over records (1,000 resamples, seed 0, from the probe's out-of-fold hits; the probe records are the discovery split,
    as D1's) and, for a check task, the free report: a probe for meta.do_answer on the same activations. Written to
    healed_probe_<task>.json; resumes by skipping when that file exists."""
    import atlas_map
    from atlas2_healed import l_on_healed

    from surgery import probes
    from surgery.atlas import measure
    cfg = ctx.cfg
    final = C.healed_probe_path(cfg, task)
    if final.exists():
        print(f"healedprobe {task}: {final} exists; nothing to do", flush=True)
        return 0
    runs = ac.read_merged("runs", cfg.sfx, "runs")
    rows = [r for r in ac.collect(runs, [task], [0], "per_k").get(task, {}).get(0, []) if r["seed"] == 0]
    if not rows:
        print(f"healedprobe {task}: no healed k=0 seed-0 run; skipped", flush=True)
        return 1
    row = rows[0]
    recs = ctx.records(task, "discovery")
    if cfg.probe_records:
        recs = recs[:cfg.probe_records]
    store = ac.ACTS / f"{task}_healed_probe{cfg.sfx}"
    model, handle = ac.rebuild_healed(ctx, row)
    adapters.merge_lora(handle)
    boot = dprobe = None
    try:
        meta = measure.build_store(model, ctx.tokenizer, task, recs, store, batch_size=ac.EVAL_BS)
        answers = meta["answers"]
        preds = np.asarray(meta["pred"])
        acc = float(np.mean(preds == np.asarray(answers)))
        X = np.load(store / "resid.npy", mmap_mode="r")
        p = atlas_map._probe(X, answers, meta["template_ids"], np.arange(len(answers)), cfg.boots, seed=0)
        if p:
            a_arr, g_arr = np.asarray(answers), np.asarray(meta["template_ids"])
            hits = np.stack([probes._oof(np.asarray(X[:, i], dtype=np.float32), a_arr, g_arr, "label", i, False) == a_arr
                             for i in range(X.shape[1])])
            boot = boot_l_on(hits, a_arr, preds, 20 if cfg.smoke else N_BOOT, SEED)
            if task in CHECK_DO:
                dans = [r["meta"]["do_answer"] for r in recs]
                d = atlas_map._probe(X, dans, meta["template_ids"], np.arange(len(dans)), 20, seed=0)
                dprobe = None if d is None else {k: d[k] for k in ("chance", "peak", "peak_layer", "L_on", "curve")}
        del X
    finally:
        import shutil
        shutil.rmtree(store, ignore_errors=True)
    chance = spec.chance(answers)
    curve = p["curve"] if p else None
    lo, never = l_on_healed(curve, chance, acc) if curve else (None, False)
    res = {"task": task, "smoke": cfg.smoke, "lr": row["lr"], "ckpt": row["ckpt"], "k": 0, "seed": 0, "n": len(answers),
           "chance": chance, "healed_accuracy": acc, "curve": curve, "lo": p["lo"] if p else None,
           "hi": p["hi"] if p else None, "L_on_healed": lo, "never": never, "boot": boot, "do_answer_probe": dprobe,
           "L_on_unmodified": C.map_doc(task, cfg.sfx).get("L_on"),
           "scoring": "option first-token accuracy", "device": str(ctx.device), "ts": ac.now()}
    ac.write(final, res)
    del model
    ac.release(ctx)
    print(f"healedprobe {task}: healed acc {acc:.3f} chance {chance:.3f} L_on_healed {lo} never {never} boot {boot} -> {final}",
          flush=True)
    return 0


def stage_bytype(ctx, size: str) -> int:
    """Free report: check accuracy by near-miss type and distance, for each check task's seed-0 winner-lr model at k 0 and 12, on the
    test split. Written to bytype.json: {task: {k: {type: [n, correct]}}}."""
    cfg = ctx.cfg
    path = ac.OUT / f"bytype{cfg.sfx}.json"
    doc = ac.read_json(path) or {"size": size, "smoke": cfg.smoke, "tasks": {}}
    runs = ac.read_merged("runs", cfg.sfx, "runs")
    for task in (t for t in cfg.tasks if t in CHECK_DO):
        recs = ctx.records(task, "test")
        do_by_id = {r["id"]: r for r in spec.read_records(CHECK_DO[task], "test")}   # full split: smoke truncates ctx.records
        coll = ac.collect(runs, [task], cfg.ks, "per_k").get(task, {})
        for k in cfg.ks:
            rows = [r for r in coll.get(k, []) if r["seed"] == 0]
            if not rows or str(k) in doc["tasks"].get(task, {}):
                continue
            model, handle = ac.rebuild_healed(ctx, rows[0])
            adapters.merge_lora(handle)
            _, preds = heal.accuracy_and_preds(model, ctx.tokenizer, task, recs, ac.EVAL_BS)
            by: dict = {}
            for r, pr in zip(recs, preds):
                n_c = by.setdefault(near_type(task, r, do_by_id[r["id"]]), [0, 0])
                n_c[0] += 1
                n_c[1] += int(pr == r["answer"])
            doc["tasks"].setdefault(task, {})[str(k)] = by
            ac.write(path, doc)
            del model
            ac.release(ctx)
            print(f"bytype {task} k={k}: {by}", flush=True)
    return 0


def exact_hits(ctx, row: dict, task: str):
    """(exact-match rate, test records, per-record 0/1 hits) of the healed model of a sweep row, as `atlas3_cuts.test_exact`
    scores it (greedy, same max_new_tokens and batch size), but keeping each record's result."""
    model, handle = ac.rebuild_healed(ctx, row)
    adapters.merge_lora(handle)
    recs = ctx.records(task, "test")
    em, texts = heal.exact_match_scores(model, ctx.tokenizer, task, recs, max_new_tokens=C.max_new_tokens(ctx, task),
                                        batch_size=ac.EVAL_BS)
    del model
    ac.release(ctx)
    return em, recs, [int(heal.exact_match(t, heal.answer_text(task, r))) for t, r in zip(texts, recs)]


def stage_test(ctx) -> int:
    """`atlas3_cuts.stage_test` for the check and bin tasks (no own-order arm), and every pass also saves its per-record hits to
    test_hits.jsonl (id, task, k, seed, lr, hit). test_results.json and the ledger are as C1's; resumes by test_results key when the run's hits are complete (R3)."""
    cfg = ctx.cfg
    runs = ac.read_merged("runs", cfg.sfx, "runs")
    path = ac.OUT / f"test_results{cfg.sfx}.json"
    hpath = ac.OUT / f"test_hits{cfg.sfx}.jsonl"
    doc = ac.read_json(path) or {"results": {}}
    ledger = ac.OUT / "test_ledger.jsonl"
    coll = ac.collect(runs, cfg.tasks, cfg.ks, "per_k")
    for task in cfg.tasks:
        for k, rows in coll.get(task, {}).items():
            for row in rows:
                if row["key"] in doc["results"] and run_hits_complete(hpath, task, k, row["seed"],
                                                                      [r["id"] for r in ctx.records(task, "test")]):
                    continue
                em, recs, hits = exact_hits(ctx, row, task)
                append_hits(hpath, task, k, row["seed"], row["lr"], recs, hits)
                doc["results"][row["key"]] = {"task": task, "k": k, "seed": row["seed"], "lr": row["lr"], "names": [f"healed_k{k}"],
                                              "n": len(recs), "accuracy": em,
                                              "chance": spec.chance([r["answer"] for r in recs]), "ts": ac.now()}
                ac.write(path, doc)
                if cfg.ledger:
                    with ledger.open("a") as fh:
                        fh.write(json.dumps({"key": row["key"], "names": [f"healed_k{k}"], "task": task, "k": k, "seed": row["seed"],
                                             "n": len(recs), "accuracy": em, "ts": ac.now()}, sort_keys=True) + "\n")
                print(f"test {row['key']} (healed_k{k}): {em:.3f}", flush=True)
    return 0


def stage_doexact(ctx, size: str, seed: int | None = None) -> int:
    """The doing side's saved shared-order checkpoints (C1's winner lr and seeds at 1.7B; stage B's whole arm at 0.6B; never the
    own-order models) re-scored by whole-answer exact match on the test split, per record: doexact_hits.jsonl, and doexact.json
    {runs: {key: task, k, seed, lr, n, accuracy, chance}}. The pair's common ids are taken from these hits at the table. `seed`
    limits the seeds (stage 1: 0). Resumes by key, but only when the hits file holds every test id of the run (R3); else re-scores."""
    cfg = ctx.cfg
    path = ac.OUT / f"doexact{cfg.sfx}.json"
    hpath = ac.OUT / f"doexact_hits{cfg.sfx}.jsonl"
    doc = ac.read_json(path) or {"size": size, "smoke": cfg.smoke, "runs": {}}
    do_tasks = ("arith",) if cfg.smoke else tuple(dict.fromkeys(DO_OF[t] for t in cfg.tasks))
    for row in doing_rows(ctx, size, do_tasks):
        if seed is not None and row["seed"] != seed:
            continue
        key = run_key(row["task"], row["k"], row["seed"], row["lr"])
        if key in doc["runs"] and run_hits_complete(hpath, row["task"], row["k"], row["seed"],
                                                    [r["id"] for r in ctx.records(row["task"], "test")]):
            print(f"skip {key}: already done", flush=True)
            continue
        em, recs, hits = exact_hits(ctx, row, row["task"])
        append_hits(hpath, row["task"], row["k"], row["seed"], row["lr"], recs, hits)
        doc["runs"][key] = {"task": row["task"], "k": row["k"], "seed": row["seed"], "lr": row["lr"], "ckpt": row["ckpt"],
                            "n": len(recs), "accuracy": em, "chance": spec.chance([r["answer"] for r in recs]), "ts": ac.now()}
        ac.write(path, doc)
        print(f"doexact {key}: exact {em:.3f} (n {len(recs)})", flush=True)
    return 0


def stage_ftdo(ctx, size: str) -> int:
    """E2-L: restricted-option argmax accuracy (first token of the option) on the test records for each saved doing-side heal
    (read-only), per record (ids and ft_correct). The exact-match re-score of the same checkpoints is `doexact`. Both sides of
    E2-L score by argmax over the restricted options; only `rule` and `hard` have multi-token answers."""
    cfg = ctx.cfg
    path = ac.OUT / f"ftdo{cfg.sfx}.json"
    doc = ac.read_json(path) or {"size": size, "smoke": cfg.smoke, "runs": {}}
    do_tasks = ("arith",) if cfg.smoke else tuple(dict.fromkeys(DO_OF[t] for t in cfg.tasks))
    doc["answer_tokens"] = {t: C.longest_answer_tokens(ctx.tokenizer, t) for t in do_tasks}
    multi = [t for t, n in doc["answer_tokens"].items() if n > 1]
    print(f"ftdo: longest answer in tokens {doc['answer_tokens']}; multi-token (first-token scoring is a partial reading): "
          f"{multi or 'none'}", flush=True)
    doc["ids"] = {t: [r["id"] for r in ctx.records(t, "test")] for t in do_tasks}
    ac.write(path, doc)
    for row in doing_rows(ctx, size, do_tasks):
        key = run_key(row["task"], row["k"], row["seed"], row["lr"])
        if key in doc["runs"]:
            print(f"skip {key}: already done", flush=True)
            continue
        task = row["task"]
        recs = ctx.records(task, "test")
        model, handle = ac.rebuild_healed(ctx, row)
        adapters.merge_lora(handle)
        ft, preds = heal.accuracy_and_preds(model, ctx.tokenizer, task, recs, ac.EVAL_BS)
        res = {"task": task, "k": row["k"], "seed": row["seed"], "lr": row["lr"], "ckpt": row["ckpt"], "n": len(recs),
               "ft_accuracy": ft, "ft_chance": spec.chance([r["answer"] for r in recs]),
               "ft_correct": [int(p == r["answer"]) for p, r in zip(preds, recs)], "ts": ac.now()}
        doc["runs"][key] = res
        ac.write(path, doc)
        del model
        ac.release(ctx)
        print(f"ftdo {key}: first-token {ft:.3f}", flush=True)
    return 0


def smoke_do_rows(runs: dict, ks) -> list[dict]:
    """Smoke only (R3): the smoke run has no heals of the doing task `arith`, so its stand-in doing side is the smoke `arith_bin` heals
    (winner lr, k in `ks`, every seed), relabelled task `arith`. They exist at the smoke cut, so doexact scores them at k 0 and the
    cut and the eligibility stage compares real common ids instead of an empty set."""
    coll = ac.collect(runs, ["arith_bin"], ks, "per_k").get("arith_bin", {})
    return [{**r, "task": "arith"} for k in ks for r in coll.get(k, [])]


def doing_rows(ctx, size: str, do_tasks) -> list[dict]:
    """The doing side's heals to re-score: `do_rows` for a real run; in smoke the `smoke_do_rows` stand-in."""
    if ctx.cfg.smoke:
        return smoke_do_rows(ac.read_merged("runs", ctx.cfg.sfx, "runs"), ctx.cfg.ks)
    return do_rows(size, do_tasks)


def do_rows(size: str, do_tasks) -> list[dict]:
    """The doing side's saved heals at k 0 and 12 (rows with seed, lr, ckpt): C1's winner-lr rows at 1.7B; stage B's whole-answer
    arm at 0.6B (one lr per task and k, the first-token sweep winner)."""
    if size == "1.7B":
        runs = B.merged_dir(C.C1_OUT, "runs", "", "runs")
        coll = ac.collect(runs, do_tasks, DO_KS, "per_k")
        return [r for t in do_tasks for k in DO_KS for r in coll.get(t, {}).get(k, [])]
    runs = B.merged_dir(Path("var/results/atlas2b"), "whole", "", "runs")
    rows = [r for r in runs.values() if r["task"] in do_tasks and r["k"] in DO_KS]
    return sorted(rows, key=lambda r: (do_tasks.index(r["task"]), r["k"], r["seed"]))


# ---------------------------------------------------------------- table

def _read(p):
    return ac.read_json(p)


def group_acc(entries, field: str) -> dict:
    """{(task, k): ([per-seed accuracies], chance)} from entries with task, k, seed and `field`."""
    out: dict = {}
    for e in sorted(entries, key=lambda e: (e["task"], e["k"], e["seed"])):
        accs, _ = out.setdefault((e["task"], e["k"]), ([], e.get("chance")))
        accs.append(e[field])
    return out


def load_run_side(size: str, smoke: bool, run_tasks) -> dict:
    """Check and bin tasks' depth numbers: L_on_healed, never, bootstrap interval, do_answer probe. (Accuracies come from the
    per-record hits, on the pair's common ids.)"""
    out = size_out(size, smoke)
    sfx = "_smoke" if smoke else ""
    side = {}
    for t in run_tasks:
        probe = _read(out / f"healed_probe_{t}{sfx}.json") or {}
        side[t] = {"depth": probe.get("L_on_healed"), "never": bool(probe.get("never")), "boot": probe.get("boot"),
                   "do_answer_probe": probe.get("do_answer_probe")}
    return side


def shared_order_results(tr: dict) -> dict:
    """Test results without the own-order models: C1's test_results.json also holds `own|...` rows (own-order, not the shared cut
    order the check and bin tasks use); they are never part of the do side."""
    return {key: v for key, v in tr.items() if not str(key).startswith("own|")}


def load_do_side(size: str, smoke: bool, run_tasks) -> dict:
    """The doing twins' existing numbers, keyed by check task: depth (C1 / D1 / stage B probes; read, never recomputed), the
    selection-split R12, and the full-test-split exact-match accuracy (C1's test_results at 1.7B, own| rows excluded; doexact at
    0.6B). The full-split accuracy is a cross-check only: the pair's R values come from the doexact hits on common ids."""
    sfx = "_smoke" if smoke else ""
    if size == "1.7B":
        tr = shared_order_results((_read(C.C1_OUT / "test_results.json") or {}).get("results", {}))
        ex = group_acc([v for v in tr.values() if v["task"] in CHECK_DO.values() and v["k"] in DO_KS], "accuracy")
        sel = {r["task"]: r.get("r12") for r in (_read(C.C1_OUT / "summary.json") or {}).get("rows", [])}
    else:
        dx = (_read(size_out(size, smoke) / f"doexact{sfx}.json") or {}).get("runs", {})
        ex = group_acc([v for v in dx.values() if v["k"] in DO_KS], "accuracy")
        sel = {}
    side = {}
    for t in (t for t in run_tasks if t in CHECK_DO):
        d = CHECK_DO[t]
        if size == "1.7B":
            probe = _read(C.C1_OUT / f"healed_probe_{d}.json") or {}
        else:
            probe = next((x for x in (_read(Path(f"var/results/{r}/healed_{d}.json")) for r in ("atlas2b", "atlas2")) if x), {})
        side[t] = {"depth": probe.get("L_on_healed"), "never": bool(probe.get("never")), "r_selection": sel.get(d),
                   "acc_full": {k: ex[(d, k)][0] for k in DO_KS if (d, k) in ex},
                   "chance_full": next((ex[(d, k)][1] for k in DO_KS if (d, k) in ex), None)}
    return side


def test_r_by_task(results: dict, ks=DO_KS) -> dict:
    """{task: R12} on the test split from test_results-style rows (task, k, seed, accuracy, chance): seed-mean accuracy at k=12
    and at k=0, `own|` rows excluded, D1's R12 (clipped 0..1). Tasks lacking either k are absent."""
    acc: dict = {}
    for v in shared_order_results(results).values():
        if v["k"] in ks:
            a = acc.setdefault(v["task"], {})
            a.setdefault(v["k"], []).append(v["accuracy"])
            a["chance"] = v["chance"]
    return {t: B.r12_value(_mean(a[ks[1]]), _mean(a[ks[0]]), a["chance"]) for t, a in acc.items() if ks[0] in a and ks[1] in a}


def rows24(size: str, smoke: bool = False) -> list[dict]:
    """The 24 tasks' (kind, L_on_healed, R) from D1's own readers, usable rows only. R is the test-split R12 where a test result
    exists (1.7B: C1's test_results; 0.6B: the doexact results, then stage B's test_results); a task without one keeps D1's
    selection-split R12 and carries r_source 'selection' (R2: labelled, not hidden)."""
    import atlas3_extra as X
    rows = X.d1_rows_17(X.C1) if size == "1.7B" else X.d1_rows_06(ROOT)
    usable, _ = X.d1_split(rows)
    sfx = "_smoke" if smoke else ""
    if size == "1.7B":
        src = (_read(C.C1_OUT / "test_results.json") or {}).get("results", {})
    else:
        src = {**((_read(Path("var/results/atlas2b/test_results.json")) or {}).get("results", {})),
               **{f"doexact|{k}": v for k, v in ((_read(size_out(size, smoke) / f"doexact{sfx}.json") or {}).get("runs", {})).items()}}
        # a doexact run replaces stage B's test rows of the same task and k
        have = {(v["task"], v["k"], v["seed"]) for k, v in src.items() if k.startswith("doexact|")}
        src = {k: v for k, v in src.items() if k.startswith("doexact|") or (v["task"], v["k"], v["seed"]) not in have}
    test = test_r_by_task(src)
    return [{"task": r["task"], "kind": r["kind"], "depth": r["L_on_healed"],
             "r": test.get(r["task"], r["r12"]), "r_source": "test" if r["task"] in test else "selection"} for r in usable]


def answers_of(task: str) -> dict:
    return {r["id"]: r["answer"] for r in spec.read_records(task, "test")}


def size_hits(size: str, smoke: bool) -> tuple[dict, dict, dict]:
    """(check and bin hits, doexact hits, ftdo hits), each {task: {(k, seed): {id: hit}}}."""
    root = size_out(size, smoke)
    sfx = "_smoke" if smoke else ""
    return (read_hits(root / f"test_hits{sfx}.jsonl"), read_hits(root / f"doexact_hits{sfx}.jsonl"),
            ftdo_hit_runs(_read(root / f"ftdo{sfx}.json") or {}))


def size_pairs(size: str, smoke: bool, chk, kcut: int, seeds=None, boot: bool = True) -> tuple[list[dict], dict]:
    """The pairs of this run on common ids (arith: unseen-expression records, all-record numbers beside), and the run-side
    probes."""
    rs, ds = load_run_side(size, smoke, chk), load_do_side(size, smoke, chk)
    rh, dh, fh = size_hits(size, smoke)
    pairs = []
    for c, b in pairs_of(chk):
        do = CHECK_DO[c]
        names = [t for t in (c, b, do) if t]
        answers = {t: answers_of(t) for t in names}
        depths = {c: rs[c], do: {k: v for k, v in ds[c].items() if k in ("depth", "never")}}
        if b:
            depths[b] = rs[b]
        extra = {k: ds[c][k] for k in ("r_selection", "acc_full", "chance_full")}
        unseen = arith_unseen_ids(names) if do == "arith" else None
        pairs.append(pair_from_hits(c, b, do, rh, dh, fh, answers, depths, kcut, seeds, unseen, boot, extra))
    return pairs, rs


def size_summary(size: str, smoke: bool, chk, kcut: int, n_perm: int = N_PERM, with_c: bool = True) -> dict:
    pairs, rs = size_pairs(size, smoke, chk, kcut)
    root = size_out(size, smoke)
    sfx = "_smoke" if smoke else ""
    ft_doc = _read(root / f"ftdo{sfx}.json") or {}
    out = {"size": size, "kcut": kcut, "pairs": pairs, "answer_tokens": ft_doc.get("answer_tokens"), **verdicts(pairs),
           "bytype": (_read(root / f"bytype{sfx}.json") or {}).get("tasks"),
           "do_answer_probe": {c: rs[c].get("do_answer_probe") for c, _ in pairs_of(chk)}}
    if with_c:
        new = []
        for p in pairs:
            new += [{"task": p["check"], "kind": "appraise", "depth": p["depth_check"], "r": p["R_check"], "r_source": "test"}]
            if p["bin"]:
                new += [{"task": p["bin"], "kind": "execute", "depth": p["depth_bin"], "r": p["R_bin"], "r_source": "test"}]
        new = [x for x in new if x["depth"] is not None and x["r"] is not None]
        r24 = rows24(size, smoke)
        out["E2-C"] = {**e2c(r24, new, n_perm), "tasks_added": [x["task"] for x in new],
                       "selection_split_tasks": [x["task"] for x in r24 if x["r_source"] == "selection"],
                       "r_basis": "test split (the 24: test_results / doexact where they exist; new tasks: common-id R); a block with any "
                                 "selection-split row has no verdict (R3)"}
    return out


def stage1_table(smoke: bool, seeds=(0,), force=None) -> dict:
    """Stage 1's eligibility at 1.7B on the pair's common ids (seed 0 of every side), no depth needed. `force` (smoke only) replaces
    the eligible count to exercise both branches."""
    chk = check_tasks(smoke, override=None)
    pairs, _ = size_pairs("1.7B", smoke, chk, 2 if smoke else 12, seeds=seeds)
    n = sum(1 for p in pairs if p["ok_kind"])
    n_used = n if force is None else int(force)
    return {"smoke": smoke, "size": "1.7B", "seeds": list(seeds), "pairs": pairs, "n_eligible_E2B": n, "forced": force is not None,
            "n_used": n_used, "need": B_NEED, "go": n_used >= B_NEED}


def render_stage1(z: dict) -> str:
    md = [f"# E2 stage 1: eligibility at Qwen3-1.7B, seed {z['seeds']}, common test ids{' (smoke)' if z['smoke'] else ''}", "",
          f"E2-B-eligible pairs: {z['n_eligible_E2B']} of {len(z['pairs'])}" + (f"; FORCED to {z['n_used']} (smoke)" if z["forced"] else "")
          + f"; need {z['need']}: **{'STAGE1_GO' if z['go'] else 'STAGE1_STOP'}**", "",
          "| check | n common | n scored | R do | R bin | R check | dR_kind [95%] | E2-B eligible |", "|" + "---|" * 8]
    for p in z["pairs"]:
        md.append(f"| {p['check']} | {p['n_common']} | {p['n_scored']} | {_f(p['R_do'])} | {_f(p['R_bin'])} | {_f(p['R_check'])} | "
                  f"{_f(p['dR_kind'])} {_ci(p['dR_boot'], 'dR_kind')} | {'y' if p['ok_kind'] else 'n'} |")
    md += ["", BARS_NOTE]
    if not z["go"]:
        md += ["", "Fewer than 3 pairs are eligible for E2-B: stop, report, and claim 6 takes the 'tangled' wording."]
    return "\n".join(md)


def stage_eligibility(smoke: bool, force=None) -> int:
    root = e2_root(smoke)
    root.mkdir(parents=True, exist_ok=True)
    z = stage1_table(smoke, force=force)
    ac.write(root / "stage1.json", z)
    md = render_stage1(z)
    (root / "stage1.md").write_text(md + "\n")
    for m in ("STAGE1_GO", "STAGE1_STOP"):
        (root / m).unlink(missing_ok=True)
    marker = "STAGE1_GO" if z["go"] else "STAGE1_STOP"
    (root / marker).write_text(f"{z['n_used']} eligible of {len(z['pairs'])} (need {z['need']})\n")
    print(md, flush=True)
    print(marker, flush=True)
    return 0


def _f(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _boot(b):
    return "n/a" if not b or b.get("lo") is None else f"[{b['lo']:.0f}, {b['hi']:.0f}]"


def _ci(boot, name):
    v = (boot or {}).get(name)
    return "" if not v else f"[{v['lo']:+.2f}, {v['hi']:+.2f}]"


def render(s: dict) -> str:
    md = [f"# Robustness E2 (revision R2): check, binary and doing tasks in matched triples{' (smoke)' if s['smoke'] else ''}", "",
          "Check tasks 26-30 and binary doing tasks 31-35 healed like their doing twins (q/v LoRA rank 10, 300 steps, shared cut order, "
          "k = 0 and 12). R12 = (seed-mean test accuracy at k=12 - chance) / (seed-mean healed k=0 - chance), clipped 0 to 1; "
          "whole-answer exact match. Every R and dR of a pair is computed on the pair's common test ids (scored by check, bin and doing "
          "models alike); arith pairs on test records whose expression is in no train or lrsel of the three tasks, the all-record "
          "numbers beside. Each dR carries a record-bootstrap 95% interval (1,000 resamples, seed 0), reported without a bar. A task's R counts only if its healed k=0 accuracy is at least chance + 0.15; a pair enters a sign "
          "count only if the side subtracted is neither at the ceiling (R >= 0.95) nor at the floor (R <= 0.05). dR_kind = R(check) - "
          "R(bin); dR_fmt = R(bin) - R(do); dR_cd = R(check) - R(do); dR_ft = R(check) - R_ft(do), R_ft scoring the doing checkpoints by "
          "argmax over the restricted options (E2-L annotates check - do only). Depth gaps are L_on_healed differences. 1.7B is primary; 0.6B is reported beside it.", ""]
    for size in ("1.7B", "0.6B"):
        z = s["sizes"].get(size)
        if not z:
            continue
        md += [f"## {size}{' (primary)' if size == '1.7B' else ''}", "",
               "| check | n scored / common | L_on do | L_on bin | L_on check | [bin 95%] | [check 95%] | R do | R bin | R check | "
               "dR_kind [95%] | ok | dR_fmt [95%] | ok | dR_cd [95%] | dR_ft | R_ft(do) |", "|" + "---|" * 17]
        for p in z["pairs"]:
            bt = p.get("dR_boot")
            md.append(f"| {p['check']} | {p.get('n_scored')} / {p.get('n_common')} | {_f(p['depth_do'])} | {_f(p['depth_bin'])} | "
                      f"{_f(p['depth_check'])} | {_boot(p['depth_boot']['bin'])} | {_boot(p['depth_boot']['check'])} | {_f(p['R_do'])} | "
                      f"{_f(p['R_bin'])} | {_f(p['R_check'])} | {_f(p['dR_kind'])} {_ci(bt, 'dR_kind')} | {'y' if p['ok_kind'] else 'n'} | "
                      f"{_f(p['dR_fmt'])} {_ci(bt, 'dR_fmt')} | {'y' if p['ok_fmt'] else 'n'} | {_f(p['dR_cd'])} {_ci(bt, 'dR_cd')} | "
                      f"{_f(p['dR_ft'])} | {_f(p['R_do_ft'])} |")
        for p in z["pairs"]:
            if p.get("all_records"):
                ar = p["all_records"]
                md.append(f"  - {p['check']}: all {ar['n']} common records (not the primary numbers): R do {_f(ar['R_do'])}, R bin "
                          f"{_f(ar['R_bin'])}, R check {_f(ar['R_check'])}, dR_kind {_f(ar['dR_kind'])}, dR_cd {_f(ar['dR_cd'])}")
        a, ac_, b, f, d, l, c = z["E2-A"], z["E2-A_check_vs_do"], z["E2-B"], z["E2-F"], z["E2-D"], z["E2-L"], z.get("E2-C")
        md += ["", f"- **E2-A** (eligible pairs only; median depth gap check - bin {_f(a['median'], 1)} over {a['n']} pairs): **{a['label']}**; "
                   f"check - do median {_f(ac_['median'], 1)} over {ac_['n']} pairs ({ac_['label']}, reported)",
               f"- **E2-B** mean dR_kind {_f(b['mean'])} over {b['n_eligible']} eligible of {b['n_pairs']} pairs; dR_kind > 0 in "
               f"{b['parts']['n_positive']}; largest |dR_kind| {_f(b['parts']['max_abs'])}: **{b['label']}**",
               f"- **E2-F** (bin - do) mean dR_fmt {_f(f['mean'])} over {f['n_eligible']} eligible; > 0 in {f['parts']['n_positive']}: "
               f"**{f['label']}**",
               f"- **E2-D** (descriptive, no p) rho(depth gap, survival gap) {_f(d['rho'])} over {d['n']} pair differences",
               f"- **E2-L** (annotation on check - do only; E2-B's label is untouched) mean dR_cd {_f(l['mean_dR_cd'])}, mean dR_ft "
               f"{_f(l['mean_dR_ft'])} over {l['n']} pairs: **{l['label']}**"]
        if c:
            t, w = c["tasks24"], c["with_new"]
            md += [f"- **E2-C** (report; E1-c's bars; R on the test split; "
                   f"{len(c['selection_split_tasks'])} of the 24 have no test R and keep the selection-split R12: "
                   f"{', '.join(c['selection_split_tasks']) or 'none'}): 24 tasks, n {t['n']}: rho(L_on_healed, R) {_f(t['rho'])} (p {_f(t['p'], 4)}), "
                   f"partial rho(kind, R | depth) {_f(t['partial'])} (p {_f(t['partial_p'], 4)}): {t['partial_verdict']}{' [descriptive]' if t.get('descriptive') else ''}. "
                   f"With {len(c['tasks_added'])} check and bin tasks, n {w['n']}: rho {_f(w['rho'])} (p {_f(w['p'], 4)}), partial "
                   f"{_f(w['partial'])} (p {_f(w['partial_p'], 4)}): {w['partial_verdict']}{' [descriptive]' if w.get('descriptive') else ''}."]
        if z.get("answer_tokens"):
            md.append("- Doing answers, longest in tokens: " + ", ".join(f"{t} {n}" for t, n in z["answer_tokens"].items())
                      + "; multi-token (restricted-option first-token scoring reads only the first token): "
                      + (", ".join(t for t, n in z["answer_tokens"].items() if n > 1) or "none"))
        dp = {c_: v for c_, v in (z.get("do_answer_probe") or {}).items() if v}
        if dp:
            md.append("- Free report, does checking compute the answer? Probe for meta.do_answer on the healed check model (peak "
                      "accuracy / chance at layer): " + "; ".join(f"{c_} {v['peak']:.2f} / {v['chance']:.2f} at {v['peak_layer']}"
                                                                  for c_, v in dp.items()))
        if z.get("bytype"):
            md += ["- Free report, check accuracy by near-miss type (k=0 -> k=12, n):"]
            for t, ks in z["bytype"].items():
                for ty in sorted({ty for k in ks.values() for ty in k}):
                    cells = [f"k={k}: {v[ty][1] / v[ty][0]:.2f} (n {v[ty][0]})" for k, v in sorted(ks.items(), key=lambda kv: int(kv[0]))
                             if ty in v]
                    md.append(f"  - {t} / {ty}: " + ", ".join(cells))
        md += [f"- {BARS_NOTE}", f"- **Claim 6 reading** ({size}): {z['claim6']['text']}", ""]
    md += ["## Claim 6 (1.7B primary)", "", s["claim6"]["text"], ""]
    return "\n".join(md)


def stage_table(smoke: bool, n_perm: int = N_PERM, override=None) -> int:
    chk = check_tasks(smoke, override=override)
    kcut = 2 if smoke else 12
    sizes = {sz: size_summary(sz, smoke, chk, kcut, n_perm) for sz in SIZES if size_out(sz, smoke).exists()}
    if "1.7B" not in sizes:
        raise SystemExit("no 1.7B results to tabulate")
    s = {"smoke": smoke, "primary": "1.7B", "run_tasks": list(chk), "sizes": sizes, "claim6": sizes["1.7B"]["claim6"],
         "bars": {"A_same": A_SAME, "A_early": A_EARLY, "B_effect": B_EFFECT, "B_none": B_NONE, "B_spread": B_SPREAD, "B_need": B_NEED, "gate_need": GATE_NEED,
                  "L_min": L_MIN, "elig_margin": ELIG_MARGIN, "r_ceil": R_CEIL, "r_floor": R_FLOOR, "n_perm": n_perm, "seed": SEED,
                  "n_boot": N_BOOT}}
    root = e2_root(smoke)
    root.mkdir(parents=True, exist_ok=True)
    ac.write(root / "summary.json", s)
    md = render(s)
    (root / "tables.md").write_text(md + "\n")
    print(md, flush=True)
    return 0


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--size", choices=sorted(SIZES), default="1.7B")
    ap.add_argument("--seed", type=int, default=None, choices=ac.SEEDS, help="sweep: the seed (default 0); doexact: only this seed")
    ap.add_argument("--force-eligible", type=int, default=os.environ.get("E2_FORCE_ELIGIBLE") and int(os.environ["E2_FORCE_ELIGIBLE"]),
                    help="eligibility, smoke only: use this as the eligible-pair count (exercises both branches)")
    ap.add_argument("--shard", default=None, help="I/N")
    ap.add_argument("--task", default=None, help="healedprobe: one check or bin task")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--list-tasks", action="store_true")
    ap.add_argument("--tasks", default=os.environ.get("E2_TASKS"),
                    help="comma list of check/bin tasks (default: the clean pairs in the gate file; $E2_TASKS also works)")
    args = ap.parse_args(argv)
    override = [t for t in (args.tasks or "").split(",") if t] or None
    chk = check_tasks(args.smoke, override=override)
    if args.list_tasks:
        print(" ".join(chk))
        return 0
    if not args.stage:
        ap.error("--stage is required")
    if args.stage == "table":
        return stage_table(args.smoke, override=override)
    if args.stage == "eligibility":
        return stage_eligibility(args.smoke, args.force_eligible if args.smoke else None)
    if args.stage == "healedprobe" and args.task not in chk:
        raise SystemExit(f"healedprobe needs --task in {list(chk)}")
    profile = size_profile(args.size, args.smoke)
    C.configure(profile)
    cfg = make_cfg(args.size, args.smoke, chk)
    ac.OUT.mkdir(parents=True, exist_ok=True)
    ctx = ac.Ctx(cfg, device=os.environ.get("ATLAS3_DEVICE") or None)
    C.check_blocks(ctx.n_layers, profile)
    if args.stage == "sweep":
        return C.stage_sweep(ctx, args.seed or 0, ac.parse_shard(args.shard))
    if args.stage == "test":
        return stage_test(ctx)
    if args.stage == "doexact":
        return stage_doexact(ctx, args.size, args.seed)
    if args.stage == "healedprobe":
        return stage_healedprobe(ctx, args.task)
    if args.stage == "bytype":
        return stage_bytype(ctx, args.size)
    return stage_ftdo(ctx, args.size)


if __name__ == "__main__":
    sys.exit(main())
