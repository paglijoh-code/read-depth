"""Post hoc Steps v3 confirmation; fixed bars: docs/ATLAS-steps3-confirm.md.

`twins` and `table` do not access a model. Training reuses Stage 1's whole-answer
objective, exact-match checkpoint selection, and shared uncut models. Set
STEPS3_CF_SMOKE=1 for isolated 20-step runs on seeds 0 and 2.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import steps3_stage1 as T  # noqa: E402
from surgery.atlas import steps3 as G  # noqa: E402

C, ac, S, heal, spec = T.C, T.ac, T.S, T.heal, T.spec
DOC = "docs/ATLAS-steps3-confirm.md"
STAMP = {"post_hoc_confirmation": True, "doc": DOC}
COPY, SWAP = S.COPY, "steps3_swap"
TASKS = {"copy": COPY, "swap": SWAP}
SEEDS = (0, 2, 3, 4, 5)
EPS = 1e-9
TWIN_SEED = 20261119
WINDOWS = {"early": (7, 8, 9, 10), "mid": (11, 12, 13, 14), "pre": (3, 4, 5, 6)}
ALLLIN = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ARMS = {
    "cf_A": dict(window="early", steps=2400, modules=T.QV, rank=10, lrs=(1e-3, 3e-4, 3e-3), tasks=(COPY, SWAP)),
    "cf_B": dict(window="early", steps=1200, modules=ALLLIN, rank=16, lrs=(1e-3, 3e-4), tasks=(COPY,)),
    "cf_C": dict(window="mid", steps=1200, modules=T.QV, rank=10, lrs=(1e-3,), tasks=(COPY, SWAP)),
    "cf_D": dict(window="early", steps=300, modules=T.QV, rank=10, lrs=tuple(ac.LRS), tasks=(COPY, SWAP)),
    "cf_F": dict(window="pre", steps=2400, modules=T.QV, rank=10, lrs=(), tasks=(COPY,)),
}


def settings(smoke=None):
    smoke = os.environ.get("STEPS3_CF_SMOKE") == "1" if smoke is None else smoke
    root = Path("var/results/steps3/confirm_smoke" if smoke else "var/results/steps3/stage1")
    return SimpleNamespace(smoke=smoke, root=root, seeds=(0, 2) if smoke else SEEDS,
                           report=root if smoke else Path("var/results/steps3/confirm"),
                           ckpt=Path("var/ckpt/steps/smoke_cf" if smoke else "var/ckpt/steps"),
                           twins=root / "swap_twins.jsonl" if smoke else Path("var/data/steps3_confirm/swap_twins.jsonl"))


def schedule(steps, smoke=False):
    return (20, 2) if smoke else (steps, steps // 100)


def window_order(shared, window):
    return T.window_order(shared, window, windows=WINDOWS)


def copy_pairs(records):
    """All complete base/flip groups, in stable group order (no per-cell cap)."""
    groups = {}
    for rec in records:
        meta = rec.get("meta", {})
        if meta.get("twin_role") in ("base", "flip") and meta.get("twin_group") is not None:
            groups.setdefault(meta["twin_group"], {})[meta["twin_role"]] = rec["id"]
    return [(g["base"], g["flip"]) for _, g in sorted(groups.items()) if "base" in g and "flip" in g]


def twin_same(pairs, items):
    counts = dict(both_right=0, first_only=0, second_only=0, neither=0)
    same = 0
    for first, second in pairs:
        a, b = items[first], items[second]
        same += a["pred"] == b["pred"]
        key = ("both_right" if b["ok"] else "first_only") if a["ok"] else ("second_only" if b["ok"] else "neither")
        counts[key] += 1
    return {"n": len(pairs), "same": same / len(pairs) if pairs else None, **counts}


def build_swap_twins(records, seed=TWIN_SEED):
    """Change one endpoint, preserving the original path/filler necessity annotations.

    Exhaustively try a seeded permutation of all single-endpoint substitutions.
    Keep only twins accepted by the generator's full necessity check, including
    its original number of necessary swaps. Input records are never mutated.
    """
    twins, skipped = [], []
    for rec in records:
        row, swaps, asked = G.parse_swap(rec["prompt"])
        candidates = [(line, slot, value) for line, pair in enumerate(swaps) for slot in (0, 1)
                      for value in range(10) if value not in pair]
        random.Random(f"{seed}:{rec['id']}").shuffle(candidates)
        for line, slot, value in candidates:
            changed = [list(pair) for pair in swaps]
            changed[line][slot] = value
            answer = G.LETTERS.index(G.simulate_swaps(row, changed)[asked])
            if answer == rec["answer"]:
                continue
            twin = copy.deepcopy(rec)
            twin.update(id=rec["id"] + "__twin", twin_of=rec["id"], answer=answer, **STAMP)
            lines = rec["prompt"].split("\n")
            a, b = changed[line]
            lines[line + 1] = f"swap slot {a} and slot {b}"
            twin["prompt"] = "\n".join(lines)
            meta = twin["meta"]
            meta["swaps"][line].update(i=a, j=b)
            source = asked
            for a, b in reversed(changed):
                source = b if source == a else a if source == b else source
            meta.update(source_slot=source, answer_letter=G.LETTERS[answer], twin_seed=seed)
            twin["template_id"] = rec["template_id"] + f"__twin_{line}_{slot}_{value}"
            try:
                G._verify_swap(twin)
            except AssertionError:
                continue
            if shortcut_hits(twin, rec["rung"]):
                continue  # same item rejection as the generator's _cell_split
            twins.append(twin)
            break
        else:
            skipped.append(rec["id"])
    return twins, skipped


MIN_BUILT_SHARE = 0.9


def shortcut_hits(rec, cell):
    """Non-expected shortcut rules that solve this item (must be empty, as in the generator)."""
    expected = G.expected(SWAP, cell)
    return [name for name, fn in G.RULES[SWAP].items() if name not in expected and fn(rec) == rec["answer"]]


def twins_gate(records, twins, skipped):
    answers = {r["id"]: r["answer"] for r in records}
    n = len(twins)
    share = sum(r["answer"] != answers[r["twin_of"]] for r in twins) / n if n else None
    rules = {}
    for name, fn in G.RULES[SWAP].items():
        scope = [r for r in twins if name not in G.expected(SWAP, r["rung"])]
        rules[name] = sum(fn(r) == r["answer"] for r in scope) / len(scope) if scope else None
    built_share = n / len(records) if records else 0.0
    passes = (share == 1.0 and all(v in (None, 0.0) for v in rules.values())
              and built_share >= MIN_BUILT_SHARE)
    return {**STAMP, "seed": TWIN_SEED, "built": n, "skipped": len(skipped), "skipped_ids": skipped,
            "built_share": built_share, "min_built_share": MIN_BUILT_SHARE,
            "answer_differs": share, "rule_accuracy_non_expected": rules, "passes": passes}


def stage_twins(cfg, records=None):
    gate_path = cfg.report / "twins_gate.json"
    if cfg.twins.exists() and gate_path.exists():
        return ac.read_json(gate_path)
    records = spec.read_records(SWAP, "test") if records is None else records
    twins, skipped = build_swap_twins(records)
    gate = twins_gate(records, twins, skipped)
    if not gate["passes"]:
        raise SystemExit("swap twin gate failed: need answer-difference 1.0, zero non-expected shortcut accuracy, "
                         f"and built share >= {MIN_BUILT_SHARE}: {gate}")
    spec.write_records(twins, cfg.twins)
    ac.write(gate_path, gate)
    return gate


def lr_plan(arm, seed, seed0_rows=(), a_rows=()):
    if arm == "cf_F":
        required = ARMS["cf_A"]["lrs"]
        rows = a_rows
    else:
        required = ARMS[arm]["lrs"] if arm in ARMS else tuple(ac.LRS)
        if seed == 0 or arm == "cf_C":
            return list(required)
        rows = seed0_rows
    if not all(any(abs(r["lr"] - lr) < 1e-12 for r in rows) for lr in required):
        source = "cf_A" if arm == "cf_F" else arm
        raise SystemExit(f"run {source} seed 0 first (all learning rates) for this task")
    return [heal.pick_lr(rows)]


class Runs(T.Runs):
    def put(self, key, row):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            doc = ac.read_json(self.path) or {"exploratory": True}
            self.runs = {**doc.get("runs", {}), key: {**row, **STAMP}}
            ac.write(self.path, {**doc, "runs": self.runs})  # rows carry STAMP; the shared file header stays Stage 1's


def setup(cfg):
    # Configure constants without S.configure(), which creates unrelated output directories.
    C.configure(None)
    for module in (ac, C, S.B):
        module.OUT, module.CKPT = cfg.root, cfg.ckpt
    train_cfg = S.make_cfg("qwen17", cfg.smoke)
    train_cfg.transfer, train_cfg.ledger = False, False
    train_cfg.steps = schedule(300, cfg.smoke)[0]
    ctx = S.make_ctx(train_cfg)
    shared = list(ctx.order)
    for window in WINDOWS:
        window_order(shared, window)
    return SimpleNamespace(root=cfg.root, ckpt_root=cfg.ckpt, cfg=train_cfg, ctx=ctx, shared=shared,
                           windows=WINDOWS, wctxs={}, runs=Runs(cfg.root / "runs.json"), confirm=cfg)


def seed_list(cfg, seed):
    if seed is not None and seed not in cfg.seeds:
        raise SystemExit(f"confirmation seeds are {cfg.seeds}, not {seed}")
    return cfg.seeds if seed is None else (seed,)


def pair_records(env, task, records, *, test=False):
    if task == COPY:
        return [], copy_pairs(records)
    if test:
        path = env.confirm.twins
        if not path.exists():
            raise SystemExit("run --stage twins first")
        import json
        ids = {r["id"] for r in records}
        twins = [r for r in (json.loads(line) for line in path.read_text().splitlines()) if r["twin_of"] in ids]
    else:
        twins, _ = build_swap_twins(records)
    if not twins:
        raise SystemExit("no swap twins available for scoring")
    return twins, [(r["twin_of"], r["id"]) for r in twins]


@contextmanager
def selection_curve(env, task, n_evals):
    """Observe training evaluations without changing the shared trainer or its EM choice.

    The trainer calls exact_match_scores once per checkpoint, then the runner
    calls it once on selection. Observe only those first n_evals calls. Restore
    the function even on failure; each process runs one training job at a time.
    """
    original = heal.exact_match_scores
    observations = []
    records = env.ctx.records(task, "selection")
    twins, pairs = pair_records(env, task, records)

    def observed(model, tok, task_, *args, **kwargs):
        result = original(model, tok, task_, *args, **kwargs)
        if len(observations) < n_evals:
            acc, preds = heal.accuracy_and_preds(model, tok, task, records, ac.EVAL_BS)
            items = {r["id"]: {"pred": int(p), "ok": int(p == r["answer"])} for r, p in zip(records, preds)}
            if twins:
                _, twin_preds = heal.accuracy_and_preds(model, tok, task, twins, ac.EVAL_BS)
                items.update({r["id"]: {"pred": int(p), "ok": int(p == r["answer"])} for r, p in zip(twins, twin_preds)})
            observations.append({"selection_accuracy": float(acc), "selection_twin": twin_same(pairs, items)})
        return result

    heal.exact_match_scores = observed
    try:
        yield observations
    finally:
        heal.exact_match_scores = original


def train_observed(env, task, n_evals, train):
    with selection_curve(env, task, n_evals) as observations:
        row = train()
    if len(row["curve"]) != len(observations):
        raise RuntimeError("training curve and confirmation observations differ in length")
    row["curve"] = [{**point, **obs} for point, obs in zip(row["curve"], observations)]
    return row


def ensure_rows(env, arm, window, task, seed, train, extra, lrs=None):
    base = dict(arm=arm, window=window, task=task)
    # Refresh at dependency boundaries: another shard may have finished cf_A.
    env.runs = Runs(env.root / "runs.json")
    lrs = lr_plan(arm, seed, env.runs.rows(**base, seed=0),
                  env.runs.rows(arm="cf_A", window="early", task=task, seed=0)) if lrs is None else lrs
    for lr in lrs:
        key = T.row_key(arm, window, task, seed, lr)
        if key not in env.runs.runs:
            row = train(lr)
            row.update(arm=arm, window=window, **extra, **STAMP)
            env.runs.put(key, row)
    rows = [env.runs.runs[T.row_key(arm, window, task, seed, lr)] for lr in lrs]
    winner = heal.pick_lr(rows)
    return next(r for r in rows if r["lr"] == winner)


def v2_runs(root=S.ROOT_OUT):
    """Read the original baseline shards independently of the confirmation output root."""
    merged = {}
    for path in sorted(Path(root).glob("runs.shard*of*.json")):
        merged.update((ac.read_json(path) or {}).get("runs", {}))
    return merged


def uncut_row(env, task, seed, train=False):
    # Existing copy baselines remain authoritative, including the v2 checkpoints.
    if not env.confirm.smoke and task == COPY and seed in (0, 2, 3):
        row = S.winner_row(v2_runs(), task, 0, seed)
        if row is not None:
            return row
        own = env.runs.rows(arm="uncut", task=task, seed=seed)
        if own:
            return own[0]
        raise SystemExit(f"no existing uncut copy row for s{seed}; Stage 1 baseline required")
    own = env.runs.rows(arm="cf_uncut", window=T.NONE, task=task, seed=seed)
    if own and (seed != 0 or len(own) == len(ac.LRS)):
        return next(r for r in own if r["lr"] == heal.pick_lr(own))
    if not train:
        raise SystemExit(f"no confirmation uncut row for {task} s{seed}; run --stage uncut first")
    lrs = None
    if task == COPY and not env.confirm.smoke:
        # Exactly Stage 1's get_k0_row learning-rate rule for new copy seeds.
        lr = ac.winner_lr(v2_runs(), task, 0, "per_k")
        if lr is None:
            raise SystemExit("no v2 seed-0 uncut copy learning-rate winner")
        lrs = [lr]
    steps, n_evals = schedule(300, env.confirm.smoke)

    def train_lr(lr):
        return train_observed(env, task, n_evals, lambda: C.run_whole_heal(
            env.ctx, task, 0, seed, lr, steps=steps, n_evals=n_evals,
            ckpt_fn=T.ckpt_fn_for(env, "cf_uncut", T.NONE)))
    return ensure_rows(env, "cf_uncut", T.NONE, task, seed, train_lr,
                       dict(kind="ctt", k=0, modules=list(T.QV), rank=10, steps=steps), lrs=lrs)


def score_model(env, model, task):
    records = env.ctx.records(task, "test")
    twins, pairs = pair_records(env, task, records, test=True)
    score = T.score_all(env.ctx, model, task, records)
    twin_score = T.score_all(env.ctx, model, task, twins) if twins else None
    items = {**score["items"], **(twin_score["items"] if twin_score else {})}
    return {"score": score, "twins_score": twin_score, "twin": twin_same(pairs, items)}


def save_result(env, path, task, seed, arm, window, row, scores, **extra):
    ac.write(path, {**STAMP, "arm": arm, "window": window, "task": task, "seed": seed, "lr": row["lr"],
                    "best_step": row.get("best_step"), "curve": row.get("curve", []), "steps": row.get("steps"),
                    "ckpt": row["ckpt"], "ts": ac.now(), **scores, **extra})


def stage_uncut(env, tasks, seed=None):
    for task in tasks:
        for sd in seed_list(env.confirm, seed):
            if not env.confirm.smoke and task == COPY and sd in (0, 2, 3):
                continue
            path = T.rpath(env.root, "cf_uncut", task, f"s{sd}")
            if path.exists():
                continue
            row = uncut_row(env, task, sd, train=True)
            model, handle = ac.rebuild_healed(env.ctx, row)
            T.adapters.merge_lora(handle)
            try:
                scores = score_model(env, model, task)
            finally:
                del model
                ac.release(env.ctx)
            save_result(env, path, task, sd, "cf_uncut", T.NONE, row, scores)


def swap_arm_dropped(env):
    """Design: Arm E is dropped (no swap training) if any uncut swap seed scores below 0.85. Not applied in smoke."""
    if env.confirm.smoke:
        return False
    for sd in env.confirm.seeds:
        doc = ac.read_json(T.rpath(env.root, "cf_uncut", SWAP, f"s{sd}"))
        if doc is None:
            raise SystemExit(f"uncut swap s{sd} missing: run --stage uncut --task swap first")
        acc = (doc.get("score") or {}).get("acc")  # saved results nest the test score under "score"
        if acc is None:
            raise SystemExit(f"uncut swap s{sd} has no score.acc")
        if acc < 0.85 - EPS:
            return True
    return False


def stage_arm(env, arm, tasks, seed=None):
    config = ARMS[arm]
    window = config["window"]
    steps, n_evals = schedule(config["steps"], env.confirm.smoke)
    for task in tasks:
        if task not in config["tasks"]:
            raise SystemExit(f"{arm} does not apply to {task}")
        if task == SWAP and swap_arm_dropped(env):
            print(f"{arm} swap: Arm E dropped (an uncut swap seed is below 0.85); not training", flush=True)
            continue
        for sd in seed_list(env.confirm, seed):
            if arm == "cf_D" and task == COPY and sd in (0, 2, 3) and not env.confirm.smoke:
                print(f"cf_D copy s{sd}: reusing Stage 1 tc early result", flush=True)
                continue
            path = T.rpath(env.root, arm, task, window, f"s{sd}")
            if path.exists():
                continue
            wctx = T.window_ctx(env, window)
            alive = heal.alive_blocks(window_order(env.shared, window), 4, env.cfg.n_layers)
            extra = dict(kind="ctt", k=4, modules=list(config["modules"]), rank=config["rank"],
                         steps=steps, alive_blocks=alive, removed=list(WINDOWS[window]))
            after_cut = None
            if arm == "cf_D":
                base = uncut_row(env, task, sd)
                extra.update(kind="tc", base={"ckpt": base["ckpt"], "seed": sd, "k": 0})
                spec_ = dict(kind="tc", k=4, window=window, base=extra["base"])
                model, _ = T.prepare_model(env, spec_)
                try:
                    after_cut = T.score_all(env.ctx, model, task, env.ctx.records(task, "test"))
                finally:
                    del model
                    ac.release(env.ctx)

                def train(lr):
                    return train_observed(env, task, n_evals, lambda: T.train_built(
                        env, spec_, task, sd, lr, steps, T.QV, 10, arm, n_evals=n_evals))
            else:
                def train(lr):
                    return train_observed(env, task, n_evals, lambda: C.run_whole_heal(
                        wctx, task, 4, sd, lr, own=True, removed=list(WINDOWS[window]),
                        arm=SimpleNamespace(modules=config["modules"], rank=config["rank"]),
                        steps=steps, n_evals=n_evals, ckpt_fn=T.ckpt_fn_for(env, arm, window)))
            # Copy D seeds 4/5 use Stage 1 tc seed 0, rather than requiring a new cf_D copy seed 0.
            lrs = None
            if arm == "cf_D" and task == COPY and not env.confirm.smoke:
                lrs = T.lr_plan(env.runs.rows(arm="tc", window="early", task=task, seed=0), sd)
            row = ensure_rows(env, arm, window, task, sd, train, extra, lrs=lrs)
            model, _ = T.build_scored(env, row)
            try:
                scores = score_model(env, model, task)
            finally:
                del model
                ac.release(env.ctx)
            save_result(env, path, task, sd, arm, window, row, scores, after_cut=after_cut)


def classify(acc, same):
    if acc is None or same is None:
        return "missing"
    if acc <= 0.60 + EPS and same >= 0.80 - EPS:
        return "stuck"
    if acc >= 0.85 - EPS and same <= 0.20 + EPS:
        return "learned"
    return "between"


def run_summary(doc, row=None, pairs=None):
    score = doc.get("score", doc.get("after_repair", {}))
    twin = doc.get("twin")
    if twin is None and pairs is not None and score.get("items"):
        twin = twin_same(pairs, score["items"])
    same = (twin or {}).get("same")
    row = row or {}
    curve = doc.get("curve", row.get("curve", []))
    return dict(acc=score.get("acc"), twin_same=same, classification=classify(score.get("acc"), same),
                lr=doc.get("lr", row.get("lr")), best_step=doc.get("best_step", row.get("best_step")),
                steps=doc.get("steps", row.get("steps")), curve=curve,
                jump_step=next((p["step"] for p in curve if p.get("selection_accuracy") is not None and p["selection_accuracy"] >= 0.85 - EPS), None))


def build_verdicts(results, seeds=SEEDS):
    """Pure: results[arm][task][seed] contains run_summary dictionaries."""
    def counts(arm, task=COPY):
        runs = results.get(arm, {}).get(task, {})
        incomplete = [s for s in seeds if runs.get(s, {}).get("steps") is not None
                      and runs[s]["steps"] < ARMS[arm]["steps"]]
        return {"stuck": sum(runs.get(s, {}).get("classification") == "stuck" for s in seeds if s not in incomplete),
                "learned": sum(runs.get(s, {}).get("classification") == "learned" for s in seeds if s not in incomplete),
                "incomplete_steps": incomplete,
                "missing": [s for s in seeds if s not in runs or runs[s].get("classification") == "missing"]}

    def stuck_bar(arm, task=COPY):
        c = counts(arm, task)
        return {**c, "label": "holds" if c["stuck"] >= 4 else "fails" if c["learned"] >= 3 else "inconclusive"}

    def learned_bar(arm):
        c = counts(arm)
        return {**c, "label": "holds" if c["learned"] >= 4 else "fails"}

    bars = {"K1": stuck_bar("cf_A"), "K2": stuck_bar("cf_B"),
            "K3": learned_bar("cf_D"), "K4": learned_bar("cf_C")}
    if bars["K4"]["label"] == "fails":
        for key in ("K1", "K2"):
            bars[key]["raw_label"] = bars[key]["label"]
            bars[key]["label"] = "not interpretable"
    a, d = counts("cf_A", SWAP), counts("cf_D", SWAP)
    uncut = {s: results.get("uncut", {}).get(SWAP, {}).get(s, {}).get("acc") for s in seeds}
    dropped = any(acc is not None and acc < 0.85 - EPS for acc in uncut.values())
    missing_uncut = [s for s, acc in uncut.items() if acc is None]
    bars["K5"] = {"label": "dropped" if dropped else "holds" if a["stuck"] >= 4 and d["learned"] >= 4
                  and not missing_uncut else "fails", "cf_A": a, "cf_D": d, "uncut_accuracy": uncut,
                  "missing_uncut": missing_uncut}
    f = counts("cf_F")
    bars["K6"] = {**f, "label": "location-specific" if f["learned"] >= 3 and bars["K1"]["label"] == "holds"
                  else "inconclusive (more damage)" if f["stuck"] >= 4 else "inconclusive"}
    return {**STAMP, "seeds": list(seeds), "bars": bars}


def load_results(cfg, copy_records=None):
    results = {}
    runs = Runs(cfg.root / "runs.json")
    legacy_pairs = None
    for arm in (*ARMS, "uncut"):
        tasks = ARMS[arm]["tasks"] if arm in ARMS else (COPY, SWAP)
        window = ARMS[arm]["window"] if arm in ARMS else T.NONE
        for task in tasks:
            for seed in cfg.seeds:
                legacy = not cfg.smoke and task == COPY and seed in (0, 2, 3) and arm in ("cf_D", "uncut")
                file_arm = "tc" if legacy and arm == "cf_D" else "uncut" if legacy else "cf_uncut" if arm == "uncut" else arm
                parts = (file_arm, task, f"s{seed}") if arm == "uncut" else (file_arm, task, window, f"s{seed}")
                doc = ac.read_json(T.rpath(cfg.root, *parts))
                if doc is None:
                    continue
                rows = runs.rows(arm=file_arm, window=window, task=task, seed=seed)
                row = next((r for r in rows if r["lr"] == doc.get("lr")), {})
                if legacy and arm == "uncut" and not row:
                    row = S.winner_row(v2_runs(), task, 0, seed) or {}
                if legacy:
                    if legacy_pairs is None:
                        legacy_pairs = copy_pairs(spec.read_records(COPY, "test") if copy_records is None else copy_records)
                summary = run_summary(doc, row, legacy_pairs if legacy else None)
                summary["source"] = T.rpath(cfg.root, *parts).name
                results.setdefault(arm, {}).setdefault(task, {})[seed] = summary
    return results, runs


def damage_sums(order_path):
    doc = ac.read_json(order_path) if order_path else None
    drops = (doc or {}).get("ppl_drop")
    if drops is None:
        return {"pre": None, "early": None}
    return {w: sum(drops[str(b)] if isinstance(drops, dict) else drops[b] for b in WINDOWS[w]) for w in ("pre", "early")}


def format_curve(curve):
    values = []
    for point in curve:
        accuracy = point.get("selection_accuracy")
        if accuracy is None:
            accuracy = f"{point.get('lrsel_exact', point.get('lrsel_accuracy', 'unavailable'))} (lrsel)"
        same = point.get("selection_twin", {}).get("same", "unavailable")
        values.append(f"{point['step']}: {accuracy}; {same}")
    return ", ".join(values)


def stage_table(cfg, order_path=T.ORDER_JSON, copy_records=None):
    results, runs = load_results(cfg, copy_records)
    verdicts = build_verdicts(results, cfg.seeds)
    verdicts["results"] = results
    verdicts["damage"] = damage_sums(order_path)
    verdicts["cf_A_seed0_3e-3"] = runs.rows(arm="cf_A", window="early", seed=0, lr=3e-3)
    verdicts["smoke"] = cfg.smoke
    lines = ["# Steps v3 POST HOC CONFIRMATION", "", f"Protocol: {DOC}", "",
             "Checkpoint choice: lrsel exact match. Curve accuracy/twins: selection split, first token.",
             "Test accuracy/twins determine the bars. Jump steps and 3e-3 are report-only.", ""]
    if cfg.smoke:
        lines += ["SMOKE ONLY: 20 steps, two seeds; the fixed five-seed bars cannot be satisfied.", ""]
    lines += ["| bar | verdict | details / missing seeds |", "|---|---|---|"]
    for bar, data in verdicts["bars"].items():
        details = {k: v for k, v in data.items() if k != "label"}
        lines.append(f"| {bar} | {data['label']} | {details} |")
    lines += ["", f"K6 summed single-block perplexity rise (order.json ppl_drop): {verdicts['damage']}",
              "Legacy curves retain lrsel exact match; historical selection first-token/twin curves were not recorded.", "",
              "| arm | task | seed | test acc | twin-same | class | lr | best step | jump step (A) | selection curve (step: accuracy; twin-same) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in (*ARMS, "uncut"):
        tasks = ARMS[arm]["tasks"] if arm in ARMS else (COPY, SWAP)
        for task in tasks:
            for seed in cfg.seeds:
                r = results.get(arm, {}).get(task, {}).get(seed)
                if r is None:
                    lines.append(f"| {arm} | {task} | {seed} | missing | | | | | | |")
                    continue
                curve = format_curve(r["curve"])
                lines.append(f"| {arm} | {task} | {seed} | {r['acc']} | {r['twin_same']} | {r['classification']} | "
                             f"{r['lr']} | {r['best_step']} | {r['jump_step'] if arm == 'cf_A' else ''} | {curve} |")
    lines += ["", "## cf_A seed 0, lr 3e-3 (report only)", ""]
    for row in verdicts["cf_A_seed0_3e-3"]:
        lines.append(f"- {row['task']}: selection exact {row.get('sel_accuracy')}, best step {row.get('best_step')}; "
                     f"curve {row.get('curve', [])}")
    if not verdicts["cf_A_seed0_3e-3"]:
        lines.append("Missing.")
    cfg.report.mkdir(parents=True, exist_ok=True)
    (cfg.report / "tables.md").write_text("\n".join(lines) + "\n")
    ac.write(cfg.report / "verdicts.json", verdicts)
    return verdicts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("twins", "uncut", "A", "B", "C", "D", "F", "table"))
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = settings()
    seed_list(cfg, args.seed)
    if args.stage in ("twins", "table"):
        if args.stage == "twins" and args.task == "copy":
            parser.error("twins applies to swap only")
        stage_twins(cfg) if args.stage == "twins" else stage_table(cfg)
        return
    arm = "cf_" + args.stage
    tasks = (TASKS[args.task],) if args.task else ARMS[arm]["tasks"] if arm in ARMS else (COPY, SWAP)
    if arm in ARMS and any(t not in ARMS[arm]["tasks"] for t in tasks):
        parser.error(f"{args.stage} does not apply to {args.task}")
    env = setup(cfg)
    if args.stage == "uncut":
        stage_uncut(env, tasks, args.seed)
    else:
        stage_arm(env, arm, tasks, args.seed)


if __name__ == "__main__":
    main()
