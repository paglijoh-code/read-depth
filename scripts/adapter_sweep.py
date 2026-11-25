"""M6 adapter-axis sweep (CONTRACT.md M6): LoRA, healed LoRA, LoReFT on Qwen3-0.6B.

  uv run python scripts/adapter_sweep.py --variant {rule,sensible} --stage train --arm {lora,heal,reft,reft0}
      --budget {10k,100k,1m} --seed S [--smoke]
  uv run python scripts/adapter_sweep.py --stage sweep --until HH:MMZ [--smoke]
  uv run python scripts/adapter_sweep.py --variant V --stage test --which {h2,h5}
  uv run python scripts/adapter_sweep.py --stage table
Results go to var/results/m6/ and adapter checkpoints to var/ckpt/m6/. Runs are resumable (a run already in
runs_*.json is skipped). Selection passes append to var/results/selection_log.jsonl and test runs to
var/results/test_ledger.jsonl; --smoke (16 train, 16 held-out, 32 selection, 4 steps, *_smoke.json) never
writes either ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from surgery import adapters, stats, steer, structure  # noqa: E402
from surgery.canaries import eligible, load_canaries, score_canaries  # noqa: E402
from surgery.eval import _free, label_logits, load_model  # noqa: E402
from surgery.hooks import apply_spec, spec_hash  # noqa: E402
from surgery.task.generator import LABELS, read_jsonl  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
SHORT = MODEL.split("/")[-1]
VARIANTS = ["rule", "sensible"]
ARMS = ["lora", "heal", "reft", "reft0"]
BUDGETS = ["10k", "100k", "1m"]
DATA = Path("var/data/m2")
M4 = Path("var/results/m4")
M5 = Path("var/results/m5")
OUT = Path("var/results/m6")
CKPT = Path("var/ckpt/m6")
BASE_CANARIES = Path("var/results/baseline") / f"{SHORT}_canaries.json"
SELECTION_LOG = Path("var/results/selection_log.jsonl")
TEST_LEDGER = Path("var/results/test_ledger.jsonl")
DEADLINE_FILE = Path("var/M6_DEADLINE")
HEAL_K = 9
QV, V = ("q_proj", "v_proj"), ("v_proj",)
# budget -> (layers, modules, rank); heal layers are "the last N surviving blocks" (None = all survivors)
LORA = {"10k": (range(23, 28), V, 1), "100k": (range(18, 28), QV, 2), "1m": (range(28), QV, 7)}
HEAL = {"10k": (5, V, 1), "100k": (10, QV, 2), "1m": (None, QV, 10)}
REFT_RANK = {"10k": 5, "100k": 49, "1m": 488}
SEEDS = [0, 1, 2]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def smoke_suffix(smoke: bool) -> str:
    return "_smoke" if smoke else ""


def runs_path(variant: str, smoke: bool) -> Path:
    return OUT / f"runs_{SHORT}_{variant}{smoke_suffix(smoke)}.json"


def ckpt_path(variant, arm, budget, seed, smoke: bool) -> Path:
    return CKPT / ("smoke" if smoke else "") / f"{variant}_{arm}_{budget}_s{seed}.pt"


def run_key(arm, budget, seed) -> str:
    return f"{arm}|{budget}|{seed}"


def plan_spec(order) -> dict:
    return {"op": "plan", "order": list(order), "scales": None}  # same form as surgery_sweep.plan_spec


# ---------------------------------------------------------------- data

class Ctx:
    """One process: the model, plus per-variant data, means and baselines loaded on first use."""

    def __init__(self, a):
        self.a = a
        self.model, self.tok = load_model(MODEL)
        self.ids = adapters.label_ids(self.model, self.tok)
        self._data, self._means, self._dirs, self._base = {}, {}, {}, {}

    def records(self, variant: str, split: str) -> list[dict]:
        key = (variant, split)
        if key not in self._data:
            recs = read_jsonl(DATA / f"{variant}_s0_{split}.jsonl")
            self._data[key] = recs[:32] if self.a.smoke and split != "discovery" else recs
        return self._data[key]

    def train_heldout(self, variant: str, seed: int):
        n_held, n_train = (16, 16) if self.a.smoke else (200, 800)
        return adapters.split_discovery(self.records(variant, "discovery"), seed, n_held, n_train)

    def means(self, variant: str):
        if variant not in self._means:
            store = Path("var/acts") / f"{SHORT}_{variant}_s0_discovery"
            self._means[variant] = torch.load(store / "means.pt", map_location="cpu", weights_only=True)
        return self._means[variant]

    def directions(self, variant: str) -> dict:
        if variant not in self._dirs:
            self._dirs[variant] = steer.load_directions(M4 / f"directions_{SHORT}_{variant}.pt")
        return self._dirs[variant]

    def baseline_pred(self, variant: str, split: str = "selection") -> list[int]:
        """Predictions of the unmodified model (the model must carry no adapter or spec when this first runs)."""
        key = (variant, split)
        if key not in self._base:
            recs = self.records(variant, split)
            sel = M5 / f"select_{SHORT}_{variant}.json"
            if split == "selection" and not self.a.smoke and sel.exists() and \
                    read_json(sel)["ids"] == [r["id"] for r in recs]:
                self._base[key] = read_json(sel)["baseline"]["pred"]
            else:
                lg = label_logits(self.model, self.tok, [r["prompt"] for r in recs], self.a.batch_size)
                self._base[key] = lg.argmax(-1).tolist()
        return self._base[key]


# ---------------------------------------------------------------- arms

class Arm:
    """An adapter attached to the model: `params` (trainable container), `forward(records) -> (B, 6)` logits,
    `close()` restores the model."""

    def __init__(self, params, forward, handles, n_params):
        self.params, self.forward, self.handles, self.n_params = params, forward, handles, n_params

    def close(self):
        for h in self.handles[::-1]:
            h.remove()
        self.handles = []


def surviving_blocks(variant: str, k: int = HEAL_K) -> list[int]:
    removed = read_json(M5 / f"excise_{SHORT}_{variant}.json")["removed_order"][:k]
    return [i for i in range(28) if i not in set(removed)]


def reft_layer(variant: str) -> int:
    return read_json(M4 / f"test_{SHORT}_{variant}_test.json")["cfg"]["layer"]


def reft_spec() -> dict:
    """The M5 rule-variant H3 spec, applied on both variants."""
    cand = read_json(M5 / f"select_{SHORT}_rule.json")["candidate"]
    if cand is None:
        raise SystemExit("no M5 rule H3 candidate for the reft arm")
    return cand["spec"]


def build_arm(ctx: Ctx, variant: str, arm: str, budget: str, seed: int) -> Arm:
    model, tok, ids = ctx.model, ctx.tok, ctx.ids
    torch.manual_seed(seed)
    full = lambda recs: adapters.full_logits(model, adapters.encode(model, tok, recs), ids)  # noqa: E731
    if arm == "lora":
        layers, modules, r = LORA[budget]
        h = adapters.add_lora(model, layers, modules, r)
        return Arm(h.modules, full, [h], adapters.count_trainable(model))
    if arm == "heal":
        n, modules, r = HEAL[budget]
        alive = surviving_blocks(variant)
        h = adapters.add_lora(model, alive if n is None else alive[-n:], modules, r)
        plan = structure.apply_plan(model, alive)
        return Arm(h.modules, full, [h, plan], adapters.count_trainable(model))
    if arm in ("reft", "reft0"):
        layer = reft_layer(variant)
        rows = None
        if arm == "reft":
            rows = ctx.directions(variant)[layer]["label_dm"]
        model.requires_grad_(False)
        module = adapters.LoReFT(model.config.hidden_size, REFT_RANK[budget], rows, seed).to(model.device)
        spec = apply_spec(model, reft_spec(), means=ctx.means(variant))
        edit = lambda: adapters.apply_loreft(model, layer, module)  # noqa: E731
        fwd = lambda recs: adapters.cached_logits(model, adapters.encode(model, tok, recs), ids, edit)  # noqa: E731
        return Arm(module, fwd, [spec], adapters.count_trainable(module))
    raise SystemExit(f"unknown arm {arm!r}")


def summarize(recs, pred, base_pred, n_boot: int) -> dict:
    y = [LABELS.index(r["label"]) for r in recs]
    groups = [r["template_id"] for r in recs]
    acc = stats.grouped_bootstrap(stats.accuracy, groups, y, pred, n_boot=n_boot)
    gain = stats.paired_grouped_bootstrap(stats.accuracy, groups, y, base_pred, pred, n_boot=n_boot)
    return dict(accuracy=acc["point"], accuracy_ci=acc, gain=gain["point"], gain_ci=gain, n=len(recs), pred=list(pred))


@torch.no_grad()
def predict(arm: Arm, recs, batch_size: int = 8) -> list[int]:
    out = [arm.forward(recs[i:i + batch_size]).argmax(-1).cpu() for i in range(0, len(recs), batch_size)]
    return torch.cat(out).tolist()


# ---------------------------------------------------------------- train

def run_train(ctx: Ctx, variant: str, arm_name: str, budget: str, seed: int) -> None:
    a = ctx.a
    rp = runs_path(variant, a.smoke)
    runs = read_json(rp) if rp.exists() else {"model": MODEL, "variant": variant, "runs": {}}
    key = run_key(arm_name, budget, seed)
    if key in runs["runs"]:
        print(f"skip {variant} {key}: already done", flush=True)
        return
    print(f"== {variant} {key} {now()}", flush=True)
    sel = ctx.records(variant, "selection")
    base_pred = ctx.baseline_pred(variant)
    heldout, train = ctx.train_heldout(variant, seed)
    steps = 4 if a.smoke else -(-len(train) // a.batch_size)
    arm = build_arm(ctx, variant, arm_name, budget, seed)
    try:
        t0 = time.perf_counter()
        state, curve = adapters.train_adapter(ctx.model, ctx.tok, train, heldout, arm.params, arm.forward, steps,
                                              batch_size=a.batch_size, seed=seed)
        train_s = time.perf_counter() - t0
        step_s = (train_s - curve[-1]["eval_seconds"]) / steps
        adapters.load_adapter_state(arm.params, state)
        cp = ckpt_path(variant, arm_name, budget, seed, a.smoke)
        cp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": state, "variant": variant, "arm": arm_name, "budget": budget, "seed": seed,
                    "n_params": arm.n_params}, cp.with_suffix(".tmp"))
        cp.with_suffix(".tmp").replace(cp)
        _free(ctx.model)
        t1 = time.perf_counter()
        pred = predict(arm, sel, a.batch_size)
        sel_s = time.perf_counter() - t1
    finally:
        arm.close()
        _free(ctx.model)
    res = summarize(sel, pred, base_pred, 2000 if a.smoke else 10_000)
    runs["runs"][key] = {"arm": arm_name, "budget": budget, "seed": seed, "n_params": arm.n_params, "steps": steps,
                         "curve": curve, "train_seconds": train_s, "step_seconds": step_s,
                         "selection_seconds": sel_s, "selection_seconds_per_example": sel_s / len(sel),
                         "checkpoint": str(cp), "selection": res, "ts": now()}
    write(rp, runs)
    if not a.smoke:
        append(SELECTION_LOG, {"ts": now(), "model": MODEL, "dtype": "fp32", "variant": variant,
                               "split": "selection", "seed": seed, "n": len(sel), "result": str(rp),
                               "intervention": f"m6-{arm_name}-{budget}-s{seed}",
                               "config": {"arm": arm_name, "budget": budget, "n_params": arm.n_params},
                               "accuracy": res["accuracy"]})
    print(f"{variant} {key}: params {arm.n_params} selection {res['accuracy']:.3f} gain {res['gain']:+.3f} "
          f"({step_s:.2f}s/step, {sel_s / len(sel) * 1000:.0f}ms/example, train {train_s:.0f}s, sel {sel_s:.0f}s)",
          flush=True)


def parse_until(text: str) -> datetime:
    hh, mm = text.rstrip("Zz").split(":")
    return datetime.now(timezone.utc).replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)


def past_deadline(until: datetime) -> bool:
    t = datetime.now(timezone.utc)
    if t >= until:
        return True
    return DEADLINE_FILE.exists() and t >= datetime.fromisoformat(DEADLINE_FILE.read_text().strip().replace("Z", "+00:00"))


def sweep_order():
    for seed in SEEDS:
        for variant in VARIANTS:
            for arm in ("heal", "lora", "reft"):
                for budget in BUDGETS:
                    yield variant, arm, budget, seed
        if seed == 0:
            for variant in VARIANTS:
                for budget in BUDGETS:
                    yield variant, "reft0", budget, seed


def stage_sweep(a):
    until = parse_until(a.until)
    ctx = Ctx(a)
    for variant, arm, budget, seed in sweep_order():
        if past_deadline(until):
            print(f"stopping: past {a.until} or {DEADLINE_FILE}", flush=True)
            return
        run_train(ctx, variant, arm, budget, seed)
    print("sweep complete", flush=True)


def stage_train(a):
    for need in ("variant", "arm", "budget"):
        if getattr(a, need) is None:
            sys.exit(f"--{need} is required for the train stage")
    run_train(Ctx(a), a.variant, a.arm, a.budget, a.seed)


# ---------------------------------------------------------------- test

def seed_stats(runs: dict, arm: str, budget: str) -> dict:
    rows = [r for r in runs["runs"].values() if r["arm"] == arm and r["budget"] == budget]
    accs = [r["selection"]["accuracy"] for r in rows]
    return {"mean": float(np.mean(accs)) if accs else None, "accs": accs, "n_seeds": len(accs),
            "gain": float(np.mean([r["selection"]["gain"] for r in rows])) if rows else None}


def ledger_has(variant: str, name: str) -> bool:
    if not TEST_LEDGER.exists():
        return False
    for line in TEST_LEDGER.read_text().splitlines():
        r = json.loads(line) if line.strip() else {}
        if r.get("model") == MODEL and r.get("variant") == variant and r.get("intervention") == name:
            return True
    return False


def load_trained(ctx: Ctx, variant: str, arm_name: str, budget: str, seed: int = 0) -> Arm:
    arm = build_arm(ctx, variant, arm_name, budget, seed)
    adapters.load_adapter_state(arm.params, torch.load(ckpt_path(variant, arm_name, budget, seed, ctx.a.smoke)
                                                       )["state"])
    return arm


def canaries_for(ctx: Ctx, arm: Arm, variant: str, arm_name: str, base_can) -> dict:
    items = load_canaries()[:: (16 if ctx.a.smoke else 1)]
    model = ctx.model
    if arm_name in ("reft", "reft0"):  # the LoReFT edit at the last prompt token of each right-padded row
        state = {"pos": None}
        h = adapters.apply_loreft(model, reft_layer(variant), arm.params, positions=lambda: state["pos"])
        try:
            got = score_canaries(model, ctx.tok, items, ctx.a.batch_size,
                                 on_batch=lambda n: state.__setitem__("pos", torch.tensor(n, dtype=torch.long) - 1))
        finally:
            h.remove()
    else:
        got = score_canaries(model, ctx.tok, items, ctx.a.batch_size)
    _free(model)
    ok, drops = eligible(base_can, got)
    return {"accuracy": got["accuracy"], "per_category": got["per_category"], "drops_pt": drops, "eligible": ok}


def stage_test(a):
    if a.which not in ("h2", "h5") or a.variant is None:
        sys.exit("--variant and --which h2|h5 are required for the test stage")
    v = a.variant
    runs = read_json(runs_path(v, a.smoke))
    models = []  # (ledger name, label, arm, budget)
    if a.which == "h2":
        best = max(BUDGETS, key=lambda b: (seed_stats(runs, "heal", b)["mean"] or -1))
        if seed_stats(runs, "heal", best)["mean"] is None:
            sys.exit("no completed heal runs")
        models = [(spec_hash(plan_spec(surviving_blocks(v))), "unhealed k=9", "excise", None),
                  (f"m6-heal-{best}-s0", f"heal {best}", "heal", best)]
        choice = {"best_heal_budget": best, "heal_selection": {b: seed_stats(runs, "heal", b) for b in BUDGETS}}
    else:
        pairs = []
        for b, b10 in (("10k", "100k"), ("100k", "1m")):
            rs, ls = seed_stats(runs, "reft", b), seed_stats(runs, "lora", b10)
            if rs["mean"] is not None and ls["mean"] is not None:
                pairs.append({"reft_budget": b, "lora_budget": b10, "reft": rs, "lora": ls,
                              "gap": ls["mean"] - rs["mean"]})
        if not pairs:
            sys.exit("no complete (reft, lora) pair")
        pick = min(pairs, key=lambda p: p["gap"])
        models = [(f"m6-reft-{pick['reft_budget']}-s0", f"reft {pick['reft_budget']}", "reft", pick["reft_budget"]),
                  (f"m6-lora-{pick['lora_budget']}-s0", f"lora {pick['lora_budget']}", "lora", pick["lora_budget"])]
        choice = {"pairs": pairs, "picked": pick}
    if not a.smoke:
        for name, label, _, _ in models:
            if ledger_has(v, name):
                print(f"refusing: test ledger already has {MODEL} {v} {name} ({label})", file=sys.stderr)
                sys.exit(2)
    ctx = Ctx(a)
    test = ctx.records(v, "test")
    base_pred = ctx.baseline_pred(v, "test")
    base_can = (score_canaries(ctx.model, ctx.tok, load_canaries()[::16], a.batch_size) if a.smoke
                else read_json(BASE_CANARIES)["canaries"])
    out = {"model": MODEL, "variant": v, "which": a.which, "split": "test", "n": len(test), "choice": choice,
           "baseline_accuracy": stats.accuracy([LABELS.index(r["label"]) for r in test], base_pred), "runs": {}}
    for name, label, arm_name, budget in models:
        t0 = time.perf_counter()
        if arm_name == "excise":
            plan = structure.apply_plan(ctx.model, surviving_blocks(v))
            try:
                lg = label_logits(ctx.model, ctx.tok, [r["prompt"] for r in test], a.batch_size)
            finally:
                plan.remove()
            pred, can = lg.argmax(-1).tolist(), None
            cfg = plan_spec(surviving_blocks(v))
        else:
            arm = load_trained(ctx, v, arm_name, budget)
            try:
                pred = predict(arm, test, a.batch_size)
                can = canaries_for(ctx, arm, v, arm_name, base_can)
            finally:
                arm.close()
            cfg = {"arm": arm_name, "budget": budget, "seed": 0}
        row = summarize(test, pred, base_pred, 2000 if a.smoke else 10_000)
        row.update(label=label, canaries=can, seconds=time.perf_counter() - t0)
        out["runs"][name] = row
        print(f"test {v} {label}: accuracy {row['accuracy']:.3f} (base {out['baseline_accuracy']:.3f}) "
              f"gain {row['gain']:+.3f}", flush=True)
        if not a.smoke:
            append(TEST_LEDGER, {"ts": now(), "model": MODEL, "dtype": "fp32", "variant": v, "split": "test",
                                 "seed": 0, "intervention": name, "config": cfg, "n": len(test),
                                 "accuracy": row["accuracy"],
                                 "result": str(OUT / f"test_{SHORT}_{v}_{a.which}.json")})
    out["verdict"] = verdict(a.which, out, choice, models)
    write(OUT / f"test_{SHORT}_{v}_{a.which}{smoke_suffix(a.smoke)}.json", out)
    print(f"{a.which} {v}: {out['verdict']}")


def verdict(which: str, out: dict, choice: dict, models) -> str:
    base = out["baseline_accuracy"]
    acc = [out["runs"][m[0]]["accuracy"] for m in models]
    if which == "h2":
        unhealed, healed = acc
        loss = base - unhealed
        out["unhealed_loss_pt"], out["recovered_pt"] = 100 * loss, 100 * (healed - unhealed)
        if healed >= base - 0.05:
            return "supported"
        return "falsified" if loss > 0 and (healed - unhealed) < 0.5 * loss else "not supported"
    reft, lora = acc
    pick = choice["picked"]
    out["test_lead_pt"] = 100 * (lora - reft)
    if reft >= lora - 0.05:
        return "supported"
    others = [p for p in choice["pairs"] if p is not pick]
    leads_all = all(p["gap"] > 0.05 for p in others)
    return "falsified" if lora - reft > 0.05 and leads_all else "not supported"


# ---------------------------------------------------------------- table

def fmt(x, pct: bool = True) -> str:
    return "n/a" if x is None else (f"{100 * x:.1f}" if pct else f"{x:.3f}")


def arm1_rows(variant: str):
    """(label, accuracy, gain) for the M5 rule H3 spec and the M4 steer reference, if their results exist."""
    rows = []
    sel_p, cross_p = M5 / f"select_{SHORT}_rule.json", M5 / f"cross_{SHORT}_rule.json"
    if sel_p.exists():
        cand = read_json(sel_p)["candidate"]
        if cand and variant == "rule":
            r = next((r for r in read_json(sel_p)["rows"] if r["spec_hash"] == cand["spec_hash"]), None)
            rows.append(("surgery H3 spec (M5)", r and r["accuracy"], r and r["gain"]))
        elif cand and cross_p.exists():
            r = next((r for r in read_json(cross_p)["rows"] if r["spec_hash"] == cand["spec_hash"]), None)
            rows.append(("surgery H3 spec (M5, transferred)", r and r["other_accuracy"], r and r["other_gain"]))
    t = M4 / f"test_{SHORT}_{variant}_test.json"
    if t.exists():
        cid = read_json(t)["cid"]
        for kind in ("grid", "tau"):
            p = M4 / f"{kind}_{SHORT}_{variant}_selection.json"
            s = p.exists() and next((s for s in read_json(p)["configs"] if s["cid"] == cid), None)
            if s:
                rows.append(("steer, 0 params (M4)", s["accuracy"], s["paired"]["point"]))
                break
    return rows


def stage_table(a):
    lines = ["# M6 adapter axis: selection-split tables", "",
             "Accuracy and gain in percentage points. Gain = adapted minus the unmodified model on the same examples.", ""]
    for v in VARIANTS:
        sel_p = M5 / f"select_{SHORT}_{v}.json"
        base = read_json(sel_p)["baseline"]["accuracy"] if sel_p.exists() else None
        lines += [f"## {v}", "", f"Unmodified baseline: {fmt(base)}", "",
                  "| arm | budget | params | seeds | mean acc | per-seed acc | mean gain |", "|---|---|---|---|---|---|---|"]
        for label, acc, gain in arm1_rows(v):
            lines.append(f"| {label} | - | - | - | {fmt(acc)} | - | {fmt(gain)} |")
        rp = runs_path(v, a.smoke)
        runs = read_json(rp) if rp.exists() else {"runs": {}}
        for arm in ("heal", "lora", "reft", "reft0"):
            for b in BUDGETS:
                rows = sorted((r for r in runs["runs"].values() if r["arm"] == arm and r["budget"] == b),
                              key=lambda r: r["seed"])
                if not rows:
                    lines.append(f"| {arm} | {b} | - | 0 | n/a | n/a | n/a |")
                    continue
                st = seed_stats(runs, arm, b)
                lines.append(f"| {arm} | {b} | {rows[0]['n_params']:,} | {st['n_seeds']} | {fmt(st['mean'])} | "
                             f"{', '.join(f's{r['seed']} {fmt(r['selection']['accuracy'])}' for r in rows)} | "
                             f"{fmt(st['gain'])} |")
        lines.append("")
        for which in ("h2", "h5"):
            tp = OUT / f"test_{SHORT}_{v}_{which}{smoke_suffix(a.smoke)}.json"
            if tp.exists():
                t = read_json(tp)
                lines += [f"### {which.upper()} on test ({v}): {t['verdict']}", "",
                          f"Baseline {fmt(t['baseline_accuracy'])}", "",
                          "| model | test acc | gain | canary acc | canary eligible |", "|---|---|---|---|---|"]
                for r in t["runs"].values():
                    c = r["canaries"]
                    lines.append(f"| {r['label']} | {fmt(r['accuracy'])} | {fmt(r['gain'])} | "
                                 f"{fmt(c and c['accuracy'])} | {c['eligible'] if c else 'n/a'} |")
                lines.append("")
    dst = OUT / f"report_tables{smoke_suffix(a.smoke)}.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n")
    print(f"-> {dst}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--stage", required=True, choices=["train", "sweep", "test", "table"])
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--budget", choices=BUDGETS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--until", default="10:40Z", help="sweep: no new run starts after this UTC time")
    ap.add_argument("--which", choices=["h2", "h5"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
