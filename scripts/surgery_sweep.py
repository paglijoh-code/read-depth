"""M5 structural-surgery sweep, one stage per process (CONTRACT.md M5).

  uv run python scripts/surgery_sweep.py --variant {rule,sensible} --stage
      {gate,excise,repswap,kvscan,kvpath,h3,select,test,cross,attrib} [--which h1|h3] [--batch-size 8] [--smoke]
Results go to var/results/m5/. Every stage is resumable (specs already in its file are skipped; a changed
data/identity is refused). Selection passes append to var/results/selection_log.jsonl and test runs to
var/results/test_ledger.jsonl; --smoke (16 examples, 2-3 steps, *_smoke.json) never writes either ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from surgery import stats, structure  # noqa: E402
from surgery.canaries import eligible, load_canaries, score_canaries  # noqa: E402
from surgery.eval import _free, label_logits, load_model  # noqa: E402
from surgery.hooks import apply_spec, spec_hash  # noqa: E402
from surgery.prompt import build_prompt  # noqa: E402
from surgery.subset import sweep_indices  # noqa: E402
from surgery.task.generator import LABELS, read_jsonl  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
SHORT = MODEL.split("/")[-1]
VARIANTS = ["rule", "sensible"]
DATA = Path("var/data/m2")
M3 = Path("var/results/m3")
M4 = Path("var/results/m4")
OUT = Path("var/results/m5")
BASE_CANARIES = Path("var/results/baseline") / f"{SHORT}_canaries.json"
SELECTION_LOG = Path("var/results/selection_log.jsonl")
TEST_LEDGER = Path("var/results/test_ledger.jsonl")
N_LAYERS = 28
KV_GROUPS = 8
EXCISE_STEPS = 10
EXCISE_CANDIDATES = 8
KV_STEP = 8
KV_STOP_LOSS = 0.10
SELECT_K = [2, 4, 6, 8, 10]
SELECT_KV = [16, 32, 48, 64, 96, 128]
MAX_H3_POINTS = 12
H3_MIN_GAIN = 0.05
H3_ADD_GAIN = 0.01


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def other_of(variant: str) -> str:
    return "sensible" if variant == "rule" else "rule"


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


def result_path(a, kind: str, variant: str | None = None, extra: str = "") -> Path:
    return OUT / f"{kind}_{SHORT}_{variant or a.variant}{extra}{'_smoke' if a.smoke else ''}.json"


# ---------------------------------------------------------------- data

class Data:
    """A record list with labels and template groups, plus the discovery-store checks of ablation_sweep.py."""

    def __init__(self, records, indices=None):
        self.indices = indices
        self.records = records
        self.y = [LABELS.index(r["label"]) for r in records]
        self.groups = [r["template_id"] for r in records]
        self.prompts = [r["prompt"] for r in records]
        self.ids = [r["id"] for r in records]


def load_store(variant: str):
    store = Path("var/acts") / f"{SHORT}_{variant}_s0_discovery"
    meta = read_json(store / "meta.json")
    if meta["model"] != MODEL or meta["variant"] != variant or meta["split"] != "discovery":
        raise ValueError("store model, variant, and discovery split must match the sweep")
    means = torch.load(store / "means.pt", map_location="cpu", weights_only=True)
    records = read_jsonl(Path(meta["data_path"]))
    if len(records) < meta["n"] or [r["id"] for r in records[:meta["n"]]] != meta["ids"]:
        raise ValueError("discovery records do not match the store's ids")
    return meta, means, records


def subset_indices(n_total: int, size: int) -> list[int]:
    """sub500 = sweep_indices(n, 500); sub200 is a seeded sample of sub500."""
    sub500 = sweep_indices(n_total, 500)
    if size == 500:
        return sub500
    return sorted(random.Random(1).sample(sub500, size))


def discovery(a, size: int):
    meta, means, records = load_store(a.variant)
    indices = sweep_indices(meta["n"], 16) if a.smoke else subset_indices(meta["n"], size)
    d = Data([records[i] for i in indices], indices)
    if (d.y != [meta["y"][i] for i in indices]
            or d.groups != [meta["template_ids"][i] for i in indices]):
        raise ValueError("discovery labels or template ids do not match the store")
    return d, means


def split_data(a, variant: str, split: str) -> Data:
    recs = read_jsonl(DATA / f"{variant}_s0_{split}.jsonl")
    return Data(recs[:16] if a.smoke else recs)


def identity(a, d: Data, split: str, name: str, variant: str | None = None) -> dict:
    return dict(model=MODEL, variant=variant or a.variant, split=split, dtype="fp32", stage=name, n=len(d.ids),
                indices=d.indices, ids=d.ids, template_ids=d.groups, y=d.y, batch_size=a.batch_size)


def mean_prompt_tokens(tok, a) -> float:
    recs = read_jsonl(DATA / f"{a.variant}_s0_selection.jsonl")
    return float(np.mean([len(tok(build_prompt(tok, r["prompt"]))["input_ids"]) for r in recs]))


# ---------------------------------------------------------------- cost of a spec

def spec_plan(spec: dict):
    """(order or None, pruned groups) a spec implies, for the FLOP and parameter counters."""
    op = spec["op"]
    if op == "plan":
        return list(spec["order"]), []
    if op == "skip":
        return [i for i in range(N_LAYERS) if i not in set(spec["layers"])], []
    if op == "ablate_kv":
        return None, [tuple(g) for g in spec["groups"]]
    if op == "compose":
        order, groups = None, []
        for s in spec["specs"]:
            o, g = spec_plan(s)
            order = o if o is not None else order
            groups += g
        return order, groups
    return None, []


def spec_cost(config, spec: dict, seq_len: float) -> dict:
    order, groups = spec_plan(spec)
    f = structure.flops_per_token(config, seq_len, order, groups)
    p = structure.param_count(config, order, groups)
    return {"flops_relative": f["relative"], "flops": f["total"], "params": p["total"],
            "params_unique": p["unique"]}


# ---------------------------------------------------------------- scoring

def correct_logp(logits, y):
    return logits.log_softmax(-1)[torch.arange(len(y)), torch.as_tensor(y)].numpy()


class Scorer:
    """Scores specs on one record list against its own baseline."""

    def __init__(self, model, tok, data: Data, batch_size: int, means, seq_len: float, n_boot: int = 2000):
        self.model, self.tok, self.d = model, tok, data
        self.batch_size, self.means, self.seq_len, self.n_boot = batch_size, means, seq_len, n_boot
        self.base_logits, self.baseline_seconds = self.logits(None)
        self.base_pred = self.base_logits.argmax(-1).tolist()
        self.base_logp = correct_logp(self.base_logits, data.y)

    def logits(self, spec):
        t0 = time.perf_counter()
        try:
            if spec is None:
                lg = label_logits(self.model, self.tok, self.d.prompts, self.batch_size)
            else:
                with apply_spec(self.model, spec, means=self.means):
                    lg = label_logits(self.model, self.tok, self.d.prompts, self.batch_size)
            return lg, time.perf_counter() - t0
        finally:
            _free(self.model)

    def baseline(self) -> dict:
        acc = stats.grouped_bootstrap(stats.accuracy, self.d.groups, self.d.y, self.base_pred, n_boot=self.n_boot)
        return dict(accuracy=acc["point"], accuracy_ci=acc, pred=self.base_pred,
                    mean_correct_label_log_prob=float(self.base_logp.mean()), seconds=self.baseline_seconds)

    def score(self, spec: dict) -> dict:
        lg, seconds = self.logits(spec)
        pred = lg.argmax(-1).tolist()
        d = self.d
        acc = stats.grouped_bootstrap(stats.accuracy, d.groups, d.y, pred, n_boot=self.n_boot)
        gain = stats.paired_grouped_bootstrap(stats.accuracy, d.groups, d.y, self.base_pred, pred,
                                              n_boot=self.n_boot)
        return dict(spec=spec, spec_hash=spec_hash(spec), accuracy=acc["point"], accuracy_ci=acc,
                    gain=gain["point"], gain_ci=gain,
                    logp_drop=float(np.mean(self.base_logp - correct_logp(lg, d.y))),
                    **spec_cost(self.model.config, spec, self.seq_len), seconds=seconds, pred=pred)


class Sweep:
    """A resumable results file: identity block, baseline, rows keyed by spec hash, stage-specific extras."""

    def __init__(self, path: Path, ident: dict, scorer: Scorer):
        self.path, self.scorer = path, scorer
        self.result = {**ident, "rows": []}
        if path.exists():
            self.result = read_json(path)
            if any(self.result.get(k) != v for k, v in ident.items()):
                raise ValueError(f"incompatible existing sweep at {path}; use the original settings")
        base = scorer.baseline()
        old = self.result.get("baseline")
        if old is not None and old["pred"] != base["pred"]:
            raise ValueError("baseline predictions changed; cannot resume this sweep")
        self.result["baseline"] = base
        self.save()
        self.by_hash = {r["spec_hash"]: r for r in self.result["rows"]}
        print(f"baseline accuracy {base['accuracy']:.3f} on n={len(scorer.d.ids)} ({base['seconds']:.1f}s)",
              flush=True)

    def save(self):
        write(self.path, self.result)

    def get(self, spec: dict, label: str = "") -> dict:
        """The row for `spec`, scoring it (and saving) if the file does not have it yet."""
        h = spec_hash(spec)
        if h not in self.by_hash:
            row = self.scorer.score(spec)
            self.result["rows"].append(row)
            self.by_hash[h] = row
            self.save()
            print(f"{label or spec_hash(spec)}: accuracy {row['accuracy']:.3f}, gain {row['gain']:+.3f}, "
                  f"logp drop {row['logp_drop']:+.3f}, flops {row['flops_relative']:.3f} "
                  f"({row['seconds']:.1f}s)", flush=True)
        return self.by_hash[h]


def damage_key(row):
    """Least damaging first: highest accuracy, tie -> smaller log-prob drop."""
    return (-row["accuracy"], row["logp_drop"])


def plan_spec(order, scales=None) -> dict:
    return {"op": "plan", "order": list(order), "scales": None if scales is None else list(scales)}


def kv_spec(groups) -> dict:
    return {"op": "ablate_kv", "mode": "mean", "groups": [list(g) for g in groups]}


def stage_context(a, size: int, name: str, extra: str = ""):
    """Loads data and model, runs the gate, builds the Scorer and the resumable Sweep for a discovery stage."""
    d, means = discovery(a, size)
    model, tok = load_model(MODEL)
    run_gate(a, model, tok, d)
    sc = Scorer(model, tok, d, a.batch_size, means, mean_prompt_tokens(tok, a))
    return d, means, model, tok, sc, Sweep(result_path(a, name, extra=extra), identity(a, d, "discovery", name), sc)


# ---------------------------------------------------------------- gate

def run_gate(a, model, tok, d: Data, write_result: bool = False) -> dict:
    prompts = d.prompts[:16]
    plain = label_logits(model, tok, prompts, a.batch_size)
    with apply_spec(model, plan_spec(range(N_LAYERS))):
        ident = label_logits(model, tok, prompts, a.batch_size)
    with apply_spec(model, {"op": "skip", "layers": [5]}):
        skipped = label_logits(model, tok, prompts, a.batch_size)
    with apply_spec(model, plan_spec([i for i in range(N_LAYERS) if i != 5])):
        dropped = label_logits(model, tok, prompts, a.batch_size)
    g = {"n": len(prompts), "identity_max_abs_diff": float((plain - ident).abs().max()),
         "drop5_vs_skip_max_abs_diff": float((skipped - dropped).abs().max())}
    g["passed"] = g["identity_max_abs_diff"] <= 1e-4 and g["drop5_vs_skip_max_abs_diff"] <= 1e-4
    print(f"gate: identity {g['identity_max_abs_diff']:.2e}, drop-5 vs skip hook "
          f"{g['drop5_vs_skip_max_abs_diff']:.2e} -> {'pass' if g['passed'] else 'FAIL'}", flush=True)
    if write_result:
        write(result_path(a, "gate"), {"model": MODEL, "variant": a.variant, "ts": now(), **g})
    if not g["passed"]:
        sys.exit("gate failed: layer-plan op does not match the plain forward / skip hook")
    return g


def stage_gate(a):
    d, _ = discovery(a, 500)
    model, tok = load_model(MODEL)
    run_gate(a, model, tok, d, write_result=True)


# ---------------------------------------------------------------- excise

def excise_candidates(removed, latest):
    """Step 1 scores every block; later steps the 8 remaining blocks that were least damaging at their most
    recent score (a block not re-scored keeps its earlier score, so the shortlist never runs dry)."""
    if latest is None:
        return list(range(N_LAYERS))
    left = [(b, r) for b, r in latest.items() if b not in removed]
    left.sort(key=lambda br: damage_key(br[1]))
    return [b for b, _ in left[:EXCISE_CANDIDATES]]


def excise_spec(removed):
    return plan_spec([i for i in range(N_LAYERS) if i not in set(removed)])


def stage_excise(a):
    d, means, model, tok, sc, sw = stage_context(a, 500, "excise")
    removed, latest, steps = [], None, []
    for step in range(1, (2 if a.smoke else a.excise_steps) + 1):
        cands = excise_candidates(removed, latest)
        rows = [(b, sw.get(excise_spec(removed + [b]), f"step {step} remove {b}")) for b in cands]
        best_b, best_row = min(rows, key=lambda br: damage_key(br[1]))
        removed = removed + [best_b]
        latest = {**(latest or {}), **dict(rows)}
        steps.append({"step": step, "removed_before": removed[:-1], "candidates": {str(b): r["spec_hash"] for b, r in rows},
                      "chosen": best_b, "spec_hash": best_row["spec_hash"], "accuracy": best_row["accuracy"]})
        print(f"step {step}: remove block {best_b} -> accuracy {best_row['accuracy']:.3f} "
              f"(removed {removed})", flush=True)
    sw.result["steps"], sw.result["removed_order"] = steps, removed
    sw.save()


# ---------------------------------------------------------------- repeat / swap

def repswap_specs(smoke: bool):
    blocks = range(6, 8) if smoke else range(6, 22)
    specs = []
    for i in blocks:
        order = list(range(i + 1)) + [i] + list(range(i + 1, N_LAYERS))
        for s in (1.0, 0.5):
            scales = None if s == 1.0 else [s if j == i + 1 else 1.0 for j in range(len(order))]
            specs.append(({"kind": "repeat", "block": i, "scale": s}, plan_spec(order, scales)))
    for i in (range(6, 8) if smoke else range(6, 21)):
        order = list(range(N_LAYERS))
        order[i], order[i + 1] = order[i + 1], order[i]
        specs.append(({"kind": "swap", "block": i}, plan_spec(order)))
    return specs


def plan_norms(model, tok, prompts, spec, means) -> list[float]:
    """Mean last-token residual norm after every position of the plan (raw block outputs). Hooks on the
    module objects fire once per call, so a repeated block records at each of its positions; the block inside
    a scaled wrapper is not recorded separately from the wrapper."""
    outs, depth = [], [0]

    def record(m, args, out):
        if depth[0] == 0:
            outs.append(float((out[0] if isinstance(out, tuple) else out)[:, -1].float().norm(dim=-1).mean()))
        return None

    def enter(m, args, kwargs):
        depth[0] += 1

    def leave(m, args, kwargs, out):
        depth[0] -= 1

    with apply_spec(model, spec, means=means):
        handles, seen = [], set()
        for layer in model.model.layers:
            if id(layer) in seen:
                continue
            seen.add(id(layer))
            if isinstance(layer, structure._Scaled):
                handles.append(layer.register_forward_pre_hook(enter, with_kwargs=True))
                handles.append(layer.register_forward_hook(leave, with_kwargs=True))
            handles.append(layer.register_forward_hook(record))
        try:
            with torch.no_grad():
                enc = tok([build_prompt(tok, p) for p in prompts], return_tensors="pt", padding=True).to(model.device)
                model(**enc, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
    _free(model)
    return outs


def stage_repswap(a):
    d, means, model, tok, sc, sw = stage_context(a, 200, "repswap")
    meta = sw.result.setdefault("meta", {})
    for info, spec in repswap_specs(a.smoke):
        row = sw.get(spec, f"{info['kind']} {info['block']}" + (f" x{info['scale']}" if "scale" in info else ""))
        meta[row["spec_hash"]] = info
        if info["kind"] == "repeat" and "norms" not in row:
            row["norms"] = plan_norms(model, tok, d.prompts[:32], spec, means)
            print(f"  norms at {len(row['norms'])} positions, last {row['norms'][-1]:.1f}", flush=True)
        sw.save()
    sw.save()


# ---------------------------------------------------------------- kv scan / path

def stage_kvscan(a):
    d, means, model, tok, sc, sw = stage_context(a, 200, "kvscan")
    for layer in range(1, 3 if a.smoke else N_LAYERS):
        for g in range(KV_GROUPS):
            sw.get(kv_spec([(layer, g)]), f"kv {layer}.{g}")
    sw.save()


def kv_order(a) -> list[list[int]]:
    """KV groups by real single-group loss ascending (tie -> smaller log-prob drop)."""
    rows = read_json(result_path(a, "kvscan"))["rows"]
    if not a.smoke and len(rows) != (N_LAYERS - 1) * KV_GROUPS:
        raise ValueError(f"kvscan has {len(rows)} groups, expected {(N_LAYERS - 1) * KV_GROUPS}")
    rows = sorted(rows, key=lambda r: (-r["gain"], r["logp_drop"]))
    return [r["spec"]["groups"][0] for r in rows]


def stage_kvpath(a):
    order = kv_order(a)
    d, means, model, tok, sc, sw = stage_context(a, 500, "kvpath")
    steps = []
    for n in range(KV_STEP, len(order) + 1, KV_STEP):
        row = sw.get(kv_spec(order[:n]), f"prune {n} groups")
        steps.append({"groups": n, "spec_hash": row["spec_hash"], "accuracy": row["accuracy"]})
        lost = sw.result["baseline"]["accuracy"] - row["accuracy"]
        if lost > KV_STOP_LOSS or (a.smoke and len(steps) >= 2):
            break
    sw.result["order"], sw.result["steps"] = order, steps
    sw.save()
    print(f"path: {len(steps)} steps, last {steps[-1]['groups']} groups at accuracy {steps[-1]['accuracy']:.3f}")


# ---------------------------------------------------------------- H3

def m3_singles(a):
    """M3 sublayer rows (always the full-sweep file; they only supply specs and discovery gains)."""
    p = M3 / f"ablation_{SHORT}_{a.variant}.json"
    if not p.exists():
        return []
    return [{"source": "m3", "spec": r["spec"], "gain": -r["accuracy_lost"], "lo": -r["accuracy_lost_ci"]["hi"]}
            for r in read_json(p)["rows"]]


def scan_singles(a, kind: str):
    p = result_path(a, kind)
    if not p.exists():
        return []
    res = read_json(p)
    meta = res.get("meta", {})
    return [{"source": kind, "spec": r["spec"], "gain": r["gain"], "lo": r["gain_ci"]["lo"],
             "info": meta.get(r["spec_hash"])} for r in res["rows"]]


def compose_spec(specs) -> dict:
    return specs[0] if len(specs) == 1 else {"op": "compose", "specs": list(specs)}


def compatible(chosen, spec) -> bool:
    """At most one plan per composition."""
    plans = sum(s["op"] == "plan" for s in chosen)
    return not (spec["op"] == "plan" and plans)


def stage_h3(a):
    cands = m3_singles(a) + scan_singles(a, "kvscan") + scan_singles(a, "repswap")
    cands.sort(key=lambda c: -c["gain"])
    if a.smoke:  # 16 examples cannot pass the gain filter; take the top 3 so the combination path runs
        cands = cands[:3]
    else:
        cands = [c for c in cands if c["gain"] >= H3_MIN_GAIN and c["lo"] > 0][:MAX_H3_POINTS]
    if not cands:
        write(result_path(a, "h3"), {"model": MODEL, "variant": a.variant, "singles": [], "combination": None,
                                     "reason": "no single intervention has discovery gain >= 5 pt with CI above 0"})
        print("no H3 candidates")
        return
    d, means, model, tok, sc, sw = stage_context(a, 500, "h3")
    # Discovery gain on the 500-example subset (M3 rows were scored on it already; the others are rescored).
    for c in cands:
        c["sub500"] = sw.get(c["spec"], f"single {c['source']}")
    cands.sort(key=lambda c: -c["sub500"]["gain"])
    singles = [{"source": c["source"], "spec": c["spec"], "spec_hash": c["sub500"]["spec_hash"],
                "gain_sub500": c["sub500"]["gain"], "gain_ci_sub500": c["sub500"]["gain_ci"],
                "gain_origin": c["gain"], "info": c.get("info")} for c in cands]
    chosen = [cands[0]["spec"]]
    current = cands[0]["sub500"]
    path = [{"added": cands[0]["spec"], "spec_hash": current["spec_hash"], "accuracy": current["accuracy"],
             "accepted": True}]
    for c in cands[1:]:
        if not compatible(chosen, c["spec"]):
            path.append({"added": c["spec"], "accepted": False, "reason": "second plan"})
            continue
        trial = sw.get(compose_spec(chosen + [c["spec"]]), "combination + 1")
        ok = trial["accuracy"] - current["accuracy"] >= H3_ADD_GAIN
        path.append({"added": c["spec"], "spec_hash": trial["spec_hash"], "accuracy": trial["accuracy"],
                     "accepted": ok})
        if ok:
            chosen, current = chosen + [c["spec"]], trial
    sw.result["singles"], sw.result["path"] = singles, path
    sw.result["combination"] = {"spec": compose_spec(chosen), "spec_hash": spec_hash(compose_spec(chosen)),
                                "n_parts": len(chosen), "accuracy": current["accuracy"], "gain": current["gain"]}
    sw.save()
    print(f"{len(singles)} singles; combination of {len(chosen)}: accuracy {current['accuracy']:.3f} "
          f"gain {current['gain']:+.3f}")


# ---------------------------------------------------------------- select

def excise_removed(a) -> list[int]:
    return read_json(result_path(a, "excise"))["removed_order"]


def best_by_accuracy(rows):
    return min(rows, key=damage_key)


def select_points(a) -> list[dict]:
    """The Pareto list of the contract + H3 candidates, each with a spec and a label."""
    pts = []
    removed = excise_removed(a)
    for k in SELECT_K:
        if k <= len(removed):
            pts.append({"label": f"excise k={k}", "kind": "excise", "k": k, "spec": excise_spec(removed[:k])})
    path = read_json(result_path(a, "kvpath"))
    for n in SELECT_KV:
        if n <= path["steps"][-1]["groups"]:
            pts.append({"label": f"kv {n} groups", "kind": "kv", "groups": n, "spec": kv_spec(path["order"][:n])})
    rs = read_json(result_path(a, "repswap"))
    meta = rs["meta"]
    for kind in ("repeat", "swap"):
        rows = [r for r in rs["rows"] if meta[r["spec_hash"]]["kind"] == kind]
        if rows:
            best = best_by_accuracy(rows)
            info = meta[best["spec_hash"]]
            pts.append({"label": f"best {kind}", "kind": kind, "info": info, "spec": best["spec"]})
    h3p = result_path(a, "h3")
    if h3p.exists():
        h3 = read_json(h3p)
        h3pts = ([{"label": "h3 combination", "kind": "h3", "h3": True, "spec": h3["combination"]["spec"]}]
                 if h3["combination"] else [])
        for s in h3["singles"]:
            h3pts.append({"label": f"h3 single {s['source']}", "kind": "h3", "h3": True, "spec": s["spec"]})
        pts += h3pts[:MAX_H3_POINTS]
    seen, out = set(), []
    for p in pts:  # the same spec can be a Pareto point and an H3 candidate: score it once, keep both tags
        h = spec_hash(p["spec"])
        p["spec_hash"] = h
        if h in seen:
            for q in out:
                if q["spec_hash"] == h:
                    q["h3"] = q.get("h3") or p.get("h3", False)
            continue
        seen.add(h)
        out.append(p)
    return out


def selection_log(a, d: Data, spec, name, acc):
    if a.smoke:
        return
    append(SELECTION_LOG, {"ts": now(), "model": MODEL, "dtype": "fp32", "variant": a.variant, "split": "selection",
                           "seed": 0, "n": len(d.ids), "result": "", "intervention": name, "config": spec,
                           "accuracy": acc})


def stage_select(a):
    points = select_points(a)
    d = split_data(a, a.variant, "selection")
    model, tok = load_model(MODEL)
    run_gate(a, model, tok, d)
    means = load_store(a.variant)[1]
    sc = Scorer(model, tok, d, a.batch_size, means, mean_prompt_tokens(tok, a), n_boot=10_000)
    sw = Sweep(result_path(a, "select"), identity(a, d, "selection", "select"), sc)
    if "logged" not in sw.result:
        sw.result["logged"] = []
        selection_log(a, d, "baseline", "baseline", sw.result["baseline"]["accuracy"])
        sw.save()
    for p in points:
        row = sw.get(p["spec"], p["label"])
        if row["spec_hash"] not in sw.result["logged"]:
            selection_log(a, d, p["spec"], row["spec_hash"], row["accuracy"])
            sw.result["logged"].append(row["spec_hash"])
            sw.save()
    sw.result["points"] = [{k: v for k, v in p.items()} for p in points]
    sw.save()
    # Canaries for every point, with the M3 gate.
    items = load_canaries()[:: (16 if a.smoke else 1)]
    if a.smoke:
        base_can = score_canaries(model, tok, items, a.batch_size)
    else:
        base_can = read_json(BASE_CANARIES)["canaries"]
    can = sw.result.setdefault("canaries", {})
    sw.result["canary_baseline"] = {"accuracy": base_can["accuracy"], "per_category": base_can["per_category"],
                                    "n": base_can["n"]}
    for p in points:
        if p["spec_hash"] in can:  # re-apply the gate to cached scores, so a gate fix takes effect
            can[p["spec_hash"]]["eligible"], can[p["spec_hash"]]["drops_pt"] = eligible(base_can, can[p["spec_hash"]])
            continue
        t0 = time.perf_counter()
        with apply_spec(model, p["spec"], means=means):
            got = score_canaries(model, tok, items, a.batch_size)
        _free(model)
        ok, drops = eligible(base_can, got)
        can[p["spec_hash"]] = {"accuracy": got["accuracy"], "per_category": got["per_category"],
                               "drops_pt": drops, "eligible": ok, "seconds": time.perf_counter() - t0}
        sw.save()
        print(f"canary {p['label']}: {got['accuracy']:.3f} eligible={ok} ({time.perf_counter() - t0:.1f}s)", flush=True)
    cand, reason = None, "no H3 point is eligible with a selection paired CI above 0"
    ranked = sorted((p for p in points if p.get("h3")),
                    key=lambda p: -sw.by_hash[p["spec_hash"]]["accuracy"])
    for p in ranked:
        row = sw.by_hash[p["spec_hash"]]
        if can[p["spec_hash"]]["eligible"] and row["gain_ci"]["lo"] > 0:
            cand, reason = {"label": p["label"], "spec": p["spec"], "spec_hash": p["spec_hash"],
                            "selection_accuracy": row["accuracy"], "selection_gain": row["gain"],
                            "selection_gain_ci": row["gain_ci"]}, None
            break
    if not ranked:
        reason = "no H3 candidates"
    sw.result["candidate"], sw.result["candidate_reason"] = cand, reason
    sw.save()
    print("H3 candidate:", cand["label"] if cand else None, reason or "")


# ---------------------------------------------------------------- test

def ledger_has(variant: str, spec_h: str) -> bool:
    if not TEST_LEDGER.exists():
        return False
    for line in TEST_LEDGER.read_text().splitlines():
        r = json.loads(line) if line.strip() else {}
        if r.get("model") == MODEL and r.get("variant") == variant and r.get("intervention") == spec_h:
            return True
    return False


def test_run(a, model, tok, d, means, sc, spec, label, store: dict, dst: Path) -> dict:
    h = spec_hash(spec)
    if h in store:
        return store[h]
    if not a.smoke and ledger_has(a.variant, h):
        print(f"refusing: test ledger already has {MODEL} {a.variant} {h}", file=sys.stderr)
        sys.exit(2)
    row = sc.score(spec)
    row["label"] = label
    store[h] = row
    if not a.smoke:
        append(TEST_LEDGER, {"ts": now(), "model": MODEL, "dtype": "fp32", "variant": a.variant, "split": "test",
                             "seed": 0, "intervention": h, "config": spec, "n": len(d.ids),
                             "accuracy": row["accuracy"], "result": str(dst)})
    print(f"test {label}: accuracy {row['accuracy']:.3f} (base {sc.baseline()['accuracy']:.3f}) gain "
          f"{row['gain']:+.3f} [{row['gain_ci']['lo']:+.3f}, {row['gain_ci']['hi']:+.3f}]", flush=True)
    return row


def stage_test(a):
    if a.which not in ("h1", "h3"):
        sys.exit("--which h1|h3 is required for the test stage")
    sel = None
    if a.which == "h3":
        sel = read_json(result_path(a, "select"))
        if sel["candidate"] is None:
            sys.exit(f"no H3 candidate for {a.variant} ({sel['candidate_reason']}): "
                     "not supported in M5, no test run")
    d = split_data(a, a.variant, "test")
    model, tok = load_model(MODEL)
    run_gate(a, model, tok, d)
    means = load_store(a.variant)[1]
    dst = result_path(a, "test", extra=f"_{a.which}")
    sc = Scorer(model, tok, d, a.batch_size, means, mean_prompt_tokens(tok, a), n_boot=10_000)
    out = read_json(dst) if dst.exists() else {}
    if out and (out.get("ids") != d.ids or out.get("batch_size") != a.batch_size):
        raise ValueError(f"incompatible existing test file {dst}")
    out = {**identity(a, d, "test", f"test_{a.which}"), "baseline": sc.baseline(), "runs": out.get("runs", {})}
    runs = out["runs"]
    if a.which == "h1":
        removed = excise_removed(a)
        row6 = test_run(a, model, tok, d, means, sc, excise_spec(removed[:6]), "excise k=6", runs, dst)
        loss6 = -row6["gain"]
        out["k6_loss_pt"] = 100 * loss6
        if loss6 <= 0.05:
            out["verdict"] = "supported"
        else:
            row5 = test_run(a, model, tok, d, means, sc, excise_spec(removed[:5]), "excise k=5", runs, dst)
            out["k5_loss_pt"] = 100 * -row5["gain"]
            out["verdict"] = "falsified" if -row5["gain"] > 0.10 else "not supported"
    else:
        cand = sel["candidate"]
        row = test_run(a, model, tok, d, means, sc, cand["spec"], f"H3 {cand['label']}", runs, dst)
        out["gain_pt"] = 100 * row["gain"]
        out["verdict"] = ("supported" if row["gain"] >= 0.05 and row["gain_ci"]["lo"] > 0 else
                          "falsified" if row["gain"] <= 0 else "not supported")
    write(dst, out)
    print(f"{a.which} {a.variant}: {out['verdict']} -> {dst}")


# ---------------------------------------------------------------- cross / anatomy / pareto

def anatomy_specs(a, sel: dict) -> list[dict]:
    """The spec rows of the anatomy matrix for this variant (the M4 steer row is handled separately)."""
    pts = {p["spec_hash"]: p for p in sel["points"]}
    rows = sel["rows"]
    by_hash = {r["spec_hash"]: r for r in rows}
    out = []
    k6 = [p for p in pts.values() if p["kind"] == "excise" and p.get("k") == 6]
    if k6:
        out.append({**k6[0], "label": "excise k=6"})
    base = sel["baseline"]["accuracy"]
    kv = [p for p in pts.values() if p["kind"] == "kv" and base - by_hash[p["spec_hash"]]["accuracy"] <= 0.05]
    if kv:
        out.append({**max(kv, key=lambda p: p["groups"]), "label": "largest KV step within 5 pt"})
    ra = [p for p in pts.values() if p["kind"] in ("repeat", "swap")]
    if ra:
        out.append({**max(ra, key=lambda p: by_hash[p["spec_hash"]]["accuracy"]), "label": "best repeat/swap"})
    if sel.get("candidate"):
        c = sel["candidate"]
        out.append({"label": "H3 candidate", "kind": "h3", "spec": c["spec"], "spec_hash": c["spec_hash"]})
    return out


def stage_cross(a):
    sel_path = result_path(a, "select")
    sel = read_json(sel_path)
    other = other_of(a.variant)
    specs = anatomy_specs(a, sel)
    model = tok = None
    cross = {"model": MODEL, "variant": a.variant, "other": other, "rows": []}
    if specs:
        d = split_data(a, other, "selection")
        model, tok = load_model(MODEL)
        means = load_store(a.variant)[1]
        sc = Scorer(model, tok, d, a.batch_size, means, mean_prompt_tokens(tok, a), n_boot=10_000)
        cross["other_baseline_accuracy"] = sc.baseline()["accuracy"]
        for s in specs:
            row = sc.score(s["spec"])
            cross["rows"].append({"label": s["label"], "spec": s["spec"], "spec_hash": s["spec_hash"],
                                  "other_accuracy": row["accuracy"], "other_gain": row["gain"],
                                  "other_gain_ci": row["gain_ci"]})
            print(f"cross {s['label']} on {other}: gain {row['gain']:+.3f}", flush=True)
    tp = M4 / f"transfer_{SHORT}_{a.variant}.json"
    if tp.exists():  # the M4 steer: this variant's config and directions on the other variant's selection split
        t = read_json(tp)
        cross["steer"] = {"cfg": t["cfg"], "source": t["config_source"], "other_gain": t["summary"]["paired"]["point"],
                          "other_gain_ci": t["summary"]["paired"]}
    write(result_path(a, "cross"), cross)
    write_pareto(a, sel)
    write_anatomy(a)


def write_pareto(a, sel: dict):
    by_hash = {r["spec_hash"]: r for r in sel["rows"]}
    pts = []
    for p in sel["points"]:
        r, c = by_hash[p["spec_hash"]], sel["canaries"][p["spec_hash"]]
        pts.append({"label": p["label"], "kind": p["kind"], "k": p.get("k"), "groups": p.get("groups"),
                    "spec_hash": p["spec_hash"], "flops_relative": r["flops_relative"], "params": r["params"],
                    "selection_accuracy": r["accuracy"], "selection_accuracy_ci": r["accuracy_ci"],
                    "selection_gain": r["gain"], "canary_accuracy": c["accuracy"],
                    "canary_per_category": c["per_category"], "eligible": c["eligible"]})
    write(result_path(a, "pareto"), {"model": MODEL, "variant": a.variant,
                                     "baseline_accuracy": sel["baseline"]["accuracy"],
                                     "canary_baseline": sel["canary_baseline"], "points": pts})


def write_anatomy(a):
    """Panel 6: written once both variants have their select and cross files."""
    files = {v: (result_path(a, "select", v), result_path(a, "cross", v)) for v in VARIANTS}
    if not all(p.exists() for pair in files.values() for p in pair):
        print("anatomy: waiting for both variants' select and cross files")
        return
    rows = []
    for v in VARIANTS:
        sel, cross = read_json(files[v][0]), read_json(files[v][1])
        by_hash = {r["spec_hash"]: r for r in sel["rows"]}
        cb = sel["canary_baseline"]["per_category"]

        def deltas(own_gain, other_gain):
            pts = lambda g: None if g is None else 100 * g  # noqa: E731
            return {f"delta_{v}_pts": pts(own_gain), f"delta_{cross['other']}_pts": pts(other_gain)}
        for x in cross["rows"]:
            r, c = by_hash[x["spec_hash"]], sel["canaries"][x["spec_hash"]]
            rows.append({"label": x["label"], "variant": v, "spec": x["spec"], **deltas(r["gain"], x["other_gain"]),
                         "delta_canary_pts": {k: 100 * (c["per_category"][k] - cb[k]) for k in cb},
                         "delta_canary_overall_pts": 100 * (c["accuracy"] - sel["canary_baseline"]["accuracy"])})
        st = cross.get("steer")
        can = M4 / f"canary_{SHORT}_{v}.json"
        cand = read_json(can)["candidate"] if can.exists() else None
        if st:
            rows.append({"label": "M4 steer", "variant": v, "cfg": st["cfg"],
                         **deltas(cand["selection_paired"]["point"] if cand else None, st["other_gain"]),
                         "delta_canary_pts": ({k: 100 * (cand["per_category"][k] - cb[k]) for k in cb} if cand else None)})
    write(OUT / f"anatomy_{SHORT}{'_smoke' if a.smoke else ''}.json", {"model": MODEL, "rows": rows})


# ---------------------------------------------------------------- attribution check

def spearman(x, y) -> dict:
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(x)}


def stage_attrib(a):
    att = read_json(M3 / f"attribution_{SHORT}_{a.variant}.json")
    heads = np.array(att["heads"])
    kv = np.array(att["kv"])
    out = {"model": MODEL, "variant": a.variant, "ts": now(), "gate_rho": 0.5}
    ch = read_json(M3 / f"ablation_{SHORT}_{a.variant}_heads.json")["rows"]
    ch = [r for r in ch if r["spec"]["heads"][0][0] != 0]
    xs = [heads[tuple(r["spec"]["heads"][0])] for r in ch]
    out["heads_excluding_layer0"] = {
        "log_prob": spearman(xs, [r["mean_correct_label_log_prob_drop"] for r in ch]),
        "accuracy_lost": spearman(xs, [r["accuracy_lost"] for r in ch])}
    ks = read_json(result_path(a, "kvscan"))["rows"]
    xs = [kv[tuple(r["spec"]["groups"][0])] for r in ks]
    out["kv_groups"] = {"log_prob": spearman(xs, [r["logp_drop"] for r in ks]),
                        "accuracy_lost": spearman(xs, [-r["gain"] for r in ks])}
    for name in ("heads_excluding_layer0", "kv_groups"):
        out[name]["gate_passed"] = out[name]["log_prob"]["rho"] >= 0.5
        print(f"{a.variant} {name}: rho(log-prob) {out[name]['log_prob']['rho']:+.2f} n={out[name]['log_prob']['n']}, "
              f"rho(accuracy lost) {out[name]['accuracy_lost']['rho']:+.2f}, gate "
              f"{'pass' if out[name]['gate_passed'] else 'fail (report only)'}")
    write(result_path(a, "attrib"), out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--stage", required=True,
                    choices=["gate", "excise", "repswap", "kvscan", "kvpath", "h3", "select", "test", "cross", "attrib"])
    ap.add_argument("--which", choices=["h1", "h3"], help="test stage: which hypothesis")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--excise-steps", type=int, default=EXCISE_STEPS,
                    help="excise stage: path length (M7 extends it to 18; earlier steps come from the cache)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.batch_size < 1:
        ap.error("--batch-size must be positive")
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    print(f"{a.stage} {a.variant}{' (smoke)' if a.smoke else ''} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
