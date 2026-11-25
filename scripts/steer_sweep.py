"""M4 steering sweep, one stage per process.

  uv run python scripts/steer_sweep.py --variant rule --stage {directions,grid,tau,canary,test,transfer}
      [--model Qwen/Qwen3-0.6B] [--n N] [--smoke]
Outputs go to var/results/m4/. Selection passes append to var/results/selection_log.jsonl and the
single test run to var/results/test_ledger.jsonl (CONTRACT.md M4). --smoke never writes either ledger.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from surgery import steer  # noqa: E402
from surgery.canaries import eligible  # noqa: E402
from surgery.eval import load_model  # noqa: E402
from surgery.task.generator import LABELS, read_jsonl  # noqa: E402

LAYERS = [6, 9, 12, 14, 16, 18, 20, 22]
ALPHAS = [-0.1, 0.05, 0.1, 0.2, 0.4]
TAUS = [0.3, 0.5, 0.7]
DATA = Path("var/data/m2")
OUT = Path("var/results/m4")
BASE = Path("var/results/baseline")
SELECTION_LOG = "var/results/selection_log.jsonl"
TEST_LEDGER = "var/results/test_ledger.jsonl"
VARIANTS = ["rule", "sensible"]


def short(model: str) -> str:
    return model.split("/")[-1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(path: str, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def out_dir(a) -> Path:
    return OUT / "smoke" if a.smoke else OUT


def dest(a, kind: str, variant: str | None = None, split: str | None = None) -> Path:
    name = f"{kind}_{short(a.model)}_{variant or a.variant}" + (f"_{split}" if split else "")
    return out_dir(a) / f"{name}.json"


def dir_path(a, variant: str) -> Path:
    return OUT / f"directions_{short(a.model)}_{variant}.pt"


def store_path(a, variant: str) -> Path:
    return Path("var/acts") / f"{short(a.model)}_{variant}_s0_discovery"


def get_directions(a, variant: str) -> dict:
    if a.smoke:
        return steer.build_directions(store_path(a, variant), [12])
    return steer.load_directions(dir_path(a, variant))


def load_split(a, variant: str, split: str) -> list[dict]:
    return read_jsonl(DATA / f"{variant}_s0_{split}.jsonl")[: (16 if a.smoke else a.n)]


def header(a, split: str, n: int, t0: float, variant: str | None = None) -> dict:
    return {"model": a.model, "variant": variant or a.variant, "split": split, "n": n,
            "wall_seconds": time.time() - t0, "ts": now()}


def grid_configs(layers, alphas) -> list[dict]:
    cfgs = []
    for l in layers:
        for d in steer.DIRECTIONS:
            for al in alphas:
                cfgs.append({"layer": l, "direction": d, "op": "add", "alpha": al, "tau": None})
        for d in ("corr_dm", "corr_probe"):
            cfgs.append({"layer": l, "direction": d, "op": "project_out", "alpha": None, "tau": None})
    return cfgs


def run_configs(model, tok, recs, cfgs, D, a, variant: str, split: str, t0: float, log: bool):
    """Evaluate configs on recs; returns (baseline summary, per-config summaries sorted by accuracy)."""
    res = steer.eval_configs(model, tok, recs, cfgs, D, a.batch_size)
    base = steer.summarize_config(recs, res["baseline"], res["baseline"])
    sums = []
    for cfg in cfgs:
        cid = steer.config_id(cfg)
        s = steer.summarize_config(recs, res["configs"][cid], res["baseline"])
        s.update(cfg=cfg, cid=cid, steered_frac=res["steered_frac"][cid])
        sums.append(s)
    sums.sort(key=lambda s: -s["accuracy"])
    if log:
        row = {"ts": now(), "model": a.model, "dtype": "fp32", "variant": variant, "split": split,
               "seed": 0, "n": len(recs), "result": ""}
        append(SELECTION_LOG, {**row, "intervention": "baseline", "accuracy": base["accuracy"]})
        for s in sums:
            append(SELECTION_LOG, {**row, "intervention": s["cid"], "config": s["cfg"],
                                   "accuracy": s["accuracy"]})
    return base, sums


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    print(f"-> {path}")


def stage_directions(a):
    t0 = time.time()
    D = steer.build_directions(store_path(a, a.variant), LAYERS)
    steer.save_directions(D, dir_path(a, a.variant))
    print(f"directions {a.variant}: layers {LAYERS}, n={D['meta']['n']} ({time.time() - t0:.0f}s) "
          f"-> {dir_path(a, a.variant)}")


def stage_grid(a):
    t0 = time.time()
    D = get_directions(a, a.variant)
    recs = load_split(a, a.variant, "selection")
    layers = [12] if a.smoke else LAYERS
    cfgs = grid_configs(layers, [0.1] if a.smoke else ALPHAS)
    model, tok = load_model(a.model)
    g = recs[:64]
    plain = steer.plain_label_logits(model, tok, g, a.batch_size).numpy()
    two = steer.eval_configs(model, tok, g, [], D, a.batch_size)["baseline"]
    gate = {"n": len(g), "max_abs_diff": float(np.abs(plain - two).max()),
            "pred_agreement": float((plain.argmax(1) == two.argmax(1)).mean())}
    print(f"equality gate: max |d| {gate['max_abs_diff']:.2e}, agreement {gate['pred_agreement']:.4f}")
    base, sums = run_configs(model, tok, recs, cfgs, D, a, a.variant, "selection", t0, not a.smoke)
    out = {**header(a, "selection", len(recs), t0), "other_gpu_jobs": None, "gate": gate,
           "baseline": base, "configs": sums}
    write(dest(a, "grid", split="selection"), out)
    print(f"baseline {base['accuracy']:.3f}; best {sums[0]['accuracy']:.3f} {sums[0]['cfg']}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def stage_tau(a):
    t0 = time.time()
    grid = read_json(dest(a, "grid", split="selection"))
    top = [s["cfg"] for s in grid["configs"][:5]]
    cfgs = [{**c, "tau": t} for c in top for t in (TAUS[:1] if a.smoke else TAUS)]
    D = get_directions(a, a.variant)
    recs = load_split(a, a.variant, "selection")
    model, tok = load_model(a.model)
    base, sums = run_configs(model, tok, recs, cfgs, D, a, a.variant, "selection", t0, not a.smoke)
    write(dest(a, "tau", split="selection"),
          {**header(a, "selection", len(recs), t0), "other_gpu_jobs": None, "baseline": base, "configs": sums})
    print(f"best conditional {sums[0]['accuracy']:.3f} {sums[0]['cfg']}")


def stage_canary(a):
    from surgery.canaries import load_canaries
    t0 = time.time()
    pool = {}
    for kind in ("grid", "tau"):
        p = dest(a, kind, split="selection")
        if p.exists():
            for s in read_json(p)["configs"]:
                pool[s["cid"]] = s
    if not pool:
        sys.exit("no grid/tau files to pool; run the grid stage first")
    ranked = sorted(pool.values(), key=lambda s: -s["accuracy"])
    base = read_json(BASE / f"{short(a.model)}_canaries.json")["canaries"]
    D = get_directions(a, a.variant)
    items = load_canaries()[: (32 if a.smoke else None)]
    model, tok = load_model(a.model)
    scored, first = [], None
    for s in ranked[:15]:
        got = steer.score_canaries_steered(model, tok, items, s["cfg"], D, a.batch_size)
        ok, drops = eligible(base, got)
        scored.append({"cid": s["cid"], "cfg": s["cfg"], "selection_accuracy": s["accuracy"],
                       "selection_paired": s["paired"], "canary_accuracy": got["accuracy"],
                       "per_category": got["per_category"], "drops_pt": drops, "eligible": ok})
        print(f"{s['cid']} sel {s['accuracy']:.3f} canary {got['accuracy']:.3f} eligible={ok}")
        if ok:
            first = scored[-1]
            break
    if first is None:
        cand, reason = None, "no eligible config among the top 15"
    elif first["selection_paired"]["lo"] > 0:
        cand, reason = first, None
    else:
        cand, reason = None, "best eligible config's selection paired CI includes 0"
    write(dest(a, "canary"), {**header(a, "canaries", len(items), t0), "other_gpu_jobs": None,
                              "baseline_canary_accuracy": base["accuracy"], "scored": scored,
                              "candidate": cand, "reason": reason})
    print("candidate:", cand["cfg"] if cand else None, reason or "")


def stage_test(a):
    t0 = time.time()
    if a.cid:  # a named config from the grid/tau pool instead of the canary candidate
        pool = {}
        for kind in ("grid", "tau"):
            p = dest(a, kind, split="selection")
            if p.exists():
                for s_ in read_json(p)["configs"]:
                    pool[s_["cid"]] = s_["cfg"]
        if a.cid not in pool:
            sys.exit(f"config {a.cid} not in the {a.variant} grid/tau pool")
        cfg, cid = pool[a.cid], a.cid
    else:
        cand = read_json(dest(a, "canary"))["candidate"]
        if cand is None:
            sys.exit(f"no candidate for {a.variant}: not supported in M4, no test run")
        cfg, cid = cand["cfg"], cand["cid"]
    ledger = Path(TEST_LEDGER)
    if not a.smoke and ledger.exists():
        for line in ledger.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get("model") == a.model and r.get("variant") == a.variant and r.get("intervention") == cid:
                print(f"refusing: test ledger already has {a.model} {a.variant} {cid}", file=sys.stderr)
                sys.exit(2)
    D = get_directions(a, a.variant)
    recs = load_split(a, a.variant, "test")
    model, tok = load_model(a.model)
    base, sums = run_configs(model, tok, recs, [cfg], D, a, a.variant, "test", t0, False)
    s = sums[0]
    h4 = "supported" if s["paired"]["lo"] > 0 else "falsified for this variant"
    out = {**header(a, "test", len(recs), t0), "other_gpu_jobs": None, "cid": cid, "cfg": cfg,
           "baseline": base, "summary": s, "h4": h4}
    dst = dest(a, "test", split=f"test_{cid}" if a.cid else "test")
    write(dst, out)
    if not a.smoke:
        append(TEST_LEDGER, {"ts": now(), "model": a.model, "dtype": "fp32", "variant": a.variant,
                             "split": "test", "seed": 0, "intervention": cid, "config": cfg,
                             "n": len(recs), "accuracy": s["accuracy"], "result": str(dst)})
    print(f"test {a.variant}: base {base['accuracy']:.3f} steered {s['accuracy']:.3f} "
          f"paired {s['paired']['point']:+.3f} [{s['paired']['lo']:+.3f}, {s['paired']['hi']:+.3f}] h4: {h4}")


def stage_transfer(a):
    t0 = time.time()
    other = "sensible" if a.variant == "rule" else "rule"
    cp = dest(a, "canary")
    cand = read_json(cp)["candidate"] if cp.exists() else None
    if cand is not None:
        cfg, source = cand["cfg"], "candidate"
    else:
        cfg, source = read_json(dest(a, "grid", split="selection"))["configs"][0]["cfg"], "best grid config"
    D = get_directions(a, a.variant)  # this variant's directions on the other variant's data
    recs = load_split(a, other, "selection")
    model, tok = load_model(a.model)
    base, sums = run_configs(model, tok, recs, [cfg], D, a, other, "selection", t0, not a.smoke)
    out = {**header(a, "selection", len(recs), t0, variant=other), "other_gpu_jobs": None,
           "config_from_variant": a.variant, "config_source": source, "directions_from": a.variant,
           "cfg": cfg, "baseline": base, "summary": sums[0]}
    write(dest(a, "transfer"), out)
    s = sums[0]
    print(f"transfer {a.variant} cfg and directions on {other}: base {base['accuracy']:.3f} steered {s['accuracy']:.3f} "
          f"paired {s['paired']['point']:+.3f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--stage", required=True,
                    choices=["directions", "grid", "tau", "canary", "test", "transfer"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cid", default=None, help="test stage: run this grid/tau config instead of the candidate")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
