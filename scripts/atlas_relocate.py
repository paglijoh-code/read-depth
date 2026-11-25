"""Relocation check (docs/ATLAS-relocate.md): probe the k=12 seed-0 healed cut models of Qwen3-1.7B (C1), as D1 probed k=0.

  ATLAS3_DEVICE=mps uv run python scripts/atlas_relocate.py --task T   # one task -> var/results/atlas3/c1/cut_probe_T.json
  uv run python scripts/atlas_relocate.py --table                       # Q1-Q3 from the saved files (markdown on stdout)
Post hoc, exploratory; no training, no network.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

K = 12
C1 = ROOT / "var/results/atlas3/c1"


def out_path(task: str) -> Path:
    return C1 / f"cut_probe_{task}.json"


def winner_row(task: str) -> dict:
    runs = json.load(open(C1 / "runs.shard0of1.json"))["runs"].values()
    lr = {r["lr"] for r in runs if r["task"] == task and r["k"] == K and r["seed"] == 1}
    assert len(lr) == 1, (task, lr)
    lr = lr.pop()
    rows = [r for r in runs if r["task"] == task and r["k"] == K and r["seed"] == 0 and r["lr"] == lr]
    assert len(rows) == 1, (task, len(rows))
    return rows[0]


def probe(task: str) -> int:
    import atlas3_cuts as C
    import atlas_cuts as ac
    import atlas_map
    from surgery import adapters
    from surgery.atlas import heal, measure, spec
    final = out_path(task)
    if final.exists():
        print(f"{final} exists; nothing to do", flush=True)
        return 0
    C.configure()
    cfg = C.make_cfg(False)
    ctx = ac.Ctx(cfg, device=os.environ.get("ATLAS3_DEVICE") or None)
    row = winner_row(task)
    recs = ctx.records(task, "discovery")
    if cfg.probe_records:
        recs = recs[:cfg.probe_records]
    store = ac.ACTS / f"{task}_cut_probe"
    model, handle = ac.rebuild_healed(ctx, row)
    adapters.merge_lora(handle)
    alive = sorted(int(b) for b in heal.alive_blocks(ctx.order, K, ctx.n_layers))
    try:
        meta = measure.build_store(model, ctx.tokenizer, task, recs, store, batch_size=ac.EVAL_BS)
        answers = meta["answers"]
        acc = float(np.mean(np.asarray(meta["pred"]) == np.asarray(answers)))
        X = np.load(store / "resid.npy", mmap_mode="r")
        p = atlas_map._probe(X, answers, meta["template_ids"], np.arange(len(answers)), cfg.boots, seed=0)
        del X
    finally:
        shutil.rmtree(store, ignore_errors=True)
    res = {"task": task, "k": K, "seed": 0, "lr": row["lr"], "ckpt": row["ckpt"], "alive": alive, "n": len(answers),
           "chance": spec.chance(answers), "cut_accuracy": acc, "curve": p["curve"] if p else None,
           "scoring": "option first-token accuracy", "device": str(ctx.device), "ts": ac.now()}
    final.write_text(json.dumps(res, indent=1))
    print(f"cutprobe {task}: acc {acc:.3f} peak {max(res['curve']) if res['curve'] else None} -> {final}", flush=True)
    return 0


def clip01(x):
    return max(0.0, min(1.0, x))


def table() -> int:
    from escape_check import weighted_within, spearman
    e1 = {r["task"]: r for r in json.load(open(ROOT / "var/results/atlas3/e1.json"))["1.7B"]["tasks"]}
    rows = []
    for t, e in e1.items():
        f0, fk = C1 / f"healed_probe_{t}.json", out_path(t)
        if not (f0.exists() and fk.exists()):
            print(f"missing {t}", file=sys.stderr)
            continue
        h0, hk = json.load(open(f0)), json.load(open(fk))
        ch, c0, cK, a0, aK = h0["chance"], h0["curve"], hk["curve"], h0["healed_accuracy"], hk["cut_accuracy"]
        L0 = h0["L_on_healed"]
        P = clip01((max(cK) - ch) / (max(c0) - ch)) if cK and max(c0) - ch > 0 else None
        O = clip01((aK - ch) / (a0 - ch)) if a0 - ch > 0 else None
        thr = ch + 0.5 * (a0 - ch)
        j = next((i for i, v in enumerate(cK or []) if v >= thr - 1e-9), None)
        alive = hk["alive"]
        rp = None if j is None else (0 if j == 0 else alive[j - 1] + 1)
        cut = [b for b in range(len(c0) - 1) if b not in alive]
        n_pre = None if L0 is None else sum(b < L0 for b in cut)
        rows.append(dict(task=t, kind=e["kind"], L0=L0, R=e["R"], P=P, O=O, rp=rp, n_pre=n_pre,
                         shift=None if (rp is None or L0 is None) else rp - L0, peakK=max(cK), peak0=max(c0), ch=ch))
    fmt = lambda x: "—" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
    print("| task | kind | L0 | n_pre | R | O (first token) | P (inside) | read point after cut | shift |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["L0"] is None, r["L0"] or 0)):
        print(f"| {r['task']} | {r['kind']} | {fmt(r['L0'])} | {fmt(r['n_pre'])} | {fmt(r['R'])} | {fmt(r['O'])} | {fmt(r['P'])} | "
              f"{'never' if r['rp'] is None else r['rp']} | {fmt(r['shift'])} |")
    failed = [r for r in rows if r["R"] < 0.5 and r["P"] is not None]
    lost = [r["task"] for r in failed if r["P"] < 0.5]
    print(f"\nQ1: {len(failed)} tasks with R < 0.5; P < 0.5 in {len(lost)} ({', '.join(lost)})")
    if len(failed) < 4:
        v = "untestable"
    else:
        s = len(lost) / len(failed)
        v = "lost inside" if s >= 0.75 else "kept inside" if s <= 0.25 else "mixed"
        print(f"Q1 share {s:.3f} -> {v}")
    ok = [r for r in rows if r["P"] is not None]
    print(f"rho(P, R) over {len(ok)} tasks: {spearman([r['P'] for r in ok], [r['R'] for r in ok]):.3f}")
    q2 = [r for r in rows if r["n_pre"] and r["R"] >= 0.9]
    print("Q2 (n_pre > 0, R >= 0.9): " + "; ".join(f"{r['task']} L0 {r['L0']} n_pre {r['n_pre']} -> {r['rp']} shift {r['shift']}" for r in q2))
    w, per, pooled = weighted_within([(r["kind"], r["L0"], r["P"]) for r in ok if r["L0"] is not None])
    print(f"Q3 weighted within-kind rho(L0, P) {w:.3f} ({', '.join(f'{k} {v:.3f}' for k, v in per.items() if v is not None)}); pooled {pooled:.3f}")
    (C1 / "relocate.json").write_text(json.dumps({"rows": rows, "Q1": v, "Q1_lost": lost}, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    sys.exit(table() if a.table else probe(a.task))
