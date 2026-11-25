"""POST HOC (2026-11-18): linear probe curves on the cut-and-healed (k12) steps models.

Not part of the fixed steps design (docs/ATLAS-steps.md). For each task and seed, rebuilds the k=12 healed winner,
stores the last-token residuals on the discovery records, and fits the same out-of-fold linear probe per layer and cell
as the uncut probe stage (no bootstraps or refits). Writes var/results/steps/cutprobe_<task>_s<seed>.json with each
cell's curve and the original block index behind every layer of the cut model.

    uv run python scripts/steps_cutprobe.py --task steps_copy --seed 0
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import atlas_cuts as ac  # noqa: E402
import steps_run as S  # noqa: E402
from surgery import adapters  # noqa: E402
from surgery.atlas import measure  # noqa: E402


def probe_healed(ctx, row: dict, task: str, seed: int, jobs: int, n_alive: int, tag: str):
    """Rebuild a healed run row, store last-token residuals on the discovery records, and fit the per-layer linear
    probe per cell. Returns ({cell: {n, chance, curve, peak}}, time the store finished)."""
    recs = ctx.records(task, "discovery")
    sd = ac.ACTS / tag
    model, handle = ac.rebuild_healed(ctx, row)
    adapters.merge_lora(handle)
    try:
        meta = measure.build_store(model, ctx.tokenizer, task, recs, sd, batch_size=ac.EVAL_BS)
    finally:
        del model
        ac.release(ctx)
    t1 = time.perf_counter()
    try:
        X = np.load(sd / "resid.npy", mmap_mode="r")
        if X.shape[1] != n_alive + 1:
            raise SystemExit(f"store has {X.shape[1]} layers, expected {n_alive + 1}")
        a_all, g_all = np.asarray(meta["answers"]), np.asarray(meta["template_ids"])
        cells = {}
        for c, idx in S.cell_index(meta["rungs"]).items():
            a = a_all[idx]
            curve, _ = S.linear_curve(np.asarray(X[idx]), a, g_all[idx], seed, jobs)
            cells[c] = {"n": int(len(idx)), "chance": S.spec.chance(a.tolist()), "curve": curve, "peak": max(curve)}
        del X
    finally:
        shutil.rmtree(sd, ignore_errors=True)
    return cells, t1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=S.TASKS)
    ap.add_argument("--seed", type=int, required=True, choices=S.SEEDS)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args(argv)
    d = S.configure("qwen17", False)
    cfg = S.make_cfg("qwen17", False)
    ctx = S.make_ctx(cfg)
    out = d.root / f"cutprobe_{args.task}_s{args.seed}.json"
    if out.exists():
        print(f"{out} exists", flush=True)
        return 0
    runs = ac.read_merged("runs", cfg.sfx, "runs")
    row = S.winner_row(runs, args.task, S.KCUT, args.seed)
    if row is None:
        raise SystemExit("no k12 winner")
    order = json.loads(Path(ac.ORDER_PATH).read_text())["shared_order"]
    alive = sorted(set(range(cfg.n_layers)) - set(order[: S.KCUT]))
    t0 = time.perf_counter()
    cells, t1 = probe_healed(ctx, row, args.task, args.seed, args.jobs, len(alive), f"cutprobe_{args.task}_s{args.seed}")
    ac.write(out, {"post_hoc": True, "task": args.task, "seed": args.seed, "k": S.KCUT, "lr": row["lr"],
                   "ckpt": row["ckpt"], "alive_blocks": alive,
                   "layer_note": "layer 0 = embeddings; layer i (i >= 1) = output of original block alive_blocks[i-1]",
                   "cells": cells, "seconds": {"store": t1 - t0, "probe": time.perf_counter() - t1}, "ts": S.now()})
    print(f"cutprobe {args.task} s{args.seed}: {len(cells)} cells, store {t1 - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
