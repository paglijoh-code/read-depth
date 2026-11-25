"""POST HOC (2026-11-18): the alternative-cut test for the copy chain.

Predictions are fixed in reports/steps/altcut_predicted.md (committed before this ran). Two k=12 cuts of Qwen3-1.7B,
same repair as the steps v2 sweep (q/v LoRA rank 10, 300 steps; seed 0 tries both learning rates, seed 2 uses seed 0's
pick): cut L removes original blocks 16-19, cut E removes 6, 8, 9, 10, and both put back v2's four least-needed blocks.
For each cut and seed: heal, score exact match on the test records, then probe the healed model per layer and cell.
Writes var/results/steps/altcut_runs.json (heal rows) and var/results/steps/altcut_<cut>_s<seed>.json.

    ATLAS3_DEVICE=mps uv run python scripts/steps_altcut.py --cut L --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import atlas_cuts as ac  # noqa: E402
import steps_cutprobe as CP  # noqa: E402
import steps_run as S  # noqa: E402
from surgery import adapters  # noqa: E402
from surgery.atlas import heal  # noqa: E402

TASK = S.COPY
SEEDS = (0, 2)
PUT_BACK = (22, 24, 25, 26)
SWAP_IN = {"L": (16, 17, 18, 19), "E": (6, 8, 9, 10)}


def cut_order(shared: list[int], cut: str) -> list[int]:
    """An order whose first 12 entries are the cut's removed blocks (v2's 12, minus PUT_BACK, plus SWAP_IN[cut])."""
    v2 = list(shared[: S.KCUT])
    if set(SWAP_IN[cut]) & set(v2) or not set(PUT_BACK) <= set(v2):
        raise SystemExit("cut definition no longer matches the shared order")
    removed = [b for b in v2 if b not in PUT_BACK] + list(SWAP_IN[cut])
    return removed + [b for b in shared if b not in removed]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cut", required=True, choices=sorted(SWAP_IN))
    ap.add_argument("--seed", type=int, required=True, choices=SEEDS)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args(argv)
    d = S.configure("qwen17", False)
    cfg = S.make_cfg("qwen17", False)
    cfg.transfer = False
    out = d.root / f"altcut_{args.cut}_s{args.seed}.json"
    if out.exists():
        print(f"{out} exists", flush=True)
        return 0
    shared = json.loads(Path(ac.ORDER_PATH).read_text())["shared_order"]
    order = cut_order(shared, args.cut)
    ctx = S.make_ctx(cfg, order=order)
    ac.CKPT = d.ckpt / f"altcut_{args.cut}"
    alive = heal.alive_blocks(order, S.KCUT, cfg.n_layers)
    runs_path = d.root / "altcut_runs.json"
    runs = ac.read_json(runs_path) or {}

    def mine(seed):
        return [r for r in runs.values() if r["cut"] == args.cut and r["seed"] == seed]

    if args.seed == 0:
        lrs = list(ac.LRS)
    else:
        seed0 = mine(0)
        if len(seed0) < len(ac.LRS):
            raise SystemExit("run seed 0 first")
        lrs = [heal.pick_lr(seed0)]
    for lr in lrs:
        key = f"{args.cut}|{ac.run_key(TASK, S.KCUT, args.seed, lr)}"
        if key not in runs:
            row = ac.run_heal(ctx, TASK, S.KCUT, args.seed, lr)
            row.update(cut=args.cut, alive_blocks=alive, post_hoc=True)
            runs[key] = row
            ac.write(runs_path, runs)
    rows = mine(args.seed)
    lr = heal.pick_lr(mine(0)) if args.seed == 0 else lrs[0]
    row = next(r for r in rows if np.isclose(r["lr"], lr))

    t0 = time.perf_counter()
    model, handle = ac.rebuild_healed(ctx, row)
    adapters.merge_lora(handle)
    try:
        test_cells, ok = S.exact_by_cell(ctx, model, TASK, ctx.records(TASK, "test"))
    finally:
        del model
        ac.release(ctx)
    t1 = time.perf_counter()
    cells, t2 = CP.probe_healed(ctx, row, TASK, args.seed, args.jobs, len(alive), f"altcut_{args.cut}_s{args.seed}")
    acc = float(np.mean(list(ok.values())))
    ac.write(out, {"post_hoc": True, "cut": args.cut, "task": TASK, "seed": args.seed, "k": S.KCUT, "lr": lr,
                   "ckpt": row["ckpt"], "alive_blocks": alive, "removed": order[: S.KCUT],
                   "sel_accuracy": row["sel_accuracy"], "test_accuracy": acc, "test_cells": test_cells,
                   "layer_note": "layer 0 = embeddings; layer i (i >= 1) = output of original block alive_blocks[i-1]",
                   "probe_cells": cells, "seconds": {"test": t1 - t0, "store": t2 - t1, "probe": time.perf_counter() - t2},
                   "ts": S.now()})
    print(f"altcut {args.cut} s{args.seed}: lr {lr:g}, test {acc:.3f} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
