#!/usr/bin/env python3
"""M9 Phase 2, stage A: probe curve of the seed-0, k=0 healed model of one of the 14 old tasks (docs/ATLAS-phase2-stageA.md, choice 13).

  uv run python scripts/atlas2_healed.py --task arith [--smoke]

The model is rebuilt as `atlas_cuts.stage_secondary` does (same helpers, imported): the winner-lr seed-0 k=0 run from
the merged var/results/atlas1/runs.shard* files, cut (nothing removed at k=0), LoRA structure, saved state, merged. The store
is built on the 1,000 `discovery` records, probed as `atlas_map._probe`, then deleted. --smoke: 64 records, 20 boots, `_smoke` names.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from surgery import adapters  # noqa: E402
from surgery.atlas import measure, metrics, spec  # noqa: E402

OUT = Path("var/results/atlas2")
ACTS = Path("var/acts/atlas2")
PHASE0_MAP = Path("var/results/atlas")


def l_on_healed(curve, chance: float, healed_acc: float) -> tuple[int | None, bool]:
    """(L_on_healed, never). First layer whose probe accuracy reaches chance + half of (healed accuracy - chance), layer
    indexing as `metrics.l_on`. Healed accuracy under 5 points above chance: (None, False). Never reached: (len(curve), True)."""
    if healed_acc - chance < metrics.MIN_GAP - metrics.EPS:
        return None, False
    hit = metrics._first_reaching(curve, chance + 0.5 * (healed_acc - chance))
    return (len(curve), True) if hit is None else (hit, False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(spec.PHASE1_TASKS))
    ap.add_argument("--boots", type=int, default=500)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    import atlas_cuts as ac
    import atlas_map
    task = args.task
    sfx = "_smoke" if args.smoke else ""
    boots = 20 if args.smoke else args.boots
    final = OUT / f"healed_{task}{sfx}.json"
    if final.exists():
        print(f"{final} exists; nothing to do")
        return 0
    cfg = ac.make_cfg(False)
    ctx = ac.Ctx(cfg)
    runs = ac.read_merged("runs", "", "runs")
    rows = [r for r in ac.collect(runs, [task], [0], ac.plan_rule(cfg)).get(task, {}).get(0, []) if r["seed"] == 0]
    if not rows:
        print(f"{task}: no healed k=0 seed-0 run", flush=True)
        return 1
    row = rows[0]
    recs = ctx.records(task, "discovery")
    if args.smoke:
        recs = recs[:64]
    store = ACTS / f"{task}_healed{sfx}"
    model, handle = ac.rebuild_healed(ctx, row)
    adapters.merge_lora(handle)
    meta = measure.build_store(model, ctx.tokenizer, task, recs, store, batch_size=ac.EVAL_BS)
    answers = meta["answers"]
    acc = float(np.mean(np.asarray(meta["pred"]) == np.asarray(answers)))
    X = np.load(store / "resid.npy", mmap_mode="r")
    p = atlas_map._probe(X, answers, meta["template_ids"], np.arange(len(answers)), boots, seed=0)
    phase0 = ac.read_json(PHASE0_MAP / f"map_{task}.json") or {}
    chance = spec.chance(answers)
    curve = p["curve"] if p else None
    lo, never = l_on_healed(curve, chance, acc) if curve else (None, False)
    res = {"task": task, "smoke": args.smoke, "lr": row["lr"], "ckpt": row["ckpt"], "n": len(answers), "chance": chance,
           "healed_accuracy": acc, "curve": curve, "lo": p["lo"] if p else None, "hi": p["hi"] if p else None,
           "L_on_healed": lo, "never": never, "L_on_phase0": phase0.get("L_on")}
    ac.write(final, res)
    shutil.rmtree(store, ignore_errors=True)
    del model
    ac.release(ctx)
    print(f"{task}: healed acc {acc:.3f} chance {chance:.3f} L_on_healed {lo} never {never} L_on_phase0 {res['L_on_phase0']} -> {final}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
