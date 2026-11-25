#!/usr/bin/env python
"""E2 post hoc depth check (2026-11-15, user: "do the free Mac depth check"; POST HOC, no bar).

The E2 pod run stopped at stage 1 before `healedprobe`, and its checkpoints were lost with the pod. This re-heals each yes/no task
at k=0, seed 0, at the stage-1 seed-0 winner lr (read from the pod's run records), on this Mac, then runs E2's own `healedprobe`
stage unchanged to get `L_on_healed`. Everything is written under var/results/atlas3/e2chk_mac/ and var/ckpt/atlas3/e2chk_mac/,
so the pod's results are never touched. Mac (MPS) heals are not bit-identical to the pod's CUDA heals; the healed k0 accuracy is
written next to the pod's for comparison.
  ATLAS3_DEVICE=mps .venv/bin/python scripts/e2_mac_depth.py [--tasks a,b] [--smoke]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_e2_run as E  # noqa: E402

POD_RUNS = Path("var/results/atlas3/e2chk/1.7B/runs.shard0of1.json")
DEFAULT = ("rule_check", "hard_check", "arith_check", "code_output_check", "code_literal_check")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(DEFAULT))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tasks = tuple(t for t in a.tasks.split(",") if t)
    E.sfx_dir = lambda smoke: "e2chk_mac_smoke" if smoke else "e2chk_mac"
    os.environ["HF_HUB_OFFLINE"] = "1"
    pod = json.loads(POD_RUNS.read_text())["runs"]
    size = "1.7B"
    profile = E.size_profile(size, a.smoke)
    E.C.configure(profile)
    cfg = E.make_cfg(size, a.smoke, tasks)
    E.ac.OUT.mkdir(parents=True, exist_ok=True)
    ctx = E.ac.Ctx(cfg, device=os.environ.get("ATLAS3_DEVICE") or None)
    E.C.check_blocks(ctx.n_layers, profile)
    # healedprobe picks the winner lr among both lrs; this folder holds only the winner's re-heal, so hand that row over directly
    E.ac.collect = lambda runs, ts, ks, rule: {t: {0: sorted((r for r in runs.values() if r["task"] == t and r["k"] == 0),
                                                              key=lambda r: r["seed"])} for t in ts}
    path = E.ac.shard_path("runs", cfg.sfx, 0, 1)
    mine = E.ac.read_json(path) or {"shard": [0, 1], "runs": {}, "post_hoc": True}
    summary = {}
    for task in tasks:
        rows = [r for r in pod.values() if r["task"] == task and r["k"] == 0 and r["seed"] == 0]
        lr = E.select_lr(rows) if rows else 3e-4
        key = E.run_key(task, 0, 0, lr)
        if key not in mine["runs"]:
            print(f"heal {key} on {ctx.device}", flush=True)
            mine["runs"][key] = E.C.run_whole_heal(ctx, task, 0, 0, lr)
            E.ac.write(path, mine)
        rc = E.stage_healedprobe(ctx, task)
        res = E.ac.read_json(E.C.healed_probe_path(cfg, task)) or {}
        summary[task] = {"lr": lr, "pod_best_lrsel": max((r["best_lrsel"] for r in rows), default=None),
                         "mac_best_lrsel": mine["runs"][key].get("best_lrsel"), "healed_accuracy": res.get("healed_accuracy"),
                         "chance": res.get("chance"), "L_on_healed": res.get("L_on_healed"), "never": res.get("never"),
                         "boot": res.get("boot"), "L_on_unmodified": res.get("L_on_unmodified"), "rc": rc}
        E.ac.write(E.ac.OUT / "summary_post_hoc.json", summary)
        print(f"{task}: {summary[task]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
