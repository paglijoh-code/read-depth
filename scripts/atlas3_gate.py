#!/usr/bin/env python3
"""M9 Phase 2, stage C0 gate (docs/ATLAS-phase2-stageC.md, C0 step 4).

  uv run python scripts/atlas3_gate.py [MODEL_DIR] [--dir var/results/atlas3]

MODEL_DIR is the model's folder under --dir (default 1.7B, the model the gate is read on). For tasks 22-25 prints BoW,
chance, BoW - chance and `L_on`. Clean: BoW - chance <= 0.10. Late: `L_on` >= 4. The gate passes if at least 3 of the 4
are clean and at least 2 of those are late. Reads baseline.json and MODEL_DIR/map_*.json only (never a cut result);
writes MODEL_DIR/gate.json. A task that fails stays out of the count but is still printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GATE_TASKS = ("code_traces_first", "syntax_outer_pair", "target_read_bare", "bind_compare")
MAX_LEAK, MIN_L_ON, MIN_CLEAN, MIN_LATE = 0.10, 4, 3, 2


def gate(rows: list[dict]) -> dict:
    """rows: {task, bow, chance, L_on}. Adds `gap` and `clean` (BoW - chance <= 0.10) to each, and the verdict."""
    out = []
    for r in rows:
        gap = r["bow"] - r["chance"]
        out.append({**r, "gap": gap, "clean": gap <= MAX_LEAK + 1e-9})
    clean = [r for r in out if r["clean"]]
    late = [r for r in clean if r["L_on"] is not None and r["L_on"] >= MIN_L_ON]
    return {"rows": out, "n_clean": len(clean), "n_clean_late": len(late),
            "pass": len(clean) >= MIN_CLEAN and len(late) >= MIN_LATE,
            "rule": f"at least {MIN_CLEAN} of {len(GATE_TASKS)} with BoW - chance <= {MAX_LEAK}, and at least {MIN_LATE} of those with L_on >= {MIN_L_ON}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", nargs="?", default="1.7B")
    ap.add_argument("--dir", type=Path, default=Path("var/results/atlas3"))
    a = ap.parse_args(argv)
    base = json.loads((a.dir / "baseline.json").read_text())["tasks"]
    rows = []
    for t in GATE_TASKS:
        m = json.loads((a.dir / a.model_dir / f"map_{t}.json").read_text())
        rows.append({"task": t, "bow": base[t]["bow"], "chance": base[t]["chance"], "L_on": m.get("L_on")})
    g = gate(rows)
    print(f"{'task':<20}{'BoW':>7}{'chance':>8}{'gap':>7}{'L_on':>6}")
    for r in g["rows"]:
        print(f"{r['task']:<20}{r['bow']:>7.3f}{r['chance']:>8.3f}{r['gap']:>7.3f}{str(r['L_on']):>6}{'' if r['clean'] else '  leaky'}")
    print(f"clean {g['n_clean']}, clean and L_on >= {MIN_L_ON}: {g['n_clean_late']} -> {'PASS' if g['pass'] else 'FAIL'}")
    (a.dir / a.model_dir / "gate.json").write_text(json.dumps(g, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
