#!/usr/bin/env python3
"""M9 Phase 2, stage A gate (docs/ATLAS-phase2-stageA.md, choice 14).

  uv run python scripts/atlas2_gate.py [--dir var/results/atlas2]

For tasks 15-18 prints BoW, chance, BoW - chance and `L_on`. The gate passes if at least 3 of the 4 have BoW - chance <= 0.10
and at least 2 of those have `L_on` >= 4. Reads baseline.json and map_*.json only (never a cut result); writes gate.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GATE_TASKS = ("arith_order", "syntax_first_close", "code_uses_first", "hard_target_read")
MAX_LEAK, MIN_L_ON, MIN_CLEAN, MIN_DEEP = 0.10, 4, 3, 2


def gate(rows: list[dict]) -> dict:
    """rows: {task, bow, chance, L_on}. Adds `gap` and `clean` (BoW - chance <= 0.10) to each, and the verdict."""
    out = []
    for r in rows:
        gap = r["bow"] - r["chance"]
        out.append({**r, "gap": gap, "clean": gap <= MAX_LEAK + 1e-9})
    clean = [r for r in out if r["clean"]]
    deep = [r for r in clean if r["L_on"] is not None and r["L_on"] >= MIN_L_ON]
    return {"rows": out, "n_clean": len(clean), "n_clean_deep": len(deep),
            "pass": len(clean) >= MIN_CLEAN and len(deep) >= MIN_DEEP,
            "rule": f"at least {MIN_CLEAN} of {len(GATE_TASKS)} with BoW - chance <= {MAX_LEAK}, and at least {MIN_DEEP} of those with L_on >= {MIN_L_ON}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("var/results/atlas2"))
    a = ap.parse_args(argv)
    base = json.loads((a.dir / "baseline.json").read_text())["tasks"]
    rows = []
    for t in GATE_TASKS:
        m = json.loads((a.dir / f"map_{t}.json").read_text())
        rows.append({"task": t, "bow": base[t]["bow"], "chance": base[t]["chance"], "L_on": m.get("L_on")})
    g = gate(rows)
    print(f"{'task':<20}{'BoW':>7}{'chance':>8}{'gap':>7}{'L_on':>6}")
    for r in g["rows"]:
        print(f"{r['task']:<20}{r['bow']:>7.3f}{r['chance']:>8.3f}{r['gap']:>7.3f}{str(r['L_on']):>6}{'' if r['clean'] else '  leaky'}")
    print(f"clean {g['n_clean']}, clean and L_on >= {MIN_L_ON}: {g['n_clean_deep']} -> {'PASS' if g['pass'] else 'FAIL'}")
    (a.dir / "gate.json").write_text(json.dumps(g, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
