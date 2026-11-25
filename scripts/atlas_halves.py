#!/usr/bin/env python
"""Glue between the gate runners and the tree reader (docs/ATLAS-gates-run.md 1.5, 1.6).

For every <root>/g1/<prompt>/<model>/<task>.json (and g2/trained/s*/...) that has its .npz beside it, adds:
  onset_half  {probe: [onset_a, onset_b], use: [...]}   item-half onsets (seeded split) that tree.icc_halves reads
  onset_fresh {probe: onset}                            the fresh-item probe onset (read only if the half ICC fails)
Idempotent; rewrites the JSON in place (atomic). Usage: python scripts/atlas_halves.py [--root var/results/atlas_gates]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surgery.atlas import gates as G  # noqa: E402


def annotate(js: Path, seed: int = 0) -> bool:
    npz_p = js.with_suffix(".npz")
    if not npz_p.exists():
        return False
    j = json.loads(js.read_text())
    if not isinstance(j.get("valid"), dict):
        return False
    npz = np.load(npz_p)
    ph = G.probe_half_curves(npz, seed=seed)["halves"]
    uh = G.use_half_curves(npz, seed=seed)["halves"]
    j["onset_half"] = {"probe": [h["onset"] for h in ph], "use": [h["onset"] for h in uh]}
    if "pred_fr" in npz.files and npz["pred_fr"].shape[-1] > 0:
        j["onset_fresh"] = {"probe": G.fresh_probe_curve(npz)["onset"]}
    tmp = js.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(j, default=G._json))
    os.replace(tmp, js)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="var/results/atlas_gates")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    root = Path(a.root)
    paths = sorted(root.glob("g1/*/*/*.json")) + sorted(root.glob("g2/trained/s*/*/*.json"))
    n = bad = 0
    for p in paths:
        try:
            n += annotate(p, a.seed)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"skip {p}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"annotated {n} of {len(paths)} result files ({bad} errors)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
