#!/usr/bin/env python3
"""POST HOC (round-5 review, 2026-11-20): conditional permutation test for the kind-given-depth partial correlation.

`scripts/atlas_robust.py` `e1c` permutes kind labels across all tasks, which breaks the kind-depth link, so its p is not a
test of "kind adds nothing given depth". Here kind is permuted only within bins of trained read depth (<=3, 4-10, >10, the
bins of `atlas2_cuts.l_on_group`), which keeps each bin's kind mix. Same statistic (`ranked_partial`), two-sided, 10,000
permutations, seed 0. Rows from `var/results/atlas3/e1.json` (per profile: task, kind, L_on_healed, R).

  .venv/bin/python scripts/kind_partial_conditional.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from atlas_robust import ranked_partial  # noqa: E402


def grp(d):
    return 0 if d <= 3 else (1 if d <= 10 else 2)


def main(n=10_000):
    e = json.loads((ROOT / "var/results/atlas3/e1.json").read_text())
    for lab, prof in e.items():
        rows = prof["tasks"]
        depth = [r["L_on_healed"] for r in rows]
        y = [r["R"] for r in rows]
        kind = np.array([1.0 if r["kind"] == "appraise" else 0.0 for r in rows])
        g = np.array([grp(d) for d in depth])
        obs = ranked_partial(kind, y, depth)
        rng = np.random.default_rng(0)
        hits_c = hits_u = 0
        for _ in range(n):
            k = kind.copy()
            for b in set(g):
                idx = np.where(g == b)[0]
                k[idx] = rng.permutation(k[idx])
            s = ranked_partial(k, y, depth)
            if s is not None and abs(s) >= abs(obs) - 1e-12:
                hits_c += 1
            s = ranked_partial(rng.permutation(kind), y, depth)
            if s is not None and abs(s) >= abs(obs) - 1e-12:
                hits_u += 1
        mix = {int(b): (int(kind[g == b].sum()), int((g == b).sum())) for b in sorted(set(g))}
        print(f"{lab}: n={len(rows)} partial {obs:+.3f}  p within-bin {(hits_c + 1) / (n + 1):.4f}  "
              f"p unrestricted {(hits_u + 1) / (n + 1):.4f}  judging/total per bin {mix}  (stored e1c p {prof['E1c']['p']:.4f})")


if __name__ == "__main__":
    main()
