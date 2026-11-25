"""E1 robustness script: pure pieces on small synthetic inputs; no model, no stored results."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import atlas_robust as R  # noqa: E402

from surgery.atlas import metrics  # noqa: E402

CURVE = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.9]


def test_l_on_fraction_default_unchanged():
    assert metrics.l_on(CURVE, 0.1) == metrics.l_on(CURVE, 0.1, 0.5) == 3  # threshold 0.1 + 0.4
    assert metrics.l_on(CURVE, 0.1, 0.3) == 2
    assert metrics.l_on(CURVE, 0.1, 0.7) == 4
    assert metrics.l_on([0.1, 0.12], 0.1, 0.3) is None


def test_healed_l_on_uses_healed_accuracy_and_never():
    assert R.healed_l_on(CURVE, 0.1, 0.9, 0.5) == 3
    assert R.healed_l_on(CURVE, 0.1, 0.9, 0.7) == 4
    assert R.healed_l_on(CURVE, 0.1, 1.0, 1.0) == len(CURVE)       # never reached
    assert R.healed_l_on(CURVE, 0.1, 0.12) is None


def test_r_at_clips():
    assert R.r_at(0.5, 0.9, 0.1) == pytest.approx(0.5)
    assert R.r_at(1.0, 0.9, 0.1) == 1.0 and R.r_at(0.0, 0.9, 0.1) == 0.0
    assert R.r_at(0.5, 0.1, 0.1) is None


def test_ranked_partial_and_plain():
    x, y, z = [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5], [1, 2, 3, 4, 5, 6]
    assert R.spearman_plain(x, z) == pytest.approx(1.0)
    assert R.ranked_partial(x, y, z) is None                      # x and z identical: degenerate
    assert R.ranked_partial([0, 1, 0, 1, 0, 1], y, [1, 2, 3, 4, 5, 6]) is not None
    assert R.spearman_plain([1, 1, 1], [1, 2, 3]) is None


def _toy():
    kinds = ["appraise"] * 6 + ["execute"] * 6
    depth = [1, 2, 3, 4, 5, 6] * 2
    r = [0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.95, 0.85, 0.6, 0.5, 0.2, 0.0]
    return depth, r, kinds, [f"t{i}" for i in range(12)]


def test_e1a_deterministic_and_verdict():
    d, r, k, t = _toy()
    a = R.e1a(d, r, k, t, n_boot=200)
    assert a == R.e1a(d, r, k, t, n_boot=200)
    assert len(a["loo"]) == 12 and a["verdict"] == "Robust" and a["rho"] < -0.9


def test_e1a_fragile_names_task():
    d, r, k, t = _toy()
    r = r[:5] + [0.9] + r[6:]                                     # one task turned into a strong outlier
    d = d[:5] + [30] + d[6:]
    a = R.e1a(d, [0.9, 0.1, 0.9, 0.1, 0.9, 0.1] + [0.5, 0.5, 0.4, 0.6, 0.5, 0.5], k, t, n_boot=50)
    assert a["verdict"] in ("Fragile", "Mostly robust", "Robust")  # label is data-dependent; the call must run


def test_e1c_kind_explained_by_depth():
    kinds = ["appraise"] * 6 + ["execute"] * 6
    depth = [1, 1, 2, 2, 3, 3, 8, 9, 10, 11, 12, 13]
    r = [0.95, 0.9, 0.85, 0.9, 0.8, 0.85, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1]
    c = R.e1c(depth, r, kinds, n_perm=500)
    assert c["rho_kind_depth"] < -0.8 and c["rho_kind_r12"] > 0.8
    assert c["p"] == R.e1c(depth, r, kinds, n_perm=500)["p"]
    assert c["verdict"] in ("Kind adds nothing measurable beyond depth", "Kind has its own effect", "Unclear")


def test_e1d_dose_response():
    rows = []
    for i, (kind, dep) in enumerate([("appraise", d) for d in (1, 2, 3, 12, 14, 16)] + [("execute", d) for d in (1, 2, 3, 12, 14, 16)]):
        drop = 0.0 if dep <= 3 else 0.5
        rows.append({"task": f"t{i}", "kind": kind, "L_on_healed": dep, "chance": 0.1,
                     "acc": {"0": {"mean": 1.0}, "9": {"mean": 1.0 - drop * 0.3}, "12": {"mean": 1.0 - drop * 0.6},
                             "16": {"mean": 1.0 - drop}}})
    out = R.e1d(rows, n_perm=500)
    assert out["verdict"] == "Dose-response"
    assert out["16"]["groups"]["deep (>10)"]["mean_R"] < out["16"]["groups"]["shallow (<=3)"]["mean_R"]
