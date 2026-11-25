"""Unit tests of the pure helpers in scripts/paper_fixpass.py, on toy data (no result files, no model)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paper_fixpass as P  # noqa: E402


def test_estimators_perfect_negative():
    kinds = ["a"] * 4 + ["b"] * 4
    x = [1, 2, 3, 4, 1, 2, 3, 4]
    y = [4, 3, 2, 1, 8, 7, 6, 5]
    w, p = P.estimators(kinds, x, y)
    assert math.isclose(w, -1.0) and -1.0 < p < 0


def test_leave_one_family_out_drops_the_family():
    rows = [{"task": f"t{i}", "kind": "a", "depth": i, "y": -i} for i in range(6)]
    fam = {f"t{i}": ("f1" if i < 2 else f"f{i}") for i in range(6)}
    res = P.leave_one_family_out(rows, fam)
    by = {r["left_out"]: r for r in res}
    assert by["f1"]["n"] == 4 and by["f1"]["tasks"] == ["t0", "t1"]
    assert all(math.isclose(r["weighted"], -1.0) for r in res)


def test_binary_depth():
    assert P.binary_depth([1, 2, 1, 14]) == [0, 1, 0, 1]


def test_paired_bootstrap_binary_equals_continuous_when_depth_is_binary():
    rows = [{"task": f"t{i}", "kind": "a" if i % 2 else "b", "depth": 1 if i < 6 else 5, "y": 1.0 - 0.05 * i - (0.3 if i >= 6 else 0)}
            for i in range(12)]
    out = P.paired_bootstrap_abs_diff(rows, n_boot=200, seed=1)
    for k in ("weighted", "pooled"):
        assert math.isclose(out[k]["abs_diff"], 0.0, abs_tol=1e-12)
        assert out[k]["lo"] <= 0.0 <= out[k]["hi"] or math.isclose(out[k]["lo"], 0.0, abs_tol=1e-12)
        assert out[k]["n_defined"] + out[k]["n_skipped"] == 200


def test_percentile_matches_linear_interpolation():
    assert P._pct([0.0, 1.0, 2.0, 3.0], 50) == 1.5
    assert P._pct([5.0], 97.5) == 5.0
    assert P._pct([], 50) is None


def test_multiset_permutations_count():
    vals = [1, 1, 1, 2]
    arr = list(P.multiset_permutations(vals))
    assert len(arr) == len(set(arr)) == P.n_distinct_permutations(vals) == 4


def test_exact_permutation_p_extreme_observed_is_one_over_n_and_tails_coincide():
    # one depth-2 label on the lowest R, three tasks tied at the top: the observed rho is the minimum, the positive tail is smaller
    y = [0.3, 0.6, 0.9, 1.0, 1.0, 1.0]
    x = [2, 1, 1, 1, 1, 1]
    e = P.exact_permutation_p(x, y)
    assert e["n_arrangements"] == 6
    assert e["hits_one_sided"] == 1 and e["hits_two_sided"] == 1
    assert math.isclose(e["p_one_sided"], 1 / 6) and math.isclose(e["p_two_sided"], 1 / 6)
    assert e["max_rho_over_arrangements"] < abs(e["observed"])


def test_exact_permutation_p_cap():
    assert P.exact_permutation_p(list(range(12)), list(range(12)), cap=1000) is None


def test_seed_survival_clips_and_guards():
    assert P.seed_survival(0.9, 1.0, 0.5) == 0.8
    assert P.seed_survival(1.2, 1.0, 0.5) == 1.0
    assert P.seed_survival(0.4, 1.0, 0.5) == 0.0
    assert P.seed_survival(0.9, 0.5, 0.5) is None


def test_n_pre_counts_blocks_strictly_below_depth():
    assert P.n_pre(1, [11, 12]) == 0
    assert P.n_pre(12, [11, 12, 13]) == 1
    assert P.n_pre(13, [11, 12, 13]) == 2
