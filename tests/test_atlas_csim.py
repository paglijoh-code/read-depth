"""C-sim tests: tiny replicate counts, checks on keys, seeding and direction of effects."""
import json

import numpy as np
import pytest

from surgery.atlas import csim as C
from surgery.atlas import tree as T


def test_spreads_from_data_reads_measured_files():
    sp = C.spreads_from_data()
    # the repo's own C-screen and P0 results (skipped quietly if they are absent on this machine)
    if not sp["probe_onsets"]:
        pytest.skip("no var/cscreen data")
    assert 0 < sp["probe_sd"] < 1 and 0 < sp["within_family_share"] < 1
    if sp["use_range"]:
        assert 0.6 < sp["use_range"][0] and sp["use_range"][1] < 0.95  # P0 use onsets are packed at 0.71-0.85
    assert sp["block_noise_sd"] is None or sp["block_noise_sd"] < 0.05
    json.dumps(sp)


def test_spreads_from_synthetic_files(tmp_path):
    for m in ("a", "b"):
        for f in range(8):
            for t in "ab":
                p = tmp_path / "cs" / m / f"f{f}_{t}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(dict(model=m, task=f"f{f}_{t}", family=f"f{f}", n_blocks=24,
                                             probe_onset=dict(rel=0.1 * f + 0.01 * (t == "b"), layers=2.4 * f + 1.0, width=0.1))))
    q = tmp_path / "p0" / "m" / "t.json"
    q.parent.mkdir(parents=True)
    q.write_text(json.dumps(dict(onset=dict(use=dict(rel=0.8, width=0.05, peak=0.7)), use=dict(random_recovery=[0.001, -0.001]))))
    sp = C.spreads_from_data(str(tmp_path / "cs"), (str(tmp_path / "p0"),), str(tmp_path / "p0"))
    assert sp["within_family_share"] < 0.05 and sp["n_family_pairs"] == 16
    assert sp["use_range"] == [0.8, 0.8] and sp["block_noise_sd"] == pytest.approx(0.001)
    assert sp["log_layer_noise_sd"] is not None


def test_onset_batch_matches_scalar():
    rng = np.random.default_rng(0)
    c = np.sort(rng.uniform(0, 1, (6, 20)), 1) * rng.uniform(0.2, 1, (6, 1)) + rng.normal(0, 0.02, (6, 20))
    ob = C.onset_batch(c)
    for i in range(6):
        o = T.onset_curve(c[i], 20)
        assert ob[i] == pytest.approx(np.nan if o is None else o["rel"], nan_ok=True)


def test_n2_power_keys_and_direction():
    sp = dict(within_family_share=0.3)
    out = C.n2_power(sp, reps=6, n_null=30, n_boot=8, n_starts=4, d_grid=(1.0, 6.0), seed=1)
    assert set(out["power"]) == {"2", "3"}
    cell = out["power"]["2"]["6.0"]
    assert {"power", "se", "power_null", "reps"} <= set(cell)
    assert cell["se"] == pytest.approx(np.sqrt(cell["power"] * (1 - cell["power"]) / 6))
    assert out["power"]["2"]["6.0"]["power"] >= out["power"]["2"]["1.0"]["power"]
    assert out["power"]["2"]["6.0"]["power"] > 0.5
    assert out["d_star"]["2"] in (1.0, 6.0, None)
    # d* picks the worse k
    assert out["d_star_worst"] is None or out["d_star_worst"] == max(v for v in out["d_star"].values())
    # seeded: same result twice, and independent of n_jobs
    again = C.n2_power(sp, reps=6, n_null=30, n_boot=8, n_starts=4, d_grid=(1.0, 6.0), seed=1)
    assert again["power"] == out["power"]


def test_n2_power_parallel_equals_serial():
    kw = dict(reps=4, n_null=20, n_boot=5, n_starts=3, d_grid=(6.0,), ks=(2,), seed=3)
    a = C.n2_power(None, n_jobs=1, **kw)
    b = C.n2_power(None, n_jobs=2, **kw)
    assert a["power"] == b["power"]


def test_n4_sim_runs():
    out = C.n4_sim(None, reps=3, n_null=15, n_boot=5, seed=0, deltas=(1.5,))
    assert {"power_by_delta", "fpr_continuum", "fpr_lineage", "fpr_max"} <= set(out)
    assert 0 <= out["fpr_max"] <= 1 and "1.5" in out["power_by_delta"]
    assert out["fpr_continuum"]["leaves"]


def test_n8_sim_noise_levels():
    sp = dict(block_noise_sd=5e-4, probe_onsets=list(np.linspace(0.05, 0.85, 40)))
    out = C.n8_sim(sp, reps=6, n_fam=15, n_boot=30, seed=0)
    assert set(out["levels"]) == {"measured", "x10", "peak10pct"}
    assert out["fpr_max"] == out["levels"]["measured"]["fpr_either"]
    # use = probe + negligible noise reads as agreement, not a gap or disagreement
    assert out["levels"]["measured"]["fpr_L8g"] == 0 and out["levels"]["measured"]["fpr_L8d"] == 0
    assert out["levels"]["measured"]["leaves"].get("L8a", 0) >= 5


def test_n5_power_key_and_direction():
    sp = dict(log_layer_noise_sd=0.1, probe_onsets=list(np.linspace(0.1, 0.8, 30)))
    lo = C.n5_power(sp, reps=4, n_boot=60, seed=0)
    assert {"power", "se", "beta"} <= set(lo) and lo["beta"] == 0.5
    hi_noise = C.n5_power(dict(sp, log_layer_noise_sd=2.0), reps=4, n_boot=60, seed=0)
    assert lo["power"] >= hi_noise["power"]


def test_run_csim_schema_and_summary():
    out = C.run_csim(dict(within_family_share=0.3), reps=4, n_null=20, n_boot=5, d_grid=(3.0,), seed=2)
    assert {"n2", "n4", "n8", "n5", "spreads_used"} <= set(out)
    cs = T.csim_summary(json.loads(json.dumps(out)))
    assert cs["n2_power_d3"] is not None and cs["n4_fpr"] is not None and cs["n8_fpr"] is not None
    assert cs["n5_power"] is not None


def test_script_csim_subcommand(tmp_path, capsys):
    import sys
    sys.path.insert(0, "scripts")
    import atlas_tree_run as run

    out = tmp_path / "csim" / "csim.json"
    run.main(["csim", "--out", str(out), "--reps", "2", "--n-null", "15", "--n-boot", "4", "--reps-n4", "2",
              "--reps-n8", "3", "--reps-n5", "2", "--cscreen", str(tmp_path / "none"), "--p0", str(tmp_path / "none"),
              "--trained", str(tmp_path / "none")])
    d = json.loads(out.read_text())
    assert d["n2"]["power"]["3"]["3.5"]["reps"] == 2 and "n5" in d and "N2 d*" in capsys.readouterr().out
