"""The task-overlap analysis (docs/ATLAS-overlap.md) on fake inputs with hand-checked values, and scripts/atlas_overlap.py
end to end on a tmp_path tree in the real file formats. No model, no network."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import atlas_overlap as CLI  # noqa: E402
import export_unembed as EU  # noqa: E402

from surgery.atlas import overlap as ov  # noqa: E402

_orig = ov.subspace_analysis


def drops28(mid, first=0.0, last=0.0):
    return [first] + list(mid) + [last]


# ---------------------------------------------------------------- vectors, similarity, Jaccard

def test_spearman_ignores_blocks_0_and_27():
    a = np.arange(26) / 100
    da, db = drops28(a, first=5.0, last=-5.0), drops28(a, first=-9.0, last=9.0)
    assert ov.block_vector(da).tolist() == pytest.approx(a.tolist())
    sim = ov.similarity_matrix(np.stack([ov.block_vector(da), ov.block_vector(db), ov.block_vector(drops28(a[::-1]))]))
    assert sim[0, 1] == pytest.approx(1.0)
    assert sim[0, 2] == pytest.approx(-1.0)
    assert sim[0, 0] == 1.0


def test_constant_vector_gives_nan_similarity():
    sim = ov.similarity_matrix(np.stack([np.zeros(26), np.arange(26.0), np.arange(26.0) ** 2]))
    assert math.isnan(sim[0, 1]) and sim[1, 2] == pytest.approx(1.0)


def test_fragile_set_and_jaccard():
    d = [0.0] * 28
    d[0], d[5], d[7], d[27] = 0.9, 0.05, 0.049, 0.9
    assert ov.fragile_set(d) == {5}
    assert ov.jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)
    assert ov.jaccard({1}, {2}) == 0.0
    assert math.isnan(ov.jaccard(set(), set()))
    j = ov.jaccard_matrix([{1, 2, 3}, {2, 3, 4}, {9}])
    assert j[0, 1] == j[1, 0] == pytest.approx(0.5) and j[0, 2] == 0.0


# ---------------------------------------------------------------- O1a, O1b

def _planted_sim(n_a, n_e, within, across, rng, noise=0.02):
    n = n_a + n_e
    s = np.full((n, n), across)
    s[:n_a, :n_a] = within
    s[n_a:, n_a:] = within
    s = s + rng.normal(0, noise, (n, n))
    s = (s + s.T) / 2
    np.fill_diagonal(s, 1.0)
    return s


def test_kind_test_planted_effect_is_small_p_and_no_effect_is_large_p():
    rng = np.random.default_rng(0)
    kinds = [0] * 7 + [1] * 7
    r = ov.kind_test(_planted_sim(7, 7, 0.8, 0.1, rng), kinds)
    assert r["method"] == "exact" and r["n_perm"] == math.comb(14, 7)
    assert r["effect"] == pytest.approx(0.7, abs=0.05)
    assert r["p"] == pytest.approx(2 / 3432)           # the labelling and its mirror image
    assert (r["n_same"], r["n_cross"]) == (42, 49)
    noise = rng.normal(0, 0.3, (14, 14))
    s = (noise + noise.T) / 2
    np.fill_diagonal(s, 1.0)
    assert ov.kind_test(s, kinds)["p"] > 0.1


def test_kind_test_uses_random_permutations_when_space_is_large():
    rng = np.random.default_rng(1)
    kinds = [0] * 10 + [1] * 10
    r = ov.kind_test(_planted_sim(10, 10, 0.8, 0.1, rng), kinds, n_perm=500)
    assert r["method"] == "permutation" and r["p"] == pytest.approx(1 / 501)
    assert r == ov.kind_test(_planted_sim(10, 10, 0.8, 0.1, np.random.default_rng(1)), kinds, n_perm=500)


def test_kind_test_hand_value():
    # 4 tasks, kinds 0 0 1 1; same pairs (0,1) and (2,3) = 0.9 and 0.5; cross pairs all 0.1
    s = np.full((4, 4), 0.1)
    s[0, 1] = s[1, 0] = 0.9
    s[2, 3] = s[3, 2] = 0.5
    r = ov.kind_test(s, [0, 0, 1, 1])
    assert r["effect"] == pytest.approx(0.7 - 0.1)
    assert r["p"] == pytest.approx(2 / 6)  # 6 labellings; the true split and its mirror image reach 0.6


def test_matching_test_planted_pairs_give_one_in_120():
    sim = np.full((10, 10), 0.1)
    for i in range(5):
        sim[i, 5 + i] = sim[5 + i, i] = 0.9
    r = ov.matching_test(sim, range(5), range(5, 10))
    assert r["n_perm"] == 120 and r["p"] == pytest.approx(1 / 120)
    assert r["effect"] == pytest.approx(0.8) and r["matched_mean"] == pytest.approx(0.9)
    assert r["n_other"] == 20
    # mis-ordering the partners gives the worst case: the observed matching is the least similar
    bad = ov.matching_test(sim, range(5), [6, 7, 8, 9, 5])
    assert bad["p"] == pytest.approx(1.0)


# ---------------------------------------------------------------- O2, O3

def test_lesion_load():
    d = [0.0] * 28
    d[3], d[7], d[9], d[20] = 0.25, -0.5, 0.125, 0.9
    assert ov.lesion_load(d, [7, 3, 9, 20], k=3) == pytest.approx(0.375)   # the negative drop counts as 0
    assert ov.lesion_load(d, [7, 3, 9, 20], k=4) == pytest.approx(1.275)


def test_spearman_test_and_interval():
    x = np.arange(14.0)
    r = ov.spearman_test(x, x * 2 + 1)
    assert r["rho"] == pytest.approx(1.0) and r["p"] == pytest.approx(1 / 10001)
    lo, hi = ov.bootstrap_interval(x, x)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    assert ov.spearman_test(rng.normal(size=14), rng.normal(size=14))["p"] > 0.05


def test_lesion_test_compares_rivals():
    tasks = list("abcdefgh")
    dmg = {t: i / 10 for i, t in enumerate(tasks)}
    res = ov.lesion_test(tasks, {t: i * 2.0 for i, t in enumerate(tasks)}, dmg,
                         {"bi": {t: -i for i, t in enumerate(tasks)}, "ce_rise": {t: i for i, t in enumerate(tasks)}})
    assert res["lesion_load"]["rho"] == pytest.approx(1.0)
    assert res["bi"]["rho"] == pytest.approx(-1.0) and res["ce_rise"]["rho"] == pytest.approx(1.0)


def test_mantel_planted_correlation():
    rng = np.random.default_rng(4)
    n = 10
    s = rng.normal(size=(n, n))
    s = (s + s.T) / 2
    np.fill_diagonal(s, 1.0)
    t = s + rng.normal(0, 0.01, (n, n))
    t = (t + t.T) / 2
    r = ov.mantel_test(s, t, n_perm=2000)
    assert r["rho"] > 0.95 and r["p"] < 0.01 and r["n_pairs"] == 45
    t_rand = rng.normal(size=(n, n))
    t_rand = (t_rand + t_rand.T) / 2
    assert ov.mantel_test(s, t_rand, n_perm=500)["p"] > 0.05


def test_mantel_drops_missing_pairs_and_symmetrise():
    tr = {"a": {"b": 0.2, "c": 0.4}, "b": {"a": 0.0}, "c": {}}
    t = ov.symmetrise(tr, ["a", "b", "c"])
    assert t[0, 1] == t[1, 0] == pytest.approx(0.1)
    assert math.isnan(t[0, 2]) and math.isnan(t[1, 2])
    sim = np.array([[1, .5, .1], [.5, 1, .2], [.1, .2, 1]])
    r = ov.mantel_test(sim, t, n_perm=10)
    assert r["n_pairs"] == 1 and math.isnan(r["rho"])      # one usable pair: no correlation


def test_holm():
    adj = ov.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.2})
    assert adj == pytest.approx({"a": 0.04, "c": 0.09, "b": 0.09, "d": 0.2})
    v = ov.verdicts({"a": 0.5, "b": 0.5, "c": -0.5, "d": 0.5}, {"a": 0.001, "b": 0.02, "c": 0.0001, "d": float("nan")})
    assert v["a"]["verdict"] == "supported" and v["a"]["p_holm"] == pytest.approx(0.003)
    assert v["b"]["verdict"] == "supported" and v["b"]["p_holm"] == pytest.approx(0.04)   # 0.02 * 2
    assert v["c"]["verdict"] == "not supported"                                          # wrong sign
    assert v["d"]["verdict"] == "not supported" and v["d"]["p_holm"] == 1.0


def test_hubs_and_first_hit():
    sets = [{1, 2, 3}, {2, 3}, {3, 9}]
    assert ov.hub_blocks(sets, 2) == {2: 2, 3: 3}
    assert ov.first_fragile_hit({9}, [5, 3, 9], (0, 1, 2, 3)) == 3
    assert ov.first_fragile_hit({9}, [5, 3, 9], (0, 1, 2)) is None
    assert ov.first_fragile_hit({5}, [5, 3, 9], (0, 6)) == 6


# ---------------------------------------------------------------- subspaces

def test_row_basis_drops_small_singular_values():
    rng = np.random.default_rng(5)
    m = rng.normal(size=(3, 40))
    m = np.vstack([m, m[0] + m[1]])
    q = ov.row_basis(m)
    assert q.shape == (3, 40) and np.allclose(q @ q.T, np.eye(3), atol=1e-8)
    assert ov.row_basis(np.zeros((3, 8))).shape == (0, 8)


def test_subspace_overlap_identical_orthogonal_and_partial():
    e = np.eye(8)
    assert ov.subspace_overlap(e[:2], e[:2]) == pytest.approx(1.0)
    assert ov.subspace_overlap(e[:2], e[2:4]) == pytest.approx(0.0)
    assert ov.subspace_overlap(e[:2], e[[0, 5]]) == pytest.approx(0.5)      # 1 shared direction / min rank 2
    assert ov.subspace_overlap(e[:1], e[:3]) == pytest.approx(1.0)          # normalised by the smaller rank
    assert math.isnan(ov.subspace_overlap(e[:0], e[:2]))


def test_class_means_are_centred_on_the_task_mean():
    x = np.array([[1.0, 0], [3, 0], [0, 2], [0, 4]], dtype=np.float32)
    m = ov.class_means(x, [0, 0, 1, 1], 3)           # class 2 absent: dropped
    assert m.dtype == np.float64 and m.shape == (2, 2)
    assert m.tolist() == [[2 - 1, 0 - 1.5], [0 - 1, 3 - 1.5]]
    assert np.linalg.matrix_rank(m) == 1      # equal class counts: the two centred means are opposite


def test_projection_removes_planted_option_direction():
    rng = np.random.default_rng(6)
    u = np.zeros((1, 12))
    u[0, 0] = 1.0
    m = rng.normal(size=(3, 12))
    m[:, 0] += 10.0
    out = ov.project_out(m, ov.row_basis(u))
    assert np.allclose(out[:, 0], 0) and np.allclose(out[:, 1:], m[:, 1:])
    assert ov.project_out(m, np.zeros((0, 12))) is m


def _two_task_acts(d=24, n=300, seed=7, shared=True):
    rng = np.random.default_rng(seed)
    labs = {t: rng.integers(0, 3, n) for t in "ab"}
    x = {}
    for t in "ab":
        base = rng.normal(0, 0.3, (n, d))
        # class signal along e0 (a direction both tasks use) plus a task-specific one
        base[:, 0] += 3.0 * labs[t] if shared else 0.0
        base[:, 1 if t == "a" else 2] += 3.0 * (labs[t] == 1)
        x[t] = base.astype(np.float32)
    return labs, x


def test_subspace_analysis_shared_direction_and_projection():
    labs, x = _two_task_acts()
    unembed = {t: np.eye(24, dtype=np.float32)[[0]] for t in "ab"}       # the shared option direction is e0
    res = ov.subspace_analysis(lambda t, l: x[t], ["a", "b"], labs, {"a": 3, "b": 3}, unembed, layers=(0,), n_shuffle=20)
    raw, null = res["raw"][0, 0, 1], res["raw_null"][0, 0, 1]
    proj, pnull = res["proj"][0, 0, 1], res["proj_null"][0, 0, 1]
    assert res["raw"][0, 1, 0] == raw and res["raw_null"][0, 1, 0] == null      # symmetric
    assert raw > 0.4                       # both span e0: one shared direction of rank 2
    assert raw > null                      # the loud shared direction also shows in shuffles, so only the excess is clean
    assert proj < 0.2 and abs(proj - pnull) < 0.1   # removing e0 removes the shared part: back to the null


def test_subspace_null_of_random_labels_is_near_zero():
    rng = np.random.default_rng(8)
    d, n = 200, 2000
    x = {t: rng.normal(size=(n, d)).astype(np.float32) for t in "ab"}
    labs = {t: rng.integers(0, 3, n) for t in "ab"}
    res = ov.subspace_analysis(lambda t, l: x[t], ["a", "b"], labs, {"a": 3, "b": 3}, None, layers=(0,), n_shuffle=10)
    assert "proj" not in res
    assert abs(res["raw"][0, 0, 1]) < 0.05 and res["raw_null"][0, 0, 1] < 0.05


def test_constant_layer_has_no_subspace():
    x = {t: np.ones((20, 6), dtype=np.float32) for t in "ab"}
    labs = {t: np.arange(20) % 2 for t in "ab"}
    res = ov.subspace_analysis(lambda t, l: x[t], ["a", "b"], labs, {"a": 2, "b": 2}, None, layers=(0,), n_shuffle=3)
    assert math.isnan(res["raw"][0, 0, 1])


def test_jsonable():
    out = ov.jsonable({"a": np.array([1.0, np.nan]), 3: np.float32(2.0), "s": {2, 1}, "i": np.int64(4)})
    assert out == {"a": [1.0, None], "3": 2.0, "s": [1, 2], "i": 4}
    json.dumps(out, allow_nan=False)


# ---------------------------------------------------------------- the unembed export layout

def test_unembed_npz_layout(tmp_path):
    matrix = np.arange(50 * 4, dtype=np.float32).reshape(50, 4)
    path = tmp_path / "sub" / "u.npz"
    EU.save_unembed(path, matrix, {"t1": [3, 7], "t2": [0, 49, 5]})
    assert path.exists() and not list(path.parent.glob("*.tmp"))
    rows, ids = EU.load_unembed(path)
    assert set(rows) == {"t1", "t2"} and rows["t1"].dtype == np.float32 and rows["t2"].shape == (3, 4)
    assert np.array_equal(rows["t1"], matrix[[3, 7]]) and ids["t2"].tolist() == [0, 49, 5]
    with np.load(path) as z:
        assert sorted(z.files) == ["ids_t1", "ids_t2", "t1", "t2"]


# ---------------------------------------------------------------- the CLI end to end

NAMES = ["a1", "a2", "a3", "e1", "e2", "e3"]
TASKS = {t: {"kind": "appraise" if t[0] == "a" else "execute", "n_options": 3, "number": i + 1}
         for i, t in enumerate(NAMES)}
PAIRS = [("a1", "e1"), ("a2", "e2"), ("a3", "e3")]
D = 16


def _build_tree(root: Path, with_unembed=True, drop_runs=False):
    rng = np.random.default_rng(10)
    a0, a1, acts = root / "atlas", root / "atlas1", root / "acts"
    a0.mkdir()
    a1.mkdir()
    order = list(rng.permutation(np.arange(1, 27)))
    order = [int(b) for b in order]
    (a0 / "order.json").write_text(json.dumps({"shared_order": order}))
    runs, summary_rows, rivals = {}, [], {}
    for i, t in enumerate(NAMES):
        drops = [float(x) for x in rng.uniform(0, 0.12, 28)]
        (a1 / f"secondary_lesion_{t}.json").write_text(json.dumps({"task": t, "drops": drops}))
        rivals[t] = {"k": 12, "bi": float(rng.uniform()), "ce_rise": float(rng.uniform())}
        summary_rows.append({"task": t, "rung": None, "k_star": int(rng.choice([0, 6, 9]))})
        for k in (0, 12):
            for lr, best in ((1e-3, 0.5), (3e-4, 0.6)):          # 3e-4 wins on lrsel at seed 0
                for seed in (0, 1, 2):
                    if lr == 1e-3 and seed:
                        continue
                    acc = 0.9 - (0.05 * i if k == 12 else 0.0) - 0.01 * seed
                    if lr == 1e-3:
                        acc = 0.0                               # the losing lr must not be used
                    key = f"{t}|k{k}|s{seed}|lr{lr:g}"
                    runs[key] = {"key": key, "task": t, "k": k, "seed": seed, "lr": lr, "best_lrsel": best,
                                 "sel_accuracy": acc}
        n = 60
        labs = rng.integers(0, 3, n)
        x = rng.normal(size=(n, 29, D)).astype(np.float16)
        x[:, :, 0] += labs[:, None]
        d = acts / f"{t}_discovery"
        d.mkdir(parents=True)
        np.save(d / "resid.npy", x)
        (d / "meta.json").write_text(json.dumps({"task": t, "n": n, "answers": [int(v) for v in labs]}))
    if drop_runs:
        runs = {}
    (a1 / "runs.shard0of1.json").write_text(json.dumps({"shard": [0, 1], "runs": runs}))
    (a1 / "plan.json").write_text(json.dumps({"lr_rule": "per_k"}))
    (a1 / "rivals.json").write_text(json.dumps(rivals))
    transfer = {t: {b: float(rng.uniform(-0.1, 0.2)) for b in NAMES if b != t} for t in NAMES}
    (a1 / "summary.json").write_text(json.dumps({"ks": [0, 6, 9, 12], "rows": summary_rows,
                                                  "transfer": {"12": transfer}}))
    if with_unembed:
        EU.save_unembed(a1 / "unembed_options.npz", rng.normal(size=(40, D)).astype(np.float32),
                        {t: [0, 1, 2] for t in NAMES})
    return a0, a1, acts


def _args(root, a0, a1, acts):
    return ["--atlas0", str(a0), "--atlas1", str(a1), "--acts", str(acts), "--unembed", str(a1 / "unembed_options.npz"),
            "--out", str(root / "out")]


def test_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(ov, "subspace_analysis", lambda *a, **k: _orig(*a, **{**k, "n_shuffle": 5}))
    a0, a1, acts = _build_tree(tmp_path)
    assert CLI.main(_args(tmp_path, a0, a1, acts), TASKS, PAIRS) == 0
    doc = json.loads((tmp_path / "out" / "overlap.json").read_text())
    md = (tmp_path / "out" / "overlap.md").read_text()
    assert md.startswith("## How the tasks overlap (secondary, plan fixed 2026-11-10 in docs/ATLAS-overlap.md)")
    assert doc["tasks"] == NAMES and doc["projected_available"] is True
    # damage uses the winning lr (3e-4) and the seed mean: k0 mean(0.9, .89, .88) minus k12 on task a1 (i=0) is 0
    assert doc["O2"]["damage"]["a1"] == pytest.approx(0.0)
    assert doc["O2"]["damage"]["e3"] == pytest.approx(0.25)
    assert set(doc["verdicts"]) == {"O1a", "O1b", "O2", "O3"}
    assert doc["O1b"]["n_perm"] == 6 and doc["O1a"]["n_perm"] == 20 and doc["O1a"]["method"] == "exact"
    assert doc["O2"]["lesion_load"]["n"] == 6
    sub = doc["subspace"]
    assert sub["layers"] == list(range(29)) and set(sub["excess_raw"]) == {"a1|e1", "a2|e2", "a3|e3"}
    assert len(sub["excess_projected"]["a1|e1"]) == 29 and set(sub["maps"]["excess_raw"]) == {"6", "12", "18", "24"}
    assert sub["raw"]["a1|e1"][1] is not None
    assert "Same directions" in md and "projected" in md.lower() and "O2" in md
    assert "Skipped" not in md


def test_cli_without_unembed_skips_projection(tmp_path, monkeypatch):
    monkeypatch.setattr(ov, "subspace_analysis", lambda *a, **k: _orig(*a, **{**k, "n_shuffle": 3}))
    a0, a1, acts = _build_tree(tmp_path, with_unembed=False)
    assert CLI.main(_args(tmp_path, a0, a1, acts), TASKS, PAIRS) == 0
    doc = json.loads((tmp_path / "out" / "overlap.json").read_text())
    md = (tmp_path / "out" / "overlap.md").read_text()
    assert doc["projected_available"] is False and doc["subspace"]["excess_projected"] == {}
    assert "projected variant was skipped" in md and "O1a" in md


def test_cli_exits_nonzero_when_phase1_input_is_missing(tmp_path, capsys):
    a0, a1, acts = _build_tree(tmp_path)
    (a1 / "rivals.json").unlink()
    assert CLI.main(_args(tmp_path, a0, a1, acts), TASKS, PAIRS) == 1
    assert "rivals.json" in capsys.readouterr().err
    (a1 / "rivals.json").write_text("{}")
    assert CLI.main(_args(tmp_path, a0, a1, acts), TASKS, PAIRS) == 1
    assert "rivals.json has no bi" in capsys.readouterr().err


def test_cli_exits_nonzero_without_healed_runs(tmp_path, capsys):
    a0, a1, acts = _build_tree(tmp_path, drop_runs=True)
    assert CLI.main(_args(tmp_path, a0, a1, acts), TASKS, PAIRS) == 1
    assert "no healed runs" in capsys.readouterr().err

