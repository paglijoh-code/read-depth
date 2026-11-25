"""Steps study run pipeline (scripts/steps_run.py): the bar logic on toy numbers, per-cell grouping, the predict-before-cutscore
refusal, the control-task permutation, the patching decision rule, the vectorised bootstrap, and site states on a tiny random
Qwen3. No real weights, no network."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import steps_run as R  # noqa: E402
from atlas2_healed import l_on_healed  # noqa: E402
from test_atlas_heal import tiny_model  # noqa: E402
from test_atlas_measure import StubTok  # noqa: E402

from surgery.atlas import measure  # noqa: E402


def dt_of(depths: dict, seeds=(0, 1, 2), noise=None):
    """{cell: {seed: {"d", "boot"}}}: `depths` {cell: depth}; replicates are depth plus the given noise vector."""
    rng = np.random.default_rng(0)
    noise = noise if noise is not None else rng.normal(0, 0.3, 200)
    return {c: {s: {"d": float(d), "boot": d + noise} for s in seeds} for c, d in depths.items()}


TERM = [(["a"], ["b"])]


# ---------------------------------------------------------------- labels

def test_label_effect_reverse_none_inconclusive():
    c = R.contrast(dt_of({"a": 10, "b": 5}), TERM, (0, 1, 2))
    assert c["delta"] == pytest.approx(5) and c["label"] == "effect"
    assert R.contrast(dt_of({"a": 5, "b": 10}), TERM, (0, 1, 2))["label"] == "reverse effect"
    assert R.contrast(dt_of({"a": 5.2, "b": 5}), TERM, (0, 1, 2))["label"] == "no effect"
    # large enough but one seed disagrees: not an effect
    dt = dt_of({"a": 10, "b": 5})
    dt["a"][1]["d"] = 4.0
    c = R.contrast(dt, TERM, (0, 1, 2))
    assert c["delta"] >= 3 and c["label"] == "inconclusive"
    # a wide interval around a middling delta
    wide = dt_of({"a": 8, "b": 5})
    for sd in (0, 1, 2):
        wide["a"][sd]["boot"] = 8 + np.random.default_rng(sd).normal(0, 6, 200)
    assert R.contrast(wide, TERM, (0, 1, 2))["label"] == "inconclusive"
    assert R.label_contrast(None, None, None, []) == "not evaluable"


def test_contrast_drops_terms_with_a_dropped_or_missing_end():
    dt = dt_of({"a": 10, "b": 5, "c": 9, "d": 8})
    terms = [(["a"], ["b"]), (["c"], ["d"])]
    full = R.contrast(dt, terms, (0, 1, 2))
    assert full["n_terms"] == 2 and full["delta"] == pytest.approx(3.0)
    part = R.contrast(dt, terms, (0, 1, 2), dropped={"d"})
    assert part["n_terms"] == 1 and part["n_skipped"] == 1 and part["delta"] == pytest.approx(5.0)
    dt["a"][2]["d"] = None                       # a seed with no depth for a cell: that term is unusable
    assert R.contrast(dt, terms, (0, 1, 2))["n_terms"] == 1
    assert R.contrast(dt, terms, (0, 1, 2), dropped={"d", "b"})["label"] == "not evaluable"


def test_s4_needs_the_gap_between_families_and_kept_ends():
    ser = R.contrast(dt_of({"a": 12, "b": 5}), TERM, (0, 1, 2))
    com = R.contrast(dt_of({"a": 6, "b": 5}), TERM, (0, 1, 2))
    assert R.s4(ser, com)["label"] == "serial steps cost more than combining"
    assert R.s4(ser, ser)["label"] == "not shown"
    weak = R.contrast(dt_of({"a": 8, "b": 5}), TERM, (0, 1, 2))
    assert R.s4(weak, com)["label"] == "not shown"            # 3 - 1 = 2 < 3
    gone = R.contrast(dt_of({"a": 12, "b": 5}), TERM, (0, 1, 2), dropped={"b"})
    assert R.s4(gone, com)["label"] == "not evaluable"


# ---------------------------------------------------------------- S5

def s5_inputs(n=12, r_fn=lambda i: 1 - i / 12):
    cells = [f"c{i}" for i in range(n)]
    depth = {c: float(i) for i, c in enumerate(cells)}
    r = {c: r_fn(i) for i, c in enumerate(cells)}
    return cells, depth, r, {c: True for c in cells}, {c: 1 + i // 3 for i, c in enumerate(cells)}


def test_s5_pass_ties_uninformative_and_drop_rules():
    cells, depth, r, kept, lens = s5_inputs()
    boot = np.full(100, -0.9)
    out = R.s5(depth, r, lens, kept, boot)
    assert out["rho"] == pytest.approx(-1.0) and out["label"].startswith("depth predicts")
    assert R.s5(depth, r, lens, kept, np.full(100, 0.2))["label"] == "not shown"        # interval reaches above 0
    assert R.s5(depth, {c: 0.99 for c in cells}, lens, kept, boot)["label"] == "uninformative"
    assert R.s5(depth, {c: 0.01 for c in cells}, lens, kept, boot)["label"] == "uninformative"
    assert R.s5(depth, {c: 0.5 + 0.005 * i for i, c in enumerate(cells)}, lens, kept, boot)["label"] == "uninformative"  # range < 0.10
    few = dict(kept)
    for c in cells[:5]:
        few[c] = False                                                                   # 7 kept: under 8
    assert R.s5(depth, r, lens, few, boot)["label"].startswith("not evaluable")
    eight = dict(kept)
    for c in cells[:4]:
        eight[c] = False
    assert R.s5(depth, r, lens, eight, boot)["n_cells"] == 8
    # ties use average ranks: constant depth has no correlation
    assert R.s5({c: 3.0 for c in cells}, r, lens, kept, boot)["label"].startswith("not evaluable")
    assert R.spearman([1, 1, 2, 3], [4, 4, 5, 6]) == pytest.approx(1.0)


def test_r_value_clips_and_refuses_flat_uncut():
    assert R.r_value(0.5, 1.0, 0.1) == pytest.approx(0.4 / 0.9)
    assert R.r_value(1.2, 1.0, 0.1) == 1.0 and R.r_value(0.0, 1.0, 0.1) == 0.0
    assert R.r_value(0.0, 1.0, 0.1, clip=False) == pytest.approx(-0.1 / 0.9)
    assert R.r_value(0.5, 0.1, 0.1) is None


# ---------------------------------------------------------------- S6

def test_s6_directional_rule():
    q = R.contrast(dt_of({"a": 10, "b": 5}), TERM, (0, 1, 2))
    ok = R.contrast(dt_of({"a": 9, "b": 5}, seeds=(0,)), TERM, (0,))
    assert R.s6(q, ok)["label"] == "directional agreement"
    small = R.contrast(dt_of({"a": 7, "b": 5}, seeds=(0,)), TERM, (0,))                  # 2 blocks < 2.6
    assert R.s6(q, small)["label"] == "no directional agreement"
    opp = R.contrast(dt_of({"a": 5, "b": 9}, seeds=(0,)), TERM, (0,))
    assert R.s6(q, opp)["label"] == "no directional agreement"
    weak_q = R.contrast(dt_of({"a": 5.5, "b": 5}), TERM, (0, 1, 2))
    assert R.s6(weak_q, ok)["label"] == "no directional agreement"                        # Qwen itself has no effect
    assert R.s6(q, ok)["smol_fraction"] == pytest.approx(4 / 24)
    assert R.S6_MIN_BLOCKS == pytest.approx(2.571, abs=0.01)


# ---------------------------------------------------------------- grouping, control task

def test_cell_index_groups_by_first_appearance():
    idx = R.cell_index(["b", "a", "b", "c", "a"])
    assert list(idx) == ["b", "a", "c"] and idx["b"].tolist() == [0, 2] and idx["a"].tolist() == [1, 4]


def test_smoke_subset_keeps_first_n_per_cell():
    recs = [{"rung": c, "i": i} for i, c in enumerate("aabbbacc")]
    assert [r["i"] for r in R.smoke_subset(recs, 2)] == [0, 1, 2, 3, 6, 7]


def test_control_labels_are_a_fixed_permutation_per_template():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 10, 300)
    t = np.array([f"t{i % 6}" for i in range(300)])
    c1, c2 = R.control_labels(a, t), R.control_labels(a, t)
    assert (c1 == c2).all()
    for u in np.unique(t):
        m = t == u
        pairs = set(zip(a[m].tolist(), c1[m].tolist()))
        assert len({p[0] for p in pairs}) == len(pairs)                    # same answer -> same control label
        assert len({p[1] for p in pairs}) == len(pairs)                    # injective
    assert (c1 != a).mean() > 0.5
    assert not np.array_equal(R.control_labels(a, t, seed=1), c1)
    one = R.control_labels(a[:50], t[:50])                                 # independent of what else is in the call
    assert (one == c1[:50]).all()


def test_choose_alpha_prefers_best_then_smaller():
    assert R.choose_alpha(lambda al: {1e-3: 0.5, 1e-2: 0.7, 1e-1: 0.7, 1.0: 0.6}[al]) == 1e-2


# ---------------------------------------------------------------- patching rule

def test_decision_block_first_at_half_else_never():
    toy = [0.0, 0.0, 0.1, 0.49, 0.5, 0.9, 1.0, 1.0]
    d = R.decision_block(toy)
    assert d["block"] == 4 and d["stable"]
    assert R.decision_block([0.1, 0.2, 0.3])["block"] == "never"
    assert R.decision_block([0.6, 0.4, 0.7])["block"] == 0 and not R.decision_block([0.6, 0.4, 0.7])["stable"]
    assert R.decision_block([None, 0.5])["block"] == 1


# ---------------------------------------------------------------- bootstrap and probes

def test_vectorised_bootstrap_matches_l_on_healed_on_the_full_sample():
    rng = np.random.default_rng(3)
    n, nl = 120, 9
    g = np.repeat(np.arange(24), 5)
    a = rng.integers(0, 4, n)
    ft = np.where(rng.random(n) < 0.8, a, (a + 1) % 4)
    corr = np.stack([np.where(rng.random(n) < min(0.2 + 0.1 * l, 0.9), a, (a + 1) % 4) for l in range(nl)])
    tab = R.group_tables(g, a, corr == a[None, :], ft == a)
    # a resample that is every group once equals the original sample
    size, cl = tab["size"], tab["cls"]
    idx = np.arange(len(size))[None, :]
    nn = size[idx].sum(1)
    acc = tab["ft"][idx].sum(1) / nn
    ch = cl[idx].sum(1).max(1) / nn
    curve = tab["corr"][idx].sum(1) / nn[:, None]
    depth = R.first_reaching_vec(curve, ch + 0.5 * (acc - ch))[0]
    exp, never = l_on_healed([float((corr[l] == a).mean()) for l in range(nl)], float(np.bincount(a).max() / n), float((ft == a).mean()))
    assert depth == (nl if never else exp)
    d, o = R.oof_boot_depths(tab, 50, 0)
    assert d.shape == (50,) and o.shape == (50,)


def test_onset_fixed_and_refit_replicate_shapes():
    assert R.onset_fixed([0.2, 0.5, 0.55, 0.9]) == 2 and R.onset_fixed([0.2, 0.3]) is None
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 5)
    a = rng.integers(0, 3, 100)
    X = rng.normal(size=(100, 3, 8)).astype(np.float32)
    X[:, 2, :3] += np.eye(3)[a] * 4                                         # layer 2 carries the label
    d, o = R.refit_replicate(X, a, g, a.copy(), 0)
    assert d == 2.0 and o == 2.0


# ---------------------------------------------------------------- predict before cutscore

def test_cutscore_refuses_without_predicted(tmp_path):
    with pytest.raises(SystemExit, match="predicted.json is missing"):
        R.check_predicted(tmp_path)
    (tmp_path / "predicted.json").write_text("{}")
    with pytest.raises(SystemExit, match="missing"):
        R.check_predicted(tmp_path)                                          # no hash file yet
    h = R.sha_of(tmp_path / "predicted.json")
    (tmp_path / "predicted.sha256").write_text(h + "\n")
    with pytest.raises(SystemExit, match="stream log"):
        (tmp_path / "stream.log").write_text("something else\n")
        R.check_predicted(tmp_path)
    R.stream(tmp_path, f"predicted.json sha256 {h}")
    assert R.check_predicted(tmp_path) == h
    (tmp_path / "predicted.json").write_text('{"x": 1}')                     # edited after the fact
    with pytest.raises(SystemExit, match="differs"):
        R.check_predicted(tmp_path)


# ---------------------------------------------------------------- site states

def test_site_states_match_the_store_and_ignore_batching(tmp_path):
    model, tok = tiny_model(), StubTok()
    recs = [{"id": f"r{i}", "task": "fact_check", "prompt": "ab" * (i + 2), "answer": i % 2, "template_id": f"t{i}",
             "rung": "c", "meta": {}} for i in range(5)]
    meta = measure.build_store(model, tok, "fact_check", recs, tmp_path, batch_size=2)
    store = np.load(tmp_path / "resid.npy")
    last = measure.site_states(model, tok, "fact_check", recs, [[-1]] * 5, batch_size=3)
    assert last.shape == (5, 1, 5, 64)
    np.testing.assert_allclose(last[:, 0].astype(np.float32), store.astype(np.float32), atol=2e-2)
    pos = [[1, -2]] * 5
    a = measure.site_states(model, tok, "fact_check", recs, pos, batch_size=1, dtype=np.float32)
    b = measure.site_states(model, tok, "fact_check", recs, pos, batch_size=5, dtype=np.float32)
    np.testing.assert_allclose(a, b, atol=1e-4)
    with pytest.raises(ValueError):
        measure.site_states(model, tok, "fact_check", recs, [[10**6]] * 5)
