"""E2 pod runner (revisions R1 and R2): verdict functions on synthetic rows, eligibility, lr tie, resumability key, first-token
chance, and the R2 items (own| rows, common ids, unseen arith, E2-L, E2-C test R, bootstrap, eligible-only E2-A, descriptive E2-D,
the staged gate). CPU only, no model."""
import json
import os
import sys
from pathlib import Path

from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import atlas_e2_run as E  # noqa: E402
from surgery.atlas import spec  # noqa: E402


def pair(ddcb, dkind, dft=None, dfmt=0.0, ok=True, ddbd=0, ddcd=None):
    return {"dd_cb": ddcb, "dd_bd": ddbd, "dd_cd": ddcb if ddcd is None else ddcd,
            "dR_kind": dkind, "ok_kind": ok, "dR_fmt": dfmt, "ok_fmt": ok, "dR_cd": dkind + dfmt if dkind is not None else None,
            "dR_ft": dkind if dft is None else dft, "ok_cd": ok}


def test_e2a():
    assert E.e2a([0, -1, -2, -3, 1])["label"] == "same or later"
    assert E.e2a([-5, -6, -7])["label"] == "check reads early"
    assert E.e2a([-3, -4, -3])["label"] == "partly earlier"
    assert E.e2a([-2, -2])["label"] == "same or later" and E.e2a([-5, -5])["label"] == "check reads early"
    assert E.e2a([None])["label"] == "undetermined"


def test_e2b_effect_bar():
    eff = "kind has its own effect"
    assert E.e2b([pair(0, 0.3), pair(0, 0.4), pair(0, 0.2), pair(0, 0.3), pair(0, -0.1)])["label"] == eff   # mean 0.22, one non-positive
    assert E.e2b([pair(0, 0.9), pair(0, 0.9), pair(0, 0.9), pair(0, -0.1), pair(0, -0.1)])["label"] == "unclear"   # two non-positive
    assert E.e2b([pair(0, 0.3), pair(0, 0.3), pair(0, 0.3)])["label"] == eff                              # three eligible is enough
    assert E.e2b([pair(0, 0.15)] * 5)["label"] == "unclear"                                                # mean under 0.20


def test_e2b_none_bar_and_not_testable():
    none = "no kind effect"
    assert E.e2b([pair(0, 0.05), pair(0, -0.05), pair(0, 0.09), pair(0, 0.0), pair(0, -0.09)])["label"] == none
    assert E.e2b([pair(0, 0.0), pair(0, 0.0), pair(0, 0.0), pair(0, 0.3), pair(0, -0.3)])["label"] == "unclear"   # a gap of 0.3 breaks the spread
    assert E.e2b([pair(0, 0.0)] * 2 + [pair(0, 0.9, ok=False)] * 3)["label"] == "not testable"             # only 2 eligible
    assert E.e2b([pair(0, None)] * 5)["label"] == "not testable"
    out = E.e2b([pair(0, 0.3)] * 3 + [pair(0, 0.9, ok=False)])
    assert out["n_eligible"] == 3 and out["n_pairs"] == 4                                                  # ineligible pairs stay out


def test_e2f_uses_the_format_gap():
    f = E.e2f([pair(0, 0.0, dfmt=0.4)] * 3)
    assert f["label"] == "format helps" and f["mean"] == pytest.approx(0.4)
    assert E.e2f([pair(0, 0.0, dfmt=0.01)] * 4)["label"] == "no format effect"


def test_eligibility_and_interior():
    assert E.r_value(0.5, 0.60, 0.5) is None                       # healed k=0 under chance + 0.15: the task does not count
    assert E.r_value(0.5, 0.65, 0.5) is not None
    assert E.r_value(None, 0.9, 0.5) is None
    assert E.interior(0.5) and not E.interior(0.97) and not E.interior(0.03) and not E.interior(None)
    c = {"acc": {0: [0.95], 12: [0.7]}, "chance": 0.5, "depth": 5}
    for r_bin_k, ok in ((0.5, True), (0.95, False), (0.9, True)):
        b = {"acc": {0: [0.95], 12: [r_bin_k]}, "chance": 0.5, "depth": 5}
        d = {"acc": {0: [0.9], 12: [0.5]}, "chance": 0.1, "depth": 6}
        assert E.build_pair("x_check", "x_bin", "x", c, b, d)["ok_kind"] == (E.interior(E._r(b, 12)) and True)


def test_e2l_annotates_check_minus_do_only():
    low = E.e2l([pair(0, 0.4, dft=0.05)] * 3)
    assert low["run"] and "answer length" in low["label"]
    ok = E.e2l([pair(0, 0.4, dft=0.3)] * 3)
    assert "passed" in ok["label"]
    assert "not triggered" in E.e2l([pair(0, 0.05, dft=0.0)] * 3)["label"]       # no check - do advantage: nothing to annotate
    off = E.e2l([{**pair(0, 0.4), "dR_ft": None}] * 3)
    assert not off["run"] and off["label"].startswith("not run")                  # ftdo missing


def test_claim6_rows():
    eff, none = "kind has its own effect", "no kind effect"
    assert E.claim6("same or later", none)["row"] == "same_none"
    assert E.claim6("check reads early", eff)["row"] == "early_effect"
    assert E.claim6("same or later", eff)["row"] == "same_effect"
    assert E.claim6("check reads early", none)["row"] == "early_none"
    assert "qualifies claim 1" in E.claim6("check reads early", none)["text"]
    assert E.claim6("partly earlier", none)["row"] == "other"
    assert E.claim6("same or later", "unclear")["row"] == "other"
    assert E.claim6("same or later", "not testable")["row"] == "other"
    c = E.claim6("same or later", none, "format helps")
    assert c["row"] == "same_none" and c["format_note"] and c["text"].endswith(E.FORMAT_NOTE)
    n = E.claim6("same or later", none, "no format effect")
    assert n["format_note"] and n["text"].endswith("survive no better than the full answer.")
    assert not E.claim6("same or later", none, "unclear")["format_note"]
    assert E.WORDING["same_none"].startswith("Checking a proposed answer forms as late as")


def test_verdicts_end_to_end():
    v = E.verdicts([pair(0, 0.4), pair(1, 0.3), pair(-1, 0.3), pair(2, 0.3), pair(0, 0.1)])
    assert v["E2-A"]["label"] == "same or later" and v["E2-B"]["label"] == "kind has its own effect"
    assert v["claim6"]["row"] == "same_effect"
    v = E.verdicts([pair(0, 0.4, dft=0.0)] * 4)
    assert "answer length" in v["E2-L"]["label"]
    assert v["E2-B"]["label"] == "kind has its own effect" and v["claim6"]["row"] == "same_effect"   # E2-L never changes E2-B or claim 6
    v = E.verdicts([pair(-7, 0.0)] * 4)
    assert v["claim6"]["row"] == "early_none"
    v = E.verdicts([pair(0, 0.0)] * 2)
    assert v["E2-B"]["label"] == "not testable" and v["claim6"]["row"] == "other"


def test_e2d_is_descriptive_with_no_p():
    ps = [pair(-i, 0.1 * i, dfmt=0.0, ddbd=0, ddcd=-i) for i in range(1, 7)]
    out = E.e2d(ps)
    assert out["n"] >= 6 and out["rho"] is not None and out["p"] is None and out["descriptive"]
    assert E.e2d([pair(0, 0.1)])["rho"] is None


def test_boot_l_on_runs():
    rng = np.random.default_rng(0)
    correct = (rng.random((6, 80)) < np.linspace(0.5, 0.95, 6)[:, None]).astype(float)
    answers = rng.integers(0, 2, 80)
    out = E.boot_l_on(correct, answers, answers, n_boot=20)
    assert out["n_boot"] == 20 and out["n_defined"] <= 20


def test_lr_tie_goes_to_3e4():
    rows = [{"lr": 1e-3, "best_lrsel": 0.9}, {"lr": 3e-4, "best_lrsel": 0.9}]
    assert E.select_lr(rows) == pytest.approx(3e-4)
    assert E.select_lr([{"lr": 1e-3, "best_lrsel": 0.95}, {"lr": 3e-4, "best_lrsel": 0.9}]) == pytest.approx(1e-3)


def test_resume_key_is_the_doing_sides():
    assert E.run_key("rule_check", 12, 1, 3e-4) == "rule_check|k12|s1|lr0.0003"
    assert E.run_key("arith_bin", 0, 0, 1e-3) == "arith_bin|k0|s0|lr0.001"


def test_first_token_r_uses_its_own_chance():
    c = {"acc": {0: [1.0], 12: [0.6]}, "chance": 0.5, "depth": 5}
    d = {"acc": {0: [1.0], 12: [0.2]}, "chance": 0.1, "ft": {0: [0.9], 12: [0.6]}, "ft_chance": 0.3, "depth": 10}
    p = E.build_pair("x_check", None, "x", c, None, d)
    assert p["R_check"] == pytest.approx(0.2) and p["R_do"] == pytest.approx(0.1 / 0.9)
    assert p["R_do_ft"] == pytest.approx((0.6 - 0.3) / (0.9 - 0.3))
    assert p["dR_ft"] == pytest.approx(0.2 - 0.5) and p["dd_cd"] == -5
    assert p["R_bin"] is None and p["dR_kind"] is None and not p["ok_kind"]
    assert spec.chance([0, 0, 1, 2]) == 0.5


def test_never_and_missing_depth_excluded():
    c = {"acc": {0: [1.0], 12: [1.0]}, "chance": 0.5, "depth": 9, "never": True}
    d = {"acc": {0: [1.0], 12: [1.0]}, "chance": 0.5, "ft": {}, "ft_chance": 0.5, "depth": 9}
    p = E.build_pair("a", None, "b", c, None, d)
    assert p["dd_cd"] is None and p["dR_ft"] is None and p["dR_cd"] == 0.0


def test_task_list_is_config(tmp_path):
    assert E.check_tasks(False, override=["rule_check", "rule_bin"]) == ("rule_check", "rule_bin")
    with pytest.raises(SystemExit):
        E.check_tasks(False, override=["rule"])
    assert E.check_tasks(True) == E.SMOKE_TASKS
    g = tmp_path / "g.json"
    names = list(E.CHECK_ORDER)
    g.write_text(json.dumps({"pairs": [{"check": c, "clean": i != 1} for i, c in enumerate(names)]}))
    got = E.check_tasks(False, gate_path=g)
    assert E.CHECK_ORDER[1] not in got and len(got) == 8 and got[:2] == (names[0], E.BIN_OF[names[0]])
    g.write_text(json.dumps({"pairs": [{"check": c, "clean": i < 2} for i, c in enumerate(names)]}))
    with pytest.raises(SystemExit):
        E.check_tasks(False, gate_path=g)                                   # fewer than 4 clean pairs: the gate stopped E2
    for n_clean, ok in ((3, False), (4, True)):
        g.write_text(json.dumps({"pairs": [{"check": c, "clean": i < n_clean} for i, c in enumerate(names)]}))
        if ok:
            assert len(E.check_tasks(False, gate_path=g)) == 2 * n_clean
        else:
            with pytest.raises(SystemExit):
                E.check_tasks(False, gate_path=g)                           # 3 clean pairs are not enough (GATE_NEED 4)
    assert E.GATE_NEED == 4 and E.B_NEED == 3
    assert E.pairs_of(("rule_check", "rule_bin", "arith_check")) == [("rule_check", "rule_bin"), ("arith_check", None)]


def test_near_type_labels():
    rec = {"meta": {"correct": True, "proposed": 1, "do_answer": 1}}
    assert E.near_type("arith_check", rec, {}) == "correct"
    wrong = {"meta": {"correct": False, "proposed": 6, "do_answer": 4}}
    assert E.near_type("arith_check", wrong, {}) == "off by +2"
    assert E.near_type("code_output_check", wrong, {}) == "distance 2"
    assert E.near_type("code_output_check", {"meta": {"correct": False, "proposed": 9, "do_answer": 4}}, {}) == "distance 3+"


def test_e2c_runs_on_synthetic_rows():
    rows = [{"task": f"t{i}", "kind": "appraise" if i % 2 else "execute", "depth": i % 7 + 1, "r": 1 - (i % 7) / 8}
            for i in range(24)]
    out = E.e2c(rows, [{"task": "c", "kind": "appraise", "depth": 12, "r": 0.1}], n_perm=50)
    assert out["tasks24"]["n"] == 24 and out["with_new"]["n"] == 25
    assert out["tasks24"]["rho"] < 0


# ---------------------------------------------------------------- R2

def test_load_do_side_excludes_own_rows(tmp_path, monkeypatch):
    c1 = tmp_path / "c1"
    c1.mkdir()
    row = lambda task, k, seed, acc: {"task": task, "k": k, "seed": seed, "lr": 3e-4, "accuracy": acc, "chance": 0.25}  # noqa: E731
    res = {"rule|k0|s0|lr0.0003": row("rule", 0, 0, 0.9), "rule|k12|s0|lr0.0003": row("rule", 12, 0, 0.5),
           "own|rule|k12|s0|lr0.0003": row("rule", 12, 0, 0.99), "own|rule|k12|s1|lr0.0003": row("rule", 12, 1, 0.99)}
    (c1 / "test_results.json").write_text(json.dumps({"results": res}))
    monkeypatch.setattr(E.C, "C1_OUT", c1)
    side = E.load_do_side("1.7B", True, ("rule_check",))["rule_check"]
    assert side["acc_full"] == {0: [0.9], 12: [0.5]}                    # the own-order 0.99 rows are not averaged in
    assert E.test_r_by_task(res)["rule"] == pytest.approx((0.5 - 0.25) / (0.9 - 0.25))


def ids_of(n, pre="r"):
    return [f"{pre}{i}" for i in range(n)]


def hits(ids, n_hit):
    return {i: int(j < n_hit) for j, i in enumerate(ids)}


def synthetic(unseen=None, seeds=None, extra_seed=False):
    ids = ids_of(20)
    rest = ids_of(20, "z")                                            # test ids only the doing side scored (and always missed)
    rh = {"x_check": {(0, 0): hits(ids, 20), (12, 0): hits(ids, 15)}, "x_bin": {(0, 0): hits(ids, 20), (12, 0): hits(ids, 17)}}
    dh = {"x": {(0, 0): {**hits(ids, 20), **hits(rest, 0)}, (12, 0): {**hits(ids, 10), **hits(rest, 0)}}}
    if extra_seed:
        for t, h in list(rh.items()) + list(dh.items()):
            h[(0, 1)], h[(12, 1)] = dict(h[(0, 0)]), dict(h[(12, 0)])
    ans = {"x_check": {i: j % 2 for j, i in enumerate(ids)}, "x_bin": {i: j % 2 for j, i in enumerate(ids)},
           "x": {i: j % 4 for j, i in enumerate(ids + rest)}}
    return E.pair_from_hits("x_check", "x_bin", "x", rh, dh, {}, ans, {"x_check": {"depth": 5}, "x_bin": {"depth": 6}, "x": {"depth": 7}},
                            12, seeds, unseen)


def test_every_r_is_on_common_ids():
    p = synthetic()
    assert p["n_common"] == 20 and p["n_scored"] == 20
    assert p["R_check"] == pytest.approx(0.5) and p["R_bin"] == pytest.approx(0.7)
    assert p["R_do"] == pytest.approx((0.5 - 0.25) / (1 - 0.25))     # the 20 extra ids the doing side alone scored are ignored
    assert p["dR_kind"] == pytest.approx(-0.2) and p["ok_kind"] and p["dd_cb"] == -1


def test_hits_file_roundtrip_and_common_ids(tmp_path):
    recs = [{"id": "a"}, {"id": "b"}]
    path = tmp_path / "h.jsonl"
    E.append_hits(path, "t", 12, 1, 3e-4, recs, [1, 0])
    E.append_hits(path, "t", 0, 1, 3e-4, recs[:1], [1])
    row = json.loads(path.read_text().splitlines()[0])
    assert row == {"id": "a", "task": "t", "k": 12, "seed": 1, "lr": 3e-4, "hit": 1}
    h = E.read_hits(path)
    assert h["t"][(12, 1)] == {"a": 1, "b": 0}
    assert E.common_ids([h["t"], {(0, 0): {"a": 1, "c": 1}, (12, 0): {"a": 0}}], (0, 12)) == ["a"]
    assert E.common_ids([h["t"], {(0, 0): {"a": 1}}], (0, 12)) == []   # a side without k=12 has no common ids


def test_arith_unseen_expression_ids():
    def rec(i, a, b):
        return {"id": i, "meta": {"a": a, "b": b, "c": None, "ops": "+"}}
    arith = {"train": [rec("t1", 1, 2)], "lrsel": [rec("l1", 3, 4)], "selection": [], "discovery": [],
             "test": [rec("x1", 1, 2), rec("x2", 5, 6), rec("x3", 3, 4)]}
    files = {("arith", s): v for s, v in arith.items()}
    files.update({("arith_check", "train"): [{"id": "t1"}], ("arith_check", "lrsel"): [{"id": "l1"}],
                  ("arith_check", "test"): [{"id": "x1"}, {"id": "x2"}, {"id": "x3"}],
                  ("arith_bin", "train"): [], ("arith_bin", "lrsel"): [], ("arith_bin", "test"): [{"id": "x1"}, {"id": "x2"}]})
    read = lambda t, s: files[(t, s)]  # noqa: E731
    assert E.arith_unseen_ids(["arith_check"], read) == {"x2"}          # x1 and x3 repeat a train / lrsel expression
    assert E.arith_unseen_ids(["arith_check", "arith_bin"], read) == {"x2"}
    assert E.arith_unseen_ids(["arith_bin"], read) == {"x1", "x2"}      # the bin task never trained on them


def test_arith_pair_scores_unseen_and_reports_all():
    p = synthetic(unseen=set(ids_of(10)))
    assert p["n_scored"] == 10 and p["n_common"] == 20 and p["all_records"]["n"] == 20
    assert p["all_records"]["R_check"] == pytest.approx(0.5)
    p2 = synthetic()
    assert "all_records" not in p2


def test_bootstrap_interval_is_seeded_and_brackets():
    a, b = synthetic(), synthetic()
    bt = a["dR_boot"]["dR_kind"]
    assert bt == b["dR_boot"]["dR_kind"] and a["dR_boot"]["n_boot"] == 1000 and a["dR_boot"]["seed"] == 0
    assert bt["lo"] <= a["dR_kind"] <= bt["hi"] and bt["n_defined"] > 0
    assert a["dR_boot"]["dR_ft"] is None                                # no ftdo hits: no ft interval


def test_ft_hits_feed_dr_ft_and_boot():
    ids = ids_of(20)
    rh = {"x_check": {(0, 0): hits(ids, 20), (12, 0): hits(ids, 15)}}
    dh = {"x": {(0, 0): hits(ids, 20), (12, 0): hits(ids, 10)}}
    fh = {"x": {(0, 0): hits(ids, 20), (12, 0): hits(ids, 12)}}
    ans = {"x_check": {i: j % 2 for j, i in enumerate(ids)}, "x": {i: j % 2 for j, i in enumerate(ids)}}
    p = E.pair_from_hits("x_check", None, "x", rh, dh, fh, ans, {}, 12)
    assert p["R_do_ft"] == pytest.approx(0.2) and p["dR_ft"] == pytest.approx(0.3)
    assert p["dR_boot"]["dR_ft"]["n_defined"] > 0


def test_ftdo_doc_to_hits():
    doc = {"ids": {"x": ["a", "b"]}, "runs": {"k": {"task": "x", "k": 12, "seed": 0, "ft_correct": [1, 0]}}}
    assert E.ftdo_hit_runs(doc) == {"x": {(12, 0): {"a": 1, "b": 0}}}
    assert E.ftdo_hit_runs({}) == {}


def test_e2a_uses_eligible_pairs_only():
    good = [{**pair(-6, 0.0), "ok_kind": True}] * 3
    bad = [{**pair(2, 0.0), "ok_kind": False, "ok_cd": False}] * 5          # ineligible pairs reading late must not count
    v = E.verdicts(good + bad)
    assert v["E2-A"]["label"] == "check reads early" and v["E2-A"]["n"] == 3


def test_e2c_test_split_r_and_labelling():
    res = {"a|k0|s0|lr1": {"task": "a", "k": 0, "seed": 0, "accuracy": 0.9, "chance": 0.5},
           "a|k12|s0|lr1": {"task": "a", "k": 12, "seed": 0, "accuracy": 0.7, "chance": 0.5},
           "b|k0|s0|lr1": {"task": "b", "k": 0, "seed": 0, "accuracy": 0.9, "chance": 0.5}}      # b has no k=12: no test R
    r = E.test_r_by_task(res)
    assert r == {"a": pytest.approx(0.5)}
    rows = [{"task": t, "kind": "execute", "depth": 3, "r": r_, "r_source": "test" if t in r else "selection"} for t, r_ in (("a", r["a"]), ("b", 0.2))]
    assert [x["r_source"] for x in rows] == ["test", "selection"]


def test_stage1_gate_markers_and_stage_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mk = lambda ok: {"check": "x_check", "n_common": 5, "n_scored": 5, "R_do": 0.1, "R_bin": 0.5, "R_check": 0.5,  # noqa: E731
                     "dR_kind": 0.0, "dR_boot": {}, "ok_kind": ok}
    for n_ok, force, marker, other in ((3, None, "STAGE1_GO", "STAGE1_STOP"), (2, None, "STAGE1_STOP", "STAGE1_GO"),
                                       (0, 3, "STAGE1_GO", "STAGE1_STOP"), (5, 0, "STAGE1_STOP", "STAGE1_GO")):
        monkeypatch.setattr(E, "size_pairs", lambda *a, _n=n_ok, **k: ([mk(i < _n) for i in range(5)], {}))
        monkeypatch.setattr(E, "check_tasks", lambda smoke, override=None: E.SMOKE_TASKS)
        assert E.stage_eligibility(True, force) == 0
        root = tmp_path / "var/results/atlas3/e2chk_smoke"
        assert (root / marker).exists() and not (root / other).exists()          # one marker at a time
        z = json.loads((root / "stage1.json").read_text())
        assert z["go"] == (marker == "STAGE1_GO") and z["need"] == 3
        assert marker in (root / "stage1.md").read_text()
    root = tmp_path / "var/results/atlas3/e2chk"
    monkeypatch.setattr(E, "size_pairs", lambda *a, **k: ([mk(True)] * 3, {}))
    E.stage_eligibility(False, None)
    assert (root / "STAGE1_GO").exists() and (root / "stage1.md").exists() and (root / "stage1.json").exists()




# ---------------------------------------------------------------- R3

def test_e2c_mixed_splits_get_no_verdict():
    rows = [{"task": f"t{i}", "kind": "appraise" if i % 2 else "execute", "depth": i % 7 + 1, "r": 1 - (i % 7) / 8, "r_source": "test"}
            for i in range(24)]
    clean = E.e2c(rows, [], n_perm=50)["tasks24"]
    assert not clean["descriptive"] and not clean["partial_verdict"].startswith("incomplete")
    mixed_rows = [{**r, "r_source": "selection"} if i < 19 else r for i, r in enumerate(rows)]
    out = E.e2c(mixed_rows, [{"task": "c", "kind": "appraise", "depth": 12, "r": 0.1, "r_source": "test"}], n_perm=50)
    t = out["tasks24"]
    assert t["partial_verdict"] == "incomplete (19 of 24 rows on the selection split)" and t["descriptive"]
    assert t["rho"] is not None and "partial" in t                            # numbers kept
    assert out["with_new"]["partial_verdict"].startswith("incomplete (19 of 25")


def test_hits_tolerate_truncated_last_line_and_write_atomically(tmp_path, capsys):
    path = tmp_path / "h.jsonl"
    recs = [{"id": "a"}, {"id": "b"}]
    E.append_hits(path, "t", 0, 0, 3e-4, recs, [1, 1])
    path.write_text(path.read_text() + '{"id": "c", "task": "t", "k": 0, "se')          # an interrupted write
    assert E.read_hits(path)["t"][(0, 0)] == {"a": 1, "b": 1}
    assert "WARNING" in capsys.readouterr().out
    E.append_hits(path, "t", 12, 0, 3e-4, recs, [0, 1])                                  # rewrites without the torn line
    assert all(json.loads(ln) for ln in path.read_text().splitlines())
    E.append_hits(path, "t", 0, 0, 3e-4, recs, [0, 0])                                   # a re-score replaces, never duplicates
    assert len(path.read_text().splitlines()) == 4 and E.read_hits(path)["t"][(0, 0)] == {"a": 0, "b": 0}
    assert not list(tmp_path.glob("*.tmp*"))
    path.write_text('{"bad\n' + json.dumps({"id": "a", "task": "t", "k": 0, "seed": 0, "lr": 1, "hit": 1}) + "\n")
    with pytest.raises(ValueError):
        E.read_hits(path)                                                                 # a bad line that is not the last one is an error


def test_doexact_rescores_a_key_with_missing_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(E.ac, "OUT", tmp_path)
    recs = [{"id": "a", "answer": 0}, {"id": "b", "answer": 1}]
    cfg = SimpleNamespace(smoke=False, sfx="", tasks=("arith_check",), ks=(0, 12))
    ctx = SimpleNamespace(cfg=cfg, records=lambda t, sp: recs)
    row = {"task": "arith", "k": 0, "seed": 0, "lr": 3e-4, "ckpt": "x"}
    monkeypatch.setattr(E, "doing_rows", lambda *a: [row])
    calls = []

    def fake(ctx_, r, t):
        calls.append(t)
        return 0.5, recs, [1, 0]
    monkeypatch.setattr(E, "exact_hits", fake)
    key = E.run_key("arith", 0, 0, 3e-4)
    E.ac.write(tmp_path / "doexact.json", {"runs": {key: {"task": "arith"}}})           # key present, hits file absent
    assert E.stage_doexact(ctx, "1.7B") == 0 and len(calls) == 1
    assert E.read_hits(tmp_path / "doexact_hits.jsonl")["arith"][(0, 0)] == {"a": 1, "b": 0}
    assert E.stage_doexact(ctx, "1.7B") == 0 and len(calls) == 1                         # complete now: skipped
    E.append_hits(tmp_path / "doexact_hits.jsonl", "arith", 0, 0, 3e-4, recs[:1], [1])    # a run missing record b
    assert E.stage_doexact(ctx, "1.7B") == 0 and len(calls) == 2
    assert len(E.read_hits(tmp_path / "doexact_hits.jsonl")["arith"][(0, 0)]) == 2


def test_smoke_doing_side_sits_at_the_smoke_cut(monkeypatch):
    runs = {"a": {"task": "arith_bin", "k": 0, "seed": 0, "lr": 3e-4, "ckpt": "p0"},
            "b": {"task": "arith_bin", "k": 2, "seed": 0, "lr": 3e-4, "ckpt": "p2"},
            "c": {"task": "arith_check", "k": 2, "seed": 0, "lr": 3e-4, "ckpt": "q2"}}
    monkeypatch.setattr(E.ac, "collect", lambda runs_, tasks, ks, rule: {"arith_bin": {k: [r for r in runs_.values() if r["task"] == "arith_bin" and r["k"] == k] for k in ks}})
    rows = E.smoke_do_rows(runs, (0, 2))
    assert [(r["task"], r["k"], r["ckpt"]) for r in rows] == [("arith", 0, "p0"), ("arith", 2, "p2")]
    # those rows give the smoke pair real common ids at k=2 (k=12 would leave none)
    ids = ids_of(8)
    rh = {"arith_check": {(0, 0): hits(ids, 8), (2, 0): hits(ids, 6)}, "arith_bin": {(0, 0): hits(ids, 8), (2, 0): hits(ids, 7)}}
    dh = {"arith": {(0, 0): hits(ids, 8), (2, 0): hits(ids, 4)}}
    ans = {t: {i: j % 2 for j, i in enumerate(ids)} for t in ("arith_check", "arith_bin", "arith")}
    assert E.pair_from_hits("arith_check", "arith_bin", "arith", rh, dh, {}, ans, {}, 2, boot=False)["n_common"] == 8
    assert E.pair_from_hits("arith_check", "arith_bin", "arith", rh, dh, {}, ans, {}, 12, boot=False)["n_common"] == 0
