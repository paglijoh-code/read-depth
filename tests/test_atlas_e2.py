"""Robustness E2, revision R1: check tasks 26-30 and binary doing tasks 31-35 (pairing with the do tasks, 50/50 balance, legality,
determinism, no cheap leak), the doing tasks unchanged, and the Mac gate's rivals and verdict. No model weights."""
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from surgery.atlas import spec, tasks  # noqa: E402

CHECK = list(tasks.CHECK_DO)
BIN = [tasks.BIN_OF_CHECK[t] for t in CHECK]
DATA = ROOT / "var/data/atlas"
N = 400
needs_data = pytest.mark.skipif(not (DATA / "manifest4.json").exists(), reason="E2 data not built")
LEAK = 0.60


def proposed(rec):
    return rec["prompt"].rsplit(tasks.CHECK_PROPOSAL, 1)[1]


def test_registry():
    assert [spec.TASKS[t].number for t in CHECK] == [26, 27, 28, 29, 30]
    assert [spec.TASKS[t].number for t in BIN] == [31, 32, 33, 34, 35]
    for t in CHECK:
        ts = spec.TASKS[t]
        assert ts.kind == "appraise" and ts.options == ("yes", "no") and ts.system.endswith("Output only the answer.")
    for t in BIN:
        ts = spec.TASKS[t]
        assert ts.kind == "execute" and ts.options == ("yes", "no") and ts.system.endswith("Output only the answer.")
    assert tasks.E2_FAMILIES == tuple(CHECK) and tasks.BIN_FAMILIES == tuple(BIN)
    assert not set(tasks.E2_FAMILIES + tasks.BIN_FAMILIES) & set(tasks.C_FAMILIES)
    assert [spec.TASKS[t].number for t in tasks.C_FAMILIES] == [22, 23, 24, 25]


def test_bin_sets_are_fixed_halves():
    for t in BIN:
        do = tasks.BIN_DO[t]
        a = tasks.bin_set(t)
        assert a == tasks.bin_set(t) and len(a) == len(spec.TASKS[do].options) // 2
        assert tasks.bin_suffix(t).startswith(tasks.BIN_QUESTION) and tasks.bin_suffix(t).endswith("?")


@pytest.mark.parametrize("task", CHECK)
def test_check_pairing_balance_legality_and_determinism(task):
    do = tasks.CHECK_DO[task]
    recs = tasks.generate(task, N, 3)
    assert recs == tasks.generate(task, N, 3) and recs != tasks.generate(task, N, 4)
    base = tasks.generate(do, N, 3)
    opts = spec.TASKS[do].options
    ans = Counter(r["answer"] for r in recs)
    assert abs(ans[0] - ans[1]) <= 1 and 0 < len(recs) <= N
    by_id = {d["id"]: d for d in base}
    assert len({r["id"] for r in recs}) == len(recs)
    for r in recs:
        d = by_id[r["id"]]
        assert r["template_id"] == d["template_id"] and r["rung"] == d["rung"]
        assert r["prompt"] == d["prompt"] + tasks.CHECK_PROPOSAL + opts[r["meta"]["proposed"]]
        assert proposed(r) in opts and r["meta"]["do_answer"] == d["answer"]
        assert r["meta"]["correct"] == (r["answer"] == 0) == (r["meta"]["proposed"] == d["answer"])


@pytest.mark.parametrize("task", BIN)
def test_bin_pairing_balance_and_determinism(task):
    do = tasks.BIN_DO[task]
    recs = tasks.generate(task, N, 3)
    assert recs == tasks.generate(task, N, 3) and recs != tasks.generate(task, N, 4)
    by_id = {d["id"]: d for d in tasks.generate(do, N, 3)}
    ans = Counter(r["answer"] for r in recs)
    assert abs(ans[0] - ans[1]) <= 1 and 0 < len(recs) <= N
    a = set(tasks.bin_set(task))
    for r in recs:
        d = by_id[r["id"]]
        assert r["prompt"] == d["prompt"] + tasks.bin_suffix(task) and r["template_id"] == d["template_id"]
        assert (r["answer"] == 0) == (d["answer"] in a)


def test_near_miss_rules():
    do = {t: {r["id"]: r for r in tasks.generate(tasks.CHECK_DO[t], 300, 7)} for t in CHECK}
    for r in tasks.generate("arith_check", 300, 7):
        if not r["meta"]["correct"]:
            d = do["arith_check"][r["id"]]
            assert abs(int(proposed(r)) - d["answer"]) == 2 and int(proposed(r)) % 2 == d["answer"] % 2 and 0 <= int(proposed(r)) <= 9
    for r in tasks.generate("rule_check", 200, 7):
        d = do["rule_check"][r["id"]]
        if not r["meta"]["correct"]:
            acts = tasks._pos_features  # noqa: F841  (position features are balanced; legality below)
            assert proposed(r) != d["meta"]["label"]
            assert f"then {proposed(r)} [end]" in "\n".join(l for l in d["prompt"].splitlines() if l.startswith("Rule "))
    for r in tasks.generate("hard_check", 200, 7):
        d = do["hard_check"][r["id"]]
        assert r["meta"]["correct"] or proposed(r) != d["meta"]["label"]
    for r in tasks.generate("code_output_check", 300, 7):
        d = do["code_output_check"][r["id"]]
        assert r["meta"]["correct"] or int(proposed(r)) != d["answer"]
    for r in tasks.generate("code_literal_check", 300, 7):
        d = do["code_literal_check"][r["id"]]
        assert r["meta"]["correct"] or int(proposed(r)) != d["answer"]


def test_loop_step_and_trace_helpers():
    code = "x = 1\nfor i in range(3):\n    x += 2\nprint(x)"
    assert tasks._loop_step_outputs(code) == [5, 9]          # one step early, one step late
    assert set(tasks._trace_values(code)) >= {1, 3, 5, 7, 2}
    assert tasks._literal_near("a = 2\nb = 5\nprint(a)", 2) == [5]
    assert tasks._literal_near("a = 2\nb = 2\nprint(a)", 2) == []
    assert tasks._assign_events("x = 1\ny = 2\nx = 3\nprint(x)")[-1][1] == 3


@needs_data
@pytest.mark.parametrize("task", CHECK + BIN)
def test_built_files_pair_with_do_files(task):
    do = tasks.CHECK_DO.get(task) or tasks.BIN_DO[task]
    for split, size in spec.SPLITS.items():
        recs, base = spec.read_records(task, split, DATA), spec.read_records(do, split, DATA)
        assert len(base) == size and 0 < len(recs) <= size
        assert abs(sum(r["answer"] == 0 for r in recs) - len(recs) / 2) <= 1
        by_id = {d["id"]: d for d in base}
        assert len(recs) == len({r["id"] for r in recs})
        for r in recs:
            d = by_id[r["id"]]
            assert r["template_id"] == d["template_id"] and r["prompt"].startswith(d["prompt"])
        assert len({r["template_id"] for r in recs}) >= 20
    train = {r["template_id"] for r in spec.read_records(task, "train", DATA)}
    assert not train & {r["template_id"] for r in spec.read_records(task, "test", DATA)}


@needs_data
@pytest.mark.parametrize("task", ["rule_check", "hard_check", "arith_check", "code_output_check"])
def test_no_cheap_leak_on_train(task):
    import atlas_e2_gate as g
    recs = spec.read_records(task, "train", DATA)
    for name, acc in g.heuristics(task, recs).items():
        if "parity" in name or "last-assigned" in name or "Rule 1" in name or "Disabled" in name:
            assert acc <= LEAK, (task, name, acc)


@needs_data
def test_doing_tasks_unchanged_by_r1():
    """Tasks 1-25 are byte-identical: every doing-task file hashes as the earlier manifests recorded."""
    seen = 0
    for name in ("manifest.json", "manifest2.json", "manifest3.json"):
        mp = DATA / name
        if mp.exists():
            for f, sha in json.loads(mp.read_text())["files"].items():
                assert hashlib.sha256((DATA / f).read_bytes()).hexdigest() == sha, f
                seen += 1
    assert seen > 0
    files = json.loads((DATA / "manifest4.json").read_text())["files"]
    assert all(f.split("_")[0] + "_" in f and (f.startswith(tuple(CHECK + BIN))) for f in files)


@needs_data
def test_manifests_match_files():
    for name in ("manifest.json", "manifest2.json", "manifest3.json", "manifest4.json"):
        mp = DATA / name
        if not mp.exists():
            continue
        for f, sha in json.loads(mp.read_text())["files"].items():
            assert hashlib.sha256((DATA / f).read_bytes()).hexdigest() == sha, f


@needs_data
def test_gate_rivals_and_verdict():
    import atlas_e2_gate as g
    row = g.gate_task("arith_bin", DATA)
    assert {"bow_whole", "side_only", "structural", "clean"} <= set(row) and 0.3 <= row["structural"] <= 0.8
    rows = []
    for c in g.CHECK_TASKS:
        rows += [{"task": c, "clean": True}, {"task": tasks.BIN_OF_CHECK[c], "clean": c != g.CHECK_TASKS[0]}]
    v = g.verdict(rows)
    assert v["n_clean"] == 4 and v["verdict"] == "gate passes" and not v["pairs"][0]["clean"]
    rows[3]["clean"] = False
    assert g.verdict(rows)["verdict"] == "stop"


# ---------------------------------------------------------------- round 2 (last gate round)

def test_arith_bin_uses_fixed_half_and_it_is_in_the_options():
    a = tasks.bin_set("arith_bin")
    assert a == tasks.BIN_SET_FIXED["arith_bin"] == (1, 5, 6, 8, 9)
    assert tasks.bin_set("code_literal_bin") == tasks.bin_set("code_literal_bin") and "code_literal_bin" not in tasks.BIN_SET_FIXED


@needs_data
def test_arith_bin_expressions_do_not_cross_train_and_test():
    core = lambda r: (r["meta"]["a"], r["meta"]["b"], r["meta"]["c"], r["meta"]["ops"])  # noqa: E731
    by_id = {r["id"]: core(r) for s in ("train", "lrsel", "test") for r in spec.read_records("arith", s, DATA)}
    fit = {by_id[r["id"]] for s in ("train", "lrsel") for r in spec.read_records("arith_bin", s, DATA)}
    assert not fit & {by_id[r["id"]] for r in spec.read_records("arith_bin", "test", DATA)}


@needs_data
@pytest.mark.parametrize("task", ["arith_check", "code_output_check", "arith_bin", "code_literal_bin"])
def test_round2_tasks_keep_200_test_records(task):
    assert len(spec.read_records(task, "test", DATA)) >= 200


def test_gate_and_run_read_r2():
    import atlas_e2_gate as g
    import atlas_e2_run as r
    assert g.NAME == "e2_gate_r2" and r.GATE.name == "e2_gate_r2.json"
    assert set(g.COPIED) == {"rule_check", "hard_check", "rule_bin", "hard_bin"}
