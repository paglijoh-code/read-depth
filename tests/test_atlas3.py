"""M9 Phase 2 stage C0: tasks 22-25, the Phase 0-2 data guard, the map --model flag and the stage C gate (no model weights)."""
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from surgery.atlas import spec, tasks  # noqa: E402

NEW = ["code_traces_first", "syntax_outer_pair", "target_read_bare", "bind_compare"]


def test_registry():
    assert [spec.TASKS[t].number for t in NEW] == [22, 23, 24, 25]
    assert all(spec.TASKS[t].kind == "appraise" and spec.TASKS[t].options == ("yes", "no") for t in NEW)
    assert spec.TASKS["target_read_bare"].system == spec.TASKS["hard_target_read"].system
    assert len(spec.PHASE1_TASKS) == 14 and not set(NEW) & set(spec.PHASE1_TASKS)
    assert tasks.C_FAMILIES == tuple(NEW) and not set(NEW) & set(tasks.NEW_FAMILIES)


@pytest.mark.parametrize("name,count", [("manifest.json", 70), ("manifest2.json", None)])
def test_old_data_files_match_manifests(name, count):
    mpath = ROOT / "var/data/atlas" / name
    if not mpath.exists():
        pytest.skip(f"no built {name}")
    files = json.loads(mpath.read_text())["files"]
    if count:
        assert len(files) == count
    for f, sha in files.items():
        assert hashlib.sha256((ROOT / "var/data/atlas" / f).read_bytes()).hexdigest() == sha, f


def test_stage_a_target_read_is_unchanged_by_the_refactor():
    a = tasks.generate("hard_target_read", 40, 3)
    assert a == tasks.generate("hard_target_read", 40, 3)
    assert all(re.search(r"^Status:", r["prompt"], re.M) and not re.search(r"^target_read=", r["prompt"], re.M) for r in a)


@pytest.mark.parametrize("task", NEW)
def test_deterministic_and_schema(task):
    a = tasks.generate(task, 80, 3)
    assert a == tasks.generate(task, 80, 3) and a != tasks.generate(task, 80, 4)
    assert len({r["id"] for r in a}) == 80
    for r in a:
        assert r["task"] == task and r["answer"] in (0, 1) and r["template_id"]


def label_dist(recs, feature):
    """Share of each feature value under each label."""
    d = {y: Counter() for y in (0, 1)}
    for r in recs:
        d[r["answer"]].update(feature(r))
    return {y: {k: v / sum(d[y].values()) for k, v in d[y].items()} for y in d}


def assert_same_dist(recs, feature, tol=0.02):
    d = label_dist(recs, feature)
    for k in set(d[0]) | set(d[1]):
        assert abs(d[0].get(k, 0) - d[1].get(k, 0)) < tol, (k, d[0].get(k, 0), d[1].get(k, 0))


# ---------------------------------------------------------------- 22 code_traces_first

def body(r):
    return r["prompt"].split("\n", 1)[1] if r["prompt"].split("\n")[0].endswith(":") else r["prompt"]


def test_trace_info_hand_built():
    code = "aa = 5\nbb = aa\ncc = bb\nprint(cc)"
    assert tasks.trace_info(code) == (2, 1)
    # a mirror: the copy line reads another constant, the label flips
    assert tasks.trace_info("aa = 5\ndd = 7\ncc = dd\nprint(cc)") == (1, 2)
    assert tasks.trace_info("aa = 5\ndd = 7\ncc = aa\nprint(cc)") == (1, 1)
    assert tasks.trace_info("aa = 5\ndd = 7\ncc = aa\nee = dd\nprint(ee)") == (1, 2)


def test_traces_labels_hops_and_balance():
    recs = tasks.generate("code_traces_first", 2000, 2)
    cells = Counter()
    for r in recs:
        code = body(r)
        lines = code.split("\n")
        assert 4 <= len(lines) <= 5 and lines[-1].startswith("print(") and run_ok(code)
        hops, root = tasks.trace_info(code)
        assert hops == r["meta"]["hops"] and (root == 1) == (r["answer"] == 0)
        cells[(hops, r["answer"])] += 1
        if hops == 2:
            assert len(lines) == 5
    assert set(h for h, _ in cells) == {1, 2}
    for h in (1, 2):
        assert cells[(h, 0)] == cells[(h, 1)], cells


def run_ok(code):
    tasks.run_py(code)
    return True


def test_traces_surface_same_under_both_labels():
    recs = tasks.generate("code_traces_first", 6000, 2)
    assert_same_dist(recs, lambda r: [len(body(r).split("\n"))])
    assert_same_dist(recs, lambda r: [str(sum(ln.split(" = ")[1].isdigit() for ln in body(r).split("\n")[:-1]))])   # constant lines
    assert_same_dist(recs, lambda r: [ln.split(" = ")[0] for ln in body(r).split("\n")[:-1]])


# ---------------------------------------------------------------- 23 syntax_outer_pair

def test_outer_pair_labels_and_identical_multiset():
    recs = tasks.generate("syntax_outer_pair", 600, 2)
    close = {")": "(", "]": "[", "}": "{"}
    cells = Counter()
    for r in recs:
        s = body(r).split()
        assert Counter(s) == Counter("( ( [ [ { { ) ) ] ] } }".split())
        stack, partner = [], {}
        for j, ch in enumerate(s):
            if ch in "([{":
                stack.append((j, ch))
            else:
                i, o = stack.pop()
                assert o == close[ch]
                partner[i] = j
        assert not stack and (partner[0] == len(s) - 1) == (r["answer"] == 0)
        assert 1 <= tasks.bracket_depth("".join(s)) <= 4
        cells[(s[0], r["answer"])] += 1
    for k in "([{":
        assert cells[(k, 0)] == cells[(k, 1)], cells
    assert Counter(r["answer"] for r in recs) == {0: 300, 1: 300}


def test_outer_pair_hand_built_and_mirror():
    assert tasks._wraps("oocc") and tasks._wraps("oococc")
    assert not tasks._wraps("ococ") and not tasks._wraps("ooccoc")
    # the mirror: the same pairs, the outer pair closed early instead of at the end
    assert tasks._wraps("o" + "oc" + "oc" + "c") and not tasks._wraps("oc" + "oc" + "oc")


# ---------------------------------------------------------------- 24 target_read_bare

def test_target_read_bare_no_status_and_balance():
    recs = tasks.generate("target_read_bare", 600, 2)
    cells = Counter()
    for r in recs:
        p = r["prompt"]
        assert "Status" not in p and not re.search(r"^[a-z_]+=(yes|no)$", p, re.M)
        assert "History (oldest first):" in p and "Disabled rules" in p and "\n\n\n" not in p
        y, cnt = tasks.hard_read_info(p)
        assert y == r["answer"] and 1 <= cnt <= 5
        cells[(cnt, y)] += 1
    for c in range(1, 6):
        assert abs(cells[(c, 0)] - cells[(c, 1)]) <= 1, cells


def test_strip_status_hand_built():
    p = "A\n\nHistory (oldest first):\n1. X\n\nStatus:\nk=yes\nj=no\n\nDisabled rules (never apply these):\n(none)"
    assert tasks.strip_status(p) == "A\n\nHistory (oldest first):\n1. X\n\nDisabled rules (never apply these):\n(none)"
    with pytest.raises(AssertionError):
        tasks.strip_status("no block here")


# ---------------------------------------------------------------- 25 bind_compare

def test_bind_compare_labels_and_mirror():
    recs = tasks.generate("bind_compare", 800, 2)
    for r in recs:
        lines = body(r).split("\n")
        assert 4 <= len(lines) <= 5
        vals = {}
        for ln in lines[:-1]:
            nm, v = ln.split(" = ")
            vals[nm] = int(v)
        assert len(vals) == len(lines) - 1 and len(set(vals.values())) == len(vals) and all(len(n) >= 2 for n in vals)
        _, n1, n2 = lines[-1].split()
        assert n1 != n2 and (vals[n1] > vals[n2]) == (r["answer"] == 0)
        assert (vals[n2] > vals[n1]) == (r["answer"] == 1)       # the mirror (swap the two values) flips it
    assert Counter(r["answer"] for r in recs) == {0: 400, 1: 400}


def test_bind_compare_surface_same_under_both_labels():
    recs = tasks.generate("bind_compare", 6000, 2)
    assert_same_dist(recs, lambda r: [ln.split(" = ")[1] for ln in body(r).split("\n")[:-1]])
    assert_same_dist(recs, lambda r: [ln.split(" = ")[0] for ln in body(r).split("\n")[:-1]])
    assert_same_dist(recs, lambda r: [len(body(r).split("\n"))])


# ---------------------------------------------------------------- templates and splits

SMALL = {"discovery": 120, "train": 100, "lrsel": 40, "selection": 60, "test": 60}


def test_strict_templates_stop_at_the_check_and_name_sets_pass(monkeypatch):
    assert tasks.NAMES_IN_TEMPLATE is True  # decided 2026-11-11 before any build
    monkeypatch.setattr(tasks, "NAMES_IN_TEMPLATE", False)
    with pytest.raises(AssertionError, match="templates"):
        tasks.make_splits("bind_compare", 9, spec.SPLITS)
    monkeypatch.setattr(tasks, "NAMES_IN_TEMPLATE", True)
    for fam in ("bind_compare", "code_traces_first"):
        splits, info = tasks.make_splits(fam, 9, SMALL)
        assert not any(info["quota_shortfall"].values())
    # a mirror (same names, same positions, values swapped) shares its template
    a = [r for r in tasks.generate("bind_compare", 400, 5)]
    by_t = {}
    for r in a:
        by_t.setdefault(r["template_id"], set()).add(r["answer"])
    assert any(len(v) == 2 for v in by_t.values())


@pytest.mark.parametrize("family", ["syntax_outer_pair", "target_read_bare"])
def test_strict_families_split(family):
    splits, info = tasks.make_splits(family, 9, SMALL)
    assert not any(info["quota_shortfall"].values())
    for items in splits.values():
        assert Counter(it["records"][family]["answer"] for it in items)[0] == len(items) // 2


# ---------------------------------------------------------------- map --model and the gate

def test_map_model_flag_defaults_to_06b(capsys):
    import atlas_map
    assert atlas_map.MODEL == "Qwen/Qwen3-0.6B"
    with pytest.raises(SystemExit):
        atlas_map.main(["--help"])
    out = capsys.readouterr().out
    assert "--model" in out and "Qwen/Qwen3-0.6B" in out


def test_gate(tmp_path):
    import atlas3_gate as g
    ok = [{"task": f"t{i}", "bow": 0.5, "chance": 0.5, "L_on": 5} for i in range(2)] + \
         [{"task": "t2", "bow": 0.55, "chance": 0.5, "L_on": 2}, {"task": "t3", "bow": 0.98, "chance": 0.5, "L_on": 9}]
    v = g.gate(ok)
    assert v["pass"] and v["n_clean"] == 3 and v["n_clean_late"] == 2
    assert not g.gate([dict(r, L_on=1) for r in ok])["pass"]
    assert not g.gate([dict(r, bow=0.9) for r in ok])["pass"]
    assert not g.gate(ok[:2] + [dict(ok[2], bow=0.7), dict(ok[3])])["pass"]       # only 2 clean
    assert g.gate([dict(r, L_on=None) for r in ok[:3]] + [ok[3]])["n_clean_late"] == 0
    # edge: a gap of exactly 0.10 is clean
    assert g.gate([{"task": "t", "bow": 0.6, "chance": 0.5, "L_on": 4}])["rows"][0]["clean"]


def test_gate_main_reads_files(tmp_path):
    import atlas3_gate as g
    (tmp_path / "1.7B").mkdir()
    base = {t: {"bow": 0.5, "chance": 0.5} for t in g.GATE_TASKS}
    (tmp_path / "baseline.json").write_text(json.dumps({"tasks": base}))
    for t in g.GATE_TASKS:
        (tmp_path / "1.7B" / f"map_{t}.json").write_text(json.dumps({"L_on": 6}))
    assert g.main(["1.7B", "--dir", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "1.7B" / "gate.json").read_text())["pass"] is True
