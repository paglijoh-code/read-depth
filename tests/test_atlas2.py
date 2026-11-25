"""M9 Phase 2 stage A: new task generators, the Phase 1 data guard, L_on_healed and the gate (no model weights)."""
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

NEW = ["arith_order", "syntax_first_close", "code_uses_first", "hard_target_read", "arith_id", "code_literal", "arith_long"]
OLD = ["route_judgement", "route_work", "code_shape", "fact_topic", "hard_hops", "rule", "sensible", "hard", "arith",
       "fact_check", "syntax_valid", "code_output", "arith_shape", "syntax_depth"]


def test_phase1_tasks_and_registry():
    assert list(spec.PHASE1_TASKS) == OLD or sorted(spec.PHASE1_TASKS, key=lambda t: spec.TASKS[t].number) == list(spec.PHASE1_TASKS)
    assert [spec.TASKS[t].number for t in spec.PHASE1_TASKS] == list(range(1, 15))
    assert [spec.TASKS[t].number for t in NEW] == list(range(15, 22))
    assert set(spec.PHASE1_TASKS) | set(NEW) <= set(spec.TASKS)  # stage C adds 22-25


def test_old_data_files_match_manifest():
    mpath = ROOT / "var/data/atlas/manifest.json"
    if not mpath.exists():
        pytest.skip("no built Phase 1 data")
    files = json.loads(mpath.read_text())["files"]
    assert len(files) == 70
    for name, sha in files.items():
        assert hashlib.sha256((ROOT / "var/data/atlas" / name).read_bytes()).hexdigest() == sha, name


@pytest.mark.parametrize("task", NEW)
def test_deterministic_and_schema(task):
    a = tasks.generate(task, 60, 3)
    assert a == tasks.generate(task, 60, 3) and a != tasks.generate(task, 60, 4)
    assert len({r["id"] for r in a}) == 60
    for r in a:
        assert r["task"] == task and 0 <= r["answer"] < len(spec.TASKS[task].options) and r["template_id"]


def toks(r):
    return Counter(r["prompt"].split())


def test_arith_order_labels_and_mirror():
    for r in tasks.generate("arith_order", 400, 2):
        body = r["prompt"].split("\n")[-1]
        t = body.split()
        assert len(t) in (3, 5) and (int(t[0]) > int(t[-1])) == (r["answer"] == 0) and t[0] != t[-1]
        mirror = " ".join([t[-1]] + t[1:-1] + [t[0]])
        assert (int(mirror.split()[0]) > int(mirror.split()[-1])) == (r["answer"] == 1)
        assert re.fullmatch(r"[0-9+\-* ]+", body)


def test_arith_order_tokens_same_under_both_labels():
    recs = tasks.generate("arith_order", 4000, 2)
    for kind in ("digit", "op"):
        dist = {y: Counter() for y in (0, 1)}
        for r in recs:
            body = r["prompt"].split("\n")[-1].split()
            dist[r["answer"]].update(t for t in body if t.isdigit() == (kind == "digit"))
        for tok in dist[0]:
            a, b = dist[0][tok] / sum(dist[0].values()), dist[1][tok] / sum(dist[1].values())
            assert abs(a - b) < 0.02, (kind, tok, a, b)


def test_syntax_first_close_labels_and_identical_multiset():
    close = {")": 0, "]": 1, "}": 2}
    for r in tasks.generate("syntax_first_close", 300, 2):
        s = r["prompt"].split("\n")[-1].split()
        assert Counter(s) == Counter("( ( [ [ { { ) ) ] ] } }".split())
        stack, first = [], None
        for ch in s:
            if ch in "([{":
                stack.append(ch)
            else:
                assert stack.pop() == "([{"[close[ch]]
                first = close[ch] if first is None else first
        assert not stack and first == r["answer"] and tasks.bracket_depth("".join(s)) <= 3


def test_code_uses_first_labels_and_tokens():
    recs = tasks.generate("code_uses_first", 600, 2)
    for r in recs:
        lines = r["prompt"].split("\n")
        names = [ln.split(" = ")[0] for ln in lines[:3]]
        assert len(set(names)) == 3 and all(len(n) >= 2 for n in names)
        m = re.fullmatch(r"print\((\w+)( [-+*] \d)?\)", lines[3])
        assert m and m.group(1) in names and (m.group(1) == names[0]) == (r["answer"] == 0)
        assert r["template_id"].startswith("+".join(sorted(names)))
    assert Counter(r["answer"] for r in recs) == {0: 300, 1: 300}


def test_hard_target_read_parse_and_balance_within_count():
    recs = tasks.generate("hard_target_read", 500, 2)
    cells = Counter()
    for r in recs:
        y, cnt = tasks.hard_read_info(r["prompt"])
        assert y == r["answer"] and 1 <= cnt <= 5 and f"read={'yes' if y == 0 else 'no'}" in r["prompt"]
        cells[(cnt, y)] += 1
    for c in range(1, 6):
        assert abs(cells[(c, 0)] - cells[(c, 1)]) <= 1, cells
    # no Status value line gives the answer away (the rules may still name the fact: "if target_read=yes ...")
    assert not any(re.search(r"^target_read=", r["prompt"], re.M) for r in recs)
    hand = "Target file: a/b.py\n\nFiles:\n- a/b.py: read=yes edited=no\n- c/d.py: read=no edited=no\n\nTests:\n"
    assert tasks.hard_read_info(hand) == (0, 1)


def test_arith_id_and_code_literal_and_long():
    for r in tasks.generate("arith_id", 300, 2):
        body = re.sub(r"^[A-Za-z: ]+?(?=[\d(])", "", r["prompt"]).rstrip("?")
        assert eval(body) == r["answer"] == r["meta"]["x"]
    for r in tasks.generate("code_literal", 300, 2):
        assert tasks.run_py(r["prompt"]) == str(r["answer"]) and 2 <= r["meta"]["n_assign"] <= 3
    for r in tasks.generate("arith_long", 300, 2):
        v = r["meta"]["value"]
        assert 10 <= v <= 999 and r["answer"] == int(str(v)[0])


@pytest.mark.parametrize("task", NEW)
def test_templates_enough(task):
    n = {"hard_target_read": 1200}.get(task, 3000)
    assert len({r["template_id"] for r in tasks.generate(task, n, 1)}) >= 100


# ---------------------------------------------------------------- L_on_healed and the gate

def test_l_on_healed():
    import atlas2_healed as h
    curve = [0.1, 0.1, 0.2, 0.6, 0.9, 0.9]
    assert h.l_on_healed(curve, 0.1, 0.9) == (3, False)          # threshold 0.5
    assert h.l_on_healed([0.1, 0.2, 0.3], 0.1, 0.9) == (3, True)   # never reached: the curve's length
    assert h.l_on_healed(curve, 0.5, 0.53) == (None, False)        # healed accuracy near chance


def test_gate():
    import atlas2_gate as g
    ok = [{"task": f"t{i}", "bow": 0.5, "chance": 0.5, "L_on": 5} for i in range(2)] + \
         [{"task": "t2", "bow": 0.55, "chance": 0.5, "L_on": 2}, {"task": "t3", "bow": 0.98, "chance": 0.5, "L_on": 9}]
    v = g.gate(ok)
    assert v["pass"] and v["n_clean"] == 3 and v["n_clean_deep"] == 2
    bad = [dict(r, L_on=1) for r in ok]
    assert not g.gate(bad)["pass"]
    leaky = [dict(r, bow=0.9) for r in ok]
    assert not g.gate(leaky)["pass"]
