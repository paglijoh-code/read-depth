"""Atlas task generators and splits (surgery/atlas/tasks.py): no model weights; the Qwen tokenizer is cached (skip if absent)."""
import ast
import functools
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

SMALL = {"discovery": 120, "train": 100, "lrsel": 40, "selection": 60, "test": 60}
SYNTH = ["code", "fact", "hard", "arith", "syntax", "rule"]


@functools.lru_cache(maxsize=None)
def gen(task, n=300, seed=4):
    return tuple(tasks.generate(task, n, seed))


@functools.lru_cache(maxsize=None)
def small_splits(family):
    splits, info = tasks.make_splits(family, 9, SMALL)
    return splits


def tok_or_skip():
    try:
        return tasks.routing_tokenizer()
    except Exception as e:                      # not cached offline
        pytest.skip(f"Qwen tokenizer not available: {e}")


@pytest.mark.parametrize("task", sorted(set(spec.TASKS) - {"route_judgement", "route_work"} - set(tasks.STEPS_TASKS)))
def test_answers_in_range_and_schema(task):
    recs = gen(task)
    n_min = 60 if task in tasks.CHECK_BASE or task in tasks.BIN_BASE else 300    # E2 R1 check and bin tasks drop items to balance
    assert n_min <= len(recs) <= 300 and len({r["id"] for r in recs}) == len(recs)
    k = len(spec.TASKS[task].options)
    for r in recs:
        assert set(r) == {"id", "task", "prompt", "answer", "template_id", "rung", "meta"}
        assert r["task"] == task and 0 <= r["answer"] < k and r["prompt"].strip() and r["template_id"]
        json.dumps(r)
        if spec.TASKS[task].rungs:
            assert r["rung"] in spec.TASKS[task].rungs


@pytest.mark.parametrize("task", sorted(spec.TASKS))
def test_option_first_tokens_distinct(task):
    ids = spec.option_token_ids(tok_or_skip(), task)
    assert len(set(ids)) == len(spec.TASKS[task].options)


@pytest.mark.parametrize("task", ["code_shape", "code_output", "fact_topic", "fact_check", "hard_hops", "hard", "arith",
                                  "syntax_depth", "syntax_valid", "rule", "sensible"])
def test_answers_balanced(task):
    recs = gen(task, 600)
    k = len(spec.TASKS[task].options)
    assert tasks.balance_dev([r["answer"] for r in recs], k) <= 0.10 + k / 600


def test_arith_shape_forced_balance():
    # a matched pair shares base items; task 9 rungs are halves, so task 13 is half "mixed" (documented in the manifest)
    cnt = Counter(r["answer"] for r in gen("arith_shape", 600))
    assert cnt[3] == 300 and all(cnt[i] == 100 for i in range(3))


@pytest.mark.parametrize("pair", spec.PAIRS)
def test_pairs_share_ids_prompts_and_templates(pair):
    a, b = tasks.generate_pair(pair[0], 120, 5)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert [r["prompt"] for r in a] == [r["prompt"] for r in b]
    assert [r["template_id"] for r in a] == [r["template_id"] for r in b]
    assert {r["task"] for r in a} == {pair[0]} and {r["task"] for r in b} == {pair[1]}


def test_generate_is_deterministic():
    for task in ("code_output", "arith", "syntax_valid", "fact_check", "hard", "rule"):
        assert tasks.generate(task, 40, 3) == tasks.generate(task, 40, 3)
    assert tasks.generate("arith", 40, 3) != tasks.generate("arith", 40, 4)


# ---------------------------------------------------------------- code

def test_code_answers_match_execution_and_shape():
    for r in gen("code_output", 600):
        out = tasks.run_py(r["prompt"])
        assert re.fullmatch(r"\d", out) and int(out) == r["answer"]
        tree = ast.parse(r["prompt"])
        loops = any(isinstance(x, (ast.For, ast.While)) for x in ast.walk(tree))
        branches = any(isinstance(x, ast.If) for x in ast.walk(tree))
        assert not (loops and branches)
    for r in gen("code_shape", 600):
        assert tasks.SHAPES[r["answer"]] == tasks.code_shape_of(r["prompt"]) == r["meta"]["shape"]


def test_code_skeletons_at_least_300():
    assert len({r["template_id"] for r in tasks.generate("code_output", 3000, 1)}) >= 300


# ---------------------------------------------------------------- facts

def test_fact_bank_size_and_falses():
    bank = tasks.fact_bank()
    per = {}
    for v in bank.values():
        per.setdefault(v["topic"], set()).add(v["s"])
        assert len(v["cands"]) >= 3 and v["true"] not in v["cands"]
    assert set(per) == set(spec.TASKS["fact_topic"].options)
    assert all(len(s) >= 120 for s in per.values()), {t: len(s) for t, s in per.items()}


def test_fact_items_half_true_template_is_subject():
    for r in gen("fact_check", 600):
        m = r["meta"]
        assert r["template_id"] == m["s"] and (r["answer"] == 0) == bool(m["truth"]) == (m["o"] == tasks.fact_bank()[f"{m['relation']}:{m['s']}"]["true"])
        assert m["s"] in r["prompt"] and m["o"] in r["prompt"]
    assert Counter(r["answer"] for r in gen("fact_check", 600)) == {0: 300, 1: 300}


# ---------------------------------------------------------------- hard

def test_hard_rungs_and_labels():
    recs = gen("hard", 300)
    assert Counter(r["rung"] for r in recs) == {"H1": 100, "H2": 100, "H3": 100}
    pair = gen("hard_hops", 300)
    for r, p in zip(recs, pair):
        assert spec.TASKS["hard"].options[r["answer"]] == r["meta"]["label"]
        assert p["answer"] == ("H1", "H2", "H3").index(r["rung"])


# ---------------------------------------------------------------- arithmetic

def _eval_left_to_right(text):
    body = re.sub(r"^(Compute: |What is |Evaluate |Calculate: |Find the value of |Work out |Simplify |Solve: |Answer: )", "", text).rstrip("?")
    return eval(body)


def test_arith_values_digits_and_rungs():
    recs = gen("arith", 600)
    for r in recs:
        assert 0 <= r["answer"] <= 9 and _eval_left_to_right(r["prompt"]) == r["answer"]
        m = r["meta"]
        ops = m["ops"]
        assert (len(ops) == 2) == (r["rung"] == "ops2") and len(set(ops)) == len(ops)
        if len(ops) == 2:
            first = tasks._APPLY[ops[0]](m["a"], m["b"])
            assert first >= 0 and tasks._APPLY[ops[1]](first, m["c"]) == r["answer"]
    assert Counter(r["rung"] for r in recs) == {"ops1": 300, "ops2": 300}
    shape = {r["id"]: r["answer"] for r in gen("arith_shape", 600)}
    for r in recs:
        assert (shape[r["id"]] == 3) == (r["rung"] == "ops2")


def test_arith_templates_at_least_200():
    assert len({r["template_id"] for r in tasks.generate("arith", 3000, 1)}) >= 200


# ---------------------------------------------------------------- syntax

def test_syntax_depth_matches_stored_answer():
    depth_recs = gen("syntax_depth", 600)
    valid_recs = gen("syntax_valid", 600)
    assert Counter(r["answer"] for r in depth_recs) == {0: 200, 1: 200, 2: 200}
    for d, v in zip(depth_recs, valid_recs):
        assert d["prompt"] == v["prompt"]
        assert tasks.bracket_depth(d["prompt"]) == d["answer"] + 1
        assert v["rung"] == f"depth{d['answer'] + 1}"
        is_json = bool(re.search(r"[\"':]", d["prompt"]))
        ok = tasks._json_valid(d["prompt"]) if is_json else tasks._balanced(d["prompt"])
        assert ok == (v["answer"] == 0)


def test_syntax_templates_at_least_200():
    assert len({r["template_id"] for r in tasks.generate("syntax_valid", 3000, 1)}) >= 200


# ---------------------------------------------------------------- splits

@pytest.mark.parametrize("family", SYNTH)
def test_splits_template_disjoint_and_sized(family):
    splits = small_splits(family)
    owner = {}
    for s, items in splits.items():
        assert len(items) == SMALL[s]
        for it in items:
            assert owner.setdefault(it["template_id"], s) == s
            for t, rec in it["records"].items():
                assert rec["template_id"] == it["template_id"]
        cap = max(1, int(spec.TEMPLATE_CAP * SMALL[s]))
        assert max(Counter(it["template_id"] for it in items).values()) <= cap
        assert len({it["template_id"] for it in items}) >= spec.MIN_TEMPLATES_PER_SPLIT


@pytest.mark.parametrize("family", ["code", "fact", "hard", "arith", "syntax"])
def test_pair_gets_same_split_for_every_base_item(family):
    splits = small_splits(family)
    a, b = tasks.FAMILY_TASKS[family]
    ids_a = {s: {it["records"][a]["id"] for it in items} for s, items in splits.items()}
    ids_b = {s: {it["records"][b]["id"] for it in items} for s, items in splits.items()}
    assert ids_a == ids_b
    for s, items in splits.items():
        assert all(it["records"][a]["prompt"] == it["records"][b]["prompt"] for it in items)


def test_splits_deterministic():
    a = tasks.make_splits("arith", 9, SMALL)[0]
    b = tasks.make_splits("arith", 9, SMALL)[0]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_build_writes_identical_bytes(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import atlas_build
    sizes = {"discovery": 60, "train": 50, "lrsel": 30, "selection": 40, "test": 40}
    m1 = atlas_build.build(tmp_path / "a", ["syntax"], sizes)
    m2 = atlas_build.build(tmp_path / "b", ["syntax"], sizes)
    assert m1["files"] == m2["files"] and len(m1["files"]) == 10
    assert (tmp_path / "a" / "syntax_valid_test.jsonl").read_bytes() == (tmp_path / "b" / "syntax_valid_test.jsonl").read_bytes()


# ---------------------------------------------------------------- routing (v5 files only)

