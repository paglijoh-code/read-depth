"""Steps study generators (surgery/atlas/steps.py): answers, twins, templates, cell counts, F4 truth table."""
import contextlib
import io
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from surgery.atlas import spec, steps

DATA = {}


def data(task):
    if task not in DATA:
        DATA[task] = steps.generate(task)
    return DATA[task]


def recs(task, split=None):
    d = data(task)
    return d[split] if split else [r for s in d.values() for r in s]


def tok_or_skip(name="Qwen/Qwen3-1.7B"):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as e:
        pytest.skip(f"tokenizer {name} not in the cache: {e}")


def run_py(code):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {})
    return buf.getvalue().strip()


@pytest.mark.parametrize("task", steps.STEPS_TASKS)
def test_deterministic(task):
    assert steps.generate(task) == data(task)


@pytest.mark.parametrize("task", steps.STEPS_TASKS)
def test_schema_and_balance(task):
    k = len(spec.TASKS[task].options)
    for r in recs(task):
        assert set(r) == {"id", "task", "prompt", "answer", "template_id", "rung", "meta"}
        assert r["task"] == task and 0 <= r["answer"] < k and r["rung"] in spec.TASKS[task].rungs
    for split in steps.SPLIT_ORDER:
        ids = [r["id"] for r in recs(task, split)]
        assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("task", steps.STEPS_TASKS)
def test_cell_counts_meet_minimums(task):
    need = {"discovery": 250, "selection": 50, "test": 100}
    for split, n in need.items():
        cnt = Counter(r["rung"] for r in recs(task, split))
        assert set(cnt) == set(spec.TASKS[task].rungs)
        assert all(v >= n for v in cnt.values()), (split, cnt)
    for split in ("train", "lrsel"):
        cnt = Counter(r["rung"] for r in recs(task, split))
        assert max(cnt.values()) - min(cnt.values()) <= (35 if task == "steps_copy" else 0)


@pytest.mark.parametrize("task", steps.STEPS_TASKS)
def test_splits_disjoint_by_template(task):
    owner = {}
    for split in steps.SPLIT_ORDER:
        for r in recs(task, split):
            assert owner.setdefault(r["template_id"], split) == split, r["template_id"]
        assert len({r["template_id"] for r in recs(task, split)}) >= spec.MIN_TEMPLATES_PER_SPLIT


def test_f1_answers_by_execution():
    for r in recs("steps_copy"):
        assert run_py(r["prompt"]) == str(r["answer"]), r["prompt"]
        assert len(r["prompt"].split("\n")) == 24 and r["prompt"].endswith(")")


def test_f1_twins():
    groups = defaultdict(dict)
    for r in recs("steps_copy"):
        groups[r["meta"]["twin_group"]][r["meta"]["twin_role"]] = r
    n_trip = 0
    for g, d in groups.items():
        base = d["base"]
        m = base["meta"]
        if m["pad"] == "A":
            assert set(d) == {"base"}
            continue
        n_trip += 1
        assert set(d) == {"base", "flip", "control"}
        assert d["flip"]["answer"] == m["decoy_digit"] != m["target_digit"]
        assert d["control"]["answer"] == base["answer"]
        for role in ("flip", "control"):
            diff = [i for i, (a, b) in enumerate(zip(base["prompt"].split("\n"), d[role]["prompt"].split("\n"))) if a != b]
            assert len(diff) == 1
            assert {x["template_id"] for x in (base, d[role])} == {base["template_id"]}
        la = base["prompt"].split("\n")
        assert [i for i, (a, b) in enumerate(zip(la, d["flip"]["prompt"].split("\n"))) if a != b] == [m["flip_link_line"] - 1]
        # same-length single-token names: only the right-hand name changed
        a, b = la[m["flip_link_line"] - 1], d["flip"]["prompt"].split("\n")[m["flip_link_line"] - 1]
        assert len(a) == len(b) and a.split(" = ")[0] == b.split(" = ")[0]
        assert len(set(re.findall(r"^(\w+) = ", base["prompt"], re.M))) == 23 - len(m["comment_lines"])
    assert n_trip > 0


def test_f1_layout_rules():
    for r in recs("steps_copy"):
        m = r["meta"]
        L, tpos = m["L"], m["tpos"]
        assert len(m["chain_lines"]) == L + 1 and m["chain_lines"][0] == {"far": 3, "middle": 11, "near": 17}[tpos]
        assert m["tpos"] in ("far", "middle", "near") and 1 <= m["Ld"] <= 4
        if m["unit"]:
            assert m["gaps_lines"] == [1] * L
        assert m["target_digit"] != m["decoy_digit"]
        lines = r["prompt"].split("\n")
        consts = sum(bool(re.fullmatch(r"\w+ = \d", x)) for x in lines)
        comments = sum(x.startswith("#") for x in lines)
        copies = sum(bool(re.fullmatch(r"\w+ = [a-z]+", x)) for x in lines)
        assert consts + copies + comments == 23


def test_f1_pad_cells_equal_token_shape():
    # a pad line replaces an assignment line with a comment of 4 (constant) or 3 (copy) words-and-hash pieces
    for r in recs("steps_copy", "selection"):
        if r["meta"]["pad"]:
            assert r["meta"]["comment_lines"]
            assert all(r["prompt"].split("\n")[l - 1].startswith("# ") for l in r["meta"]["comment_lines"])


def test_f1_twin_token_counts_equal():
    tok = tok_or_skip()
    groups = defaultdict(dict)
    for r in recs("steps_copy", "selection")[:300]:
        groups[r["meta"]["twin_group"]][r["meta"]["twin_role"]] = r
    for d in groups.values():
        n = len(tok(spec.build_prompt(tok, "steps_copy", d["base"]["prompt"]))["input_ids"])
        for role in ("flip", "control"):
            if role in d:
                assert len(tok(spec.build_prompt(tok, "steps_copy", d[role]["prompt"]))["input_ids"]) == n


def test_f1_token_meta_present_after_annotation():
    tok = tok_or_skip()
    r = json.loads(json.dumps(recs("steps_copy", "selection")[0]))
    steps.annotate_f1_tokens(tok, r)
    m = r["meta"]
    assert m["src_to_query_tok"] > m["final_to_query_tok"] > 0
    assert len(m["gaps_tok"]) == m["L"] and m["flip_after_tok"] and m["final_binding_tok"]


def test_f2_answers_and_partials():
    for r in recs("steps_ops"):
        m = r["meta"]
        assert m["s"] + m["m"] == steps.OPS_TOTAL and len(m["qsigns"]) == m["s"] == int(r["rung"][1])
        lines = r["prompt"].split("\n")
        assert len(lines) == 3 and lines[2] == f"What is the value of {m['qlabel']}?"
        assert lines[m["qline"] - 1].startswith(m["qlabel"] + " =")
        assert "gnore" not in r["prompt"]
        exprs = {l[0]: l.split(" = ")[1] for l in lines[:2]}
        assert eval(exprs[m["qlabel"]]) == r["answer"]
        for ex in exprs.values():
            toks = ex.split()
            v = int(toks[0])
            for sg, x in zip(toks[1::2], toks[2::2]):
                v = v + int(x) if sg == "+" else v - int(x)
                assert 0 <= v <= 9
            assert 0 <= int(toks[0]) <= 9
        assert {a for a, b in [(lines[0][0], lines[1][0])]} <= {"A", "B"}


def test_f2_balance_and_counterbalance():
    for cell in ("s1", "s2", "s3", "s4"):
        rs = [r for r in recs("steps_ops", "discovery") if r["rung"] == cell]
        cnt = Counter(r["answer"] for r in rs)
        assert max(cnt.values()) - min(cnt.values()) <= 2
        assert {(r["meta"]["qlabel"], r["meta"]["qline"]) for r in rs} == {("A", 1), ("A", 2), ("B", 1), ("B", 2)}


# ---- F4: recompute everything from the JSON, without steps.py's helpers
_J = json.loads(Path(steps.__file__).with_name("steps_concepts.json").read_text())
_C = _J["concepts"]


def _true(concept, feature, pol):
    return (concept in _J["features"][feature]["true"]) == pol


def _n_needed(answer, clues):
    for size in range(1, len(clues) + 1):
        for sub in itertools.combinations(clues, size):
            left = [c for c in _C if all(_true(c, f, p) for f, p in sub)]
            if left == [answer]:
                return size
    return 0


def test_f4_truth_table_size_and_concepts():
    assert 25 <= len(_J["features"]) <= 40 and len(_C) == 10
    assert tuple(_C) == spec.TASKS["steps_describe"].options


def test_f4_items():
    sample = recs("steps_describe", "selection") + recs("steps_describe", "test")[:200]
    for r in sample:
        m = r["meta"]
        concept = _C[r["answer"]]
        assert m["concept"] == concept and len(m["features"]) == 6 == len(set(m["features"]))
        clues = list(zip(m["features"], m["polarities"]))
        assert all(_true(concept, f, p) for f, p in clues)
        assert _n_needed(concept, clues) == m["n_needed"] == int(r["rung"][1])
        # texts are the stated phrasings
        assert r["prompt"].split("\n") == [_J["features"][f]["pos" if p else "neg"][v]
                                           for f, p, v in zip(m["features"], m["polarities"], m["variants"])]
        # essential clue: in every settling subset; placed last (near) or first (far)
        e = m["placed_essential_position"]
        assert e == (5 if m["pos"] == "near" else 0) and e in m["essential_positions"]
        for i in m["essential_positions"]:
            rest = [c for j, c in enumerate(clues) if j != i]
            assert not any(_n_needed(concept, rest) for _ in [0]) or _n_needed(concept, rest) == 0
        prefix = next(k for k in range(1, 7) if _n_needed(concept, clues[:k]) == 1 or
                      [c for c in _C if all(_true(c, f, p) for f, p in clues[:k])] == [concept])
        assert prefix == m["settle_prefix"]


def test_f4_concepts_balanced_and_all_cells_filled():
    for cell in spec.TASKS["steps_describe"].rungs:
        rs = [r for r in recs("steps_describe", "discovery") if r["rung"] == cell]
        cnt = Counter(r["answer"] for r in rs)
        assert len(cnt) == 10 and max(cnt.values()) == min(cnt.values()) == 25


def test_f4_phrasing_holdout():
    for r in recs("steps_describe", "train") + recs("steps_describe", "lrsel"):
        assert 2 not in r["meta"]["variants"]
    assert any(2 in r["meta"]["variants"] for r in recs("steps_describe", "test"))


def test_f4_clue_token_table_matches_qwen():
    tok = tok_or_skip()
    for p, n in list(_J["phrase_tokens_qwen"].items())[::7]:
        assert len(tok.encode(p, add_special_tokens=False)) == n


@pytest.mark.parametrize("name", ["Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct"])
@pytest.mark.parametrize("task", steps.STEPS_TASKS)
def test_option_first_tokens_distinct(task, name):
    spec.option_token_ids(tok_or_skip(name), task)


def test_token_counts_match_across_cells_qwen():
    tok = tok_or_skip()
    for task in steps.STEPS_TASKS:
        by = defaultdict(list)
        for r in recs(task, "selection"):
            by[r["rung"]].append(len(tok(spec.build_prompt(tok, task, r["prompt"]))["input_ids"]))
        means = {c: sum(v) / len(v) for c, v in by.items()}
        grand = sum(means.values()) / len(means)
        assert all(abs(m / grand - 1) <= 0.02 for m in means.values()), (task, means)


def test_f1_filler_constant_after_last_copy_and_print_line():
    for r in recs("steps_copy", "selection"):
        m = r["meta"]
        if m["pad"]:
            continue
        lines = r["prompt"].split("\n")
        consts = [i + 1 for i, x in enumerate(lines) if re.fullmatch(r"\w+ = \d", x)]
        assert max(consts) > m["chain_lines"][-1] and max(consts) not in (m["chain_lines"][0], m["decoy_src_line"])
        assert m["print_line"] == 24


def test_f2_answer_tables_balanced():
    for cell in ("s1", "s2", "s3", "s4"):
        rs = [r for r in recs("steps_ops", "discovery") if r["rung"] == cell]
        for key in (lambda r: r["meta"]["qops"][0], lambda r: r["meta"]["qops"][-1]):
            # no first/last operand equals the answer, and each operand value spreads over answers
            assert all(key(r) != r["answer"] for r in rs)
            per_val = Counter((r["answer"], key(r)) for r in rs)
            assert max(per_val.values()) <= 0.2 * len(rs) / 10 * 10 + 5
