"""Steps v3 generators (surgery/atlas/steps3.py): determinism, balance, necessity, order dependence, templates, rules, solvers."""
import copy
import itertools
from collections import Counter, defaultdict

import pytest

from surgery.atlas import spec, steps3

DATA = {}
TASKS = steps3.TASKS3


def data(task):
    if task not in DATA:
        DATA[task] = steps3.generate(task)
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


def sample(task, k=400):
    rs = recs(task)
    return rs[::max(1, len(rs) // k)]


@pytest.mark.parametrize("task", TASKS)
def test_deterministic(task):
    assert steps3.generate(task) == data(task)


@pytest.mark.parametrize("task", TASKS)
def test_schema_and_sizes(task):
    sizes = steps3.cell_sizes(task)
    for r in recs(task):
        assert set(r) == {"id", "task", "prompt", "answer", "template_id", "rung", "meta"}
        assert r["task"] == task and 0 <= r["answer"] < 10 and r["rung"] in spec.TASKS[task].rungs
    for split in steps3.SPLIT_ORDER:
        ids = [r["id"] for r in recs(task, split)]
        assert len(set(ids)) == len(ids)
        cnt = Counter(r["rung"] for r in recs(task, split))
        assert dict(cnt) == sizes[split]


@pytest.mark.parametrize("task", TASKS)
def test_answers_balanced_per_cell_and_split(task):
    for split in steps3.SPLIT_ORDER:
        for cell in spec.TASKS[task].rungs:
            cnt = Counter(r["answer"] for r in recs(task, split) if r["rung"] == cell)
            assert len(cnt) == 10 and max(cnt.values()) == min(cnt.values()), (split, cell, cnt)


@pytest.mark.parametrize("task", TASKS)
def test_splits_disjoint_by_template(task):
    owner = {}
    for split in steps3.SPLIT_ORDER:
        for r in recs(task, split):
            assert owner.setdefault(r["template_id"], split) == split, r["template_id"]
        tpl = Counter(r["template_id"] for r in recs(task, split))
        assert len(tpl) >= spec.MIN_TEMPLATES_PER_SPLIT
        assert max(tpl.values()) / len(recs(task, split)) <= spec.TEMPLATE_CAP


@pytest.mark.parametrize("task", TASKS)
def test_solver_recomputes_every_stored_answer(task):
    for r in recs(task):
        assert steps3.SOLVERS[task](r["prompt"]) == r["answer"], r["id"]


@pytest.mark.parametrize("task", TASKS)
def test_necessity_holds_on_a_sample(task):
    for r in sample(task):
        steps3.verify(r)


def test_necessity_check_catches_a_dead_step():
    r = copy.deepcopy(next(r for r in recs("steps3_total") if r["rung"] == "m2"))
    lines = r["prompt"].split("\n")
    k = next(i for i, l in enumerate(lines) if l.startswith("x = x "))
    lines[k] = "x = x + 0"          # b = 0 would make the step dead
    r["prompt"] = "\n".join(lines)
    with pytest.raises(AssertionError):
        steps3.verify(r)


def test_swap_necessity_recomputed_from_meta():
    for r in recs("steps3_swap"):
        m = r["meta"]
        sw = [(s["i"], s["j"]) for s in m["swaps"]]
        full = steps3.simulate_swaps(m["row"], sw)[m["p"]]
        assert full == m["answer_letter"] == spec.TASKS["steps3_swap"].options[r["answer"]]
        for k, s in enumerate(m["swaps"]):
            rest = steps3.simulate_swaps(m["row"], [x for kk, x in enumerate(sw) if kk != k])[m["p"]]
            assert (rest != full) == s["on_path"]
        assert sum(s["on_path"] for s in m["swaps"]) == m["m"] and len(sw) == 4
        assert len(set(m["row"])) == 10


def test_swap_fillers_do_not_change_answer():
    for r in recs("steps3_swap"):
        m = r["meta"]
        on = [(s["i"], s["j"]) for s in m["swaps"] if s["on_path"]]
        assert steps3.simulate_swaps(m["row"], on)[m["p"]] == m["answer_letter"]
        assert steps3.solve_swap(r["prompt"]) == r["answer"]


def test_swap_order_matters():
    # reversing the on-path swaps changes the answer in most m >= 2 items (never required to be all, but must happen)
    changed = tot = 0
    for r in recs("steps3_swap"):
        m = r["meta"]
        if m["m"] < 2:
            continue
        on = [(s["i"], s["j"]) for s in m["swaps"] if s["on_path"]]
        tot += 1
        changed += steps3.simulate_swaps(m["row"], on[::-1])[m["p"]] != m["answer_letter"]
    assert changed > 0.5 * tot


def test_order_dependence_for_steps3_order():
    for r in recs("steps3_order"):
        m = r["meta"]
        if m["m"] < 2:
            continue
        a, ans = m["start"], r["answer"]
        st = [(s["sign"], s["b"]) for s in m["steps"]]
        outs = set()
        for p in itertools.permutations(st):
            v = a
            for sg, b in p:
                v = steps3._apply(v, sg, b) % 10
            outs.add(v)
        assert len(outs) > 1 and ans in outs, r["id"]
        for k in range(len(st)):                       # dropping any step changes the answer
            v = a
            for kk, (sg, b) in enumerate(st):
                if kk != k:
                    v = steps3._apply(v, sg, b) % 10
            assert v != ans, r["id"]


def test_total_steps_in_range_and_nonzero():
    for r in recs("steps3_total"):
        lines, pv = steps3.parse_program(r["prompt"])
        steps3.run_lines(lines, strict=True)
        assert all(l[4] != 0 for l in lines if l[0] == "sum")
        assert [(s["sign"], s["b"]) for s in r["meta"]["steps"]] == [(l[3], l[4]) for l in lines if l[0] == "sum" and l[1] == "x"]


def test_mix_structure():
    for r in recs("steps3_mix"):
        m = r["meta"]
        lines, pv = steps3.parse_program(r["prompt"])
        assert len(lines) == steps3.MIX_L and pv == m["print_var"]
        assert len(m["hop_lines"]) == m["h"] and len(m["op_lines"]) == m["m"]
        assert len(m["decoy_hop_lines"]) == m["h"] and len(m["decoy_op_lines"]) == m["m"]
        assert m["decoy_value"] != r["answer"]
        assert all(a["alt_value"] != a["chain_value"] for a in m["hop_alts"])
        if m["h"]:
            assert max(m["op_lines"]) > max(m["hop_lines"])


@pytest.mark.parametrize("task", TASKS)
def test_rules_are_wrong_except_where_expected(task):
    for r in sample(task, 600):
        skip = steps3.expected(task, r["rung"])
        for name, fn in steps3.RULES[task].items():
            if name not in skip:
                assert fn(r) != r["answer"], (r["id"], name)


def test_expected_rules_are_the_solution():
    for task in TASKS:
        for r in recs(task, "selection"):
            for name in steps3.expected(task, r["rung"]):
                assert steps3.RULES[task][name](r) == r["answer"], (r["id"], name)


# ---------------------------------------------------------------- rule functions on hand-written examples
def test_swap_rules_hand_example():
    row = list("ABCDEFGHIJ")
    prompt = ("Slots: " + ", ".join(f"{i} {c}" for i, c in enumerate(row)) +
              "\nswap slot 0 and slot 3\nswap slot 3 and slot 5\nswap slot 7 and slot 8\nswap slot 8 and slot 9\n"
              "Which letter is in slot 5?")
    assert steps3.solve_swap(prompt) == 0                       # A
    meta = {"row": row, "p": 5, "swaps": [{"i": 0, "j": 3, "on_path": True}, {"i": 3, "j": 5, "on_path": True},
                                          {"i": 7, "j": 8, "on_path": False}, {"i": 8, "j": 9, "on_path": False}]}
    rec = {"prompt": prompt, "meta": meta, "answer": 0}
    R = steps3.RULES["steps3_swap"]
    assert R["initial_letter_p"](rec) == 5                      # F
    assert R["last_swap_only"](rec) == 3                        # D
    assert R["first_swap_only"](rec) == 5                       # F


def test_total_rules_hand_example():
    prompt = "x = 3\ny = 5\nz = 1\nx = x + 4\ny = y - 2\nx = x - 1\nprint(x)"
    assert steps3.solve_program(prompt) == 6
    rec = {"prompt": prompt, "answer": 6, "meta": {"start": 3, "steps": [{"sign": "+", "b": 4}, {"sign": "-", "b": 1}]}}
    R = steps3.RULES["steps3_total"]
    assert R["start_only"](rec) == 3 and R["last_op_on_start"](rec) == 2
    assert R["first_literal"](rec) == 3 and R["last_literal"](rec) == 1


def test_order_rules_hand_example():
    prompt = "y = 7\nx = 3\nz = 1\nx = (x * 2) % 10\ny = (y + 2) % 10\nx = (x + 5) % 10\nprint(x)"
    assert steps3.solve_program(prompt) == 1
    assert steps3.literal_digits(prompt) == [7, 3, 1, 2, 2, 5]      # the modulus 10 is not a literal
    rec = {"prompt": prompt, "answer": 1, "meta": {"start": 3, "steps": [{"sign": "*", "b": 2}, {"sign": "+", "b": 5}]}}
    R = steps3.RULES["steps3_order"]
    assert R["last_op_on_start"](rec) == 8 and R["first_literal"](rec) == 7 and R["last_literal"](rec) == 5


def test_mix_rules_hand_example():
    prompt = "aaaa = 2\nbbbb = 4\ncccc = 7\ndddd = bbbb\ndddd = dddd + 3\ncccc = cccc - 2\nprint(dddd)"
    assert steps3.solve_program(prompt) == 7
    rec = {"prompt": prompt, "answer": 7, "meta": {"print_var": "dddd", "value_before_last_hop": 4, "decoy_value": 5}}
    R = steps3.RULES["steps3_mix"]
    assert R["sum_all_operands"](rec) == 3          # 2 + 3 - 2
    assert R["own_ops_first_literal"](rec) == 5     # 2 + 3
    assert R["stop_one_hop_early"](rec) == 4 and R["decoy_value"](rec) == 5
    assert R["nearest_binding"](rec) == 7           # last constant line is cccc = 7


def test_expected_sets():
    assert steps3.expected("steps3_swap", "m1") == {"last_swap_only", "first_swap_only"}
    assert steps3.expected("steps3_swap", "m2") == set()
    assert steps3.expected("steps3_total", "m1") == {"last_op_on_start"} and steps3.expected("steps3_order", "m3") == set()
    assert steps3.expected("steps3_mix", "h0_m2") == {"stop_one_hop_early"} and steps3.expected("steps3_mix", "h2_m2") == set()


# ---------------------------------------------------------------- tokenizers (skipped when the cache is absent)
@pytest.mark.parametrize("name", ["Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct"])
@pytest.mark.parametrize("task", TASKS)
def test_option_first_tokens_distinct(task, name):
    spec.option_token_ids(tok_or_skip(name), task)


@pytest.mark.parametrize("task", TASKS)
def test_token_counts_match_across_cells_qwen(task):
    tok = tok_or_skip()
    by = defaultdict(list)
    for r in recs(task, "selection"):
        by[r["rung"]].append(len(tok(spec.build_prompt(tok, task, r["prompt"]))["input_ids"]))
    means = {c: sum(v) / len(v) for c, v in by.items()}
    grand = sum(means.values()) / len(means)
    assert (max(means.values()) - min(means.values())) / grand <= 0.02, (task, means)
