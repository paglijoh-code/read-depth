"""Steps v3 Stage 1 runner (scripts/steps3_stage1.py): windows and orders, seed and lr rules, R, the bar verdicts, the copy
error rules and their null, the words-only floor, generation classification, and the table stage on fabricated files.
No model is loaded."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import steps3_stage1 as T  # noqa: E402
import steps_run as S  # noqa: E402

from surgery.atlas import heal  # noqa: E402

SHARED = [25, 26, 24, 22, 13, 12, 23, 14, 21, 15, 11, 20, 18, 8, 19, 17, 16, 10, 9, 7, 6, 5, 4, 3, 2, 1]


# ---------------------------------------------------------------- windows, seeds, lr

def test_each_window_removes_exactly_its_four_blocks():
    assert set(T.WINDOWS) == {"early", "mid", "late", "last", "spread"}
    for w, blocks in T.WINDOWS.items():
        order = T.window_order(SHARED, w)
        assert sorted(order) == sorted(SHARED) and order[:4] == list(blocks)
        alive = T.alive_for(SHARED, w, 28)
        assert sorted(set(range(28)) - set(alive)) == sorted(blocks) and len(alive) == 24
        assert T.removed_blocks(SHARED, w) == list(blocks)


def test_v2k12_is_the_first_twelve_of_the_shared_order():
    assert T.window_order(SHARED, T.V2K) == SHARED and T.removed_blocks(SHARED, T.V2K) == SHARED[:12]
    assert len(T.alive_for(SHARED, T.V2K, 28)) == 16
    assert T.alive_for(SHARED, T.NONE, 28) == list(range(28))
    with pytest.raises(ValueError):
        T.window_order([1, 2, 3], "mid")


def test_lr_plan_seed0_both_others_use_seed0_pick():
    rows = [{"lr": 1e-3, "best_lrsel": 0.9}, {"lr": 3e-4, "best_lrsel": 0.8}]
    assert T.lr_plan([], 0) == [1e-3, 3e-4]
    assert T.lr_plan(rows, 2) == [1e-3] and T.lr_plan(rows, 3) == [heal.pick_lr(rows)]
    tie = [{"lr": 1e-3, "best_lrsel": 0.9}, {"lr": 3e-4, "best_lrsel": 0.9}]
    assert T.lr_plan(tie, 1) == [3e-4]
    with pytest.raises(SystemExit):
        T.lr_plan(rows[:1], 2)


def test_seed_sets_and_usability_threshold():
    assert T.SEEDS[S.COPY] == (0, 2, 3) and T.SEEDS[S.OPS] == (0, 1, 2)
    assert T.usable(0.90) and T.usable(0.97) and not T.usable(0.8999) and not T.usable(None)
    with pytest.raises(SystemExit):
        T.seeds_of(S.COPY, 1)
    assert T.seeds_of(S.COPY, None) == (0, 2, 3) and T.seeds_of(S.OPS, 2) == (2,)


def test_arms_follow_the_d2_machinery():
    assert T.ARMS["mlp"][1:] == (2, 1) and T.ARMS["long"][1:] == (10, 4) and T.ARMS["both"][1:] == (2, 4)
    assert T.ARMS["long"][0] == ("q_proj", "v_proj") and len(T.ARMS["mlp"][0]) == 7


# ---------------------------------------------------------------- generation classes, score assembly

def test_classify_generation_stop_exact_repeat():
    c = T.classify_generation
    assert c("7", "7") == {"first_ok": True, "exact": True, "repeat": False}
    assert c("7\n", "7")["exact"] and c(" 7.", "7")["exact"]
    assert c("77", "7") == {"first_ok": True, "exact": False, "repeat": True}
    assert c("7 8", "7") == {"first_ok": True, "exact": False, "repeat": False}
    assert c("8", "7") == {"first_ok": False, "exact": False, "repeat": False}
    assert c("", "7")["first_ok"] is False


def test_assemble_score_rates_and_stop_given_correct():
    opts = [str(i) for i in range(10)]
    recs = [{"id": f"a{i}", "rung": "c1" if i < 4 else "c2", "answer": i % 3} for i in range(8)]
    preds = [0, 1, 0, 0, 1, 2, 2, 0]                                 # ok: 1,1,0,1,1,1,0,0
    full = [1, 1, 0, 0, 1, 1, 1, 1]
    texts = ["0", "11", "5", "0\n", "1", "2", "3", ""]               # first_ok: 1,1,0,1 | 1,1,0,0
    s = T.assemble_score(recs, opts, preds, full, texts)
    assert s["acc"] == pytest.approx(5 / 8) and s["top1_full"] == pytest.approx(6 / 8)
    c1 = s["cells"]["c1"]
    assert c1["n"] == 4 and c1["acc"] == pytest.approx(3 / 4) and c1["n_first_ok"] == 3
    assert c1["stop_given_correct"] == pytest.approx(2 / 3) and c1["repeat"] == pytest.approx(1 / 4)
    assert s["cells"]["c2"]["stop_given_correct"] == pytest.approx(1.0)
    assert s["items"]["a1"] == {"pred": 1, "ok": 1, "full": 1, "gen": "11"}
    json.dumps(s)


# ---------------------------------------------------------------- R

def _score(accs, chance=0.1):
    return {"acc": float(np.mean(list(accs.values()))), "cells": {c: {"acc": a, "chance": chance} for c, a in accs.items()}}


def test_r_value_is_chance_adjusted_per_cell_and_grouped():
    unc = _score({"L1_far_pad": 1.0, "L1_far_pad_keep": 0.9, "L1_far": 1.0, "L1_near": 0.9})
    sc = _score({"L1_far_pad": 0.55, "L1_far_pad_keep": 0.1, "L1_far": 0.1, "L1_near": 0.9})
    g = T.r_groups(S.COPY, sc, unc)
    assert g["by_cell"]["L1_far_pad"] == pytest.approx(0.45 / 0.9) and g["by_cell"]["L1_far_pad_keep"] == 0.0
    assert g["fetch"] == pytest.approx((0.5 + 0.0) / 2)              # only the two L1_*_pad* cells
    assert g["select"] == pytest.approx((0.0 + 1.0) / 2)
    ops_u = _score({"s1": 1.0, "s2": 0.9, "s3": 0.9, "s4": 0.1})
    ops_s = _score({"s1": 1.0, "s2": 0.5, "s3": 0.9, "s4": 0.9})
    g = T.r_groups(S.OPS, ops_s, ops_u)
    assert g["by_cell"]["s2"] == pytest.approx(0.4 / 0.8) and g["by_cell"]["s4"] is None   # uncut at chance: no R
    assert g["ops24"] == pytest.approx((0.5 + 1.0) / 2)


def test_exact_match_r_uses_exact_not_first_token():
    def sc(acc, exact):
        return {"acc": acc, "cells": {"s2": {"acc": acc, "exact": exact, "chance": 0.1}}}
    g = T.r_groups(S.OPS, sc(1.0, 0.2), sc(1.0, 1.0))
    assert g["ops24"] == 1.0
    assert T.r_groups(S.OPS, sc(1.0, 0.2), sc(1.0, 1.0), "exact")["ops24"] == pytest.approx(0.1 / 0.9)


# ---------------------------------------------------------------- verdicts

def per_window(vals):
    return {w: dict(vals.get(w, {})) for w in T.WINDOWS}


def test_w1_holds_fails_not_evaluable():
    seeds = (0, 2, 3)
    ok = {w: {s: 0.9 for s in seeds} for w in T.WINDOWS}
    assert T.verdict_w1(ok, seeds)["label"] == "holds"
    bad = {w: dict(v) for w, v in ok.items()}
    bad["mid"][2] = 0.79
    assert T.verdict_w1(bad, seeds)["label"] == "fails"
    missing = {w: dict(v) for w, v in ok.items()}
    del missing["last"][3]
    assert T.verdict_w1(missing, seeds)["label"] == "not evaluable"
    missing["mid"][0] = 0.1                        # a known failure outranks missing data
    assert T.verdict_w1(missing, seeds)["label"] == "fails"
    assert T.verdict_w1({}, seeds)["label"] == "not evaluable"
    assert T.verdict_w1({w: {s: None for s in seeds} for w in T.WINDOWS}, seeds)["label"] == "not evaluable"


def test_w2_names_the_window_and_last_cannot_satisfy_it_alone():
    seeds = (0, 1, 2)
    low, high = {s: 0.2 for s in seeds}, {s: 0.9 for s in seeds}
    v = T.verdict_w2({"mid": low, "late": high, "last": low}, seeds)
    assert v["label"] == "holds" and v["named"] == ["mid"]
    v = T.verdict_w2({"mid": low, "late": low}, seeds)
    assert v["named"] == ["mid", "late"]
    v = T.verdict_w2({"mid": high, "late": high, "last": low}, seeds)       # W-last low is not enough
    assert v["label"] == "fails" and v["named"] == [] and v["last"] == "holds" and "cannot satisfy" in v["note"]
    v = T.verdict_w2({"last": low}, seeds)                                  # mid and late missing
    assert v["label"] == "not evaluable" and v["named"] == [] and v["last"] == "holds"
    v = T.verdict_w2({"mid": {0: 0.2, 1: 0.2, 2: 0.7}, "late": {0: 0.2, 1: 0.2}}, seeds)
    assert v["per_window"] == {"mid": "fails", "late": "not evaluable"} and v["label"] == "not evaluable"
    assert T.verdict_w2({"mid": {0: 0.5, 1: 0.5, 2: 0.5}}, seeds)["per_window"]["mid"] == "holds"   # <= 0.50 inclusive


def test_wcontrol_general_damage():
    seeds = (0, 1, 2)
    assert T.verdict_wcontrol({s: 0.3 for s in seeds}, {}, seeds)["label"] == "fails"
    v = T.verdict_wcontrol({0: 0.3, 1: 0.9, 2: 0.4}, {0: 1.0}, seeds)
    assert v["label"] == "holds" and v["general_damage"] is False
    assert T.verdict_wcontrol({0: 0.3}, {}, seeds)["label"] == "not evaluable"
    assert T.verdict_wcontrol({0: 0.3, 1: 0.9}, {}, seeds)["label"] == "holds"


def test_v1_v2_v3():
    seeds = (0, 2, 3)
    v1 = T.verdict_v1({0: 0.1, 2: 0.4, 3: 0.5}, seeds)
    assert v1["label"] == "holds" and v1["reading"] is None
    v1 = T.verdict_v1({0: 0.9, 2: 0.95, 3: 0.85}, seeds)
    assert v1["label"] == "fails" and "redundant" in v1["reading"]
    assert T.verdict_v1({0: 0.1, 2: 0.4}, seeds)["label"] == "not evaluable"
    assert T.verdict_v2({0: 0.8, 2: 0.9, 3: 1.0}, seeds)["label"] == "holds"
    assert T.verdict_v2({0: 0.8, 2: 0.79, 3: 1.0}, seeds)["label"] == "fails"
    assert T.verdict_v2({}, seeds)["label"] == "not evaluable"
    s3 = (0, 1, 2)
    rep = {"mid": {s: 0.3 for s in s3}, "late": {s: 0.8 for s in s3}}
    assert T.verdict_v3(rep, ["mid"], s3)["label"] == "holds"
    assert T.verdict_v3(rep, ["mid", "late"], s3)["label"] == "fails"
    assert T.verdict_v3(rep, [], s3)["label"] == "not evaluable"
    assert T.verdict_v3({"mid": {0: 0.3}}, ["mid"], s3)["label"] == "not evaluable"


def test_b1_classes_and_label():
    seeds = (0, 1, 2)
    assert T.b1_class({0: 0.9, 1: 0.8, 2: 0.95}, seeds) == "rebuilt"
    assert T.b1_class({0: 0.1, 1: 0.5, 2: 0.2}, seeds) == "not rebuilt"
    assert T.b1_class({0: 0.9, 1: 0.2, 2: 0.9}, seeds) == "mixed"
    assert T.b1_class({0: 0.6, 1: 0.7, 2: 0.9}, seeds) == "mixed"
    assert T.b1_class({0: 0.9, 1: 0.2}, seeds) == "mixed"                  # already decided despite a missing seed
    assert T.b1_class({0: 0.9, 1: 0.95}, seeds) == "not evaluable"
    vals = {("late", a): {s: 0.1 for s in seeds} for a in T.ARMS}
    vals.update({("v2k12", a): {s: 0.05 for s in seeds} for a in T.ARMS})
    v = T.verdict_b1(vals, seeds)
    assert v["label"] == "holds" and set(v["classes"].values()) == {"not rebuilt"}
    vals[("late", "mlp")] = {0: 0.9, 1: 0.1, 2: 0.9}
    assert T.verdict_b1(vals, seeds)["label"] == "fails"
    part = {("late", "mlp"): {s: 0.9 for s in seeds}}                       # the other arms never ran
    assert T.verdict_b1(part, seeds)["label"] == "not evaluable"
    assert T.verdict_b1({}, seeds)["label"] == "not evaluable"


def test_e1_needs_all_seeds_of_a_source():
    def cell(p):
        return {rule: {"passes": p} for rule in T.RULES}
    ok = {"src": {"0": {g: cell(True) for g in T.GROUPS}, "2": {g: cell(True) for g in T.GROUPS}}}
    assert T.verdict_e1(ok)["label"] == "holds"
    one_bad = {"src": {"0": {g: cell(True) for g in T.GROUPS}, "2": {g: cell(False) for g in T.GROUPS}}}
    assert T.verdict_e1(one_bad)["label"] == "fails"
    nodata = {"src": {"0": {g: cell(None) for g in T.GROUPS}}}
    assert T.verdict_e1(nodata)["label"] == "not evaluable" and T.verdict_e1({})["label"] == "not evaluable"


# ---------------------------------------------------------------- copy errors

PROG = "\n".join(["a = 3", "b = a", "# note", "c = 7", "d = b", "e = 9", "f = e", "print(d)"])


def rec(i, cell, prompt, answer, decoy, rest=None):
    return {"id": i, "rung": cell, "prompt": prompt, "answer": answer, "meta": {"decoy_digit": decoy}}


def test_rule_digits_on_a_hand_written_program():
    d = T.rule_digits(rec("x", "L1_far", PROG, 3, 9))
    assert d == {"first_source": 3, "last_assignment": 9, "decoy_end": 9}
    pad = "x = 1\n# y = 2\nz = 4\nprint(z)\nw = 8"                       # nothing after print counts, comments are skipped
    assert T.rule_digits(rec("x", "L1_far", pad, 4, None)) == {"first_source": 1, "last_assignment": 4, "decoy_end": None}
    assert T.rule_digits(rec("x", "L1_far", "print(a)", 4, 2))["first_source"] is None
    assert T.position_group("L2_middle") == "middle" and T.position_group("L1_far_unit") == "far"
    assert T.position_group("L1_far_pad_keep") == "far" and T.position_group("s2") is None


def test_keep_for_rule_excludes_answer_and_other_rules_digits():
    d = {"first_source": 3, "last_assignment": 9, "decoy_end": 9}
    assert T.keep_for_rule("first_source", 5, d)
    assert not T.keep_for_rule("first_source", 3, d)                         # equals the right answer
    assert not T.keep_for_rule("last_assignment", 5, d)                      # equals decoy-end's digit
    assert not T.keep_for_rule("decoy_end", 5, d)
    assert not T.keep_for_rule("first_source", 5, {**d, "first_source": None})
    assert T.keep_for_rule("last_assignment", 5, {"first_source": None, "last_assignment": 9, "decoy_end": 1})


def make_error_recs(n=60):
    rng = np.random.default_rng(1)
    recs, preds = [], {}
    for i in range(n):
        first, last, decoy, ans = rng.choice(10, 4)
        prompt = f"a = {first}\nb = {last}\nprint(b)"
        cell = ["L1_far", "L2_middle", "L3_near"][i % 3]
        r = rec(f"r{i}", cell, prompt, int(ans), int(decoy))
        recs.append(r)
        preds[r["id"]] = int(first)                                           # the model always says the first source
    return recs, preds


def test_error_analysis_first_source_passes_and_overlaps_are_dropped():
    recs, preds = make_error_recs(300)
    res = T.error_analysis(recs, preds, n_shuffle=200)
    for g, cellname in (("far", "L1_far"), ("middle", "L2_middle"), ("near", "L3_near")):
        fs = res[g]["first_source"]
        assert fs["share"] == 1.0 and fs["passes"] is True and fs["null"] < 0.25 and fs["n"] > 0
        assert fs["n"] <= fs["n_wrong"]
        assert res[g]["last_assignment"]["passes"] is False
    # items whose first-source digit equals the answer are right (not wrong) or excluded: overlapping decoy/first digit dropped
    both = [rec("o1", "L1_far", "a = 4\nb = 6\nprint(b)", 1, 4), rec("o2", "L1_far", "a = 4\nb = 6\nprint(b)", 1, 5)]
    res = T.error_analysis(both, {"o1": 4, "o2": 4}, n_shuffle=10)
    assert res["far"]["first_source"]["n"] == 1 and res["far"]["decoy_end"]["n"] == 1   # o1: decoy == first source, dropped
    assert res["near"]["first_source"]["passes"] is None                      # no wrong answers in that group


def test_shuffle_null_is_deterministic_and_correct_answers_are_not_counted():
    recs, preds = make_error_recs(120)
    a, b = T.error_analysis(recs, preds, n_shuffle=100), T.error_analysis(recs, preds, n_shuffle=100)
    assert a == b
    assert T.error_analysis(recs, preds, n_shuffle=100, seed=1) != a
    right = {r["id"]: r["answer"] for r in recs}
    assert all(e["n"] == 0 for g in T.GROUPS for e in T.error_analysis(recs, right, 10)[g].values())


# ---------------------------------------------------------------- words-only floor

def test_words_only_probe_on_a_toy_set():
    def mk(i, label, cell):
        word = ["apple", "banana", "cherry"][label]
        return {"id": f"{cell}{i}", "rung": cell, "prompt": f"fruit {word} basket {i % 7}", "answer": label}
    train = [mk(i, i % 3, c) for i in range(60) for c in ("c1", "c2")]
    test = [mk(i, i % 3, "c1") for i in range(30)] + [{**mk(i, 0, "c2"), "prompt": "nothing here"} for i in range(30)]
    out = T.words_only(train, test)
    assert out["c1"]["acc"] == 1.0 and out["c1"]["chance"] == pytest.approx(1 / 3)
    assert out["c2"]["chance"] == 1.0 and out["c2"]["n_test"] == 30 and out["c2"]["acc"] < 1.0
    one_class = T.words_only([{"id": "t", "rung": "c", "prompt": "a b", "answer": 2}] * 5, [{"id": "u", "rung": "c", "prompt": "a", "answer": 2}])
    assert one_class["c"]["acc"] == 1.0


def test_half_peak_layer_and_block_mapping():
    curve = [0.1, 0.1, 0.2, 0.6, 0.9, 0.95, 0.95]
    assert T.half_peak_layer(curve, 0.1) == 3                              # threshold 0.1 + 0.5 * 0.85 = 0.525 -> 0.6 at 3
    assert T.layer_to_block(0, [0, 1, 2]) == "embed" and T.layer_to_block(3, [0, 1, 2, 5, 6]) == 2
    assert T.layer_to_block(None, [0]) is None


# ---------------------------------------------------------------- table stage on fabricated files

def fab_score(accs, chance=0.1, stop=0.0):
    cells = {c: {"n": 100, "acc": a, "chance": chance, "top1_full": a, "n_first_ok": int(100 * a),
                 "stop_given_correct": stop, "exact": 0.0, "repeat": 0.0} for c, a in accs.items()}
    mean = float(np.mean(list(accs.values())))
    return {"n": 100, "acc": mean, "top1_full": mean, "n_first_ok": 50, "stop_given_correct": stop, "exact": 0.0,
            "repeat": 0.0, "chance": chance, "cells": cells, "items": {}}


COPY_CELLS = ["L1_far", "L1_near", "L1_far_pad", "L1_far_pad_keep"]
OPS_CELLS = ["s1", "s2", "s3", "s4"]


def put(root, name, doc):
    Path(root, name).write_text(json.dumps(doc))


def fabricate(root):
    for s in (0, 2, 3):
        put(root, f"uncut_steps_copy_s{s}.json", {"usable": True, "score": fab_score({c: 1.0 for c in COPY_CELLS})})
    put(root, "uncut_steps_copy_s1.json", {"usable": False, "score": fab_score({c: 0.47 for c in COPY_CELLS})})
    for s in (0, 1, 2):
        put(root, f"uncut_steps_ops_s{s}.json", {"usable": True, "score": fab_score({c: 1.0 for c in OPS_CELLS})})
    for w in T.WINDOWS:
        for s in (0, 2, 3):
            put(root, f"ctt_steps_copy_{w}_s{s}.json", {"score": fab_score({c: 0.95 for c in COPY_CELLS})})
            put(root, f"tc_steps_copy_{w}_s{s}.json", {
                "after_cut": fab_score({c: 0.1 if w == "late" else 0.9 for c in COPY_CELLS}),
                "after_repair": fab_score({c: 0.95 for c in COPY_CELLS}), "probe": None})
        for s in (0, 1, 2):
            low = w in ("mid", "late")
            put(root, f"ctt_steps_ops_{w}_s{s}.json", {"score": fab_score({c: 0.2 if low else 0.95 for c in OPS_CELLS})})
            put(root, f"tc_steps_ops_{w}_s{s}.json", {"after_cut": fab_score({c: 0.1 for c in OPS_CELLS}),
                                                      "after_repair": fab_score({c: 0.2 for c in OPS_CELLS}), "probe": None})
    for s in (0, 2, 3):
        put(root, f"tc_steps_copy_none_s{s}.json", {"after_cut": None, "after_repair": fab_score({c: 1.0 for c in COPY_CELLS})})
    for cut in ("late", T.V2K):
        for arm in T.ARMS:
            for s in (0, 1, 2):
                put(root, f"arms_{cut}_{arm}_s{s}.json", {"steps": 300, "curve": [{"step": 100, "lrsel_accuracy": 0.2}],
                                                          "score": fab_score({c: 0.2 for c in OPS_CELLS})})
    for s in (0, 2, 3):
        put(root, f"ftonly_s{s}.json", {"score": fab_score({c: 0.95 for c in COPY_CELLS}, stop=0.9)})


def test_table_stage_on_fabricated_files(tmp_path):
    fabricate(tmp_path)
    order = tmp_path / "order.json"
    order.write_text(json.dumps({"base_ppl": 10.0, "ppl": {str(b): 10.0 + b for b in range(28)}}))
    v = T.stage_table(tmp_path, order_path=order)
    bars = v["bars"]
    assert v["seeds"] == {"steps_copy": [0, 2, 3], "steps_ops": [0, 1, 2]}
    assert bars["W1"]["label"] == "holds"
    assert bars["W2"]["label"] == "holds" and bars["W2"]["named"] == ["mid", "late"]
    assert bars["W-control"]["label"] == "holds"                          # spread spares ops
    assert bars["V1"]["label"] == "holds" and bars["V2"]["label"] == "holds"
    assert bars["V3"]["label"] == "holds"
    assert bars["B1"]["label"] == "holds" and set(bars["B1"]["classes"].values()) == {"not rebuilt"}
    assert bars["E1"]["label"] == "not evaluable"                         # errors stage not run
    md = (tmp_path / "tables.md").read_text()
    assert "EXPLORATORY" in md and "Unrepaired damage" in md and "| late |" in md and "control (+300 steps)" in md
    assert "First-token-only control" in md and "first-token only" in md and "exact-match R" in md and "whole answer" in md
    assert "(order.json" in md and json.loads((tmp_path / "verdicts.json").read_text())["bars"]["W1"]["label"] == "holds"
    # without the order file the table says what is missing
    T.stage_table(tmp_path, order_path=None)
    assert "not available" in (tmp_path / "tables.md").read_text()


def test_table_stage_with_nothing_run(tmp_path):
    v = T.stage_table(tmp_path, order_path=None)
    assert {b["label"] for b in v["bars"].values()} == {"not evaluable"}


def test_errors_stage_reads_saved_items(tmp_path):
    recs, preds = make_error_recs(90)
    put(tmp_path, "uncut_steps_copy_s0.json", {"usable": True, "score": fab_score({"L1_far": 1.0})})
    put(tmp_path, "ctt_steps_copy_late_s0.json", {"score": {**fab_score({"L1_far": 0.5}),
                                                            "items": {i: {"pred": p} for i, p in preds.items()}}})
    res = T.stage_errors(tmp_path, recs=recs, n_shuffle=50)
    assert res["seeds"] == [0] and list(res["sources"]) == ["ctt:late"]
    assert res["sources"]["ctt:late"]["0"]["far"]["first_source"]["passes"] is True
    assert (tmp_path / "errors.json").exists()
    v = T.build_verdicts(T.load_results(tmp_path), {S.COPY: [0], S.OPS: []}, res)
    assert v["bars"]["E1"]["label"] == "holds"


def test_ctt600_is_budget_matched_and_kept_out_of_the_table_and_probe():
    import steps3_stage1 as M
    assert M.MATCHED_STEPS == 2 * M.STD_STEPS
    assert "ctt600" in M.STAGES
    assert set(M.CTT600_CELLS[M.S.COPY]) == {"early", "mid"}
    assert set(M.CTT600_CELLS[M.S.OPS]) == {"early", "late", "spread"}
    from pathlib import Path
    assert not Path("ctt600_steps_copy_early_s0.json").match("ctt_*.json")
