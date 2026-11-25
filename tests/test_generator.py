"""Rule-engine and generator tests (fast, no model)."""
import hashlib
import json
import random
from collections import Counter

import pytest

from surgery.task.generator import (
    FACT_KEYS, LABELS, SENSIBLE_RULES, Knobs, firing_rules, generate, has_conflict, next_action, render_prompt,
    wrong_label, write_jsonl, read_jsonl, _facts, _sample_state, _cond_holds,
)

TARGET = "src/a.py"


def mk(read=False, edited=False, tests=("not_run",), history=None, ambiguous=False,
       path_known=True, rules=()):
    """Hand-built state. History is built to be consistent with the flags unless given."""
    if history is None:
        history = []
        if read:
            history.append({"action": "READ_FILE", "arg": TARGET})
        if edited:
            history.append({"action": "EDIT_FILE", "arg": TARGET})
        for i, st in enumerate(tests):
            if st != "not_run":
                history.append({"action": "RUN_TEST", "arg": f"test_{i}", "result": st})
    return {
        "task": {"description": "Fix", "target": TARGET, "ambiguous": ambiguous, "path_known": path_known},
        "files": [{"name": TARGET, "read": read, "edited": edited}],
        "tests": [{"name": f"test_{i}", "status": st} for i, st in enumerate(tests)],
        "history": history,
        "disabled": [],
        "rules": [{"cond": [list(c) for c in cond], "action": a} for cond, a in rules],
    }


def test_each_label_reachable():
    cases = {
        "READ_FILE": mk(rules=[([("target_read", "no")], "READ_FILE")]),
        "EDIT_FILE": mk(read=True, rules=[([("target_read", "yes"), ("target_edited", "no")], "EDIT_FILE")]),
        "RUN_TEST": mk(read=True, edited=True, rules=[([("target_edited", "yes"), ("test_not_run", "yes")], "RUN_TEST")]),
        "SEARCH": mk(path_known=False, rules=[([("path_known", "no"), ("searched", "no")], "SEARCH")]),
        "ASK_USER": mk(ambiguous=True, rules=[([("task_ambiguous", "yes"), ("asked_user", "no")], "ASK_USER")]),
        "DONE": mk(read=True, edited=True, tests=("pass",), rules=[([("target_edited", "yes"), ("all_tests_pass", "yes")], "DONE")]),
    }
    assert set(cases) == set(LABELS)
    for label, st in cases.items():
        assert next_action(st) == label


def test_precedence_first_firing_rule_wins():
    both = [([("target_read", "no")], "READ_FILE"), ([("target_read", "no")], "SEARCH")]
    st = mk(rules=both)
    assert next_action(st) == "READ_FILE"
    assert wrong_label(st) == "SEARCH"
    assert has_conflict(st)
    st["rules"].reverse()
    assert next_action(st) == "SEARCH"
    assert wrong_label(st) == "READ_FILE"


def test_lower_rule_used_when_higher_does_not_fire():
    st = mk(read=True, rules=[([("target_read", "no")], "READ_FILE"), ([("target_edited", "no")], "EDIT_FILE")])
    assert [r["action"] for r in firing_rules(st)] == ["EDIT_FILE"]
    assert next_action(st) == "EDIT_FILE"
    assert not has_conflict(st)


def test_disabled_rules_never_fire():
    st = mk(rules=[([("target_read", "no")], "READ_FILE")])
    st["disabled"] = [{"cond": [["target_read", "no"]], "action": "DONE"}]
    assert next_action(st) == "READ_FILE"
    assert [r["action"] for r in firing_rules(st)] == ["READ_FILE"]


def test_agreeing_rules_use_naive_wrong_label():
    st = mk(rules=[([("target_read", "no")], "READ_FILE"), ([("file_unread", "yes")], "READ_FILE")])
    assert next_action(st) == "READ_FILE"
    assert wrong_label(st) != "READ_FILE"


def test_no_firing_rule_raises():
    with pytest.raises(ValueError):
        next_action(mk(rules=[([("target_read", "yes")], "DONE")]))


def test_derived_facts():
    st = mk(read=True, tests=("fail", "pass"), history=[
        {"action": "RUN_TEST", "arg": "test_0", "result": "fail"},
        {"action": "RUN_TEST", "arg": "test_0", "result": "fail"},
        {"action": "SEARCH", "arg": "x"}, {"action": "READ_FILE", "arg": TARGET}])
    f = _facts(st)
    assert set(f) == set(FACT_KEYS)
    assert f["repeat_failure"] == "yes" and f["searched"] == "yes" and f["test_failing"] == "yes"
    assert f["all_tests_pass"] == "no" and f["asked_user"] == "no" and f["target_edited"] == "no"


def test_determinism():
    assert generate(30, 5) == generate(30, 5)
    assert generate(30, 5) != generate(30, 6)


def test_labels_match_rule_engine_and_schema():
    recs = generate(150, 3)
    for r in recs:
        assert set(r) == {"id", "prompt", "label", "wrong_label", "knobs", "state", "template_id", "variant"}
        assert r["variant"] == "rule"
        assert r["template_id"] == "+".join(sorted({fact for rule in r["state"]["rules"]
                                                  for fact, _ in rule["cond"]}))
        assert r["label"] == next_action(r["state"])
        assert r["wrong_label"] == wrong_label(r["state"])
        assert r["wrong_label"] in LABELS and r["wrong_label"] != r["label"]
        assert r["prompt"] == render_prompt(r["state"])
        assert set(r["knobs"]) == set(vars(Knobs()))
        # history consistent with flags
        st = r["state"]
        for f in st["files"]:
            reads = [h for h in st["history"] if h["action"] == "READ_FILE" and h["arg"] == f["name"]]
            edits = [h for h in st["history"] if h["action"] == "EDIT_FILE" and h["arg"] == f["name"]]
            assert f["read"] == bool(reads) and f["edited"] == bool(edits)
        for t in st["tests"]:
            runs = [h for h in st["history"] if h["action"] == "RUN_TEST" and h["arg"] == t["name"]]
            assert (t["status"] == "not_run") == (not runs)
            if runs:
                assert runs[-1]["result"] == t["status"]
        assert json.loads(json.dumps(r)) == r


def test_class_balance_and_conflicts():
    recs = generate(500, 0)
    c = Counter(r["label"] for r in recs)
    assert set(c) == set(LABELS)
    assert max(c.values()) / 500 <= 0.35
    share = sum(has_conflict(r["state"]) for r in recs) / 500
    assert 0.08 < share < 0.35


def test_every_action_listed_once_in_active_plus_disabled():
    for r in generate(60, 2):
        st = r["state"]
        acts = [x["action"] for x in st["rules"] + st["disabled"]]
        assert sorted(acts) == sorted(LABELS)
        assert 1 <= len(st["rules"]) <= Knobs().n_rules


def test_jsonl_roundtrip(tmp_path):
    recs = generate(10, 1)
    p = tmp_path / "x.jsonl"
    write_jsonl(recs, p)
    assert read_jsonl(p) == recs


def test_rule_sample50_regression():
    recs = generate(50, 12345)
    s = json.dumps([[r['id'], r['prompt'], r['label'], r['wrong_label'], r['state']]
                    for r in recs], sort_keys=True)
    assert hashlib.sha256(s.encode()).hexdigest() == "1d8611114a48a19472fa85ffb85d5936866fb9aed8b5b3bf6a5662ca5f2b908e"


def test_rule_m1_eval500_regression():
    s = json.dumps([[r['prompt'], r['label']] for r in generate(500, 0)])
    assert hashlib.sha256(s.encode()).hexdigest() == "4d0c8b6e7d4ca59609b752d7453c2b5f674851d74a5f1434da5577d38c8232b5"


# A reviewed action table, intentionally independent of the production library.
EXPECTED_SENSIBLE = {
    "ASK_USER": [
        "task_ambiguous=yes", "repeat_failure=yes,asked_user=no",
        "task_ambiguous=yes,asked_user=no", "path_known=no,searched=yes,asked_user=no",
        "task_ambiguous=yes,target_edited=no", "task_ambiguous=yes,test_failing=yes",
        "repeat_failure=yes,task_ambiguous=yes", "task_ambiguous=yes,test_not_run=yes",
    ],
    "SEARCH": [
        "path_known=no,searched=no", "file_unread=yes,searched=no",
        "path_known=no,target_read=no", "path_known=no,task_ambiguous=no",
        "path_known=no,asked_user=yes", "path_known=no,test_failing=yes",
        "file_unread=yes,searched=no,target_edited=no", "path_known=no,test_not_run=yes",
    ],
    "READ_FILE": [
        "target_read=no", "target_read=no,path_known=yes", "target_read=no,task_ambiguous=no",
        "target_read=no,searched=yes", "target_read=no,asked_user=yes",
        "target_read=no,test_failing=yes", "target_read=no,test_not_run=yes",
        "file_unread=yes,repeat_failure=yes",
    ],
    "EDIT_FILE": [
        "test_failing=yes", "target_edited=no",
        "target_read=yes,target_edited=no,test_failing=yes", "target_read=yes,target_edited=no",
        "test_failing=yes,task_ambiguous=no", "test_failing=yes,asked_user=yes",
        "target_edited=no,path_known=yes,searched=yes", "test_failing=yes,repeat_failure=no",
    ],
    "RUN_TEST": [
        "test_not_run=yes", "target_edited=yes,tests_run=no", "target_edited=yes,test_not_run=yes",
        "target_read=yes,test_not_run=yes", "test_not_run=yes,task_ambiguous=no",
        "test_not_run=yes,asked_user=yes", "test_not_run=yes,searched=yes",
        "test_not_run=yes,repeat_failure=no",
    ],
    "DONE": [
        "all_tests_pass=yes,target_edited=yes", "all_tests_pass=yes,target_edited=yes,test_not_run=no",
        "all_tests_pass=yes,target_edited=yes,task_ambiguous=no",
        "all_tests_pass=yes,target_edited=yes,target_read=yes",
        "all_tests_pass=yes,target_edited=yes,tests_run=yes",
        "all_tests_pass=yes,target_edited=yes,test_failing=no",
        "all_tests_pass=yes,target_edited=yes,repeat_failure=no",
        "all_tests_pass=yes,target_edited=yes,path_known=yes",
        "all_tests_pass=yes,target_edited=yes,asked_user=yes",
        "all_tests_pass=yes,target_edited=yes,searched=yes",
        "all_tests_pass=yes,target_edited=yes,file_unread=no",
    ],
}


def test_sensible_library_expected_actions_and_reachable_conditions():
    expected = {tuple(tuple(term.split("=")) for term in condition.split(",")): action
                for action, conditions in EXPECTED_SENSIBLE.items() for condition in conditions}
    assert dict(SENSIBLE_RULES) == expected
    assert len(SENSIBLE_RULES) == len(expected) >= 30
    assert all(count >= 4 for count in Counter(action for _, action in SENSIBLE_RULES).values())
    for cond, _ in SENSIBLE_RULES:
        assert 1 <= len(cond) <= 3
        assert all(fact in FACT_KEYS and value in ("yes", "no") for fact, value in cond)
    # Witnesses come from the actual sampler, including its history constraints.
    rng = random.Random(43)
    unseen = set(expected)
    for _ in range(10000):
        facts = _facts(_sample_state(rng, Knobs()))
        unseen = {cond for cond in unseen if not _cond_holds(cond, facts)}
        if not unseen:
            break
    assert not unseen


def test_sensible_balance_and_valid_records():
    recs = generate(600, 0, variant="sensible")
    assert Counter(r["label"] for r in recs) == {label: 100 for label in LABELS}
    library = set(SENSIBLE_RULES)
    for r in recs:
        assert r["variant"] == "sensible"
        assert r["label"] == next_action(r["state"])
        assert r["wrong_label"] == wrong_label(r["state"]) != r["label"]
        assert r["prompt"] == render_prompt(r["state"])
        st = r["state"]
        assert r["template_id"] == "+".join(sorted({f for rule in st["rules"] for f, _ in rule["cond"]}))
        active = {(tuple(map(tuple, rule["cond"])), rule["action"]) for rule in st["rules"]}
        disabled = {(tuple(map(tuple, rule["cond"])), rule["action"]) for rule in st["disabled"]}
        assert len(active) == len(st["rules"])
        assert len(disabled) == len(st["disabled"])
        assert active <= library and disabled <= library and not active & disabled
        assert len(st["disabled"]) == len(LABELS) - len({rule["action"] for rule in st["rules"]})


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="unknown variant"):
        generate(1, 0, variant="typo")


def test_sensible_determinism_and_seed_difference():
    assert generate(12, 0, variant="sensible") == generate(12, 0, variant="sensible")
    assert generate(12, 0, variant="sensible") != generate(12, 1, variant="sensible")


def test_sensible_knobs_and_decoy_count():
    knobs = Knobs(min_rules=3, n_rules=3, n_off_extra=2)
    recs = generate(18, 4, knobs, variant="sensible")
    for r in recs:
        st = r["state"]
        assert len(st["rules"]) == 3
        assert len(st["disabled"]) == 8 - len({rule["action"] for rule in st["rules"]})
