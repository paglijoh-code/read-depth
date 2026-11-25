"""Mock coding-workflow next-action task generator.

A hidden state describes a mock coding session: a task (target file, whether the
request is ambiguous, whether the target path is known), a list of files (read /
edited flags), a list of tests (not_run / pass / fail), an action history, and a
list of active workflow rules in precedence order (highest first).

Decision procedure (`next_action`, a pure function of the state dict):
  1. Status facts (FACT_KEYS, yes/no) are derived from files, tests and history, e.g.
     target_read, test_failing, asked_user, repeat_failure (>= 2 failed runs in history).
  2. `state["rules"]` are the active rules, highest precedence first. Each is
     {"cond": [[fact, "yes"|"no"], ..], "action": label}; a condition is an AND of
     1..cond_size facts with distinct random actions in the `rule` variant; `sensible`
     draws distinct entries from SENSIBLE_RULES instead.
  3. The first active rule whose condition holds decides the label; lower-precedence
     firing rules are ignored. (The generator never emits a state where none fires.)
  4. `state["disabled"]` rules are shown under "Disabled rules (never apply these)" and
     never fire. In `rule` they use exactly the actions the active rules do not, so
     every prompt lists each action once. In `sensible` they are unused library entries.

`wrong_label` is the action of the highest-precedence firing rule whose action differs
from the label (the answer if precedence were ignored by one step); if all firing rules
agree it is the "naive workflow" action (read, edit, test, done), or the next label in
LABELS order if that equals the true label.

Every prompt has the same sections (task, files, tests, history, status, disabled rules,
rules) and vocabulary; distractor files/tests use the same words. Generation
rejection-samples on (label, conflict) so classes are balanced and about `conflict_rate`
of the examples have firing rules that disagree (examples with a single active rule
cannot conflict, so the realised share is a little lower).

CLI: uv run python -m surgery.task.generator --n 500 --seed 0 --out var/data/x.jsonl
     uv run python -m surgery.task.generator --sample50
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

BASE_LABELS = ["READ_FILE", "EDIT_FILE", "RUN_TEST", "SEARCH", "ASK_USER", "DONE"]
HARD_LABELS = BASE_LABELS + ["COMMIT", "BUILD", "FORMAT", "DEBUG", "MERGE", "REVERT"]
# Task family, chosen per process (Study B2 sets SURGERY_FAMILY=hard): every importer sees the matching labels.
FAMILY = os.environ.get("SURGERY_FAMILY", "base")
if FAMILY not in ("base", "hard"):
    raise ValueError(f"SURGERY_FAMILY must be 'base' or 'hard', got {FAMILY!r}")
LABELS = list(HARD_LABELS if FAMILY == "hard" else BASE_LABELS)
SAMPLE50_SEED = 12345


@dataclass
class Knobs:
    n_rules: int = 2            # max enabled rules per example
    min_rules: int = 1          # min enabled rules (count drawn uniformly from min..max)
    history_extra: int = 2      # max repeated reads / re-runs added to the history
    n_distractor_files: int = 2
    n_distractor_tests: int = 2
    conflict_rate: float = 0.2  # share of examples where firing rules disagree
    n_off_extra: int = 0        # extra disabled rules beyond one per remaining action
    cond_size: int = 2          # max status facts per rule condition


# ---------------------------------------------------------------- facts / rules

FACT_KEYS = ["task_ambiguous", "path_known", "asked_user", "searched", "tests_run", "target_read",
             "target_edited", "test_failing", "test_not_run", "all_tests_pass", "file_unread",
             "repeat_failure"]

# Fixed workflow policies, independent of the sampled state. Conditions are ANDs;
# precedence still decides between applicable policies that recommend different steps.
SENSIBLE_RULES = (
    ((("task_ambiguous", "yes"),), "ASK_USER"),
    ((("repeat_failure", "yes"), ("asked_user", "no")), "ASK_USER"),
    ((("task_ambiguous", "yes"), ("asked_user", "no")), "ASK_USER"),
    ((("path_known", "no"), ("searched", "yes"), ("asked_user", "no")), "ASK_USER"),
    ((("task_ambiguous", "yes"), ("target_edited", "no")), "ASK_USER"),
    ((("task_ambiguous", "yes"), ("test_failing", "yes")), "ASK_USER"),
    ((("repeat_failure", "yes"), ("task_ambiguous", "yes")), "ASK_USER"),
    ((("task_ambiguous", "yes"), ("test_not_run", "yes")), "ASK_USER"),

    ((("path_known", "no"), ("searched", "no")), "SEARCH"),
    # Locate unread context before deciding which file to inspect.
    ((("file_unread", "yes"), ("searched", "no")), "SEARCH"),
    ((("path_known", "no"), ("target_read", "no")), "SEARCH"),
    ((("path_known", "no"), ("task_ambiguous", "no")), "SEARCH"),
    ((("path_known", "no"), ("asked_user", "yes")), "SEARCH"),
    ((("path_known", "no"), ("test_failing", "yes")), "SEARCH"),
    ((("file_unread", "yes"), ("searched", "no"), ("target_edited", "no")), "SEARCH"),
    ((("path_known", "no"), ("test_not_run", "yes")), "SEARCH"),

    ((("target_read", "no"),), "READ_FILE"),
    ((("target_read", "no"), ("path_known", "yes")), "READ_FILE"),
    ((("target_read", "no"), ("task_ambiguous", "no")), "READ_FILE"),
    ((("target_read", "no"), ("searched", "yes")), "READ_FILE"),
    ((("target_read", "no"), ("asked_user", "yes")), "READ_FILE"),
    ((("target_read", "no"), ("test_failing", "yes")), "READ_FILE"),
    ((("target_read", "no"), ("test_not_run", "yes")), "READ_FILE"),
    # Re-read relevant context before attempting another fix after repeated failures.
    ((("file_unread", "yes"), ("repeat_failure", "yes")), "READ_FILE"),

    ((("test_failing", "yes"),), "EDIT_FILE"),
    ((("target_edited", "no"),), "EDIT_FILE"),
    ((("target_read", "yes"), ("target_edited", "no"), ("test_failing", "yes")), "EDIT_FILE"),
    ((("target_read", "yes"), ("target_edited", "no")), "EDIT_FILE"),
    ((("test_failing", "yes"), ("task_ambiguous", "no")), "EDIT_FILE"),
    ((("test_failing", "yes"), ("asked_user", "yes")), "EDIT_FILE"),
    ((("target_edited", "no"), ("path_known", "yes"), ("searched", "yes")), "EDIT_FILE"),
    ((("test_failing", "yes"), ("repeat_failure", "no")), "EDIT_FILE"),

    ((("test_not_run", "yes"),), "RUN_TEST"),
    ((("target_edited", "yes"), ("tests_run", "no")), "RUN_TEST"),
    ((("target_edited", "yes"), ("test_not_run", "yes")), "RUN_TEST"),
    ((("target_read", "yes"), ("test_not_run", "yes")), "RUN_TEST"),
    ((("test_not_run", "yes"), ("task_ambiguous", "no")), "RUN_TEST"),
    ((("test_not_run", "yes"), ("asked_user", "yes")), "RUN_TEST"),
    ((("test_not_run", "yes"), ("searched", "yes")), "RUN_TEST"),
    ((("test_not_run", "yes"), ("repeat_failure", "no")), "RUN_TEST"),

    ((("all_tests_pass", "yes"), ("target_edited", "yes")), "DONE"),
    ((("all_tests_pass", "yes"), ("target_edited", "yes"), ("test_not_run", "no")), "DONE"),
    ((("all_tests_pass", "yes"), ("target_edited", "yes"), ("task_ambiguous", "no")), "DONE"),
    ((("all_tests_pass", "yes"), ("target_edited", "yes"), ("target_read", "yes")), "DONE"),
    ((("all_tests_pass", "yes"), ("target_edited", "yes"), ("tests_run", "yes")), "DONE"),
    ((("all_tests_pass", "yes"), ("target_edited", "yes"), ("test_failing", "no")), "DONE"),
    ((("all_tests_pass", "yes"), ("target_edited", "yes"), ("repeat_failure", "no")), "DONE"),
    ((("all_tests_pass", "yes"), ("target_edited", "yes"), ("path_known", "yes")), "DONE"),
    ((("all_tests_pass", "yes"), ("target_edited", "yes"), ("asked_user", "yes")), "DONE"),
    ((("all_tests_pass", "yes"), ("target_edited", "yes"), ("searched", "yes")), "DONE"),
    ((("all_tests_pass", "yes"), ("target_edited", "yes"), ("file_unread", "no")), "DONE"),
)


def _yn(b: bool) -> str:
    return "yes" if b else "no"


def _facts(s: dict) -> dict:
    """Derive the yes/no status facts from files, tests and history."""
    hist = s["history"]
    acts = [h["action"] for h in hist]
    tf = next(f for f in s["files"] if f["name"] == s["task"]["target"])
    st = [t["status"] for t in s["tests"]]
    fails = sum(1 for h in hist if h["action"] == "RUN_TEST" and h.get("result") == "fail")
    f = {
        "task_ambiguous": s["task"]["ambiguous"],
        "path_known": s["task"]["path_known"],
        "asked_user": "ASK_USER" in acts,
        "searched": "SEARCH" in acts,
        "tests_run": "RUN_TEST" in acts,
        "target_read": tf["read"],
        "target_edited": tf["edited"],
        "test_failing": "fail" in st,
        "test_not_run": "not_run" in st,
        "all_tests_pass": all(x == "pass" for x in st),
        "file_unread": any(not x["read"] for x in s["files"]),
        "repeat_failure": fails >= 2,
    }
    return {k: _yn(v) for k, v in f.items()}


def _cond_holds(cond: list, facts: dict) -> bool:
    return all(facts[k] == v for k, v in cond)


def cond_text(cond: list) -> str:
    return " and ".join(f"{k}={v}" for k, v in cond)


def firing_rules(state: dict) -> list[dict]:
    """Active rules ({"cond": [[fact, "yes"|"no"], ..], "action"}) whose condition holds, in precedence order."""
    f = _facts(state)
    return [r for r in state["rules"] if _cond_holds(r["cond"], f)]


def next_action(state: dict) -> str:
    fr = firing_rules(state)
    if not fr:
        raise ValueError("no rule fires for this state")
    return fr[0]["action"]


def wrong_label(state: dict) -> str:
    label = next_action(state)
    for r in firing_rules(state):
        if r["action"] != label:
            return r["action"]
    f = _facts(state)
    if f["target_read"] == "no":
        naive = "READ_FILE"
    elif f["target_edited"] == "no" or f["test_failing"] == "yes":
        naive = "EDIT_FILE"
    elif f["test_not_run"] == "yes":
        naive = "RUN_TEST"
    else:
        naive = "DONE"
    if naive == label:
        naive = LABELS[(LABELS.index(label) + 1) % len(LABELS)]
    return naive


def has_conflict(state: dict) -> bool:
    return len({r["action"] for r in firing_rules(state)}) > 1


# ---------------------------------------------------------------- state sampling

_DIRS = ["src", "lib", "app", "core", "utils", "api"]
_STEMS = ["parser", "config", "loader", "router", "cache", "client", "server", "schema",
          "queue", "worker", "auth", "logger", "metrics", "index", "models", "session"]
_VERBS = ["Fix the bug in", "Add input validation to", "Refactor", "Update the error handling in",
          "Improve the output of", "Change the default behaviour of"]


def _sample_state(rng: random.Random, k: Knobs) -> dict:
    names = [f"{d}/{s}.py" for d in _DIRS for s in _STEMS]
    files_n = [names.pop(rng.randrange(len(names))) for _ in range(1 + k.n_distractor_files)]
    tests_n = [f"test_{s}" for s in rng.sample(_STEMS, 1 + k.n_distractor_tests)]
    target = files_n[0]
    path_known = rng.random() < 0.7
    files = []
    for i, n in enumerate(files_n):
        read = rng.random() < 0.5
        edited = read and rng.random() < (0.5 if i == 0 else 0.25)
        if i == 0 and not path_known:
            read = edited = False
        files.append({"name": n, "read": read, "edited": edited})
    rng.shuffle(files)
    tests = [{"name": n, "status": rng.choices(["not_run", "pass", "fail"], [0.35, 0.35, 0.3])[0]}
             for n in tests_n]
    searched = rng.random() < 0.3
    asked = rng.random() < 0.3
    ambiguous = rng.random() < 0.35

    # per-entity event chains (order inside a chain is kept), then random interleave
    chains: list[list[dict]] = []
    extra = rng.randint(0, k.history_extra)
    for f in files:
        ch = []
        if f["read"]:
            ch.append({"action": "READ_FILE", "arg": f["name"]})
            if extra and rng.random() < 0.5:
                extra -= 1
                ch.append({"action": "READ_FILE", "arg": f["name"]})
        if f["edited"]:
            ch.append({"action": "EDIT_FILE", "arg": f["name"]})
        if ch:
            chains.append(ch)
    for t in tests:
        if t["status"] == "not_run":
            continue
        runs = 1
        if extra and rng.random() < 0.5:
            extra -= 1
            runs = 2
        results = [rng.choice(["pass", "fail"]) for _ in range(runs - 1)] + [t["status"]]
        chains.append([{"action": "RUN_TEST", "arg": t["name"], "result": r} for r in results])
    if searched:
        chains.append([{"action": "SEARCH", "arg": rng.choice(_STEMS)}])
    if asked:
        chains.append([{"action": "ASK_USER", "arg": "clarify the request"}])
    history = []
    while chains:
        c = rng.choice(chains)
        history.append(c.pop(0))
        if not c:
            chains.remove(c)
    return {
        "task": {"description": f"{rng.choice(_VERBS)} {target}", "target": target,
                 "ambiguous": ambiguous, "path_known": path_known},
        "files": files, "tests": tests, "history": history, "rules": [], "disabled": [],
    }


def _rand_cond(rng: random.Random, k: Knobs) -> list:
    keys = rng.sample(FACT_KEYS, rng.randint(1, k.cond_size))
    return [[key, rng.choice(["yes", "no"])] for key in keys]


def _sample_rules(rng: random.Random, k: Knobs) -> tuple[list[dict], list[dict]]:
    """(active rules in precedence order, disabled rules). Active rules are random
    conjunctions of 1..cond_size status facts with distinct random actions. The
    disabled rules use exactly the remaining actions (plus n_off_extra random ones), so
    every prompt lists each action the same number of times and the action words say
    nothing about the label."""
    acts = rng.sample(LABELS, rng.randint(min(k.min_rules, k.n_rules), k.n_rules))
    off = [a for a in LABELS if a not in acts] + [rng.choice(LABELS) for _ in range(k.n_off_extra)]
    rng.shuffle(off)
    return ([{"cond": _rand_cond(rng, k), "action": a} for a in acts],
            [{"cond": _rand_cond(rng, k), "action": a} for a in off])


def _sample_sensible_rules(rng: random.Random, k: Knobs) -> tuple[list[dict], list[dict]]:
    count = rng.randint(min(k.min_rules, k.n_rules), k.n_rules)
    # random.sample supplies both distinct entries and a random precedence order.
    active = rng.sample(range(len(SENSIBLE_RULES)), count)
    actions = {SENSIBLE_RULES[i][1] for i in active}
    off_count = len(LABELS) - len(actions) + k.n_off_extra
    remaining = [i for i in range(len(SENSIBLE_RULES)) if i not in active]
    disabled = rng.sample(remaining, off_count)

    def rules(indices):
        return [{"cond": [list(pair) for pair in SENSIBLE_RULES[i][0]],
                 "action": SENSIBLE_RULES[i][1]} for i in indices]

    return rules(active), rules(disabled)


def render_prompt(s: dict) -> str:
    t = s["task"]
    out = [f"Task: {t['description']}.",
           f"Target file: {t['target']}",
           f"Task is ambiguous: {_yn(t['ambiguous'])}",
           f"Target path is known: {_yn(t['path_known'])}",
           "", "Files:"]
    for f in s["files"]:
        out.append(f"- {f['name']}: read={_yn(f['read'])} edited={_yn(f['edited'])}")
    out += ["", "Tests:"]
    for x in s["tests"]:
        out.append(f"- {x['name']}: {x['status']}")
    out += ["", "History (oldest first):"]
    if not s["history"]:
        out.append("(empty)")
    for i, h in enumerate(s["history"], 1):
        line = f"{i}. {h['action']} {h['arg']}"
        if "result" in h:
            line += f" -> {h['result']}"
        out.append(line)
    out += ["", "Status:"] + [f"{k}={v}" for k, v in _facts(s).items()]
    out += ["", "Disabled rules (never apply these):"]
    for i, r in enumerate(s["disabled"], 1):
        out.append(f"Disabled {i}: if {cond_text(r['cond'])}, then {r['action']} [end]")
    out += ["", "Rules (highest priority first):"]
    for i, r in enumerate(s["rules"], 1):
        out.append(f"Rule {i}: if {cond_text(r['cond'])}, then {r['action']} [end]")
    return "\n".join(out)


def generate(n: int, seed: int, knobs: Knobs | None = None, variant: str = "rule") -> list[dict]:
    if variant not in ("rule", "sensible"):
        raise ValueError(f"unknown variant: {variant!r}")
    sample_rules = _sample_rules if variant == "rule" else _sample_sensible_rules
    k = knobs or Knobs()
    rng = random.Random(seed)
    order: list[str] = []
    recs = []
    for i in range(n):
        if not order:
            order = LABELS[:]
            rng.shuffle(order)
        want = order.pop()
        want_conf = rng.random() < k.conflict_rate
        tries = 0
        while True:
            tries += 1
            s = _sample_state(rng, k)
            s["rules"], s["disabled"] = sample_rules(rng, k)
            if not firing_rules(s):
                continue
            if next_action(s) != want:
                continue
            if tries < 400 and has_conflict(s) != want_conf:
                continue
            break
        label = next_action(s)
        recs.append({"id": f"{seed}-{i}", "prompt": render_prompt(s), "label": label,
                     "wrong_label": wrong_label(s), "knobs": asdict(k), "state": s,
                     "variant": variant,
                     "template_id": "+".join(sorted({fact for rule in s["rules"]
                                                     for fact, _ in rule["cond"]}))})
    return recs


def write_jsonl(recs: list[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv=None) -> None:
    d = Knobs()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", choices=("rule", "sensible"), default="rule")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample50", action="store_true", help="write data/sample50.jsonl")
    ap.add_argument("--n-rules", type=int, default=d.n_rules)
    ap.add_argument("--min-rules", type=int, default=d.min_rules)
    ap.add_argument("--history-extra", type=int, default=d.history_extra)
    ap.add_argument("--distractor-files", type=int, default=d.n_distractor_files)
    ap.add_argument("--distractor-tests", type=int, default=d.n_distractor_tests)
    ap.add_argument("--conflict-rate", type=float, default=d.conflict_rate)
    ap.add_argument("--n-off-extra", type=int, default=d.n_off_extra)
    ap.add_argument("--cond-size", type=int, default=d.cond_size)
    a = ap.parse_args(argv)
    k = Knobs(n_rules=a.n_rules, min_rules=a.min_rules, history_extra=a.history_extra, n_distractor_files=a.distractor_files,
              n_distractor_tests=a.distractor_tests, conflict_rate=a.conflict_rate,
              n_off_extra=a.n_off_extra, cond_size=a.cond_size)
    if a.sample50:
        recs, out = generate(50, SAMPLE50_SEED, k, a.variant), "data/sample50.jsonl"
    else:
        recs, out = generate(a.n, a.seed, k, a.variant), a.out or "var/data/eval500.jsonl"
    write_jsonl(recs, out)
    dist = {l: sum(r["label"] == l for r in recs) for l in LABELS}
    print(f"wrote {len(recs)} records to {out}; labels {dist}")


if __name__ == "__main__":
    main()
