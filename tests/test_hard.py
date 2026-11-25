"""Study B2 task family (surgery/task/hard.py): the family switch, answers derivable from the prompt alone (an independent
reference solver that parses only the prompt text), label balance, hop depth, shortcut baselines, the b-calibrate walk
and the splits. Run with -s to see the mean prompt length per rung."""
import functools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from surgery.task.generator import FACT_KEYS, HARD_LABELS  # noqa: E402
from surgery.task.hard import ALL_KNOBS, BETWEEN, HTIERS, HardKnobs, generate_hard  # noqa: E402
from surgery.task.splits import _make_splits  # noqa: E402

N = 2000
SEED = 7


def run_py(code, env_family=None, cwd=ROOT):
    env = {k: v for k, v in os.environ.items() if k != "SURGERY_FAMILY"}
    env["PYTHONPATH"] = str(ROOT)
    if env_family is not None:
        env["SURGERY_FAMILY"] = env_family
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=cwd)


@functools.lru_cache(maxsize=None)
def items(name):
    return generate_hard(N, SEED, ALL_KNOBS[name])


# ---------------------------------------------------------------- independent reference solver (prompt text only)

def tokenize(text):
    return re.findall(r"\(|\)|\w+=\w+|\w+", text)


def parse_expr(tokens):
    """or > and > not/paren/atom, left to right; returns a nested tuple."""
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def eat():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def p_or():
        parts = [p_and()]
        while peek() == "or":
            eat()
            parts.append(p_and())
        return parts[0] if len(parts) == 1 else ("or", parts)

    def p_and():
        parts = [p_unary()]
        while peek() == "and":
            eat()
            parts.append(p_unary())
        return parts[0] if len(parts) == 1 else ("and", parts)

    def p_unary():
        t = eat()
        if t == "not":
            return ("not", p_unary())
        if t == "(":
            e = p_or()
            assert eat() == ")"
            return e
        if "=" in t:
            k, v = t.split("=")
            return ("fact", k, v)
        return ("flag", t)

    e = p_or()
    assert pos == len(tokens), tokens
    return e


def ev(e, facts, flag):
    t = e[0]
    if t == "fact":
        return facts[e[1]] == e[2]
    if t == "flag":
        return flag(e[1])
    if t == "not":
        return not ev(e[1], facts, flag)
    if t == "and":
        return all(ev(c, facts, flag) for c in e[1])
    return any(ev(c, facts, flag) for c in e[1])


def flags_in(e):
    t = e[0]
    if t == "flag":
        return {e[1]}
    if t == "not":
        return flags_in(e[1])
    if t in ("and", "or"):
        return set().union(*(flags_in(c) for c in e[1]))
    return set()


def read_prompt(prompt):
    d = {"files": [], "tests": [], "hist": [], "status": {}, "flags": {}, "rules": []}
    sec = None
    for ln in prompt.split("\n"):
        if not ln.strip():
            continue
        if ln.startswith("Target file: "):
            d["target"] = ln[len("Target file: "):]
        elif ln.startswith("Task is ambiguous: "):
            d["ambiguous"] = ln.endswith("yes")
        elif ln.startswith("Target path is known: "):
            d["path_known"] = ln.endswith("yes")
        elif ln == "Files:":
            sec = "files"
        elif ln == "Tests:":
            sec = "tests"
        elif ln.startswith("History"):
            sec = "hist"
        elif ln.startswith("Status"):
            sec = "status"
        elif ln == "Flags:":
            sec = "flags"
        elif ln.startswith("Disabled rules"):
            sec = "disabled"
        elif ln.startswith("Rules (highest priority first"):
            sec = "rules"
        elif sec == "files":
            m = re.fullmatch(r"- (\S+): read=(yes|no) edited=(yes|no)", ln)
            d["files"].append((m[1], m[2] == "yes", m[3] == "yes"))
        elif sec == "tests":
            m = re.fullmatch(r"- (\S+): (not_run|pass|fail)", ln)
            d["tests"].append(m[2])
        elif sec == "hist" and ln != "(empty)":
            m = re.fullmatch(r"\d+\. ([A-Z_]+) (.*?)(?: -> (pass|fail))?", ln)
            d["hist"].append((m[1], m[3]))
        elif sec == "status":
            m = re.fullmatch(r"(\w+)=(yes|no)", ln)
            d["status"][m[1]] = m[2]
        elif sec == "flags" and ln.endswith("[end]"):
            m = re.fullmatch(r"(\w+) holds if (.*) \[end\]", ln)
            d["flags"][m[1]] = parse_expr(tokenize(m[2]))
        elif sec == "rules":
            m = re.fullmatch(r"Rule \d+: (?:otherwise|if (.*)), then ([A-Z_]+) \[end\]", ln)
            d["rules"].append((parse_expr(tokenize(m[1])) if m[1] else None, m[2]))
    return d


def derive_facts(d):
    acts = [a for a, _ in d["hist"]]
    target = next(f for f in d["files"] if f[0] == d["target"])
    y = lambda b: "yes" if b else "no"  # noqa: E731
    f = {"task_ambiguous": d["ambiguous"], "path_known": d["path_known"], "asked_user": "ASK_USER" in acts,
         "searched": "SEARCH" in acts, "tests_run": "RUN_TEST" in acts, "target_read": target[1],
         "target_edited": target[2], "test_failing": "fail" in d["tests"], "test_not_run": "not_run" in d["tests"],
         "all_tests_pass": all(t == "pass" for t in d["tests"]), "file_unread": any(not x[1] for x in d["files"]),
         "repeat_failure": sum(1 for a, r in d["hist"] if a == "RUN_TEST" and r == "fail") >= 2}
    facts = {k: y(v) for k, v in f.items()}
    for k, v in d["status"].items():   # a listed fact must agree with what the event log says
        assert facts[k] == v, (k, v, facts[k])
    return facts


def flag_vals(d, facts, force=None, all_false=False):
    memo = {}

    def val(n):
        if n not in memo:
            memo[n] = (force[1] if force and n == force[0] else False if all_false
                       else ev(d["flags"][n], facts, val))
        return memo[n]

    for n in d["flags"]:
        val(n)
    return memo


def first_rule(d, facts, fv, skip_flag_rules=False):
    for i, (cond, action) in enumerate(d["rules"]):
        if skip_flag_rules and cond is not None and flags_in(cond):
            continue
        if cond is None or ev(cond, facts, fv.__getitem__):
            return i, action
    raise AssertionError("no rule fires")


def flag_depth(d):
    depth = {}

    def dp(n):
        if n not in depth:
            deps = flags_in(d["flags"][n])
            depth[n] = 1 + max((dp(m) for m in deps), default=0)
        return depth[n]

    for n in d["flags"]:
        dp(n)
    return depth


def ref_solve(prompt):
    """(label, deciding rule index, hops the deciding rule depends on, flip changes the answer)."""
    d = read_prompt(prompt)
    facts = derive_facts(d)
    fv = flag_vals(d, facts)
    i, label = first_rule(d, facts, fv)
    depth = flag_depth(d)
    seen, todo = set(), list(flags_in(d["rules"][i][0])) if d["rules"][i][0] else []
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            todo += flags_in(d["flags"][n])
    used = max((depth[n] for n in seen), default=0)
    flip = any(first_rule(d, facts, flag_vals(d, facts, (n, not fv[n])))[1] != label
               for n in seen if depth[n] == used and used)
    return label, i, used, flip


# ---------------------------------------------------------------- family switch

def test_family_switch_labels_and_tiers():
    code = ("from surgery.task.generator import LABELS; from surgery import capability as C; "
            "from surgery.prompt import SYSTEM; import json; print(json.dumps([LABELS, list(C.TIERS), SYSTEM]))")
    base = run_py(code)
    assert base.returncode == 0, base.stderr
    labels, tiers, system = json.loads(base.stdout)
    assert len(labels) == 6 and tiers == ["R1", "R2", "R3", "R4"] and "COMMIT" not in system
    explicit = run_py(code, "base")
    assert json.loads(explicit.stdout) == [labels, tiers, system]
    hard = run_py(code, "hard")
    assert hard.returncode == 0, hard.stderr
    hl, ht, hs = json.loads(hard.stdout)
    assert len(hl) == 12 and hl[:6] == labels and ht == ["H1", "H2", "H3", "H4", "H5"]
    assert all(l in hs for l in hl) and hl == HARD_LABELS
    bad = run_py(code, "nonsense")
    assert bad.returncode != 0 and "SURGERY_FAMILY" in bad.stderr


def test_ladder_matches_contract():
    shape = lambda k: (k.n_rules, k.n_flags, k.hops, k.hide_status, k.or_not, k.status_share)  # noqa: E731
    assert list(HTIERS) == ["H1", "H2", "H3", "H4", "H5"]
    assert [shape(HTIERS[n]) for n in HTIERS] == [
        (8, 0, 0, False, True, 1.0), (8, 2, 1, False, True, 1.0), (8, 4, 2, False, True, 1.0),
        (8, 4, 2, True, True, 1.0), (12, 6, 3, True, True, 1.0)]
    assert {u: (n, shape(k)) for u, (n, k) in BETWEEN.items()} == {
        "H1": ("H0", (6, 0, 0, False, False, 1.0)), "H2": ("H1b", (8, 1, 1, False, True, 1.0)),
        "H3": ("H2b", (8, 3, 2, False, True, 1.0)), "H4": ("H3b", (8, 4, 2, False, True, 0.5)),
        "H5": ("H4b", (10, 5, 3, True, True, 1.0))}
    for k in ALL_KNOBS.values():
        assert (k.history_extra, k.n_distractor_files, k.n_distractor_tests, k.conflict_rate, k.cond_size) == (
            5, 5, 5, 0.6, 3)


# ---------------------------------------------------------------- validity, balance, depth

@pytest.mark.parametrize("name", list(ALL_KNOBS))
def test_answer_derivable_from_prompt_alone(name):
    recs = items(name)
    k = ALL_KNOBS[name]
    assert len(recs) == N
    for r in recs:
        label, idx, used, flip = ref_solve(r["prompt"])
        assert label == r["label"], r["prompt"]
        assert used == r["hops_used"] and flip == r["flip_sensitive"] and idx + 1 == r["decided_by"]
        assert r["wrong_label"] != r["label"] and r["wrong_label"] in HARD_LABELS
    sample = recs[0]["prompt"]
    assert "Rules (highest priority first" in sample and ("Flags:" in sample) == (k.n_flags > 0)
    assert ("Status" in sample) == (not k.hide_status)
    assert sample.index("History") < sample.index("Rules")
    for r in recs[:200]:   # sections keep one order
        p = r["prompt"]
        order = [p.index(h) for h in ("Task:", "Files:", "Tests:", "History", "Disabled rules", "Rules (")]
        assert order == sorted(order)
        if k.n_flags:
            assert p.index("History") < p.index("Flags:") < p.index("Disabled rules")
    if k.status_share < 1:   # half the facts listed, the rest only derivable
        assert {len(re.findall(r"^\w+=(?:yes|no)$", r["prompt"], re.M)) for r in recs} == {6}
    assert all(r["label"] in HARD_LABELS and r["variant"] == "rule" for r in recs)
    for r in recs[:100]:   # every item has exactly the n_rules distinct actions plus disabled ones for the rest
        acts = [a for _, a in read_prompt(r["prompt"])["rules"]]
        assert len(acts) == k.n_rules == len(set(acts))
        assert len(re.findall(r"^Disabled \d+:", r["prompt"], re.M)) == len(HARD_LABELS) - k.n_rules


@pytest.mark.parametrize("name", list(ALL_KNOBS))
def test_label_balance_hops_and_catchall(name):
    recs = items(name)
    k = ALL_KNOBS[name]
    counts = {l: sum(r["label"] == l for r in recs) for l in HARD_LABELS}
    assert all(0.75 * N / 12 <= c <= 1.25 * N / 12 for c in counts.values()), counts
    assert sum(r["decided_by"] == k.n_rules for r in recs) / N <= 0.25
    print(f"\n{name}: mean prompt chars {sum(len(r['prompt']) for r in recs) / N:.0f}, labels min/max "
          f"{min(counts.values())}/{max(counts.values())}, catch-all share "
          f"{sum(r['decided_by'] == k.n_rules for r in recs) / N:.3f}", end=" ")
    if k.hops:
        assert max(r["hops_used"] for r in recs) == k.hops
        assert sum(r["hops_used"] == k.hops for r in recs) / N >= 0.70
        assert sum(r["hops_used"] == k.hops and r["flip_sensitive"] for r in recs) / N >= 0.50
        assert any(max(f["depth"] for f in r["state"]["flags"]) == k.hops for r in recs)
    else:
        assert all(r["hops_used"] == 0 for r in recs)
    assert all(r["knobs"]["n_rules"] == k.n_rules for r in recs[:10])


@pytest.mark.parametrize("name", [n for n, k in ALL_KNOBS.items() if k.n_flags])
def test_flat_shortcuts_score_low(name):
    recs = items(name)
    flat = ignore = 0
    for r in recs:
        d = read_prompt(r["prompt"])
        facts = derive_facts(d)
        flat += first_rule(d, facts, flag_vals(d, facts, all_false=True))[1] == r["label"]
        ignore += first_rule(d, facts, flag_vals(d, facts), skip_flag_rules=True)[1] == r["label"]
    print(f"\n{name}: flags-all-false {flat / N:.3f}, first flag-free rule {ignore / N:.3f}", end=" ")
    assert flat / N <= 0.40 and ignore / N <= 0.40


def test_flag_names_and_shuffled_order():
    recs = items("H3")
    assert sum(r["decided_by"] < 8 for r in recs) / N > 0.9   # decided before the catch-all
    r = recs[0]
    names = {f["name"] for f in r["state"]["flags"]}
    assert len(names) == 4 and not names & set(FACT_KEYS)
    assert len({tuple(f["name"] for f in x["state"]["flags"]) for x in recs[:50]}) > 10   # printed order is shuffled


def test_generation_is_deterministic_and_variant_checked():
    assert [r["id"] for r in generate_hard(5, 3, HTIERS["H2"])] == [f"3-{i}" for i in range(5)]
    assert generate_hard(20, 3, HTIERS["H2"]) == generate_hard(20, 3, HTIERS["H2"])
    with pytest.raises(ValueError):
        generate_hard(2, 0, HTIERS["H1"], variant="sensible")
    with pytest.raises(ValueError):
        generate_hard(2, 0, HardKnobs(n_rules=13))


# ---------------------------------------------------------------- calibration walk (stub scores)

def stub_walk(acc, **kw):
    import capability_sweep as S
    ran = []

    def run(name):
        ran.append(name)
        return {"accuracy": acc[name]}

    cached = kw.pop("cached", {})
    frozen = S.walk_hard(cached, run, **kw)
    return frozen, ran


def test_walk_freezes_first_in_band():
    assert stub_walk({"H1": 0.7}) == ("H1", ["H1"])
    assert stub_walk({"H1": 0.99, "H2": 0.97, "H3": 0.9}) == ("H3", ["H1", "H2", "H3"])   # 90 is in band
    assert stub_walk({"H1": 0.99, "H2": 0.6}) == ("H2", ["H1", "H2"])                         # 60 is in band


def test_walk_runs_between_rung_after_a_drop():
    assert stub_walk({"H1": 0.99, "H2": 0.40, "H1b": 0.75}) == ("H1b", ["H1", "H2", "H1b"])
    assert stub_walk({"H1": 0.99, "H2": 0.97, "H3": 0.99, "H4": 0.3, "H3b": 0.5}) == (
        None, ["H1", "H2", "H3", "H4", "H3b"])
    assert stub_walk({"H1": 0.95, "H2": 0.95, "H3": 0.95, "H4": 0.95, "H5": 0.2, "H4b": 0.8}) == (
        "H4b", ["H1", "H2", "H3", "H4", "H5", "H4b"])


def test_walk_h5_above_90_stops_with_none():
    frozen, ran = stub_walk({n: 0.95 for n in HTIERS})
    assert frozen is None and ran == list(HTIERS)


def test_walk_h1_below_60_runs_h0_against_old_r4():
    assert stub_walk({"H1": 0.4, "H0": 0.8}) == ("H0", ["H1", "H0"])
    assert stub_walk({"H1": 0.4, "H0": 0.95}) == (None, ["H1", "H0"])
    assert stub_walk({"H1": 0.4, "H0": 0.3}) == (None, ["H1", "H0"])


def test_walk_uses_cached_rungs():
    frozen, ran = stub_walk({"H3": 0.7}, cached={"H1": {"accuracy": 0.99}, "H2": {"accuracy": 0.95}})
    assert frozen == "H3" and ran == ["H3"]


# ---------------------------------------------------------------- sweep under the hard family

SWEEP = ("import sys; sys.path.insert(0, 'scripts'); import capability_sweep as S; ")


def test_sweep_paths_names_and_study_a_refusal(tmp_path):
    base = run_py(SWEEP + "print(S.OUT, S.CKPT, S.CAP_DATA, S.PFX, S.HARD)")
    assert base.stdout.split() == ["var/results/cap", "var/ckpt/cap", "var/data/cap", "cap-b-", "False"], base.stderr
    hard = run_py(SWEEP + "print(S.OUT, S.CKPT, S.CAP_DATA, S.PFX, S.HARD, S.TEST_LEDGER_B, "
                  "sorted(S.C.TIERS), S.C.knobs_of('H0').n_rules)", "hard")
    assert hard.stdout.split() == ["var/results/cap/hard", "var/ckpt/cap/hard", "var/data/cap/hard", "cap-b2-", "True",
                                   "var/results/cap/hard/test_ledger_b.jsonl", "['H1',", "'H2',", "'H3',", "'H4',",
                                   "'H5']", "6"], hard.stderr
    a = run_py(SWEEP + "S.main(['--stage', 'a-tune'])", "hard")
    assert a.returncode != 0 and "Study A" in a.stderr
    v = run_py(SWEEP + "S.main(['--stage', 'b-calibrate', '--variant', 'sensible'])", "hard")
    assert v.returncode != 0 and "only variant rule" in v.stderr


def test_table_runs_under_hard_family_and_says_b2(tmp_path):
    out = tmp_path / "var" / "results" / "cap" / "hard"
    out.mkdir(parents=True)
    (out / "calib_rule.json").write_text(json.dumps({"frozen": "H1b", "rungs": {"H1": {
        "accuracy": 0.99, "label_counts": {}, "mean_prompt_tokens": 800, "step_seconds": 0.5, "hops": 0}}}))
    (out / "b_test_rule.json").write_text(json.dumps({
        "baseline_accuracy": 0.5, "verdict_capacity": "out of capacity", "verdict_cut": "cutting stays free",
        "runs": {"cap-b2-k0-g0-s0": {"accuracy": 0.7, "selection_mean": 0.7, "flops": {"relative": 1.0}}}}))
    code = ("import sys; sys.path.insert(0, %r); import capability_sweep as S; S.main(['--stage', 'table'])"
            % str(ROOT / "scripts"))
    r = run_py(code, "hard", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    text = (out / "report_tables.md").read_text()
    assert "Study B2" in text and "H-B2 capacity: out of capacity. H-B2 cut: cutting stays free." in text
    assert "Study A" not in text
    assert (run_py(code, None, cwd=tmp_path).returncode == 0)   # base family on the same folder still works


# ---------------------------------------------------------------- splits

def test_splits_build_with_contract_sizes():
    sizes = {"discovery": 3000, "selection": 1500, "test": 1500}
    t0 = time.perf_counter()
    splits, manifest = _make_splits("rule", 2, sizes, knobs=HTIERS["H3"])
    secs = time.perf_counter() - t0
    print(f"\nsplits H3 3000/1500/1500: {secs:.1f}s, pool {manifest['pool_size']}, templates {manifest['template_counts']}",
          end=" ")
    assert secs < 60
    assert {n: len(r) for n, r in splits.items()} == sizes
    tmpl = {n: {r["template_id"] for r in rows} for n, rows in splits.items()}
    assert not (tmpl["discovery"] & tmpl["selection"] or tmpl["discovery"] & tmpl["test"] or tmpl["selection"] & tmpl["test"])
    assert all(len(t) >= 50 for t in tmpl.values())
    assert {r["label"] for r in splits["test"]} == set(HARD_LABELS)   # (manifest counts follow LABELS, i.e. the family)
    ids = [r["id"] for rows in splits.values() for r in rows]
    assert len(ids) == len(set(ids)) == 6000
