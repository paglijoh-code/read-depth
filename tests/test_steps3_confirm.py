"""Confirmation protocol tests; synthetic data and fake training, no model loads."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import steps3_confirm as M  # noqa: E402


def cfg_at(tmp_path, smoke=False):
    cfg = M.settings(smoke)
    cfg.root, cfg.report = tmp_path / "runs", tmp_path / "report"
    cfg.twins, cfg.ckpt = tmp_path / "data" / "twins.jsonl", tmp_path / "checkpoints"
    return cfg


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def copy_records():
    return [{"id": role + str(g), "meta": {"twin_role": role, "twin_group": g}}
            for g in (2, 1) for role in ("flip", "base")]


def test_twin_same_counts_and_missing_predictions_are_loud():
    pairs = [(str(i), str(i + 4)) for i in range(4)]
    items = {"0": dict(pred=1, ok=1), "4": dict(pred=2, ok=1),
             "1": dict(pred=1, ok=1), "5": dict(pred=1, ok=0),
             "2": dict(pred=2, ok=0), "6": dict(pred=2, ok=1),
             "3": dict(pred=1, ok=0), "7": dict(pred=1, ok=0)}
    assert M.twin_same(pairs, items) == dict(n=4, same=0.75, both_right=1, first_only=1, second_only=1, neither=1)
    assert M.twin_same([], {}) == dict(n=0, same=None, both_right=0, first_only=0, second_only=0, neither=0)
    with pytest.raises(KeyError):
        M.twin_same(pairs, {})


def test_copy_pair_groups_are_complete_sorted_and_uncapped():
    records = copy_records() + [{"id": "solo", "meta": {"twin_group": 3, "twin_role": "base"}},
                                {"id": "plain", "meta": {}},
                                {"id": "control", "meta": {"twin_group": 1, "twin_role": "control"}}]
    assert M.copy_pairs(records) == [("base1", "flip1"), ("base2", "flip2")]


@pytest.mark.parametrize("cell", ["m1", "m2", "m3", "m4"])
def test_swap_twins_on_real_generator_output(cell):
    records = M.G._cell_split(M.SWAP, cell, "test", 20, 7, {})
    before = copy.deepcopy(records)
    twins, skipped = M.build_swap_twins(records)
    assert (twins, skipped) == M.build_swap_twins(records)
    assert records == before
    assert len(twins) > 0 and len(twins) + len(skipped) == len(records)
    by_id = {r["id"]: r for r in records}
    for twin in twins:
        original = by_id[twin["twin_of"]]
        assert twin["id"] == original["id"] + "__twin"
        assert twin["answer"] != original["answer"] == M.G.solve_swap(original["prompt"])
        assert M.G.solve_swap(twin["prompt"]) == twin["answer"]
        M.G._verify_swap(twin)
        a, b = original["prompt"].splitlines(), twin["prompt"].splitlines()
        assert len(a) == len(b) and a[0] == b[0] and a[-1] == b[-1]
        assert sum(x != y for x, y in zip(a, b)) == 1
        row, sw, asked = M.G.parse_swap(original["prompt"])
        row2, sw2, asked2 = M.G.parse_swap(twin["prompt"])
        assert row == row2 and asked == asked2
        assert sum(x != y for pair, pair2 in zip(sw, sw2) for x, y in zip(pair, pair2)) == 1
        assert twin["post_hoc_confirmation"] and twin["doc"] == M.DOC
    gate = M.twins_gate(records, twins, skipped)
    assert gate["answer_differs"] == 1.0 and gate["passes"]
    assert gate["built"] == len(twins) and gate["skipped"] == len(skipped)
    for twin in twins:
        assert M.shortcut_hits(twin, twin["rung"]) == []
    for name, acc in gate["rule_accuracy_non_expected"].items():
        assert acc in (None, 0.0), name
    if cell == "m1":
        assert gate["rule_accuracy_non_expected"]["last_swap_only"] is None


def test_twins_gate_rejects_shortcut_leaks_and_low_coverage():
    records = M.G._cell_split(M.SWAP, "m2", "test", 20, 7, {})
    twins, skipped = M.build_swap_twins(records)
    leak = copy.deepcopy(twins[0])
    leak["answer"] = M.G._swap_rule_initial(leak)  # a twin the initial-letter shortcut solves
    gate = M.twins_gate(records, [leak, *twins[1:]], skipped)
    assert not gate["passes"] and gate["rule_accuracy_non_expected"]["initial_letter_p"] > 0
    few = M.twins_gate(records, twins[:2], [r["id"] for r in records[2:]])
    assert few["answer_differs"] == 1.0 and not few["passes"]


def test_swap_arm_dropped_reads_uncut_swap_and_skips_in_smoke(tmp_path):
    env = SimpleNamespace(root=tmp_path, confirm=SimpleNamespace(smoke=False, seeds=(0, 2)))
    with pytest.raises(SystemExit):
        M.swap_arm_dropped(env)
    for sd, acc in ((0, 0.99), (2, 0.84)):
        M.ac.write(M.T.rpath(tmp_path, "cf_uncut", M.SWAP, f"s{sd}"), {"score": {"acc": acc}})
    assert M.swap_arm_dropped(env)
    M.ac.write(M.T.rpath(tmp_path, "cf_uncut", M.SWAP, "s2"), {"score": {"acc": 0.85}})
    assert not M.swap_arm_dropped(env)
    env.confirm.smoke = True
    assert not M.swap_arm_dropped(env)


def test_twins_gate_stage_resumes_without_model(tmp_path, monkeypatch):
    cfg = cfg_at(tmp_path)
    records = M.G._cell_split(M.SWAP, "m3", "test", 8, 4, {})
    gate = M.stage_twins(cfg, records)
    assert cfg.twins.exists() and (cfg.report / "twins_gate.json").exists()
    monkeypatch.setattr(M, "build_swap_twins", lambda _: pytest.fail("must resume"))
    assert M.stage_twins(cfg) == gate
    assert not M.twins_gate([], [], ["x"])["passes"]


@pytest.mark.parametrize("acc,same,label", [(0.60, 0.80, "stuck"), (0.85, 0.20, "learned"),
    (0.60 + 1e-10, 0.80 - 1e-10, "stuck"), (0.85 - 1e-10, 0.20 + 1e-10, "learned"),
    (0.60 + 1e-6, 0.80, "between"), (0.60, 0.80 - 1e-6, "between"),
    (0.85 - 1e-6, 0.20, "between"), (0.85, 0.20 + 1e-6, "between"),
    (1.0, 0.99, "between"), (None, 0.9, "missing"), (0.9, None, "missing")])
def test_classify_inclusive_edges(acc, same, label):
    assert M.classify(acc, same) == label


def summaries(label, n=5):
    acc, same = {"stuck": (0.5, 0.95), "learned": (0.95, 0.05), "between": (0.7, 0.5)}[label]
    return {s: M.run_summary({"score": {"acc": acc}, "twin": {"same": same}}) for s in M.SEEDS[:n]}


def synthetic():
    out = {arm: {M.COPY: summaries("learned" if arm in ("cf_C", "cf_D") else "stuck")}
           for arm in M.ARMS}
    out["cf_A"][M.SWAP], out["cf_D"][M.SWAP] = summaries("stuck"), summaries("learned")
    out["uncut"] = {M.SWAP: summaries("learned")}
    return out


def bars(results):
    return M.build_verdicts(results)["bars"]


def test_all_holds_and_k6_more_damage():
    b = bars(synthetic())
    assert all(b[f"K{i}"]["label"] == "holds" for i in range(1, 6))
    assert b["K6"]["label"] == "inconclusive (more damage)"


def test_k1_requires_full_2400_step_budget():
    r = synthetic()
    for run in r["cf_A"][M.COPY].values():
        run["steps"] = 1200
    b = bars(r)
    assert b["K1"]["label"] == "inconclusive"
    assert b["K1"]["incomplete_steps"] == list(M.SEEDS)


@pytest.mark.parametrize("arm,key", [("cf_A", "K1"), ("cf_B", "K2")])
def test_stuck_bar_holds_fails_inconclusive_and_missing(arm, key):
    r = synthetic()
    r[arm][M.COPY] = summaries("stuck", 4)
    assert bars(r)[key]["label"] == "holds" and bars(r)[key]["missing"] == [5]
    r[arm][M.COPY] = summaries("learned", 3)
    assert bars(r)[key]["label"] == "fails"
    r[arm][M.COPY] = summaries("between")
    assert bars(r)[key]["label"] == "inconclusive"
    r[arm][M.COPY] = summaries("stuck", 3)
    assert bars(r)[key]["label"] == "inconclusive" and bars(r)[key]["missing"] == [4, 5]


@pytest.mark.parametrize("arm,key", [("cf_D", "K3"), ("cf_C", "K4")])
def test_learned_bars_four_of_five_or_fail(arm, key):
    r = synthetic()
    r[arm][M.COPY] = summaries("learned", 4)
    assert bars(r)[key]["label"] == "holds"
    r[arm][M.COPY] = summaries("learned", 3)
    assert bars(r)[key]["label"] == "fails"
    assert bars(r)[key]["missing"] == [4, 5]


def test_k4_controls_interpretability_even_when_k1_or_k2_fails():
    r = synthetic()
    r["cf_C"][M.COPY] = summaries("between")
    r["cf_B"][M.COPY] = summaries("learned")
    b = bars(r)
    assert b["K1"]["label"] == b["K2"]["label"] == "not interpretable"
    assert b["K1"]["raw_label"] == "holds" and b["K2"]["raw_label"] == "fails"


def test_k5_requires_both_arms_and_usable_uncut_and_drops_any_low_seed():
    r = synthetic()
    r["cf_A"][M.SWAP] = summaries("stuck", 4)
    r["cf_D"][M.SWAP] = summaries("learned", 4)
    assert bars(r)["K5"]["label"] == "holds"
    r["cf_D"][M.SWAP] = summaries("learned", 3)
    assert bars(r)["K5"]["label"] == "fails"
    r = synthetic()
    r["uncut"][M.SWAP][5]["acc"] = 0.85
    assert bars(r)["K5"]["label"] == "holds"
    r["uncut"][M.SWAP][5]["acc"] = 0.85 - 1e-6
    assert bars(r)["K5"]["label"] == "dropped"
    del r["uncut"][M.SWAP][5]
    assert bars(r)["K5"]["label"] == "fails" and bars(r)["K5"]["missing_uncut"] == [5]


def test_k6_one_directional_reading():
    r = synthetic()
    r["cf_F"][M.COPY] = summaries("learned", 3)
    assert bars(r)["K6"]["label"] == "location-specific"
    r["cf_A"][M.COPY] = summaries("learned")
    assert bars(r)["K6"]["label"] == "inconclusive"
    r["cf_A"][M.COPY] = summaries("stuck")
    r["cf_C"][M.COPY] = summaries("stuck")
    assert bars(r)["K6"]["label"] == "inconclusive"  # K1 not interpretable
    r["cf_F"][M.COPY] = summaries("stuck", 4)
    assert bars(r)["K6"]["label"] == "inconclusive (more damage)"
    r["cf_F"][M.COPY] = summaries("between")
    assert bars(r)["K6"]["label"] == "inconclusive"


def test_lr_rules_A_F_C_and_D():
    rows = [{"lr": lr, "best_lrsel": score} for lr, score in zip(M.ARMS["cf_A"]["lrs"], (0.4, 0.5, 0.9))]
    assert M.lr_plan("cf_A", 0) == [1e-3, 3e-4, 3e-3]
    assert M.lr_plan("cf_A", 2, rows) == [3e-3]
    for seed in M.SEEDS:
        assert M.lr_plan("cf_F", seed, a_rows=rows) == [M.heal.pick_lr(rows)]
        assert M.lr_plan("cf_C", seed) == [1e-3]
    with pytest.raises(SystemExit, match="cf_A seed 0 first"):
        M.lr_plan("cf_F", 0, seed0_rows=rows)
    with pytest.raises(SystemExit, match="cf_A seed 0 first"):
        M.lr_plan("cf_A", 2, rows[:2])
    assert M.lr_plan("cf_D", 0) == list(M.ac.LRS)
    assert M.lr_plan("cf_D", 5, rows[:2]) == [M.heal.pick_lr(rows[:2])]


def test_windows_smoke_paths_and_every_schedule(monkeypatch):
    monkeypatch.setenv("STEPS3_CF_SMOKE", "1")
    cfg = M.settings()
    assert cfg.root == cfg.report == Path("var/results/steps3/confirm_smoke")
    assert cfg.ckpt == Path("var/ckpt/steps/smoke_cf") and cfg.twins.is_relative_to(cfg.root)
    assert cfg.seeds == (0, 2)
    assert M.settings(False).root == M.T.OUT
    assert M.settings(False).report == Path("var/results/steps3/confirm")
    for a in M.ARMS.values():
        assert M.schedule(a["steps"], True) == (20, 2)
        assert M.schedule(a["steps"]) == (a["steps"], a["steps"] // 100)
    assert M.schedule(300, True) == (20, 2)
    with pytest.raises(SystemExit, match="confirmation seeds"):
        M.seed_list(cfg, 4)
    for window, blocks in M.WINDOWS.items():
        order = M.window_order(range(1, 28), window)
        assert order[:4] == list(blocks) and sorted(order) == list(range(1, 28))
        with pytest.raises(ValueError, match="outside the shared order"):
            M.window_order([1, 2, 3], window)
    env = SimpleNamespace(ckpt_root=cfg.ckpt)
    assert M.T.ckpt_dir(env, "cf_A", "early") == cfg.ckpt / "steps3_cf_A_early"


def test_runs_merge_does_not_clobber_parallel_rows(tmp_path):
    path = tmp_path / "runs.json"
    write(path, {"exploratory": True, "runs": {"legacy": {"arm": "tc"}}})
    a, b = M.Runs(path), M.Runs(path)
    a.put("a", {"arm": "cf_A"})
    b.put("b", {"arm": "cf_B"})
    a.put("c", {"arm": "cf_C"})
    doc = json.loads(path.read_text())
    assert set(doc["runs"]) == {"legacy", "a", "b", "c"}
    assert "post_hoc_confirmation" not in doc  # the shared Stage 1 file header is left alone
    assert doc["runs"]["legacy"] == {"arm": "tc"}
    assert doc["runs"]["a"]["post_hoc_confirmation"]


def test_ensure_rows_sweeps_once_resumes_and_F_uses_A(tmp_path):
    cfg = cfg_at(tmp_path)
    env = SimpleNamespace(root=cfg.root, runs=M.Runs(cfg.root / "runs.json"))
    calls = []

    def train(lr):
        calls.append(lr)
        return dict(lr=lr, seed=0, task=M.COPY, best_lrsel=lr)

    extra = dict(kind="ctt")
    winner = M.ensure_rows(env, "cf_A", "early", M.COPY, 0, train, extra)
    assert calls == [1e-3, 3e-4, 3e-3] and winner["lr"] == 3e-3
    M.ensure_rows(env, "cf_A", "early", M.COPY, 0, train, extra)
    assert len(calls) == 3
    M.ensure_rows(env, "cf_F", "pre", M.COPY, 0, train, extra)
    assert calls[-1] == 3e-3 and len(calls) == 4


def test_uncut_smoke_does_not_read_real_baselines(tmp_path, monkeypatch):
    cfg = cfg_at(tmp_path, True)
    env = SimpleNamespace(root=cfg.root, confirm=cfg, runs=M.Runs(cfg.root / "runs.json"),
                          ctx=object(), ckpt_root=cfg.ckpt)
    monkeypatch.setattr(M.ac, "read_merged", lambda *a: pytest.fail("real baseline read"))
    monkeypatch.setattr(M, "v2_runs", lambda *a: pytest.fail("real v2 baseline read"))
    monkeypatch.setattr(M.T, "get_k0_row", lambda *a, **k: pytest.fail("real uncut row"))
    monkeypatch.setattr(M, "train_observed", lambda e, t, n, f: f())
    calls = []

    def fake_train(ctx, task, k, seed, lr, **kw):
        calls.append((seed, lr, kw))
        return dict(task=task, seed=seed, lr=lr, best_lrsel=lr, ckpt="smoke.pt", steps=kw["steps"])

    monkeypatch.setattr(M.C, "run_whole_heal", fake_train)
    for seed in cfg.seeds:
        M.uncut_row(env, M.COPY, seed, train=True)
    assert len(calls) == 3
    assert all(c[2]["steps"] == 20 and c[2]["n_evals"] == 2 for c in calls)
    assert all(r["steps"] == 20 for r in env.runs.rows())


def test_selection_observer_preserves_exact_selection_and_restores_function(monkeypatch):
    records = [{"id": "base", "answer": 0, "meta": {"twin_group": 1, "twin_role": "base"}},
               {"id": "flip", "answer": 1, "meta": {"twin_group": 1, "twin_role": "flip"}}]
    env = SimpleNamespace(ctx=SimpleNamespace(records=lambda *args: records))
    original = lambda *args, **kwargs: (0.125, ["unchanged"])
    monkeypatch.setattr(M.heal, "exact_match_scores", original)
    monkeypatch.setattr(M.heal, "accuracy_and_preds", lambda *args: (1.0, [0, 1]))
    with M.selection_curve(env, M.COPY, 2) as observations:
        for _ in range(3):
            assert M.heal.exact_match_scores(None, None, M.COPY) == (0.125, ["unchanged"])
    assert M.heal.exact_match_scores is original and len(observations) == 2
    assert observations[0]["selection_accuracy"] == 1.0
    assert observations[0]["selection_twin"]["same"] == 0.0
    with pytest.raises(RuntimeError), M.selection_curve(env, M.COPY, 2):
        raise RuntimeError("failed train")
    assert M.heal.exact_match_scores is original


def test_original_uncut_rows_are_reused_without_retraining(tmp_path, monkeypatch):
    cfg = cfg_at(tmp_path)
    env = SimpleNamespace(root=cfg.root, confirm=cfg, runs=M.Runs(cfg.root / "runs.json"))
    rows = {str(i): dict(task=M.COPY, k=0, seed=sd, lr=lr, best_lrsel=lr, ckpt=f"original-{sd}.pt")
            for i, (sd, lr) in enumerate(((0, 1e-3), (0, 3e-4), (2, 1e-3)))}
    monkeypatch.setattr(M, "v2_runs", lambda: rows)
    monkeypatch.setattr(M.C, "run_whole_heal", lambda *a, **k: pytest.fail("must reuse"))
    assert M.uncut_row(env, M.COPY, 0)["ckpt"] == "original-0.pt"
    assert M.uncut_row(env, M.COPY, 2)["ckpt"] == "original-2.pt"
    env.runs.put("copy3", dict(arm="uncut", task=M.COPY, seed=3, ckpt="stage1-3.pt"))
    assert M.uncut_row(env, M.COPY, 3)["ckpt"] == "stage1-3.pt"


def test_v2_reader_uses_original_shards_independent_of_ac_output(tmp_path, monkeypatch):
    root = tmp_path / "v2"
    write(root / "runs.shard0of2.json", {"runs": {"old0": {"lr": 1e-3}}})
    write(root / "runs.shard1of2.json", {"runs": {"old1": {"lr": 3e-4}}})
    monkeypatch.setattr(M.ac, "OUT", tmp_path / "confirmation")
    assert set(M.v2_runs(root)) == {"old0", "old1"}


@pytest.mark.parametrize("arm,seed,smoke", [("cf_A", 0, False), ("cf_B", 0, False),
    ("cf_C", 5, False), ("cf_F", 2, False), ("cf_D", 4, False), ("cf_D", 0, True)])
def test_arm_wiring_and_finished_file_resume(tmp_path, monkeypatch, arm, seed, smoke):
    cfg = cfg_at(tmp_path, smoke)
    env = SimpleNamespace(root=cfg.root, confirm=cfg, ckpt_root=cfg.ckpt, cfg=SimpleNamespace(n_layers=28),
                          shared=list(range(1, 28)), runs=M.Runs(cfg.root / "runs.json"),
                          ctx=SimpleNamespace(records=lambda *a: []))
    for lr in M.ARMS["cf_A"]["lrs"] if arm == "cf_F" else ():  # F reads A's picked rate; A itself must train
        env.runs.put(M.T.row_key("cf_A", "early", M.COPY, 0, lr),
                     dict(arm="cf_A", window="early", task=M.COPY, seed=0, lr=lr, best_lrsel=lr,
                          ckpt="existing-A.pt", curve=[], best_step=100, steps=2400))
    # Real A seed 0 is already trained above; its scoring should still resume.
    for lr in M.ac.LRS:
        env.runs.put(M.T.row_key("tc", "early", M.COPY, 0, lr),
                     dict(arm="tc", window="early", task=M.COPY, seed=0, lr=lr, best_lrsel=lr))
    monkeypatch.setattr(M.T, "window_ctx", lambda *a: "window-context")
    monkeypatch.setattr(M, "uncut_row", lambda *a: {"ckpt": "uncut.pt"})
    monkeypatch.setattr(M.T, "prepare_model", lambda *a: ("fake-model", []))
    monkeypatch.setattr(M.T, "build_scored", lambda *a: ("fake-model", []))
    monkeypatch.setattr(M.T, "score_all", lambda *a: {"acc": 0.9})
    monkeypatch.setattr(M.ac, "release", lambda *a: None)
    monkeypatch.setattr(M, "score_model", lambda *a: {"score": {"acc": 0.9}, "twin": {"same": 0.1}})
    monkeypatch.setattr(M, "train_observed", lambda e, t, n, f: f())
    calls = []

    def fake_ctt(ctx, task, k, sd, lr, **kw):
        calls.append(kw)
        assert kw["own"] and kw["removed"] == list(M.WINDOWS[M.ARMS[arm]["window"]])
        assert kw["arm"].modules == M.ARMS[arm]["modules"] and kw["arm"].rank == M.ARMS[arm]["rank"]
        return dict(task=task, seed=sd, lr=lr, best_lrsel=lr, ckpt="fake.pt", curve=[], best_step=100)

    def fake_tc(env_, spec_, task, sd, lr, steps, modules, rank, arm_, **kw):
        calls.append({**kw, "steps": steps})
        assert spec_["base"]["ckpt"] == "uncut.pt" and spec_["kind"] == "tc"
        assert arm_ == "cf_D" and modules == M.T.QV and rank == 10
        return dict(task=task, seed=sd, lr=lr, best_lrsel=lr, ckpt="fake.pt", curve=[], best_step=100)

    monkeypatch.setattr(M.C, "run_whole_heal", fake_ctt)
    monkeypatch.setattr(M.T, "train_built", fake_tc)
    M.stage_arm(env, arm, (M.COPY,), seed)
    expected_steps, expected_evals = M.schedule(M.ARMS[arm]["steps"], smoke)
    assert len(calls) > 0
    assert all(kw["steps"] == expected_steps and kw["n_evals"] == expected_evals for kw in calls)
    path = M.T.rpath(cfg.root, arm, M.COPY, M.ARMS[arm]["window"], f"s{seed}")
    result = json.loads(path.read_text())
    assert result["post_hoc_confirmation"] and result["twin"]["same"] == 0.1
    assert result["steps"] == expected_steps
    if arm == "cf_F":
        assert result["lr"] == 3e-3
    if arm == "cf_D" and not smoke:
        assert result["lr"] == 1e-3
    monkeypatch.setattr(M.T, "window_ctx", lambda *a: pytest.fail("must skip finished file"))
    M.stage_arm(env, arm, (M.COPY,), seed)


def test_table_legacy_D_and_uncut_plus_new_rows_and_curves(tmp_path):
    cfg = cfg_at(tmp_path)
    records = copy_records()
    items = {r["id"]: {"pred": int(r["id"].startswith("flip")), "ok": 1} for r in records}
    curve = [{"step": 100, "selection_accuracy": 0.6, "lrsel_exact": 0.4},
             {"step": 200, "selection_accuracy": 0.85, "lrsel_exact": 0.7},
             {"step": 300, "selection_accuracy": 0.9, "lrsel_exact": 0.8}]
    for seed in (0, 2, 3):
        write(M.T.rpath(cfg.root, "tc", M.COPY, "early", f"s{seed}"),
              {"after_cut": {"acc": 0.5}, "after_repair": {"acc": 0.95, "items": items}, "lr": 1e-3})
        write(M.T.rpath(cfg.root, "uncut", M.COPY, f"s{seed}"), {"score": {"acc": 0.99, "items": items}})
    for seed in (4, 5):
        write(M.T.rpath(cfg.root, "cf_D", M.COPY, "early", f"s{seed}"),
              {**M.STAMP, "score": {"acc": 0.95}, "twin": {"same": 0.05}})
    write(M.T.rpath(cfg.root, "cf_A", M.COPY, "early", "s0"),
          {**M.STAMP, "score": {"acc": 0.9}, "twin": {"same": 0.1}, "lr": 3e-3, "curve": curve, "best_step": 300})
    M.Runs(cfg.root / "runs.json").put("a", dict(arm="cf_A", window="early", task=M.COPY, seed=0,
                                                lr=3e-3, curve=curve, best_step=300, sel_accuracy=0.88))
    order = tmp_path / "order.json"
    write(order, {"ppl_drop": list(range(28))})
    verdicts = M.stage_table(cfg, order, records)
    assert verdicts["bars"]["K3"]["label"] == "holds"
    assert verdicts["results"]["cf_D"][M.COPY][0]["twin_same"] == 0.0
    assert verdicts["results"]["cf_A"][M.COPY][0]["jump_step"] == 200
    assert verdicts["damage"] == {"pre": 18, "early": 34}
    assert len(verdicts["cf_A_seed0_3e-3"]) == 1
    table = (cfg.report / "tables.md").read_text()
    assert "100: 0.6" in table and "missing" in table and "3e-3" in table
    assert json.loads((cfg.report / "verdicts.json").read_text())["post_hoc_confirmation"]
    assert M.damage_sums(None) == {"pre": None, "early": None}
    write(order, {"ppl_drop": {str(i): i for i in range(28)}})
    assert M.damage_sums(order) == {"pre": 18, "early": 34}


def test_empty_table_reports_missing_and_does_not_load_records(tmp_path, monkeypatch):
    cfg = cfg_at(tmp_path)
    monkeypatch.setattr(M.spec, "read_records", lambda *a: pytest.fail("unnecessary data read"))
    v = M.stage_table(cfg, None)
    assert v["bars"]["K4"]["label"] == "fails" and v["bars"]["K1"]["label"] == "not interpretable"
    assert v["bars"]["K1"]["missing"] == list(M.SEEDS)


def test_stage1_table_and_probe_ignore_confirmation_files(tmp_path, monkeypatch):
    before = M.T.stage_table(tmp_path, order_path=None)
    for name in ("cf_A_steps_copy_early_s0.json", "cf_D_steps_copy_early_s0.json", "cf_uncut_steps_copy_s0.json"):
        (tmp_path / name).write_text("not even valid json")
    assert M.T.stage_table(tmp_path, order_path=None) == before
    assert not list(tmp_path.glob("ctt_*.json")) and not list(tmp_path.glob("tc_*.json"))
    monkeypatch.setattr(M.T, "build_scored", lambda *a: pytest.fail("probe accessed confirmation"))
    M.T.stage_probe(SimpleNamespace(root=tmp_path), 1)


@pytest.mark.parametrize("stage,task", [("B", "swap"), ("F", "swap"), ("twins", "copy")])
def test_cli_rejects_inapplicable_tasks_before_setup(stage, task, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["steps3_confirm.py", "--stage", stage, "--task", task])
    monkeypatch.setattr(M, "setup", lambda *a: pytest.fail("must reject before setup"))
    with pytest.raises(SystemExit) as err:
        M.main()
    assert err.value.code == 2
    assert "appl" in capsys.readouterr().err


def test_cf_D_copy_legacy_seeds_are_skipped_outside_smoke(tmp_path, monkeypatch):
    cfg = cfg_at(tmp_path, False)
    env = SimpleNamespace(root=cfg.root, confirm=cfg)
    monkeypatch.setattr(M.T, "train_built", lambda *a, **k: pytest.fail("legacy seed must not retrain"))
    for sd in (0, 2, 3):
        M.stage_arm(env, "cf_D", (M.COPY,), sd)
        assert not M.T.rpath(cfg.root, "cf_D", M.COPY, "early", f"s{sd}").exists()
