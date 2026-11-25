"""M9 Phase 2 stage C1 (scripts/atlas3_cuts.py): lr winner, own-order block choice, R12/H2-E readings, the stages end to
end on a tiny random Qwen3 with a stub tokenizer, resumability, and that atlas_order.py's defaults are unchanged. No real
weights, no network."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import atlas3_cuts as C  # noqa: E402
import atlas_cuts as S  # noqa: E402
from test_atlas2b import EotTok, long_records  # noqa: E402
from test_atlas_heal import dirs, fake_row, tiny_model, toy_records  # noqa: E402,F401

from surgery.atlas import heal, spec  # noqa: E402


def lesion(drops):
    return {"task": "t", "rows": [{"block": b, "accuracy": 0.5, "drop": d} for b, d in enumerate(drops)]}


# ---------------------------------------------------------------- lr winner

def test_lr_winner_and_tie_rule_per_task_k():
    runs = {}
    for lr, lrsel in ((1e-3, 0.8), (3e-4, 0.7)):
        r = fake_row("arith", 12, 0, lr, 0.5, lrsel)
        runs[r["key"]] = r
    assert S.winner_lr(runs, "arith", 12, "per_k") == 1e-3
    runs = {}
    for lr, lrsel in ((1e-3, 0.7), (3e-4, 0.7)):
        r = fake_row("arith", 12, 0, lr, 0.5, lrsel)
        runs[r["key"]] = r
    assert S.winner_lr(runs, "arith", 12, "per_k") == 3e-4           # tie goes to 3e-4
    assert S.winner_lr(runs, "arith", 9, "per_k") is None            # nothing yet at another k
    assert S.unit_lrs(runs, "arith", 12, 0, "per_k") == [1e-3, 3e-4] and S.unit_lrs(runs, "arith", 12, 1, "per_k") == [3e-4]


# ---------------------------------------------------------------- own order

def test_own_order_picks_least_damaging_and_breaks_ties_by_index():
    drops = [0.9] * 28
    drops[27] = -1.0                       # the last block is never a candidate
    drops[0] = -1.0                        # nor block 0
    for b, d in ((5, 0.0), (3, 0.0), (20, 0.0), (9, 0.01), (8, 0.01), (2, -0.02)):
        drops[b] = d
    order = C.own_order(lesion(drops))
    assert order[:6] == [2, 3, 5, 20, 8, 9]                          # drop ascending, ties by lower block index
    assert 0 not in order and 27 not in order and len(order) == 26 and len(set(order)) == 26
    assert C.own_order(lesion([0.0] * 28))[:12] == list(range(1, 13))   # all tied: the first twelve blocks
    with pytest.raises(ValueError):
        C.own_order({"rows": lesion(drops)["rows"][:10]})


def test_own_ctx_cuts_its_own_order(monkeypatch):
    cfg = C.make_cfg(True)
    base = S.Ctx(cfg, model=tiny_model(), tok=EotTok(), order=[2, 1], device="cpu")
    own = C.OwnCtx(base, [1, 2])
    model, alive = own.cut(1)
    assert alive == [0, 2, 3] and model.config.num_hidden_layers == 3
    assert base.cut(1)[1] == [0, 1, 3]
    assert len(base.pristine.model.layers) == 4


# ---------------------------------------------------------------- readings

def test_h2e_pass_and_each_fail_branch():
    shared = {"arith": 0.1, "code_output": 0.2, "route_work": 0.9, "code_uses_first": 0.8}
    ok = C.h2e(shared, {"arith": 0.6, "code_output": 0.5, "route_work": 0.95, "code_uses_first": 0.85})
    assert ok["label"] == "supported" and ok["text"] == C.H2E_TEXT and ok["failed"] == []
    low_doing = C.h2e(shared, {"arith": 0.35, "code_output": 0.5, "route_work": 0.95, "code_uses_first": 0.85})
    assert low_doing["label"] == "not_supported" and len(low_doing["failed"]) == 1 and "arith" in low_doing["failed"][0]
    big_judging = C.h2e(shared, {"arith": 0.6, "code_output": 0.5, "route_work": 1.0, "code_uses_first": 0.85})
    assert big_judging["label"] == "not_supported" and "route_work" in big_judging["failed"][0]
    both = C.h2e(shared, {"arith": 0.2, "code_output": 0.5, "route_work": 0.9, "code_uses_first": 0.95})
    assert len(both["failed"]) == 2 and both["text"] is None
    edge = C.h2e(shared, {"arith": 0.4, "code_output": 0.5, "route_work": 1.0, "code_uses_first": 0.8})
    assert edge["gains"]["arith"]["gain"] == pytest.approx(0.3) and "arith" not in " ".join(edge["failed"])   # 0.30 passes
    assert "route_work" in " ".join(edge["failed"])                                                          # 0.10 fails
    assert C.h2e(shared, {"arith": 0.6})["label"] == "undetermined"


def test_word_tasks_follow_gate_and_reading():
    assert C.word_tasks([]) == C.WORD_BASE
    assert C.word_tasks(["bind_compare", "target_read_bare", "code_traces_first"]) == \
        C.WORD_BASE + ("code_traces_first", "bind_compare")
    r12 = {t: 0.9 for t in C.word_tasks(["bind_compare"])}
    assert C.reading_word(r12, C.word_tasks(["bind_compare"]))["label"] == "not_explained"
    assert C.reading_word({}, C.WORD_BASE)["label"] == "undetermined"


def test_r12_clipping_and_drop_rule_are_stage_bs():
    assert C.B.r12_value(0.52, 0.9, 0.5) == pytest.approx(0.05) and C.B.r12_value(0.3, 0.9, 0.5) == 0.0
    assert C.B.r12_value(1.0, 0.9, 0.5) == 1.0 and C.B.is_dropped(0.59, 0.5) and not C.B.is_dropped(0.6, 0.5)


def test_table_on_fake_numbers(dirs, monkeypatch):
    monkeypatch.setattr(C, "OUT", dirs / "c1")
    monkeypatch.setattr(C, "MAPS", dirs / "maps")
    monkeypatch.setattr(C, "BASELINES", (dirs / "b.json",))
    monkeypatch.setattr(S, "OUT", dirs / "c1")
    S.OUT.mkdir(parents=True)
    cfg = C.make_cfg(False)
    cfg.ks = (0, 12)
    tasks = ["arith_order", "syntax_first_close", "code_uses_first", "route_work", "arith", "code_output", "rule",
             "code_traces_first"]
    cfg.tasks = tuple(tasks) + ("arith_long",)
    cfg.table_tasks = tuple(tasks)
    l_on = {"arith_order": 1, "syntax_first_close": 2, "code_uses_first": 4, "route_work": 1, "arith": 5, "code_output": 6,
            "rule": 2, "code_traces_first": 7}
    a12 = {"arith": 0.6, "code_output": 0.6, "rule": 0.6}                 # doing tasks fail under the shared order
    runs = {}
    for t in cfg.tasks:
        for k, a in ((0, 0.9), (12, a12.get(t, 0.88))):
            for seed in (0, 1, 2):
                r = fake_row(t, k, seed, 3e-4, a, 0.7, chance=0.5)
                runs[r["key"]] = r
            r = fake_row(t, k, 0, 1e-3, a - 0.1, 0.6, chance=0.5)
            runs[r["key"]] = r
        if t != "arith_long":
            S.write(C.MAPS / f"map_{t}.json", {"L_on": l_on[t]})
    S.write(C.BASELINES[0], {"tasks": {t: {"bow": 0.5 + 0.01 * i, "chance": 0.5} for i, t in enumerate(tasks)}})
    S.write(C.MAPS / "gate.json", {"rows": [{"task": "code_traces_first", "clean": True},
                                            {"task": "bind_compare", "clean": False}]})
    S.write(S.OUT / "runs.shard0of1.json", {"runs": runs})
    own = {}
    for t, a in (("arith", 0.88), ("code_output", 0.8), ("route_work", 0.9), ("code_uses_first", 0.9)):
        for seed in (0, 1, 2):
            r = fake_row(t, 12, seed, 3e-4, a, 0.7, chance=0.5)
            r["removed"] = [1, 2]
            own[r["key"]] = r
    S.write(S.OUT / "ownorder.shard0of1.json", {"runs": own})
    s = C.build_summary(cfg, C.load_all(cfg), n_perm=100)
    rows = {r["task"]: r for r in s["rows"]}
    assert len(rows) == 9 and rows["arith_long"]["L_on"] is None and rows["arith"]["r12"] == pytest.approx(0.1 / 0.4)
    assert rows["route_work"]["r12"] == pytest.approx(0.38 / 0.4) and rows["arith"]["surface"]["gap"] == pytest.approx(0.04)
    e = s["verdict"]["H2-E"]
    assert e["label"] == "supported" and e["gains"]["arith"]["own"] == pytest.approx(0.95)
    assert e["gains"]["arith"]["shared"] == pytest.approx(0.25) and e["gains"]["route_work"]["gain"] == pytest.approx(0.05)
    w = s["verdict"]["word_shortcut"]
    assert list(w["tasks"]) == list(C.WORD_BASE) + ["code_traces_first"] and w["label"] == "not_explained"
    assert s["own_order"]["arith"]["removed"] == [1, 2] and "arith_long" not in s["verdict"]["H2-A"]["tasks"]
    md = C.render_tables(s)
    assert "## H2-E" in md and "## All tasks" in md


# ---------------------------------------------------------------- stages on the tiny model

def make_ctx(monkeypatch, cfg):
    recs = {}

    def read_records(task, split, root=None):
        n = {"discovery": 12, "train": 18, "lrsel": 8, "selection": 10, "test": 9}[split]
        if (task, split) not in recs:
            recs[(task, split)] = long_records(task, n) if task == "arith_long" else toy_records(task, n)
        return recs[(task, split)]

    monkeypatch.setattr(spec, "read_records", read_records)
    return S.Ctx(cfg, model=tiny_model(), tok=EotTok(), order=[2, 1], device="cpu")


def tiny_cfg():
    cfg = C.make_cfg(True)
    cfg.tasks, cfg.table_tasks, cfg.own_tasks, cfg.ks, cfg.kcut, cfg.steps, cfg.n = (
        ("code_shape", "arith", "arith_long"), ("code_shape", "arith"), ("code_shape", "arith"), (0, 2), 2, 4, None)
    return cfg


def test_stages_end_to_end_on_tiny_model(dirs, monkeypatch):
    monkeypatch.setattr(C, "OUT", dirs / "c1")
    monkeypatch.setattr(C, "MAPS", dirs / "maps")
    monkeypatch.setattr(C, "BASELINES", ())
    monkeypatch.setattr(S, "OUT", dirs / "c1")
    S.OUT.mkdir(parents=True)
    cfg = tiny_cfg()
    ctx = make_ctx(monkeypatch, cfg)
    for t in cfg.own_tasks:
        S.write(C.MAPS / f"lesion_{t}.json", lesion([0.0, 0.5, 0.1, 0.3]))      # candidates 1 and 2: block 2 first
    # seed 0 over two shards, then seeds 1 and 2
    assert C.stage_sweep(ctx, 0, (0, 2)) == 0 and C.stage_sweep(ctx, 0, (1, 2)) == 0
    runs = S.read_merged("runs", "_smoke", "runs")
    assert len(runs) == 3 * 2 * 2                                  # 3 tasks x 2 k x 2 lrs
    r = runs[S.run_key("arith", 2, 0, 3e-4)]
    assert r["n_alive"] == 2 and r["n"] == 10 and 0 <= r["sel_accuracy"] <= 1 and len(r["curve"]) == 4
    assert "lrsel_exact" in r["curve"][0] and Path(r["ckpt"]).exists() and r["max_new_tokens"] == 1 + C.MNT_MARGIN
    assert runs[S.run_key("arith_long", 2, 0, 3e-4)]["max_new_tokens"] == 2 + C.MNT_MARGIN       # values 7, 17, 27
    before = {k: v["ts"] for k, v in runs.items()}
    assert C.stage_sweep(ctx, 0, (0, 2)) == 0                      # resumed: nothing re-run
    assert {k: v["ts"] for k, v in S.read_merged("runs", "_smoke", "runs").items()} == before
    for seed in (1, 2):
        assert C.stage_sweep(ctx, seed, (0, 2)) == 0 and C.stage_sweep(ctx, seed, (1, 2)) == 0
    runs = S.read_merged("runs", "_smoke", "runs")
    assert len(runs) == 3 * 2 * 4
    # own order: block 2 is the least damaging, so k=2 cuts blocks 2 and 1; lr is the sweep winner
    assert C.stage_ownorder(ctx, (0, 2)) == 0 and C.stage_ownorder(ctx, (1, 2)) == 0
    own = S.read_merged("ownorder", "_smoke", "runs")
    assert len(own) == 2 * 3
    for row in own.values():
        assert row["removed"] == [2, 1] and row["n_alive"] == 2 and "own_" in row["ckpt"] and Path(row["ckpt"]).exists()
        assert row["lr"] == S.winner_lr(runs, row["task"], 2, "per_k")
    before = {k: v["ts"] for k, v in own.items()}
    assert C.stage_ownorder(ctx, (0, 2)) == 0
    assert {k: v["ts"] for k, v in S.read_merged("ownorder", "_smoke", "runs").items()} == before
    # test split: each (task, k) winner (3 seeds) and every own-order model; no ledger in smoke
    assert C.stage_test(ctx) == 0
    tr = json.loads((S.OUT / "test_results_smoke.json").read_text())["results"]
    assert len(tr) == 3 * 2 * 3 + 2 * 3 and all(r["n"] == 9 for r in tr.values())
    assert sum("ownorder_k2" in r["names"] for r in tr.values()) == 6 and not (S.OUT / "test_ledger.jsonl").exists()
    n_tr = len(tr)
    C.stage_test(ctx)
    assert len(json.loads((S.OUT / "test_results_smoke.json").read_text())["results"]) == n_tr
    # table
    assert C.stage_table(cfg) == 0
    s = json.loads((S.OUT / "summary_smoke.json").read_text())
    assert {r["task"] for r in s["rows"]} == {"code_shape", "arith", "arith_long"} and (S.OUT / "tables_smoke.md").exists()
    assert s["own_order"]["arith"]["removed"] == [2, 1] and s["verdict"]["H2-E"]["label"] in ("supported", "not_supported",
                                                                                              "undetermined")


def test_ownorder_without_sweep_winner_reports_missing(dirs, monkeypatch):
    monkeypatch.setattr(C, "OUT", dirs / "c1")
    monkeypatch.setattr(S, "OUT", dirs / "c1")
    S.OUT.mkdir(parents=True)
    ctx = make_ctx(monkeypatch, tiny_cfg())
    assert C.stage_ownorder(ctx, (0, 1)) == 1


def test_table_smoke_with_no_results(dirs, monkeypatch):
    monkeypatch.setattr(C, "OUT", dirs / "c1")
    monkeypatch.setattr(S, "OUT", dirs / "c1")
    S.OUT.mkdir(parents=True)
    assert C.stage_table(C.make_cfg(True)) == 0
    s = json.loads((S.OUT / "summary_smoke.json").read_text())
    assert s["verdict"]["H2-A"]["label"] == "undetermined" and s["verdict"]["H2-E"]["label"] == "undetermined"


def test_plan_sizes_and_projection():
    assert len(C.HEAL_TASKS) == 25 and "arith_long" in C.HEAL_TASKS
    assert C.KS == (0, 9, 12, 16) and C.OWN_TASKS == ("arith", "code_output", "route_work", "code_uses_first")
    tokens = {t: 100.0 + i for i, t in enumerate(C.HEAL_TASKS)}
    small, big = C.HEAL_TASKS[0], C.HEAL_TASKS[-1]
    p = C.project_c1({small: 1.0, big: 2.0}, {small: 0.1, big: 0.2}, tokens, 28, C.KS, 300)
    assert p["heals"] == 25 * 4 * 4 + 12 and p["total"] == pytest.approx(p["sweep"] + p["ownorder"] + p["test"])
    fb = C.project_c1({small: 1.0, big: 2.0}, {small: 0.1, big: 0.2}, tokens, 28, C.KS_FALLBACK, 300)
    assert fb["total"] < p["total"] and fb["heals"] == 25 * 4 * 3 + 12


# ---------------------------------------------------------------- order.json

def test_atlas_order_defaults_unchanged():
    import atlas_order as O
    assert O.MODEL == "Qwen/Qwen3-0.6B" and O.OUT == Path("var/results/atlas")
    real = Path(__file__).resolve().parent.parent / "var/results/atlas/order.json"
    if real.exists():
        assert json.loads(real.read_text())["model"] == "Qwen/Qwen3-0.6B"
    import inspect
    assert 'default=str(OUT / "order.json")' in inspect.getsource(O.main)


# ---------------------------------------------------------------- H2-E2 follow-up (healed-model own order)

def test_e2_own_order_reads_healed_lesion_file_and_names_do_not_collide(dirs, monkeypatch):
    monkeypatch.setattr(S, "OUT", dirs / "c1")
    S.OUT.mkdir(parents=True)
    cfg = C.make_cfg(False)
    drops = [0.9] * 28
    for b, d in ((4, 0.0), (2, 0.0), (9, -0.1)):
        drops[b] = d
    S.write(C.healed_lesion_path(cfg, "arith"), lesion(drops))
    assert C.healed_lesion_path(cfg, "arith").name == "healed_lesion_arith.json"
    assert C.load_own_order2(cfg, "arith")[:3] == [9, 2, 4]
    # key, checkpoint, shard-file and ledger-key names are disjoint from ownorder's
    assert C.e2_ckpt_path(cfg, "arith", 12, 0, 3e-4).name == "ownorder2_arith_k12_s0_lr0.0003.pt"
    assert C.e2_ckpt_path(cfg, "arith", 12, 0, 3e-4) != C.own_ckpt_path(cfg, "arith", 12, 0, 3e-4)
    assert not C.e2_ckpt_path(cfg, "arith", 12, 0, 3e-4).name.startswith("own_")
    S.write(S.shard_path("ownorder", "", 0, 1), {"runs": {"a|k12|s0|lr0.0003": {}}})
    S.write(S.shard_path(C.E2_PREFIX, "", 0, 1), {"runs": {"b|k12|s0|lr0.0003": {}}})
    assert list(S.read_merged("ownorder", "", "runs")) == ["a|k12|s0|lr0.0003"]
    assert list(S.read_merged(C.E2_PREFIX, "", "runs")) == ["b|k12|s0|lr0.0003"]
    assert "healedlesion" in C.STAGES and C.STAGES[:5] == ["probe", "sweep", "ownorder", "test", "table"]


def test_table2_gains_and_reading_on_fake_numbers(dirs, monkeypatch):
    monkeypatch.setattr(C, "MAPS", dirs / "maps")
    monkeypatch.setattr(C, "BASELINES", ())
    monkeypatch.setattr(S, "OUT", dirs / "c1")
    S.OUT.mkdir(parents=True)
    cfg = C.make_cfg(False)
    cfg.ks = (0, 12)
    shared12 = {"arith": 0.6, "code_output": 0.6, "route_work": 0.88, "code_uses_first": 0.5}
    own12 = {"arith": 0.88, "code_output": 0.8, "route_work": 0.9, "code_uses_first": 0.5}
    e2_12 = {"arith": 0.9, "code_output": 0.7, "route_work": 0.9, "code_uses_first": 0.88}
    runs, own, own2 = {}, {}, {}
    for t in C.OWN_TASKS:
        for k, a in ((0, 0.9), (12, shared12[t])):
            for seed in (0, 1, 2):
                r = fake_row(t, k, seed, 3e-4, a, 0.7, chance=0.5)
                runs[r["key"]] = r
            r = fake_row(t, k, 0, 1e-3, a - 0.1, 0.6, chance=0.5)
            runs[r["key"]] = r
        for d, a, rem in ((own, own12[t], [1, 2]), (own2, e2_12[t], [2, 3])):
            for seed in (0, 1, 2):
                r = fake_row(t, 12, seed, 3e-4, a, 0.7, chance=0.5)
                r["removed"] = rem
                d[r["key"]] = r
    s = C.build_summary_e2(cfg, runs, own, own2, {})
    a, co, ru, cu = (s["tasks"][t] for t in ("arith", "code_output", "route_work", "code_uses_first"))
    assert a["r12_shared"] == pytest.approx(0.25) and a["r12_own"] == pytest.approx(0.95)
    assert a["r12_healed_own"] == pytest.approx(1.0) and a["gain_e2"] == pytest.approx(0.75)
    assert co["gain_e2"] == pytest.approx(0.25) and ru["gain_e2"] == pytest.approx(0.05)
    assert cu["gain_e"] == pytest.approx(0.0) and cu["gain_e2"] == pytest.approx(0.95)
    assert a["removed_own"] == [1, 2] and a["removed_healed_own"] == [2, 3] and a["blocks_in_common"] == [2]
    e2 = s["H2-E2"]
    assert e2["label"] == "not_supported" and len(e2["failed"]) == 2            # code_output 0.25 < 0.30; code_uses_first 0.95
    assert any("code_output" in x for x in e2["failed"]) and any("code_uses_first" in x for x in e2["failed"])
    assert s["H2-E"]["label"] == "supported"                                      # the as-run order still reads supported
    md = C.render_tables_e2(s)
    assert "H2-E2" in md and "| arith |" in md
    # a passing E2
    e2_12.update(code_output=0.85, code_uses_first=0.52)
    for t in ("code_output", "code_uses_first"):
        for r in own2.values():
            if r["task"] == t:
                r["sel_accuracy"] = e2_12[t]
    assert C.build_summary_e2(cfg, runs, own, own2, {})["H2-E2"]["label"] == "supported"


def test_e2_stages_end_to_end_on_tiny_model(dirs, monkeypatch):
    monkeypatch.setattr(C, "OUT", dirs / "c1")
    monkeypatch.setattr(C, "MAPS", dirs / "maps")
    monkeypatch.setattr(C, "BASELINES", ())
    monkeypatch.setattr(S, "OUT", dirs / "c1")
    monkeypatch.setattr(S, "ACTS", dirs / "acts")
    S.OUT.mkdir(parents=True)
    cfg = tiny_cfg()
    cfg.own_tasks = ("code_shape", "arith")
    ctx = make_ctx(monkeypatch, cfg)
    assert C.stage_ownorder2(ctx, (0, 1)) == 1                       # no sweep, no healed lesion yet
    assert C.stage_healedlesion(ctx, "arith") == 1                   # no healed k=0 run yet
    for seed in (0, 1, 2):
        assert C.stage_sweep(ctx, seed, (0, 1)) == 0
    for t in cfg.own_tasks:
        assert C.stage_healedlesion(ctx, t) == 0
    les = json.loads(C.healed_lesion_path(cfg, "arith").read_text())
    n = ctx.n_layers
    assert [r["block"] for r in les["rows"]] == list(range(1, n - 1)) and les["n"] == 12
    assert Path(les["ckpt"]).exists() and not any((dirs / "acts").glob("*healed_lesion*"))
    order = C.load_own_order2(cfg, "arith", n)
    assert C.stage_ownorder2(ctx, (0, 1)) == 0
    own2 = S.read_merged(C.E2_PREFIX, "_smoke", "runs")
    assert len(own2) == 2 * 3 and not S.read_merged("ownorder", "_smoke", "runs")
    for row in own2.values():
        assert row["removed"] == C.load_own_order2(cfg, row["task"], n)[:2] and "ownorder2_" in row["ckpt"]
        assert Path(row["ckpt"]).exists()
    before = {k: v["ts"] for k, v in own2.items()}
    assert C.stage_ownorder2(ctx, (0, 1)) == 0
    assert {k: v["ts"] for k, v in S.read_merged(C.E2_PREFIX, "_smoke", "runs").items()} == before
    assert C.stage_test2(ctx) == 0
    tr = json.loads((S.OUT / "test_results_e2_smoke.json").read_text())["results"]
    assert len(tr) == 6 and all(k.startswith("ownorder2|") for k in tr) and not (S.OUT / "test_results_smoke.json").exists()
    assert C.stage_table2(cfg) == 0
    s = json.loads((S.OUT / "summary_e2_smoke.json").read_text())
    assert set(s["tasks"]) == {"code_shape", "arith"} and (S.OUT / "tables_e2_smoke.md").exists() and order
