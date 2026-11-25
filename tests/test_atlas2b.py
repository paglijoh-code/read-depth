"""M9 Phase 2 stage B: whole-answer training pieces on a tiny random Qwen3 with a stub tokenizer, the statistics of
scripts/atlas2_cuts.py on constructed data, and its stages end to end on the tiny model. No real weights, no network."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import atlas2_cuts as B  # noqa: E402
import atlas_cuts as S  # noqa: E402
from test_atlas_heal import StubTok, dirs, fake_row, make_ctx, tiny, tiny_cfg, tiny_model, toy_records  # noqa: E402,F401

from surgery.atlas import heal, spec  # noqa: E402

EOT = 31


class EotTok(StubTok):
    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return EOT


@pytest.fixture
def stage_dirs(dirs, monkeypatch):
    for name in ("P1_DIR", "MAP_OLD", "MAP_NEW", "HEALED_OLD", "M5_DIR"):
        monkeypatch.setattr(B, name, dirs / name.lower())
    return dirs


def long_records(task, n):
    recs = toy_records(task, n)
    for i, r in enumerate(recs):
        r["meta"] = {"value": 10 * (i % 3) + 7}
    return recs


# ---------------------------------------------------------------- library

def test_whole_target_ends_with_eot_and_loss_masks_prompt(tiny):
    tok, task = EotTok(), "code_shape"
    recs = toy_records(task, 4)
    for r in recs:
        prompt, target = heal.whole_target(tok, task, r)
        assert target == tok.encode(spec.TASKS[task].options[r["answer"]]) + [EOT]
        assert prompt == tok.encode(spec.build_prompt(tok, task, r["prompt"]))
    ids, mask, labels, tmax = heal.whole_batch(tok, task, recs, "cpu")
    assert tmax == max(len(heal.whole_target(tok, task, r)[1]) for r in recs)
    for i, r in enumerate(recs):
        p, t = heal.whole_target(tok, task, r)
        kept = labels[i][labels[i] != -100].tolist()
        assert kept == t and ids[i, -1].item() == EOT and labels[i, -1].item() == EOT
        assert (labels[i, :len(labels[i]) - len(t)] == -100).all()      # prompt and padding masked
        assert mask[i].sum().item() == len(p) + len(t)
    # the loss equals cross-entropy of the target tokens only, over the full logits
    loss = heal.whole_loss(tiny, (ids, mask, labels, tmax)).item()
    logits = tiny(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1].float()
    want = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                                             ignore_index=-100).item()
    assert loss == pytest.approx(want, abs=1e-5)


def test_answer_text_for_arith_long_uses_value():
    assert heal.answer_text("arith_long", {"answer": 1, "meta": {"value": 127}}) == "127"
    assert heal.answer_text("arith", {"answer": 4, "meta": {}}) == "4"
    assert heal.answer_text("code_shape", {"answer": 2}) == "neither"


def test_train_whole_lowers_loss(tiny):
    tok, task = EotTok(), "code_shape"
    train, held = toy_records(task, 24), toy_records(task, 12)
    heal.cut_model(tiny, [0, 1, 3])
    h = heal.add_heal(tiny, 3, rank=4, seed=0)
    enc = heal.whole_batch(tok, task, train, "cpu")
    with torch.no_grad():
        before = heal.whole_loss(tiny, enc).item()
    state, curve, tim = heal.train_whole(tiny, tok, task, train, held, h.modules, 60, 1e-2, batch_size=8, seed=0,
                                         n_evals=4, log=lambda *_: None)
    from surgery import adapters
    adapters.load_adapter_state(h.modules, state)
    with torch.no_grad():
        after = heal.whole_loss(tiny, enc).item()
    assert after < before - 0.1
    assert [c["step"] for c in curve] == [15, 30, 45, 60]
    best = max(c["lrsel_exact"] for c in curve)
    assert tim["best_lrsel"] == best and tim["best_step"] == next(c["step"] for c in curve if c["lrsel_exact"] == best)


def test_exact_match_scores_cuts_at_eot(tiny):
    tok, task = EotTok(), "arith"
    recs = toy_records(task, 5)
    em, texts = heal.exact_match_scores(tiny, tok, task, recs, max_new_tokens=4, batch_size=2)
    assert len(texts) == 5 and 0.0 <= em <= 1.0 and all(len(t) <= 4 for t in texts)


def test_perm_within_keeps_groups():
    rng = np.random.default_rng(0)
    vals, groups = np.arange(8), np.array(list("aabbbcdd"))
    out = heal.perm_within(vals, groups, rng)
    for g in set(groups):
        assert sorted(out[groups == g]) == sorted(vals[groups == g])
    assert out[5] == 5                       # a singleton group cannot move
    assert any(heal.perm_within(vals, groups, rng).tolist() != vals.tolist() for _ in range(5))


# ---------------------------------------------------------------- statistics

def test_r12_clipping_and_drop_rule():
    assert B.r12_value(0.85, 0.90, 0.50) == pytest.approx(0.875)
    assert B.r12_value(0.95, 0.90, 0.50) == 1.0           # above the healed k=0 score: clipped to 1
    assert B.r12_value(0.40, 0.90, 0.50) == 0.0           # below chance: clipped to 0
    assert B.is_dropped(0.59, 0.50) and not B.is_dropped(0.60, 0.50) and not B.is_dropped(0.95, 0.50)


def test_h2a_monotone_case_is_minus_one_with_small_p():
    kinds = ["appraise"] * 5 + ["execute"] * 5
    pred = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    y = [0.9, 0.8, 0.7, 0.6, 0.5, 0.95, 0.85, 0.75, 0.65, 0.55]
    out = B.h2a_test(pred, y, kinds, n_perm=2000)
    assert out["rho"] == pytest.approx(-1.0) and out["p"] < 0.01 and out["n"] == {"appraise": 5, "execute": 5}
    assert B.label_a(out["rho"], out["p"], 0.3) == "supported"
    assert B.label_a(out["rho"], out["p"], -1.0) != "supported"      # the rival must be beaten
    assert B.label_a(-0.3, 0.5, None) == "partial" and B.label_a(0.0, 0.5, None) == "not_supported"
    # ties everywhere in one kind: that kind drops out, the other still counts
    one = B.h2a_test([1, 1, 1, 1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6],
                     ["appraise"] * 4 + ["execute"] * 4, n_perm=0)
    assert one["by_kind"]["appraise"] is None and one["rho"] == pytest.approx(-1.0) and one["p"] is None


def test_h2b_permutation_p_on_constructed_cases():
    groups = [0] * 6 + [1] * 6
    kinds = (["appraise"] * 3 + ["execute"] * 3) * 2
    y = [0.9, 0.95, 0.9, 0.1, 0.15, 0.1] * 2
    out = B.h2b_test(groups, y, kinds, n_perm=2000)
    assert out["stat"] == pytest.approx(0.8) and out["p"] < 0.01 and out["label"] == "supported"
    assert set(out["groups"]) == {"0", "1"}
    null = B.h2b_test(groups, [0.5] * 12, kinds, n_perm=500)
    assert null["stat"] == pytest.approx(0.0) and null["label"] == "not_supported" and null["p"] > 0.5
    # a group with one kind only is left out of the statistic
    mixed = B.h2b_test([0, 0, 1, 1], [0.9, 0.1, 0.5, 0.5], ["appraise", "execute", "appraise", "appraise"], n_perm=0)
    assert mixed["stat"] == pytest.approx(0.8) and list(mixed["groups"]) == ["0"]
    assert [B.l_on_group(x) for x in (0, 1, 3, 4, 10, 11, 28)] == [0, 0, 0, 1, 1, 2, 2]


def test_readings():
    ok = {"arith_order": 0.9, "syntax_first_close": 0.85, "code_uses_first": 0.8, "hard_target_read": 0.1}
    assert B.reading_word(ok)["label"] == "not_explained" and B.reading_word(ok)["task18"] == 0.1
    assert B.reading_word({**ok, "arith_order": 0.2, "syntax_first_close": 0.4, "code_uses_first": 0.5})["label"] == "rested_on"
    assert B.reading_word({**ok, "arith_order": 0.5})["label"] == "mixed"
    assert B.reading_word({"arith_order": 0.9})["label"] == "undetermined"
    assert B.reading_zero_step({"arith_id": 0.8, "code_literal": 0.9})["label"] == "working_steps"
    assert B.reading_zero_step({"arith_id": 0.5, "code_literal": 0.1})["label"] == "kind_beyond_steps"
    assert B.reading_zero_step({"arith_id": 0.9, "code_literal": 0.1})["label"] == "mixed"
    assert [B.label_d(x) for x in (0.95, 0.9, 0.7, 0.49, None)] == ["confirmed", "confirmed", "partial", "harness_differs",
                                                                     "undetermined"]


def test_h2c_cell_count():
    cfg = B.make_cfg(False)
    cfg.whole_tasks = ("arith_id", "arith_long")
    cfg.whole_ks = (0, 12)
    whole = {}
    for task, k, em, ft in [("arith_id", 0, 0.90, 0.92), ("arith_id", 12, 0.50, 0.60), ("arith_long", 0, 0.30, 0.31),
                            ("arith_long", 12, 0.10, 0.12)]:
        for seed in (0, 1, 2):
            for lr, lrsel in ((1e-3, 0.4), (3e-4, 0.5)) if task == "arith_long" and seed == 0 else ((3e-4, 0.5),):
                r = {"key": f"whole|{task}|k{k}|s{seed}|lr{lr:g}", "task": task, "k": k, "seed": seed, "lr": lr,
                     "best_lrsel": lrsel, "exact_match": em, "first_token_accuracy": ft}
                whole[r["key"]] = r
    sweep = {"arith_id": {0: [{"sel_accuracy200": 0.92}] * 3, 12: [{"sel_accuracy200": 0.60}] * 3}}
    out = B.h2c_cells(cfg, whole, sweep)
    assert out["n_cells"] == 4 and out["n_within"] == 3 and out["complete"] and out["need"] == 16
    assert out["label"] == "not_supported"
    cell = {(c["task"], c["k"]): c for c in out["cells"]}
    assert cell[("arith_id", 12)]["within"] is False and cell[("arith_long", 0)]["n_seeds"] == 3   # winner lr only
    assert B.h2c_cells(cfg, {}, {})["label"] == "incomplete"
    out20 = B.h2c_cells(B.make_cfg(False), {}, {})
    assert out20["n_cells"] == 20


# ---------------------------------------------------------------- cuts and Phase 1 behaviour

def test_ordercheck_alive_set_leaves_block_27_out():
    removed = [16, 25, 23, 8, 27, 21, 24, 17, 13, 22, 4, 26]
    alive = B.alive_from_removed(removed, 12, 28)
    assert len(alive) == 16 and 27 not in alive and 0 in alive and alive == sorted(alive)
    assert not set(alive) & set(removed)
    with pytest.raises(ValueError):
        heal.alive_blocks(removed, 12, 28)           # the Phase 1 guard would refuse this set
    assert B.alive_from_removed(removed, 4, 28)[-1] == 27   # block 27 not yet cut at k=4
    with pytest.raises(ValueError):
        B.alive_from_removed([1, 1], 2, 28)


def test_removed_ctx_cuts_the_m5_set(stage_dirs, monkeypatch):
    cfg = tiny_cfg()
    base = make_ctx(monkeypatch, cfg)
    rctx = B.RemovedCtx(base, [3, 1])
    model, alive = rctx.cut(1)
    assert alive == [0, 1, 2] and len(model.model.layers) == 3
    model, alive = rctx.cut(2)
    assert alive == [0, 2] and model.config.num_hidden_layers == 2
    assert len(base.pristine.model.layers) == 4        # the shared pristine model is untouched


def test_phase1_run_heal_transfer_default_and_switch(dirs, monkeypatch):
    cfg = tiny_cfg()
    ctx = make_ctx(monkeypatch, cfg)
    assert set(S.run_heal(ctx, "arith", 1, 0, 1e-3)["transfer"]) == {"code_shape"}      # unset: still scored
    cfg.transfer = False
    assert S.run_heal(ctx, "arith", 1, 0, 1e-3)["transfer"] == {}
    cfg.transfer = True
    assert set(S.run_heal(ctx, "arith", 1, 0, 1e-3)["transfer"]) == {"code_shape"}


# ---------------------------------------------------------------- stages on the tiny model

def make_ctx2(monkeypatch, cfg):
    recs = {}

    def read_records(task, split, root=None):
        n = {"discovery": 12, "train": 18, "lrsel": 8, "selection": 10, "test": 9}[split]
        if (task, split) not in recs:
            recs[(task, split)] = long_records(task, n) if task == "arith_long" else toy_records(task, n)
        return recs[(task, split)]

    monkeypatch.setattr(spec, "read_records", read_records)
    return S.Ctx(cfg, model=tiny_model(), tok=EotTok(), order=[2, 1], device="cpu")


def tiny_cfg2():
    cfg = B.make_cfg(True)
    cfg.tasks, cfg.ks, cfg.whole_tasks, cfg.whole_ks, cfg.n = ("arith_id",), (0, 2), ("arith_id", "arith_long"), (0, 2), None
    cfg.steps, cfg.kcut = 4, 2
    cfg.order_tasks = ("code_shape", "arith")        # the stub tokenizer cannot tell the planner labels apart
    return cfg


def test_sweep_ordercheck_whole_test_table_on_tiny_model(stage_dirs, monkeypatch):
    monkeypatch.setattr(S, "ACTS", stage_dirs / "acts")
    cfg = tiny_cfg2()
    ctx = make_ctx2(monkeypatch, cfg)
    removed = [3, 1, 2]
    B.M5_DIR.mkdir(parents=True)
    for t in cfg.order_tasks:
        (B.M5_DIR / f"excise_Qwen3-0.6B_{t}.json").write_text(json.dumps({"removed_order": removed}))
    # sweep (the Phase 1 stage, reused), no transfer
    for seed in (0, 1, 2):
        assert S.stage_sweep(ctx, seed, (0, 1)) == 0
    runs = S.read_merged("runs", "_smoke", "runs")
    assert len(runs) == 2 * (2 + 1 + 1)
    assert all(r["transfer"] == {} for r in runs.values())
    # order check, two shards: 2 tasks x (2 lrs + 2 seeds) heals, block 3 (the last of the tiny model) cut
    assert B.stage_ordercheck(ctx, (0, 2)) == 0 and B.stage_ordercheck(ctx, (1, 2)) == 0
    order = S.read_merged("ordercheck", "_smoke", "runs")
    assert len(order) == 8 and all(r["n_alive"] == 2 and r["removed"] == [3, 1] for r in order.values())
    assert all(Path(r["ckpt"]).exists() and "ordercheck_" in r["ckpt"] for r in order.values())
    before = {k: v["ts"] for k, v in order.items()}
    assert B.stage_ordercheck(ctx, (0, 2)) == 0
    assert {k: v["ts"] for k, v in S.read_merged("ordercheck", "_smoke", "runs").items()} == before    # resumed
    assert S.OUT / "ordercheck_smoke.shard0of2.json" in list(S.OUT.glob("*.json"))
    # whole-answer check: arith_id at the sweep winner, arith_long at both lrs then the winner
    assert B.stage_whole(ctx, (0, 2)) == 0 and B.stage_whole(ctx, (1, 2)) == 0
    whole = S.read_merged("whole", "_smoke", "runs")
    assert len(whole) == 2 * 3 + 2 * 4
    r = next(iter(whole.values()))
    assert r["n"] == 10 and 0 <= r["exact_match"] <= 1 and len(r["curve"]) == 4 and Path(r["ckpt"]).exists()
    sweep_lr = S.winner_lr(runs, "arith_id", 2, "per_k")
    assert {x["lr"] for x in whole.values() if x["task"] == "arith_id" and x["k"] == 2} == {sweep_lr}
    long_k2 = [x for x in whole.values() if x["task"] == "arith_long" and x["k"] == 2]
    assert sorted((x["seed"], x["lr"]) for x in long_k2).count((0, 1e-3)) == 1 and len(long_k2) == 4
    assert B.stage_whole(ctx, (0, 2)) == 0 and len(S.read_merged("whole", "_smoke", "runs")) == len(whole)
    # test split: new tasks (healed k=0 and k*) then the order-check models; no ledger in smoke
    assert B.stage_test(ctx) == 0
    tr = json.loads((S.OUT / "test_results_smoke.json").read_text())["results"]
    names = {n for r in tr.values() for n in r["names"]}
    assert {"healed_k0", "ordercheck_k2"} <= names and not (S.OUT / "test_ledger.jsonl").exists()
    assert sum("ordercheck_k2" in r["names"] for r in tr.values()) == 6 and all(r["n"] == 9 for r in tr.values())
    n_tr = len(tr)
    B.stage_test(ctx)
    assert len(json.loads((S.OUT / "test_results_smoke.json").read_text())["results"]) == n_tr
    # table
    assert B.stage_table(cfg) == 0
    s = json.loads((S.OUT / "summary_smoke.json").read_text())
    assert {r["task"] for r in s["rows"]} == {"arith_order", "arith_id"} and "H2-C" in s["verdict"]
    assert s["verdict"]["H2-C"]["n_cells"] == 4 and (S.OUT / "tables_smoke.md").exists()


def test_table_on_fake_inputs(stage_dirs, monkeypatch):
    cfg = B.make_cfg(False)
    cfg.ks = (0, 12)
    app = ["route_judgement", "arith_order", "syntax_first_close", "code_uses_first", "hard_target_read"]
    exe = ["rule", "sensible", "arith_id", "code_literal"]
    cfg.table_tasks = tuple(app + exe)
    cfg.whole_tasks, cfg.whole_ks = ("rule", "arith_long"), (0, 12)
    old_rows, new_rows = {}, {}
    l_on = {"route_judgement": 1, "arith_order": 1, "syntax_first_close": 2, "code_uses_first": 4, "hard_target_read": 13,
            "rule": 1, "sensible": 2, "arith_id": 1, "code_literal": 5}
    for t in app + exe:
        a12 = 0.88 if t in app else 0.52          # appraisal survives, execution does not
        tgt = new_rows if t in B.NEW6 else old_rows
        for k, a in ((0, 0.90), (12, a12)):
            for seed in (0, 1, 2):
                r = fake_row(t, k, seed, 3e-4, a, 0.7, chance=0.5)
                tgt[r["key"]] = r
            r = fake_row(t, k, 0, 1e-3, a - 0.1, 0.6, chance=0.5)
            tgt[r["key"]] = r
        B.ac.write(B.MAP_OLD / f"map_{t}.json", {"L_on": l_on[t]})
        if t in B.NEW6:
            B.ac.write(B.MAP_NEW / f"map_{t}.json", {"L_on": l_on[t]})
        B.ac.write(S.OUT / f"healed_{t}.json", {"L_on_healed": l_on[t]})
        B.ac.write(B.HEALED_OLD / f"healed_{t}.json", {"L_on_healed": l_on[t]})
    for d, tasks in ((B.MAP_OLD, [t for t in app + exe if t not in B.NEW6]), (B.MAP_NEW, [t for t in app + exe if t in B.NEW6])):
        B.ac.write(d / "baseline.json", {"tasks": {t: {"bow": 0.5 + 0.02 * i, "chance": 0.5} for i, t in enumerate(tasks)}})
    B.ac.write(B.P1_DIR / "runs.shard0of1.json", {"runs": old_rows})
    B.ac.write(S.OUT / "runs.shard0of1.json", {"runs": new_rows})
    order = {}
    for t, a in (("rule", 0.95), ("sensible", 0.7)):
        for seed in (0, 1, 2):
            r = fake_row(t, 12, seed, 3e-4, a)
            r["removed"] = [1, 2]
            order[r["key"]] = r
        r = fake_row(t, 12, 0, 1e-3, a, 0.4)
        order[r["key"]] = r
    B.ac.write(S.OUT / "ordercheck.shard0of1.json", {"runs": order})
    whole = {}
    for t in cfg.whole_tasks:
        for k in (0, 12):
            for seed in (0, 1, 2):
                lrs = (1e-3, 3e-4) if t == "arith_long" and seed == 0 else (3e-4,)
                for lr in lrs:
                    key = f"whole|{S.run_key(t, k, seed, lr)}"
                    whole[key] = {"key": key, "task": t, "k": k, "seed": seed, "lr": lr, "best_lrsel": 0.5,
                                  "exact_match": 0.8, "first_token_accuracy": 0.82}
    B.ac.write(S.OUT / "whole.shard0of1.json", {"runs": whole})
    B.ac.write(S.OUT / "test_results.json", {"results": {"k": {"task": "rule", "k": 12, "names": ["ordercheck_k12"],
                                                              "accuracy": 0.9, "chance": 0.5}}})
    s = B.build_summary(cfg, B.load_all(cfg), n_perm=200)
    rows = {r["task"]: r for r in s["rows"]}
    assert len(rows) == 9 and rows["arith_id"]["r12"] == pytest.approx((0.52 - 0.5) / 0.4)
    assert rows["route_judgement"]["r12"] == pytest.approx(0.95) and rows["rule"]["healed_k0"] == pytest.approx(0.9)
    assert rows["arith_id"]["surface"] is not None and rows["rule"]["L_on_healed"] == 1
    v = s["verdict"]
    assert v["H2-B"]["stat"] > 0.3 and set(v["H2-B"]["groups"]) <= {"0", "1", "2"}
    assert v["H2-B_drop18"]["tasks"] == [t for t in v["H2-B"]["tasks"] if t != "hard_target_read"]
    assert v["H2-D"]["label"] == "confirmed" and v["H2-D"]["sensible"]["mean"] == pytest.approx(0.7)
    assert v["H2-C"]["n_cells"] == 4 and v["H2-C"]["n_within"] == 2 and v["H2-C"]["complete"]   # arith_long only
    assert v["word_shortcut"]["label"] == "not_explained" and v["zero_step"]["label"] == "kind_beyond_steps"
    md = B.render_tables(s)
    assert "## All tasks" in md and "H2-D" in md and "ordercheck_k12" in md
    # a task under chance + 0.10 is dropped and listed
    new_rows = json.loads((S.OUT / "runs.shard0of1.json").read_text())["runs"]
    for key, r in new_rows.items():
        if r["task"] == "code_uses_first" and r["k"] == 0:
            r["sel_accuracy"] = 0.55
    B.ac.write(S.OUT / "runs.shard0of1.json", {"runs": new_rows})
    s2 = B.build_summary(cfg, B.load_all(cfg), n_perm=0)
    assert s2["verdict"]["dropped"] == ["code_uses_first"] and s2["verdict"]["word_shortcut"]["label"] == "undetermined"


def test_table_smoke_with_no_results(stage_dirs):
    cfg = B.make_cfg(True)
    assert B.stage_table(cfg) == 0
    s = json.loads((S.OUT / "summary_smoke.json").read_text())
    assert s["verdict"]["H2-A"]["label"] == "undetermined" and s["verdict"]["H2-C"]["label"] == "incomplete"
