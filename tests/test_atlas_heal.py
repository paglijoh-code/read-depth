"""M9 Phase 1 cut-and-heal: library pieces on a tiny random Qwen3 with a stub tokenizer, the shard reader, and the stages
of scripts/atlas_cuts.py end to end on that tiny model. No real weights, no network."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import atlas_cuts as S  # noqa: E402

from surgery import adapters  # noqa: E402
from surgery.atlas import heal, measure, metrics, spec  # noqa: E402

GRID = metrics.K_GRID


class Batch(dict):
    def to(self, device):
        return Batch({k: v.to(device) for k, v in self.items()})


class StubTok:
    """Character-level tokenizer into the tiny vocab (ord % 32); left padding with id 0."""
    pad_token_id = 0
    padding_side = "left"

    @staticmethod
    def _ids(text):
        return [ord(c) % 32 for c in text]

    def apply_chat_template(self, messages, **kw):
        return "|".join(m["content"] for m in messages)

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["".join(chr(64 + int(i) % 26) for i in row) for row in ids]

    def __call__(self, text, add_special_tokens=False, return_tensors="pt", padding=False, **kw):
        texts = [text] if isinstance(text, str) else list(text)
        seqs = [self._ids(t)[:kw.get("max_length") or 10 ** 9] for t in texts]
        n = max(map(len, seqs))
        ids = torch.zeros(len(seqs), n, dtype=torch.long)
        mask = torch.zeros(len(seqs), n, dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, n - len(s):] = torch.tensor(s)
            mask[r, n - len(s):] = 1
        return Batch(input_ids=ids, attention_mask=mask)


def tiny_model():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    model = Qwen3ForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    return model


@pytest.fixture
def tiny():
    return tiny_model()


def toy_records(task, n, rungs=False):
    """Three prompt classes the model can tell apart; answer = class."""
    n_opt = len(spec.TASKS[task].options)
    texts = ["xx xx", "yy yy", "zz zz"]
    return [{"id": f"{task}{i}", "task": task, "prompt": f"{texts[i % 3]} {i % 5}", "answer": (i % 3) % n_opt,
             "template_id": f"t{i % 4}", "rung": (["ops1", "ops2"][i % 2] if rungs else None), "meta": {}}
            for i in range(n)]


# ---------------------------------------------------------------- library

def test_alive_blocks_keeps_ends_and_order():
    order = [25, 24, 26, 12, 13, 23, 14, 22, 15, 21, 11, 20, 18, 16, 10, 17, 19, 8, 9, 6, 7, 5, 2, 1, 4, 3]
    for k in GRID:
        alive = heal.alive_blocks(order, k, 28)
        assert len(alive) == 28 - k and alive == sorted(alive) and alive[0] == 0 and alive[-1] == 27
        assert not set(alive) & set(order[:k])
    with pytest.raises(ValueError):
        heal.alive_blocks([0, 1, 2], 1, 4)       # block 0 may not be cut
    with pytest.raises(ValueError):
        heal.alive_blocks([3, 1, 2], 1, 4)       # nor the last block
    with pytest.raises(ValueError):
        heal.alive_blocks([1, 2], 3, 4)


def test_cut_and_add_heal_shapes(tiny):
    alive = heal.alive_blocks([2, 1], 1, 4)
    heal.cut_model(tiny, alive)
    assert len(tiny.model.layers) == 3 and tiny.config.num_hidden_layers == 3
    h = heal.add_heal(tiny, 3, rank=10, seed=0)
    assert len(h.modules) == 6 and adapters.count_trainable(tiny) == 3 * 10 * ((64 + 64) + (64 + 32))
    assert all(isinstance(tiny.model.layers[i].self_attn.q_proj, adapters.LoRALinear) for i in range(3))


def test_train_heal_lowers_loss_and_keeps_best(tiny):
    tok, task = StubTok(), "code_shape"
    train, held = toy_records(task, 24), toy_records(task, 12)
    heal.cut_model(tiny, heal.alive_blocks([2, 1], 1, 4))
    h = heal.add_heal(tiny, 3, rank=4, seed=0)
    fwd = heal.option_forward(tiny, tok, task)
    y = torch.tensor([r["answer"] for r in train])
    with torch.no_grad():
        before = torch.nn.functional.cross_entropy(fwd(train), y).item()
    state, curve, tim = heal.train_heal(tiny, tok, task, train, held, h.modules, 60, 1e-2, batch_size=8, seed=0,
                                        log=lambda *_: None)
    adapters.load_adapter_state(h.modules, state)
    with torch.no_grad():
        after = torch.nn.functional.cross_entropy(fwd(train), y).item()
    assert after < before - 0.05
    assert [c["step"] for c in curve] == [15, 30, 45, 60]
    best = max(c["lrsel_accuracy"] for c in curve)
    assert tim["best_lrsel"] == best and tim["best_step"] == next(c["step"] for c in curve if c["lrsel_accuracy"] == best)
    assert tim["step_seconds"] > 0 and set(state) and all("base_layer" not in k for k in state)
    # the base weights did not move
    ref = tiny_model()
    heal.cut_model(ref, heal.alive_blocks([2, 1], 1, 4))
    assert torch.equal(tiny.model.layers[0].mlp.up_proj.weight, ref.model.layers[0].mlp.up_proj.weight)


def test_lora_state_roundtrips_to_same_logits(tiny, tmp_path):
    tok, task = StubTok(), "code_shape"
    pristine = copy.deepcopy(tiny)
    recs = toy_records(task, 9)
    heal.cut_model(tiny, [0, 1, 3])
    h = heal.add_heal(tiny, 3, rank=4, seed=3)
    state, _, _ = heal.train_heal(tiny, tok, task, recs, recs[:6], h.modules, 8, 1e-2, n_evals=2, log=lambda *_: None)
    adapters.load_adapter_state(h.modules, state)
    path = tmp_path / "c.pt"
    S.save_ckpt(path, state, {"k": 1})
    want = measure.option_logits(tiny, tok, task, recs)
    other = copy.deepcopy(pristine)
    heal.cut_model(other, [0, 1, 3])
    h2 = heal.add_heal(other, 3, rank=4, seed=99)  # a different random init
    adapters.load_adapter_state(h2.modules, torch.load(path, map_location="cpu")["state"])
    got = measure.option_logits(other, tok, task, recs)
    assert np.abs(got - want).max() < 1e-5
    # merging keeps the logits too
    adapters.merge_lora(h2)
    assert np.abs(measure.option_logits(other, tok, task, recs) - want).max() < 1e-4


def test_pick_lr_tie_rule():
    rows = lambda a, b: [{"lr": 1e-3, "best_lrsel": a}, {"lr": 3e-4, "best_lrsel": b}]  # noqa: E731
    assert heal.pick_lr(rows(0.8, 0.7)) == 1e-3
    assert heal.pick_lr(rows(0.7, 0.8)) == 3e-4
    assert heal.pick_lr(rows(0.75, 0.75)) == 3e-4
    assert heal.pick_lr([{"lr": 1e-3, "best_lrsel": 0.5}]) == 1e-3


def test_bootstrap_rho_monotone_case_is_minus_one():
    ks = list(GRID)
    acc, pred = {}, {}
    for i, kstar in enumerate(reversed(ks)):          # predictor rises, k* falls: one task per grid value
        t = f"t{i}"
        pred[t] = float(i)
        acc[t] = {k: [0.9, 0.88, 0.92] if k <= kstar else [0.5, 0.52, 0.48] for k in ks}
    out = heal.bootstrap_rho(acc, pred, GRID, n_boot=300)
    assert out["rho"] == pytest.approx(-1.0)
    assert out["hi"] < 0 and out["lo"] <= out["hi"] and out["n_valid"] > 0
    assert out["kstar"] == {f"t{i}": k for i, k in enumerate(reversed(ks))}
    assert heal.bootstrap_rho({"a": acc["t0"]}, {"a": 1.0}, GRID)["rho"] is None   # under 3 tasks


def test_spearman_ties_and_constant():
    assert heal.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)
    assert np.isnan(heal.spearman([1, 1, 1], [1, 2, 3]))


def test_exact_match():
    assert heal.exact_match("  7. \n", "7")
    assert heal.exact_match("YES", "YES") and heal.exact_match("yes .", "yes")
    assert not heal.exact_match("7 and more", "7") and not heal.exact_match("77", "7")
    assert not heal.exact_match("Yes", "yes") and not heal.exact_match("7..", "7")


def test_generate_answers_on_excised_model(tiny):
    tok, task = StubTok(), "arith"
    heal.cut_model(tiny, [0, 3])
    texts = heal.generate_answers(tiny, tok, task, toy_records(task, 5), max_new_tokens=3, batch_size=2)
    assert len(texts) == 5 and all(len(t) == 3 for t in texts)


def test_project_hours_orders_rules():
    tokens = {"route_work": 400.0, "arith": 100.0, "code_shape": 150.0}
    p = heal.project_hours({"route_work": 1.0, "arith": 0.3}, tokens, 0.5, ks=GRID)
    assert p["heals"] == {"main": 72, "fallback": 57}
    assert p["fallback"]["sweep"] < p["main"]["sweep"] and p["main"]["total"] > p["main"]["sweep"]
    assert heal.project_hours({"route_work": 1.0, "arith": 0.3}, tokens, None)["main"]["ceiling"] is None


# ---------------------------------------------------------------- script: shard files, stages

@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "OUT", tmp_path / "res")
    monkeypatch.setattr(S, "CKPT", tmp_path / "ckpt")
    monkeypatch.setattr(S, "ACTS", tmp_path / "acts")
    monkeypatch.setattr(S, "MAP_DIR", tmp_path / "map")
    monkeypatch.setattr(S, "GEN", tmp_path / "gen")
    S.OUT.mkdir(parents=True)
    return tmp_path


def fake_row(task, k, seed, lr, sel, lrsel=0.5, transfer=None, chance=0.4, rung=None):
    return {"key": S.run_key(task, k, seed, lr), "task": task, "k": k, "seed": seed, "lr": lr, "best_lrsel": lrsel,
            "sel_accuracy": sel, "sel_accuracy200": sel, "chance": chance, "transfer": transfer or {},
            "rung_acc": rung or {}, "rung_chance": {g: 0.5 for g in (rung or {})}, "ckpt": "x.pt"}


def test_read_merged_merges_shard_files(dirs):
    S.write(S.OUT / "runs.shard0of2.json", {"shard": [0, 2], "runs": {"a": {"v": 1}}})
    S.write(S.OUT / "runs.shard1of2.json", {"shard": [1, 2], "runs": {"b": {"v": 2}}})
    S.write(S.OUT / "runs_smoke.shard0of1.json", {"runs": {"c": {"v": 3}}})
    assert S.read_merged("runs", "", "runs") == {"a": {"v": 1}, "b": {"v": 2}}
    assert S.read_merged("runs", "_smoke", "runs") == {"c": {"v": 3}}
    assert S.parse_shard("1/3") == (1, 3) and S.parse_shard(None) == (0, 1)


def test_unit_shards_cover_each_unit_once():
    for rule in ("per_k", "fallback"):
        seen = [[S.unit_shard(ti, ki, 3, rule) for ki in range(6)] for ti in range(14)]
        assert all(0 <= s < 3 for row in seen for s in row)
    assert len({S.unit_shard(0, ki, 3, "fallback") for ki in range(6)}) == 1   # one task, one shard under the fallback


def test_winner_and_unit_lrs():
    runs = {r["key"]: r for r in [fake_row("t", 6, 0, 1e-3, 0.9, 0.8), fake_row("t", 6, 0, 3e-4, 0.9, 0.7),
                                  fake_row("t", 0, 0, 1e-3, 0.9, 0.6), fake_row("t", 0, 0, 3e-4, 0.9, 0.6)]}
    assert S.winner_lr(runs, "t", 6, "per_k") == 1e-3 and S.winner_lr(runs, "t", 0, "per_k") == 3e-4
    assert S.winner_lr(runs, "t", 6, "fallback") == 3e-4          # k=0's pick, reused at every k
    assert S.unit_lrs(runs, "t", 6, 0, "per_k") == list(S.LRS)
    assert S.unit_lrs(runs, "t", 6, 0, "fallback") == [3e-4]
    assert S.unit_lrs(runs, "t", 6, 1, "per_k") == [1e-3]
    assert S.unit_lrs(runs, "u", 6, 1, "per_k") == []


def test_table_on_fake_results(dirs):
    cfg = S.make_cfg(False)
    cfg.tasks = ("arith_shape", "arith", "code_shape")
    accs = {"arith_shape": {0: 0.90, 6: 0.90, 12: 0.80}, "arith": {0: 0.80, 6: 0.78, 12: 0.76},
            "code_shape": {0: 0.95, 6: 0.80, 12: 0.60}}
    runs = {}
    for t, by_k in accs.items():
        others = [b for b in cfg.tasks if b != t]
        for k, a in by_k.items():
            for seed in (0, 1, 2):
                rung = {"ops1": a + 0.05, "ops2": a - 0.05} if t == "arith" else None
                tr = {b: 0.5 + 0.01 * k for b in others}
                runs.update({r["key"]: r for r in [fake_row(t, k, seed, 3e-4, a + 0.01 * (seed - 1), 0.7, tr, 0.4, rung)]})
            runs.update({r["key"]: r for r in [fake_row(t, k, 0, 1e-3, a - 0.1, 0.6, {}, 0.4)]})
    S.write(S.OUT / "runs.shard0of1.json", {"runs": runs})
    S.write(S.OUT / "cutonly.shard0of1.json", {"cut": {f"{t}|k{k}": {"acc": 0.4, "acc200": 0.3, "rung_acc": {"ops1": 0.45}}
                                                       for t in cfg.tasks for k in (0, 6, 12)}})
    for t, lo in (("arith_shape", 1), ("arith", 5), ("code_shape", 9)):
        S.write(S.MAP_DIR / f"map_{t}.json", {"L_on": lo, "L_lens": lo + 1, "fragile": [lo]})
    S.write(S.OUT / "rivals.json", {t: {"bi": 0.1 * i, "ce_rise": 0.2 * i} for i, t in enumerate(cfg.tasks)})
    S.stage_table(cfg)
    s = json.loads((S.OUT / "summary.json").read_text())
    md = (S.OUT / "tables.md").read_text()
    rows = {(r["task"], r["rung"]): r for r in s["rows"]}
    a = rows[("arith_shape", None)]
    assert a["lr"]["6"] == 3e-4 and a["healed_k0"] == pytest.approx(0.90) and a["k_star"] == 6
    assert a["unmodified"] == pytest.approx(0.4) and a["rel_drop"] == pytest.approx(0.0, abs=1e-9)
    assert rows[("arith", None)]["k_star"] == 12 and rows[("code_shape", None)]["k_star"] == 0
    assert rows[("code_shape", None)]["rel_drop"] == pytest.approx(0.0)
    assert rows[("arith", "ops1")]["k_star"] == 12 and ("arith", "ops2") in rows
    assert a["acc"]["12"]["min"] < a["acc"]["12"]["max"] and a["L_on"] == 1
    # T[a,b] at k=12: after-heal 0.62 minus cut-only 0.3
    assert s["transfer"]["12"]["arith"]["code_shape"] == pytest.approx(0.32)
    assert len(s["transfer_cells_ge_5pts"]) == 6 and s["headline_k"] == 12
    assert s["verdict"]["H-R2"]["pairs"] == [{"appraise": "arith_shape", "execute": "arith", "diff": -6}]
    assert s["verdict"]["H-R1"]["L_on"]["n_tasks"] == 3 and s["verdict"]["H-R1"]["L_on"]["rho"] is not None
    assert s["verdict"]["H-R3"]["arith"]["k_star"] == [12, 12]
    assert "## Accuracy by k" in md and "### k=12" in md and "Not run" in md


def test_table_cli_smoke_with_no_results(dirs):
    cfg = S.make_cfg(True)
    s = S.build_summary(cfg, S.load_all(cfg))
    assert s["rows"] == [] and S.render_tables(s).startswith("# Atlas Phase 1 tables (smoke)")


def make_ctx(monkeypatch, cfg, order=(2, 1)):
    recs = {}

    def read_records(task, split, root=None):
        n = {"discovery": 12, "train": 18, "lrsel": 8, "selection": 10, "test": 9}[split]
        key = (task, split)
        if key not in recs:
            recs[key] = toy_records(task, n, rungs=bool(spec.TASKS[task].rungs))
        return recs[key]

    monkeypatch.setattr(spec, "read_records", read_records)
    return S.Ctx(cfg, model=tiny_model(), tok=StubTok(), order=list(order), device="cpu")


def tiny_cfg():
    cfg = S.make_cfg(True)
    cfg.tasks, cfg.ks, cfg.rival_k, cfg.ceiling_k, cfg.headline_k, cfg.n, cfg.ledger = ("code_shape", "arith"), (0, 1, 2), 2, 2, 2, None, False
    return cfg


def test_stages_end_to_end_on_tiny_model(dirs, monkeypatch):
    cfg = tiny_cfg()
    ctx = make_ctx(monkeypatch, cfg)
    base_w = ctx.pristine.model.layers[1].mlp.up_proj.weight.clone()
    # seed 0 across two shards, then seed 1 reads the winners
    assert S.stage_sweep(ctx, 0, (0, 2)) == 0 and S.stage_sweep(ctx, 0, (1, 2)) == 0
    runs = S.read_merged("runs", "_smoke", "runs")
    assert len(runs) == 2 * 3 * 2 and (S.OUT / "runs_smoke.shard0of2.json").exists() and (S.OUT / "runs_smoke.shard1of2.json").exists()
    row = runs[S.run_key("arith", 2, 0, 3e-4)]
    assert row["n_alive"] == 2 and set(row["transfer"]) == {"code_shape"} and len(row["sel_preds"]) == 10
    assert set(row["rung_acc"]) == {"ops1", "ops2"} and Path(row["ckpt"]).exists() and len(row["curve"]) == 4
    assert row["timings"]["step_seconds"] > 0
    before = {k: v["ts"] for k, v in runs.items()}
    assert S.stage_sweep(ctx, 0, (0, 2)) == 0                      # resumed: nothing re-run
    assert {k: v["ts"] for k, v in S.read_merged("runs", "_smoke", "runs").items()} == before
    assert S.stage_sweep(ctx, 1, (0, 2)) == 0 and S.stage_sweep(ctx, 1, (1, 2)) == 0
    runs = S.read_merged("runs", "_smoke", "runs")
    assert len(runs) == 2 * 3 * 3                                   # winner lr only for seed 1
    assert all(S.run_key(t, k, 1, S.winner_lr(runs, t, k, "per_k")) in runs for t in cfg.tasks for k in cfg.ks)
    assert torch.equal(ctx.pristine.model.layers[1].mlp.up_proj.weight, base_w)   # the clean copy is never modified
    # the saved state rebuilds the scored model
    model, h = S.rebuild_healed(ctx, row)
    assert heal.accuracy_and_preds(model, ctx.tok, "arith", ctx.records("arith", "selection"))[1] == row["sel_preds"]
    # cut-only, in two shards
    assert S.stage_cutonly(ctx, (0, 2)) == 0 and S.stage_cutonly(ctx, (1, 2)) == 0
    assert len(S.read_merged("cutonly", "_smoke", "cut")) == 6
    # fallback plan: seed 0 at k>0 uses the k=0 winner only
    S.write(S.OUT / "plan_smoke.json", {"lr_rule": "fallback"})
    assert S.plan_rule(cfg) == "fallback"
    S.write(S.OUT / "plan_smoke.json", {"lr_rule": "per_k"})
    # secondary, generative, rivals, test, table
    assert S.stage_secondary(ctx, (0, 1)) == 0
    sec = json.loads((S.OUT / "secondary_arith_smoke.json").read_text())
    assert len(sec["lens"]["all"]["curve"]) == 5 and "fragile" in sec and not (S.ACTS / "arith_smoke").exists()
    assert S.stage_generative(ctx) == 0
    gen = json.loads((S.OUT / "generative_smoke.json").read_text())
    assert set(gen["arith"]) == {"unmodified", "healed_k0", "healed_k2"} and gen["arith"]["healed_k2"]["n"] == 10
    assert S.stage_rivals(ctx) == 0
    riv = json.loads((S.OUT / "rivals_smoke.json").read_text())
    assert set(riv) == {"code_shape", "arith"} and riv["arith"]["bi"] > 0 and riv["arith"]["k"] == 2
    assert S.stage_test(ctx) == 0
    tr = json.loads((S.OUT / "test_results_smoke.json").read_text())["results"]
    assert all(r["n"] == 9 for r in tr.values()) and not (S.OUT / "test_ledger.jsonl").exists()
    assert S.stage_ceiling(ctx) == 0 and not (S.OUT / "ceiling_smoke.json").exists()   # general data absent: skipped
    S.stage_table(cfg)
    s = json.loads((S.OUT / "summary_smoke.json").read_text())
    assert {r["task"] for r in s["rows"]} == {"code_shape", "arith"} and s["ks"] == [0, 1, 2]
    assert s["generative"]["arith"]["healed_k0"]["n"] == 10 and s["test"]


def test_test_ledger_written_once_per_model(dirs, monkeypatch):
    cfg = tiny_cfg()
    cfg.tasks, cfg.ks, cfg.ledger = ("code_shape",), (0, 2), True
    ctx = make_ctx(monkeypatch, cfg)
    for seed in (0, 1, 2):
        assert S.stage_sweep(ctx, seed, (0, 1)) == 0
    S.stage_test(ctx)
    S.stage_test(ctx)                                               # second call must not query again
    lines = [json.loads(l) for l in (S.OUT / "test_ledger.jsonl").read_text().splitlines()]
    keys = [l["key"] for l in lines]
    assert len(keys) == len(set(keys)) and len(keys) >= 3 and all(l["n"] == 9 for l in lines)


def test_ceiling_stage_with_general_data(dirs, monkeypatch):
    cfg = tiny_cfg()
    ctx = make_ctx(monkeypatch, cfg)
    S.GEN.mkdir()
    (S.GEN / "manifest.json").write_text("{}")
    rows = [{"text": f"general text number {i} about nothing"} for i in range(40)]
    for name in ("train", "dev"):
        (S.GEN / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    base_w = ctx.pristine.model.layers[1].mlp.up_proj.weight.clone()
    assert S.stage_ceiling(ctx) == 0
    doc = json.loads((S.OUT / "ceiling_smoke.json").read_text())["runs"]
    assert set(doc) == {"ceiling|k2|s0", "ceiling|k2|s1", "ceiling|k2|s2"}
    r = doc["ceiling|k2|s0"]
    assert set(r["scores"]) == {"code_shape", "arith"} and len(r["curve"]) == 4 and r["timings"]["step_seconds"] > 0
    assert all(c["general_dev_kl"] >= 0 for c in r["curve"])
    assert torch.equal(ctx.pristine.model.layers[1].mlp.up_proj.weight, base_w)
    mtime = (S.OUT / "ceiling_smoke.json").stat().st_mtime_ns
    assert S.stage_ceiling(ctx) == 0 and (S.OUT / "ceiling_smoke.json").stat().st_mtime_ns == mtime   # resumed


def test_probe_writes_plan(dirs, monkeypatch):
    cfg = tiny_cfg()
    cfg.ks = (0, 1, 2)
    ctx = make_ctx(monkeypatch, cfg)
    monkeypatch.setattr(S, "ROUTE_LONG", "code_shape")
    assert S.stage_probe(ctx, 1.0) == 0
    plan = json.loads((S.OUT / "plan_smoke.json").read_text())
    assert plan["lr_rule"] in ("per_k", "fallback") and plan["projection"]["main"]["sweep"] > 0
