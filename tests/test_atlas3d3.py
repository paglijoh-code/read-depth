"""Step D3 (docs/ATLAS-phase2-stageC.md, "D3 bars"; scripts/atlas3_cuts.py --model smol17): the model profiles, a tiny random
Llama through cut, q/v LoRA heal and merge, the C1/D1 stages and the D3 table on it, 24-block order and n_pre arithmetic, and the
D3 table on synthetic rows (each label, the drop rule). No real weights, no network."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import atlas2_cuts as B  # noqa: E402
import atlas3_cuts as C  # noqa: E402
import atlas3_extra as X  # noqa: E402
import atlas_cuts as S  # noqa: E402
from test_atlas2b import EotTok, long_records  # noqa: E402
from test_atlas_heal import dirs, fake_row, toy_records  # noqa: E402,F401

from surgery import adapters  # noqa: E402
from surgery.atlas import heal, measure, spec  # noqa: E402


def tiny_llama(layers=4):
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                         num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=True)
    model = LlamaForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    return model


# ---------------------------------------------------------------- profiles

def test_qwen17_profile_is_todays_configuration():
    p = C.resolve_profile("qwen17")
    assert (p.model, p.maps, p.out, p.ckpt, p.acts) == ("Qwen/Qwen3-1.7B", Path("var/results/atlas3/1.7B"),
                                                         Path("var/results/atlas3/c1"), Path("var/ckpt/atlas3"),
                                                         Path("var/acts/atlas3"))
    assert (p.n_layers, p.kcut, p.ks) == (28, 12, (0, 9, 12, 16))
    assert (p.model, p.maps, p.out, p.ckpt, p.acts) == (C.MODEL, C.MAPS, C.OUT, C.CKPT, C.ACTS) and p.ks == C.KS
    assert C.resolve_profile("qwen17", smoke=True) is p                     # smoke keeps C1's folders
    assert vars(C.make_cfg(False, p)) == vars(C.make_cfg(False)) and vars(C.make_cfg(True, p)) == vars(C.make_cfg(True))
    cfg = C.make_cfg(False)
    assert (cfg.kcut, cfg.ks, cfg.tasks, cfg.n_own, cfg.fallback_grid) == (12, C.KS, C.HEAL_TASKS, 12, C.KS_FALLBACK)


def test_smol17_profile_and_cfg():
    p = C.resolve_profile("smol17")
    assert p.model == "HuggingFaceTB/SmolLM2-1.7B-Instruct" and p.n_layers == 24 and p.kcut == 10 and p.ks == (0, 10)
    assert round(0.43 * 24) == p.kcut
    assert (p.maps, p.out, p.ckpt, p.acts) == (Path("var/results/atlas3/smol17"), Path("var/results/atlas3/d3"),
                                                Path("var/ckpt/atlas3/d3"), Path("var/acts/atlas3/d3"))
    sm = C.resolve_profile("smol17", smoke=True)
    assert sm.out == Path("var/results/atlas3/d3_smoke") and sm.maps == Path("var/results/atlas3/smol17_smoke")
    assert sm.ckpt == Path("var/ckpt/atlas3/d3_smoke") and sm.acts == Path("var/acts/atlas3/d3_smoke") and sm.n_layers == 24
    cfg = C.make_cfg(False, p)
    assert cfg.kcut == 10 and cfg.ks == (0, 10) and cfg.steps == 300 and cfg.sfx == "" and cfg.n_own == 0
    assert len(cfg.tasks) == 24 and cfg.tasks == cfg.table_tasks and C.LONG_TASK not in cfg.tasks
    assert sorted(spec.TASKS[t].number for t in cfg.tasks) == [n for n in range(1, 26) if n != 21]
    assert cfg.fallback_grid is None
    smoke = C.make_cfg(True, p)
    assert smoke.sfx == "_smoke" and smoke.ks == (0, 2) and smoke.kcut == 2 and smoke.n_own == 0 and not smoke.ledger
    assert C.PROFILE_STAGES["smol17"] == ("probe", "sweep", "healedprobe", "d3table") and C.D3_STAGES == ["d3table"]


def test_configure_rebinds_this_module_and_the_phase1_module(monkeypatch):
    for mod, names in ((C, ("MODEL", "MAPS", "ORDER_PATH", "OUT", "CKPT", "ACTS")), (S, ("MODEL", "ORDER_PATH", "OUT", "CKPT", "ACTS")),
                       (B, ("OUT", "CKPT", "ACTS"))):
        for n in names:
            monkeypatch.setattr(mod, n, getattr(mod, n))
    C.configure(C.resolve_profile("smol17"))
    assert C.MODEL == S.MODEL == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    assert C.ORDER_PATH == S.ORDER_PATH == Path("var/results/atlas3/smol17/order.json") and C.MAPS == Path("var/results/atlas3/smol17")
    assert C.OUT == S.OUT == B.OUT == Path("var/results/atlas3/d3") and S.CKPT == B.CKPT == Path("var/ckpt/atlas3/d3")
    assert S.ACTS == B.ACTS == Path("var/acts/atlas3/d3")
    C.configure(C.resolve_profile("qwen17"))
    assert C.MODEL == S.MODEL == "Qwen/Qwen3-1.7B" and S.ORDER_PATH == Path("var/results/atlas3/1.7B/order.json")
    assert C.OUT == S.OUT == Path("var/results/atlas3/c1") and S.CKPT == Path("var/ckpt/atlas3")


def test_block_count_must_match_the_profile():
    C.check_blocks(24, C.resolve_profile("smol17"))
    C.check_blocks(28, C.resolve_profile("qwen17"))
    C.check_blocks(4, None)
    with pytest.raises(SystemExit):
        C.check_blocks(28, C.resolve_profile("smol17"))


def test_main_refuses_other_stages_for_smol17_and_d3table_for_qwen():
    with pytest.raises(SystemExit):
        C.main(["--model", "smol17", "--stage", "ownorder"])
    with pytest.raises(SystemExit):
        C.main(["--stage", "d3table"])
    assert C.STAGES[:5] == ["probe", "sweep", "ownorder", "test", "table"] and C.STAGES[-1] == "healedprobe"


def test_projection_without_ownorder_arm():
    tokens = {t: 100.0 + i for i, t in enumerate(C.HEAL_TASKS[:24])}
    small, big = C.HEAL_TASKS[0], C.HEAL_TASKS[23]
    p = C.project_c1({small: 1.0, big: 2.0}, {small: 0.1, big: 0.2}, tokens, 24, (0, 10), 300, 10, n_own=0)
    assert p["heals"] == 24 * 4 * 2 and p["ownorder"] == 0 and p["total"] == pytest.approx(p["sweep"] + p["test"])


# ---------------------------------------------------------------- Llama through the library

def test_llama_excise_lora_heal_merge():
    model = tiny_llama(4)
    base = model.model.layers[3]
    alive = heal.alive_blocks([2, 1], 2, 4)
    assert alive == [0, 3]
    heal.cut_model(model, alive)
    assert model.config.num_hidden_layers == 2 and model.model.layers[1] is base
    assert [l.self_attn.layer_idx for l in model.model.layers] == [0, 1]
    handle = heal.add_heal(model, len(alive), 10, 0)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(handle.modules) == 4 and len(trainable) == 8 and all(n.endswith((".A", ".B")) for n in trainable)
    tok, task = EotTok(), "code_shape"
    recs = toy_records(task, 24)
    state, curve, tim = heal.train_whole(model, tok, task, recs[:16], recs[16:], handle.modules, 3, 1e-3, batch_size=4,
                                         seed=0, n_evals=1, max_new_tokens=3)
    adapters.load_adapter_state(handle.modules, state)
    ids = torch.tensor([[3, 4, 5, 6]])
    with torch.no_grad():
        before = model(input_ids=ids).logits
    assert adapters.merge_lora(handle) == 4
    with torch.no_grad():
        after = model(input_ids=ids).logits
    assert torch.allclose(before, after, atol=1e-5) and not any(isinstance(m, adapters.LoRALinear) for m in model.modules())
    assert len(curve) == 1 and tim["best_step"] == 3


def test_llama_store_and_lesion_sweep(tmp_path):
    model = tiny_llama(4)
    tok, task = EotTok(), "code_shape"
    recs = toy_records(task, 12)
    out = tmp_path / "store"
    meta = measure.build_store(model, tok, task, recs, out, batch_size=4)
    assert meta["n_layers"] == 4 and meta["d_model"] == 64
    means = torch.load(out / "means.pt", map_location="cpu", weights_only=True)
    les = measure.lesion_sweep(model, tok, task, recs, means, batch_size=4)
    assert [r["block"] for r in les["rows"]] == [0, 1, 2, 3]


def test_llama_shared_order_pieces():
    """What atlas_order.py computes (block influence, skip-one perplexity, the order), on a 4-block Llama; the skipped-block
    forward must leave the model intact afterwards."""
    from surgery.atlas import metrics
    model, tok = tiny_llama(4), EotTok()
    text = "the quick brown fox jumps over the lazy dog " * 8
    bi = measure.block_influence(model, tok, text, window=32)
    res = measure.ppl_drop(model, tok, text, max_tokens=64)
    assert len(bi) == 4 and len(res["drop"]) == 4 and len(model.model.layers) == 4 and model.config.num_hidden_layers == 4
    order = metrics.shared_order(bi, 4)
    assert sorted(order) == [1, 2]
    with pytest.raises(ValueError):
        metrics.shared_order(bi, 24)


# ---------------------------------------------------------------- stages on a tiny Llama

def tiny_d3_cfg():
    cfg = C.make_cfg(True, C.resolve_profile("smol17", True))
    cfg.tasks = cfg.table_tasks = ("code_shape", "arith")
    cfg.ks, cfg.kcut, cfg.steps, cfg.n = (0, 2), 2, 4, None
    return cfg


def make_llama_ctx(monkeypatch, cfg):
    recs = {}

    def read_records(task, split, root=None):
        n = {"discovery": 12, "train": 18, "lrsel": 8, "selection": 10, "test": 9}[split]
        if (task, split) not in recs:
            recs[(task, split)] = long_records(task, n) if task == "arith_long" else toy_records(task, n)
        return recs[(task, split)]

    monkeypatch.setattr(spec, "read_records", read_records)
    return S.Ctx(cfg, model=tiny_llama(4), tok=EotTok(), order=[2, 1], device="cpu")


def test_d3_stages_end_to_end_on_tiny_llama(dirs, monkeypatch):
    for n in ("OUT", "MAPS"):
        monkeypatch.setattr(C, n, dirs / n.lower())
    monkeypatch.setattr(C, "BASELINES", (dirs / "base.json",))
    monkeypatch.setattr(C, "ORDER_PATH", dirs / "maps" / "order.json")
    monkeypatch.setattr(C, "C1_OUT", dirs / "c1")
    monkeypatch.setattr(S, "OUT", dirs / "out")
    monkeypatch.setattr(S, "ACTS", dirs / "acts")
    S.OUT.mkdir(parents=True)
    cfg = tiny_d3_cfg()
    ctx = make_llama_ctx(monkeypatch, cfg)
    assert ctx.n_layers == 4
    for seed in (0, 1, 2):
        assert C.stage_sweep(ctx, seed, (0, 2)) == 0 and C.stage_sweep(ctx, seed, (1, 2)) == 0
    runs = S.read_merged("runs", "_smoke", "runs")
    assert len(runs) == 2 * 2 * 4 and runs[S.run_key("arith", 2, 0, 3e-4)]["n_alive"] == 2
    assert all(Path(r["ckpt"]).exists() for r in runs.values())
    before = {k: v["ts"] for k, v in runs.items()}
    assert C.stage_sweep(ctx, 0, (0, 2)) == 0                               # resumed: nothing re-run
    assert {k: v["ts"] for k, v in S.read_merged("runs", "_smoke", "runs").items()} == before
    assert C.stage_probe(ctx, 1.0) == 0
    pr = json.loads((S.OUT / "probe_smoke.json").read_text())
    assert set(pr["projection"]) == {"main"} and pr["projection"]["main"]["ownorder"] == 0
    S.write(C.MAPS / "map_arith_smoke.json", {"L_on": 2})
    for t in cfg.table_tasks:
        assert C.stage_healedprobe(ctx, t) == 0
    d = json.loads(C.healed_probe_path(cfg, "arith").read_text())
    assert d["k"] == 0 and d["seed"] == 0 and d["n"] == 12 and d["L_on_unmodified"] == 2
    S.write(C.ORDER_PATH, {"shared_order": [2, 1]})
    S.write(C.BASELINES[0], {"tasks": {t: {"bow": 0.6, "chance": 0.5} for t in cfg.table_tasks}})
    assert C.stage_d3table(cfg) == 0
    s = json.loads((S.OUT / "summary_smoke.json").read_text())
    assert (S.OUT / "tables_smoke.md").exists() and s["cut"] == [1, 2] and s["kcut"] == 2
    assert {r["task"] for r in s["rows"]} == {"code_shape", "arith"}
    assert s["D3-A"]["label"] in ("replicates", "weak", "does not replicate", "undetermined")
    assert s["D3-X"]["label"] == "untestable"


def test_d3table_with_no_results(dirs, monkeypatch):
    monkeypatch.setattr(C, "OUT", dirs / "out")
    monkeypatch.setattr(C, "MAPS", dirs / "maps")
    monkeypatch.setattr(C, "ORDER_PATH", dirs / "maps" / "order.json")
    monkeypatch.setattr(C, "BASELINES", ())
    monkeypatch.setattr(C, "C1_OUT", dirs / "c1")
    monkeypatch.setattr(S, "OUT", dirs / "out")
    S.OUT.mkdir(parents=True)
    cfg = tiny_d3_cfg()
    with pytest.raises(SystemExit):
        C.stage_d3table(cfg)                                                # no shared order yet
    S.write(C.ORDER_PATH, {"shared_order": [3, 2, 1]})
    assert C.stage_d3table(cfg) == 0
    s = json.loads((S.OUT / "summary_smoke.json").read_text())
    assert s["D3-A"]["label"] == "undetermined" and s["D3-X"]["label"] == "untestable" and all(r["no_runs"] for r in s["rows"])


# ---------------------------------------------------------------- 24 blocks: own order and n_pre

def lesion24(drops):
    return {"task": "t", "rows": [{"block": b, "accuracy": 0.5, "drop": d} for b, d in enumerate(drops)]}


def test_own_order_with_24_blocks():
    drops = [0.9] * 24
    drops[0] = drops[23] = -1.0                                             # never candidates
    for b, d in ((5, 0.0), (3, 0.0), (20, 0.0), (9, 0.01), (8, 0.01), (2, -0.02)):
        drops[b] = d
    order = C.own_order(lesion24(drops), 24)
    assert order[:6] == [2, 3, 5, 20, 8, 9] and len(order) == 22 and 0 not in order and 23 not in order
    with pytest.raises(ValueError):
        C.own_order(lesion24(drops))                                        # 28-block default does not fit a 24-block sweep
    assert heal.alive_blocks(order, 10, 24)[0] == 0 and len(heal.alive_blocks(order, 10, 24)) == 14
    with pytest.raises(ValueError):
        heal.alive_blocks([23], 1, 24)


def test_n_pre_counts_cut_blocks_before_the_read_point():
    cut = list(range(10, 20))
    cfg = SimpleNamespace(ks=(0, 10), kcut=10, table_tasks=("arith", "rule"))
    runs = {}
    for t in cfg.table_tasks:
        for k, a in ((0, 0.9), (10, 0.7)):
            r = fake_row(t, k, 0, 3e-4, a, 0.7, chance=0.5)
            runs[r["key"]] = r
            r = fake_row(t, k, 0, 1e-3, a, 0.6, chance=0.5)
            runs[r["key"]] = r
    probes = {"arith": {"L_on_healed": 12, "never": False, "healed_accuracy": 0.9}, "rule": {"L_on_healed": 25, "never": True}}
    rows = {r["task"]: r for r in C.d3_task_rows(cfg, runs, probes, {}, {}, cut)}
    assert rows["arith"]["n_pre"] == 2 and rows["rule"]["n_pre"] == 10 and rows["rule"]["never"]
    probes["arith"]["L_on_healed"] = 10
    assert {r["task"]: r for r in C.d3_task_rows(cfg, runs, probes, {}, {}, cut)}["arith"]["n_pre"] == 0           # block 10 is not before read point 10
    assert rows["arith"]["r"] == pytest.approx((0.7 - 0.5) / (0.9 - 0.5))


# ---------------------------------------------------------------- the table on synthetic rows

def synth(depths, rs, kinds_gap=None, k0=0.9, chance=0.5):
    """Rows for all 24 table tasks via d3_task_rows. depths/rs: lists in table-task number order."""
    cfg = SimpleNamespace(ks=(0, 10), kcut=10, n_perm=200, smoke=False,
                          table_tasks=C.make_cfg(False, C.resolve_profile("smol17")).table_tasks)
    tasks = sorted(cfg.table_tasks, key=lambda t: spec.TASKS[t].number)
    runs, probes, maps, base = {}, {}, {}, {}
    for i, t in enumerate(tasks):
        for k, a in ((0, k0), (10, chance + rs[i] * (k0 - chance))):
            for seed in (0, 1, 2):
                r = fake_row(t, k, seed, 3e-4, a, 0.7, chance=chance)
                runs[r["key"]] = r
            r = fake_row(t, k, 0, 1e-3, a, 0.6, chance=chance)
            runs[r["key"]] = r
        probes[t] = {"L_on_healed": depths[i], "never": False, "healed_accuracy": k0}
        maps[t] = {"L_on": depths[i]}
        base[t] = {"bow": chance + (0.05 if kinds_gap is None else kinds_gap[i]), "chance": chance}
    return cfg, tasks, runs, probes, maps, base


def summarise(monkeypatch, depths, rs, cut=tuple(range(10, 20)), gap=None):
    monkeypatch.setattr(X, "N_PERM", 200)
    cfg, tasks, runs, probes, maps, base = synth(depths, rs, gap)
    rows = C.d3_task_rows(cfg, runs, probes, maps, base, list(cut))
    return C.build_summary_d3(cfg, rows, list(cut), {"arith": 0.5}, n_perm=200), rows


DEPTHS = list(range(2, 26))                                                   # 24 distinct depths, task order


def test_replicates_and_holds(monkeypatch):
    r_down = [round(1.0 - i * 0.04, 3) for i in range(24)]                  # deeper read point, lower R
    s, rows = summarise(monkeypatch, DEPTHS, r_down)
    assert s["D3-A"]["rho"] < -0.9 and s["D3-A"]["p"] < 0.05 and s["D3-A"]["label"] == "replicates" and s["D3-A"]["n"] == 24
    assert s["D3-X"]["n_zero"] == 9 and s["D3-X"]["label"] == "holds" and s["D3-X"]["rho"] <= -0.3   # depths 2..10 have n_pre 0
    assert [r["n_pre"] for r in rows][:10] == [0] * 9 + [1] and s["dropped"] == [] and s["excluded"]["missing"] == []
    assert s["d2_tasks"]["arith"] == {"smol_r": rows[[r["task"] for r in rows].index("arith")]["r"], "qwen_r12": 0.5}
    assert set(s["d2_tasks"]) == set(C.D2_DEEP + C.D2_SHALLOW) and s["judging_minus_doing"] is not None
    md = C.render_tables_d3(s)
    assert "**replicates**" in md and "**holds**" in md and "Judging minus doing" in md and "R12 Qwen3" in md


def test_does_not_replicate_and_not_shown(monkeypatch):
    r_up = [round(0.3 + i * 0.03, 3) for i in range(24)]
    s, _ = summarise(monkeypatch, DEPTHS, r_up)
    assert s["D3-A"]["rho"] > 0.9 and s["D3-A"]["label"] == "does not replicate"
    assert s["D3-X"]["n_zero"] == 9 and s["D3-X"]["label"] == "not shown"


def test_weak_when_the_surface_rival_is_as_strong(monkeypatch):
    r_down = [round(1.0 - i * 0.04, 3) for i in range(24)]
    gap = [0.01 + 0.01 * i for i in range(24)]                              # BoW gap grows with depth: rival rho == main rho
    s, _ = summarise(monkeypatch, DEPTHS, r_down, gap=gap)
    assert s["D3-A"]["rho"] < -0.9 and s["D3-A"]["p"] < 0.05
    assert abs(s["D3-A"]["rival_rho"]) == pytest.approx(abs(s["D3-A"]["rho"])) and s["D3-A"]["label"] == "weak"


def test_untestable_when_few_tasks_have_n_pre_zero(monkeypatch):
    r_down = [round(1.0 - i * 0.04, 3) for i in range(24)]
    s, _ = summarise(monkeypatch, DEPTHS, r_down, cut=tuple(range(3, 13)))        # n_pre = 0 only for depth 2 and 3
    assert s["D3-X"]["n_zero"] == 2 and s["D3-X"]["label"] == "untestable"
    s, _ = summarise(monkeypatch, DEPTHS, r_down, cut=tuple(range(5, 15)))         # depths 2..5: four tasks
    assert s["D3-X"]["n_zero"] == 4 and s["D3-X"]["label"] == "untestable"


def test_label_boundaries():
    L = C.d3_label_a
    assert L(-0.40, 0.01, 0.1) == "replicates" and L(-0.3999, 0.01, 0.1) == "weak"
    assert L(-0.60, 0.05, 0.1) == "weak" and L(-0.60, 0.049, 0.1) == "replicates"          # p < 0.05 strictly
    assert L(-0.60, 0.01, -0.60) == "weak" and L(-0.60, 0.01, 0.59) == "replicates" and L(-0.60, 0.01, None) == "replicates"
    assert L(-0.20, 0.5, 0.0) == "weak" and L(-0.1999, 0.5, 0.0) == "does not replicate" and L(0.3, 0.9, 0.0) == "does not replicate"
    assert L(None, None, None) == "undetermined" and L(float("nan"), None, None) == "undetermined"
    X_ = C.d3_label_x
    assert X_(8, -0.30) == "holds" and X_(8, -0.2999) == "not shown" and X_(7, -0.9) == "untestable"
    assert X_(12, None) == "not shown" and X_(12, float("nan")) == "not shown" and X_(0, None) == "untestable"


def test_drop_rule_at_chance_plus_020():
    assert C.d3_dropped(0.69, 0.5) and not C.d3_dropped(0.70, 0.5) and not C.d3_dropped(0.7001, 0.5)
    cfg = SimpleNamespace(ks=(0, 10), kcut=10, table_tasks=("arith", "rule", "hard"))
    runs = {}
    for t, k0 in (("arith", 0.69), ("rule", 0.70), ("hard", 0.95)):
        for k, a in ((0, k0), (10, 0.6)):
            for seed in (0, 1):
                r = fake_row(t, k, seed, 3e-4, a, 0.7, chance=0.5)
                runs[r["key"]] = r
            r = fake_row(t, k, 0, 1e-3, a, 0.6, chance=0.5)
            runs[r["key"]] = r
    rows = {r["task"]: r for r in C.d3_task_rows(cfg, runs, {}, {}, {}, [1, 2])}
    assert rows["arith"]["dropped"] and rows["arith"]["r"] is None
    assert not rows["rule"]["dropped"] and rows["rule"]["r"] == pytest.approx(0.1 / 0.2)
    assert rows["hard"]["r"] == pytest.approx(0.1 / 0.45) and rows["hard"]["healed_missing"]
    # a dropped task is out of every correlation and listed
    monkey_cfg = SimpleNamespace(ks=(0, 10), kcut=10, smoke=False, n_perm=50, table_tasks=cfg.table_tasks)
    s = C.build_summary_d3(monkey_cfg, list(rows.values()), [1, 2], None, n_perm=50)
    assert s["dropped"] == ["arith"] and "arith" not in s["D3-A"]["tasks"]
    assert s["no_L_on"] == ["rule", "hard"] and "Dropped" in C.render_tables_d3(s)


def test_r_is_clipped_to_zero_one():
    cfg = SimpleNamespace(ks=(0, 10), kcut=10, table_tasks=("arith", "rule"))
    runs = {}
    for t, a10 in (("arith", 0.3), ("rule", 1.0)):
        for k, a in ((0, 0.9), (10, a10)):
            r = fake_row(t, k, 0, 3e-4, a, 0.7, chance=0.5)
            runs[r["key"]] = r
            r = fake_row(t, k, 0, 1e-3, a, 0.6, chance=0.5)
            runs[r["key"]] = r
    rows = {r["task"]: r["r"] for r in C.d3_task_rows(cfg, runs, {}, {}, {}, [1])}
    assert rows == {"arith": 0.0, "rule": 1.0}


