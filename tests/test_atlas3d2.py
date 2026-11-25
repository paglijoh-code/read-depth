"""Step D2 (scripts/atlas3_cuts.py, docs/ATLAS-phase2-stageC.md "D2 bars"): label thresholds, lift arithmetic, own-k0
reporting, name disjointness from C1 and E2, the contiguous cut, the all-linear parameter count, and the three arms end to
end on a tiny random Qwen3. No real weights, no network."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import atlas3_cuts as C  # noqa: E402
import atlas_cuts as S  # noqa: E402
from test_atlas3c1 import make_ctx, tiny_cfg  # noqa: E402,F401
from test_atlas_heal import dirs, fake_row, tiny_model  # noqa: E402,F401

from surgery import adapters  # noqa: E402
from types import SimpleNamespace  # noqa: E402

QWEN17 = SimpleNamespace(hidden_size=2048, intermediate_size=6144, num_attention_heads=16, num_key_value_heads=8,
                         head_dim=128)


def test_label_thresholds_at_the_boundaries():
    assert C.d2_label(0.30, 0.9, 0.5) == "artifact" and C.d2_label(0.45 - 0.15, 0.1, 0.9) == "artifact"
    assert C.d2_label(0.2999, 0.1, 0.9) == "partial"
    assert C.d2_label(0.1499, 0.4, 0.5) == "holds"
    assert C.d2_label(0.45 - 0.30, 0.4, 0.5) == "partial"             # 0.15 (float noise) is not under 0.15
    assert C.d2_label(0.15, 0.4, 0.5) == "partial"
    assert C.d2_label(0.0, 0.5, 0.5) == "partial" and C.d2_label(0.0, 0.6, 0.5) == "partial"   # deep mean must be lower
    assert C.d2_label(-0.2, 0.2, 0.9) == "holds"
    assert C.d2_label(None, 0.2, 0.9) == "undetermined"


def c1_rows(tasks, k0=0.9, r12=0.5, chance=0.5):
    return {t: {"task": t, "chance": chance, "healed_k0": k0, "r12": r12} for t in tasks}


def arm_runs(arm, accs, k0=None, lrs=None):
    """accs {task: acc12}; fake rows: seeds 0-2 at the first lr, seed 0 only at the second (lrsel lower), and one k0 row per
    lr when k0 is given. The long arm has one lr."""
    lrs = lrs or ((3e-4,) if arm == "long" else (3e-4, 1e-3))
    runs = {}
    for t, a in accs.items():
        for j, lr in enumerate(lrs):
            for seed in (0, 1, 2) if j == 0 else (0,):
                r = fake_row(t, 12, seed, lr, a, 0.7 - 0.1 * j, chance=0.5)
                runs[r["key"]] = r
            if k0 is not None:
                r = fake_row(t, 0, 0, lr, k0 - 0.1 * j, 0.7 - 0.1 * j, chance=0.5)
                runs[r["key"]] = r
    return runs


def test_lift_arithmetic_on_fake_rows():
    deep, shallow = ("a", "b"), ("c", "d")
    c1 = c1_rows(deep + shallow, r12=0.2)
    # arm acc 0.8 -> R12 (0.8-0.5)/0.4 = 0.75 on deep; 0.58 -> 0.2 on shallow (no change)
    runs = arm_runs("alllin", {"a": 0.8, "b": 0.8, "c": 0.58, "d": 0.58})
    s = C.d2_arm_summary(runs, "alllin", c1, deep, shallow, 12)
    assert s["tasks"]["a"]["r12"] == pytest.approx(0.75) and s["tasks"]["a"]["delta"] == pytest.approx(0.55)
    assert s["lift_deep"] == pytest.approx(0.55) and s["lift_shallow"] == pytest.approx(0.0)
    assert s["diff"] == pytest.approx(0.55) and s["deep_mean"] == pytest.approx(0.75)
    assert s["shallow_mean"] == pytest.approx(0.2) and s["label"] == "artifact"
    s = C.d2_arm_summary(arm_runs("alllin", {"a": 0.6, "b": 0.6, "c": 0.7, "d": 0.7}), "alllin", c1, deep, shallow, 12)
    assert s["lift_deep"] == pytest.approx(0.25 - 0.2) and s["lift_shallow"] == pytest.approx(0.5 - 0.2)
    assert s["label"] == "holds"                                        # diff -0.25, deep 0.25 < shallow 0.5
    s = C.d2_arm_summary({}, "contig", c1, deep, shallow, 12)
    assert s["label"] == "undetermined" and s["tasks"]["a"]["r12"] is None


def test_winner_lr_rule_and_long_arm_single_lr():
    runs = arm_runs("alllin", {"a": 0.9})
    for r in runs.values():
        r["best_lrsel"] = 0.9 if r["lr"] == 1e-3 else 0.7
        r["sel_accuracy"] = 0.9 if r["lr"] == 1e-3 else 0.6
    assert C.d2_arm_acc(runs, "alllin", "a", 12) == (1e-3, [0.9])           # only seed 0 ran at the winner here
    one = arm_runs("long", {"a": 0.7}, lrs=(1e-3,))
    assert C.d2_arm_acc(one, "long", "a", 12) == (1e-3, [0.7, 0.7, 0.7])
    assert C.d2_arm_acc({}, "long", "a", 12) == (None, [])


def test_own_k0_reported_only_when_it_differs_by_more_than_005():
    c1 = c1_rows(["a"], k0=0.9)
    same = C.d2_task_row(arm_runs("alllin", {"a": 0.7}, k0=0.94), "alllin", "a", c1["a"], 12)
    assert same["arm_k0"] == pytest.approx(0.94) and not same["k0_differs"] and same["r12_own_k0"] is None
    edge = C.d2_task_row(arm_runs("alllin", {"a": 0.7}, k0=0.95), "alllin", "a", c1["a"], 12)
    assert not edge["k0_differs"]                                       # exactly 0.05 does not count
    far = C.d2_task_row(arm_runs("alllin", {"a": 0.7}, k0=0.7), "alllin", "a", c1["a"], 12)
    assert far["k0_differs"] and far["r12_own_k0"] == pytest.approx(1.0) and far["r12"] == pytest.approx(0.5)
    contig = C.d2_task_row(arm_runs("contig", {"a": 0.7}), "contig", "a", c1["a"], 12)
    assert contig["arm_k0"] is None and contig["r12_own_k0"] is None


def test_names_are_disjoint_from_c1_and_e2():
    cfg = C.make_cfg(False)
    names = {C.d2_ckpt_path(a, cfg, "arith", 12, 0, 3e-4).name for a in C.D2_ARMS}
    assert names == {f"d2_{a}_arith_k12_s0_lr0.0003.pt" for a in ("alllin", "contig", "long")}
    others = {C.e2_ckpt_path(cfg, "arith", 12, 0, 3e-4).name, C.own_ckpt_path(cfg, "arith", 12, 0, 3e-4).name,
              S.ckpt_path(cfg, "arith", 12, 0, 3e-4).name}
    assert not names & others
    prefixes = {C.d2_prefix(a) for a in C.D2_ARMS}
    assert len(prefixes) == 3 and not prefixes & {"runs", "ownorder", C.E2_PREFIX}
    assert C.D2_OUT == Path("var/results/atlas3/d2") and C.D2_OUT != C.OUT
    assert C.STAGES[:5] == ["probe", "sweep", "ownorder", "test", "table"] and C.STAGES[-1] == "healedprobe" and C.D2_STAGES == ["d2sweep", "d2table"]


def test_plan_constants_and_contiguous_cut():
    assert list(C.D2_CONTIG) == list(range(15, 27)) and len(C.D2_CONTIG) == 12
    assert C.D2_DEEP == ("code_output", "arith", "bind_compare", "rule")
    assert C.D2_SHALLOW == ("code_traces_first", "route_judgement")
    cfg = C.make_cfg(False)
    order = C.d2_contig_order(cfg, 28)
    assert order[:12] == list(range(15, 27)) and sorted(order) == list(range(1, 27))
    from surgery.atlas import heal
    assert heal.alive_blocks(order, 12, 28) == list(range(15)) + [27]
    arms = C.D2_ARMS
    assert (arms["alllin"].rank, arms["alllin"].mult, len(arms["alllin"].modules)) == (2, 1, 7)
    assert (arms["contig"].rank, arms["contig"].mult) == (10, 1) and (arms["long"].rank, arms["long"].mult) == (10, 4)
    assert C.make_cfg(False).steps * arms["long"].mult == 1200
    assert [len(C.d2_units(cfg, a, s)) for a in ("alllin", "contig", "long") for s in (0, 1)] == [12, 6, 6, 6, 12, 6]


def test_alllin_param_count_within_ten_percent():
    a, b = C.check_alllin_params(QWEN17)
    assert (a, b) == (77824, 71680) and abs(a - b) <= 0.10 * b


def test_block_params_match_add_lora_on_a_real_module_set():
    model = tiny_model()
    handle = adapters.add_lora(model, [0, 1], modules=C.D2_ARMS["alllin"].modules, r=2)
    n = sum(p.numel() for p in handle.modules.parameters() if p.requires_grad)
    assert n == 2 * C.block_params(model.config, C.D2_ARMS["alllin"].modules, 2)
    assert len(handle.modules) == 14
    handle.remove()
    assert not any(isinstance(m, adapters.LoRALinear) for m in model.modules())


def test_d2_lrs_follow_the_c1_rule():
    c1 = {"arith": {12: 1e-3, 0: 3e-4}}
    assert C.d2_lrs("long", {}, "arith", 12, 1, c1) == [1e-3] and C.d2_lrs("long", {}, "arith", 0, 0, c1) == [3e-4]
    assert C.d2_lrs("long", {}, "rule", 12, 0, c1) == []
    assert C.d2_lrs("alllin", {}, "arith", 12, 0, c1) == list(S.LRS) and C.d2_lrs("alllin", {}, "arith", 12, 1, c1) == []


def test_d2_arms_end_to_end_on_tiny_model(dirs, monkeypatch):
    monkeypatch.setattr(C, "C1_OUT", dirs / "c1")
    monkeypatch.setattr(C, "D2_OUT", dirs / "d2")
    monkeypatch.setattr(S, "OUT", dirs / "d2")
    S.OUT.mkdir(parents=True)
    cfg = tiny_cfg()
    cfg.d2_deep, cfg.d2_shallow, cfg.d2_contig = ("arith",), ("code_shape",), (2, 1)
    ctx = make_ctx(monkeypatch, cfg)
    S.write(dirs / "c1" / "summary_smoke.json", {"rows": [
        {"task": t, "chance": 0.3, "healed_k0": 0.9, "r12": 0.4, "lr": {"0": 3e-4, "2": 1e-3}} for t in ("arith", "code_shape")]})
    for seed in (0, 1, 2):
        for arm in ("long", "alllin", "contig"):
            assert C.stage_d2sweep(ctx, arm, seed, (0, 2)) == 0 and C.stage_d2sweep(ctx, arm, seed, (1, 2)) == 0
    for arm, expect in (("alllin", 2 * (2 + 1 + 1) + 2 * 2), ("contig", 2 * (2 + 1 + 1)), ("long", 2 * 3 + 2)):
        runs = S.read_merged(C.d2_prefix(arm), "_smoke", "runs")
        assert len(runs) == expect, arm
        for r in runs.values():
            assert f"d2_{arm}_" in r["ckpt"] and Path(r["ckpt"]).exists() and r["arm"] == arm
            assert r["steps"] == cfg.steps * C.D2_ARMS[arm].mult and r["rank"] == C.D2_ARMS[arm].rank
        if arm == "contig":
            assert all(r["removed"] == [2, 1] and r["n_alive"] == 2 for r in runs.values())
        if arm == "long":
            assert {r["lr"] for r in runs.values() if r["k"] == 2} == {1e-3}
    assert not S.read_merged("runs", "_smoke", "runs") and not S.read_merged("ownorder", "_smoke", "runs")
    before = {k: v["ts"] for k, v in S.read_merged(C.d2_prefix("alllin"), "_smoke", "runs").items()}
    assert C.stage_d2sweep(ctx, "alllin", 0, (0, 2)) == 0               # resumed: nothing re-run
    assert {k: v["ts"] for k, v in S.read_merged(C.d2_prefix("alllin"), "_smoke", "runs").items()} == before
    assert C.stage_d2table(cfg) == 0
    s = json.loads((S.OUT / "summary_smoke.json").read_text())
    assert set(s["arms"]) == {"alllin", "contig", "long"} and (S.OUT / "tables_smoke.md").exists()
    assert s["arms"]["contig"]["tasks"]["arith"]["n_seeds"] == 3 and s["arms"]["long"]["label"] in (
        "artifact", "holds", "partial")


def test_long_arm_without_c1_lr_reports_missing(dirs, monkeypatch):
    monkeypatch.setattr(C, "C1_OUT", dirs / "c1")
    monkeypatch.setattr(S, "OUT", dirs / "d2")
    S.OUT.mkdir(parents=True)
    cfg = tiny_cfg()
    cfg.smoke = False
    cfg.d2_deep, cfg.d2_shallow, cfg.d2_contig = ("arith",), ("code_shape",), (2, 1)
    assert C.stage_d2sweep(make_ctx(monkeypatch, cfg), "long", 0, (0, 1)) == 1
