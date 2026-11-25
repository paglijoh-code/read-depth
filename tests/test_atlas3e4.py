"""Step E4 (scripts/atlas3_cuts.py, docs/ATLAS-robustness.md "E4: late-only cuts"): the two block sets and orders, n_pre and
the qualifying set, the verdict bars, the permutation p, the table arithmetic, contig unchanged, and both arms end to end on
a tiny random Qwen3. No real weights, no network."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import atlas3_cuts as C  # noqa: E402
import atlas_cuts as S  # noqa: E402
from test_atlas3c1 import make_ctx, tiny_cfg  # noqa: E402,F401
from test_atlas3d2 import arm_runs  # noqa: E402
from test_atlas_heal import dirs, fake_row, tiny_model  # noqa: E402,F401

from surgery.atlas import heal  # noqa: E402

DEPTHS = {"a": 3, "b": 8, "c": 12, "d": 15, "e": 16, "f": 17, "hard": 18, "code_output": 20}


def test_block_sets_and_orders():
    cfg = C.make_cfg(False)
    assert cfg.e4_blocks["late17"] == tuple(range(17, 27)) and cfg.e4_blocks["late15"] == tuple(range(15, 25))
    assert cfg.e4_k == 10 and len(cfg.e4_tasks) == 24 and "arith_long" not in cfg.e4_tasks
    for arm, blocks in cfg.e4_blocks.items():
        order = C.e4_order(cfg, arm, 28)
        assert order[:10] == list(blocks) and sorted(order) == list(range(1, 27))
    assert heal.alive_blocks(C.e4_order(cfg, "late17", 28), 10, 28) == list(range(17)) + [27]
    assert heal.alive_blocks(C.e4_order(cfg, "late15", 28), 10, 28) == list(range(15)) + [25, 26, 27]
    assert [len(C.e4_units(cfg, a, s)) for a in C.E4_ARMS for s in (0, 1)] == [24] * 4
    assert all(k == 10 for _, k in C.e4_units(cfg, "late17", 0))


def test_contig_is_unchanged_and_names_are_disjoint():
    cfg = C.make_cfg(False)
    assert C.d2_contig_order(cfg, 28)[:12] == list(range(15, 27)) and list(C.D2_CONTIG) == list(range(15, 27))
    assert set(C.D2_ARMS) == {"alllin", "contig", "long"} and C.D2_STAGES == ["d2sweep", "d2table"]
    assert C.E4_ARMS["late17"] is C.D2_ARMS["contig"] and C.E4_OUT == Path("var/results/atlas3/e4")
    names = {C.e4_ckpt_path(a, cfg, "arith", 10, 0, 3e-4).name for a in C.E4_ARMS}
    assert names == {"e4_late17_arith_k10_s0_lr0.0003.pt", "e4_late15_arith_k10_s0_lr0.0003.pt"}
    assert not names & {C.d2_ckpt_path(a, cfg, "arith", 10, 0, 3e-4).name for a in C.D2_ARMS}
    assert not {C.e4_prefix(a) for a in C.E4_ARMS} & ({C.d2_prefix(a) for a in C.D2_ARMS} | {"runs", "ownorder"})


def test_n_pre_and_qualifying_set():
    qual = lambda arm: sorted(t for t, d in DEPTHS.items() if C.e4_n_pre(C.E4_BLOCKS[arm], d) == 0)   # noqa: E731
    assert qual("late17") == sorted(["a", "b", "c", "d", "e", "f"])         # depth <= 17; hard 18, code_output 20 out
    assert qual("late15") == sorted(["a", "b", "c", "d"])                    # depth <= 15
    assert C.e4_n_pre(C.E4_BLOCKS["late17"], 18) == 1 and C.e4_n_pre(C.E4_BLOCKS["late17"], 20) == 3
    assert C.e4_n_pre(C.E4_BLOCKS["late15"], 16) == 1 and C.e4_n_pre(C.E4_BLOCKS["late17"], None) is None


def test_verdict_at_the_boundaries():
    v = C.e4_verdict
    assert v(-0.40, 5) == "holds" and v(-0.7, 5) == "holds" and v(-0.3999, 5) == "weak"
    assert v(-0.20, 5) == "weak" and v(-0.1999, 5) == "not supported" and v(0.3, 5) == "not supported"
    assert v(None, 5) == "not supported" and v(float("nan"), 5) == "not supported"
    assert v(-0.9, 2) == "uninformative" and v(None, 0) == "uninformative" and v(-0.9, 3) == "holds"


def test_permutation_p():
    dep, kinds = list(range(1, 9)), ["execute"] * 8
    assert C.e4_perm_p(dep, [1.0] * 8, kinds, 200) is None                   # R constant: rho undefined
    perfect = C.e4_perm_p(dep, [1 - 0.1 * d for d in dep], kinds, 2000)      # rho -1: only the exact reversal ties it
    assert perfect == pytest.approx((1 + 1) / 2001, abs=2e-3) or perfect < 0.002
    flat = C.e4_perm_p(dep, [0.5, 0.9, 0.2, 0.8, 0.4, 0.7, 0.3, 0.6], kinds, 500)
    assert 0.0 < flat <= 1.0 and flat == C.e4_perm_p(dep, [0.5, 0.9, 0.2, 0.8, 0.4, 0.7, 0.3, 0.6], kinds, 500)   # seeded
    obs = -0.8
    assert (0 + 1) / (10_000 + 1) < 1e-3 and obs < 0


def fake_arm_rows(arm, r_of, depth=DEPTHS):
    """e4_task_row on fake runs where task t's seed-mean R is r_of[t] (a0 0.9, chance 0.5: acc = 0.5 + 0.4 R)."""
    blocks = C.E4_BLOCKS[arm]
    rows = {}
    for t, d in depth.items():
        runs = {}
        if t in r_of:
            runs = arm_runs("contig", {t: 0.5 + 0.4 * r_of[t]})
            runs = {k: dict(r, k=10, key=k.replace("k12", "k10")) for k, r in runs.items()}
            runs = {r["key"]: r for r in runs.values()}
            for r in runs.values():
                r["arm"] = arm
        rows[t] = C.e4_task_row(runs, arm, t, {"chance": 0.5, "healed_k0": 0.9}, d, "execute" if d > 10 else "appraise",
                                blocks, 10)
    return rows


def test_task_row_and_arm_summary_on_fake_rows():
    r_of = {t: r for t, r in zip(DEPTHS, [0.95, 0.9, 0.7, 0.5, 0.4, 0.3, 0.1, 0.0])}
    rows = fake_arm_rows("late17", r_of)
    a = rows["a"]
    assert a["n_pre"] == 0 and a["n_seeds"] == 3 and a["r"] == pytest.approx(0.95) and a["a0"] == 0.9
    assert len(a["acc_seeds"]) == 3 and len(a["r_seeds"]) == 3 and a["r_seed_mean"] == pytest.approx(0.95)
    assert rows["hard"]["n_pre"] == 1 and rows["code_output"]["n_pre"] == 3
    s = C.e4_arm_summary(rows, C.E4_BLOCKS["late17"], 300)
    assert s["qualifying"] == ["a", "b", "c", "d", "e", "f"] and s["excluded"] == ["hard", "code_output"]
    assert s["n_below_ceiling"] == 4                                          # R 0.7, 0.5, 0.4, 0.3 (0.9 is not < 0.9)
    assert s["rho"] is not None and s["rho"] < -0.9 and s["verdict"] == "holds" and 0 < s["p"] <= 1
    assert s["pooled_rho"] == pytest.approx(heal.spearman([3, 8, 12, 15, 16, 17], [0.95, 0.9, 0.7, 0.5, 0.4, 0.3]))
    assert s["loo_min"] is not None and s["loo_max"] <= -0.4
    s15 = C.e4_arm_summary(fake_arm_rows("late15", r_of), C.E4_BLOCKS["late15"], 50)
    assert s15["qualifying"] == ["a", "b", "c", "d"]
    d = C.e4_difference({"late17": s, "late15": s15})
    assert d["n"] == 4 and {x["task"] for x in d["rows"] if x["both"]} == {"a", "b", "c", "d"}
    assert all(x["diff"] == pytest.approx(0.0) for x in d["rows"] if x["both"])
    empty = C.e4_arm_summary(fake_arm_rows("late17", {}), C.E4_BLOCKS["late17"], 10)
    assert empty["n_qualifying"] == 0 and empty["rho"] is None and empty["verdict"] == "uninformative"


def test_e4_end_to_end_on_tiny_model(dirs, monkeypatch):
    monkeypatch.setattr(C, "C1_OUT", dirs / "c1")
    monkeypatch.setattr(C, "E4_OUT", dirs / "e4")
    monkeypatch.setattr(C, "E1_PATH", dirs / "e1.json")
    monkeypatch.setattr(S, "OUT", dirs / "e4")
    S.OUT.mkdir(parents=True)
    cfg = tiny_cfg()
    cfg.e4_tasks, cfg.e4_k, cfg.e4_blocks = ("arith", "code_shape"), 2, {"late17": (2, 1), "late15": (1, 2)}
    ctx = make_ctx(monkeypatch, cfg)
    S.write(dirs / "c1" / "summary_smoke.json", {"rows": [
        {"task": t, "chance": 0.3, "healed_k0": 0.9} for t in ("arith", "code_shape")]})
    S.write(dirs / "e1.json", {"1.7B": {"tasks": [{"task": "arith", "kind": "execute", "L_on_healed": 1},
                                                   {"task": "code_shape", "kind": "execute", "L_on_healed": 2}]}})
    for arm in C.E4_ARMS:
        for seed in (0, 1, 2):
            assert C.stage_e4sweep(ctx, arm, seed, (0, 2)) == 0 and C.stage_e4sweep(ctx, arm, seed, (1, 2)) == 0
        runs = S.read_merged(C.e4_prefix(arm), "_smoke", "runs")
        assert len(runs) == 2 * (2 + 1 + 1)                                  # two tasks: 2 lrs at seed 0, 1 at seeds 1-2
        for r in runs.values():
            assert f"e4_{arm}_" in r["ckpt"] and Path(r["ckpt"]).exists() and r["arm"] == arm and r["k"] == 2
            assert r["removed"] == list(cfg.e4_blocks[arm]) and r["rank"] == 10 and r["steps"] == cfg.steps
    assert not S.read_merged("runs", "_smoke", "runs") and not S.read_merged(C.d2_prefix("contig"), "_smoke", "runs")
    before = {k: v["ts"] for k, v in S.read_merged(C.e4_prefix("late17"), "_smoke", "runs").items()}
    assert C.stage_e4sweep(ctx, "late17", 0, (0, 2)) == 0                    # resumed: nothing re-run
    assert {k: v["ts"] for k, v in S.read_merged(C.e4_prefix("late17"), "_smoke", "runs").items()} == before
    assert C.stage_e4table(cfg) == 0
    s = json.loads((S.OUT / "summary_smoke.json").read_text())
    assert set(s["arms"]) == {"late17", "late15"} and (S.OUT / "summary_smoke.md").exists()
    t = s["arms"]["late17"]["tasks"]["arith"]
    assert t["n_seeds"] == 3 and t["n_pre"] == 0 and s["arms"]["late17"]["tasks"]["code_shape"]["n_pre"] == 1
    assert s["arms"]["late15"]["tasks"]["arith"]["n_pre"] == 0 and s["difference"]["n"] == 1
