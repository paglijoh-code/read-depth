"""Step D1 (docs/ATLAS-phase2-stageC.md): the `healedprobe` stage of scripts/atlas3_cuts.py on a tiny random Qwen3 and the
pure D1 functions of scripts/atlas3_extra.py on synthetic rows. No real weights, no network."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import atlas3_cuts as C  # noqa: E402
import atlas3_extra as X  # noqa: E402
import atlas_cuts as S  # noqa: E402
from test_atlas3c1 import make_ctx, tiny_cfg  # noqa: E402,F401
from test_atlas_heal import dirs  # noqa: E402,F401


# ---------------------------------------------------------------- stage

def test_stage_registered_last_and_path_name():
    assert C.STAGES[-1] == "healedprobe" and C.STAGES[:5] == ["probe", "sweep", "ownorder", "test", "table"]
    assert C.healed_probe_path(C.make_cfg(False), "arith").name == "healed_probe_arith.json"
    assert C.healed_probe_path(C.make_cfg(True), "arith").name == "healed_probe_arith_smoke.json"
    big, small = C.make_cfg(False), C.make_cfg(True)
    assert (big.boots, big.probe_records, small.boots, small.probe_records) == (500, None, 20, 64)


def test_stage_resumes_and_reports_missing_run(dirs, monkeypatch):
    monkeypatch.setattr(C, "OUT", dirs / "c1")
    monkeypatch.setattr(S, "OUT", dirs / "c1")
    S.OUT.mkdir(parents=True)
    cfg = C.make_cfg(True)
    ctx = S.Ctx.__new__(S.Ctx)                  # nothing may be loaded or touched
    ctx.cfg = cfg
    assert C.stage_healedprobe(ctx, "arith") == 1           # no k=0 seed-0 run
    S.write(C.healed_probe_path(cfg, "arith"), {"task": "arith"})
    assert C.stage_healedprobe(ctx, "arith") == 0           # final file exists


def test_stage_on_tiny_model(dirs, monkeypatch):
    monkeypatch.setattr(C, "OUT", dirs / "c1")
    monkeypatch.setattr(C, "MAPS", dirs / "maps")
    monkeypatch.setattr(C, "BASELINES", ())
    monkeypatch.setattr(S, "OUT", dirs / "c1")
    monkeypatch.setattr(S, "ACTS", dirs / "acts")
    S.OUT.mkdir(parents=True)
    cfg = tiny_cfg()
    ctx = make_ctx(monkeypatch, cfg)
    for seed in (0,):
        assert C.stage_sweep(ctx, seed, (0, 1)) == 0
    S.write(C.MAPS / "map_arith.json", {"L_on": 2})
    assert C.stage_healedprobe(ctx, "arith") == 0
    d = json.loads(C.healed_probe_path(cfg, "arith").read_text())
    assert d["k"] == 0 and d["seed"] == 0 and d["n"] == 12 and d["L_on_unmodified"] == 2 and d["smoke"] is True
    assert d["scoring"] == "option first-token accuracy" and Path(d["ckpt"]).exists()
    assert set(d) >= {"task", "lr", "chance", "healed_accuracy", "curve", "lo", "hi", "L_on_healed", "never", "ts"}
    assert not any((dirs / "acts").glob("*healed_probe*"))
    ts = d["ts"]
    assert C.stage_healedprobe(ctx, "arith") == 0
    assert json.loads(C.healed_probe_path(cfg, "arith").read_text())["ts"] == ts
    C.stage_healedprobe(ctx, "code_shape")
    assert json.loads(C.healed_probe_path(cfg, "code_shape").read_text())["L_on_unmodified"] is None


# ---------------------------------------------------------------- D1 functions

def test_main_untouched():
    assert callable(X.main) and callable(X.x1) and callable(X.h2b) and callable(X.d1_main)


def test_compare_thresholds_at_the_boundary():
    assert X.d1_compare(-0.60, -0.50) == "sharper" and X.d1_compare(-0.599, -0.50) == "same"
    assert X.d1_compare(-0.40, -0.50) == "weaker" and X.d1_compare(-0.401, -0.50) == "same"
    assert X.d1_compare(-0.50, -0.50) == "same" and X.d1_compare(None, -0.5) is None


def mk(task, kind, lon, healed, r12, never=False, gap=0.1, missing=False):
    return {"task": task, "kind": kind, "L_on": lon, "L_on_healed": healed, "never": never, "healed_accuracy": 0.9,
            "r12": r12, "surface": {"bow": 0.5, "chance": 0.4, "gap": gap}, **({"healed_missing": True} if missing else {})}


def synth(n=8, healed_rev=False):
    rows = []
    for i in range(n):
        kind = "appraise" if i % 2 == 0 else "execute"
        lon = 2 + 2 * (i // 2)
        r12 = 1.0 - 0.2 * (i // 2)
        rows.append(mk(f"t{i}", kind, lon, (20 - lon) if healed_rev else lon, r12, gap=0.05 * (i % 3)))
    return rows


def test_split_excludes_never_none_and_missing():
    rows = synth(4) + [mk("n1", "appraise", 3, 28, 0.5, never=True), mk("n2", "execute", 3, None, 0.5),
                       mk("n3", "execute", 3, None, 0.5, missing=True)]
    usable, ex = X.d1_split(rows)
    assert len(usable) == 4 and ex == {"never": ["n1"], "none": ["n2"], "missing": ["n3"]}
    d = X.d1_size(rows, [0, 1, 2], "1.7B", 50)
    assert d["n_usable"] == 4 and d["n_rows"] == 7 and d["x1"]["n_tasks"] == 4
    assert all(p["L_on"] == q["L_on_healed"] for p, q in zip(d["x1"]["per_task"], usable))


def test_size_sharper_and_weaker_with_synthetic_rows():
    same = X.d1_size(synth(8), [0, 1, 2], "m", 50)
    assert same["healed"]["rho"] == pytest.approx(same["unmod_same"]["rho"]) and same["compare"] == "same"
    flipped = X.d1_size(synth(8, healed_rev=True), [0, 1, 2], "m", 50)
    assert flipped["compare"] == "weaker" and flipped["healed"]["rho"] > flipped["unmod_same"]["rho"]


def fake(label, compare, rho, missing=()):
    return {"label": label, "compare": compare, "healed": {"rho": rho}, "n_rows": 24,
            "excluded": {"never": [], "none": [], "missing": list(missing)}}


def test_decision_branches():
    ok6, weak6 = fake("supported", "same", -0.6), fake("supported", "weaker", -0.6)
    assert X.d1_decision(fake("supported", "sharper", -0.7), ok6)[0] == "healed_primary"
    assert X.d1_decision(fake("supported", "same", -0.7), ok6)[0] == "healed_primary"
    assert X.d1_decision(fake("partial", "weaker", -0.3), ok6)[0] == "unmodified_primary_report_both"
    assert X.d1_decision(fake("partial", "same", -0.1), ok6)[0] == "depth_claim_unmodified_only_reconsider_D3"
    assert X.d1_decision(fake("not_supported", "weaker", -0.1), ok6)[0] == "depth_claim_unmodified_only_reconsider_D3"
    assert X.d1_decision(fake("supported", "same", -0.7), weak6)[0] == "no_rule_matched"      # supported, but 0.6B weaker
    assert X.d1_decision(fake("partial", "same", -0.3), ok6)[0] == "no_rule_matched"


def test_decision_incomplete_and_markdown():
    d17 = X.d1_size(synth(4) + [mk("m", "execute", 3, None, 0.5, missing=True)], [0, 1], "1.7B", 20)
    d06 = X.d1_size(synth(6), [0, 1], "0.6B", 20)
    dec, text = X.d1_decision(d17, d06)
    assert dec is None and text.startswith("INCOMPLETE 4/5")
    md = X.d1_markdown(d17, d06, synth(4), synth(6), dec, text)
    assert "test (blind)" in md and "description (already seen)" in md and "decision: null" in md and "INCOMPLETE 4/5" in md
    assert "| task | kind | L_on | L_on_healed | never | healed acc | n_pre (healed) | R12 |" in md
