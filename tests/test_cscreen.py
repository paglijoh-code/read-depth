from types import SimpleNamespace

import numpy as np

from surgery.atlas import cscreen as C
from surgery.atlas.p0_toy import ToyAdapter, make_toy_items

SEL = ["EleutherAI/pythia-160m", "HuggingFaceTB/SmolLM-135M"]


# ---------------------------------------------------------------- run_task on the toy
class FakeAd(ToyAdapter):
    tok = object()
    model_id = "toy"

    def encode(self, prompts):
        return [[int(t) for t in p.split()] for p in prompts]

    def filter_items(self, items):
        return [x for x in items if x["id"] != "drop-me"], sum(x["id"] == "drop-me" for x in items)

    def answer_logfreq(self, opt_ids):
        return {"lm": {"mean": -1.0}}


def _fam(n=240):
    ids, y, an, y_an, sf = make_toy_items(n, seed=1)
    items = []
    for i in range(n):
        j = lambda v: " ".join(map(str, v))
        items.append({"id": f"toy-{i}", "prompt": j(ids[i]), "answer": y[i], "template": f"t{i % 6}",
                      "split": "train" if i < 160 else "test", "ans_twin": {"prompt": j(an[i]), "answer": y_an[i]},
                      "surf_twin": {"prompt": j(sf[i])}})
    items.append({**items[0], "id": "drop-me"})
    task = SimpleNamespace(name="toy_a", family="toy", type_label="surface form", options=[" a", " b", " c", " d"])
    return SimpleNamespace(
        get_task=lambda n: task, generate=lambda n, seed=0: items,
        split_items=lambda it: ([x for x in it if x["split"] == "train"], [x for x in it if x["split"] == "test"]),
        rivals=lambda it, n: {"bow": 0.25, "near_miss": {"r": 0.3}, "position": None},
        check_options=lambda tok, t: [1, 2, 3, 4],
        tokcheck=SimpleNamespace(check_items=lambda it, o, toks: {"bad_ids": ["toy-3", "toy-4"], "ok": False}))


def test_run_task_toy():
    fam = _fam()
    r = C.run_task(FakeAd(), "toy_a", fam, toks={"x": 1}, n_perm=2)
    for k in ("probe", "twin", "swapped", "probe_onset", "early_read", "answer_logfreq", "timing", "rivals", "family",
              "type_label", "n_blocks", "model", "task", "dropped_alignment", "dropped_filter"):
        assert k in r
    assert "fits" not in r["probe"]
    assert set(r["timing"]) == {"states_s", "probe_s", "twin_s", "total_s"}
    assert r["dropped_alignment"] == 2 and r["dropped_filter"] == 1 and r["n_blocks"] == 8
    assert r["probe"]["valid"] and r["probe"]["peak_selectivity"] > 0.5
    assert np.allclose(r["swapped"]["t"], -np.array(r["twin"]["t"]))
    assert r["twin"]["peak"] == -r["swapped"]["trough"]
    assert r["twin"]["valid_sym"] != r["swapped"]["valid_sym"] or not r["twin"]["valid_sym"]
    import json
    json.dumps(r, default=lambda o: o.item())
    assert C.run_task(FakeAd(), "toy_a", fam, toks=None, n_perm=2)["dropped_alignment"] == 0


def test_early_read_flag():
    ok = {"valid": True}
    assert C.early_read(ok, {"rel": 0.1, "layers": 3})
    assert C.early_read(ok, {"rel": 0.5, "layers": 1.0})
    assert not C.early_read(ok, {"rel": 0.11, "layers": 1.5})
    assert not C.early_read({"valid": False}, {"rel": 0.0, "layers": 0})
    assert not C.early_read(ok, None)


# ---------------------------------------------------------------- verdicts
def R(sel=0.5, valid=True, early=False, bow=0.25, nm=None, pos=None, drop_a=0, drop_f=0, peak=0.2, trough=-0.05,
      swap_valid=False):
    return {"probe": {"valid": valid, "peak_selectivity": sel}, "early_read": early,
            "rivals": {"bow": bow, "near_miss": nm or {"a": 0.3, "b": 0.2}, "position": pos},
            "dropped_alignment": drop_a, "dropped_filter": drop_f,
            "twin": {"peak": peak, "trough": trough}, "swapped": {"valid_sym": swap_valid}}


FAM = SimpleNamespace(FAMILIES={"f1": ("surface form", ("f1_a", "f1_b"))})


def V(per_task):
    """per_task = {task: (run_m1, run_m2)}"""
    res = {m: {t: r[i] for t, r in per_task.items()} for i, m in enumerate(SEL)}
    return C.verdicts(res, FAM)


def good():
    return {"f1_a": (R(), R()), "f1_b": (R(), R())}


def test_verdict_kept_and_signal():
    v = V(good())
    f = v["families"]["f1"]
    assert f["kept"] and f["kept_strict"] and f["reasons"] == [] and abs(f["signal"] - 0.5) < 1e-9
    assert v["tasks"]["f1_a"]["twins_ok"] and v["tasks"]["f1_a"]["probe_valid_models"] == SEL


def test_verdict_rejections():
    cases = {
        "rivals_ok": dict(bow=0.5), "aligned_ok": dict(drop_a=1), "twin_peak_gt_dip": dict(peak=0.1, trough=-0.2),
        "swap_fails": dict(swap_valid=True),
    }
    for key, kw in cases.items():
        g = good()
        g["f1_a"] = (R(**kw), R())
        v = V(g)
        assert not v["tasks"]["f1_a"][key], key
        assert not v["families"]["f1"]["kept"]
        assert any(key in r for r in v["families"]["f1"]["reasons"]), key
    g = good()
    g["f1_a"] = (R(nm={"a": 0.40}), R())
    assert not V(g)["tasks"]["f1_a"]["rivals_ok"]
    g["f1_a"] = (R(pos=0.05), R())
    assert not V(g)["tasks"]["f1_a"]["rivals_ok"]
    g["f1_a"] = (R(drop_f=2), R())
    assert not V(g)["tasks"]["f1_a"]["aligned_ok"]
    g["f1_a"] = (R(), {"error": "boom"})
    v = V(g)
    assert not v["families"]["f1"]["kept"] and any("error" in r for r in v["families"]["f1"]["reasons"])


def test_kept_vs_strict_and_none_valid():
    g = good()
    g["f1_b"] = (R(valid=False, sel=0.1), R(valid=False, sel=0.0))
    f = V(g)["families"]["f1"]
    assert f["kept"] and not f["kept_strict"]
    g["f1_a"] = (R(valid=False), R(valid=False))
    f = V(g)["families"]["f1"]
    assert not f["kept"] and "probe_valid_nowhere" in f["reasons"]


def test_early_read_rule():
    g = good()
    g["f1_a"] = (R(early=True), R(early=False))  # 1 of 2 valid -> flagged
    g["f1_b"] = (R(early=True, valid=False), R(early=False))  # valid only where not early -> not flagged
    v = V(g)
    assert v["tasks"]["f1_a"]["early_read"] and not v["tasks"]["f1_b"]["early_read"]
    assert v["families"]["f1"]["early_tasks"] == 1
    g["f1_a"] = (R(early=True, valid=False), R(valid=False))
    assert not V(g)["tasks"]["f1_a"]["early_read"]  # valid nowhere


# ---------------------------------------------------------------- pick
TYPES = ["surface form", "fact lookup", "binding/reference", "arithmetic", "code tracing", "rule application",
         "multi-hop composition"]


def synth(n_per_type=6, early=(), kept_off=(), sig=None):
    fv, i = {}, 0
    for ty in TYPES:
        for k in range(n_per_type):
            f = f"{ty[:3]}{k}"
            fv[f] = {"type_label": ty, "kept": f not in kept_off, "kept_strict": True,
                     "signal": (sig or {}).get(f, 1.0 - 0.01 * k), "early_tasks": 2 if f in early else 0}
            i += 1
    return fv


def test_pick_stratified_and_deterministic():
    fv = synth(6)  # 42 kept
    p = C.pick(fv, seed=3)
    assert p["status"] == "OK" and len(p["main"]) == 30 and len(p["reserve"]) == 6
    assert not set(p["main"]) & set(p["reserve"])
    counts = list(p["per_type"]["main"].values())
    assert max(counts) - min(counts) <= 1
    # best-signal first within a type: k=0 is always taken before k=1
    for ty in TYPES:
        ks = sorted(int(f[3:]) for f in p["main"] if f.startswith(ty[:3]))
        assert ks == list(range(len(ks)))
    assert C.pick(fv, seed=3) == p
    assert C.pick(fv, seed=4)["type_order"] != p["type_order"] or C.pick(fv, seed=5)["type_order"] != p["type_order"]


def test_pick_ties_seeded():
    fv = synth(6, sig={f: 1.0 for f in synth(6)})
    a, b = C.pick(fv, seed=0), C.pick(fv, seed=1)
    assert a["main"] != b["main"] or a["reserve"] != b["reserve"]
    assert C.pick(fv, seed=0) == a


def test_pick_early_cap():
    fv0 = synth(6)
    early = {f for f in fv0 if f.endswith("0")}  # the best family of each type is an early reader
    p = C.pick(synth(6, early=early), seed=0)
    assert p["early_read_tasks"]["main"] <= 9 and p["early_read_tasks"]["reserve"] <= 1
    assert sum(2 for f in p["main"] if f in early) == p["early_read_tasks"]["main"]
    assert p["status"] == "OK"


def test_pick_status():
    off = lambda n: {f for f in list(synth(6))[:n]}
    assert C.pick(synth(6, kept_off=off(6)))["status"] == "OK"  # 36 kept
    p = C.pick(synth(6, kept_off=off(12)))  # 30 kept: main shrinks to 24, reserve 6
    assert p["status"] == "OK_MIN" and len(p["main"]) == 24 and len(p["reserve"]) == 6
    p = C.pick(synth(6, kept_off=off(14)))  # 28 kept
    assert p["status"] == "SHORT"
    p = C.pick(synth(6, kept_off=set(synth(6))))
    assert p["status"] == "SHORT" and p["main"] == [] and p["reserve"] == []


# ---------------------------------------------------------------- v13
BIG = "HuggingFaceTB/SmolLM-1.7B"


def V13(per_task, big=None, fresh=None, art=None):
    res = {m: {t: r[i] for t, r in per_task.items()} for i, m in enumerate(SEL)}
    pr = {BIG: big} if big is not None else {}
    if fresh is None:
        fresh = {t: {m: {"rivals": {"bow": 0.25, "near_miss": {"a": 0.26}, "position": None}} for m in SEL}
                 for t in per_task}
    return C.verdicts_v13(res, FAM, probe_results=pr, fresh_rivals=fresh, artefacts=art)


def test_v13_probe_only_model_counts():
    g = good()
    g["f1_a"] = (R(valid=False), R(valid=False))
    g["f1_b"] = (R(valid=False), R(valid=False))
    assert not V13(g)["families"]["f1"]["kept"]
    big = {"f1_a": {"probe": {"valid": True, "peak_selectivity": 0.6}, "early_read": False}, "f1_b": {"error": "x"}}
    f = V13(g, big=big)["families"]["f1"]
    assert f["kept"] and not f["kept_strict"] and f["valid_only_on_probe_models"] and not f["kept_v12"]
    # twins and rivals still come from the small models only
    g["f1_b"] = (R(valid=False, swap_valid=True), R(valid=False))
    assert not V13(g, big=big)["families"]["f1"]["kept"]


def test_v13_fresh_rivals_and_artefact():
    fresh = {t: {m: {"rivals": {"bow": 0.25, "near_miss": {"a": 0.26}, "position": None}} for m in SEL}
             for t in ("f1_a", "f1_b")}
    fresh["f1_b"][SEL[1]]["rivals"]["near_miss"] = {"a": 0.40}
    f = V13(good(), fresh=fresh)["families"]["f1"]
    assert not f["kept"] and any("fresh_rivals" in r for r in f["reasons"])
    assert not V13(good(), fresh={})["families"]["f1"]["kept"]  # missing fresh rivals -> not kept
    g = good()
    g["f1_a"] = (R(), R(valid=False))
    g["f1_b"] = (R(), R(valid=False))
    art = {"f1_a": {"pythia-160m": {"artefact": True}}, "f1_b": {"pythia-160m": {"artefact": True}}}
    f = V13(g, art=art)["families"]["f1"]
    assert not f["kept"] and "probe_valid_nowhere" in f["reasons"]
    assert V13(g, art={"f1_a": {"pythia-160m": {"artefact": True}}})["families"]["f1"]["kept"]


def test_pick_v13_joint_cap_and_skips():
    fv0 = synth(6)
    early = {f for f in fv0 if f[3:] == "0"}  # 7 families x 2 early tasks: 5 fit the joint cap of 10, 2 are skipped
    p = C.pick_v13(synth(6, early=early), seed=0)
    assert p["early_cap_tasks"] == 10 and p["early_read_tasks"]["total"] <= 10
    assert p["early_read_tasks"]["total"] == 2 * sum(1 for f in p["main"] + p["reserve"] if f in early)
    assert len(p["skipped_early"]) == 2 and all(f in early for f in p["skipped_early"])
    assert not set(p["skipped_early"]) & set(p["main"] + p["reserve"])
    assert p["status"] == "OK" and len(p["main"]) == 30 and len(p["reserve"]) == 6
    assert C.pick_v13(synth(6, early=early), seed=0) == p
    q = C.pick_v13(synth(6, kept_off={f for f in list(fv0)[:14]}))  # 28 kept
    assert q["status"] == "SHORT"
