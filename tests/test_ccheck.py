import types

import numpy as np
import pytest
from scipy.stats import binom

from surgery.atlas import ccheck as C
from surgery.atlas import p0_measures as M


def _blocks(n, nb, signal_from, rng, amp=3.0):
    y = np.arange(n) % 4
    S = rng.standard_normal((n, nb, 12)).astype(np.float32)
    for b in range(signal_from, nb):
        S[:, b, :3] += amp * np.eye(4)[y][:, :3] - amp / 4
    return S, y


@pytest.fixture(scope="module")
def sig():
    rng = np.random.default_rng(0)
    S, y = _blocks(600, 5, 2, rng)
    return S, y, np.arange(600) % 10


def _pc(sig, y_tr=None, **kw):
    S, y, g = sig
    return M.probe_curve(S[:400], y[:400] if y_tr is None else y_tr, g[:400], S[400:], y[400:], seed=0, **kw)


def test_control_is_permutation_mean_and_selectivity(sig):
    r = _pc(sig, n_perm=8)
    assert r["n_perm"] == 8 and r["test"] == "binomial_bonferroni" and r["bow_acc"] is None
    assert np.allclose(r["control_acc"], r["perm_mean"])
    assert np.all(np.abs(np.array(r["perm_mean"]) - 0.25) < 0.1)
    acc, ctrl = np.array(r["acc"]), np.array(r["control_acc"])
    assert np.allclose(r["selectivity"], (acc - ctrl) / 0.75)
    assert r["peak_block"] == int(np.argmax(r["selectivity"])) and r["fit_s"] > 0 and len(r["fits"]) == 5
    assert r["valid"] and r["p_perm"] is not None


def test_bow_raises_control_and_lowers_selectivity(sig):
    a, b = _pc(sig, n_perm=4), _pc(sig, n_perm=4, bow_acc=0.6)
    assert np.allclose(b["control_acc"], np.maximum(b["perm_mean"], 0.6)) and b["bow_acc"] == 0.6
    assert b["peak_selectivity"] < a["peak_selectivity"]
    assert np.allclose(b["selectivity"], (np.array(b["acc"]) - np.array(b["control_acc"])) / 0.75)
    assert b["p_binom_bonf"] > a["p_binom_bonf"]
    hi = _pc(sig, n_perm=4, bow_acc=0.99)  # a rival at ceiling leaves nothing for the state to add
    assert not hi["valid"]


def test_binomial_p_uses_peak_block_control(sig):
    r = _pc(sig, n_perm=4, bow_acc=0.5)
    pb = r["peak_block"]
    k = int(round(r["acc"][pb] * 200))
    p = binom.sf(k - 1, 200, min(max(r["control_acc"][pb], 0.25), 1 - 1e-9))
    assert r["p_binom"] == pytest.approx(p) and r["p_binom_bonf"] == pytest.approx(min(1, 5 * p))


def test_n_perm_zero(sig):
    r = _pc(sig, n_perm=0)
    assert r["p_perm"] is None and r["n_perm"] == 0 and np.allclose(r["control_acc"], 0.25) and r["valid"]
    r = _pc(sig, n_perm=0, bow_acc=0.4)
    assert np.allclose(r["control_acc"], 0.4)


def test_noise_invalid_signal_valid(sig):
    S, y, g = sig
    assert _pc(sig, n_perm=5)["valid"]
    yp = np.random.default_rng(1).permutation(y[:400])
    assert not _pc(sig, y_tr=yp, n_perm=5)["valid"]


# ---- onset reproduction
def _map(curve, control, chance, l_on):
    return {"task": "t", "model": "m", "chance": chance, "L_on": l_on, "control_curve": control,
            "probe": {"all": {"curve": curve, "control": control, "chance": chance, "L_on": l_on}}}


def test_onset_check_pass_and_fail():
    curve = [0.25, 0.25, 0.5, 1.0, 1.0]
    ctrl = [0.25] * 5
    # sel = [0, 0, .333, 1, 1] -> half at 2 + (0.5 - .333)/.667 = 2.25; l_on: first layer >= 0.625 -> 3
    ok = C.onset_check_map(_map(curve, ctrl, 0.25, 3))
    assert ok["pass"] and ok["B"] == 4 and ok["recompute_match"] and ok["diff"] == pytest.approx(2.25 - 3 + 0.0)
    bad = C.onset_check_map(_map(curve, ctrl, 0.25, 1))
    assert not bad["pass"] and not bad["recompute_match"]
    none = C.onset_check_map(_map([0.25] * 5, ctrl, 0.25, None))  # flat: both undefined -> agree
    assert none["pass"] and none["v11_layers"] is None
    assert C.onset_check_map({"probe": {"all": {"curve": curve, "chance": 0.25}}}) is None  # no stored control
    assert C.onset_check_map({"probe": {}}) is None


def test_onset_baseline_forced_to_zero():
    r = C.onset_check_map(_map([0.5, 1.0, 1.0], [0.9, 0.5, 0.5], 0.5, 1))  # control above acc at block 0
    assert r["v11_layers"] == pytest.approx(0.5) and r["pass"]


# ---- tasks
def test_tasks_checks_pass_on_p0_generators():
    r = C.check_tasks()
    assert r["pass"] and r["n_pass"] == r["n_tasks"] == 8
    assert r["tasks"]["surf_copy"]["rivals"]["position"] is not None
    assert r["tasks"]["hop_ctx"]["rivals"]["position"] is None
    assert all(v["duplicate_prompts_train_test"] == 0 and v["template_overlap"] == 0 for v in r["tasks"].values())
    assert r["tasks"]["fact_continent"]["class_words"]["overlap"] == 0
    models = {"m1": {"tokenizer": {n: "ok" for n in C.T.TASK_NAMES}},
              "m2": {"tokenizer": {**{n: "ok" for n in C.T.TASK_NAMES}, "arith_add": "arith_add: option '5' is 2 tokens"}}}
    s = C.check_tasks(models)["tasks"]["arith_add"]["single_token"]
    assert s["n_pass"] == 1 and s["n_fail"] == 1 and "m2" in s["failures"]


def test_leak_detector_flags_a_leak():
    task = C.T.get_task("fact_color")
    it = {"prompt": "Anna says the colour of the red is", "answer": 0,
          "ans_twin": {"prompt": "Anna says the colour of the snow is", "answer": 3}}
    assert C._leaks(task, [it])["violations"] == 1


# ---- tamper on the toy
@pytest.fixture(scope="module")
def tam():
    return C.toy_tamper()


def test_tamper_toy(tam):
    runs = tam["runs"]
    assert all(runs["untampered"]["valid"][s] for s in ("probe", "twin", "use"))
    assert not any(runs["shuffled_labels"]["valid"][s] for s in ("probe", "use"))
    assert not any(runs["swapped_twins"]["valid"][s] for s in ("twin", "use"))
    # the filters a tamper cannot reach are unchanged: twin ignores labels, probe ignores twins
    assert runs["shuffled_labels"]["valid"]["twin"] and runs["swapped_twins"]["valid"]["probe"]
    assert tam["pass"]


def test_tamper_report_fails_when_a_signal_survives():
    ok = {"valid": {"probe": False, "twin": False, "use": False, "arrival": False}}
    live = {"valid": {"probe": True, "twin": False, "use": False, "arrival": False}}
    assert C.tamper_report({"untampered": live, "shuffled_labels": ok, "swapped_twins": ok})["pass"]
    assert not C.tamper_report({"untampered": live, "shuffled_labels": live, "swapped_twins": ok})["pass"]


def test_toy_check_passes():
    r = C.toy_check()
    assert r["pass"] and [x["k"] for x in r["runs"]] == [4, 2]
    for x in r["runs"]:
        assert x["probe_valid"] and x["n_pairs"] >= 40 and x["pass"]


# ---- models subcommand logic
class _Card(dict):
    pass


def _info(gated=False, tags=(), lic="apache-2.0"):
    return types.SimpleNamespace(gated=gated, card_data=_Card(license=lic, license_name=None, license_link=None),
                                 tags=list(tags), created_at=None, last_modified=None, sha="abc")


class _Tok:
    def __init__(self, tpl=None, multi=()):
        self.chat_template, self.multi = tpl, multi

    def __call__(self, text, add_special_tokens=False):
        n = 2 if text.strip() in self.multi else 1
        base = abs(hash(text)) % 1000
        return types.SimpleNamespace(input_ids=[base + i for i in range(n)])


def _patch(monkeypatch, infos, cfgs, toks, disk=100.0):
    monkeypatch.setattr(C, "fetch_info", lambda m: infos[m])
    monkeypatch.setattr(C, "fetch_config", lambda m: cfgs[m])
    monkeypatch.setattr(C, "fetch_tokenizer", lambda m: toks[m])
    monkeypatch.setattr(C, "free_disk_gb", lambda: disk)


def _cfg(n):
    return types.SimpleNamespace(model_type="llama", num_hidden_layers=n, hidden_size=32)


def test_models_logic(monkeypatch, tmp_path):
    ids = ["a/gated", "a/ok", "a/wrong", "a/Chat-Instruct"]
    infos = {"a/gated": _info(gated="manual"), "a/ok": _info(tags=["license:apache-2.0", "base_model:x"]),
             "a/wrong": _info(), "a/Chat-Instruct": _info(tags=["conversational"])}
    cfgs = {"a/ok": _cfg(12), "a/wrong": _cfg(10), "a/Chat-Instruct": _cfg(12)}
    toks = {"a/ok": _Tok(), "a/wrong": _Tok(multi=("red",)), "a/Chat-Instruct": _Tok(tpl="{{x}}")}
    _patch(monkeypatch, infos, cfgs, toks)
    monkeypatch.setattr(C, "MODELS", [("a/gated", 12, "fitted"), ("a/ok", 12, "fitted"), ("a/wrong", 12, "fitted"),
                                      ("a/Chat-Instruct", 12, "fitted")])
    calls = []
    monkeypatch.setattr(C, "load_probe", lambda mid, tok_ok, dev: calls.append(mid) or {"load_s": 0.0})
    out = C.run_models(ids, out_dir=tmp_path)
    assert calls == ["a/ok", "a/wrong", "a/Chat-Instruct"]  # the gated model is recorded, not loaded
    assert out["a/gated"]["load"] == "skipped: gated" and out["a/gated"]["hub"]["gated"] == "manual"
    assert out["a/ok"]["blocks_match"] is True and out["a/ok"]["hub"]["tags_base_model"] == ["base_model:x"]
    assert out["a/ok"]["base"]["base_ok"] is True and out["a/ok"]["tokenizer"]["surf_copy"] == "ok"
    assert out["a/wrong"]["blocks_match"] is False and "2 tokens" in out["a/wrong"]["tokenizer"]["surf_copy"]
    assert out["a/Chat-Instruct"]["base"]["base_ok"] is False
    assert (tmp_path / "models.json").exists() and "models: done a/ok" in (tmp_path / "status.txt").read_text()
    # resumable: a second run skips everything
    C.run_models(ids, out_dir=tmp_path)
    assert calls == ["a/ok", "a/wrong", "a/Chat-Instruct"]
    C.run_models(["a/ok"], out_dir=tmp_path, force=True)
    assert calls[-1] == "a/ok" and len(calls) == 4


def test_models_no_load_and_disk(monkeypatch, tmp_path):
    _patch(monkeypatch, {"a/ok": _info()}, {"a/ok": _cfg(12)}, {"a/ok": _Tok()}, disk=3.0)
    called = []
    r = C.check_model("a/ok", 12, "fitted", no_load=True, loader=lambda *a: called.append(1))
    assert r["load"] == "skipped: --no-load" and not called
    r = C.check_model("a/ok", 12, "fitted", loader=lambda *a: called.append(1))
    assert r["load"] == "skipped: disk" and not called
    monkeypatch.setattr(C, "free_disk_gb", lambda: 80.0)
    r = C.check_model("a/ok", 12, "fitted", loader=lambda *a: called.append(1) or {"ok": 1})
    assert r["load"] == {"ok": 1} and called == [1]


def test_models_list_matches_brief():
    assert len(C.MODELS) == 17 and C.MODELS[-1] == ("TinyLlama/TinyLlama_v1.1", 22, "held out")
