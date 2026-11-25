import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from surgery import probes

ROOT = Path(__file__).resolve().parent.parent


def _load_export():
    spec = importlib.util.spec_from_file_location(
        "map_export", ROOT / "scripts" / "map_export.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_store(path, n=120, n_pos=4, d=16, n_templates=12, signal_layer=2,
               seed=0):
    """Fake store following the M3 store spec."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 6, n)
    tids = np.arange(n) % n_templates
    resid = rng.normal(size=(n, n_pos, d)).astype(np.float16)
    onehot = np.eye(6)[y] * 6.0
    resid[:, signal_layer, :6] = (resid[:, signal_layer, :6] + onehot).astype(np.float16)
    lens = rng.normal(size=(n, n_pos, 6)).astype(np.float32)
    final = lens[:, -1].copy()
    path.mkdir(parents=True)
    np.save(path / "resid.npy", resid)
    np.save(path / "lens.npy", lens)
    np.save(path / "final_logits.npy", final)
    meta = dict(model="fake", variant="rule", split="discovery", data_path="x",
                n=n, n_layers=n_pos - 1, d_model=d, n_heads=4, n_kv_heads=2,
                head_dim=4, ids=[f"id{i}" for i in range(n)],
                template_ids=[f"t{t}" for t in tids], y=y.tolist(),
                pred=final.argmax(1).tolist(), label_token_ids=list(range(6)))
    (path / "meta.json").write_text(json.dumps(meta))
    return meta


def test_groupkfold_never_splits_a_template():
    groups = np.arange(100) % 10
    X = np.zeros((100, 1))
    for train, test in GroupKFold(n_splits=5).split(X, groups, groups):
        assert not set(groups[train]) & set(groups[test])


def test_oof_uses_template_disjoint_folds_and_covers_all_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5)).astype(np.float32)
    y = rng.integers(0, 3, 100)
    groups = np.arange(100) % 10
    out = probes._oof(X, y, groups, "label", 0, permute=False)
    assert out.shape == (100,) and set(out) <= {0, 1, 2}


def test_control_near_chance_on_random_data():
    rng = np.random.default_rng(1)
    n = 600
    X = rng.normal(size=(n, 3, 20)).astype(np.float32)
    y = rng.integers(0, 6, n)
    groups = np.arange(n) % 30
    res = probes.probe_by_layer(X, y, groups, "label", n_boot=50)
    for r in res:
        assert abs(r["control"] - 1 / 6) < 0.08
        assert abs(r["metric"] - 1 / 6) < 0.08
    correct = rng.integers(0, 2, n)
    res = probes.probe_by_layer(X, correct, groups, "correct", n_boot=50)
    for r in res:
        assert abs(r["control"] - 0.5) < 0.1


def test_planted_signal_found_at_its_layer():
    rng = np.random.default_rng(2)
    n, d = 600, 20
    y = rng.integers(0, 6, n)
    X = rng.normal(size=(n, 4, d)).astype(np.float32)
    X[:, 2, :6] += np.eye(6)[y] * 5
    groups = np.arange(n) % 30
    res = probes.probe_by_layer(X, y, groups, "label", n_boot=50)
    assert res[2]["metric"] > 0.9 and res[2]["lo"] > 0.8
    assert all(res[i]["metric"] < 0.4 for i in (0, 1, 3))
    assert res[2]["control"] < 0.4
    # correctness planted at layer 1
    c = rng.integers(0, 2, n)
    X2 = rng.normal(size=(n, 3, d)).astype(np.float32)
    X2[:, 1, 0] += c * 4
    res = probes.probe_by_layer(X2, c, groups, "correct", n_boot=50)
    assert res[1]["metric"] > 0.9 and res[0]["metric"] < 0.65


def test_export_schema(tmp_path):
    meta = make_store(tmp_path / "store")
    mod = _load_export()
    out = tmp_path / "maps.json"
    mod.export(tmp_path / "store", out, "fake", "rule", probe_n=80, n_pca=50,
               n_boot=20, log=lambda *_: None)
    r = json.loads(out.read_text())
    assert {"model", "variant", "n", "n_layers"} <= set(r)
    assert r["n"] == 120 and r["n_layers"] == 3
    assert r["probe_subset"]["used"] and r["probe_subset"]["n"] == 80
    assert len(r["layers"]) == 4
    L = r["layers"][2]
    for k in ("label_probe", "correct_probe"):
        assert {"metric", "lo", "hi", "control", "control_lo", "control_hi"} <= set(L[k])
    assert {"lens_accuracy", "lens_p_correct", "norm_mean", "norm_p5",
            "norm_p95"} <= set(L)
    assert len(L["pca"]["x"]) == len(L["pca"]["y"]) == 50 == len(r["pca_labels"])
    assert L["pca"]["x"][0] == round(L["pca"]["x"][0], 3)
    assert r["layers"][2]["label_probe"]["metric"] > r["layers"][0]["label_probe"]["metric"]
    assert len(r["lens_examples"]) == 6
    assert sorted(e["label"] for e in r["lens_examples"]) == list(range(6))
    assert all(len(e["probs"]) == 4 and len(e["probs"][0]) == 6
               for e in r["lens_examples"])
    assert np.shape(r["lens_by_label"]) == (6, 4)
    assert L["norm_p5"] <= L["norm_mean"] <= L["norm_p95"]
