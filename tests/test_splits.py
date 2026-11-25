"""Real generator pools plus focused tests of template allocation (no model)."""
import json
import random
from collections import Counter

import pytest

from surgery.task import splits
from surgery.task.generator import LABELS, generate, read_jsonl


@pytest.fixture(scope="module")
def pools():
    # Full rule generation takes <10 s; full sensible generation takes ~35 s.
    # Reuse real deterministic pools to exercise allocation repeatedly within 20 s.
    return {"rule": generate(12000, 0), "sensible": generate(1800, 0, variant="sensible")}


@pytest.fixture
def cached_pools(monkeypatch, pools):
    def sample(n, seed, variant):
        assert n <= len(pools[variant])
        return pools[variant][:n]
    monkeypatch.setattr(splits, "generate", sample)


@pytest.mark.parametrize("variant,sizes,pool_n", [
    ("rule", {"discovery": 3000, "selection": 1500, "test": 1500}, 12000),
    ("sensible", {"discovery": 400, "selection": 300, "test": 300}, 1800),
])
def test_real_pool_splits_and_determinism(cached_pools, variant, sizes, pool_n):
    result = splits.make_splits(variant, 0, sizes, pool_n)
    again = splits.make_splits(variant, 0, sizes, pool_n)
    seen = set()
    for name, rows in result.items():
        assert len(rows) == sizes[name]
        templates = {r["template_id"] for r in rows}
        assert len(templates) >= 50
        assert not seen & templates
        seen |= templates
        counts = Counter(r["label"] for r in rows)
        assert set(counts) == set(LABELS)
        assert all(.08 <= counts[label] / len(rows) <= .30 for label in LABELS)
        assert [r["id"] for r in rows] == [r["id"] for r in again[name]]
    # Hold the generated pool fixed to isolate the seed's effect on template shuffle.
    other_seed = splits.make_splits(variant, 1, sizes, pool_n)
    assert [r["id"] for r in result["discovery"]] != [r["id"] for r in other_seed["discovery"]]


def test_pool_grows_when_whole_templates_exhaust_it(monkeypatch):
    calls = []

    def sample(n, seed, variant):
        calls.append((n, seed, variant))
        return [{"id": str(i), "template_id": f"t{i // 2}", "label": LABELS[i % 6]}
                for i in range(n)]

    monkeypatch.setattr(splits, "generate", sample)
    result, manifest = splits._make_splits("rule", 0, {"a": 100, "b": 100}, pool_n=100)
    assert calls == [(100, 0, "rule"), (200, 0, "rule")]
    assert manifest["pool_size"] == 200
    assert manifest["sizes"] == {"a": 100, "b": 100}
    assert manifest["template_counts"] == {"a": 50, "b": 50}
    assert len({r["id"] for rows in result.values() for r in rows}) == 200


def test_templates_dealt_whole_capped_and_tails_dropped():
    pool = [{"id": f"t{i}-{j}", "template_id": f"t{i}"} for i in range(300) for j in range(8)]
    result = splits._partition(pool, 7, {"a": 400, "b": 200})
    assert {k: len(v) for k, v in result.items()} == {"a": 400, "b": 200}
    owner = {}
    for name, rows in result.items():
        for r in rows:
            assert owner.setdefault(r["template_id"], name) == name  # one split per template
    cap = {"a": round(splits.TEMPLATE_CAP * 400), "b": round(splits.TEMPLATE_CAP * 200)}
    per = Counter((name, r["template_id"]) for name, rows in result.items() for r in rows)
    assert all(n <= cap[name] for (name, _), n in per.items())
    for (name, t), n in per.items():  # what a split keeps of a template is a prefix
        assert [r["id"] for r in result[name] if r["template_id"] == t] == [f"{t}-{j}" for j in range(n)]
    assert splits._partition(pool, 7, {"a": 400, "b": 200}) == result


def test_coarse_templates_raise(monkeypatch):
    monkeypatch.setattr(splits, "generate", lambda *args, **kwargs: [
        {"id": str(i), "template_id": "one", "label": "DONE"} for i in range(200)])
    with pytest.raises(ValueError, match="cannot fill splits.*pool template sizes"):
        splits.make_splits("rule", 0, {"discovery": 100})


@pytest.mark.parametrize("sizes,pool_n", [({}, None), ({"a": 0}, None), ({"a": -1}, None),
                                         ({"a": 1.5}, None), ({"a": 100}, 0)])
def test_bad_sizes_rejected(sizes, pool_n):
    with pytest.raises(ValueError):
        splits.make_splits("rule", 0, sizes, pool_n)


def test_cli_files_and_manifest(cached_pools, monkeypatch, tmp_path):
    sizes = {"discovery": 600, "selection": 300, "test": 300}
    monkeypatch.setitem(splits.SPLIT_SIZES, 0, sizes)
    splits.main(["make", "--variant", "rule", "--seed", "0", "--out-dir", str(tmp_path)])
    manifest = json.loads((tmp_path / "rule_s0_manifest.json").read_text())
    assert manifest["sizes"] == sizes
    assert manifest["pool_size"] == 2400
    assert manifest["variant"] == "rule" and manifest["seed"] == 0
    for name, size in sizes.items():
        rows = read_jsonl(tmp_path / f"rule_s0_{name}.jsonl")
        assert len(rows) == size
        assert manifest["template_counts"][name] == len({r["template_id"] for r in rows})
        assert manifest["label_counts"][name] == Counter(r["label"] for r in rows)
    assert manifest["template_size_distribution"]["max"] == 66


def test_production_seed_sizes():
    assert splits.SPLIT_SIZES == {
        0: {"discovery": 3000, "selection": 1500, "test": 1500},
        1: {"discovery": 3000, "selection": 1500, "test": 1500},
    }
