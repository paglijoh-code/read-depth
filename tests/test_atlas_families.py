import collections
import functools

import pytest

from surgery.atlas import families as F
from surgery.atlas.families import core
from surgery.atlas.families import tokcheck as K

NAMES = F.TASK_NAMES
_TOKS = {}


@functools.lru_cache(maxsize=None)
def items_of(name, seed=0):
    return F.generate(name, seed=seed)


@functools.lru_cache(maxsize=None)
def rivals_of(name):
    return F.rivals(items_of(name), name)


def distinct_tokenizers():
    """The 7 distinct tokenizers, loaded once per process; skips the caller if any of the 17 cannot load."""
    if "d" not in _TOKS:
        pytest.importorskip("transformers")
        toks = K.load_tokenizers()
        _TOKS["d"] = K.distinct(toks) if len(toks) == len(K.TOKENIZER_IDS) else None
    if _TOKS["d"] is None:
        pytest.skip("not all tokenizers are cached")
    return _TOKS["d"]


@pytest.fixture(params=NAMES)
def name(request):
    return request.param


def test_sizes_balance_templates(name):
    it = items_of(name)
    tr, te = F.split_items(it)
    assert len(tr) == 400 and len(te) == 200
    for part in (tr, te):
        assert set(collections.Counter(x["answer"] for x in part)) == {0, 1, 2, 3}
        assert set(collections.Counter(x["answer"] for x in part).values()) == {len(part) // 4}
    ttr, tte = {x["template"] for x in tr}, {x["template"] for x in te}
    assert len(ttr) >= 8 and len(tte) == 4 and not (ttr & tte)
    assert len(F.get_task(name).spec.templates) >= 12
    assert all(len(x["prompt"].split()) < 60 for x in it)


def test_deterministic_per_seed():
    n = NAMES[0]
    assert F.generate(n, seed=0) == items_of(n)
    assert F.generate(n, seed=1) != items_of(n)


def test_held_out_class_words(name):
    spec = F.get_task(name).spec
    if not spec.class_pools:
        pytest.skip("no class pools")
    tr, te = F.split_items(items_of(name))
    wtr, wte = {x["slots"][spec.cls_key] for x in tr}, {x["slots"][spec.cls_key] for x in te}
    assert not (wtr & wte)
    for c, pool in enumerate(spec.class_pools):
        assert len(pool) >= 6 and len(core._split_words(pool, "test")) >= 2, (name, c)
    assert wte <= {w for p in spec.class_pools for w in core._split_words(p, "test")}


def test_oracle_and_twins_semantics(name):
    spec = F.get_task(name).spec
    for x in items_of(name):
        assert spec.solve(x["prompt"]) == x["answer"], x["prompt"]
        assert spec.solve(x["ans_twin"]["prompt"]) == x["ans_twin"]["answer"], x["ans_twin"]["prompt"]
        assert spec.solve(x["surf_twin"]["prompt"]) == x["surf_twin"]["answer"], x["surf_twin"]["prompt"]
        assert x["ans_twin"]["answer"] != x["answer"] and 0 <= x["ans_twin"]["answer"] < 4
        assert x["surf_twin"]["answer"] == x["answer"]
        assert x["ans_twin"]["prompt"] != x["prompt"] and x["surf_twin"]["prompt"] != x["prompt"]


def test_rivals_near_chance(name):
    r = rivals_of(name)
    task = F.get_task(name)
    assert r["bow"] <= 0.35, r
    assert len(r["near_miss"]) >= 2 and all(v <= 0.35 for v in r["near_miss"].values()), r
    if task.copy:
        assert r["position"] is not None and r["position"] <= 0.35, r
    else:
        assert r["position"] is None


def test_registry_families():
    fam = collections.Counter(F.get_task(n).family for n in NAMES)
    assert set(fam.values()) == {2}
    for fam_name, (label, tasks) in F.FAMILIES.items():
        assert label in F.TYPE_LABELS and tasks == (f"{fam_name}_a", f"{fam_name}_b")
    assert len(F.TYPE_LABELS) == 7


def test_check_options_in_context():
    tok = next(iter(distinct_tokenizers().values()))
    for n in NAMES:
        assert len(set(F.check_options(tok, n))) == 4


def test_tokenizer_alignment_and_options(name):
    toks = distinct_tokenizers()
    task = F.get_task(name)
    res = K.check_items(items_of(name), task.options, toks)
    assert res["ok"], (name, res["per_tokenizer"], res["bad_ids"][:5])
