import collections
import re

import pytest

from surgery.atlas import p0_tasks as T

TOKENIZERS = ["EleutherAI/pythia-160m", "HuggingFaceTB/SmolLM-135M", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen3-0.6B"]


@pytest.fixture(scope="module", params=T.TASK_NAMES)
def items(request):
    return request.param, T.generate(request.param)


def _words(p):
    return re.findall(r"[A-Za-z]+|\d|[^\sA-Za-z\d]", p)  # words, single digits, punctuation


def test_sizes_splits_and_balance(items):
    name, it = items
    tr, te = T.split_items(it)
    assert len(tr) == 400 and len(te) == 200
    for part in (tr, te):
        assert set(collections.Counter(x["answer"] for x in part).values()) == {len(part) // 4}
    ttr, tte = {x["template"] for x in tr}, {x["template"] for x in te}
    assert len(ttr) >= 10 and len(tte) >= 3 and not (ttr & tte)


def test_deterministic():
    assert T.generate("hop_ctx", seed=0) == T.generate("hop_ctx", seed=0)
    assert T.generate("hop_ctx", seed=0) != T.generate("hop_ctx", seed=1)


def test_twins(items):
    name, it = items
    for x in it:
        w, a, s = _words(x["prompt"]), _words(x["ans_twin"]["prompt"]), _words(x["surf_twin"]["prompt"])
        assert len(w) == len(a) == len(s)
        da = [i for i in range(len(w)) if w[i] != a[i]]
        ds = [i for i in range(len(w)) if w[i] != s[i]]
        assert len(da) == 1 and len(ds) == 1, (x["prompt"], x["ans_twin"]["prompt"])
        assert da[0] < len(w) - 1 and ds[0] < len(w) - 1 and w[-1] == a[-1] == s[-1]
        assert x["ans_twin"]["answer"] != x["answer"] and x["surf_twin"]["answer"] == x["answer"]
        assert 0 <= x["ans_twin"]["answer"] < 4


def test_family_list_and_pool_sizes():
    fam = collections.Counter(T.get_task(n).family for n in T.TASK_NAMES)
    assert set(fam.values()) == {2} and len(fam) == 4
    assert sum(len(c) for c in T.COUNTRIES) >= 60 and sum(len(c) for c in T.COLOR_OBJ) >= 40


def test_answer_semantics():
    it = {x["id"]: x for x in T.generate("arith_add")}
    for x in it.values():
        a, b = map(int, re.findall(r"(\d)\+(\d)", x["prompt"])[0])
        assert a + b - 5 == x["answer"]
    for x in T.generate("arith_two"):
        a, b, c = map(int, re.findall(r"(\d)\+(\d)-(\d)", x["prompt"])[0])
        assert a + b - c - 3 == x["answer"]
    for x in T.generate("fact_continent"):
        assert any(x["slots"]["c"] in cs for cs in [T.COUNTRIES[x["answer"]]])
    for x in T.generate("hop_ctx"):
        assert x["slots"]["cont"][x["slots"]["ppl"].index(x["slots"]["q"])] == x["answer"]
        assert x["prompt"].endswith("lives on the continent of")


def test_word_split_between_train_and_test():
    it = T.generate("fact_continent")
    tr, te = T.split_items(it)
    assert not ({x["slots"]["c"] for x in tr} & {x["slots"]["c"] for x in te})


def test_rivals_near_chance_where_designed():
    for n in ("surf_first", "fact_continent", "fact_color", "hop_fact", "hop_ctx", "arith_two"):
        r = T.rivals(T.generate(n), n)
        assert r["bow"] < 0.45 and r["near_miss"] < 0.45, (n, r)
    r = T.rivals(T.generate("surf_copy"), "surf_copy")
    assert r["position"] == 1.0  # a copy task: the position rival is the solution
    assert T.rivals(T.generate("hop_fact"), "hop_fact")["position"] is None


class _FakeTok:
    def __init__(self, multi):
        self.multi, self.vocab = multi, {}

    def __call__(self, s, add_special_tokens=False):
        class R:
            pass
        r = R()
        r.input_ids = [1, 2] if s.strip() == self.multi else [self.vocab.setdefault(s, len(self.vocab) + 10)]
        return r


def test_check_options_raises_on_multi_token():
    T.check_options(_FakeTok("zzz"), "fact_continent")
    with pytest.raises(ValueError):
        T.check_options(_FakeTok("Asia"), "fact_continent")


@pytest.mark.parametrize("tok_id", TOKENIZERS)
def test_options_and_pools_single_token_real_tokenizers(tok_id):
    tr = pytest.importorskip("transformers")
    try:
        tok = tr.AutoTokenizer.from_pretrained(tok_id, local_files_only=True)
    except Exception:
        pytest.skip(f"{tok_id} not cached")
    for n in T.TASK_NAMES:
        ids = T.check_options(tok, n)
        assert len(set(ids)) == 4
    # item/twin token lengths agree for (nearly) all items
    for n in T.TASK_NAMES:
        it = T.generate(n)
        bad = sum(
            len({len(tok(x[k]["prompt"] if k != "item" else x["prompt"], add_special_tokens=False).input_ids)
                 for k in ("item", "ans_twin", "surf_twin")}) > 1
            for x in it)
        assert bad <= 0.05 * len(it), (tok_id, n, bad)
