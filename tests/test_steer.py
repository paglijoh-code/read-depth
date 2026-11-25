import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pytest
import torch

from surgery import steer
from surgery.canaries import score_canaries
from surgery.eval import label_logits

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def setup():
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", padding_side="left")
    except Exception:
        pytest.skip("tokenizer not cached")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    model = Qwen3ForCausalLM(cfg).eval()
    recs = [{"id": f"r{i}", "template_id": i, "label": lab, "prompt": p} for i, (lab, p) in enumerate([
        ("READ_FILE", "short"),
        ("EDIT_FILE", "a somewhat longer prompt with more words in it to vary the length"),
        ("RUN_TEST", "medium length prompt here")])]
    return model, tok, recs


def hand_D(d=4):
    lab_dir = torch.nn.functional.normalize(
        torch.arange(1, 6 * d + 1, dtype=torch.float32).reshape(6, d), dim=-1)
    # correctness probe: P(correct) = sigmoid(h[0])
    return {0: {"norm_mean": 2.0, "corr_dm": torch.tensor([0., 1, 0, 0]),
                "corr_probe": torch.tensor([0., 0, 1, 0]), "label_dm": lab_dir, "label_probe": lab_dir,
                "corr_mu": torch.zeros(d), "corr_scale": torch.ones(d),
                "corr_coef": torch.tensor([1., 0, 0, 0]), "corr_b": torch.tensor(0.),
                "lab_mu": torch.zeros(d), "lab_scale": torch.ones(d),
                "lab_coef": torch.eye(6, d)[[0, 1, 2, 3, 0, 1]],  # class k reads coordinate k % 4
                "lab_b": torch.zeros(6)}}, lab_dir


def test_steer_vectors():
    D, lab_dir = hand_D()
    h = torch.tensor([[-3., 1, 2, 0], [3., 1, 2, 0]])
    cfg = {"layer": 0, "direction": "corr_dm", "op": "add", "alpha": 0.5, "tau": None}
    new, mask = steer.steer_vectors(h, cfg, D)
    assert mask.all()
    torch.testing.assert_close(new, h + torch.tensor([0., 1, 0, 0]) * 1.0)
    # tau mask: P(correct) = sigmoid(h0); row 0 ~0.047 < 0.5, row 1 ~0.95
    new, mask = steer.steer_vectors(h, {**cfg, "tau": 0.5}, D)
    assert mask.tolist() == [True, False]
    torch.testing.assert_close(new[1], h[1])
    assert not torch.equal(new[0], h[0])
    # project_out
    po = {"layer": 0, "direction": "corr_probe", "op": "project_out", "alpha": None, "tau": None}
    new, _ = steer.steer_vectors(h, po, D)
    assert torch.allclose(new @ torch.tensor([0., 0, 1, 0]), torch.zeros(2), atol=1e-6)
    torch.testing.assert_close(new[:, [0, 1, 3]], h[:, [0, 1, 3]])
    # alpha 0 is identity
    new, _ = steer.steer_vectors(h, {**cfg, "alpha": 0.0}, D)
    torch.testing.assert_close(new, h)
    # label rows: predicted class = argmax of lab_coef @ h; coef row k picks coordinate k % 4
    lab = {"layer": 0, "direction": "label_dm", "op": "add", "alpha": 1.0, "tau": None}
    hh = torch.tensor([[0., 5, 0, 0]])  # class 1 (coordinate 1) and class 5 tie; argmax takes first -> 1
    new, _ = steer.steer_vectors(hh, lab, D)
    torch.testing.assert_close(new, hh + 2.0 * lab_dir[1])


def test_config_validation():
    with pytest.raises(ValueError):
        steer.validate_config({"layer": 0, "direction": "label_dm", "op": "project_out", "alpha": None})
    with pytest.raises(ValueError):
        steer.validate_config({"layer": 0, "direction": "corr_dm", "op": "add", "alpha": None})
    assert steer.config_id({"a": 1}) == steer.config_id({"a": 1})


def rand_D(model, layer, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = model.config.hidden_size
    u = lambda *s: torch.nn.functional.normalize(torch.randn(*s, generator=g), dim=-1)  # noqa: E731
    return {layer: {"norm_mean": 3.0, "corr_dm": u(d), "corr_probe": u(d), "label_dm": u(6, d),
                    "label_probe": u(6, d), "corr_mu": torch.zeros(d), "corr_scale": torch.ones(d),
                    "corr_coef": torch.randn(d, generator=g), "corr_b": torch.tensor(0.),
                    "lab_mu": torch.zeros(d), "lab_scale": torch.ones(d),
                    "lab_coef": torch.randn(6, d, generator=g), "lab_b": torch.zeros(6)}}


def test_apply_steer_position_only(setup):
    model, tok, _ = setup
    D = rand_D(model, 0)
    cfg = {"layer": 0, "direction": "corr_dm", "op": "add", "alpha": 1.0, "tau": None}
    x = torch.randint(0, 1000, (2, 6))
    cap = {}
    with torch.no_grad():
        hk = model.model.layers[0].register_forward_hook(lambda m, a, o: cap.__setitem__("o", o))
        model(input_ids=x)
        plain = (cap["o"][0] if isinstance(cap["o"], tuple) else cap["o"]).clone()
        hk.remove()
        with steer.apply_steer(model, cfg, D, positions=torch.tensor([2, 4])):
            hk = model.model.layers[0].register_forward_hook(lambda m, a, o: cap.__setitem__("o", o))
            model(input_ids=x)
            hk.remove()
    steered = cap["o"][0] if isinstance(cap["o"], tuple) else cap["o"]
    diff = (steered - plain).abs().sum(-1) > 1e-6
    assert diff.tolist() == [[False, False, True, False, False, False],
                             [False, False, False, False, True, False]]


def test_eval_configs_matches_plain(setup):
    model, tok, recs = setup
    D = rand_D(model, 1)
    cfg = {"layer": 1, "direction": "label_probe", "op": "add", "alpha": 0.7, "tau": 0.9}
    cfg2 = {"layer": 1, "direction": "corr_probe", "op": "project_out", "alpha": None, "tau": None}
    res = steer.eval_configs(model, tok, recs, [cfg, cfg2], D, batch_size=3)
    plain = label_logits(model, tok, [r["prompt"] for r in recs], 3).numpy()
    np.testing.assert_allclose(res["baseline"], plain, atol=1e-4)
    np.testing.assert_allclose(steer.plain_label_logits(model, tok, recs, 3).numpy(), plain, atol=1e-6)
    for c in (cfg, cfg2):
        with steer.apply_steer(model, c, D):
            ref = label_logits(model, tok, [r["prompt"] for r in recs], 3).numpy()
        np.testing.assert_allclose(res["configs"][steer.config_id(c)], ref, atol=1e-4)
    assert 0 <= res["steered_frac"][steer.config_id(cfg)] <= 1
    s = steer.summarize_config(recs, res["configs"][steer.config_id(cfg)], res["baseline"], n_boot=50)
    assert len(s["pred"]) == 3 and "ece" in s and "mcnemar" in s


def test_build_directions(tmp_path):
    rng = np.random.default_rng(0)
    N, L, d = 120, 3, 8
    y = rng.integers(0, 6, N)
    pred = np.where(rng.random(N) < 0.5, y, (y + 1) % 6)
    resid = rng.normal(size=(N, L + 1, d)).astype(np.float16)
    resid[np.arange(N), 2, :] += y[:, None] * 0.5
    np.save(tmp_path / "resid.npy", resid)
    (tmp_path / "meta.json").write_text(json.dumps({"y": y.tolist(), "pred": pred.tolist()}))
    D = steer.build_directions(tmp_path, [0, 1])
    assert D["meta"]["n"] == N and D["meta"]["layers"] == [0, 1]
    for l in (0, 1):
        e = D[l]
        assert e["norm_mean"] > 0
        for k in ("corr_dm", "corr_probe"):
            assert e[k].shape == (d,) and abs(float(e[k].norm()) - 1) < 1e-5
        for k in ("label_dm", "label_probe"):
            assert e[k].shape == (6, d)
            torch.testing.assert_close(e[k].norm(dim=-1), torch.ones(6), atol=1e-5, rtol=0)
        assert e["lab_coef"].shape == (6, d) and e["lab_b"].shape == (6,) and e["corr_coef"].shape == (d,)
    steer.save_directions(D, tmp_path / "d.pt")
    assert steer.load_directions(tmp_path / "d.pt")["meta"]["n"] == N


def test_canaries_on_batch(setup):
    model, tok, _ = setup
    items = [{"prompt": "What is two plus two?", "options": ["four", "five"], "answer": 0, "category": "syntax"},
             {"prompt": "Name a color", "options": ["red", "a very long option here", "x"], "answer": 1,
              "category": "factual"}]
    a = score_canaries(model, tok, items, batch_size=4)
    seen = []
    b = score_canaries(model, tok, items, batch_size=4, on_batch=seen.append)
    assert a["scores"] == b["scores"] and a["pred"] == b["pred"] and seen
    # steered hook hits n_prompt - 1 on every row
    D = rand_D(model, 0)
    cfg = {"layer": 0, "direction": "corr_dm", "op": "add", "alpha": 1.0, "tau": None}
    n_prompts = [len(tok(f"Question: {it['prompt']}\nAnswer:",
                         add_special_tokens=False)["input_ids"]) for it in items]
    orig = steer.apply_steer

    seq = []

    def spy(m, c, DD, positions=None):
        def pos():
            p = positions()
            seq.append(p.clone())
            return p
        return orig(m, c, DD, positions=pos)

    steer.apply_steer = spy
    try:
        steer.score_canaries_steered(model, tok, items, cfg, D, batch_size=8)
    finally:
        steer.apply_steer = orig
    got = sorted(int(v) for p in seq for v in p)
    expect = sorted(n - 1 for n in n_prompts for _ in items[n_prompts.index(n)]["options"])
    assert got == expect
