import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pytest
import torch

from surgery.store import build_store

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
    recs = [
        {"id": f"r{i}", "template_id": i, "label": lab, "prompt": p}
        for i, (lab, p) in enumerate([
            ("READ_FILE", "short"),
            ("EDIT_FILE", "a somewhat longer prompt with more words in it to vary the length"),
            ("RUN_TEST", "medium length prompt here"),
        ])
    ]
    return model, tok, recs


def test_store(setup, tmp_path):
    from surgery.prompt import build_prompt
    model, tok, recs = setup
    meta = build_store(model, tok, recs, tmp_path, batch_size=3)
    L, d = 2, 64
    resid = np.load(tmp_path / "resid.npy")
    lens = np.load(tmp_path / "lens.npy")
    final = np.load(tmp_path / "final_logits.npy")
    assert resid.shape == (3, L + 1, d) and resid.dtype == np.float16
    assert lens.shape == (3, L + 1, 6) and lens.dtype == np.float32
    assert final.shape == (3, 6)
    assert meta["pred"] == final.argmax(-1).tolist() and meta["head_dim"] == 16
    np.testing.assert_allclose(lens[:, L], final, atol=1e-4)

    enc = tok([build_prompt(tok, r["prompt"]) for r in recs], return_tensors="pt", padding=True)
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    # last entry of hidden_states is post-norm in HF; compare layer 1 output via norm
    normed = hs[-1][:, -1]
    got = model.model.norm(torch.tensor(resid[:, L]).float())
    np.testing.assert_allclose(got.detach().numpy(), normed.numpy(), atol=2e-2)
    np.testing.assert_allclose(resid[:, 0], hs[0][:, -1].numpy(), atol=1e-2)

    # brute force means, one prompt at a time (no padding)
    sums = {"attn": torch.zeros(L, d), "mlp": torch.zeros(L, d), "heads": torch.zeros(L, 4, 16)}
    cnt = 0
    caps = {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(
            lambda m, a, o, i=i: caps.__setitem__(("attn", i), o[0])))
        hooks.append(layer.mlp.register_forward_hook(lambda m, a, o, i=i: caps.__setitem__(("mlp", i), o)))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
            lambda m, a, i=i: caps.__setitem__(("heads", i), a[0])))
    for r in recs:
        e = tok([build_prompt(tok, r["prompt"])], return_tensors="pt")
        with torch.no_grad():
            model(**e)
        cnt += e["input_ids"].shape[1]
        for i in range(L):
            sums["attn"][i] += caps[("attn", i)][0].sum(0)
            sums["mlp"][i] += caps[("mlp", i)][0].sum(0)
            sums["heads"][i] += caps[("heads", i)][0].reshape(-1, 4, 16).sum(0)
    for h in hooks:
        h.remove()
    means = torch.load(tmp_path / "means.pt")
    for k in sums:
        torch.testing.assert_close(means[k], sums[k] / cnt, atol=1e-5, rtol=1e-4)
