import numpy as np
import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from surgery.atlas import p0_harness as H
from surgery.atlas import p0_measures as M

V = 100


def _tiny(arch, seed=0):
    torch.manual_seed(seed)
    kw = dict(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
              max_position_embeddings=64)
    if arch == "neox":
        return GPTNeoXForCausalLM(GPTNeoXConfig(**kw))
    if arch == "llama":
        return LlamaForCausalLM(LlamaConfig(**kw, num_key_value_heads=2))
    if arch == "qwen2":
        return Qwen2ForCausalLM(Qwen2Config(**kw, num_key_value_heads=2))
    import transformers
    cls = {"qwen3": ("Qwen3Config", "Qwen3ForCausalLM"), "olmo": ("OlmoConfig", "OlmoForCausalLM"),
           "olmo2": ("Olmo2Config", "Olmo2ForCausalLM")}[arch]
    if not hasattr(transformers, cls[0]):
        pytest.skip(f"{cls[0]} missing from the installed transformers")
    extra = dict(head_dim=8) if arch == "qwen3" else {}
    return getattr(transformers, cls[1])(getattr(transformers, cls[0])(**kw, **extra, num_key_value_heads=2))


@pytest.fixture(params=["neox", "llama", "qwen2", "qwen3", "olmo", "olmo2"])
def ad(request):
    return H.Adapter(model=_tiny(request.param), device="cpu")


OPT = [5, 17, 33, 71]


def _ids(rng, n, T):
    return [[int(t) for t in rng.integers(0, V, T)] for _ in range(n)]


def test_states_logits_match_full_forward_and_batching(ad):
    rng = np.random.default_rng(0)
    ids = _ids(rng, 5, 7) + _ids(rng, 3, 9)
    S, L = ad.collect_ids(ids, OPT, batch_size=2)
    assert S.shape == (8, ad.n_blocks + 1, 32) and L.shape == (8, 4)
    for i, x in enumerate(ids):
        with torch.no_grad():
            o = ad.model(torch.tensor([x]), output_hidden_states=True)
        assert np.allclose(L[i], o.logits[0, -1, OPT].numpy(), atol=1e-5)
        hs = o.hidden_states  # [embed, after block 1, ..., after block B (post final norm)]
        assert np.allclose(S[i, 0], hs[0][0, -1].numpy(), atol=1e-5)
        assert np.allclose(S[i, 1], hs[1][0, -1].numpy(), atol=1e-5)
        # raw last block output != post-norm last hidden state, and the lens on it reproduces the logits
    lens = ad.lens_logits(S, OPT)
    assert np.allclose(lens[:, -1], L, atol=1e-4)


def test_unpatched_and_self_patch_equal_full_forward(ad):
    rng = np.random.default_rng(1)
    x = _ids(rng, 1, 8)[0]
    S, L = ad.collect_ids([x], OPT)
    rows = [(None, None)] + [(b, S[0, b]) for b in range(1, ad.n_blocks + 1)]
    out = ad.patched_logits(x, rows, OPT, chunk=3)  # chunking path
    assert out.shape == (len(rows), 4)
    assert np.allclose(out, L[0][None], atol=1e-4)


def test_patch_with_other_state_matches_manual_hook(ad):
    rng = np.random.default_rng(2)
    a, b = _ids(rng, 2, 8)
    b[:-1], b[-1] = b[:-1], a[-1]
    Sa, La = ad.collect_ids([a], OPT)
    Sb, Lb = ad.collect_ids([b], OPT)
    for L in range(1, ad.n_blocks):
        got = ad.patched_logits(a, [(L, Sb[0, L])], OPT)[0]
        # manual: full forward on a with block L's last-token output replaced
        def f(m, i, o):
            h = (o[0] if isinstance(o, tuple) else o).clone()
            h[:, -1] = torch.tensor(Sb[0, L])
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        hk = ad.layers[L - 1].register_forward_hook(f)
        with torch.no_grad():
            ref = ad.model(torch.tensor([a])).logits[0, -1, OPT].numpy()
        hk.remove()
        assert np.allclose(got, ref, atol=1e-4)
    # replacing the last block's state with another item's changes the answer to that item's logits at block B
    assert np.allclose(ad.patched_logits(a, [(ad.n_blocks, Sb[0, -1])], OPT)[0], Lb[0], atol=1e-4)


def test_make_rows_and_use_arrival_shapes(ad):
    rng = np.random.default_rng(3)
    B, d = ad.n_blocks, ad.d
    xs = _ids(rng, 1, 8)[0]
    xa, xf = list(xs), list(xs)
    xa[2], xf[3] = (xs[2] + 1) % V, (xs[3] + 1) % V
    S, _ = ad.collect_ids([xs, xa, xf], OPT)
    Qp = np.stack([np.linalg.qr(rng.standard_normal((d, 3)))[0] for _ in range(B + 1)]).astype(np.float32)
    Qr = M.random_subspaces(d, B, seed=0)
    res = H.run_use_arrival(ad, xs, S[0], S[1], S[2], Qp, Qr, OPT)
    assert res["probe"].shape == (B - 1, 4) and res["rand"].shape == (B - 1, 10, 4)
    # whole-state swap with the answer twin at block L, run through the remaining blocks, equals the twin's own run
    # only when later positions did not matter; at least it must equal the manual patched run
    ref = ad.patched_logits(xs, [(1, S[1, 1])], OPT)[0]
    assert np.allclose(res["whole"][0], ref, atol=1e-4)
    # swap in a subspace: identity when twin == item
    same = H.swap_subspace(S[0, 1], S[0, 1], Qp[1])
    assert np.allclose(same, S[0, 1], atol=1e-5)
    h2 = H.swap_subspace(S[0, 1], S[1, 1], Qp[1])
    assert np.allclose(Qp[1].T @ h2, Qp[1].T @ S[1, 1], atol=1e-4)
    rest = lambda h: h - Qp[1] @ (Qp[1].T @ h)
    assert np.allclose(rest(h2), rest(S[0, 1]), atol=1e-4)


class _Tok:
    """Whitespace tokenizer stand-in: one token per word."""
    vocab = {}

    def __call__(self, text, add_special_tokens=False):
        class R:
            pass
        r = R()
        r.input_ids = [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]
        return r


def test_filter_items_drops_length_mismatch():
    ad = H.Adapter(model=_tiny("llama"), device="cpu", tok=_Tok())
    mk = lambda p, a, s: {"prompt": p, "ans_twin": {"prompt": a}, "surf_twin": {"prompt": s}}
    items = [mk("a b c in", "a x c in", "a b y in"), mk("a b c in", "a x y c in", "a b y in"),
             mk("a b c in", "a b c on", "a b y in")]
    kept, dropped = ad.filter_items(items)
    assert len(kept) == 1 and dropped == 2


def test_unsupported_arch_raises():
    from transformers import GPT2Config, GPT2LMHeadModel
    with pytest.raises(ValueError):
        H.Adapter(model=GPT2LMHeadModel(GPT2Config(n_layer=1, n_embd=16, n_head=2, vocab_size=50)), device="cpu")


class _TokBos:
    def __init__(self, bos, eos=None):
        self.bos_token_id, self.eos_token_id = bos, eos


def test_answer_logfreq_two_readings(ad):
    ad.tok = _TokBos(1, 2)
    r = ad.answer_logfreq(OPT)
    assert r["prompt_token_id"] == 1
    for k in ("lm", "block0"):
        v = np.array(r[k]["values"])
        assert v.shape == (4,) and np.all(v < 0) and abs(r[k]["mean"] - v.mean()) < 1e-6
    with torch.no_grad():
        ref = torch.log_softmax(ad.model(torch.tensor([[1]])).logits[0, -1], -1)[OPT].numpy()
    assert np.allclose(r["lm"]["values"], ref, atol=1e-5)
    # block0 reading = lens on the embedding state
    S, _ = ad.collect_ids([[1]], OPT)
    full = torch.log_softmax(ad.unembed(ad.final_norm(torch.tensor(S[0, 0]))[None]).float(), -1)[0, OPT].detach().numpy()
    assert np.allclose(r["block0"]["values"], full, atol=1e-4)
    ad.tok = _TokBos(None, 2)  # no BOS: falls back to EOS
    assert ad.answer_logfreq(OPT)["prompt_token_id"] == 2


def test_kv_chunk_caps_by_cache_bytes():
    # pythia-2.8b-like (MHA, 32 layers, d 2560, fp32), 250-token prefix: 164 MB a row -> 4 GB fits 26 rows
    neox = GPTNeoXConfig(num_hidden_layers=32, hidden_size=2560, num_attention_heads=32)
    assert H.kv_row_bytes(neox, 250, 4) == 2 * 32 * 250 * 2560 * 4
    assert H.kv_chunk(neox, 250, 4, 128, 4 * 2**30) == 26
    # GQA (Qwen2.5-3B-like: 36 layers, 2 kv heads x 128) is not capped; a tiny budget still gives at least 1 row
    qwen = Qwen2Config(num_hidden_layers=36, hidden_size=2048, num_attention_heads=16, num_key_value_heads=2)
    assert H.kv_chunk(qwen, 250, 4, 128, 4 * 2**30) == 128
    assert H.kv_chunk(neox, 250, 4, 128, 1) == 1


def test_patched_logits_same_under_kv_cap(ad, monkeypatch):
    rng = np.random.default_rng(1)
    x = _ids(rng, 1, 8)[0]
    S, L = ad.collect_ids([x], OPT)
    rows = [(None, None)] + [(b, S[0, b]) for b in range(1, ad.n_blocks + 1)]
    full = ad.patched_logits(x, rows, OPT)
    monkeypatch.setenv("ATLAS_PATCH_KV_GB", "1e-9")  # forces 1 row per chunk
    assert np.allclose(ad.patched_logits(x, rows, OPT), full, atol=1e-5)
