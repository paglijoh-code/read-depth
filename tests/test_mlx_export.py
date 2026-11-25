"""M8 export on a tiny random Qwen3 (CPU, fp32): rebuild, export, and the real mlx-lm against PyTorch.

On macOS these fail (not skip) when mlx is absent; elsewhere the module is skipped. The MLX models are never mocked.
"""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

if sys.platform == "darwin":
    import mlx.core as mx
else:
    pytest.importorskip("mlx")
    import mlx.core as mx

import mlx_lm
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model as mlx_load_model
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from surgery import adapters, structure
from surgery import export as E

mx.set_default_device(mx.cpu)

L, NH, NKV, HD, D, V = 5, 8, 4, 16, 64, 32
ALIVE = [0, 1, 3, 4]
# excised numbering: layers keep 1, 3, 4 and 2 groups (non-contiguous pruned groups)
GROUPS = [(0, 0), (0, 2), (0, 3), (1, 1), (3, 0), (3, 2)]
IDS = [3, 7, 9, 1, 4, 11, 2, 30]


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=V, hidden_size=D, intermediate_size=128, num_hidden_layers=L,
                         num_attention_heads=NH, num_key_value_heads=NKV, head_dim=HD, tie_word_embeddings=True)
    model = Qwen3ForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():  # init std 0.02 makes logits ~0; use real-sized activations so 1e-3 means something
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (0.15 if p.ndim > 1 else 0.1) + (1.0 if p.ndim == 1 else 0.0))
    return model


@pytest.fixture
def tok():
    vocab = {f"w{i}": i for i in range(V - 1)} | {"[UNK]": V - 1}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]")


def lora_state(model, n, modules, rank, seed):
    """A random adapter state with the right keys and shapes (taken from a throwaway wrapper)."""
    probe = copy.deepcopy(model)
    h = adapters.add_lora(probe, range(n), modules, rank)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in h.modules:
            m.A.copy_(torch.randn(m.A.shape, generator=g) * 0.3)
            m.B.copy_(torch.randn(m.B.shape, generator=g) * 0.1)
    return adapters.adapter_state(h.modules)


CFG = {"lr": 1e-3, "steps": 1, "rank": 3, "modules": ["q_proj", "v_proj"]}


def final_for_folded(tiny, base, means, seed=12):
    probe = copy.deepcopy(tiny)
    structure.excise(probe, ALIVE)
    h = adapters.add_lora(probe, range(len(ALIVE)), CFG["modules"], CFG["rank"])
    adapters.load_adapter_state(h.modules, base)
    adapters.merge_lora(h)
    structure.fold_kv_groups(probe, means, GROUPS)
    return lora_state(probe, len(ALIVE), CFG["modules"], CFG["rank"], seed)


@pytest.fixture
def point(tiny):
    """(pristine copy, base state, final state, means) for the A-style point."""
    n = len(ALIVE)
    base = lora_state(tiny, n, CFG["modules"], CFG["rank"], 11)
    means = torch.randn(n, NH, HD, generator=torch.Generator().manual_seed(3)) / 5
    final = final_for_folded(tiny, base, means)
    return base, final, means


@torch.no_grad()
def pt_logits(model, ids=IDS):
    return model(input_ids=torch.tensor([ids])).logits[0].float().numpy()


def mlx_logits(path, ids=IDS, cache=False, trust=False):
    model, _ = mlx_load_model(Path(path), trust_remote_code=trust)
    x = mx.array(ids)[None]
    if not cache:
        out = model(x)[0]
    else:
        c = make_prompt_cache(model)
        model(x[:, :-1], cache=c)
        out = model(x[:, -1:], cache=c)[0]
    mx.eval(out)
    import numpy as np
    return np.array(out.astype(mx.float32))


# ---------------------------------------------------------------- required: a, b, c

def test_a_excised_merged_export_matches_pytorch_in_stock_mlx(tiny, tmp_path):
    h = adapters.add_lora(tiny, range(L), ["q_proj", "v_proj"], 3)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for m in h.modules:
            m.A.copy_(torch.randn(m.A.shape, generator=g) * 0.3)
            m.B.copy_(torch.randn(m.B.shape, generator=g) * 0.1)
    adapters.merge_lora(h)
    structure.excise(tiny, ALIVE)
    man = E.export_mlx(tiny, tmp_path / "m", "fp32")
    assert man["needs_model_file"] is False and man["n_layers"] == 4
    cfg = json.loads((tmp_path / "m" / "config.json").read_text())
    assert "model_file" not in cfg and "kv_heads_per_layer" not in cfg and cfg["num_hidden_layers"] == 4
    assert not (tmp_path / "m" / E.MODEL_FILE).exists()
    ref = pt_logits(tiny)
    assert ref.std() > 0.1
    out = mlx_logits(tmp_path / "m")
    assert out.shape == ref.shape and abs(out - ref).max() < 1e-3
    with pytest.raises(FileExistsError):
        E.export_mlx(tiny, tmp_path / "m", "fp32")
    assert not (tmp_path / "m.tmp").exists()


def test_b_uneven_kv_heads_and_o_bias_via_model_file(tiny, point, tok, tmp_path):
    base, final, means = point
    model = E.rebuild(copy.deepcopy(tiny), ALIVE, CFG, final, base, GROUPS, means)
    kv = [l.self_attn.k_proj.out_features // HD for l in model.model.layers]
    assert kv == [1, 3, 4, 2]
    ref = pt_logits(model)
    assert ref.std() > 0.1
    man = E.export_mlx(model, tmp_path / "m", "fp32", tokenizer=tok)
    assert man["needs_model_file"] and man["kv_heads_per_layer"] == kv
    cfg = json.loads((tmp_path / "m" / "config.json").read_text())
    assert cfg["model_file"] == E.MODEL_FILE and cfg["kv_heads_per_layer"] == kv
    assert cfg["o_proj_bias"] == [True, True, False, True]
    assert (tmp_path / "m" / E.MODEL_FILE).exists()
    with pytest.raises(ValueError):  # not loadable without opting in to the model file
        mlx_load_model(tmp_path / "m")
    # load_model path
    full = mlx_logits(tmp_path / "m", trust=True)
    assert abs(full - ref).max() < 1e-3
    # KV cache: prefill n-1 tokens, then one token
    cached = mlx_logits(tmp_path / "m", cache=True, trust=True)
    assert abs(cached[-1] - ref[-1]).max() < 1e-3
    # mlx_lm.load with the saved tokenizer
    m2, tk = mlx_lm.load(str(tmp_path / "m"), trust_remote_code=True)
    x = mx.array(IDS)[None]
    c = make_prompt_cache(m2)
    m2(x[:, :-1], cache=c)
    last = m2(x[:, -1:], cache=c)[0, -1]
    mx.eval(last)
    import numpy as np
    assert abs(np.array(last) - ref[-1]).max() < 1e-3
    assert abs(np.array(m2(x)[0]) - ref).max() < 1e-3
    assert [c_.keys.shape[1] for c_ in c] == kv  # per-layer KV heads in the cache
    # generate_step yields tokens
    toks = [int(t) for t, _ in generate_step(mx.array(IDS), m2, max_tokens=5,
                                             sampler=lambda p: mx.argmax(p, axis=-1))]
    assert len(toks) == 5 and all(0 <= t < V for t in toks)
    assert toks[0] == int(np.argmax(ref[-1]))


def test_c_parameter_counts_match_structure(tiny, point, tmp_path):
    cfg0 = copy.deepcopy(tiny.config)
    base, final, means = point
    model = E.rebuild(copy.deepcopy(tiny), ALIVE, CFG, final, base, GROUPS, means)
    man = E.export_mlx(model, tmp_path / "m", "fp32")
    shapes = {k: tuple(v.shape) for k, v in load_file(str(tmp_path / "m" / "model.safetensors")).items()}
    orig = [(ALIVE[layer], g) for layer, g in GROUPS]
    want = structure.param_count(cfg0, order=ALIVE, pruned_groups=orig)["total"]
    assert E.block_param_count(shapes) == want == man["block_params"]
    size = structure.model_size(model)["total"]
    assert sum(int(torch.Size(s).numel()) for s in shapes.values()) == size == man["n_params"]
    assert "lm_head.weight" not in shapes  # tied embedding stored once
    plain = structure.param_count(cfg0)["total"]  # unpruned reference differs
    assert plain > want


def test_untied_export_keeps_lm_head(tmp_path):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=V, hidden_size=D, intermediate_size=128, num_hidden_layers=2, num_attention_heads=NH,
                      num_key_value_heads=NKV, head_dim=HD, tie_word_embeddings=False)
    model = Qwen3ForCausalLM(cfg).eval()
    E.export_mlx(model, tmp_path / "m", "fp32")
    assert "lm_head.weight" in load_file(str(tmp_path / "m" / "model.safetensors"))


# ---------------------------------------------------------------- rebuild

def test_rebuild_matches_the_frontier_sequence(tiny, point):
    base, final, means = point
    got = E.rebuild(copy.deepcopy(tiny), ALIVE, CFG, final, base, GROUPS, means)
    ref = copy.deepcopy(tiny)  # the M7 test path, as written in frontier_sweep.build_point + stage_test
    structure.excise(ref, ALIVE)
    h = adapters.add_lora(ref, range(len(ALIVE)), CFG["modules"], CFG["rank"])
    adapters.load_adapter_state(h.modules, base)
    adapters.merge_lora(h)
    structure.fold_kv_groups(ref, means, GROUPS)
    h = adapters.add_lora(ref, range(len(ALIVE)), CFG["modules"], CFG["rank"])
    adapters.load_adapter_state(h.modules, final)
    adapters.merge_lora(h)
    for (k1, v1), (k2, v2) in zip(got.state_dict().items(), ref.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)
    assert abs(pt_logits(got) - pt_logits(ref)).max() == 0


def test_rebuild_without_groups_is_excise_plus_merge(tiny):
    state = lora_state(tiny, 4, CFG["modules"], CFG["rank"], 21)
    got = E.rebuild(copy.deepcopy(tiny), ALIVE, CFG, state)
    ref = copy.deepcopy(tiny)
    structure.excise(ref, ALIVE)
    h = adapters.add_lora(ref, range(4), CFG["modules"], CFG["rank"])
    adapters.load_adapter_state(h.modules, state)
    adapters.merge_lora(h)
    assert abs(pt_logits(got) - pt_logits(ref)).max() == 0
    assert abs(pt_logits(got) - pt_logits(tiny)).max() > 1e-3


def test_rebuild_raises_on_mismatched_adapter_state(tiny):
    good = lora_state(tiny, 4, CFG["modules"], CFG["rank"], 21)
    missing = {k: v for k, v in list(good.items())[1:]}
    extra = {**good, "99.A": torch.zeros(1)}
    wrong_rank = lora_state(tiny, 4, CFG["modules"], 2, 21)
    wrong_layers = lora_state(tiny, 3, CFG["modules"], CFG["rank"], 21)
    for bad in (missing, extra, wrong_rank, wrong_layers):
        with pytest.raises(ValueError, match="adapter state"):
            E.rebuild(copy.deepcopy(tiny), ALIVE, CFG, bad)


def test_rebuild_needs_means_for_groups(tiny, point):
    base, final, _ = point
    with pytest.raises(ValueError):
        E.rebuild(copy.deepcopy(tiny), ALIVE, CFG, final, base, GROUPS, None)


def test_load_point_with_injected_paths(tiny, tmp_path):
    ck, m5 = tmp_path / "ck", tmp_path / "m5"
    ck.mkdir(), m5.mkdir()
    (m5 / "excise_Qwen3-0.6B_rule.json").write_text(json.dumps({"removed_order": [2, 5, 6]}))
    E.MODELS["t"] = E.Spec("t", "rule", 1, "base.pt", "final.pt", CFG)
    try:
        assert E.alive_for("rule", 1, m5)[:3] == [0, 1, 3] and len(E.alive_for("rule", 1, m5)) == 27
        with pytest.raises(ValueError):
            E.alive_for("rule", 4, m5)
        # k=1 removes block 2 of a 28-block numbering; the tiny model has 5 blocks, so use the lora-style full path
        E.MODELS["t"] = E.Spec("t", "rule", 0, None, "final.pt", CFG)
        state = lora_state(tiny, L, CFG["modules"], CFG["rank"], 31)
        torch.save({"state": state}, ck / "final.pt")
        orig_n = E.N_LAYERS
        E.N_LAYERS = L
        try:
            got = E.load_point("t", ckpt_dir=ck, m5=m5, pristine=copy.deepcopy(tiny))
        finally:
            E.N_LAYERS = orig_n
        ref = E.rebuild(copy.deepcopy(tiny), list(range(L)), CFG, state)
        assert abs(pt_logits(got) - pt_logits(ref)).max() == 0
    finally:
        del E.MODELS["t"]


def test_registry_matches_contract():
    assert len(E.MODELS) == 8
    a = E.MODELS["rule-A"]
    assert (a.k, a.base, a.final) == (12, "rule_k12_tune_lr0.001_st300_r10_qv.pt",
                                      "rule_k12_g48_lr0.001_st300_r10_qv_s0.pt")
    assert E.MODELS["sensible-A"].final == "sensible_k12_g64_lr0.001_st300_r10_qv_s0.pt"
    assert E.MODELS["rule-B"].final == "rule_k18_tune_lr0.003_st300_r10_qv.pt"
    lora = E.MODELS["sensible-lora-1m"]
    assert lora.final == "sensible_recheck_lora_1m_lr0.0003_st300.pt" and lora.cfg["rank"] == 7 and lora.k == 0
    assert E.MODELS["rule-k12"].final == "rule_k12_tune_lr0.001_st300_r10_qv.pt"
    assert E.variants_of("unmodified") == ["rule", "sensible"] and E.variants_of("rule-A") == ["rule"]


# ---------------------------------------------------------------- dtype, parity gate

def test_bf16_export_has_bf16_tensors_and_loads(tiny, tmp_path):
    structure.excise(tiny, ALIVE)
    man = E.export_mlx(tiny, tmp_path / "m", "bf16")
    w = load_file(str(tmp_path / "m" / "model.safetensors"))
    assert all(v.dtype == torch.bfloat16 for v in w.values())
    assert man["weight_bytes"] < E.export_mlx(tiny, tmp_path / "f", "fp32")["weight_bytes"]
    model, _ = mlx_load_model(tmp_path / "m")
    out = model(mx.array(IDS)[None])
    mx.eval(out)
    assert out.dtype == mx.bfloat16
    import numpy as np
    got = np.array(out.astype(mx.float32))
    assert np.isfinite(got).all() and abs(got[0] - pt_logits(tiny)).max() < 0.5  # bf16 rounding, loose


def test_export_rejects_unknown_dtype(tiny, tmp_path):
    with pytest.raises(ValueError):
        E.export_mlx(tiny, tmp_path / "m", "fp16")


def test_parity_gate_edges():
    a = [0] * 100
    b99 = [0] * 99 + [1]
    b98 = [0] * 98 + [1, 1]
    assert E.parity_gate(a, a, 0.9, 0.9)["passed"]
    assert E.parity_gate(a, b99, 0.9, 0.9)["passed"]  # exactly 99%
    assert not E.parity_gate(a, b98, 0.9, 0.9)["passed"]
    assert E.parity_gate(a, a, 0.905, 0.9)["passed"]  # exactly 0.5 points
    assert not E.parity_gate(a, a, 0.9051, 0.9)["passed"]
    assert not E.parity_gate(a, a, 0.894, 0.9)["passed"]
    assert E.parity_gate(a, b99, 0.9, 0.9)["agreement"] == pytest.approx(0.99)
    with pytest.raises(ValueError):
        E.parity_gate([], [], 0.0, 0.0)
    with pytest.raises(ValueError):
        E.parity_gate([0], [0, 1], 0.0, 0.0)


def test_block_param_count_ignores_norms_embedding_and_bias():
    w = {"model.embed_tokens.weight": (32, 64), "model.layers.0.self_attn.q_proj.weight": (128, 64),
         "model.layers.0.self_attn.o_proj.bias": (64,), "model.layers.0.self_attn.q_norm.weight": (16,),
         "model.layers.0.mlp.down_proj.weight": (64, 128), "model.norm.weight": (64,)}
    assert E.block_param_count(w) == 128 * 64 + 64 * 128
