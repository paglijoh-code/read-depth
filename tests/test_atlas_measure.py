"""Atlas store, block lesion, block influence and perplexity on a tiny random Qwen3 with a stub tokenizer."""
import json

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from surgery.atlas import measure, spec

TASK = "code_shape"  # options loop / branch / neither: first characters l, b, n


class Batch(dict):
    def to(self, device):
        return Batch({k: v.to(device) for k, v in self.items()})


class StubTok:
    """Character-level tokenizer into the tiny vocab (ord % 32); left padding with id 0."""
    pad_token_id = 0

    @staticmethod
    def _ids(text):
        return [ord(c) % 32 for c in text]

    def apply_chat_template(self, messages, **kw):
        return "|".join(m["content"] for m in messages)

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def __call__(self, text, add_special_tokens=False, return_tensors="pt", padding=False, **kw):
        texts = [text] if isinstance(text, str) else list(text)
        seqs = [self._ids(t) for t in texts]
        n = max(map(len, seqs))
        ids = torch.zeros(len(seqs), n, dtype=torch.long)
        mask = torch.zeros(len(seqs), n, dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, n - len(s):] = torch.tensor(s)
            mask[r, n - len(s):] = 1
        return Batch(input_ids=ids, attention_mask=mask)


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    model = Qwen3ForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    return model


@pytest.fixture
def records():
    texts = ["for i in range(3): x", "if a: b = 1 else", "x = 1 y = 2", "while k: k -= 1", "a = b + c", "if q: pass",
             "for z in y: z", "m = n"]
    return [{"id": f"r{i}", "task": TASK, "prompt": t, "answer": i % 3, "template_id": f"t{i % 4}",
             "rung": None if i % 2 else "r", "meta": {}} for i, t in enumerate(texts)]


def test_store_shapes_lens_equals_final_and_means(tiny, records, tmp_path):
    tok = StubTok()
    meta = measure.build_store(tiny, tok, TASK, records, tmp_path, batch_size=3)
    L, d, n = 4, 64, len(records)
    resid = np.load(tmp_path / "resid.npy")
    lens = np.load(tmp_path / "lens.npy")
    final = np.load(tmp_path / "final_logits.npy")
    assert resid.shape == (n, L + 1, d) and resid.dtype == np.float16
    assert lens.shape == (n, L + 1, 3) and final.shape == (n, 3)
    assert np.abs(lens[:, -1] - final).max() < 1e-3  # last-layer lens is the model's own option logits
    assert np.isfinite(lens).all()
    means = torch.load(tmp_path / "means.pt", weights_only=True)
    assert means["attn"].shape == (L, d) and means["mlp"].shape == (L, d)
    saved = json.loads((tmp_path / "meta.json").read_text())
    assert saved["ids"] == [r["id"] for r in records] and saved["answers"] == [0, 1, 2, 0, 1, 2, 0, 1]
    assert saved["rungs"][0] == "r" and saved["rungs"][1] is None
    assert saved["chance"] == pytest.approx(3 / 8) and saved["pred"] == meta["pred"] == final.argmax(-1).tolist()
    # batch size does not change the store (left padding is handled)
    measure.build_store(tiny, tok, TASK, records, tmp_path / "b1", batch_size=1)
    assert np.abs(np.load(tmp_path / "b1" / "lens.npy") - lens).max() < 1e-3


def test_block_lesion_replaces_both_sublayers_and_removes_hooks(tiny, records, tmp_path):
    tok = StubTok()
    measure.build_store(tiny, tok, TASK, records, tmp_path)
    means = torch.load(tmp_path / "means.pt", weights_only=True)
    base = measure.option_logits(tiny, tok, TASK, records)
    from surgery.hooks import apply_spec

    with apply_spec(tiny, measure.block_spec(1), means=means):
        both = measure.option_logits(tiny, tok, TASK, records)
    with apply_spec(tiny, {"op": "ablate", "site": "attn", "mode": "mean", "layers": [1]}, means=means):
        attn_only = measure.option_logits(tiny, tok, TASK, records)
    with apply_spec(tiny, {"op": "ablate", "site": "mlp", "mode": "mean", "layers": [1]}, means=means):
        mlp_only = measure.option_logits(tiny, tok, TASK, records)
    assert np.abs(both - base).max() > 1e-4
    assert np.abs(both - attn_only).max() > 1e-4 and np.abs(both - mlp_only).max() > 1e-4

    # the block output is exactly the input plus the two mean vectors (both sites replaced at once)
    seen = {}
    h = tiny.model.layers[1].register_forward_hook(lambda m, a, k, o: seen.update(x=k.get("hidden_states", a[0] if a else None),
                                                                                  y=o[0] if isinstance(o, tuple) else o),
                                                   with_kwargs=True)
    enc = tok(["for i in range(3): x"])
    with apply_spec(tiny, measure.block_spec(1), means=means), torch.no_grad():
        tiny(**enc)
    h.remove()
    expect = seen["x"] + means["attn"][1] + means["mlp"][1]
    assert (seen["y"] - expect).abs().max() < 1e-4

    # hooks are gone afterwards
    assert all(not m._forward_hooks for m in (tiny.model.layers[1], tiny.model.layers[1].self_attn, tiny.model.layers[1].mlp))
    assert np.array_equal(measure.option_logits(tiny, tok, TASK, records), base)


def test_lesion_sweep_resumable_and_restores(tiny, records, tmp_path):
    tok = StubTok()
    measure.build_store(tiny, tok, TASK, records, tmp_path)
    means = torch.load(tmp_path / "means.pt", weights_only=True)
    base = measure.option_logits(tiny, tok, TASK, records)
    path = tmp_path / "lesion.json"
    part = measure.lesion_sweep(tiny, tok, TASK, records, means, path=path, blocks=[0, 1])
    assert [r["block"] for r in part["rows"]] == [0, 1] and len(part["drops"]) == 2
    first_seconds = part["rows"][0]["seconds"]
    full = measure.lesion_sweep(tiny, tok, TASK, records, means, path=path)
    assert len(full["drops"]) == 4 and full["rows"][0]["seconds"] == first_seconds  # blocks 0,1 not recomputed
    answers = np.array([r["answer"] for r in records])
    assert full["base_accuracy"] == pytest.approx(float((base.argmax(-1) == answers).mean()))
    for row in full["rows"]:
        assert row["drop"] == pytest.approx(full["base_accuracy"] - row["accuracy"])
    assert np.array_equal(measure.option_logits(tiny, tok, TASK, records), base)
    with pytest.raises(ValueError):
        measure.lesion_sweep(tiny, tok, TASK, records[:4], means, path=path)


TEXT = "the quick brown fox jumps over the lazy dog. " * 12


def test_block_influence_matches_direct_computation(tiny):
    tok = StubTok()
    before = [len(m._forward_hooks) for m in tiny.model.layers]
    bi = measure.block_influence(tiny, tok, TEXT, window=64)
    assert [len(m._forward_hooks) for m in tiny.model.layers] == before  # our hooks are removed
    assert len(bi) == 4 and all(-1e-6 <= v <= 2 + 1e-6 for v in bi)
    # one window: compare with hidden states from output_hidden_states (blocks 0..2 inputs/outputs are unnormalised)
    ids = tok(TEXT[:60])["input_ids"]
    with torch.no_grad():
        hs = tiny(input_ids=ids, output_hidden_states=True).hidden_states
    for i in range(3):
        c = torch.nn.functional.cosine_similarity(hs[i], hs[i + 1], dim=-1).mean().item()
        assert measure.block_influence(tiny, tok, TEXT[:60], window=64)[i] == pytest.approx(1 - c, abs=1e-5)


def test_ppl_drop_skip_changes_ppl_and_restores(tiny):
    tok = StubTok()
    out = measure.ppl_drop(tiny, tok, TEXT, max_tokens=200)
    assert out["n_tokens"] == 200 and len(out["drop"]) == 4 and out["base_ppl"] > 1
    assert any(abs(x) > 1e-6 for x in out["drop"])
    again = measure.ppl_drop(tiny, tok, TEXT, max_tokens=200, blocks=[2])
    assert again["base_ppl"] == pytest.approx(out["base_ppl"])
    assert again["drop"][0] == pytest.approx(out["drop"][2])
    rows = []
    measure.ppl_drop(tiny, tok, TEXT, max_tokens=200, blocks=[0, 1], done={0: out["ppl"][0]},
                     on_row=lambda b, p: rows.append(dict(p)))
    assert len(rows) == 1 and set(rows[0]) == {0, 1}  # block 0 reused, only block 1 computed
