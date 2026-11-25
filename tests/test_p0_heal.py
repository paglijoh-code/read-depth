import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from surgery import adapters
from surgery.atlas import p0_tasks as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import p0_heal as P  # noqa: E402

V = 3000


class _Tok:
    """No-download tokenizer: words, single digits and punctuation, each a hashed id (a leading space is part of the word)."""

    def __call__(self, text, add_special_tokens=False):
        parts = re.findall(r"\s?[A-Za-z]+|\d|[^\sA-Za-z\d]", text)
        return type("E", (), {"input_ids": [zlib.crc32(p.encode()) % V for p in parts]})()


def _tiny(seed=0):
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256))


@pytest.mark.parametrize("name", T.TASK_NAMES)
def test_lora_split_train_templates_only_and_deterministic(name):
    items = T.generate(name, seed=0)
    train_t = {x["template"] for x in items if x["split"] == "train"}
    test_t = {x["template"] for x in items if x["split"] == "test"}
    lo = T.generate_lora(name)
    assert len(lo) == 2300 and lo == T.generate_lora(name)
    assert {x["template"] for x in lo} <= train_t and not ({x["template"] for x in lo} & test_t)
    assert {x["split"] for x in lo} == {"lora"}
    assert not ({x["id"] for x in lo} & {x["id"] for x in items})
    assert 0 < T.unique_prompts(lo) <= 2300
    assert lo != T.generate_lora(name, seed=1)
    assert items == T.generate(name, seed=0)  # existing splits unchanged by the new function


def test_lora_split_uses_train_side_class_words():
    tr = {w for w in T.COUNTRIES[0] if w in set(T._split_words(T.COUNTRIES[0], "train"))}
    te = set(T._split_words(T.COUNTRIES[0], "test"))
    c = [x["slots"]["c"] for x in T.generate_lora("fact_continent") if x["answer"] == 0]
    assert set(c) <= tr and not (set(c) & te)


def _setup(seed=0):
    model, tok = _tiny(seed), _Tok()
    opt_ids = T.check_options(tok, "fact_color")
    (x, y), _, _ = P.task_data(tok, "fact_color", limit_train=16)
    return model, tok, opt_ids, x, y


def _randomise(handle, seed=3):
    g = torch.Generator().manual_seed(seed)
    for n, p in handle.modules.named_parameters():
        if "base_layer." not in n:
            p.data = torch.randn(p.shape, generator=g) * 0.2


def test_adapter_round_trip_gives_same_logits(tmp_path):
    model, tok, opt_ids, x, y = _setup()
    h = P.attach_lora(model, seed=1)
    _randomise(h)
    with torch.no_grad():
        before = P.option_logits(model, x, opt_ids, "cpu")
    P.save_adapter(h, tmp_path / "adapter.safetensors")
    h.remove()
    h2 = P.attach_lora(model, seed=2)  # different init
    P.load_adapter(h2, tmp_path / "adapter.safetensors")
    with torch.no_grad():
        after = P.option_logits(model, x, opt_ids, "cpu")
    assert torch.allclose(before, after, atol=1e-6)
    base = _tiny()
    h3 = P.attach_lora(base)
    h3.remove()
    with torch.no_grad():
        assert not torch.allclose(P.option_logits(base, x, opt_ids, "cpu"), before, atol=1e-3)


def test_merged_logits_equal_unmerged(tmp_path):
    model, tok, opt_ids, x, y = _setup()
    h = P.attach_lora(model, seed=1)
    _randomise(h)
    with torch.no_grad():
        unmerged = P.option_logits(model, x, opt_ids, "cpu")
    P.save_adapter(h, tmp_path / "a.safetensors")
    fresh = _tiny()
    h2 = P.attach_lora(fresh)
    P.load_adapter(h2, tmp_path / "a.safetensors")
    assert adapters.merge_lora(h2) == 6
    with torch.no_grad():
        merged = P.option_logits(fresh, x, opt_ids, "cpu")
    assert torch.allclose(unmerged, merged, atol=1e-4)


def test_right_padded_batch_matches_single_prompts():
    model, tok, opt_ids, x, y = _setup()
    with torch.no_grad():
        both = P.option_logits(model, x, opt_ids, "cpu")
        one = torch.cat([P.option_logits(model, [i], opt_ids, "cpu") for i in x])
    assert torch.allclose(both, one, atol=1e-4)


def test_heal_task_runs_and_leaves_model_clean(tmp_path):
    model, tok = _tiny(), _Tok()
    ref = model.model.layers[0].self_attn.q_proj
    r = P.heal_task(model, tok, "fact_color", 1e-3, 0, tmp_path, "cpu", steps=3, limit_train=16, log=lambda *a: None)
    assert model.model.layers[0].self_attn.q_proj is ref
    d = tmp_path / "fact_color_lr0.001_s0"
    assert (d / "adapter.safetensors").exists() and (d / "heal.json").exists()
    assert {"wall_seconds", "steps", "lr", "seed", "dev_acc", "test_acc", "base_test_acc", "device"} <= set(r)


def test_pick_lr_takes_higher_mean_dev_acc():
    rows = [{"lr": 1e-3, "dev_acc": a} for a in (0.9, 0.5, 0.5, 0.5)] + [{"lr": 3e-4, "dev_acc": a} for a in (0.6, 0.6, 0.6, 0.6)]
    assert P.pick_lr(rows) == 3e-4  # mean 0.6 beats 0.6; tie goes to 3e-4
    rows[0]["dev_acc"] = 1.0
    assert P.pick_lr(rows) == 1e-3  # mean 0.625 beats 0.6, though 3e-4 wins 3 of 4 tasks
    rows[0]["dev_acc"] = 0.5
    assert P.pick_lr(rows) == 3e-4


def test_run_plan_selects_lr_and_logs(tmp_path):
    acc = {1e-3: 0.9, 3e-4: 0.4}
    calls = []

    def fake(task, lr, seed):
        calls.append((task, lr, seed))
        return {"task": task, "lr": lr, "seed": seed, "wall_seconds": 1.0, "dev_acc": acc[lr], "test_acc": 0.5}

    done = P.run_plan(fake, tmp_path, log=lambda *a: None)
    assert len(calls) == 12 and done["n_heals"] == 12 and done["chosen_lr"] == 1e-3
    assert [c[2] for c in calls] == [0] * 8 + [1] * 4 and {c[1] for c in calls[8:]} == {1e-3}
    assert {c[0] for c in calls} == set(P.PLAN_TASKS)
    lines = (tmp_path / "heal_log.tsv").read_text().splitlines()
    assert len(lines) == 12 and len(lines[0].split("\t")) == 8
    assert (tmp_path / "plan_done.json").exists()
