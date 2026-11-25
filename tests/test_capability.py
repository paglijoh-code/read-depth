"""Capability-study pieces on a tiny random Qwen3: distillation KL, combined training, bf16 checkpoints, the
general-data filter, harder-tier generation, the pick rule and verdict bars."""
import copy
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from surgery import adapters, capability as C
from surgery.task.generator import LABELS, Knobs, generate, next_action
from surgery.task.splits import _partition, make_splits

L, D, V = 3, 32, 64


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=V, hidden_size=D, intermediate_size=64, num_hidden_layers=L,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = Qwen3ForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    return model


class Enc(dict):
    def to(self, device):
        return Enc({k: v.to(device) for k, v in self.items()})


class CharTok:
    """Characters mod V as ids, 0 as padding; honours padding_side like a HF tokenizer."""
    padding_side = "left"

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True, max_length=128):
        ids = [[1 + ord(c) % (V - 1) for c in t][:max_length] for t in texts]
        n = max(map(len, ids))
        x, m = torch.zeros(len(ids), n, dtype=torch.long), torch.zeros(len(ids), n, dtype=torch.long)
        for i, s in enumerate(ids):
            lo = n - len(s) if self.padding_side == "left" else 0
            x[i, lo:lo + len(s)], m[i, lo:lo + len(s)] = torch.tensor(s), 1
        return Enc(input_ids=x, attention_mask=m)


def task_rows(n=16):
    g = torch.Generator().manual_seed(1)
    return [{"x": torch.randint(1, V, (6,), generator=g).tolist(), "label": LABELS[i % len(LABELS)]} for i in range(n)]


def make_forward(model):
    def fwd(recs):
        x = torch.tensor([r["x"] for r in recs])
        return model(input_ids=x, use_cache=False).logits[:, -1, :len(LABELS)].float()
    return fwd


GEN = [{"text": "def f(x):\n    return x + %d  # some code" % i} for i in range(12)]


# ---------------------------------------------------------------- KL

def test_kl_zero_for_identical_and_positive_otherwise():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(2, 5, V, generator=g)
    b = torch.randn(2, 5, V, generator=g)
    mask = torch.ones(2, 5, dtype=torch.long)
    assert C.distill_kl(a, a, mask).abs() < 1e-6
    assert C.distill_kl(a, b, mask) > 0


def test_kl_ignores_masked_positions():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(2, 5, V, generator=g), torch.randn(2, 5, V, generator=g)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    a2, b2 = a.clone(), b.clone()
    a2[mask == 0], b2[mask == 0] = torch.randn(int((mask == 0).sum()), V), torch.randn(int((mask == 0).sum()), V)
    assert torch.allclose(C.distill_kl(a, b, mask), C.distill_kl(a2, b2, mask))
    # temperature scaling: T^2 times the per-token KL at T
    t = C.distill_kl(a, b, mask, T=2.0)
    s, te = torch.log_softmax(a / 2, -1), torch.log_softmax(b / 2, -1)
    manual = (4 * (te.exp() * (te - s)).sum(-1))[mask == 1].mean()
    assert torch.allclose(t, manual, atol=1e-6)


def test_general_dev_kl_zero_for_teacher_copy(tiny):
    teacher = copy.deepcopy(tiny)
    assert C.general_dev_kl(tiny, teacher, CharTok(), GEN, batch_size=4) < 1e-6


# ---------------------------------------------------------------- combined training

def frozen_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("model.layers.")}


def test_full_trainable_names_and_freezing(tiny):
    named = C.full_trainable(tiny)
    assert {n for n, _ in named} == {n for n, p in tiny.named_parameters() if n.startswith("model.layers.")}
    assert all(p.requires_grad for _, p in named)
    assert not tiny.model.embed_tokens.weight.requires_grad and not tiny.lm_head.weight.requires_grad
    assert not tiny.model.norm.weight.requires_grad


def test_train_combined_changes_layers_only(tiny):
    teacher = copy.deepcopy(tiny)
    teacher_before = {n: p.detach().clone() for n, p in teacher.named_parameters()}
    before_frozen = frozen_state(tiny)
    before_layers = {n: p.detach().clone() for n, p in tiny.model.layers.named_parameters()}
    named = C.full_trainable(tiny)
    rows = task_rows()
    state, curve = C.train_combined(tiny, teacher, CharTok(), rows, rows[:8], GEN, GEN[:4], [p for _, p in named],
                                    make_forward(tiny), steps=6, lr=1e-2, lam=1.0, batch_size=4, n_evals=3)
    for n, p in tiny.named_parameters():
        if n in before_frozen:
            assert torch.equal(p, before_frozen[n]), n
    assert any(not torch.equal(p, before_layers[n]) for n, p in tiny.model.layers.named_parameters())
    assert set(state) == {f"model.layers.{n}" for n in before_layers}
    assert [c["step"] for c in curve] == [2, 4, 6] and sum(c["kept"] for c in curve) == 1
    assert all(c["general_dev_kl"] >= 0 for c in curve)
    assert all(not p.requires_grad for p in teacher.parameters())  # train_combined freezes the teacher
    assert all(torch.equal(p, teacher_before[n]) for n, p in teacher.named_parameters())


def test_zero_task_weight_skips_task_loss(tiny):
    teacher = copy.deepcopy(tiny)
    named = C.full_trainable(tiny)
    rows = task_rows()
    _, curve = C.train_combined(tiny, teacher, CharTok(), rows, rows[:4], GEN, GEN[:4], [p for _, p in named],
                                make_forward(tiny), steps=2, lr=1e-2, lam=1.0, batch_size=4, n_evals=1, task_weight=0.0)
    assert curve[0]["task_loss"] == 0.0 and curve[0]["kept"]


def test_checkpoint_roundtrip_within_bf16_rounding(tiny, tmp_path):
    original = copy.deepcopy(tiny)
    teacher = copy.deepcopy(tiny)
    named = C.full_trainable(tiny)
    rows = task_rows()
    state, _ = C.train_combined(tiny, teacher, CharTok(), rows, rows[:4], GEN, GEN[:4], [p for _, p in named],
                                make_forward(tiny), steps=4, lr=1e-2, lam=1.0, batch_size=4, n_evals=2)
    path = tmp_path / "t.pt"
    C.save_trainables(path, state)
    assert all(v.dtype == torch.bfloat16 for v in torch.load(path)["state"].values())
    fresh = copy.deepcopy(original)
    assert C.load_trainables(fresh, path) == len(state)
    x = torch.tensor([[3, 5, 7, 9, 11, 13]])
    with torch.no_grad():
        got, trained = fresh(input_ids=x).logits, tiny(input_ids=x).logits
        expected = copy.deepcopy(original)
        for n, p in expected.named_parameters():
            if n in state:
                p.copy_(state[n].to(torch.bfloat16).to(p.dtype))
        assert torch.allclose(got, expected(input_ids=x).logits, atol=1e-6)
    assert (got - trained).abs().max() < 0.1  # bf16 rounding only
    with pytest.raises(KeyError):
        C.load_trainables(fresh, {"model.layers.0.nope": torch.zeros(1)})


def test_lora_training_then_merge_matches_unmerged(tiny):
    teacher = copy.deepcopy(tiny)
    h = adapters.add_lora(tiny, range(L), ("q_proj", "k_proj", "v_proj", "o_proj"), r=3)
    params = [p for p in h.modules.parameters() if p.requires_grad]
    rows = task_rows()
    state, _ = C.train_combined(tiny, teacher, CharTok(), rows, rows[:4], GEN, GEN[:4], params, make_forward(tiny),
                                steps=4, lr=1e-2, lam=1.0, batch_size=4, n_evals=2)
    assert all(k.endswith((".A", ".B")) for k in state)
    C.load_trainables(tiny, state)
    x = torch.tensor([[3, 5, 7, 9, 11, 13]])
    with torch.no_grad():
        unmerged = tiny(input_ids=x).logits
    adapters.merge_lora(h)
    with torch.no_grad():
        assert (tiny(input_ids=x).logits - unmerged).abs().max() < 1e-4


# ---------------------------------------------------------------- general-data filter

def test_drop_reason():
    canaries = ["Which of these strings has balanced brackets?", "What is the capital city of France?"]
    ppl = " ".join(f"w{i}" for i in range(40))
    flt = C.filter_sets(canaries, ppl)
    assert C.drop_reason("Hello.\n  WHICH of these STRINGS has   balanced brackets? yes", None, *flt) == "canary_prompt"
    near = "Which of these strings has balanced brackets or not?"
    assert C.drop_reason("Question: " + near + "\nAnswer: x", near, *flt) == "canary_jaccard"
    run = " ".join(f"w{i}" for i in range(10, 22))  # a shared 12-word run
    assert C.drop_reason("prefix words then " + run + " and more", None, *flt) == "ppl_overlap"
    clean = "Question: How do I bake bread?\nAnswer: Mix flour, water, salt and yeast, then bake."
    assert C.drop_reason(clean, "How do I bake bread?", *flt) is None
    assert C.drop_reason("w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11", None, *flt) is None  # 11 words: no 12-word shingle


# ---------------------------------------------------------------- tiers

@pytest.mark.parametrize("tier", list(C.TIERS))
def test_tier_generates_valid_balanced_examples(tier):
    knobs = C.TIERS[tier]
    recs = generate(240, 7, knobs)
    assert len(recs) == 240
    assert {l: sum(r["label"] == l for r in recs) for l in LABELS} == {l: 40 for l in LABELS}
    assert all(next_action(r["state"]) == r["label"] for r in recs)
    assert recs[0]["knobs"] == asdict(knobs)
    assert len(recs[0]["state"]["rules"]) <= knobs.n_rules


def test_tier_tuples_match_contract():
    k = C.TIERS["R1"]
    assert (k.n_rules, k.cond_size, k.n_distractor_files, k.n_distractor_tests, k.history_extra, k.conflict_rate) == \
        (3, 2, 3, 3, 3, 0.4)
    k = C.TIERS["R3"]
    assert (k.n_rules, k.cond_size, k.n_distractor_files, k.history_extra, k.conflict_rate) == (6, 3, 5, 5, 0.6)
    assert C.TIERS["R2"].n_rules == 4 and C.TIERS["R4"].cond_size == 4 and C.TIERS["R4"].conflict_rate == 0.7


def test_make_splits_default_unchanged_and_knobs_pass_through():
    sizes = {"a": 300, "b": 300}
    direct = _partition(generate(2 * sum(sizes.values()), 0, variant="rule"), 0, sizes)
    assert direct is not None
    assert make_splits("rule", 0, sizes) == direct
    assert make_splits("rule", 0, sizes, knobs=None) == direct
    tiered = make_splits("rule", 0, sizes, knobs=C.TIERS["R2"])
    assert tiered["a"][0]["knobs"] == asdict(C.TIERS["R2"]) and tiered != direct
    assert direct["a"][0]["knobs"] == asdict(Knobs())


# ---------------------------------------------------------------- pick rule and verdicts

def row(name, heldout, kl, **kw):
    return {"name": name, "heldout": heldout, "dev_kl": kl, **kw}


def test_pick_a_lowest_kl_among_gate_never_ceiling_ignores_canaries():
    rows = [row("a", 0.995, 0.30, canaries={"accuracy": 0.9}), row("b", 0.99, 0.20, canaries={"accuracy": 0.1}),
            row("c", 0.98, 0.01), row("ceil", 1.0, 0.001, ceiling=True)]
    pick, fb = C.pick_a(rows)
    assert pick["name"] == "b" and fb is None
    rows[0]["canaries"], rows[1]["canaries"] = {"accuracy": 0.1}, {"accuracy": 0.9}
    assert C.pick_a(rows)[0]["name"] == "b"


def test_pick_a_none_returns_highest_heldout_fallback():
    rows = [row("a", 0.80, 0.1), row("b", 0.95, 0.5), row("ceil", 1.0, 0.0, ceiling=True)]
    pick, fb = C.pick_a(rows)
    assert pick is None and fb["name"] == "b"
    assert C.pick_a([row("ceil", 1.0, 0.0, ceiling=True)]) == (None, None)


def test_rank_key_gate_then_kl():
    assert C.rank_key(0.99, 0.5) > C.rank_key(0.98, 0.0)
    assert C.rank_key(0.995, 0.1) > C.rank_key(0.995, 0.2)
    assert C.rank_key(0.9, 9.0) > C.rank_key(0.8, 0.0)
    assert C.rank_key(0.0, 0.1, task_weight=0) > C.rank_key(1.0, 0.2, task_weight=0)


def test_verdict_a_bars_at_their_edges():
    assert C.verdict_a(0.97, 0.58) == "supported"
    assert C.verdict_a(0.97, 0.579) == "partial"
    assert C.verdict_a(0.97, 0.48) == "partial"
    assert C.verdict_a(0.97, 0.479) == "not supported"
    assert C.verdict_a(0.969, 0.9) == "not supported"
    assert C.verdict_a(1.0, 290 / 500) == "supported" and C.verdict_a(1.0, 240 / 500) == "partial"


def test_verdict_b_bars_at_their_edges():
    assert C.verdict_b_capacity(0.90) == "300 steps was the limit"
    assert C.verdict_b_capacity(0.899) == "out of capacity"
    assert C.verdict_b_cut(0.80, 0.75) == "cutting stays free"
    assert C.verdict_b_cut(0.80, 0.749) == "cutting costs capability"
    assert C.verdict_b_cut(0.70, 0.75) == "cutting stays free"


# ---------------------------------------------------------------- the pod script

def test_capability_sweep_help_loads_no_model(monkeypatch, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import capability_sweep as CS

    def boom(*a, **k):
        raise AssertionError("loaded a model")
    monkeypatch.setattr(CS.F, "load_model", boom)
    with pytest.raises(SystemExit) as e:
        CS.main(["--help"])
    assert e.value.code == 0 and "--stage" in capsys.readouterr().out
    assert CS.a_key(12, 48, CS.mk_spec("lora", 1e-3, 0.3), 1) == "k12|g48|lora_lr0.001_lam0.3|s1"
