"""Fast tests for surgery/atlas/gates.py: toy adapter and tiny random HF models, a fake tokenizer, no downloads."""
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, Qwen2Config, Qwen2ForCausalLM

from surgery import adapters
from surgery.atlas import families as fam
from surgery.atlas import gates as G
from surgery.atlas import p0_measures as M
from surgery.atlas.p0_harness import Adapter, analyze_task
from surgery.atlas.p0_toy import ToyAdapter, make_toy_items

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import p0_heal  # noqa: E402

V = 3000


class _Tok:
    """No-download tokenizer: words, single digits and punctuation, each a hashed id (a leading space is part of the word)."""

    bos_token_id, eos_token_id = 1, 2

    def __call__(self, text, add_special_tokens=False):
        parts = re.findall(r"\s?[A-Za-z]+|\d|[^\sA-Za-z\d]", text)
        return type("E", (), {"input_ids": [zlib.crc32(p.encode()) % V for p in parts]})()


def _tiny(arch, seed=0):
    torch.manual_seed(seed)
    kw = dict(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
              max_position_embeddings=1024)
    if arch == "neox":
        return GPTNeoXForCausalLM(GPTNeoXConfig(**kw))
    return Qwen2ForCausalLM(Qwen2Config(**kw, num_key_value_heads=2))


@pytest.fixture(scope="module")
def tasks():
    """Three family tasks whose options are single tokens under the fake tokenizer and whose prompts have no newline."""
    out, tok = [], _Tok()
    for n in fam.TASK_NAMES:
        try:
            fam.check_options(tok, fam.get_task(n))
        except ValueError:
            continue
        if not any("\n" in x["prompt"] for x in fam.generate(n, 0)):
            out.append(n)
        if len(out) == 3:
            break
    assert len(out) == 3
    return out


@pytest.fixture(scope="module")
def ad():
    return Adapter(model=_tiny("qwen2"), tok=_Tok(), device="cpu")


# ---------------------------------------------------------------- items

def test_lora_items_train_templates_every_family():
    for n in fam.TASK_NAMES:
        tr_t, te_t = G.train_test_templates(n)
        te_gen = {x["template"] for x in fam.generate(n, 0) if x["split"] == "test"}
        assert set(te_t) == te_gen
        lo = G.lora_items(n)
        assert len(lo) == 2300 and {x["split"] for x in lo} == {"lora"}
        assert {x["template"] for x in lo} <= set(tr_t) and not ({x["template"] for x in lo} & set(te_t))
        assert np.bincount([x["answer"] for x in lo], minlength=4).tolist() == [575] * 4
        assert lo[0]["id"] == f"{n}-lora-0"
    n = fam.TASK_NAMES[0]
    assert G.lora_items(n) == G.lora_items(n) and G.lora_items(n) != G.lora_items(n, seed=1)


def test_fresh_items_test_templates_disjoint_balanced():
    for n in fam.TASK_NAMES[:12]:
        gen = fam.generate(n, 0)
        te_t = {x["template"] for x in gen if x["split"] == "test"}
        fr = G.fresh_items(n)
        assert len(fr) == 200 and fr == G.fresh_items(n)
        assert {x["template"] for x in fr} == te_t
        assert fr[3]["id"] == f"{n}-fresh-3"
        assert not ({x["id"] for x in fr} & {x["id"] for x in gen})
        assert np.bincount([x["answer"] for x in fr], minlength=4).tolist() == [50] * 4
        seen = {x["prompt"] for x in gen}
        assert sum(x["prompt"] in seen for x in fr) <= 4, n  # redraws make a clash rare, even for small pools
        assert all(x["ans_twin"]["answer"] != x["answer"] for x in fr)


def test_exemplar_classes_balanced_and_independent_of_answer(tasks):
    n = tasks[0]
    c = np.array([G.exemplar_classes(n, i) for i in range(400)])
    assert np.bincount(c[:, 0], minlength=4).tolist() == [100] * 4 and np.bincount(c[:, 1], minlength=4).tolist() == [100] * 4
    assert c.tolist() == [list(G.exemplar_classes(n, i)) for i in range(400)]
    te = [x for x in fam.generate(n, 0) if x["split"] == "test"]
    for pos in (0, 1):
        tab = np.zeros((4, 4))
        for i, x in enumerate(te):
            tab[G.exemplar_classes(n, i)[pos], x["answer"]] += 1
        exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
        assert ((tab - exp) ** 2 / exp).sum() < 27.9  # chi-square, 9 df, p = 0.001


def test_twoshot_exemplars_use_train_templates_only(tasks, monkeypatch):
    for n in tasks:
        task = fam.get_task(n)
        tr_t, te_t = G.train_test_templates(n)
        gen = fam.generate(n, 0)
        orig, used = task.spec.render, []
        monkeypatch.setattr(task.spec, "render", lambda tpl, s: (used.append(tpl), orig(tpl, s))[1])
        for i, x in enumerate(gen):
            used.clear()
            G.exemplar_prefix(n, i, x["template"])
            assert len(used) == 2 and set(used) <= set(tr_t) and not (set(used) & set(te_t))
            assert x["template"] not in used  # excluding the item's own template
        monkeypatch.undo()


def test_twoshot_prefix_format_and_twins_share_it(tasks, ad):
    for n in tasks:
        task = fam.get_task(n)
        b, plain = G.build_task(ad, n, "twoshot"), G.build_task(ad, n, "plain")
        assert len(b["kept"]) > 0
        gen = {x["id"]: i for i, x in enumerate(fam.generate(n, 0))}
        by_id = {x["id"]: x for x in plain["kept"]}
        for x in b["kept"][:60]:
            pre = x["prefix"]
            assert pre.endswith("\n\n") and x["prompt"].startswith(pre)
            assert x["ans_twin"]["prompt"].startswith(pre) and x["surf_twin"]["prompt"].startswith(pre)
            if x["id"] in by_id:
                assert x["prompt"] == pre + by_id[x["id"]]["prompt"]
            e1, e2, rest = pre.split("\n\n")
            c1, c2 = G.exemplar_classes(n, gen[x["id"]])
            assert rest == "" and e1.endswith(task.options[c1]) and e2.endswith(task.options[c2])
            assert pre == G.exemplar_prefix(n, gen[x["id"]], x["template"])


def test_twoshot_rival_recomputed_and_ids_differ(tasks, ad):
    n = tasks[0]
    b2, b1 = G.build_task(ad, n, "twoshot"), G.build_task(ad, n, "plain")
    assert b2["rivals"] == fam.rivals(b2["kept"], n)
    assert b2["ids_te"] != b1["ids_te"] and all(len(a) > len(c) for a, c in zip(b2["ids_te"], b1["ids_te"]))
    # twins stay position-matched: item, answer twin, surface twin have equal length and last token
    assert all(len(a) == len(c) == len(d) for a, c, d in zip(b2["ids_te"], b2["ids_an"], b2["ids_sf"]))


def test_build_task_with_fresh(tasks, ad):
    b = G.build_task(ad, tasks[0], "twoshot", with_fresh=True, n_fresh=40)
    assert len(b["ids_fr"]) == len(b["y_fr"]) == len(b["fr"]) and 0 < len(b["fr"]) <= 40
    assert 0.0 <= b["bow_fr"] <= 1.0 and all(x["prefix"] for x in b["fr"])


# ---------------------------------------------------------------- measure on the toy

def _toy_b(seed_te=1, n_tr=300, n_te=100, n_fr=60):
    ids, y, an, y_an, sf = make_toy_items(n_tr + n_te, seed=seed_te)
    fr = make_toy_items(n_fr, seed=7)
    b = {"opt_ids": list(range(4)), "ids_tr": ids[:n_tr], "y_tr": y[:n_tr], "g_tr": list(np.arange(n_tr) % 10),
         "ids_te": ids[n_tr:], "y_te": y[n_tr:], "ids_an": an[n_tr:], "y_an": y_an[n_tr:], "ids_sf": sf[n_tr:],
         "rivals": {"bow": 0.25}, "ids_fr": fr[0], "y_fr": fr[1], "bow_fr": 0.25}
    return b


def _same(a, c, path=""):
    if isinstance(a, dict):
        assert a.keys() == c.keys(), path
        for k in a:
            _same(a[k], c[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(c), path
        for i, (x, z) in enumerate(zip(a, c)):
            _same(x, z, f"{path}/{i}")
    elif isinstance(a, float):
        assert a == pytest.approx(c, abs=1e-9, nan_ok=True), path
    else:
        assert a == c, path


@pytest.fixture(scope="module")
def toy_run():
    t = ToyAdapter(B=8, k=4)
    b = _toy_b()
    kw = dict(n_perm=6, n_flip=500, n_draw=200, max_pairs=200)
    out, arr = G.measure_arrays(t, b, **kw)
    ref = analyze_task(t, b["ids_tr"], b["y_tr"], b["g_tr"], b["ids_te"], b["y_te"], b["ids_an"], b["y_an"],
                       b["ids_sf"], b["opt_ids"], bow_acc=0.25, **{k: v for k, v in kw.items()})
    return t, b, out, arr, ref


def test_measure_json_equals_analyze_task(toy_run):
    _, _, out, _, ref = toy_run
    drop = lambda d: {k: v for k, v in d.items() if k not in ("timing",)}  # noqa: E731
    o, r = drop(out), drop(ref)
    o.pop("n_fresh")  # the only extra field
    for d in (o["probe"], r["probe"]):
        d.pop("fit_s")
    # the one deliberate difference: twin validity / onset use valid_sym
    assert o["valid"]["twin"] == ref["twin"]["valid_sym"]
    if r["valid"]["twin"] == r["twin"]["valid_sym"]:
        _same(o, r)
    else:
        for k in ("valid", "onset"):
            o.pop(k), r.pop(k)
        _same(o, r)
    assert out["use"]["valid"] and out["n_pairs_used"] > 40


def test_measure_arrays_shapes_and_consistency(toy_run):
    t, b, out, arr, _ = toy_run
    B, n, m, P = 8, 100, 60, out["n_pairs_used"]
    K = B - 1
    assert arr["pred_te"].shape == (B + 1, n) and arr["pred_perm"].shape == (6, B + 1, n)
    assert arr["pred_fr"].shape == (B + 1, m) and arr["pred_fr_perm"].shape == (6, B + 1, m)
    assert arr["y_te"].shape == (n,) and arr["y_fr"].shape == (m,) and arr["d_ans"].shape == (n, B + 1)
    assert arr["d_item"].shape == (P,) and arr["d_probe"].shape == (P, K) and arr["d_rand"].shape == (P, 10, K)
    assert arr["d_whole"].shape == (P, K) and arr["pair_idx"].shape == (P,)
    # arrays reproduce the JSON
    acc = (arr["pred_te"] == arr["y_te"]).mean(1)
    assert np.allclose(acc, out["probe"]["acc"])
    assert np.allclose((arr["pred_perm"] == arr["y_te"]).mean(2).mean(0), out["probe"]["perm_mean"])
    assert np.allclose(arr["d_ans"].mean(0), out["twin"]["d_ans"], atol=1e-6)
    u = M.use_scores(arr["d_item"], arr["d_twin"], arr["d_probe"], arr["d_rand"], arr["d_psurf"], n_flip=10)
    assert np.allclose(u["use"], out["use"]["use"], atol=1e-5)
    # the fresh predictions come from the same fits: planted key is recovered from block k on
    assert (arr["pred_fr"][5] == arr["y_fr"]).mean() > 0.95


def test_half_curves_recompose_full_curve(toy_run, tmp_path):
    _, _, out, arr, _ = toy_run
    G.save_npz(tmp_path / "x.npz", arr)
    npz = np.load(tmp_path / "x.npz")
    hc = G.probe_half_curves(npz, seed=3)
    h1, h2 = hc["halves"]
    n = h1["n"] + h2["n"]
    assert n == 100 and not (set(h1["idx"]) & set(h2["idx"]))
    assert np.allclose((h1["n"] * h1["acc"] + h2["n"] * h2["acc"]) / n, out["probe"]["acc"])
    assert np.allclose(h1["selectivity"], (h1["acc"] - h1["control"]) / 0.75) and h1["onset"]["layers"] > 3
    assert hc["halves"][0]["idx"].tolist() == G.probe_half_curves(npz, seed=3)["halves"][0]["idx"].tolist()
    uc = G.use_half_curves(npz, seed=3)
    u1, u2 = uc["halves"]
    P = out["n_pairs_used"]
    assert u1["n_pairs"] + u2["n_pairs"] == P and len(u1["use"]) == 8
    assert 3 < u1["onset"]["layers"] <= 4.05 and 3 < u2["onset"]["layers"] <= 4.05
    fc = G.fresh_probe_curve(npz)
    assert fc["n"] == 60 and np.allclose(fc["acc"], (arr["pred_fr"] == arr["y_fr"]).mean(1))
    assert fc["onset"]["layers"] > 3


def test_run_measure_end_to_end_writes_json_and_npz(tasks, ad, tmp_path):
    r = G.run_measure(ad, tasks[0], "plain", tmp_path / "t.json", tmp_path / "t.npz", n_perm=2, n_fresh=20,
                      log=lambda *a: None)
    assert (tmp_path / "t.json").exists() and (tmp_path / "t.npz").exists() and not list(tmp_path.glob("*.tmp"))
    npz = np.load(tmp_path / "t.npz")
    assert npz["pred_te"].shape[1] == r["n_test"] and npz["pred_fr"].shape[1] == r["n_fresh"]
    assert r["prompt"] == "plain" and r["task"] == tasks[0] and "answer_logfreq" in r and "rivals" in r
    assert len(r["probe"]["acc"]) == ad.n_blocks + 1


# ---------------------------------------------------------------- robustness arm

def test_meandiff_subspace_orthonormal_rank3():
    rng = np.random.default_rng(0)
    y = np.tile(np.arange(4), 25)
    X = rng.standard_normal((100, 16)) + 3 * np.eye(16)[y] @ rng.standard_normal((16, 16))
    Q = G.meandiff_subspace(X, y)
    assert Q.shape == (16, 3) and np.allclose(Q.T @ Q, np.eye(3), atol=1e-5)
    means = np.stack([X[y == c].mean(0) for c in range(4)]) - X.mean(0)
    assert np.linalg.matrix_rank(means) == 3 and np.allclose(Q @ (Q.T @ means.T), means.T, atol=1e-4)  # spans the class means


def test_robust_toy_onsets_at_planted_block_and_c1_matches_main(toy_run):
    t, b, out, _, _ = toy_run
    r = G.robust_built(t, b, Cs=(0.1, 1, 10), n_flip=200)
    assert set(r["variants"]) == {"C=0.1", "C=1", "C=10", "meandiff"}
    for v, x in r["variants"].items():
        assert x["valid"] and 3 < x["onset"]["layers"] <= 4.05, v
    assert r["robust"] and r["onset_shift_vs_C1"]["C=1"] == 0
    assert np.allclose(r["variants"]["C=1"]["use"], out["use"]["use"], atol=1e-5)  # same fits, pairs, controls


# ---------------------------------------------------------------- fused QKV LoRA and heal

def _capture_k_q(model, ids):
    """(q, k) fed to the first layer's rotary embedding in GPT-NeoX attention."""
    import transformers.models.gpt_neox.modeling_gpt_neox as mod
    got = []
    orig = mod.apply_rotary_pos_emb
    mod.apply_rotary_pos_emb = lambda q, k, c, s, *a, **kw: (got.append((q.detach().clone(), k.detach().clone())), orig(q, k, c, s, *a, **kw))[1]
    try:
        model(torch.tensor([ids]))
    finally:
        mod.apply_rotary_pos_emb = orig
    return got[0]


def test_fused_qkv_mask_selects_q_and_v_rows():
    H, hs = 4, 8
    m = adapters.qv_row_mask(H, hs).view(H, 3, hs)
    assert m.sum() == 2 * H * hs and (m[:, 0] == 1).all() and (m[:, 2] == 1).all() and (m[:, 1] == 0).all()
    model = _tiny("neox")
    lin = model.gpt_neox.layers[0].attention.query_key_value
    lora = adapters.FusedQKVLoRA(lin, 2, H)
    torch.nn.init.normal_(lora.B, std=0.5)
    x = torch.randn(5, 32)
    d = (lora(x) - lin(x)).view(5, H, 3 * hs).chunk(3, -1)  # the model's own q | k | v split
    assert d[1].abs().max() == 0 and d[0].abs().max() > 0 and d[2].abs().max() > 0
    assert "mask" not in lora.state_dict()


def test_fused_lora_changes_q_v_not_k_in_the_model_and_survives_steps():
    model = _tiny("neox").eval()
    ids = [int(i) for i in np.random.default_rng(0).integers(0, V, 9)]
    with torch.no_grad():
        q0, k0 = _capture_k_q(model, ids)
        lg0 = model(torch.tensor([ids])).logits
    h = G.attach_lora_any(model, seed=1)
    assert len(h.modules) == 3 and sum(p.numel() for p in h.modules.parameters() if p.requires_grad) == 3 * (10 * 32 + 96 * 10)
    for mod in h.modules:
        torch.nn.init.normal_(mod.B, std=0.3)
    opt = torch.optim.AdamW([p for p in h.modules.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        loss = model(torch.tensor([ids])).logits[0, -1].pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        q1, k1 = _capture_k_q(model, ids)
        lg1 = model(torch.tensor([ids])).logits
    assert torch.equal(k0, k1) and not torch.allclose(q0, q1) and not torch.allclose(lg0, lg1)
    h.remove()
    with torch.no_grad():
        assert torch.allclose(model(torch.tensor([ids])).logits, lg0, atol=1e-6)


@pytest.mark.parametrize("arch", ["neox", "qwen2"])
def test_adapter_round_trip_and_merge_any_arch(arch, tmp_path):
    model, tok = _tiny(arch), _Tok()
    ids = [[int(i) for i in np.random.default_rng(1).integers(0, V, 7)] for _ in range(3)]
    opt = [5, 17, 33, 71]
    h = G.attach_lora_any(model, seed=1)
    g = torch.Generator().manual_seed(3)
    for n, p in h.modules.named_parameters():
        if "base_layer." not in n:
            p.data = torch.randn(p.shape, generator=g) * 0.2
    with torch.no_grad():
        before = G.option_logits_any(model, ids, opt, "cpu")
    p0_heal.save_adapter(h, tmp_path / "a.safetensors")
    fresh = _tiny(arch)
    h2 = G.attach_lora_any(fresh, seed=9)
    p0_heal.load_adapter(h2, tmp_path / "a.safetensors")
    assert adapters.merge_lora(h2) == (3 if arch == "neox" else 6)
    with torch.no_grad():
        assert torch.allclose(before, G.option_logits_any(fresh, ids, opt, "cpu"), atol=1e-4)
    if arch == "neox":  # merged k rows unchanged
        assert torch.equal(fresh.gpt_neox.layers[0].attention.query_key_value.weight.view(4, 3, 8, 32)[:, 1],
                           _tiny(arch).gpt_neox.layers[0].attention.query_key_value.weight.view(4, 3, 8, 32)[:, 1])


@pytest.mark.parametrize("arch", ["neox", "qwen2"])
def test_heal_task_any_runs_cleanly_and_load_trained(arch, tasks, tmp_path, monkeypatch):
    model, tok = _tiny(arch), _Tok()
    if arch == "neox":
        ref = model.gpt_neox.layers[0].attention.query_key_value
    else:
        ref = model.model.layers[0].self_attn.q_proj
    r = G.heal_task_any(model, tok, tasks[0], 1e-3, 0, tmp_path, "cpu", steps=3, limit_train=16, log=lambda *a: None)
    cur = model.gpt_neox.layers[0].attention.query_key_value if arch == "neox" else model.model.layers[0].self_attn.q_proj
    assert cur is ref
    d = tmp_path / f"{tasks[0]}_lr0.001_s0"
    assert (d / "adapter.safetensors").exists() and (d / "heal.json").exists()
    assert {"wall_seconds", "steps", "lr", "seed", "dev_acc", "test_acc", "base_test_acc", "device", "arch"} <= set(r)
    assert r["n_train"] == 16 and r["n_dev"] == 300
    monkeypatch.setattr(G, "Adapter", lambda mid, device="cpu", dtype=None: Adapter(model=_tiny(arch), tok=tok, device=device))
    ad2 = G.load_trained("x", d / "adapter.safetensors", "cpu")
    assert ad2.merged_modules == (3 if arch == "neox" else 6) and ad2.model_id == "x"
    ids = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    _, L = ad2.collect_ids(ids, [5, 17, 33, 71])
    assert L.shape == (2, 4)


def test_heal_constants_match_p0_heal():
    assert (G.N_LORA, G.N_TRAIN, G.RANK, G.SCALE, G.BATCH, G.STEPS) == (
        p0_heal.N_LORA, p0_heal.N_TRAIN, p0_heal.RANK, p0_heal.SCALE, p0_heal.BATCH, p0_heal.STEPS)
    assert G.pick_lr([{"lr": 1e-3, "dev_acc": 0.5}, {"lr": 3e-4, "dev_acc": 0.5}]) == 3e-4


# ---------------------------------------------------------------- seeded draws

def test_seeded_draws_deterministic_and_nested():
    tasks50 = [f"t{i:02d}" for i in range(50)]
    a, c = G.lr_tasks(tasks50), G.seed_tasks(tasks50)
    assert a == G.lr_tasks(list(reversed(tasks50))) and len(a) == 6 == len(set(a))
    assert c == G.seed_tasks(list(reversed(tasks50))) and len(c) == 12 == len(set(c)) and c[:6] == a
    assert G.lr_tasks(tasks50, seed=1) != a
    models = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen3-0.6B-Base",
              "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
              "allenai/OLMo-2-0425-1B"]
    fams = [f"f{i:02d}" for i in range(25)]
    d = G.robust_draw(models, fams)
    assert d == G.robust_draw(list(reversed(models)), list(reversed(fams)))
    assert len(d["models"]) == 3 and "Qwen2.5" in d["models"][0] and "SmolLM2" in d["models"][1] and "pythia" in d["models"][2]
    assert len(d["families"]) == 6 and set(d["families"]) <= set(fams)
    assert G.robust_draw(["EleutherAI/pythia-410m"], fams)["models"] == ["EleutherAI/pythia-410m"]


# ---------------------------------------------------------------- runner script (tiny model, fake tokenizer)

def _run_cli(R, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["atlas_gates_run.py", *argv])
    with pytest.raises(SystemExit) as e:
        R.main()
    return e.value.code


def test_runner_models_and_default_tasks():
    import atlas_gates_run as R
    assert len(R.ALL_MODELS) == 15 and len(set(R.ALL_MODELS)) == 15
    assert "TinyLlama/TinyLlama_v1.1" in R.ALL_MODELS and "allenai/OLMo-2-0425-1B" in R.ALL_MODELS
    sizes = dict(R.MODELS)
    assert [sizes[m] for m in R.ALL_MODELS] == sorted(sizes.values())
    pick = Path(__file__).resolve().parent.parent / PICK_REL
    if pick.exists():
        import json
        assert len(json.loads(pick.read_text())["main"]) == 25


PICK_REL = "var/cscreen/pick_v13.json"


def test_runner_smoke_flow_adapt_lr_heal_trained(tasks, tmp_path, monkeypatch):
    import atlas_gates_run as R
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(R, "_load_model", lambda c, mid: (_tiny("neox"), _Tok()))
    monkeypatch.setattr(G, "Adapter", lambda mid, device="cpu", dtype=None: Adapter(model=_tiny("neox"), tok=_Tok(), device=device))
    def tiny_ad(c, mid):
        a = Adapter(model=_tiny("neox"), tok=_Tok(), device="cpu")
        a.model_id = mid
        return a

    monkeypatch.setattr(R, "_load_adapter", tiny_ad)
    common = ["--smoke", "--root", str(tmp_path / "res"), "--models", "x/pythia-tiny", "--tasks", ",".join(tasks[:2]),
              "--device", "cpu", "--n-perm", "2"]
    assert _run_cli(R, monkeypatch, "adapt", *common) == 0
    root = tmp_path / "res" / "smoke"
    t0 = G.lr_tasks(tasks[:2])[0]
    assert (root / "g2" / "adapt" / "pythia-tiny" / f"{t0}_lr0.0003_s0" / "heal.json").exists()
    assert _run_cli(R, monkeypatch, "lr", *common) == 0
    import json
    assert json.loads((root / "g2" / "lr_choice.json").read_text())["pythia-tiny"] in (1e-3, 3e-4)
    n_heal_lines = len((root / "g2" / "heal_log.tsv").read_text().splitlines())
    assert n_heal_lines == 1 + 3  # adapt + the 3 other lr-stage heals (the 3e-4 one of the first task was reused)
    assert (tmp_path / "var" / ".upload_now").exists()
    assert _run_cli(R, monkeypatch, "heal", "--seed", "1", *common) == 0
    lr = json.loads((root / "g2" / "lr_choice.json").read_text())["pythia-tiny"]
    assert (root / "g2" / "heals" / "pythia-tiny" / f"{tasks[0]}_lr{lr:g}_s1" / "heal.json").exists()
    assert _run_cli(R, monkeypatch, "heal", "--seed", "1", *common) == 0  # resumable: nothing new
    assert len((root / "g2" / "heal_log.tsv").read_text().splitlines()) == n_heal_lines + 2
    assert _run_cli(R, monkeypatch, "trained", "--seed", "1", *common) == 0
    r = json.loads((root / "g2" / "trained" / "s1" / "pythia-tiny" / f"{tasks[0]}.json").read_text())
    assert r["trained"] and "learnable" in r and r["heal"]["steps"] == 10 and r["lr"] == lr
    assert (root / "g2" / "trained" / "s1" / "pythia-tiny" / f"{tasks[0]}.npz").exists()
    assert _run_cli(R, monkeypatch, "measure", "--prompt", "twoshot", *common) == 0
    assert (root / "g1" / "twoshot" / "pythia-tiny" / f"{tasks[1]}.npz").exists()
    assert _run_cli(R, monkeypatch, "robust", *common) == 0
    assert _run_cli(R, monkeypatch, "heal", "--seed", "0", *common) == 0
    assert _run_cli(R, monkeypatch, "robust_trained", *common) == 0
    rt = [json.loads(p.read_text()) for p in (root / "g2" / "robust" / "pythia-tiny").glob("*.json")]
    assert len(rt) == 2 and all(x["trained"] and x["lr"] == lr and "onsets" in x for x in rt)
    assert "ERROR" not in (root / "status.txt").read_text()
