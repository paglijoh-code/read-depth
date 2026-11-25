"""M8 speed harness: guards, rounds, summary, H8, and one real measurement on a tiny exported model (CPU)."""
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

if sys.platform == "darwin":
    import mlx.core as mx
else:
    pytest.importorskip("mlx")
    import mlx.core as mx

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from surgery import export as E
from surgery import speed as S

mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parents[1]
PS_HEAD = "  PID  %CPU ARGS\n"


def fake_run(ollama=None, ps="", batt="Now drawing from 'AC Power'\n"):
    """ollama: CompletedProcess-like, or an exception instance to raise."""
    def run(cmd, **kw):
        if cmd[0] == "ollama":
            if isinstance(ollama, Exception):
                raise ollama
            return ollama
        if cmd[0] == "ps":
            return SimpleNamespace(returncode=0, stdout=PS_HEAD + ps, stderr="")
        if cmd[0] == "pmset":
            return SimpleNamespace(returncode=0, stdout=batt, stderr="")
        raise AssertionError(cmd)
    return run


def done(stdout="NAME ID SIZE PROCESSOR UNTIL\n", rc=0, stderr=""):
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------- guards

def test_quiet_when_ollama_empty_and_nothing_running():
    q = S.quiet_state(fake_run(done(), "  101   2.0 /usr/bin/python3 foo.py\n"))
    assert q["clear"] and q["reasons"] == []


def test_loaded_ollama_model_is_not_clear():
    q = S.quiet_state(fake_run(done("NAME ID SIZE\nqwen3.8:27b abc 30 GB 100% GPU 4 minutes\n")))
    assert not q["clear"] and "qwen3.8:27b" in q["reasons"][0]


@pytest.mark.parametrize("job", [
    "/usr/bin/python scripts/frontier_sweep.py --stage sweep", "python scripts/adapter_sweep.py x",
    "python scripts/mlx_parity.py", "python scripts/mlx_export.py"])
def test_known_gpu_job_is_not_clear(job):
    q = S.quiet_state(fake_run(done(), f"  4242  90.0 {job}\n"))
    assert not q["clear"] and "known GPU job" in q["reasons"][0]


def test_own_and_parent_pid_are_excluded():
    line = f"  {os.getpid()}  90.0 python scripts/frontier_sweep.py\n  {os.getppid()}  50.0 scripts/mlx_export.py\n"
    assert S.quiet_state(fake_run(done(), line))["clear"]


def test_busy_processes_are_logged_top_five_over_20_percent():
    ps = "".join(f"  {200 + i}  {30 + i}.0 /bin/work{i}\n" for i in range(7)) + "  300  5.0 /bin/idle\n"
    q = S.quiet_state(fake_run(done(), ps))
    assert q["clear"] and len(q["others"]) == 5
    assert "work6" in q["others"][0] and not any("idle" in o for o in q["others"])


def test_ollama_missing_or_down_counts_as_nothing_loaded():
    q = S.quiet_state(fake_run(FileNotFoundError("ollama")))
    assert q["clear"] and "not installed" in q["others"][0]
    down = done("", rc=1, stderr="Error: could not connect to ollama app, is it running?")
    q = S.quiet_state(fake_run(down))
    assert q["clear"] and "not reachable" in q["others"][0]


def test_other_ollama_failure_is_not_clear():
    q = S.quiet_state(fake_run(done("", rc=2, stderr="panic: something else")))
    assert not q["clear"] and "exited 2" in q["reasons"][0]


def test_ps_failure_is_not_clear():
    def run(cmd, **kw):
        if cmd[0] == "ps":
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")
        return done()
    q = S.quiet_state(run)
    assert not q["clear"] and "ps failed" in q["reasons"][0]


def test_on_mains():
    assert S.on_mains(fake_run(batt="Now drawing from 'AC Power'\n -InternalBattery-0 100%\n"))
    assert not S.on_mains(fake_run(batt="Now drawing from 'Battery Power'\n"))
    assert not S.on_mains(fake_run(batt=""))


def test_harness_refuses_when_a_fake_loaded_model_is_reported(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    exe = fake / "ollama"
    exe.write_text("#!/bin/sh\necho 'NAME ID SIZE PROCESSOR UNTIL'\n"
                   "echo 'fake:1b abc 1 GB 100% GPU 4 minutes from now'\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    results = tmp_path / "out" / "speed.jsonl"
    env = {**os.environ, "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}"}
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "mlx_speed.py"), "run", "--rounds", "1",
                        "--results", str(results), "--mlx", str(tmp_path / "none")],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    assert p.returncode == 2, p.stderr
    assert "refusing" in p.stderr and "ollama has a model loaded" in p.stderr
    assert not results.exists() and not results.parent.exists()  # no child started, nothing written


# ---------------------------------------------------------------- rounds, summary, H8

def test_pick_prompts_is_fixed_and_deterministic():
    recs = [{"id": i} for i in range(1500)]
    a, b = S.pick_prompts(recs), S.pick_prompts(recs)
    assert a == b and len(a) == 100 and len({r["id"] for r in a}) == 100
    assert a != S.pick_prompts(recs, seed=9)
    assert len(S.pick_prompts(recs[:10])) == 10


def test_round_orders_rotate():
    o = S.round_orders(["a", "b", "c"], 4)
    assert o == [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"], ["a", "b", "c"]]
    assert all(sorted(x) == ["a", "b", "c"] for x in o)


def row(model, variant, rnd, prefill, peak=1000.0, valid=True, **kw):
    return {"model": model, "variant": variant, "dtype": "bf16", "round": rnd, "valid": valid,
            "prefill_tps": prefill, "peak_mem_bytes": peak, "decode_tps": 50.0, **kw}


def test_summarize_drops_round_one_and_invalid_rounds():
    rows = [row("unmodified", "rule", 1, 1.0), row("unmodified", "rule", 2, 100.0), row("unmodified", "rule", 3, 120.0),
            row("unmodified", "rule", 4, 5000.0, round_valid=False), row("unmodified", "rule", 5, 110.0),
            row("rule-A", "rule", 4, 7.0), row("rule-A", "rule", 6, 200.0, error="x")]
    s = S.summarize(rows)
    e = s[("unmodified", "rule", "bf16")]
    assert e["n"] == 3 and e["prefill_tps"] == {"median": 110.0, "min": 100.0, "max": 120.0}
    assert ("rule-A", "rule", "bf16") not in s  # rounds 4 (invalid round) and 6 (error) are gone


def test_a_failed_model_does_not_invalidate_the_others():
    rows = [row("unmodified", "rule", 2, 100.0), row("rule-A", "rule", 2, 900.0, valid=False),
            row("rule-k12", "rule", 3, 150.0), row("rule-A", "rule", 3, 900.0, error="crash")]
    s = S.summarize(rows)
    assert ("unmodified", "rule", "bf16") in s and ("rule-k12", "rule", "bf16") in s
    assert ("rule-A", "rule", "bf16") not in s  # so H8 reports unmeasured, not a verdict
    assert S.h8_verdict(s)["verdict"] == "unmeasured"


def summary(**ratios):
    base = {"prefill_tps": {"median": 100.0}, "peak_mem_bytes": {"median": 1000.0}}
    out = {("unmodified", v, "bf16"): base for v in ("rule", "sensible")}
    for name, v in (("rule-A", "rule"), ("sensible-A", "sensible")):
        p, m = ratios.get(name, (2.0, 0.6))
        out[(name, v, "bf16")] = {"prefill_tps": {"median": 100.0 * p}, "peak_mem_bytes": {"median": 1000.0 * m}}
    return out


def test_h8_verdicts():
    assert S.h8_verdict(summary())["verdict"] == "supported"
    assert S.h8_verdict(summary(**{"rule-A": (1.6, 0.75)}))["verdict"] == "supported"  # bars are inclusive
    assert S.h8_verdict(summary(**{"rule-A": (1.29, 0.5)}))["verdict"] == "falsified"
    assert S.h8_verdict(summary(**{"sensible-A": (1.0, 0.5)}))["verdict"] == "falsified"
    assert S.h8_verdict(summary(**{"rule-A": (1.5, 0.6)}))["verdict"] == "not supported"
    assert S.h8_verdict(summary(**{"rule-A": (2.0, 0.8)}))["verdict"] == "not supported"
    assert S.h8_verdict(summary(**{"rule-A": (1.3, 0.6)}))["verdict"] == "not supported"  # 1.3 is not below 1.3


def test_h8_unmeasured_never_gives_a_verdict():
    s = summary()
    del s[("sensible-A", "sensible", "bf16")]
    assert S.h8_verdict(s)["verdict"] == "unmeasured"
    s = summary()
    del s[("unmodified", "rule", "bf16")]
    assert S.h8_verdict(s)["verdict"] == "unmeasured"
    s = summary(**{"rule-A": (1.0, 1.0)})  # would be falsified, but a missing metric makes it unmeasured
    del s[("rule-A", "rule", "bf16")]["peak_mem_bytes"]
    assert S.h8_verdict(s)["verdict"] == "unmeasured"
    assert S.h8_verdict({})["verdict"] == "unmeasured"
    assert S.h8_verdict(S.summarize([row("unmodified", "rule", 1, 1.0)]))["verdict"] == "unmeasured"


# ---------------------------------------------------------------- one real measurement

def test_measure_one_on_a_tiny_exported_model(tmp_path):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=8, num_key_value_heads=4, head_dim=16, tie_word_embeddings=True)
    model = Qwen3ForCausalLM(cfg).eval()
    vocab = {f"w{i}": i for i in range(31)} | {"[UNK]": 31}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]")
    E.export_mlx(model, tmp_path / "m", "bf16", tokenizer=tok)
    prompts = [[1, 2, 3, 4, 5, 6], [7, 8, 9], [10, 11, 12, 13]]
    r = S.measure_one(tmp_path / "m", prompts, decode_n=2, decode_tokens=6)
    assert r["n_prompts"] == 3 and r["prefill_tokens"] == 13 and r["decode_tokens"] == 2 * 5
    for k in ("prefill_tps", "decode_tps", "tta_median_s", "peak_mem_bytes", "rss_bytes"):
        assert r[k] > 0
    assert r["weight_bytes"] == (tmp_path / "m" / "model.safetensors").stat().st_size
