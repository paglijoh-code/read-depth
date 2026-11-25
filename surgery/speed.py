"""M8 speed harness pieces: quiet-machine guard, power check, one-model measurement, rounds, summary, H8.

Mlx is imported inside `measure_one` only, so the guard and the statistics import anywhere. Prompts come from the
selection split (never test); the test ledger is not touched.
"""
from __future__ import annotations

import os
import random
import re
import resource
import statistics
import subprocess
import time
from pathlib import Path

KNOWN_GPU_JOBS = [
    r"scripts/\w+_sweep\.py", r"scripts/mlx_parity\.py", r"scripts/mlx_export\.py",
]
_SERVER_DOWN = re.compile(r"could not connect|connection refused|is it running|server not|not reachable", re.I)
CPU_NOISE = 20.0
METRICS = ["prefill_tps", "decode_tps", "tta_median_s", "peak_mem_bytes", "rss_bytes", "weight_bytes"]


def pick_prompts(records, n: int = 100, seed: int = 8) -> list:
    """n selection records chosen by a fixed seed, in sample order (deterministic)."""
    idx = random.Random(seed).sample(range(len(records)), min(n, len(records)))
    return [records[i] for i in idx]


# ---------------------------------------------------------------- guards

def quiet_state(run=subprocess.run) -> dict:
    """{"clear", "reasons", "others"}. Not clear when `ollama ps` lists a loaded model, a known GPU job runs, or a
    check itself fails (fail loud). A missing ollama binary or an unreachable server counts as nothing loaded."""
    reasons, others = [], []
    try:
        r = run(["ollama", "ps"], capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        others.append("ollama not installed: nothing loaded")
    except Exception as e:  # noqa: BLE001
        reasons.append(f"ollama ps failed to run: {e!r}")
    else:
        text = (r.stdout or "") + (r.stderr or "")
        if r.returncode == 0:
            rows = [l for l in (r.stdout or "").splitlines()[1:] if l.strip()]
            if rows:
                reasons.append("ollama has a model loaded: " + "; ".join(" ".join(l.split()[:3]) for l in rows))
        elif _SERVER_DOWN.search(text):
            others.append("ollama server not reachable: nothing loaded")
        else:
            reasons.append(f"ollama ps exited {r.returncode}: {text.strip()[:200]}")
    skip = {os.getpid(), os.getppid()}
    try:
        r = run(["ps", "-axo", "pid,pcpu,args"], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"exit {r.returncode}: {(r.stderr or '').strip()[:200]}")
    except Exception as e:  # noqa: BLE001
        reasons.append(f"ps failed: {e!r}")
    else:
        busy = []
        for line in (r.stdout or "").splitlines()[1:]:
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            try:
                pid, cpu = int(parts[0]), float(parts[1])
            except ValueError:
                continue
            args = parts[2]
            if pid in skip or args.startswith("ps -axo"):
                continue
            hit = next((p for p in KNOWN_GPU_JOBS if re.search(p, args)), None)
            if hit:
                reasons.append(f"known GPU job running: pid {pid} {args[:100]}")
            elif cpu > CPU_NOISE:
                busy.append((cpu, pid, args))
        others += [f"{c:.0f}% pid {p} {a[:100]}" for c, p, a in sorted(busy, reverse=True)[:5]]
    return {"clear": not reasons, "reasons": reasons, "others": others}


def on_mains(run=subprocess.run) -> bool:
    """True when `pmset -g batt` says AC power. Battery, or anything unreadable, is False."""
    try:
        r = run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return False
    first = (r.stdout or "").splitlines()[:1]
    return r.returncode == 0 and bool(first) and "AC Power" in first[0]


# ---------------------------------------------------------------- measurement

def _weight_bytes(model_dir: Path) -> int:
    return sum(p.stat().st_size for p in Path(model_dir).glob("model*.safetensors"))


def measure_one(model_dir, prompts_ids, decode_n: int = 10, decode_tokens: int = 128) -> dict:
    """Run in a fresh process. Batch 1; every prompt gets a new cache (no prompt-cache reuse). One untimed forward
    on the first prompt first, so kernel compilation is not in the first timed prompt."""
    import json

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    model, _tok = load(str(model_dir), trust_remote_code="model_file" in config)

    def forward(ids):
        cache = make_prompt_cache(model)
        logits = model(mx.array(ids)[None], cache=cache)[:, -1, :]
        return logits, cache

    logits, cache = forward(prompts_ids[0])
    mx.eval(logits, [c.state for c in cache])
    del logits, cache

    total_tokens, total_s, answer_s = 0, 0.0, []
    for ids in prompts_ids:
        t0 = time.perf_counter()
        logits, cache = forward(ids)
        mx.eval(logits, [c.state for c in cache])
        t1 = time.perf_counter()
        mx.eval(mx.argmax(logits, axis=-1))
        t2 = time.perf_counter()
        total_tokens += len(ids)
        total_s += t1 - t0
        answer_s.append(t2 - t0)
        del logits, cache

    dec_tokens, dec_s = 0, 0.0
    for ids in prompts_ids[:decode_n]:
        stamps = []
        for _tok_id, _ in generate_step(mx.array(ids), model, max_tokens=decode_tokens,
                                        sampler=lambda x: mx.argmax(x, axis=-1)):
            stamps.append(time.perf_counter())  # no EOS stop: the token stream is never inspected
        if len(stamps) > 1:
            dec_tokens += len(stamps) - 1
            dec_s += stamps[-1] - stamps[0]
    return {
        "prefill_tps": total_tokens / total_s, "prefill_tokens": total_tokens, "n_prompts": len(prompts_ids),
        "tta_median_s": statistics.median(answer_s),
        "decode_tps": dec_tokens / dec_s if dec_s else None, "decode_tokens": dec_tokens,
        "peak_mem_bytes": int(mx.get_peak_memory()), "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "weight_bytes": _weight_bytes(model_dir),
    }


# ---------------------------------------------------------------- rounds and summary

def round_orders(names, rounds: int) -> list[list]:
    """Round r starts r places further along `names` (rotation), so no model always runs first or last."""
    names = list(names)
    n = len(names)
    return [names[r % n:] + names[:r % n] for r in range(rounds)] if n else [[] for _ in range(rounds)]


def _good(row) -> bool:
    return bool(row.get("valid", True)) and "error" not in row


def summarize(rows, warmup_rounds: int = 1) -> dict:
    """{(model, variant, dtype): {"n": rounds used, metric: {"median", "min", "max"}}}. Round 1 is warm-up. A
    machine failure (round_valid false: contention, battery) drops the whole round; a model failure (valid false:
    crash, parity not passed) drops only that model's row, so the other models still report."""
    bad = {r["round"] for r in rows if not r.get("round_valid", True)}
    groups: dict = {}
    for r in rows:
        if r["round"] <= warmup_rounds or r["round"] in bad or not _good(r):
            continue
        groups.setdefault((r["model"], r["variant"], r.get("dtype", "bf16")), []).append(r)
    out = {}
    for key, rs in groups.items():
        entry = {"n": len(rs)}
        for m in METRICS:
            vals = [r[m] for r in rs if r.get(m) is not None]
            if vals:
                entry[m] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
        out[key] = entry
    return out


H8_SUPPORT_PREFILL, H8_SUPPORT_PEAK, H8_FALSIFY_PREFILL = 1.6, 0.75, 1.3
H8_MODELS = [("rule-A", "rule"), ("sensible-A", "sensible")]


def h8_verdict(summary: dict, dtype: str = "bf16") -> dict:
    """bf16, each A model against unmodified on the same variant. supported: prefill >= 1.6x and peak <= 0.75x on
    both; falsified: either A under 1.3x prefill; otherwise not supported. Missing data is "unmeasured"."""
    ratios = {}
    for name, v in H8_MODELS:
        a, b = summary.get((name, v, dtype)), summary.get(("unmodified", v, dtype))
        try:
            ratios[name] = {"prefill": a["prefill_tps"]["median"] / b["prefill_tps"]["median"],
                            "peak": a["peak_mem_bytes"]["median"] / b["peak_mem_bytes"]["median"]}
        except (TypeError, KeyError, ZeroDivisionError):
            return {"verdict": "unmeasured", "ratios": ratios, "missing": f"{name}/{v}"}
    if any(r["prefill"] < H8_FALSIFY_PREFILL for r in ratios.values()):
        verdict = "falsified"
    elif all(r["prefill"] >= H8_SUPPORT_PREFILL and r["peak"] <= H8_SUPPORT_PEAK for r in ratios.values()):
        verdict = "supported"
    else:
        verdict = "not supported"
    return {"verdict": verdict, "ratios": ratios}
