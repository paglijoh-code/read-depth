"""M8 step 4: MLX speed runs (bf16), one fresh process per (model, variant) per round.

  uv run python scripts/mlx_speed.py run [--rounds 5] [--models ...] [--dtype bf16]
  uv run python scripts/mlx_speed.py report
  uv run python scripts/mlx_speed.py one --model-dir D --prompts FILE      (child; prints one JSON line)
`run` exits 2 and starts nothing unless `ollama ps` is empty, no known GPU job runs, and the Mac is on mains.
Each child is followed by a quiet check, and so is each round; a round failing any check gets round_valid:false.
A row gets valid:false when its child crashed or its model has not passed the full parity gate (mlx_parity.py).
Rows are appended to var/results/m8/speed.jsonl. Round 1 is warm-up and is dropped by the report.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgery import speed as S  # noqa: E402  (no torch: the `one` child must not carry torch in its RSS)

DATA = Path("var/data/m2")
OUT = Path("var/results/m8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def refuse(msg: str) -> None:
    print(f"mlx_speed: refusing to run: {msg}", file=sys.stderr)
    sys.exit(2)


def guard(quiet=None) -> dict:
    q = quiet or S.quiet_state()
    if not q["clear"]:
        refuse("machine not quiet: " + "; ".join(q["reasons"]))
    if not S.on_mains():
        refuse("not on mains power (or `pmset -g batt` unreadable)")
    return q


def prompt_files(variants, out: Path) -> dict:
    """Tokenize the 100 fixed selection prompts per variant once; the child reads the ids from a file."""
    from transformers import AutoTokenizer
    from surgery import export as E
    from surgery.prompt import build_prompt
    from surgery.task.generator import read_jsonl
    tok = AutoTokenizer.from_pretrained(E.MODEL)
    files = {}
    out.mkdir(parents=True, exist_ok=True)
    for v in variants:
        recs = S.pick_prompts(read_jsonl(DATA / f"{v}_s0_selection.jsonl"))
        p = out / f"prompts_{v}.json"
        p.write_text(json.dumps({"variant": v, "ids": [r["id"] for r in recs],
                                 "tokens": [tok(build_prompt(tok, r["prompt"]))["input_ids"] for r in recs]}))
        files[v] = p
    return files


def parity_ok(name, v):
    p = OUT / "parity.json"
    if not p.exists():
        return None
    row = json.loads(p.read_text()).get(name, {}).get(v)
    return None if row is None or not row.get("full") else bool(row["passed"])


def cmd_run(a) -> None:
    guard()
    from surgery import export as E
    names = a.models.split(",")
    for n in names:
        if n not in E.MODELS:
            sys.exit(f"unknown model {n!r}")
    entries = [(n, v) for n in names for v in E.variants_of(n)]
    dirs = {e: Path(a.mlx) / e[0] / a.dtype for e in entries}
    for d in dirs.values():
        if not (d / "model.safetensors").exists():
            sys.exit(f"{d} missing: run scripts/mlx_export.py first")
    results = Path(a.results)
    files = prompt_files(sorted({v for _, v in entries}), results.parent)
    for rnd, order in enumerate(S.round_orders(entries, a.rounds), start=1):
        rows, ok = [], True
        for name, v in order:
            cmd = [sys.executable, str(Path(__file__).resolve()), "one", "--model-dir", str(dirs[(name, v)]),
                   "--prompts", str(files[v]), "--decode-n", str(a.decode_n), "--decode-tokens", str(a.decode_tokens)]
            p = subprocess.run(cmd, capture_output=True, text=True)
            row = {"ts": now(), "round": rnd, "model": name, "variant": v, "dtype": a.dtype}
            if p.returncode == 0:
                row.update(json.loads(p.stdout.strip().splitlines()[-1]))
            else:
                row["error"] = (p.stderr or p.stdout)[-400:]
            q = S.quiet_state()
            row["others"] = q["others"]
            if not q["clear"]:
                ok, row["quiet_reasons"] = False, q["reasons"]
            parity = parity_ok(name, v)  # a row is invalid until its model has passed the full parity gate
            row["parity"] = {True: "passed", False: "failed", None: "not run"}[parity]
            row["valid"] = "error" not in row and parity is True
            rows.append(row)
            print(f"round {rnd} {name}/{v}: " + (f"prefill {row['prefill_tps']:.0f} tok/s, decode "
                  f"{row['decode_tps'] or 0:.0f} tok/s" if "error" not in row else "FAILED"), flush=True)
        qr = S.quiet_state()
        if not qr["clear"] or not S.on_mains():
            ok = False
        results.parent.mkdir(parents=True, exist_ok=True)
        with results.open("a") as fh:
            for r in rows:
                r["round_valid"] = ok
                r["round_others"] = qr["others"]
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"round {rnd} {'valid' if ok else 'INVALID'}", flush=True)


def cmd_one(a) -> None:
    prompts = json.loads(Path(a.prompts).read_text())["tokens"]
    print(json.dumps(S.measure_one(a.model_dir, prompts, a.decode_n, a.decode_tokens)))


def table(summary: dict) -> str:
    def f(entry, key, scale=1.0, fmt="{:.0f}"):
        m = entry.get(key)
        return "n/a" if not m else f"{fmt.format(m['median'] * scale)} ({fmt.format(m['min'] * scale)}-" \
                                   f"{fmt.format(m['max'] * scale)})"
    lines = ["| model | variant | dtype | rounds | prefill tok/s | decode tok/s | time to answer ms | peak MB (mx) "
             "| RSS MB | weights MB |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (m, v, d), e in sorted(summary.items()):
        lines.append(f"| {m} | {v} | {d} | {e['n']} | {f(e, 'prefill_tps')} | {f(e, 'decode_tps')} | "
                     f"{f(e, 'tta_median_s', 1e3, '{:.1f}')} | {f(e, 'peak_mem_bytes', 1e-6)} | "
                     f"{f(e, 'rss_bytes', 1e-6)} | {f(e, 'weight_bytes', 1e-6)} |")
    return "\n".join(lines)


def cmd_report(a) -> None:
    p = Path(a.results)
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    summary = S.summarize(rows)
    print(table(summary))
    invalid = sorted({f"{r['model']}/{r['variant']} (parity {r.get('parity')})" for r in rows if not r.get("valid", True)})
    if invalid:
        print("\nInvalid rows, left out of the table: " + ", ".join(invalid))
    h = S.h8_verdict(summary, a.dtype)
    print(f"\nH8 ({a.dtype}): **{h['verdict']}**")
    for name, r in h["ratios"].items():
        print(f"- {name}: prefill {r['prefill']:.2f}x, peak memory {r['peak']:.2f}x of unmodified")
    if h.get("missing"):
        print(f"- missing or invalid data for {h['missing']}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--models", default="unmodified,rule-A,rule-B,sensible-A,rule-lora-1m,sensible-lora-1m,rule-k12,sensible-k12")
    r.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    r.add_argument("--mlx", default="var/mlx")
    r.add_argument("--results", default=str(OUT / "speed.jsonl"))
    r.add_argument("--decode-n", type=int, default=10)
    r.add_argument("--decode-tokens", type=int, default=128)
    o = sub.add_parser("one")
    o.add_argument("--model-dir", required=True)
    o.add_argument("--prompts", required=True)
    o.add_argument("--decode-n", type=int, default=10)
    o.add_argument("--decode-tokens", type=int, default=128)
    rep = sub.add_parser("report")
    rep.add_argument("--results", default=str(OUT / "speed.jsonl"))
    rep.add_argument("--dtype", default="bf16")
    a = ap.parse_args(argv)
    {"run": cmd_run, "one": cmd_one, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
