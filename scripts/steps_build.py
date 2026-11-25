#!/usr/bin/env python3
"""Build the three steps-study tasks (docs/ATLAS-steps.md) into var/data/atlas and write manifest_steps.json.

    HF_HUB_OFFLINE=1 uv run python scripts/steps_build.py [--out DIR]

Splits are dealt by whole template (surgery/atlas/steps.py), seed spec.SPLIT_SEED. F1 records get Qwen-token distances
in meta (needs the Qwen3-1.7B tokenizer in the offline HF cache). Deterministic: two runs give identical bytes.
Writes {task}_{split}.jsonl and manifest_steps.json (counts per split per cell, answer balance per cell, templates, sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from surgery.atlas import spec, steps  # noqa: E402

QWEN = "Qwen/Qwen3-1.7B"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: Path, seed: int = spec.SPLIT_SEED) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN)
    manifest = {"split_seed": seed, "sizes": {t: steps.cell_sizes(t) for t in steps.STEPS_TASKS}, "tasks": {}, "files": {}}
    for task in steps.STEPS_TASKS:
        splits = steps.generate(task, seed)
        k = len(spec.TASKS[task].options)
        entry = {"number": spec.TASKS[task].number, "options": list(spec.TASKS[task].options), "splits": {}}
        for split, recs in splits.items():
            if task == "steps_copy":
                for r in recs:
                    steps.annotate_f1_tokens(tok, r)
            path = spec.data_path(task, split, out)
            spec.write_records(recs, path)
            manifest["files"][path.name] = sha256(path)
            cells: dict[str, dict] = {}
            for cell in sorted({r["rung"] for r in recs}):
                sub = [r for r in recs if r["rung"] == cell]
                cnt = Counter(r["answer"] for r in sub)
                cells[cell] = {"n": len(sub), "answers": [cnt.get(i, 0) for i in range(k)],
                               "chance": round(spec.chance([r["answer"] for r in sub]), 4),
                               "templates": len({r["template_id"] for r in sub})}
            tpl = Counter(r["template_id"] for r in recs)
            entry["splits"][split] = {"n": len(recs), "templates": len(tpl),
                                      "max_template_share": round(max(tpl.values()) / len(recs), 4), "cells": cells}
        manifest["tasks"][task] = entry
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=spec.DATA_DIR)
    a = ap.parse_args(argv)
    manifest = build(a.out)
    mpath = a.out / "manifest_steps.json"
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for task, e in manifest["tasks"].items():
        print(task, {s: (v["n"], v["templates"]) for s, v in e["splits"].items()})
    print("wrote", mpath)


if __name__ == "__main__":
    main()
