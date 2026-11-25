#!/usr/bin/env python3
"""Build the 14 atlas tasks (or, with --only, the stage A families, manifest2.json, or the stage C families, manifest3.json) into var/data/atlas (docs/ATLAS-phase0.md, Brief A).

    HF_HUB_OFFLINE=1 uv run python scripts/atlas_build.py [--out DIR] [--only FAMILY ...]

Splits (spec.SPLITS, seed spec.SPLIT_SEED) are dealt by whole template and cell-balanced; the two tasks of a matched
pair get the same base items in the same split. Writes {task}_{split}.jsonl and manifest.json (counts, answer balance,
templates per split, chance per split, sha256 of every file). Deterministic: two runs give identical bytes.
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

from surgery.atlas import spec, tasks  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: Path, families: list[str] | None = None, sizes: dict | None = None, seed: int = spec.SPLIT_SEED) -> dict:
    sizes = sizes or spec.SPLITS
    manifest = {"split_seed": seed, "sizes": sizes, "template_cap": spec.TEMPLATE_CAP,
                "min_templates_per_split": spec.MIN_TEMPLATES_PER_SPLIT, "tasks": {}, "files": {}}
    for family in families or list(tasks.PHASE1_FAMILIES):
        splits, info = tasks.make_splits(family, seed, sizes)
        for task in tasks.FAMILY_TASKS[family]:
            k = len(spec.TASKS[task].options)
            entry = {"number": spec.TASKS[task].number, "family": family, "options": list(spec.TASKS[task].options),
                     "pool": info, "splits": {}}
            for split, items in splits.items():
                recs = [it["records"][task] for it in items]
                path = spec.data_path(task, split, out)
                spec.write_records(recs, path)
                manifest["files"][path.name] = sha256(path)
                ans = [r["answer"] for r in recs]
                cnt = Counter(ans)
                tpl = Counter(r["template_id"] for r in recs)
                rungs = Counter(r["rung"] for r in recs if r["rung"])
                entry["splits"][split] = {
                    "n": len(recs), "answers": [cnt.get(i, 0) for i in range(k)],
                    "max_dev_from_uniform": round(tasks.balance_dev(ans, k), 4),
                    "templates": len(tpl), "max_template_share": round(max(tpl.values()) / len(recs), 4),
                    "chance": round(spec.chance(ans), 4), "rungs": dict(sorted(rungs.items()))}
            manifest["tasks"][task] = entry
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=spec.DATA_DIR)
    ap.add_argument("--only", nargs="*", choices=sorted(tasks.FAMILY_TASKS), default=None)
    a = ap.parse_args(argv)
    fams = a.only or tasks.PHASE1_FAMILIES
    groups = {"manifest.json": tasks.PHASE1_FAMILIES, "manifest2.json": tasks.NEW_FAMILIES, "manifest3.json": tasks.C_FAMILIES}
    names = {n for n, g in groups.items() if any(f in g for f in fams)}
    if len(names) > 1:
        ap.error("build Phase 1, stage A and stage C families separately (they have separate manifests)")
    manifest = build(a.out, a.only)
    mpath = a.out / names.pop()   # manifest.json (the 14 old tasks) and manifest2.json are never rewritten by stage C
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for task, e in sorted(manifest["tasks"].items(), key=lambda kv: kv[1]["number"]):
        tr = {s: v["templates"] for s, v in e["splits"].items()}
        dev = max(v["max_dev_from_uniform"] for v in e["splits"].values())
        ch = {s: v["chance"] for s, v in e["splits"].items()}
        print(f"{e['number']:>2} {task:<16} n={sum(v['n'] for v in e['splits'].values())} templates={tr} "
              f"max_dev={dev} chance={ch} pool={e['pool']['pool_items']} short={sum(e['pool']['quota_shortfall'].values())}")
    print("wrote", mpath)


if __name__ == "__main__":
    main()
