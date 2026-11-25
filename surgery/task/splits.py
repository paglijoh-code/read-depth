"""Deterministic splits that keep every active-rule template in one split.

CLI: python -m surgery.task.splits make --variant rule --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from surgery.task.generator import LABELS, Knobs, generate, write_jsonl
from surgery.task.hard import HardKnobs, generate_hard

SPLIT_SIZES = {
    0: {"discovery": 3000, "selection": 1500, "test": 1500},
    # Seed 1 is built exactly like seed 0 so its discovery split has the same template
    # mix; only its discovery split is used (seed-agreement gate), the rest is never read.
    1: {"discovery": 3000, "selection": 1500, "test": 1500},
}
MIN_TEMPLATES = 50
MAX_POOL_FACTOR = 16  # give up when the pool is this many times the total split size


TEMPLATE_CAP = 0.03  # at most this share of a split from one template


def _partition(pool: list[dict], seed: int, sizes: dict[str, int]) -> dict[str, list[dict]] | None:
    """Deal whole templates, in seeded shuffle order, to the split furthest below its size,
    so every split gets a similar mix of template sizes. A template contributes at most
    TEMPLATE_CAP of its split; its unused records are dropped, never moved to another split."""
    groups = defaultdict(list)
    for record in pool:
        groups[record["template_id"]].append(record)
    templates = sorted(groups)
    random.Random(seed).shuffle(templates)
    splits = {name: [] for name in sizes}
    for template in templates:
        open_ = [n for n in sizes if len(splits[n]) < sizes[n]]
        if not open_:
            break
        name = max(open_, key=lambda n: (1 - len(splits[n]) / sizes[n], sizes[n]))
        cap = max(1, round(TEMPLATE_CAP * sizes[name]))
        room = sizes[name] - len(splits[name])
        splits[name].extend(groups[template][:min(cap, room)])
    if any(len(splits[n]) < sizes[n] for n in sizes):
        return None
    return splits


def _template_sizes(pool: list[dict]) -> dict:
    counts = sorted(Counter(r["template_id"] for r in pool).values())
    return {"templates": len(counts), "min": counts[0],
            "p50": counts[(len(counts) - 1) // 2],
            "p90": counts[(len(counts) - 1) * 90 // 100],
            "p99": counts[(len(counts) - 1) * 99 // 100], "max": counts[-1]}


def _make_splits(variant: str, seed: int, sizes: dict[str, int],
                 pool_n: int | None = None, knobs: Knobs | HardKnobs | None = None) -> tuple[dict[str, list[dict]], dict]:
    if not sizes or any(not isinstance(n, int) or n <= 0 for n in sizes.values()):
        raise ValueError("split sizes must be positive integers")
    if pool_n is not None and pool_n <= 0:
        raise ValueError("pool_n must be positive")
    pool_n = pool_n if pool_n is not None else 2 * sum(sizes.values())
    while True:
        if isinstance(knobs, HardKnobs):
            pool = generate_hard(pool_n, seed, knobs, variant=variant)   # template_id: facts the flags and rules read
        else:
            pool = (generate(pool_n, seed, variant=variant) if knobs is None
                    else generate(pool_n, seed, knobs, variant=variant))
        splits = _partition(pool, seed, sizes)
        if splits is not None:
            break
        if pool_n >= MAX_POOL_FACTOR * sum(sizes.values()):
            raise ValueError(f"cannot fill splits {sizes} from a pool of {pool_n}; "
                             f"pool template sizes: {_template_sizes(pool)}")
        pool_n *= 2

    template_counts = {name: len({r["template_id"] for r in rows})
                       for name, rows in splits.items()}
    distribution = _template_sizes(pool)
    for name, count in template_counts.items():
        if count < MIN_TEMPLATES:
            raise ValueError(f"{name} has {count} templates; need >= {MIN_TEMPLATES}; "
                             f"pool template sizes: {distribution}")
    manifest = {
        "variant": variant, "seed": seed, "pool_size": pool_n,
        "sizes": {name: len(rows) for name, rows in splits.items()},
        "template_counts": template_counts,
        "label_counts": {name: {label: sum(r["label"] == label for r in rows)
                                 for label in LABELS} for name, rows in splits.items()},
        "template_size_distribution": distribution,
    }
    return splits, manifest


def make_splits(variant: str, seed: int, sizes: dict[str, int],
                pool_n: int | None = None, knobs: Knobs | None = None) -> dict[str, list[dict]]:
    """Generate a pool, shuffle sorted templates, and fill splits in sizes' order.

    Grow the pool geometrically if whole-template assignment leaves too few rows.
    Raise if a completed split contains fewer than 50 distinct templates.
    """
    return _make_splits(variant, seed, sizes, pool_n, knobs)[0]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make")
    make.add_argument("--variant", choices=("rule", "sensible"), required=True)
    make.add_argument("--seed", type=int, choices=(0, 1), required=True)
    make.add_argument("--out-dir", type=Path, default=Path("var/data/m2"))
    args = parser.parse_args(argv)
    splits, manifest = _make_splits(args.variant, args.seed, SPLIT_SIZES[args.seed])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.variant}_s{args.seed}"
    for name, records in splits.items():
        write_jsonl(records, args.out_dir / f"{prefix}_{name}.jsonl")
    path = args.out_dir / f"{prefix}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
