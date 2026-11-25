"""Rank discovery-set attention heads by the loss change from mean ablation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from surgery.attribution import head_attribution  # noqa: E402
from surgery.eval import load_model  # noqa: E402
from surgery.task.generator import read_jsonl  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--n", type=int)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args(argv)
    if args.n is not None and args.n < 1:
        ap.error("--n must be positive")
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")

    short = args.model.rstrip("/").split("/")[-1]
    store = Path("var/acts") / f"{short}_{args.variant}_s0_discovery"
    means = torch.load(store / "means.pt", map_location="cpu", weights_only=True)["heads"]
    records = read_jsonl(f"var/data/m2/{args.variant}_s0_discovery.jsonl")[:args.n]
    if not records:
        ap.error("discovery records must not be empty")
    model, tok = load_model(args.model, dtype="fp32")
    start = time.perf_counter()
    scores = head_attribution(model, tok, records, means, batch_size=args.batch_size)
    seconds = time.perf_counter() - start
    heads = scores["heads"].tolist()
    ranked = sorted(((layer, head) for layer, row in enumerate(heads)
                     for head in range(len(row))), key=lambda pair: heads[pair[0]][pair[1]])
    top = list(reversed(ranked[-10:]))
    bottom = ranked[:10]
    # Small models can have fewer than twenty heads; evaluate each only once.
    check_heads = list(dict.fromkeys(top + bottom))
    result = {
        "model": args.model, "variant": args.variant, "n": len(records),
        "heads": heads, "kv": scores["kv"].tolist(), "seconds": seconds,
        "check_heads": {"top": top, "bottom": bottom},
    }
    out_dir = Path("var/results/m3")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"attribution_{short}_{args.variant}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    checks = out_dir / f"check_heads_{short}_{args.variant}.json"
    checks.write_text(json.dumps(check_heads, indent=2) + "\n")
    print(f"attributed {len(records)} examples in {seconds:.1f}s -> {out}; checks -> {checks}")


if __name__ == "__main__":
    main()
