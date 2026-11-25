"""Build the M3 activation store for one model and variant.

  uv run python scripts/build_store.py --model Qwen/Qwen3-0.6B --variant rule [--n N] [--check-m2]
Writes var/acts/{short}_{variant}_s0_discovery/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

from surgery.eval import load_model  # noqa: E402
from surgery.store import build_store, short_name  # noqa: E402
from surgery.task.generator import read_jsonl  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--variant", default="rule")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--check-m2", action="store_true")
    a = ap.parse_args(argv)
    short = short_name(a.model)
    data_path = f"var/data/m2/{a.variant}_s0_discovery.jsonl"
    recs = read_jsonl(data_path)[: a.n]
    out_dir = Path("var/acts") / f"{short}_{a.variant}_s0_discovery"
    model, tok = load_model(a.model, dtype="fp32")
    t0 = time.time()
    meta = build_store(model, tok, recs, out_dir, a.batch_size,
                       {"model": a.model, "variant": a.variant, "data_path": data_path})
    secs = time.time() - t0
    print(f"built {meta['n']} examples in {secs:.1f}s ({secs / meta['n'] * 40:.1f}s per 40) -> {out_dir}")
    if a.check_m2:
        base = json.loads(Path(f"var/results/baseline/{short}_{a.variant}_s0_discovery.json").read_text())
        n = meta["n"]
        assert meta["pred"] == base["pred"][:n], "pred mismatch vs M2 baseline"
        fl = np.load(out_dir / "final_logits.npy")
        delta = np.abs(fl - np.array(base["logits"][:n], dtype=np.float32)).max()
        print(f"check-m2: preds match on {n}; max |dlogit| = {delta:.3e}")


if __name__ == "__main__":
    main()
