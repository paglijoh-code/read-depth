"""M9 atlas: shared cut order for Phase 1. ShortGPT block influence and skip-one-block perplexity on data/ppl_heldout.txt.

  uv run python scripts/atlas_order.py [--model NAME] [--out PATH]

Writes var/results/atlas/order.json (BI, perplexity drop per block, shared order). Perplexity rows resume from
var/results/atlas/order.partial.json. The order reads only the model and data/ppl_heldout.txt (no task results), so
`--model Qwen/Qwen3-1.7B --out var/results/atlas3/1.7B/order.json` gives the 1.7B order by the same rule; the partial file
sits beside --out. Defaults reproduce Phase 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgery.atlas import measure, metrics  # noqa: E402
from surgery.canaries import load_ppl_text  # noqa: E402
from surgery.eval import load_model  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
OUT = Path("var/results/atlas")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=str(OUT / "order.json"), help="final order.json path")
    args = ap.parse_args(argv)
    final = Path(args.out)
    part = final.with_name("order.partial.json")
    if final.exists():
        print(f"{final} exists; nothing to do")
        return
    text = load_ppl_text()
    model, tok = load_model(args.model, dtype="fp32")
    L = len(model.model.layers)
    state = json.loads(part.read_text()) if part.exists() else {}
    if "bi" not in state:
        t0 = time.time()
        state["bi"] = measure.block_influence(model, tok, text)
        measure._write_json(part, state)
        print(f"block influence done ({time.time() - t0:.0f}s)", flush=True)

    def save(base, ppl):
        state.update(base_ppl=base, ppl={str(k): v for k, v in ppl.items()})
        measure._write_json(part, state)
        print(f"ppl rows {len(ppl)}/{L}", flush=True)

    done = {int(k): v for k, v in state.get("ppl", {}).items()}
    res = measure.ppl_drop(model, tok, text, done=done, on_row=save)
    order = metrics.shared_order(state["bi"], L)
    out = {"model": args.model, "text": "data/ppl_heldout.txt", "n_layers": L, "bi": state["bi"], "base_ppl": res["base_ppl"],
           "ppl_tokens": res["n_tokens"], "ppl": {str(k): v for k, v in res["ppl"].items()}, "ppl_drop": res["drop"],
           "shared_order": order}
    measure._write_json(final, out)
    part.unlink(missing_ok=True)
    print(f"shared order {order} -> {final}", flush=True)


if __name__ == "__main__":
    main()
