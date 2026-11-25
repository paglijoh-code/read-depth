"""M8 step 2: rebuild each M7 model (excise, heal, fold, merge) and export it to MLX.

  uv run python scripts/mlx_export.py [--models rule-A,rule-B,...] [--dtypes fp32,bf16] [--out var/mlx]
Writes var/mlx/<name>/{fp32,bf16} (unmodified is one export) and records var/results/m8/export.json. Existing
exports are skipped. Run it only on a quiet machine (loads the 0.6B model on CPU).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgery import export as E  # noqa: E402

RESULT = Path("var/results/m8/export.json")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(E.MODELS))
    ap.add_argument("--dtypes", default="fp32,bf16")
    ap.add_argument("--out", default="var/mlx")
    a = ap.parse_args(argv)
    names, dtypes, out = a.models.split(","), a.dtypes.split(","), Path(a.out)
    for n in names:
        if n not in E.MODELS:
            sys.exit(f"unknown model {n!r}; known: {', '.join(E.MODELS)}")
    for d in dtypes:
        if d not in E.DTYPES:
            sys.exit(f"unknown dtype {d!r}")
    done = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    todo = [(n, d) for n in names for d in dtypes if not (out / n / d / "model.safetensors").exists()]
    if not todo:
        print("nothing to export")
        return
    from surgery.eval import load_model
    pristine, tok = load_model(E.MODEL, device="cpu")
    for name in dict.fromkeys(n for n, _ in todo):
        model = E.load_point(name, pristine=copy.deepcopy(pristine))
        for d in [d for n, d in todo if n == name]:
            man = E.export_mlx(model, out / name / d, d, tokenizer=tok)
            done.setdefault(name, {})[d] = man
            RESULT.parent.mkdir(parents=True, exist_ok=True)
            tmp = RESULT.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(done, indent=2) + "\n")
            tmp.replace(RESULT)
            print(f"{name}/{d}: {man['weight_bytes'] / 1e6:.0f} MB, {man['n_layers']} layers, "
                  f"model file {man['needs_model_file']}", flush=True)
        del model


if __name__ == "__main__":
    main()
