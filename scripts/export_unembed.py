"""Export the unembedding rows of every atlas task's option tokens (docs/ATLAS-overlap.md, Inputs).

  uv run python scripts/export_unembed.py [--out var/results/atlas1/unembed_options.npz]

Run on the pod before it is terminated. Reads the Qwen3-0.6B tokenizer and the tied embedding matrix
(`model.embed_tokens.weight`, which is also the output projection) and the final RMSNorm weight (`model.norm.weight`)
from the local HF cache through safetensors, offline, without building the model. The residual reaches the logits through
that norm, so the direction in residual space that drives an option's logit is its embedding row times the norm weight
(up to the per-record RMS scale); those products are what is saved. The npz holds, per task, `{task}` ->
(n_options, d_model) fp32 rows and `ids_{task}` -> the option token ids those rows came from.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
EMBED_KEY = "model.embed_tokens.weight"
NORM_KEY = "model.norm.weight"
DEFAULT_OUT = Path("var/results/atlas1/unembed_options.npz")


def save_unembed(path, matrix: np.ndarray, ids_by_task: dict) -> None:
    """One npz: `{task}` -> matrix[ids] as fp32, `ids_{task}` -> the ids."""
    arrays = {}
    for task, ids in ids_by_task.items():
        arrays[task] = np.asarray(matrix[list(ids)], dtype=np.float32)
        arrays[f"ids_{task}"] = np.asarray(ids, dtype=np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def load_unembed(path) -> tuple[dict, dict]:
    """(rows_by_task, ids_by_task) from a file written by `save_unembed`."""
    with np.load(path) as z:
        tasks = [k for k in z.files if not k.startswith("ids_")]
        return {t: z[t] for t in tasks}, {t: z[f"ids_{t}"] for t in tasks}


def _tensor(root: Path, key: str) -> np.ndarray:
    from safetensors import safe_open
    for f in sorted(root.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as st:
            if key in st.keys():
                return st.get_tensor(key).float().numpy()
    raise SystemExit(f"{key} not found in {root}")


def load_readout() -> np.ndarray:
    """Embedding (= unembedding) matrix times the final norm weight, (vocab, d_model) fp32."""
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(MODEL, local_files_only=True))
    return _tensor(root, EMBED_KEY) * _tensor(root, NORM_KEY)[None, :]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer
    from surgery.atlas import spec
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = {t: spec.option_token_ids(tok, t) for t in spec.PHASE1_TASKS}
    save_unembed(args.out, load_readout(), ids)
    print(f"wrote {args.out}: {len(ids)} tasks", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
