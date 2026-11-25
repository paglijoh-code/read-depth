#!/usr/bin/env python
"""Export the activation store to one JSON the dashboard reads.

  uv run python scripts/map_export.py --model Qwen3-0.6B --variant rule

Reads var/acts/{short}_{variant}_s0_discovery/ and writes
var/results/m3/maps_{short}_{variant}.json: per-layer label/correctness probes,
logit-lens accuracy, residual-norm stats, a 2D PCA of 300 seeded examples per
layer, lens probabilities for 6 examples (one per label) and the mean lens
probability of the true label by true label.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from surgery import probes  # noqa: E402

LABELS = ["READ_FILE", "EDIT_FILE", "RUN_TEST", "SEARCH", "ASK_USER", "DONE"]


def short_name(model):
    return model.split("/")[-1]


def _round(a, nd=4):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def export(store, out_path, model, variant, probe_n=1500, n_pca=300, seed=0,
           n_boot=500, n_jobs=1, log=print):
    store = Path(store)
    meta = json.loads((store / "meta.json").read_text())
    resid = np.load(store / "resid.npy", mmap_mode="r")
    lens = np.load(store / "lens.npy")
    n, n_pos, _ = resid.shape
    y = np.asarray(meta["y"])
    pred = np.asarray(meta["pred"])
    groups = np.asarray(meta["template_ids"])
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # lens: softmax over the six label logits
    z = lens - lens.max(axis=2, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=2, keepdims=True)
    lens_acc = (lens.argmax(axis=2) == y[:, None]).mean(axis=0)
    p_true = p[np.arange(n), :, y]  # (N, L+1)
    lens_p_true = p_true.mean(axis=0)
    by_label = [p_true[y == c].mean(axis=0) if (y == c).any()
                else np.zeros(n_pos) for c in range(6)]
    ex_idx = []
    for c in range(6):
        cand = np.flatnonzero(y == c)
        ex_idx.append(int(rng.choice(cand)) if len(cand) else -1)

    # residual norms, float32, chunked
    norms = np.empty((n, n_pos), dtype=np.float32)
    for s in range(0, n, 500):
        blk = np.asarray(resid[s:s + 500], dtype=np.float32)
        norms[s:s + 500] = np.linalg.norm(blk, axis=2)
    log(f"norms/lens {time.time() - t0:.1f}s")

    # PCA on seeded examples
    pca_idx = np.sort(rng.choice(n, size=min(n_pca, n), replace=False))
    pca_blk = np.asarray(resid[pca_idx], dtype=np.float32)
    pca = []
    for i in range(n_pos):
        A = pca_blk[:, i] - pca_blk[:, i].mean(axis=0)
        _, sv, vt = np.linalg.svd(A, full_matrices=False)
        comp = vt[:2].copy()
        for k in range(len(comp)):
            if comp[k, np.argmax(np.abs(comp[k]))] < 0:
                comp[k] = -comp[k]
        xy = A @ comp.T
        if xy.shape[1] < 2:
            xy = np.pad(xy, ((0, 0), (0, 2 - xy.shape[1])))
        pca.append(dict(x=np.round(xy[:, 0].astype(float), 3).tolist(),
                        y=np.round(xy[:, 1].astype(float), 3).tolist()))
    log(f"pca {time.time() - t0:.1f}s")

    # probes (optionally on a seeded subset)
    probe_subset = n > probe_n
    sub = np.sort(rng.choice(n, size=probe_n, replace=False)) if probe_subset \
        else np.arange(n)
    X = np.asarray(resid[sub])
    correct = (pred == y).astype(int)
    lab = probes.probe_by_layer(X, y[sub], groups[sub], "label", seed,
                                n_boot, n_jobs)
    log(f"label probes {time.time() - t0:.1f}s")
    cor = probes.probe_by_layer(X, correct[sub], groups[sub], "correct", seed,
                                n_boot, n_jobs)
    log(f"correctness probes {time.time() - t0:.1f}s")

    layers = []
    for i in range(n_pos):
        layers.append(dict(
            layer=i,
            label_probe={k: round(v, 4) for k, v in lab[i].items() if k != "layer"},
            correct_probe={k: round(v, 4) for k, v in cor[i].items() if k != "layer"},
            lens_accuracy=round(float(lens_acc[i]), 4),
            lens_p_correct=round(float(lens_p_true[i]), 4),
            norm_mean=round(float(norms[:, i].mean()), 3),
            norm_p5=round(float(np.percentile(norms[:, i], 5)), 3),
            norm_p95=round(float(np.percentile(norms[:, i], 95)), 3),
            pca=pca[i]))

    result = dict(
        model=model, variant=variant, n=int(n), n_layers=int(meta["n_layers"]),
        d_model=int(resid.shape[2]), labels=LABELS,
        accuracy=round(float((pred == y).mean()), 4),
        probe_subset=dict(used=bool(probe_subset), n=int(len(sub)),
                          n_boot=n_boot),
        pca_labels=y[pca_idx].tolist(), pca_ids=[meta["ids"][j] for j in pca_idx],
        layers=layers,
        lens_examples=[dict(index=j, id=meta["ids"][j], label=int(y[j]),
                            pred=int(pred[j]),
                            probs=_round(p[j], 4)) for j in ex_idx if j >= 0],
        lens_by_label=_round(by_label, 4),
        seconds=round(time.time() - t0, 1))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, separators=(",", ":")))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--probe-n", type=int, default=1500,
                    help="seeded subset size for probes (default 1500)")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", help="override store directory")
    ap.add_argument("--out", help="override output path")
    a = ap.parse_args()
    short = short_name(a.model)
    store = a.store or ROOT / "var/acts" / f"{short}_{a.variant}_s0_discovery"
    out = a.out or ROOT / "var/results/m3" / f"maps_{short}_{a.variant}.json"
    r = export(store, out, short, a.variant, a.probe_n, seed=a.seed,
               n_boot=a.n_boot, n_jobs=a.n_jobs)
    print(f"wrote {out} in {r['seconds']}s")


if __name__ == "__main__":
    main()
