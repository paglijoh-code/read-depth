"""Residual-stream steering at the decision position (M4).

Directions are fit on a discovery store only. `layer l` means the output of decoder layer l
(`resid[:, l + 1]` in the store; the hook goes on `model.model.layers[l]`).
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from surgery import stats  # noqa: E402
from surgery.eval import _free, label_logits  # noqa: E402
from surgery.hooks import Handle, spec_hash  # noqa: E402
from surgery.prompt import build_prompt, label_token_ids  # noqa: E402
from surgery.task.generator import LABELS  # noqa: E402

DIRECTIONS = ("corr_dm", "corr_probe", "label_dm", "label_probe")
OPS = ("add", "project_out")


def _unit(v, eps: float = 1e-12):
    v = np.asarray(v, dtype=np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def _fit_probe(X, t, n_classes):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(sc.transform(X), t)
    coef, b = clf.coef_, clf.intercept_
    if n_classes == 2:  # sklearn stores one row for the positive class
        return sc.mean_, sc.scale_, coef[0], float(b[0])
    full = np.zeros((n_classes, X.shape[1]))
    fb = np.full(n_classes, -1e4)  # a class absent from training is never predicted
    full[clf.classes_] = coef
    fb[clf.classes_] = b
    return sc.mean_, sc.scale_, full, fb


def build_directions(store_dir, layers) -> dict:
    """Fit all four directions per layer on every row of the store (no selection/test data)."""
    store_dir = Path(store_dir)
    resid = np.load(store_dir / "resid.npy", mmap_mode="r")
    meta = json.loads((store_dir / "meta.json").read_text())
    y = np.asarray(meta["y"])
    c = (np.asarray(meta["pred"]) == y).astype(int)
    f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    out = {}
    for l in layers:
        X = np.asarray(resid[:, l + 1], dtype=np.float32)
        d = {"norm_mean": float(np.linalg.norm(X, axis=1).mean())}
        if c.min() == c.max():
            raise ValueError("predictions are all correct or all wrong; no correctness direction")
        d["corr_dm"] = f(_unit(X[c == 1].mean(0) - X[c == 0].mean(0)))
        mu, sc, coef, b = _fit_probe(X, c, 2)
        d.update(corr_mu=f(mu), corr_scale=f(sc), corr_coef=f(coef), corr_b=torch.tensor(b, dtype=torch.float32))
        d["corr_probe"] = f(_unit(coef / sc))
        mu, sc, coef, b = _fit_probe(X, y, len(LABELS))
        d.update(lab_mu=f(mu), lab_scale=f(sc), lab_coef=f(coef), lab_b=f(b))
        w = coef / sc
        d["label_probe"] = f(_unit(w - w.mean(0)))
        d["label_dm"] = f(_unit(np.stack([X[y == k].mean(0) if (y == k).any() else X.mean(0)
                                          for k in range(len(LABELS))]) - X.mean(0)))
        out[int(l)] = d
    out["meta"] = {"store": str(store_dir), "n": int(len(y)), "layers": [int(l) for l in layers]}
    return out


def save_directions(d: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, path)


def load_directions(path) -> dict:
    return torch.load(path, weights_only=False)


def config_id(cfg: dict) -> str:
    return spec_hash(cfg)


def validate_config(cfg: dict) -> None:
    if cfg.get("direction") not in DIRECTIONS:
        raise ValueError(f"unknown direction {cfg.get('direction')!r}")
    if cfg.get("op") not in OPS:
        raise ValueError(f"unknown op {cfg.get('op')!r}")
    if not isinstance(cfg.get("layer"), int):
        raise ValueError("config needs an int 'layer'")
    if cfg["op"] == "project_out" and cfg["direction"].startswith("label_"):
        raise ValueError("project_out needs a label-agnostic direction")
    if cfg["op"] == "add" and cfg.get("alpha") is None:
        raise ValueError("add needs alpha")
    if cfg.get("tau") is not None and not 0 <= cfg["tau"] <= 1:
        raise ValueError("tau must be in [0, 1]")


def steer_vectors(h: torch.Tensor, cfg: dict, D: dict):
    """h (B, d) float32 -> (h_new, mask); steered fraction = mask.float().mean(). Rows outside the mask are unchanged."""
    validate_config(cfg)
    ld = D[cfg["layer"]]
    ld = {k: (v.to(h.device) if torch.is_tensor(v) else v) for k, v in ld.items()}
    if cfg.get("tau") is None:
        mask = torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
    else:
        z = ((h - ld["corr_mu"]) / ld["corr_scale"]) @ ld["corr_coef"] + ld["corr_b"]
        mask = torch.sigmoid(z) < cfg["tau"]
    dvec = ld[cfg["direction"]]
    if cfg["direction"].startswith("label_"):
        s = ((h - ld["lab_mu"]) / ld["lab_scale"]) @ ld["lab_coef"].T + ld["lab_b"]
        dvec = dvec[s.argmax(-1)]  # (B, d)
    else:
        dvec = dvec.expand_as(h)
    if cfg["op"] == "add":
        new = h + cfg["alpha"] * ld["norm_mean"] * dvec
    else:
        new = h - (h * dvec).sum(-1, keepdim=True) * dvec
    new = torch.where(mask[:, None], new, h)
    return new, mask


def apply_steer(model, cfg: dict, D: dict, positions=None) -> Handle:
    """Forward hook on layer cfg['layer']. positions: None = last position; LongTensor (B,); or a callable."""
    validate_config(cfg)

    def hook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        B, T, _ = out.shape
        pos = positions() if callable(positions) else positions
        pos = (torch.full((B,), T - 1, dtype=torch.long) if pos is None
               else torch.as_tensor(pos, dtype=torch.long)).to(out.device)
        rows = torch.arange(B, device=out.device)
        h = out[rows, pos].float()
        new, mask = steer_vectors(h, cfg, D)
        handle.steered.append(float(mask.float().mean()))
        out = out.clone()
        out[rows, pos] = new.to(out.dtype)
        return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out

    handle = Handle([model.model.layers[cfg["layer"]].register_forward_hook(hook)])
    handle.steered = []
    return handle


@torch.no_grad()
def eval_configs(model, tok, records, configs, D, batch_size: int = 8) -> dict:
    """Prefix-cached evaluation: run the prompt minus its last token once per batch, then re-run
    only the last token for the baseline and for each config."""
    ids = torch.tensor(label_token_ids(tok), device=model.device)
    cids = [config_id(c) for c in configs]
    base, outs = [], {cid: [] for cid in cids}
    fracs = {cid: [] for cid in cids}

    def last_step(enc, past):
        lg = model(input_ids=enc["input_ids"][:, -1:], attention_mask=enc["attention_mask"],
                   past_key_values=past, use_cache=True, logits_to_keep=1).logits[:, -1, :]
        return lg[:, ids].float().cpu()

    for i in range(0, len(records), batch_size):
        enc = tok([build_prompt(tok, r["prompt"]) for r in records[i:i + batch_size]],
                  return_tensors="pt", padding=True).to(model.device)
        past = model(input_ids=enc["input_ids"][:, :-1], attention_mask=enc["attention_mask"][:, :-1],
                     use_cache=True).past_key_values
        base.append(last_step(enc, past))
        past.crop(-1)  # drop the re-run last token
        for cfg, cid in zip(configs, cids):
            with apply_steer(model, cfg, D) as h:
                outs[cid].append(last_step(enc, past))
            fracs[cid].append(float(np.mean(h.steered)))
            past.crop(-1)  # drop the re-run last token
        del enc, past
        _free(model)
    return {"baseline": torch.cat(base).numpy(),
            "configs": {cid: torch.cat(v).numpy() for cid, v in outs.items()},
            "steered_frac": {cid: float(np.mean(v)) for cid, v in fracs.items()}}


def plain_label_logits(model, tok, records, batch_size: int = 8):
    return label_logits(model, tok, [r["prompt"] for r in records], batch_size)


def summarize_config(records, logits, base_logits, seed: int = 0, n_boot: int = 2000) -> dict:
    y = np.array([LABELS.index(r["label"]) for r in records])
    groups = np.array([r["template_id"] for r in records])
    logits, base_logits = np.asarray(logits), np.asarray(base_logits)
    pred, bpred = logits.argmax(1), base_logits.argmax(1)
    probs = stats.softmax(logits)
    ent = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(1)
    return {
        "accuracy": stats.accuracy(y, pred),
        "paired": stats.paired_grouped_bootstrap(stats.accuracy, groups, y, bpred, pred,
                                                 n_boot=n_boot, seed=seed),
        "mcnemar": stats.mcnemar(bpred == y, pred == y),
        "entropy": float(ent.mean()),
        "ece": stats.ece(y, probs),
        "per_label_accuracy": {l: (float((pred[y == k] == k).mean()) if (y == k).any() else None)
                               for k, l in enumerate(LABELS)},
        "pred": pred.tolist(),
        "n": len(records),
    }


def score_canaries_steered(model, tok, items, cfg, D, batch_size: int = 8) -> dict:
    """Canary scoring with the steer applied at the last prompt token of every row (right-padded)."""
    from surgery.canaries import score_canaries
    state = {"pos": None}
    cfg = {**cfg, "tau": None}  # worst case: unconditional
    with apply_steer(model, cfg, D, positions=lambda: state["pos"]):
        return score_canaries(model, tok, items, batch_size,
                              on_batch=lambda n_prompt: state.__setitem__(
                                  "pos", torch.tensor(n_prompt, dtype=torch.long) - 1))
