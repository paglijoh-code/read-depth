#!/usr/bin/env python
"""Depth-atlas gates 1 and 2 runner (docs/ATLAS-gates-run.md). One subcommand per stage, all resumable: an existing
output JSON (or heal.json) is skipped unless --force. Results go to --root (default var/results/atlas_gates); progress is
appended to <root>/status.txt. The per-model loops load each model once, smallest first.

  download                      snapshot_download every model, report sizes (needs the network; HF_HUB_OFFLINE unset)
  measure --prompt plain|twoshot  g1/<prompt>/<short>/<task>.{json,npz}   (probe, twin, lens, use, arrival, per-item arrays)
  robust_trained                g2/robust/<short>/<task>.json              (same 12 tasks on the healed models, post hoc)
  robust                        g1/robust/<short>/<task>.json              (C grid + mean-difference subspace, seeded draw)
  adapt                         g2/adapt/<short>/...                        (C-adapt: 1 heal per model, first LR task, 3e-4)
  lr                            6 tasks x 2 lrs x models, seed 0 -> g2/heals/..., g2/lr_choice.json
  heal --seed 0|1               all tasks (seed 0) / the 12 seed tasks (seed 1) at each model's chosen lr
  trained --seed 0|1            merge each adapter, measure (plain) -> g2/trained/s<seed>/<short>/<task>.{json,npz}

--smoke on any stage: 1 model (default EleutherAI/pythia-410m) x 2 tasks, heals of 10 steps on 64 train items, output
under <root>/smoke/. Every heal appends a line to <root>/g2/heal_log.tsv and touches var/.upload_now.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (HF id, parameters in millions): smallest first. Fourteen fitted models plus TinyLlama (held out).
MODELS = [
    ("HuggingFaceTB/SmolLM2-135M", 135), ("HuggingFaceTB/SmolLM2-360M", 360), ("EleutherAI/pythia-410m", 410),
    ("Qwen/Qwen2.5-0.5B", 494), ("Qwen/Qwen3-0.6B-Base", 600), ("EleutherAI/pythia-1b", 1010),
    ("TinyLlama/TinyLlama_v1.1", 1100), ("allenai/OLMo-2-0425-1B", 1480), ("allenai/OLMo-1B-0724-hf", 1180),
    ("EleutherAI/pythia-1.4b", 1410), ("Qwen/Qwen2.5-1.5B", 1540), ("Qwen/Qwen3-1.7B-Base", 1720),
    ("HuggingFaceTB/SmolLM2-1.7B", 1710), ("EleutherAI/pythia-2.8b", 2800), ("Qwen/Qwen2.5-3B", 3090),
]
ALL_MODELS = [m for m, _ in sorted(MODELS, key=lambda x: x[1])]
SMOKE_MODEL = "EleutherAI/pythia-410m"
LRS = (1e-3, 3e-4)
ADAPT_LR = 3e-4
LEARNABLE = 0.45
PICK = Path("var/cscreen/pick_v13.json")


def short(mid: str) -> str:
    return mid.split("/")[-1]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def touch_upload():
    p = Path("var/.upload_now")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


class Ctx:
    def __init__(self, a):
        from surgery.atlas import families as fam
        from surgery.atlas import gates as G
        self.G, self.fam, self.a = G, fam, a
        self.smoke = a.smoke
        self.root = Path(a.root) / ("smoke" if a.smoke else "")
        self.root.mkdir(parents=True, exist_ok=True)
        if a.models == "all":
            self.models = [SMOKE_MODEL] if a.smoke else list(ALL_MODELS)
        else:
            self.models = a.models.split(",")
        if a.tasks == "main":
            main = json.loads(PICK.read_text())["main"]
            tasks = [t for f in main for t in fam.FAMILIES[f][1]]
        else:
            tasks = a.tasks.split(",")
        self.tasks = tasks[:2] if (a.smoke and a.tasks == "main") else tasks
        self.steps = 10 if a.smoke else G.STEPS
        self.limit_train = 64 if a.smoke else None
        self._device = a.device

    @property
    def device(self):
        return self.G._p0h().pick_device(self._device)

    def say(self, msg):
        line = f"{_now()} {msg}"
        with open(self.root / "status.txt", "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def done(self, p: Path) -> bool:
        return p.exists() and not self.a.force


def _free():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _err(c, where, e):
    tb = traceback.extract_tb(e.__traceback__)[-1]
    c.say(f"ERROR {where}: {type(e).__name__}: {e} ({tb.filename}:{tb.lineno})")


def _load_adapter(c, mid):
    from surgery.atlas.p0_harness import Adapter
    ad = Adapter(mid, device=c.device)
    ad.model_id = mid
    c.say(f"{short(mid)}: loaded in {ad.load_seconds:.0f}s on {c.device}, B={ad.n_blocks} d={ad.d}")
    return ad


def _summary(r):
    pc, on = r["probe"], r["onset"]
    f = lambda o: "-" if o is None else f"{o['rel']:.2f}"  # noqa: E731
    return (f"acc={r['test_item_acc']:.2f} pairs={r['n_pairs_used']} probe[sel={pc['peak_selectivity']:.2f} "
            f"onset={f(on['probe'])}] twin[onset={f(on['twin'])}] use[onset={f(on['use'])}] arr[onset={f(on['arrival'])}] "
            f"valid={ {k: int(v) for k, v in r['valid'].items()} } {r['timing']['total_s']:.0f}s")


# ---------------------------------------------------------------- stages

def cmd_download(c):
    from huggingface_hub import snapshot_download
    total = 0
    for mid in c.models:
        t0 = time.time()
        try:
            try:
                p = snapshot_download(mid, ignore_patterns=["*.bin", "*.pt", "*.h5", "*.msgpack", "*.onnx", "*.gguf", "original/*"])
                if not list(Path(p).glob("*.safetensors")):
                    p = snapshot_download(mid)
            except Exception:  # noqa: BLE001
                p = snapshot_download(mid)
            size = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
            total += size
            c.say(f"download {mid}: {size / 1e9:.2f} GB in {time.time() - t0:.0f}s -> {p}")
        except Exception as e:  # noqa: BLE001
            _err(c, f"download {mid}", e)
    c.say(f"download total {total / 1e9:.2f} GB")
    return 0


def cmd_measure(c):
    G, bad = c.G, 0
    for mid in c.models:
        out = c.root / "g1" / c.a.prompt / short(mid)
        todo = [t for t in c.tasks if not c.done(out / f"{t}.json")]
        if not todo:
            c.say(f"{short(mid)} {c.a.prompt}: nothing to do")
            continue
        ad = _load_adapter(c, mid)
        for t in todo:
            try:
                r = G.run_measure(ad, t, c.a.prompt, out / f"{t}.json", out / f"{t}.npz", n_perm=c.a.n_perm,
                                  log=lambda *a: None)
                c.say(f"{short(mid)} {c.a.prompt} {t} {_summary(r)}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                _err(c, f"{short(mid)} {c.a.prompt} {t}", e)
        del ad
        _free()
    return 1 if bad else 0


def robust_plan(c):
    """(models, tasks): the seeded draw of 3 models and 6 families from what was asked for."""
    fams = sorted({c.fam.get_task(t).family for t in c.tasks})
    d = c.G.robust_draw(c.models, fams)
    tasks = [t for t in c.tasks if c.fam.get_task(t).family in d["families"]]
    if c.a.robust_one_task:
        seen, keep = set(), []
        for t in tasks:
            f = c.fam.get_task(t).family
            if f not in seen:
                seen.add(f)
                keep.append(t)
        tasks = keep
    return d, tasks


def cmd_robust(c):
    G, bad = c.G, 0
    d, tasks = robust_plan(c)
    (c.root / "g1" / "robust").mkdir(parents=True, exist_ok=True)
    (c.root / "g1" / "robust" / "draw.json").write_text(json.dumps({**d, "tasks": tasks}, indent=1))
    c.say(f"robust draw: models={d['models']} families={d['families']} tasks={len(tasks)}")
    for mid in d["models"]:
        out = c.root / "g1" / "robust" / short(mid)
        todo = [t for t in tasks if not c.done(out / f"{t}.json")]
        if not todo:
            continue
        ad = _load_adapter(c, mid)
        for t in todo:
            try:
                t0 = time.time()
                r = G.run_robust(ad, t)
                p = out / f"{t}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(r, default=G._json))
                on = {k: (None if v is None else round(v, 2)) for k, v in r["onsets"].items()}
                c.say(f"{short(mid)} robust {t} onsets={on} robust={r.get('robust')} {time.time() - t0:.0f}s")
            except Exception as e:  # noqa: BLE001
                bad += 1
                _err(c, f"{short(mid)} robust {t}", e)
        del ad
        _free()
    return 1 if bad else 0


def cmd_robust_trained(c):
    """The robustness arm on the trained kind (post hoc; DECISIONS 2026-11-24): the gate-1 arm's families, every model asked for."""
    G, bad = c.G, 0
    fams = sorted({c.fam.get_task(t).family for t in c.tasks})
    d = G.robust_draw(ALL_MODELS, fams)
    tasks = [t for t in c.tasks if c.fam.get_task(t).family in d["families"]]
    base = c.root / "g2" / "robust"
    base.mkdir(parents=True, exist_ok=True)
    (base / "draw.json").write_text(json.dumps({"models": c.models, "families": d["families"], "tasks": tasks}, indent=1))
    c.say(f"robust_trained: models={len(c.models)} families={d['families']} tasks={len(tasks)}")
    for mid in c.models:
        lr = model_lr(c, mid)
        if lr is None:
            c.say(f"{short(mid)} robust_trained: no chosen lr, skipped")
            bad += 1
            continue
        for t in tasks:
            p = base / short(mid) / f"{t}.json"
            if c.done(p):
                continue
            hd = heal_dir(c.root / "g2" / "heals", mid, t, lr, c.a.seed)
            if not (hd / "heal.json").exists():
                c.say(f"{short(mid)} robust_trained {t}: no heal at {hd}, skipped")
                bad += 1
                continue
            try:
                t0 = time.time()
                heal = json.loads((hd / "heal.json").read_text())
                ad = G.load_trained(mid, hd / "adapter.safetensors", c.device)
                r = G.run_robust(ad, t)
                r.update(trained=True, lr=lr, heal_seed=c.a.seed, heal_test_acc=heal["test_acc"],
                         learnable=bool(heal["test_acc"] >= LEARNABLE))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(r, default=G._json))
                on = {k: (None if v is None else round(v, 2)) for k, v in r["onsets"].items()}
                c.say(f"{short(mid)} robust_trained {t} pairs={r['n_pairs_used']} onsets={on} robust={r.get('robust')} "
                      f"{time.time() - t0:.0f}s")
                del ad
            except Exception as e:  # noqa: BLE001
                bad += 1
                _err(c, f"{short(mid)} robust_trained {t}", e)
            _free()
    return 1 if bad else 0


def heal_dir(base: Path, mid, task, lr, seed) -> Path:
    return base / short(mid) / f"{task}_lr{lr:g}_s{seed}"


def _load_model(c, mid):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(c.device).eval()
    c.say(f"{short(mid)}: model loaded on {c.device} in {time.time() - t0:.0f}s")
    return model, tok


def heal_jobs(c, mid, jobs, base, reuse=None):
    """jobs = [(task, lr, seed)]. Skips a finished heal, copies one from `reuse` (a directory with the same layout), else
    heals. Returns the number of failures."""
    G, bad, ctx = c.G, 0, {}
    cache = {}
    for task, lr, seed in jobs:
        d = heal_dir(base, mid, task, lr, seed)
        if c.done(d / "heal.json"):
            continue
        src = heal_dir(reuse, mid, task, lr, seed) if reuse else None
        if src and (src / "heal.json").exists() and not c.a.force:
            shutil.copytree(src, d, dirs_exist_ok=True)
            c.say(f"{short(mid)} heal {task} lr{lr:g} s{seed}: reused {src}")
            continue
        if "model" not in ctx:
            ctx["model"], ctx["tok"] = _load_model(c, mid)
        try:
            a = _now()
            r = G.heal_task_any(ctx["model"], ctx["tok"], task, lr, seed, base / short(mid), c.device, steps=c.steps,
                                limit_train=c.limit_train, base_cache=cache, log=lambda *x: None)
            log = c.root / "g2" / "heal_log.tsv"
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a") as f:
                f.write("\t".join(map(str, [a, _now(), mid, task, lr, seed, f"{r['wall_seconds']:.1f}",
                                            f"{r['dev_acc']:.4f}", f"{r['test_acc']:.4f}"])) + "\n")
            touch_upload()
            c.say(f"{short(mid)} heal {task} lr{lr:g} s{seed}: wall {r['wall_seconds']:.0f}s dev {r['dev_acc']:.3f} "
                  f"test {r['test_acc']:.3f} (base {r['base_test_acc']:.3f})")
        except Exception as e:  # noqa: BLE001
            bad += 1
            _err(c, f"{short(mid)} heal {task} lr{lr:g} s{seed}", e)
            _free()
    if "model" in ctx:
        del ctx["model"]
        _free()
    return bad


def cmd_adapt(c):
    task = c.G.lr_tasks(c.tasks)[0]
    c.say(f"adapt task {task} lr {ADAPT_LR:g}")
    return min(1, sum(heal_jobs(c, mid, [(task, ADAPT_LR, 0)], c.root / "g2" / "adapt") for mid in c.models))


def choose_lr(c, mid):
    """The model's lr from its lr-stage heals (6 tasks x 2 lrs, seed 0): `pick_lr`. None if any is missing."""
    G = c.G
    rows = []
    for task in G.lr_tasks(c.tasks):
        for lr in LRS:
            p = heal_dir(c.root / "g2" / "heals", mid, task, lr, 0) / "heal.json"
            if not p.exists():
                return None
            rows.append(json.loads(p.read_text()))
    return G.pick_lr(rows)


def cmd_lr(c):
    G, bad = c.G, 0
    jobs = [(t, lr, 0) for t in G.lr_tasks(c.tasks) for lr in LRS]
    p = c.root / "g2" / "lr_choice.json"
    for mid in c.models:
        bad += heal_jobs(c, mid, jobs, c.root / "g2" / "heals", reuse=c.root / "g2" / "adapt")
        lr = choose_lr(c, mid)
        if lr is not None:
            ch = json.loads(p.read_text()) if p.exists() else {}
            ch[short(mid)] = lr
            p.write_text(json.dumps(ch, indent=1))
            c.say(f"{short(mid)} chosen lr {lr:g}")
    return 1 if bad else 0


def model_lr(c, mid):
    if c.a.lr is not None:
        return c.a.lr
    p = c.root / "g2" / "lr_choice.json"
    ch = json.loads(p.read_text()) if p.exists() else {}
    return ch.get(short(mid)) or choose_lr(c, mid)


def heal_tasks(c, seed):
    return c.tasks if seed == 0 else c.G.seed_tasks(c.tasks)


def cmd_heal(c):
    bad = 0
    tasks = heal_tasks(c, c.a.seed)
    for mid in c.models:
        lr = model_lr(c, mid)
        if lr is None:
            c.say(f"{short(mid)} heal: no chosen lr yet (run the lr stage first or pass --lr), skipped")
            bad += 1
            continue
        bad += heal_jobs(c, mid, [(t, lr, c.a.seed) for t in tasks], c.root / "g2" / "heals")
    return 1 if bad else 0


def cmd_trained(c):
    G, bad = c.G, 0
    tasks = heal_tasks(c, c.a.seed)
    for mid in c.models:
        lr = model_lr(c, mid)
        out = c.root / "g2" / "trained" / f"s{c.a.seed}" / short(mid)
        if lr is None:
            c.say(f"{short(mid)} trained: no chosen lr, skipped")
            bad += 1
            continue
        for t in tasks:
            if c.done(out / f"{t}.json"):
                continue
            hd = heal_dir(c.root / "g2" / "heals", mid, t, lr, c.a.seed)
            if not (hd / "heal.json").exists():
                c.say(f"{short(mid)} trained {t}: no heal at {hd}, skipped")
                bad += 1
                continue
            try:
                heal = json.loads((hd / "heal.json").read_text())
                ad = G.load_trained(mid, hd / "adapter.safetensors", c.device)
                r = G.run_measure(ad, t, "plain", out / f"{t}.json", out / f"{t}.npz", n_perm=c.a.n_perm,
                                  log=lambda *a: None,
                                  extra={"trained": True, "lr": lr, "heal_seed": c.a.seed, "heal": heal,
                                         "learnable": bool(heal["test_acc"] >= LEARNABLE)})
                c.say(f"{short(mid)} trained s{c.a.seed} {t} heal_test={heal['test_acc']:.2f} {_summary(r)}")
                del ad
            except Exception as e:  # noqa: BLE001
                bad += 1
                _err(c, f"{short(mid)} trained s{c.a.seed} {t}", e)
            _free()
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["download", "measure", "robust", "robust_trained", "adapt", "lr", "heal", "trained"])
    ap.add_argument("--root", default="var/results/atlas_gates")
    ap.add_argument("--models", default="all", help="comma list of HF ids, or all (the 15 models, smallest first)")
    ap.add_argument("--tasks", default="main", help="comma list, or main (the 50 tasks of var/cscreen/pick_v13.json main)")
    ap.add_argument("--prompt", choices=["plain", "twoshot"], default="plain", help="measure")
    ap.add_argument("--seed", type=int, default=0, choices=[0, 1], help="heal / trained")
    ap.add_argument("--lr", type=float, default=None, help="heal / trained: override the chosen lr")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--robust-one-task", action="store_true", help="robust: one task per drawn family (default both)")
    a = ap.parse_args()
    c = Ctx(a)
    c.say(f"START {a.stage} models={len(c.models)} tasks={len(c.tasks)} smoke={a.smoke} force={a.force}")
    rc = {"download": cmd_download, "measure": cmd_measure, "robust": cmd_robust, "robust_trained": cmd_robust_trained, "adapt": cmd_adapt, "lr": cmd_lr,
          "heal": cmd_heal, "trained": cmd_trained}[a.stage](c)
    c.say(f"END {a.stage} rc={rc}")
    touch_upload()
    sys.exit(rc)


if __name__ == "__main__":
    main()
