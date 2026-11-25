"""C-check (proposal section 5, C4, scripted): model ids, blocks, licences, single-token answers, leaks, template splits,
rivals, the onset rule against paper one's stored curves, tamper tests and the planted toy under the v11 probe test.

Pure functions that return JSON-able dicts; `scripts/ccheck_run.py` is the CLI that writes them to var/ccheck/.
"""
from __future__ import annotations

import gc
import glob
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

from surgery.atlas import p0_measures as M
from surgery.atlas import p0_tasks as T
from surgery.atlas.metrics import l_on

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "var" / "ccheck"
MIN_FREE_GB = 25
PROBE_TEXT = " The capital of France is"

# (id, expected blocks, role). Roles: fitted unless noted.
MODELS = [
    ("EleutherAI/pythia-160m", 12, "screening"), ("HuggingFaceTB/SmolLM-135M", 30, "screening"),
    ("Qwen/Qwen3-0.6B-Base", 28, "fitted"), ("Qwen/Qwen3-1.7B-Base", 28, "fitted"),
    ("Qwen/Qwen2.5-0.5B", 24, "fitted"), ("Qwen/Qwen2.5-1.5B", 28, "fitted"), ("Qwen/Qwen2.5-3B", 36, "fitted"),
    ("HuggingFaceTB/SmolLM2-135M", 30, "fitted"), ("HuggingFaceTB/SmolLM2-360M", 32, "fitted"),
    ("HuggingFaceTB/SmolLM2-1.7B", 24, "fitted"), ("EleutherAI/pythia-410m", 24, "fitted"),
    ("EleutherAI/pythia-1b", 16, "fitted"), ("EleutherAI/pythia-1.4b", 24, "fitted"),
    ("EleutherAI/pythia-2.8b", 32, "fitted"), ("allenai/OLMo-2-0425-1B", 16, "fitted"),
    ("allenai/OLMo-1B-0724-hf", 16, "fitted"), ("TinyLlama/TinyLlama_v1.1", 22, "held out"),
]


# ---------------------------------------------------------------- output plumbing
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def status(msg, out_dir=None):
    out_dir = Path(out_dir or OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    line = f"{now()} {msg}"
    with open(out_dir / "status.txt", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def write_json(name, obj, out_dir=None):
    out_dir = Path(out_dir or OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=_jsonable))
    return p


# ---------------------------------------------------------------- models
def fetch_info(mid):
    """Hugging Face model_info, no token."""
    from huggingface_hub import HfApi
    return HfApi().model_info(mid)


def fetch_config(mid):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(mid)


def fetch_tokenizer(mid):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(mid)


def free_disk_gb():
    return shutil.disk_usage(os.path.expanduser("~")).free / 1e9


def _card(info):
    c = getattr(info, "card_data", None) or getattr(info, "cardData", None) or {}
    if not isinstance(c, dict):
        c = c.to_dict() if hasattr(c, "to_dict") else {k: getattr(c, k) for k in dir(c) if not k.startswith("_")}
    return c


def _iso(x):
    return x.isoformat() if hasattr(x, "isoformat") else (None if x is None else str(x))


def hub_record(info):
    card = _card(info)
    tags = list(getattr(info, "tags", None) or [])
    return {"gated": getattr(info, "gated", None), "license": card.get("license"),
            "license_name": card.get("license_name"), "license_link": card.get("license_link"),
            "tags_license": [t for t in tags if t.startswith("license:")],
            "tags_base_model": [t for t in tags if t.startswith("base_model:")],
            "tags_conversational": "conversational" in tags,
            "created_at": _iso(getattr(info, "created_at", None) or getattr(info, "createdAt", None)),
            "last_modified": _iso(getattr(info, "last_modified", None) or getattr(info, "lastModified", None)),
            "sha": getattr(info, "sha", None)}


def base_check(mid, tok, hub):
    """base_ok = no instruct/chat/-it in the id AND (no chat_template OR the repo is not tagged conversational).
    Raw evidence is recorded for judgement."""
    name_flag = re.search(r"instruct|chat|-it(?![a-z])", mid, re.I) is not None
    tpl = getattr(tok, "chat_template", None) if tok is not None else None
    return {"name_flag": name_flag, "chat_template_present": bool(tpl), "tagged_conversational": hub["tags_conversational"],
            "base_ok": bool((not name_flag) and ((not tpl) or (not hub["tags_conversational"])))}


def tokenizer_check(tok):
    out = {}
    for name in T.TASK_NAMES:
        try:
            T.check_options(tok, name)
            out[name] = "ok"
        except ValueError as e:
            out[name] = str(e)
    return out


def load_probe(mid, tok_ok, device="mps"):
    """Load on `device`, one forward, answer-token frequency for the tasks whose options are single tokens; free it."""
    import torch
    from surgery.atlas.p0_harness import Adapter
    rec = {}
    ad = None
    try:
        t0 = time.time()
        ad = Adapter(mid, device=device)
        rec["load_s"] = time.time() - t0
        rec["n_blocks"], rec["d"] = ad.n_blocks, ad.d
        ids = ad.encode([PROBE_TEXT])[0]
        with torch.no_grad():
            lg = ad.model(torch.tensor([ids], device=ad.device)).logits[0, -1].float().cpu().numpy()
        rec["logits_finite"] = bool(np.isfinite(lg).all())
        rec["answer_logfreq"] = {}
        for name, ok in tok_ok.items():
            if ok == "ok":
                rec["answer_logfreq"][name] = ad.answer_logfreq(T.check_options(ad.tok, name))
    finally:
        del ad
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    return rec


def check_model(mid, expected_blocks, role, no_load=False, device="mps", loader=None):
    rec = {"id": mid, "role": role, "expected_blocks": expected_blocks, "checked_at": now()}
    try:
        hub = hub_record(fetch_info(mid))
    except Exception as e:  # network, auth, missing repo
        rec["hub_error"] = f"{type(e).__name__}: {e}"
        return rec
    rec["hub"] = hub
    if hub["gated"]:
        rec["load"] = "skipped: gated"
        return rec
    try:
        cfg = fetch_config(mid)
        rec["config"] = {"model_type": cfg.model_type, "num_hidden_layers": cfg.num_hidden_layers,
                         "hidden_size": cfg.hidden_size}
        rec["blocks_match"] = bool(cfg.num_hidden_layers == expected_blocks)
        tok = fetch_tokenizer(mid)
        rec["base"] = base_check(mid, tok, hub)
        rec["tokenizer"] = tokenizer_check(tok)
    except Exception as e:
        rec["config_error"] = f"{type(e).__name__}: {e}"
        return rec
    if no_load:
        rec["load"] = "skipped: --no-load"
        return rec
    free = free_disk_gb()
    rec["free_disk_gb"] = free
    if free < MIN_FREE_GB:
        rec["load"] = "skipped: disk"
        return rec
    try:
        rec["load"] = (loader or load_probe)(mid, rec["tokenizer"], device)
    except Exception as e:
        rec["load"] = f"error: {type(e).__name__}: {e}"
    return rec


def run_models(ids=None, no_load=False, force=False, out_dir=None, device="mps"):
    out_dir = Path(out_dir or OUT)
    path = out_dir / "models.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    table = {m[0]: m for m in MODELS}
    for mid in (ids or [m[0] for m in MODELS]):
        _, nb, role = table.get(mid, (mid, None, "unlisted"))
        if mid in done and not force:
            status(f"models: skip {mid} (done)", out_dir)
            continue
        status(f"models: start {mid}", out_dir)
        done[mid] = check_model(mid, nb, role, no_load=no_load, device=device)
        write_json("models.json", done, out_dir)
        r = done[mid]
        status(f"models: done {mid} gated={r.get('hub', {}).get('gated')} blocks_match={r.get('blocks_match')} "
               f"base_ok={r.get('base', {}).get('base_ok')} load={'ok' if isinstance(r.get('load'), dict) else r.get('load')}",
               out_dir)
    return done


# ---------------------------------------------------------------- tasks
def _word_in(word, text):
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", text) is not None


def _expr_stripped(prompt):
    """Arithmetic prompts: drop the operand expression, so an operand equal to the result is not a leak."""
    return re.sub(r"\d+(?:[+\-]\d+)+=", "=", prompt)


def _leaks(task, items):
    """Answer word in its own prompt (item and answer twin). `strict` counts every occurrence; `informative` counts
    those where some other option is absent from the prompt (an answer that stands out). A prompt that lists all four
    options (hop_ctx's context names every continent) leaks nothing by itself."""
    opts = [o.strip() for o in task.options]
    strict = informative = 0
    for x in items:
        for p, a in ((x["prompt"], x["answer"]), (x["ans_twin"]["prompt"], x["ans_twin"]["answer"])):
            q = _expr_stripped(p)
            present = [_word_in(o, q if o.isdigit() else p) for o in opts]
            if present[a]:
                strict += 1
                informative += not all(present)
    return {"violations": informative, "strict_violations": strict, "checked": 2 * len(items)}


def _split_pools(spec):
    """class -> (train words, test words) for split-word tasks; None for tasks without a class-word pool."""
    if not isinstance(spec, T.WordTask) or not spec.split_words:
        return None
    return [(T._split_words(p, "train"), T._split_words(p, "test")) for p in spec.class_pools]


def check_tasks(models=None):
    """Per-task C-check of the P0 generators (seed 0). `models` = the models.json dict (single-token summary)."""
    out = {}
    for name in T.TASK_NAMES:
        task = T.get_task(name)
        items = T.generate(name, seed=0)
        tr, te = T.split_items(items)
        rec = {"n_items": len(items), "n_train": len(tr), "n_test": len(te)}
        # (a) single-token answers, per tokenizer, from models.json
        if models:
            res = {mid: m["tokenizer"][name] for mid, m in models.items() if "tokenizer" in m}
            rec["single_token"] = {"n_pass": sum(v == "ok" for v in res.values()), "n_fail": sum(v != "ok" for v in res.values()),
                                   "failures": {k: v for k, v in res.items() if v != "ok"}}
        else:
            rec["single_token"] = "no models.json yet (run `models`)"
        # (b) leaks
        if task.copy:
            rec["leaks"] = {"note": "copy task: the answer is in the prompt by design", "violations": 0,
                            "position_rival_defined": T.rival_position(items, name) is not None}
            leak_ok = rec["leaks"]["position_rival_defined"]
        else:
            rec["leaks"] = _leaks(task, items)
            leak_ok = rec["leaks"]["violations"] == 0
        pt = {x["prompt"] for x in tr}
        dup = sum(x["prompt"] in pt for x in te)
        rec["duplicate_prompts_train_test"] = dup
        # (c) template and class-word splits
        both = {x["template"] for x in tr} & {x["template"] for x in te}
        rec["template_overlap"] = len(both)
        pools = _split_pools(task.spec)
        if pools is None:
            rec["class_words"] = {"note": "no split class-word pool (copy task or non-word task)", "overlap": 0, "ok": True}
        else:
            tr_w, te_w = set(), set()
            for a, b in pools:
                tr_w |= set(a)
                te_w |= set(b)
            key = task.spec.cls_key
            used_tr = {x["slots"][key] for x in tr}
            used_te = {x["slots"][key] for x in te}
            ov = len(tr_w & te_w) + len(used_tr & used_te) + len(used_tr - tr_w) + len(used_te - te_w)
            rec["class_words"] = {"n_train_pool": len(tr_w), "n_test_pool": len(te_w), "overlap": ov, "ok": ov == 0}
        # (d) rivals
        kept = items
        r = T.rivals(kept, name)
        isf = lambda v: isinstance(v, float) and 0 <= v <= 1
        rec["rivals"] = r
        riv_ok = isf(r["bow"]) and isf(r["near_miss"]) and (isf(r["position"]) if task.copy else r["position"] is None)
        rec["checks"] = {"leaks": leak_ok, "duplicates": dup == 0, "template_split": not both,
                         "class_word_split": rec["class_words"]["ok"], "rivals": bool(riv_ok)}
        rec["pass"] = all(rec["checks"].values())
        out[name] = rec
    return {"tasks": out, "n_pass": sum(v["pass"] for v in out.values()), "n_tasks": len(out),
            "pass": all(v["pass"] for v in out.values())}


# ---------------------------------------------------------------- onset reproduction
def onset_check_map(d, tol=1.0):
    """One stored paper-one map. None if it has no probe curve or no stored control."""
    a = (d.get("probe") or {}).get("all") if isinstance(d.get("probe"), dict) else None
    if not a or "curve" not in a:
        return None
    ctrl = a.get("control") or d.get("control_curve")
    if not isinstance(ctrl, list):
        return None
    curve, chance = np.asarray(a["curve"], float), float(a.get("chance", d.get("chance")))
    sel = (curve - np.asarray(ctrl, float)) / (1 - chance)
    sel[0] = 0.0  # baseline 0 at the embedding (4.3 item 7)
    B = len(curve) - 1
    on = M.onset(sel, B)
    stored = d.get("L_on")
    recomputed = l_on([float(c) for c in curve], chance)
    v11 = None if on is None else on["layers"]
    if v11 is None or stored is None:
        diff, ok = None, v11 is None and stored is None
    else:
        diff = float(v11 - stored)
        ok = abs(diff) <= tol
    return {"model": d.get("model"), "task": d.get("task"), "B": B, "L_on": stored, "L_on_recomputed": recomputed,
            "recompute_match": recomputed == stored, "v11_layers": v11, "diff": diff, "pass": bool(ok),
            "smoke": d.get("smoke")}


def onset_check(root=None):
    root = Path(root or REPO / "var" / "results")
    files = sorted(glob.glob(str(root / "atlas*" / "**" / "map_*.json"), recursive=True))
    rows, skipped = [], []
    for f in files:
        rel = os.path.relpath(f, root)
        d = json.loads(Path(f).read_text())
        r = onset_check_map(d)
        if r is None:
            skipped.append(rel)
            continue
        if r["model"] is None:
            r["model"] = f"(default, {Path(rel).parent.name})"
        rows.append({"file": rel, **r})
    n_pass = sum(r["pass"] for r in rows)
    return {"rows": rows, "skipped_no_control": skipped, "n_files": len(files), "n_checked": len(rows),
            "n_pass": n_pass, "n_fail": len(rows) - n_pass,
            "n_recompute_mismatch": sum(not r["recompute_match"] for r in rows),
            "note": "v11 onset uses the selectivity curve (control subtracted, divided by 1 - chance, baseline 0); paper "
                    "one used accuracy from chance. The difference is the point of the check."}


# ---------------------------------------------------------------- tamper
SIGNALS = ("probe", "twin", "use")


def run_summary(r):
    pc, td, us, ar = r["probe"], r["twin"], r["use"], r["arrival"]
    return {"valid": dict(r["valid"]),
            "probe": {"peak_selectivity": pc["peak_selectivity"], "p_binom_bonf": pc["p_binom_bonf"]},
            "twin": {"peak": td["peak"], "p_perm": td["p_perm"], "trough": td["trough"], "valid_sym": td["valid_sym"]},
            "n_pairs": r["n_pairs_qualifying"],
            "use": None if us is None else {"peak": us["peak"], "p_signflip": us["p_signflip"]},
            "arrival_onset": None if r["onset"]["arrival"] is None else r["onset"]["arrival"]["layers"]}


def tamper_variants(d, seed=0):
    """Field overrides for the two tampered runs of a built task (`d` has y_tr, y_te, y_an, ids_an, ids_sf)."""
    rng = np.random.default_rng(seed)
    shuffled = {k: np.asarray(d[k])[rng.permutation(len(d[k]))] for k in ("y_tr", "y_te", "y_an")}
    swapped = {"ids_an": d["ids_sf"], "ids_sf": d["ids_an"], "y_an": list(d["y_te"])}
    # train labels only: test labels and twins stay true, so the use curve keeps its pairs and must fail on its merits
    # (with all labels shuffled, use fails mostly through the 40-pair floor)
    train_only = {"y_tr": np.asarray(d["y_tr"])[np.random.default_rng(seed + 1).permutation(len(d["y_tr"]))]}
    return {"shuffled_labels": shuffled, "shuffled_train_labels": train_only, "swapped_twins": swapped}


# Added in C-check review: reported, not part of pass (a finding for the desk, see reports/ccheck_depth_atlas.md)
DESCRIPTIVE = ("shuffled_train_labels",)
# Which filters a tamper can possibly move: the twin curve ignores labels, the probe ignores twins.
APPLICABLE = {"untampered": (), "shuffled_labels": ("probe", "use"), "shuffled_train_labels": ("probe", "use"),
              "swapped_twins": ("twin", "use")}


def tamper_report(runs):
    """runs = {"untampered": summary, "shuffled_labels": ..., "swapped_twins": ...} -> pass/fail."""
    chk = {}
    for k in [k for k in APPLICABLE if k != "untampered" and k in runs]:
        v = runs[k]["valid"]
        if k in DESCRIPTIVE:
            chk[k] = {"descriptive": True, "still_valid": [s for s in SIGNALS if v[s]]}
            continue
        chk[k] = {"all_signals_invalid": not any(v[s] for s in SIGNALS),
                  "applicable_invalid": not any(v[s] for s in APPLICABLE[k]),
                  "still_valid": [s for s in SIGNALS if v[s]]}
    return {"checks": chk, "applicable": {k: list(v) for k, v in APPLICABLE.items()},
            "untampered_valid": runs["untampered"]["valid"],
            "pass": all(c["applicable_invalid"] for c in chk.values() if not c.get("descriptive")),
            # with the proposed twin repair (peak > |trough|) in place of the v11 twin filter
            "pass_repaired": all(not any((runs[k].get("twin", {}).get("valid_sym", runs[k]["valid"]["twin"]) if s == "twin"
                                          else runs[k]["valid"][s])
                                         for s in APPLICABLE[k]) for k in chk if not chk[k].get("descriptive")),
            "note": "Probe ignores twins and twin divergence ignores labels, so shuffled labels cannot move the twin "
                    "filter and swapped twins cannot move the probe filter; pass = every filter the tamper can move fails."}


def tamper(ad, d, analyze, seed=0):
    """analyze(**overrides) -> analyze_task result on the same items. Runs as is, shuffled labels, swapped twins."""
    runs = {"untampered": run_summary(analyze())}
    for k, over in tamper_variants(d, seed).items():
        runs[k] = run_summary(analyze(**over))
    return {"runs": runs, **tamper_report(runs)}


# ---------------------------------------------------------------- toy
def toy_data(k=4, B=8, n=600, n_tr=400, seed=1):
    from surgery.atlas.p0_toy import ToyAdapter, make_toy_items
    ad = ToyAdapter(B=B, k=k)
    ids, y, an, y_an, sf = make_toy_items(n, seed=seed)
    y = np.asarray(y)
    groups = np.arange(n) % 10
    d = {"ids_tr": ids[:n_tr], "y_tr": y[:n_tr], "g_tr": groups[:n_tr], "ids_te": ids[n_tr:], "y_te": y[n_tr:],
         "ids_an": an[n_tr:], "y_an": np.asarray(y_an[n_tr:]), "ids_sf": sf[n_tr:], "opt_ids": list(range(4))}
    return ad, d


def analyze_built(ad, d, n_perm=20, bow_acc=None, **over):
    from surgery.atlas.p0_harness import analyze_task
    b = {**d, **over}
    return analyze_task(ad, b["ids_tr"], b["y_tr"], b["g_tr"], b["ids_te"], b["y_te"], b["ids_an"], b["y_an"],
                        b["ids_sf"], b["opt_ids"], n_perm=n_perm, n_flip=2000, n_draw=500, bow_acc=bow_acc)


def toy_run(k=4, B=8, n_perm=20):
    ad, d = toy_data(k=k, B=B)
    r = analyze_built(ad, d, n_perm=n_perm)
    on, ar = r["onset"]["use"], r["onset"]["arrival"]
    ok = bool(on is not None and ar is not None and k - 1 < on["layers"] <= k + 1e-9
              and ar["layers"] >= on["layers"] - 0.05)
    return {"k": k, "B": B, "use_onset_layers": None if on is None else on["layers"],
            "arrival_onset_layers": None if ar is None else ar["layers"], "probe_valid": r["valid"]["probe"],
            "probe_p_binom_bonf": r["probe"]["p_binom_bonf"], "n_pairs": r["n_pairs_qualifying"], "pass": ok}


def toy_check():
    runs = [toy_run(4, 8), toy_run(2, 8)]
    return {"runs": runs, "pass": all(r["pass"] for r in runs)}


def toy_tamper(n_perm=20):
    ad, d = toy_data()
    return tamper(ad, d, lambda **over: analyze_built(ad, d, n_perm=n_perm, **over))
