"""M9 atlas: collect var/results/atlas/map_*.json (and baseline.json, if present) into summary.json and table sections.

  uv run python scripts/atlas_report.py

Writes var/results/atlas/summary.json and var/results/atlas/tables.md (paste into reports/atlas_map.md; prose is
written by hand). One row per task and rung. BoW numbers come from var/results/atlas/baseline.json, read tolerantly:
either {task: entry} or {"tasks": {task: entry}}, an entry holding a number or {"point": x} under accuracy /
bow_accuracy / acc, and optional {"rungs": {rung: same}}. Without it the BoW and surface-leaky columns stay empty.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgery.atlas import spec  # noqa: E402

OUT = Path("var/results/atlas")
LEAKY_ABS, LEAKY_GAP = 0.90, 0.05


def _num(x):
    if isinstance(x, dict):
        x = x.get("point", x.get("accuracy"))
    return float(x) if isinstance(x, (int, float)) else None


def _entry_acc(entry):
    if not isinstance(entry, dict):
        return _num(entry)
    for key in ("bow", "accuracy", "bow_accuracy", "acc", "selection_accuracy"):
        if key in entry and _num(entry[key]) is not None:
            return _num(entry[key])
    return None


def load_baseline(path=OUT / "baseline.json"):
    """-> {task: {rung or None: accuracy}}"""
    if not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text())
    data = data.get("tasks", data) if isinstance(data, dict) else {}
    out = {}
    for task, entry in data.items():
        if task not in spec.TASKS:
            continue
        out[task] = {None: _entry_acc(entry)}
        rungs = entry.get("rungs") if isinstance(entry, dict) else None
        for r, e in (rungs or {}).items():
            out[task][r] = _entry_acc(e)
    return out


def surface_leaky(bow, peak):
    """Phase 0 choice 6: BoW at least 90%, or within 5 points of the probe peak. None if either is unknown."""
    if bow is None or peak is None:
        return None
    return bool(bow >= LEAKY_ABS - 1e-9 or peak - bow <= LEAKY_GAP + 1e-9)


def rows_for(m, baseline):
    task = m["task"]
    ts = spec.TASKS[task]
    rows = []
    for rung in [None, *ts.rungs]:
        lens = m["lens"]["all"] if rung is None else m["lens"]["rungs"].get(rung)
        probe = m["probe"]["all"] if rung is None else m["probe"].get(rung)
        if lens is None:
            continue
        bow = baseline.get(task, {}).get(rung)
        peak = probe["peak"] if probe else None
        rows.append({
            "task": task, "number": ts.number, "rung": rung, "kind": ts.kind, "needs": ts.needs,
            "chance": lens["chance"], "base_accuracy": lens["final"], "bow": bow,
            "probe_peak": peak, "probe_peak_layer": probe["peak_layer"] if probe else None,
            "L_on": probe["L_on"] if probe else None, "L_lens": lens["L_lens"],
            "fragile": m["fragile"] if rung is None else None,
            "surface_leaky": surface_leaky(bow, peak),
        })
    return rows


def _f(x, pct=True):
    if x is None:
        return "-"
    return f"{100 * x:.1f}" if pct else str(x)


def table(rows):
    head = ("| # | task | rung | kind | needs | chance % | base acc % | BoW % | probe peak % | peak layer | L_on | L_lens "
            "| fragile blocks | surface-leaky |\n|" + "---|" * 14)
    lines = [head]
    for r in rows:
        fr = "-" if r["fragile"] is None else (", ".join(map(str, r["fragile"])) or "none")
        leaky = "-" if r["surface_leaky"] is None else ("yes" if r["surface_leaky"] else "no")
        lines.append(f"| {r['number']} | {r['task']} | {r['rung'] or 'all'} | {r['kind']} | {r['needs']} | {_f(r['chance'])} "
                     f"| {_f(r['base_accuracy'])} | {_f(r['bow'])} | {_f(r['probe_peak'])} | {_f(r['probe_peak_layer'], False)} "
                     f"| {_f(r['L_on'], False)} | {_f(r['L_lens'], False)} | {fr} | {leaky} |")
    return "\n".join(lines)


def main():
    baseline = load_baseline()
    rows, missing = [], []
    for task in spec.PHASE1_TASKS:
        p = OUT / f"map_{task}.json"
        if not p.exists():
            missing.append(task)
            continue
        rows += rows_for(json.loads(p.read_text()), baseline)
    order = json.loads((OUT / "order.json").read_text()) if (OUT / "order.json").exists() else None
    summary = {"rows": rows, "missing_tasks": missing, "baseline_present": bool(baseline),
               "shared_order": order["shared_order"] if order else None}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = ["## Map table\n", table(rows), ""]
    if order:
        md += ["## Shared cut order (least important first)\n", ", ".join(map(str, order["shared_order"])), "",
               "| block | BI | ppl drop |\n|---|---|---|"]
        md += [f"| {i} | {b:.4f} | {d:+.3f} |" for i, (b, d) in enumerate(zip(order["bi"], order["ppl_drop"]))]
    if missing:
        md += ["", f"Missing maps: {', '.join(missing)}"]
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} rows, missing {missing or 'none'} -> {OUT / 'summary.json'}, {OUT / 'tables.md'}")


if __name__ == "__main__":
    main()
