"""Saves the paper-two numbers that were first computed inline: cells under 0.45 accuracy, seed rank correlations,
N5 without early tasks, twin against read. Post hoc; saved data only."""
import sys, json
sys.path.insert(0, '.')
import numpy as np
from scipy.stats import spearmanr
from surgery.atlas import tree as T
rows = T.rows_from_results('var/results/atlas_gates')
out = {}
tr = T.Tab(rows, 'trained', 'plain', 0, None)
t1 = T.Tab(rows, 'trained', 'plain', 1, None)
out['tab_attrs'] = [a for a in dir(tr) if not a.startswith('_')]
for meth in ('probe', 'use'):
    common = [t for t in tr.tasks if t in t1.tasks]
    r = []
    for m in tr.models:
        if m not in t1.models: continue
        a = np.array([tr.ons[meth][tr.models.index(m), tr.tasks.index(t)] for t in common])
        b = np.array([t1.ons[meth][t1.models.index(m), t1.tasks.index(t)] for t in common])
        g = np.isfinite(a) & np.isfinite(b)
        if g.sum() >= 5: r.append(spearmanr(a[g], b[g])[0])
    out['seed_rho_' + meth] = dict(models=len(r), median=round(float(np.median(r)), 3), tasks=len(common))
def fam(tab, X, ok):
    return np.array([[np.nanmean(X[i, ok & (tab.task_fam == f)]) if np.isfinite(X[i, ok & (tab.task_fam == f)]).any() else np.nan
                      for f in range(len(tab.families))] for i in range(tab.M)])
for kind, shot in (('trained', 'plain'), ('unmodified', 'plain'), ('unmodified', next(r['prompt'] for r in rows if r['kind'] == 'unmodified' and r['prompt'] != 'plain'))):
    try: tab = T.Tab(rows, kind, shot, 0, None)
    except Exception as e: out[f'{kind}_{shot}'] = repr(e); continue
    o = out[f'{kind}_{shot}'] = {}
    for lbl, ok in (('all', np.ones(tab.T, bool)), ('no_early', ~tab.early_task)):
        F, G = fam(tab, tab.ons['twin'], ok), fam(tab, tab.ons['probe'], ok)
        for mn in (8, 10, 12):
            r = [spearmanr(F[i, g], G[i, g])[0] for i in range(tab.M) for g in [np.isfinite(F[i]) & np.isfinite(G[i])] if g.sum() >= mn]
            if r: o[f'twin_read_{lbl}_min{mn}'] = [len(r)] + [round(float(x), 2) for x in (np.median(r), min(r), max(r))]
        ti, mi, y, xb = T.n5_points(tab, 'probe'); k = ok[ti]
        b, c, _ = T.n5_fit(ti[k], mi[k], y[k], xb[k], tab.T, tab.M, 300, 0)
        o[f'n5_{lbl}'] = [round(float(b), 2), [round(float(x), 2) for x in c], int(k.sum())]
low = [(m, t) for i, m in enumerate(tr.models) for k, t in enumerate(tr.tasks)
       if np.isfinite(tr.ons['probe'][i, k]) and np.isfinite(tr.acc[i, k]) and tr.acc[i, k] < 0.45]
out['under_0p45_valid_read'] = low
ac = {(r['model'], r['task']): r['trained_acc'] for r in rows if r['kind'] == 'trained' and r['seed'] == 0 and r['prompt'] == 'plain'}
out['learned'] = [sum(v >= .45 for v in ac.values()), len(ac)]
out.pop('tab_attrs', None)
print(json.dumps(out, indent=1))
json.dump(out, open('var/results/atlas_gates/tree/paper2_extra.json', 'w'), indent=1)
