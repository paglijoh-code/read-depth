"""Post hoc numbers for paper two (shared depth order). Saved data only; the frozen verdicts are untouched."""
import sys, json, itertools
sys.path.insert(0, '.')
import numpy as np
from scipy.stats import spearmanr
from surgery.atlas import tree as T

rows = T.rows_from_results('var/results/atlas_gates')
out = {}


def pairs(F, min_common=15, mask=None):
    r = {}
    for i, j in itertools.combinations(range(F.shape[0]), 2):
        g = np.isfinite(F[i]) & np.isfinite(F[j])
        if mask is not None:
            g &= mask
        if g.sum() >= min_common:
            r[i, j] = spearmanr(F[i, g], F[j, g])[0]
    return r


def boot(F, n=1000, seed=0, sel=None, mc=15):
    rng = np.random.default_rng(seed)
    v = []
    cols = np.flatnonzero(np.isfinite(F).any(0))
    for _ in range(n):
        c = rng.choice(cols, len(cols))
        G = F[:, c]
        p = pairs(G, mc)
        v.append(np.median([x for k, x in p.items() if sel is None or sel(k)] or [np.nan]))
    return [round(float(x), 2) for x in np.nanpercentile(v, [2.5, 97.5])]


for kind in ('trained', 'unmodified'):
    tab = T.Tab(rows, kind, 'plain', 0, None)
    o = out[kind] = {}
    F, cnt, keep = T.family_values(tab, 'probe', drop_early=True)
    F = np.where(keep[None, :], F, np.nan)
    lin = tab.lineage
    p = pairs(F)
    same = [v for (i, j), v in p.items() if lin[i] == lin[j]]
    diff = [v for (i, j), v in p.items() if lin[i] != lin[j]]
    o['all'] = [len(p), round(float(np.median(list(p.values()))), 2), round(min(p.values()), 2), round(max(p.values()), 2)]
    o['families'] = int(keep.sum())
    o['same_lineage'] = [len(same), round(float(np.median(same)), 2)]
    o['cross_lineage'] = [len(diff), round(float(np.median(diff)), 2), boot(F, sel=lambda k: lin[k[0]] != lin[k[1]])]
    o['share_above_0.5'] = round(float(np.mean([v >= .5 for v in p.values()])), 2)
    o['leave_lineage'] = {L: round(float(np.median([v for (i, j), v in p.items() if L not in (lin[i], lin[j])])), 2) for L in sorted(set(lin))}
    types = np.array(tab.fam_type)
    o['leave_type'] = {}
    for t in sorted(set(types[keep])):
        q = pairs(F, 12, types != t)
        o['leave_type'][t] = [len(q), round(float(np.median(list(q.values()))), 2) if q else None, int((keep & (types != t)).sum())]
    # within type: demean ranks is too coarse, so remove type means from each model's family onsets
    G = F.copy()
    for t in set(types):
        c = types == t
        with np.errstate(all='ignore'):
            G[:, c] -= np.nanmean(G[:, c], 1, keepdims=True)
    multi = np.array([keep[(types == t)].sum() >= 2 for t in types])
    q = pairs(G, 12, multi)
    o['type_mean_removed'] = [len(q), round(float(np.median(list(q.values()))), 2) if q else None, int((keep & multi).sum())]
    # also drop every task flagged early in ANY model (stricter)
    keepon = tab.ons['probe'].copy()
    anyearly = tab.early_cell.any(0)
    tab.ons['probe'][:, anyearly] = np.nan
    F2, c2, k2 = T.family_values(tab, 'probe', drop_early=True)
    F2 = np.where(k2[None, :], F2, np.nan)
    q = pairs(F2, 12)
    o['strict_drop'] = dict(dropped=int(anyearly.sum()), tasks=[t for t, a in zip(tab.tasks, anyearly) if a], families=int(k2.sum()), n_pairs=len(q), rho=round(float(np.median(list(q.values()))), 2), ci=boot(F2, mc=12))
    tab.ons['probe'] = keepon
    Fa, ca, ka = T.family_values(tab, 'probe', drop_early=False)
    q = pairs(np.where(ka[None, :], Fa, np.nan))
    o['with_early'] = [len(q), round(float(np.median(list(q.values()))), 2)]
    # reliability ceiling: half-split family rank agreement within model
    o['types'] = {t: int((types[keep] == t).sum()) for t in set(types[keep])}
    if kind == 'trained':
        U, cu, ku = T.family_values(tab, 'use', drop_early=False)
        Fp = Fa
        for nm, de in (('all', False), ('noearly', True)):
            P, _, _ = T.family_values(tab, 'probe', drop_early=de)
            Uu, _, _ = T.family_values(tab, 'use', drop_early=de)
            rr, sp, su, gap = [], [], [], []
            for m in range(tab.M):
                g = np.isfinite(P[m]) & np.isfinite(Uu[m])
                if g.sum() < 12: continue
                rr.append(spearmanr(P[m, g], Uu[m, g])[0]); sp.append(P[m, g].std(ddof=1)); su.append(Uu[m, g].std(ddof=1))
                gap.append(np.median(Uu[m, g] - P[m, g]))
            o['read_use_' + nm] = dict(models=len(rr), rho=round(float(np.median(rr)), 2), rho_range=[round(min(rr), 2), round(max(rr), 2)], sd_read=round(float(np.median(sp)), 3), sd_use=round(float(np.median(su)), 3), gap=round(float(np.median(gap)), 2), gap_range=[round(min(gap), 2), round(max(gap), 2)])
        u = tab.ons['use'][np.isfinite(tab.ons['use'])]; pr = tab.ons['probe'][np.isfinite(tab.ons['probe'])]
        o['use_q'] = np.round(np.percentile(u, [5, 25, 50, 75, 95]), 2).tolist()
        o['read_q'] = np.round(np.percentile(pr, [5, 25, 50, 75, 95]), 2).tolist()
        o['n_use_cells'] = int(len(u))
    ft = {}
    for i, f in enumerate(tab.families):
        ft[f] = dict(type=str(types[i]), read=float(np.nanmedian(Fa[:, i])), n=int(np.isfinite(Fa[:, i]).sum()), early=bool(not keep[i]))
        if kind == 'trained':
            ft[f]['use'] = float(np.nanmedian(U[:, i]))
    o['fam'] = ft
    o['models'] = {m: dict(lin=lin[i], blocks=int(tab.n_blocks[i]), params=float(tab.n_params[i]), rel=tab.release[i]) for i, m in enumerate(tab.models)}
    o['early_tasks'] = [t for t, e in zip(tab.tasks, tab.early_task) if e]
json.dump(out, open('var/results/atlas_gates/tree/paper2_numbers.json', 'w'), indent=1, default=float)
for k in out:
    print(k)
    for a, b in out[k].items():
        print('  ', a, b if a not in ('fam',) else '...')
