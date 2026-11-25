"""Planned descriptive readings left out of the first draft: arrival onset (whole last-token state swap) against use and
read onset, and twin divergence against read, trained kind seed 0, 14 fitted models. No model runs."""
import sys, glob, json
sys.path.insert(0, '.')
import numpy as np
from scipy.stats import spearmanr
from surgery.atlas import tree as T
rows = T.rows_from_results('var/results/atlas_gates')
tab = T.Tab(rows, 'trained', 'plain', 0, None)
rel = {}
for f in glob.glob('var/results/atlas_gates/g2/trained/s0/*/*.json'):
    j = json.load(open(f))
    a = (j.get('onset') or {}).get('arrival')
    if j.get('valid', {}).get('arrival') and a: rel[(j['model'].split('/')[-1], j['task'])] = a['rel']
A = np.full((tab.M, tab.T), np.nan)
for i, m in enumerate(tab.models):
    for k, t in enumerate(tab.tasks):
        A[i, k] = rel.get((m.split('/')[-1], t), np.nan)
U, P = tab.ons['use'], tab.ons['probe']
q = lambda x: np.round(np.percentile(x[np.isfinite(x)], [25, 50, 75]), 3).tolist()
def fam(X):
    return np.array([[np.nanmean(X[i, tab.task_fam == f]) if np.isfinite(X[i, tab.task_fam == f]).any() else np.nan
                      for f in range(len(tab.families))] for i in range(tab.M)])
FA, FU, FP, FT = fam(A), fam(U), fam(P), fam(tab.ons['twin'])
def rho(X, Y):
    r = []
    for i in range(tab.M):
        g = np.isfinite(X[i]) & np.isfinite(Y[i])
        if g.sum() >= 12: r.append(spearmanr(X[i, g], Y[i, g])[0])
    return [len(r), round(float(np.median(r)), 2), round(min(r), 2), round(max(r), 2)] if r else None
d = (U - A)[np.isfinite(U) & np.isfinite(A)]
out = dict(n_arrival=int(np.isfinite(A).sum()), arrival_q=q(A), use_q=q(U), n_both=int(len(d)),
           use_minus_arrival_q=np.round(np.percentile(d, [25, 50, 75]), 3).tolist(), share_within_0p05=round(float((abs(d) <= .05).mean()), 2),
           sd_arrival=round(float(np.median(np.nanstd(FA, 1, ddof=1))), 3), sd_use=round(float(np.median(np.nanstd(FU, 1, ddof=1))), 3),
           rho_use_arrival=rho(FU, FA), rho_read_arrival=rho(FP, FA), rho_twin_read=rho(FT, FP), rho_twin_arrival=rho(FT, FA),
           n_probe_cells=int(np.isfinite(P).sum()), n_use_cells=int(np.isfinite(U).sum()))
print(json.dumps(out, indent=1))
json.dump(out, open('var/results/atlas_gates/tree/paper2_arrival.json', 'w'), indent=1)
