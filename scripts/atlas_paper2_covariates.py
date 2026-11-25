"""Post hoc: does the shared read-depth order survive holding prompt length, answer-token frequency and answer mention fixed?"""
import sys, json, glob, itertools, re
sys.path.insert(0, '.')
import numpy as np
from scipy.stats import spearmanr, rankdata
from surgery.atlas import tree as T
import surgery.atlas.families as F
rows = T.rows_from_results('var/results/atlas_gates')
out = {}
ex = {}
L, MEN = {}, {}
tasks = sorted({r['task'] for r in rows})
for t in tasks:
    it = F.generate(t, seed=0, n_train=8, n_test=120)
    it = [x for x in it if x['split'] == 'test']
    L[t] = float(np.mean([len(x['prompt'].split()) for x in it]))
    opts = [o.strip() for o in F.get_task(t).options]
    MEN[t] = float(np.mean([opts[x['answer']] in re.findall(r'[A-Za-z0-9]+', x['prompt']) for x in it]))
    ex[t] = dict(prompt=it[0]['prompt'], options=opts, answer=opts[it[0]['answer']], at=it[0]['ans_twin']['prompt'], st=it[0]['surf_twin']['prompt'])
json.dump(ex, open('var/results/atlas_gates/tree/paper2_examples.json', 'w'), indent=1)
lf = {}
for p in glob.glob('var/results/atlas_gates/g1/plain/*/*.json'):
    j = json.load(open(p)); v = j.get('answer_logfreq')
    v = v['lm']['mean'] if isinstance(v, dict) and 'lm' in v else None
    if v is not None: lf.setdefault(j['task'], []).append(v)
lf = {t: float(np.median(v)) for t, v in lf.items()}
def resid(y, X):
    Z = np.column_stack([np.ones(len(y))] + X); b = np.linalg.lstsq(Z, y, rcond=None)[0]; return y - Z @ b
for kind in ('trained', 'unmodified'):
    tab = T.Tab(rows, kind, 'plain', 0, None)
    Fv, c, keep = T.family_values(tab, 'probe', drop_early=True)
    fam_tasks = {i: [t for t, f, e in zip(tab.tasks, tab.task_fam, tab.early_task) if f == i and not e] for i in range(len(tab.families))}
    cov = {}
    for nm, d in (('length', L), ('logfreq', lf), ('mention', MEN)):
        cov[nm] = np.array([np.mean([d[t] for t in fam_tasks[i] if t in d]) if any(t in d for t in fam_tasks[i]) else np.nan for i in range(len(tab.families))])
    mean = np.nanmedian(np.where(keep, Fv, np.nan), 0)
    g0 = np.isfinite(mean)
    o = out[kind] = {'corr_with_median_depth': {k: round(float(spearmanr(mean[g0], v[g0])[0]), 2) for k, v in cov.items()}}
    for nm in (('length',), ('logfreq',), ('mention',), ('length', 'logfreq', 'mention')):
        r = []
        for i, j in itertools.combinations(range(tab.M), 2):
            g = np.isfinite(Fv[i]) & np.isfinite(Fv[j]) & keep & np.isfinite(sum(cov[n] for n in nm))
            if g.sum() < 15: continue
            X = [rankdata(cov[n][g]) for n in nm]
            r.append(np.corrcoef(resid(rankdata(Fv[i, g]), X), resid(rankdata(Fv[j, g]), X))[0, 1])
        o['partial_' + '+'.join(nm)] = [len(r), round(float(np.median(r)), 2)]
print(json.dumps(out, indent=1))
json.dump(out, open('var/results/atlas_gates/tree/paper2_covariates.json', 'w'), indent=1)
