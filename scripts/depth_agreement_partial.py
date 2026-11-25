"""POST HOC: partial Spearman of cross-model depth given bag-of-words gap (control 1)."""
import json
import numpy as np
from scipy.stats import rankdata, pearsonr
c = {r['task']: r for r in json.load(open('var/results/atlas3/c1/summary.json'))['rows']}
e = {r['task']: r['L_on_healed'] for r in json.load(open('var/results/atlas3/e1.json'))['1.7B']['tasks']}
s = {r['task']: r for r in json.load(open('var/results/atlas3/d3/summary.json'))['rows']}
T = [t for t in e if t in s and t in c]
def partial(x, y, z):
    x, y, z = (rankdata(v) for v in (x, y, z))
    rx = x - np.polyval(np.polyfit(z, x, 1), z); ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return pearsonr(rx, ry)[0]
def boot(f, idx, n=5000):
    rng = np.random.default_rng(0); out = []
    for _ in range(n):
        i = rng.choice(idx, len(idx))
        try:
            v = f(i)
            if np.isfinite(v): out.append(v)
        except Exception: pass
    return np.percentile(out, [2.5, 97.5]).round(2)
for name, qa, sa in [('healed', lambda t: e[t], lambda t: s[t]['L_on_healed']), ('unmod', lambda t: c[t]['L_on'], lambda t: s[t]['L_on'])]:
    Q = np.array([qa(t) for t in T], float); S = np.array([sa(t) for t in T], float)
    Z = np.array([(c[t]['surface']['gap'] + s[t]['surface']['bow'] - s[t]['surface'].get('chance', c[t]['chance'])) / 2 for t in T], float)
    for sub, idx in [('all', np.arange(len(T))), ('deep', np.array([i for i in range(len(T)) if Q[i] > 1 and S[i] > 1]))]:
        f = lambda i: partial(Q[i], S[i], Z[i])
        print(name, sub, len(idx), 'partial rho|BoW', round(f(idx), 3), boot(f, idx))
