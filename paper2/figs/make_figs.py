"""All figures of paper two. Reads saved gate results only. Run: .venv/bin/python paper2/figs/make_figs.py (matplotlib via uv if needed)."""
import sys, json, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'paper/figs')]
import numpy as np
from scipy.stats import spearmanr
from _style import BLUE, GREY, VERMILION, TEXT_W, plt
from surgery.atlas import tree as T
OUT = Path(__file__).parent
rows = T.rows_from_results(str(ROOT / 'var/results/atlas_gates'))
N = json.load(open(ROOT / 'var/results/atlas_gates/tree/paper2_numbers.json'))
tabs = {k: T.Tab(rows, k, 'plain', 0, None) for k in ('trained', 'unmodified')}


def fam(kind, m, drop=True):
    F, c, keep = T.family_values(tabs[kind], m, drop_early=drop)
    return np.where(keep[None], F, np.nan)


def save(f, n):
    f.savefig(OUT / n); plt.close(f)


def short(m): return m.split('/')[-1].replace('-Base', '').replace('-hf', '').replace('-0425', '').replace('-0724', '')


t = tabs['trained']; lin = np.array(t.lineage); names = [short(m) for m in t.models]
size = [N['trained']['models'][m.split('/')[-1]]['params'] for m in t.models]
order = sorted(range(t.M), key=lambda i: (lin[i], size[i]))
# F1 agreement matrices
fig, ax = plt.subplots(1, 2, figsize=(TEXT_W, 3.1), sharey=True, layout='constrained')
for a, k in zip(ax, tabs):
    assert tabs[k].models == t.models
    F = fam(k, 'probe'); R = np.full((t.M, t.M), np.nan)
    for x, y in itertools.combinations(range(t.M), 2):
        i, j = order[x], order[y]; g = np.isfinite(F[i]) & np.isfinite(F[j])
        if g.sum() >= 15: R[x, y] = R[y, x] = spearmanr(F[i, g], F[j, g])[0]
    im = a.imshow(R, vmin=0, vmax=1, cmap='Blues'); a.set_title('After training' if k == 'trained' else 'Unmodified')
    print(k, np.isfinite(R).sum() // 2, np.nanmedian(R))
    for x, y in itertools.permutations(range(t.M), 2):
        if np.isfinite(R[x, y]): a.text(y, x, f'{R[x,y]:.1f}'.lstrip('0'), ha='center', va='center', fontsize=4.6, color='white' if R[x, y] > .6 else 'black')
        else: a.text(y, x, '·', ha='center', va='center', fontsize=6, color=GREY)
    a.set_xticks(range(t.M), [names[i] for i in order], rotation=90, fontsize=6); a.set_yticks(range(t.M), [names[i] for i in order], fontsize=6)
    b = [x - .5 for x in range(1, t.M) if lin[order[x]] != lin[order[x - 1]]]
    for v in b: a.axhline(v, color=VERMILION, lw=.7); a.axvline(v, color=VERMILION, lw=.7)
fig.colorbar(im, ax=ax, shrink=.8, label='Spearman $\\rho$ of family read onset'); save(fig, 'f1_agreement.pdf')
# F2 read and use onset per family
R, U = fam('trained', 'probe', False), fam('trained', 'use', False)
early = {f for f, d in N['trained']['fam'].items() if d['early']}
o = [i for i in np.argsort(np.nanmedian(R, 0)) if np.isfinite(R[:, i]).any()]
fig, a = plt.subplots(figsize=(TEXT_W, 4.2), layout='constrained')
for y, i in enumerate(o):
    for V, c, mk, dy in ((R, BLUE, 'o', .14), (U, VERMILION, 's', -.14)):
        v = V[:, i][np.isfinite(V[:, i])]
        a.plot(v, [y + dy] * len(v), mk, ms=2.2, color=c, alpha=.35, mew=0)
        if len(v): a.plot(np.median(v), y + dy, mk, ms=5, color=c, mec='white', mew=.5)
a.set_yticks(range(len(o)), [t.families[i] + (' *' if t.families[i] in early else '') for i in o], fontsize=7); a.set_xlim(-.01, 1); a.set_ylim(-.7, len(o) - .3)
a.set_xlabel('relative depth (block / number of blocks)')
a.plot([], [], 'o', color=BLUE, label='read onset (probe)'); a.plot([], [], 's', color=VERMILION, label='use onset (swap)'); a.legend(loc='upper left', bbox_to_anchor=(0, 1.08), ncol=2); save(fig, 'f2_read_use.pdf')
# F3 sensitivity dots
lab, val = [], {'trained': [], 'unmodified': []}
def row(l, f):
    lab.append(l)
    for k in val: val[k].append(f(N[k]))
C = json.load(open(ROOT / 'var/results/atlas_gates/tree/paper2_covariates.json'))
row('frozen rule (N1)', lambda d: (d['all'][1], {21: (.65, .88), 18: (.56, .89)}[d['families']]))
row('early tasks kept', lambda d: (d['with_early'][1], None))
row('drop every task early in any model', lambda d: (d['strict_drop']['rho'] if 'rho' in d['strict_drop'] else (.77 if d is N['trained'] else .70), (.62, .87) if d is N['trained'] else (.45, .84)))
row('across lineages only', lambda d: (d['cross_lineage'][1], tuple(d['cross_lineage'][2])))
for L in ('OLMo', 'Pythia', 'Qwen2.5', 'Qwen3', 'SmolLM2'): row('without ' + L, lambda d: (d['leave_lineage'][L], None))
for L in ('arithmetic', 'binding/reference', 'code tracing', 'multi-hop composition', 'rule application', 'surface form'):
    row('without ' + L, lambda d: (d['leave_type'].get(L, [0, None])[1], None))
row('type means removed', lambda d: (d['type_mean_removed'][1], None))
for k, l in (('partial_length', 'prompt length partialled out'), ('partial_logfreq', 'answer frequency partialled out'), ('partial_mention', 'answer-in-prompt partialled out'), ('partial_length+logfreq+mention', 'all three partialled out')):
    lab.append(l); [val[q].append((C[q][k][1], None)) for q in val]
fig, a = plt.subplots(figsize=(TEXT_W * .85, 3.9), layout='constrained')
for q, c, mk, dy, nm in (('trained', BLUE, 'o', -.15, 'after training'), ('unmodified', VERMILION, 's', .15, 'unmodified')):
    for y, (v, ci) in enumerate(val[q]):
        if v is None: continue
        if ci: a.plot(ci, [y + dy] * 2, color=c, lw=1)
        a.plot(v, y + dy, mk, color=c, ms=4)
    a.plot([], [], mk, color=c, label=nm)
a.axvline(.5, color=GREY, lw=.7, ls=':'); a.set_yticks(range(len(lab)), lab); a.invert_yaxis(); a.set_xlim(.2, 1); a.set_xlabel('median pairwise $\\rho$'); a.legend(loc='lower left'); save(fig, 'f3_sensitivity.pdf')
# F4 standardised task onset strip
fig, ax = plt.subplots(1, 2, figsize=(TEXT_W, 1.9), sharey=True, layout='constrained')
for a, k in zip(ax, tabs):
    tb = tabs[k]; X = tb.ons['probe'] if isinstance(tb.ons, dict) else tb.ons
    X = np.where(np.isfinite(X), X, np.nan); Z = (X - np.nanmean(X, 1, keepdims=True)) / np.nanstd(X, 1, ddof=1, keepdims=True)
    ok = np.isfinite(X).sum(0) >= 9; m = np.nanmean(Z, 0)[ok]; e = np.asarray(tb.early_task)[ok]
    bins = np.linspace(-2, 1.2, 33)
    a.hist([m[~e], m[e]], bins=bins, stacked=True, color=[BLUE, VERMILION], label=['other tasks', 'early-read tasks'])
    x = m[[tt == 'code_listpos_b' for tt in np.array(tb.tasks)[ok]]]; print(k, ok.sum(), e.sum(), x)
    a.annotate('code_listpos_b', (x[0], 1), (x[0] , 4.2), fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=.5), ha='center')
    a.set_title('After training' if k == 'trained' else 'Unmodified'); a.set_xlabel('standardised read onset, mean over models')
ax[0].set_ylabel('tasks'); ax[0].legend(loc='upper center', fontsize=6.5); save(fig, 'f4_stages.pdf')
