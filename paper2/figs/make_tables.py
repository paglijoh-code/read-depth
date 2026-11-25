"""Writes tab_families.tex, tab_models.tex and tab_sens.tex from saved results."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; P = ROOT / 'paper2'; D = ROOT / 'var/results/atlas_gates/tree'
N = json.load(open(D / 'paper2_numbers.json')); E = json.load(open(D / 'paper2_examples.json'))
esc = lambda s: s.replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('#', r'\#').replace('{', r'\{').replace('}', r'\}').replace('\n', r' $\hookleftarrow$ ').replace('&', r'\&').replace('>>>', r'{>}{>}{>}').replace('–', '--')
o = [r'\begin{tabular}{llccc@{\hspace{1.2em}}p{0.42\linewidth}}', r'\toprule', r'Family & Type & Read, unm. & Read, tr. & Use, tr. & Example (first task): prompt $\rightarrow$ answer \\', r'\midrule']
U, T = N['unmodified']['fam'], N['trained']['fam']
for f, d in sorted(T.items(), key=lambda x: (x[1]['type'], x[1]['read'])):
    e = E[f + '_a']; pr = e['prompt']
    s = lambda q: r'$^{*}$' if q['early'] else ''
    o.append(f"\\task{{{esc(f)}}} & {d['type']} & {U[f]['read']:.2f}{s(U[f])} & {d['read']:.2f}{s(d)} & {d['use']:.2f} & {{\\scriptsize {esc(pr)} $\\rightarrow$ \\emph{{{esc(e['answer'])}}}}} \\\\")
o += [r'\bottomrule', r'\end{tabular}']; (P / 'tab_families.tex').write_text('\n'.join(o) + '\n')
o = [r'\begin{tabular}{llccc}', r'\toprule', r'Model & Lineage & Blocks & Parameters & Hub repo created \\', r'\midrule']
C = {'Qwen3': 'qwen3', 'Qwen2.5': 'qwen25', 'SmolLM2': 'smollm2', 'Pythia': 'biderman2023pythia', 'OLMo': 'groeneveld2024olmo,olmo2'}; seen = set()
for m, d in N['trained']['models'].items():
    p = d['params']; l = d['lin'] + ('' if d['lin'] in seen else r'~\citep{' + C[d['lin']] + '}'); seen.add(d['lin'])
    o.append(f"{esc(m)} & {l} & {d['blocks']} & {p/1e9:.2f}B & {d['rel']} \\\\" if p >= 1e9 else f"{esc(m)} & {l} & {d['blocks']} & {p/1e6:.0f}M & {d['rel']} \\\\")
o += [r'\midrule', r'TinyLlama\_v1.1 & held out~\citep{zhang2024tinyllama} & 22 & 1.1B & --- \\', r'\bottomrule', r'\end{tabular}']; (P / 'tab_models.tex').write_text('\n'.join(o) + '\n')
