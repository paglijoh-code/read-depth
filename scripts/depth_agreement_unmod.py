import json
from scipy.stats import spearmanr
import numpy as np
c=json.load(open('var/results/atlas3/c1/summary.json'))['rows']
e=json.load(open('var/results/atlas3/e1.json'))['1.7B']['tasks']
s=json.load(open('var/results/atlas3/d3/summary.json'))['rows']
q={r['task']:r for r in c}
qh={r['task']:r['L_on_healed'] for r in e}
sm={r['task']:r for r in s}
print('bind_compare unmod/healed 1.7B', q.get('bind_compare',{}).get('L_on'), qh.get('bind_compare'))
common=[t for t in qh if t in sm and t in q]
print(len(common), 'common; smol dropped:', [t for t in common if sm[t].get('dropped')], 'qwen dropped', [t for t in common if q[t].get('dropped')])
rows=[(t,q[t]['L_on'],qh[t],sm[t]['L_on'],sm[t]['L_on_healed']) for t in common]
for r in rows: print(r)
def rho(a,b):
    a,b=np.array(a,float),np.array(b,float); return spearmanr(a,b).correlation
def boot(a,b,n=5000):
    rng=np.random.default_rng(0); a,b=np.array(a,float),np.array(b,float); out=[]
    for _ in range(n):
        i=rng.integers(0,len(a),len(a)); 
        if len(set(a[i]))>1 and len(set(b[i]))>1: out.append(spearmanr(a[i],b[i]).correlation)
    return np.percentile(out,[2.5,97.5]).round(2)
QU=[r[1] for r in rows]; QH=[r[2] for r in rows]; SU=[r[3] for r in rows]; SH=[r[4] for r in rows]
print('healed vs healed all', round(rho(QH,SH),3), boot(QH,SH))
print('unmod vs unmod all', round(rho(QU,SU),3), boot(QU,SU))
deep=[i for i,r in enumerate(rows) if r[2]>1 and r[4]>1]
print('healed deep n',len(deep), round(rho([QH[i] for i in deep],[SH[i] for i in deep]),3), boot([QH[i] for i in deep],[SH[i] for i in deep]))
deepu=[i for i,r in enumerate(rows) if r[1]>1 and r[3]>1]
print('unmod deep n',len(deepu), round(rho([QU[i] for i in deepu],[SU[i] for i in deepu]),3), boot([QU[i] for i in deepu],[SU[i] for i in deepu]))
print('within-model unmod vs healed: qwen', round(rho(QU,QH),3), 'smol', round(rho(SU,SH),3))
print('cross unmod-qwen vs healed-smol', round(rho(QU,SH),3), 'healed-qwen vs unmod-smol', round(rho(QH,SU),3))
print('ties at 1: unmod qwen',sum(x==1 for x in QU),'smol',sum(x==1 for x in SU),'healed',sum(x==1 for x in QH),sum(x==1 for x in SH))

# POST HOC, round-4 review (2026-11-20): the 22 tasks SmolLM2 learned (drops the two routing tasks it failed),
# and leave-one-family-out over the paper's seven families (PAPER_FAMILY in scripts/paper_fixpass.py).
import sys; sys.path.insert(0, 'scripts')
from paper_fixpass import PAPER_FAMILY
keep=[i for i,r in enumerate(rows) if not sm[r[0]].get('dropped')]
print('22-task set n', len(keep))
print('  healed vs healed', round(rho([QH[i] for i in keep],[SH[i] for i in keep]),3), boot([QH[i] for i in keep],[SH[i] for i in keep]))
print('  unmod vs unmod', round(rho([QU[i] for i in keep],[SU[i] for i in keep]),3), boot([QU[i] for i in keep],[SU[i] for i in keep]))
kd=[i for i in keep if QH[i]>1 and SH[i]>1]; ku=[i for i in keep if QU[i]>1 and SU[i]>1]
print('  healed depth>1 both n',len(kd), round(rho([QH[i] for i in kd],[SH[i] for i in kd]),3))
print('  unmod depth>1 both n',len(ku), round(rho([QU[i] for i in ku],[SU[i] for i in ku]),3))
fams=sorted({PAPER_FAMILY[r[0]] for r in rows})
for lab,A,B in [('healed',QH,SH),('unmod',QU,SU)]:
    vals=[]
    for f in fams:
        idx=[i for i,r in enumerate(rows) if PAPER_FAMILY[r[0]]!=f]
        vals.append((f,round(rho([A[i] for i in idx],[B[i] for i in idx]),3)))
    print('LOFO',lab,len(fams),'families', vals, 'range', min(v for _,v in vals), max(v for _,v in vals))
