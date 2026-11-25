import sys;sys.path.insert(0,'.')
import numpy as np
from surgery.atlas import tree as T
def heldout():
    pass
for kind,root in(('trained','var/results/atlas_gates'),('unmodified','var/results/atlas_gates')):
    rows=T.rows_from_results(root); pass
    tab=T.Tab(rows,kind,'plain',0,None); ms=tab.models
    x,n=T.task_level(tab); good=np.isfinite(x); xs=x[good]; fam=tab.task_fam[good]; early=tab.early_task[good]; names=np.array(tab.tasks)[good]
    d=T.n2_decide(xs,fam,early,1000,1000)
    lab=d['_fit']['labels']; lo=min(np.unique(lab),key=lambda c:np.median(xs[lab==c])); m=lab==lo
    print('\n',kind,len(ms),'models',len(xs),'tasks; verdict',d['verdict'],'dbic',round(d['dbic'],1),'early tasks',int(early.sum()))
    print('low cluster',m.sum(),'of which early',int(early[m].sum()),'not early:',[(a,round(b,2)) for a,b in zip(names[m&~early],xs[m&~early])],'early in high:',list(names[~m&early]))
    print('low max',xs[m].max(),'high min',xs[~m].min(), sorted(names[m]))
    for lbl,keep in(('drop early',~early),('drop low cluster',~m)):
        d2=T.n2_decide(xs[keep],fam[keep],early[keep],1000,1000)
        print(lbl,keep.sum(),'k',d2['k'],d2['verdict'],'dbic',round(d2['dbic'],1),'q95',d2['null_q95'],'p',d2['p_null'],'stab',d2['stability'],'sizes',np.unique(d2['_fit']['labels'],return_counts=True)[1])
