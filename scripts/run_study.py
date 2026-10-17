"""Reproduce all numerical study outputs without network access or model APIs."""
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy.stats import binom, multinomial

from stoprisk.core import Counts, confidence_rectangles, cp, envelope, fixed_risk, optimal_risk

ROOT=Path(__file__).resolve().parents[1]
MENUS={
    'balanced':[(.02,.08),(.01,.02)],
    'rare':[(.0002,.001),(.0001,.0001)],
    'zero-harm':[(0,.01),(0,.1)],
    'switching':[(.05,.5),(.01,.01)],
    'low-catch':[(.01,.0001),(.005,.0002)],
    'asymmetric':[(.001,.3),(.03,.05)],
}


def batch_rect(u,c,n,k=2,method='factor'):
    tail=.05/(4*k)
    if method=='factor':
        a,b=cp(u+c,n,tail);d,e=cp(u,u+c,tail)
    else:
        a,b=cp(u,n,tail);d,e=cp(c,n,tail)
    return np.stack([a,b,d,e],axis=-1)


def simulation():
    rng=np.random.default_rng(1720406); rows=[];reps=10000
    for name,menu in MENUS.items():
        uc=np.array(menu);u,c=uc.T; truth=float(optimal_risk(u,c,100))
        for n in [50,200,1000,5000]:
            draws=np.stack([rng.multinomial(n,[a,b,1-a-b],reps) for a,b in menu],axis=1)
            nu,nc=draws[...,0],draws[...,1]
            plugin=optimal_risk(nu/n,nc/n,100)
            # Preserve all generated count tables, without pretending they are
            # observed LLM outcomes.
            np.savez_compressed(ROOT/'results'/f'synthetic_{name}_{n}.npz',counts=draws,truth=truth)
            for method in ['factor','hazard']:
                lo,hi=envelope(batch_rect(nu,nc,n,method=method),100,method)
                covered=(lo<=truth+1e-12)&(hi>=truth-1e-12)
                rows.append(dict(condition=name,n=n,repetitions=reps,method=method,
                    true_risk=truth,coverage=covered.mean(),coverage_mcse=np.sqrt(covered.mean()*(1-covered.mean())/reps),
                    mean_width=np.mean(hi-lo),width_mcse=np.std(hi-lo,ddof=1)/np.sqrt(reps),
                    plugin_mean=plugin.mean(),plugin_bias=np.mean(plugin-truth),
                    zero_plugin_fraction=np.mean((plugin==0)&(truth>0))))
            print(name,n,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results/simulation.csv',index=False)


def exact_coverage():
    n=8; counts=np.array([(u,c,n-u-c) for u in range(n+1) for c in range(n-u+1)])
    indices=np.array([(a,b) for a in range(len(counts)) for b in range(len(counts))])
    joint=counts[indices];rows=[]
    for name,menu in MENUS.items():
        prob=[multinomial.pmf(counts,n,[u,c,1-u-c]) for u,c in menu]
        mass=prob[0][indices[:,0]]*prob[1][indices[:,1]]
        assert abs(mass.sum()-1)<1e-12
        u,c=np.array(menu).T
        for t in [1,5,20]:
            truth=float(optimal_risk(u,c,t))
            for method in ['factor','hazard']:
                lo,hi=envelope(batch_rect(joint[...,0],joint[...,1],n,method=method),t,method)
                rows.append(dict(condition=name,n_per_action=n,horizon=t,method=method,
                    exact_coverage=float(mass[(lo<=truth+1e-13)&(hi>=truth-1e-13)].sum()),count_tables=len(mass)))
    pd.DataFrame(rows).to_csv(ROOT/'results/exact_coverage.csv',index=False)


def bootstrap_boundary():
    # Enumerate the original multinomial count distribution exactly; the
    # conditional bootstrap quantile is a binomial quantile, with no MC noise.
    n=100;t=100
    counts=np.array([(u,c,n-u-c) for u in range(n+1) for c in range(n-u+1)])
    nu,nc=counts[:,:2].T;m=nu+nc
    r=np.divide(nu,m,out=np.zeros_like(nu,dtype=float),where=m>0)
    rb=np.divide(binom.ppf(.95,m,r),m,out=np.zeros_like(r),where=m>0)
    s=m/n
    bootstrap_upper=rb*(-np.expm1(t*np.log1p(-np.minimum(s,1-1e-16))))
    bootstrap_upper[s==1]=rb[s==1]
    rect=batch_rect(nu[:,None],nc[:,None],n,k=1)
    _,upper=envelope(rect,t)
    rows=[]
    for name,menu in MENUS.items():
        for j,(u,c) in enumerate(menu):
            mass=multinomial.pmf(counts,n,[u,c,1-u-c]);truth=float(fixed_risk(u,c,t))
            for method,bound in [('conditional percentile bootstrap',bootstrap_upper),('factor upper',upper)]:
                rows.append(dict(condition=name,action=j,n=n,true_risk=truth,method=method,
                    exact_upper_coverage=float(mass[bound>=truth-1e-12].sum()),
                    probability_zero_harm=float((1-u)**n)))
    pd.DataFrame(rows).to_csv(ROOT/'results/bootstrap_boundary.csv',index=False)


def public_data():
    path=ROOT/'data/upstream_identity_all_viable.csv';df=pd.read_csv(path)
    eligible=df[(df.n_distinct_honest==1)&(df.honest_flag_rate==0)]
    eligible=eligible.groupby('monitor').filter(lambda g: len(g)==2)
    out=[];counts_by_monitor={}
    for _,row in eligible.iterrows():
        caught=100*row.tpr0; landed_caught=row.n_succ*row.tpr0_succ
        assert abs(caught-round(caught))<1e-8
        assert abs(landed_caught-round(landed_caught))<1e-8
        x=Counts(100,int(row.n_succ)-round(landed_caught),round(caught))
        counts_by_monitor.setdefault(row.monitor,[]).append(x)
        out.append(dict(attacker=row.attacker,monitor=row.monitor,n=x.n,harm=x.harm,caught=x.caught,
                        neutral=x.n-x.harm-x.caught,threshold=0,n_success=int(row.n_succ)))
    pd.DataFrame(out).to_csv(ROOT/'data/derived_counts.csv',index=False)
    results=[]
    for monitor,counts in sorted(counts_by_monitor.items()):
        u=np.array([x.harm/x.n for x in counts]);c=np.array([x.caught/x.n for x in counts])
        # The upstream table filtered the original fixed 24-cell design using
        # observed successes. Correct over ALL original candidates, including
        # the unreported cells, rather than over the eight retained cells.
        rect=confidence_rectangles(counts,family_size=24)
        for t in [1,5,10,100,1000]:
            lo,hi=envelope(rect,t)
            adaptive=float(optimal_risk(u,c,t))
            results.append(dict(monitor=monitor,horizon=t,plugin_adaptive=adaptive,
                plugin_best_fixed=float(np.max(fixed_risk(u,c,t))),lower=float(lo),upper=float(hi),
                confidence_family_cells=24))
    pd.DataFrame(results).to_csv(ROOT/'results/public_analysis.csv',index=False)


def main():
    start=time.monotonic();public_data();exact_coverage();bootstrap_boundary();simulation()
    # Hash only files used/produced by this main run. Figure scripts and the
    # exploratory acceleration outputs have a separate lifecycle.
    inputs=[ROOT/'src/stoprisk/core.py',ROOT/'scripts/run_study.py']
    inputs+=list((ROOT/'data').glob('*'))
    outputs=[ROOT/'results'/name for name in ['simulation.csv','exact_coverage.csv',
             'bootstrap_boundary.csv','public_analysis.csv']]
    outputs+=list((ROOT/'results').glob('synthetic_*.npz'))
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in inputs+outputs if p.is_file()}
    meta=dict(seed=1720406,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
              elapsed_seconds=time.monotonic()-start,files=files)
    (ROOT/'results/run_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Complete',meta['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
