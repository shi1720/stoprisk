from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from stoprisk.core import confidence_rectangles, Counts, envelope, fixed_risk, optimal_risk
from stoprisk.fast import factor_upper_fast

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
BLUE='#205c9b';ORANGE='#ce6e29';GREY='#63717d';GREEN='#22755a'


def save(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.pdf')
    (ROOT/'manuscript/figures').mkdir(exist_ok=True)
    fig.savefig(ROOT/'manuscript/figures'/f'{name}.pdf')
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=200)
    plt.close(fig)


def schematic():
    fig,ax=plt.subplots(figsize=(7.2,2.4));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,3)
    nodes=[(.2,1.1,2.2,.8,'Fixed evaluation menu\ncounts $(U_j,C_j,N_j)$',BLUE),
           (3.15,1.1,2.4,.8,'Joint confidence event\n$[s_j^-,s_j^+]\\times[r_j^-,r_j^+]$',GREEN),
           (6.35,1.1,3.2,.8,'Bellman envelope\nall tested attacks, all horizons',ORANGE)]
    for x,y,w,h,label,col in nodes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=col,alpha=.08,edgecolor=col))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',color=col)
    for x,y in [(2.5,3),(5.7,6.2)]:ax.annotate('',xy=(y,1.5),xytext=(x,1.5),arrowprops={'arrowstyle':'->','color':GREY})
    ax.text(5,.35,'Scope: fixed stationary kernels; immediate effective shutdown; menu chosen before evaluation.',ha='center',fontsize=8,color=GREY)
    save(fig,'pipeline')


def simulation():
    df=pd.read_csv(ROOT/'results/simulation.csv');names=['balanced','rare','zero-harm','switching','low-catch','asymmetric']
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.2),sharex=True,sharey=True)
    for ax,name in zip(axs.flat,names):
        for method,color in [('factor',BLUE),('hazard',ORANGE)]:
            g=df[(df.condition==name)&(df.method==method)]
            ax.plot(g.n,g.mean_width,'o-',ms=3,color=color,label=method)
        ax.set_xscale('log');ax.set_title(name);ax.set_ylim(0,1.03);ax.grid(alpha=.18)
    axs[0,0].legend(frameon=False,fontsize=8)
    fig.supxlabel('Independent evaluation samples per attack');fig.supylabel('Mean 95% interval width');fig.tight_layout()
    save(fig,'interval_widths')


def switching():
    u=np.array([.05,.01]);c=np.array([.5,.01]);ts=np.arange(1,151)
    values,policy=optimal_risk(u,c,int(ts.max()),True)
    adaptive=np.array([optimal_risk(u,c,int(t)) for t in ts])
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.6))
    axs[0].plot(ts,adaptive,label='optimal switching',color=BLUE)
    for j,(color,label) in enumerate([(ORANGE,'fixed A: high decisive rate'),(GREEN,'fixed B: low decisive rate')]):
        axs[0].plot(ts,fixed_risk(u[j],c[j],ts.max())*np.ones_like(ts) if False else [fixed_risk(u[j],c[j],int(t)) for t in ts],label=label,color=color,ls='--')
    axs[0].set(xlabel='Deployment opportunities',ylabel='Harm before catch');axs[0].legend(fontsize=6.5,frameon=False)
    axs[1].step(ts,policy,where='post',color=BLUE)
    axs[1].set(xlabel='Opportunities remaining',ylabel='Optimal next attack',yticks=[0,1],yticklabels=['A','B'],ylim=[-.2,1.2])
    for ax in axs:ax.grid(alpha=.18)
    fig.tight_layout();save(fig,'switching')
    (ROOT/'results/switching_example.json').write_text(json.dumps({'u':u.tolist(),'c':c.tolist(),
      'switch_remaining_horizon':int(np.flatnonzero(policy==1)[0]+1),
      'horizon10_adaptive':float(optimal_risk(u,c,10)),
      'horizon10_best_fixed':float(max(fixed_risk(u,c,10))),
      'horizon100_adaptive':float(values if ts.max()==100 else optimal_risk(u,c,100))},indent=2)+'\n')


def public():
    d=pd.read_csv(ROOT/'results/public_analysis.csv');d=d[d.horizon==100].sort_values('plugin_adaptive')
    names={'opus48':'Opus 4.8','gpt55':'GPT-5.5','geminipro':'Gemini 3.1 Pro','gpt41mini':'GPT-4.1 mini'}
    fig,ax=plt.subplots(figsize=(7.2,2.5));y=np.arange(len(d))
    ax.errorbar(d.plugin_adaptive,y,xerr=[d.plugin_adaptive-d.lower,d.upper-d.plugin_adaptive],fmt='o',color=BLUE,capsize=4)
    ax.set(yticks=y,yticklabels=[names[x] for x in d.monitor],xlabel='Modeled harm-before-catch probability, horizon 100',xlim=[0,1])
    ax.grid(axis='x',alpha=.2);fig.tight_layout();save(fig,'public_intervals')


def acceleration():
    # Exploratory addition after the frozen primary study: compare the optional
    # block accelerator with the reference recursion at feasible horizons.
    rng=np.random.default_rng(42);rows=[];max_error=0.
    for k in [2,8,32]:
        raw=rng.random((k,4));raw[:,:2]=np.sort(raw[:,:2],axis=1);raw[:,2:]=np.sort(raw[:,2:],axis=1)
        for t in [100,10000,10**6,10**12]:
            start=time.perf_counter();value,blocks=factor_upper_fast(raw,t);duration=time.perf_counter()-start
            error=None;ref_time=None
            if t<=10000:
                start=time.perf_counter();ref=float(envelope(raw,t)[1]);ref_time=time.perf_counter()-start
                error=abs(value-ref);max_error=max(max_error,error)
            rows.append(dict(actions=k,horizon=t,value=value,blocks=len(blocks),seconds=duration,
                             reference_seconds=ref_time,absolute_error=error))
    pd.DataFrame(rows).to_csv(ROOT/'results/acceleration.csv',index=False)
    print('Acceleration maximum checked error:',max_error)


if __name__=='__main__':
    schematic();simulation();switching();public();acceleration()
