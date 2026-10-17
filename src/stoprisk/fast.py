"""Block evaluation of scalar upper envelopes of affine probability maps.

This optional numerical accelerator does not change the statistical method.
All returned numbers use floating-point arithmetic, not interval arithmetic.
"""
import math
from numbers import Integral
import numpy as np


def affine_iterate(intercepts, slopes, horizon, *, decisive=None):
    """Iterate v <- max_j (a_j+b_j*v), starting at zero, in blocks.

    Require a,b >= 0 and a+b <= 1. An active line cannot return after a
    strictly greater slope becomes optimal along the nondecreasing orbit.
    Binary search locates the final valid application in each active block.
    Return (value, [(line_index, number_of_steps), ...]).
    """
    a=np.asarray(intercepts,float); b=np.asarray(slopes,float)
    if (a.ndim!=1 or a.shape!=b.shape or len(a)==0 or
        not isinstance(horizon,Integral) or horizon<0 or
        np.any(a<0) or np.any(b<0) or np.any(b>1) or np.any(a+b>1+1e-12)):
        raise ValueError('Invalid affine probability maps or horizon')
    s=1-b if decisive is None else np.asarray(decisive,float)
    if s.shape!=a.shape or np.any(s<0) or np.any(s>1) or np.any(a>s+1e-12):
        raise ValueError('Invalid decisive probabilities')
    v=0.; remaining=int(horizon);blocks=[]
    while remaining:
        values=v+a-s*v
        best=values.max()
        # Exact ties go to larger slope; avoid a tolerance that changes a map.
        active=np.flatnonzero(values==best)
        j=int(active[np.argmin(s[active])])
        aj,bj=float(a[j]),float(b[j])
        sj=float(s[j])
        if sj==0:
            blocks.append((j,remaining));return v,blocks
        r=aj/sj
        def geometric(m):
            if not m:return v
            if sj==1:return aj
            return v+(r-v)*(-math.expm1(m*math.log1p(-sj)))
        candidates=[(aj-a[k])/(sj-s[k]) for k in range(len(a)) if s[k]<sj]
        crossing=min(candidates,default=math.inf)
        # m applications are valid when each input v_0,...,v_(m-1)
        # stays in this line's active region. A crossing output is permitted.
        low,high=1,remaining
        while low<high:
            mid=(low+high+1)//2
            if geometric(mid-1)<=crossing:
                low=mid
            else:
                high=mid-1
        m=low;v=geometric(m);blocks.append((j,m));remaining-=m
        if len(blocks)>2*len(a)+4:
            raise ArithmeticError('Numerical boundary cycling; use reference iteration')
    return float(v),blocks


def factor_upper_fast(rectangles,horizon):
    z=np.asarray(rectangles,float)
    if z.ndim!=2 or z.shape[1]!=4 or np.any(z<0) or np.any(z>1):
        raise ValueError('Expected valid (actions,4) interval array')
    sl,su,rl,ru=z.T
    if np.any(sl>su) or np.any(rl>ru):raise ValueError('Inverted intervals')
    s=np.stack([sl,su],axis=1).ravel()
    r=np.repeat(ru,2)
    return affine_iterate(s*r,1-s,horizon,decisive=s)
