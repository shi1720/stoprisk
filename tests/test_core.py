import itertools
import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.stats import binom
from stoprisk.core import Counts, confidence_rectangles, cp, envelope, fixed_risk, optimal_risk


def test_bellman_against_all_open_loop_sequences():
    # Before absorption there is only one state, so every history collapses
    # to a sequence. Enumerate independently with exact rational arithmetic.
    from fractions import Fraction as F
    menu = [(F(1,20), F(1,2)), (F(1,100), F(1,100))]
    for horizon in range(1,10):
        values=[]
        for sequence in itertools.product(range(2), repeat=horizon):
            survival=F(1); harm=F(0)
            for j in sequence:
                u,c=menu[j]; harm+=survival*u; survival*=1-u-c
            values.append(harm)
        assert optimal_risk([.05,.01],[.5,.01],horizon) == pytest.approx(float(max(values)), abs=1e-14)


def test_fixed_policy_against_direct_sum_and_bellman():
    for u,c in [(0,0),(.001,.01),(.2,.8),(1,0),(0,1)]:
        for t in [0,1,2,10,1000]:
            direct=sum(u*(1-u-c)**i for i in range(t))
            assert fixed_risk(u,c,t)==pytest.approx(direct, abs=1e-13)
            assert optimal_risk([u],[c],t)==pytest.approx(direct, abs=1e-13)


def test_hazard_extrema_against_independent_lp():
    rng=np.random.default_rng(8)
    for _ in range(150):
        u,c,_=rng.dirichlet([1,1,1]); v=rng.random()
        ul,cl=rng.random(2)*[u,c]
        uu,cu=[u,c]+rng.random(2)*(1-np.array([u,c]))
        objective=np.array([1-v,-v])
        low=linprog(objective,A_ub=[[1,1]],b_ub=[1],bounds=[(ul,uu),(cl,cu)],method='highs')
        high=linprog(-objective,A_ub=[[1,1]],b_ub=[1],bounds=[(ul,uu),(cl,cu)],method='highs')
        assert low.success and high.success
        assert ul+(1-ul-min(cu,1-ul))*v == pytest.approx(v+low.fun,abs=1e-12)
        assert min(uu,1-cl)+(1-min(uu,1-cl)-cl)*v == pytest.approx(v-high.fun,abs=1e-12)


def test_factor_extrema_against_all_rectangle_corners():
    rng=np.random.default_rng(19)
    for _ in range(500):
        sl,su=sorted(rng.random(2));rl,ru=sorted(rng.random(2));v=rng.random()
        corners=[s*r+(1-s)*v for s in [sl,su] for r in [rl,ru]]
        assert v+(sl if rl>=v else su)*(rl-v)==pytest.approx(min(corners))
        assert v+(su if ru>=v else sl)*(ru-v)==pytest.approx(max(corners))


def test_cp_exact_error_via_enumeration():
    for n in [1,2,5,10,30,100]:
        x=np.arange(n+1);lo,hi=cp(x,n,.0125)
        for p in np.linspace(0,1,101):
            mass=binom.pmf(x,n,p)
            assert mass[lo>p+1e-14].sum()<=.0125+1e-12
            assert mass[hi<p-1e-14].sum()<=.0125+1e-12


def test_zero_events_and_invalid_inputs():
    z=confidence_rectangles([Counts(100,0,0)])
    assert tuple(z[0,[0,2,3]])==(0,0,1)
    assert envelope(z,100)[1]>.98
    assert envelope(z,0)==(0,0)
    for args in [(0,0,0),(2,2,1),(2,-1,0),(3.5,0,0)]:
        with pytest.raises(ValueError): Counts(*args)
    with pytest.raises(ValueError): optimal_risk([.7],[.5],10)
    with pytest.raises(ValueError): fixed_risk(.1,.1,-1)
    with pytest.raises(ValueError): confidence_rectangles([Counts(3,0,1)],family_size=0)


def test_envelopes_contain_every_stationary_corner_menu():
    z=np.array([[.01,.2,.02,.9],[.05,.8,.1,.6]])
    for choices in itertools.product(range(4),repeat=2):
        corners=[[(s,r) for s in x[:2] for r in x[2:]] for x in z]
        picked=[corners[j][a] for j,a in enumerate(choices)]
        u=[s*r for s,r in picked];c=[s*(1-r) for s,r in picked]
        for t in [1,2,10,50]:
            lo,hi=envelope(z,t)
            assert lo-1e-13<=optimal_risk(u,c,t)<=hi+1e-13


def test_menu_switching_is_strictly_stronger_at_some_horizon():
    u=np.array([.05,.01]);c=np.array([.5,.01]);t=10
    assert optimal_risk(u,c,t) > max(fixed_risk(u,c,t)) + .01
