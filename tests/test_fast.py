import numpy as np
import pytest
from stoprisk.core import envelope, optimal_risk
from stoprisk.fast import affine_iterate, factor_upper_fast


def test_accelerator_against_unrolled_bellman():
    rng=np.random.default_rng(20261017)
    for k in [1,2,4,8,16]:
        for _ in range(25):
            probs=rng.dirichlet([.3,.3,.3],size=k)
            u,c=probs[:,:2].T
            t=int(rng.integers(1,2000))
            val,blocks=affine_iterate(u,1-u-c,t)
            assert val==pytest.approx(optimal_risk(u,c,t),abs=2e-12)
            assert sum(m for _,m in blocks)==t
            slopes=[1-u[j]-c[j] for j,_ in blocks]
            assert all(y>=x-1e-13 for x,y in zip(slopes,slopes[1:]))
            assert len(blocks)<=k


def test_factor_accelerator_and_large_horizon():
    z=np.array([[.001,.01,0,.1],[.03,.1,.2,.6]])
    for t in [0,1,2,10,100,1000,10000]:
        val,blocks=factor_upper_fast(z,t)
        assert val==pytest.approx(envelope(z,t)[1],abs=2e-12)
        assert len(blocks)<=2*len(z)
    val,blocks=factor_upper_fast(z,10**12)
    assert val==pytest.approx(.6,abs=2e-12)
    assert len(blocks)<=4


def test_degenerate_maps():
    assert affine_iterate([0],[1],100)==(0,[(0,100)])
    assert affine_iterate([.2,.1],[0,0],100)[0]==.2
    assert affine_iterate([0,0],[0,1],100)[0]==0
    assert affine_iterate([.5,.25],[.5,.75],100)[0]>1-1e-12


def test_tiny_decisive_probability_against_high_precision():
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec=60
        truth=float(Decimal('.5')*(1-(1-Decimal('1e-12'))**(10**12)))
    z=np.array([[1e-12,1e-12,.5,.5]])
    value,_=factor_upper_fast(z,10**12)
    assert value==pytest.approx(truth,abs=1e-14)
