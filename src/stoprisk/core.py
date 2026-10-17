"""All outcomes are mutually exclusive: uncaught harm, catch, continue.

The confidence claim requires independent identically distributed outcomes
within each fixed action, fixed sample sizes, and a prespecified finite menu.
Deployment choices may depend on past history, but must precede the next draw.
This is not a bound over untested attacks or context-dependent hazards.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class Counts:
    n: int
    harm: int
    caught: int

    def __post_init__(self):
        if any(not isinstance(x, Integral) or isinstance(x, bool)
               for x in (self.n, self.harm, self.caught)):
            raise ValueError('Counts must be integers')
        if self.n < 1 or min(self.harm, self.caught) < 0 or self.harm + self.caught > self.n:
            raise ValueError('Require n >= 1 and 0 <= harm+caught <= n')


def cp(x, n, tail):
    """Clopper-Pearson limits, with error <= tail in EACH tail.

    n=0 returns [0,1], including the conditional zero-decisive case.
    """
    if not 0 < tail < .5:
        raise ValueError('tail must lie in (0,.5)')
    x, n = np.broadcast_arrays(np.asarray(x), np.asarray(n))
    if np.any(x < 0) or np.any(x > n) or np.any(n < 0):
        raise ValueError('Invalid binomial counts')
    lo = np.where(x == 0, 0., beta.ppf(tail, x, n-x+1))
    hi = np.where(x == n, 1., beta.isf(tail, x+1, n-x))
    return np.asarray(lo), np.asarray(hi)


def confidence_rectangles(counts, delta=.05, family_size=None, method='factor'):
    """Return per-action (lower1, upper1, lower2, upper2) rectangles.

    factor: decisive probability s=u+c, harm share r=u/(u+c).
    hazard: separate harm probability u and catch probability c.
    family_size may exceed len(counts) to cover a larger reported family.
    Dependence across actions is allowed; within-action IID is required.
    """
    if not counts or not 0 < delta < 1:
        raise ValueError('Need nonempty counts and delta in (0,1)')
    k = len(counts) if family_size is None else family_size
    if not isinstance(k, Integral) or k < len(counts):
        raise ValueError('family_size must cover every supplied action')
    n = np.array([x.n for x in counts])
    u = np.array([x.harm for x in counts])
    c = np.array([x.caught for x in counts])
    tail = delta / (4*k)
    if method == 'factor':
        sl, su = cp(u+c, n, tail)
        rl, ru = cp(u, u+c, tail)
        return np.stack([sl, su, rl, ru], axis=-1)
    if method == 'hazard':
        ul, uu = cp(u, n, tail)
        cl, cu = cp(c, n, tail)
        return np.stack([ul, uu, cl, cu], axis=-1)
    raise ValueError('Unknown method')


def fixed_risk(u, c, horizon):
    """Known-kernel harm-before-catch probability for a fixed action."""
    _horizon(horizon)
    u, c = np.broadcast_arrays(np.asarray(u, float), np.asarray(c, float))
    if np.any(u < 0) or np.any(c < 0) or np.any(u+c > 1+1e-12):
        raise ValueError('Invalid probabilities')
    s = np.minimum(u+c, 1.)
    r = np.divide(u, s, out=np.zeros_like(s), where=s > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        event = -np.expm1(horizon*np.log1p(-s)) if horizon else np.zeros_like(s)
    return r*event


def _horizon(horizon):
    if not isinstance(horizon, Integral) or isinstance(horizon, bool) or horizon < 0:
        raise ValueError('horizon must be a nonnegative integer')


def optimal_risk(u, c, horizon, return_policy=False):
    """Exact Bellman value over history-adaptive selection from a fixed menu.

    Final array dimension is the action dimension; other dimensions batch.
    Policy[t-1] chooses the action with t opportunities remaining.
    """
    _horizon(horizon)
    u, c = np.broadcast_arrays(np.asarray(u, float), np.asarray(c, float))
    if u.ndim < 1 or not u.shape[-1] or np.any(u < 0) or np.any(c < 0) or np.any(u+c > 1+1e-12):
        raise ValueError('Invalid nonempty probability menu')
    value = np.zeros(u.shape[:-1]); policies = []
    for _ in range(horizon):
        q = u + np.maximum(0., 1-u-c)*value[..., None]
        if return_policy:
            policies.append(np.argmax(q, axis=-1))
        value = np.max(q, axis=-1)
    return (value, np.asarray(policies)) if return_policy else value


def envelope(rectangles, horizon, method='factor'):
    """Lower/upper Bellman bounds; validity is simultaneous at ALL horizons.

    Rectangles have shape (..., actions, 4). Rectangular uncertainty may
    reselect parameters each period. The resulting bounds are conservative
    for stationary unknown kernels; no stationary sharpness is claimed.
    """
    _horizon(horizon)
    z = np.asarray(rectangles, float)
    if z.ndim < 2 or z.shape[-1] != 4 or not z.shape[-2]:
        raise ValueError('Expected (..., actions, 4) rectangles')
    a, b, d, e = np.moveaxis(z, -1, 0)
    if np.any(z < 0) or np.any(z > 1) or np.any(a > b) or np.any(d > e):
        raise ValueError('Invalid interval endpoints')
    lower = np.zeros(z.shape[:-2]); upper = lower.copy()
    for _ in range(horizon):
        lv, uv = lower[..., None], upper[..., None]
        if method == 'factor':
            lq = lv + np.where(d >= lv, a, b)*(d-lv)
            uq = uv + np.where(e >= uv, b, a)*(e-uv)
        elif method == 'hazard':
            if np.any(a+d > 1+1e-12):
                raise ValueError('Empty hazard simplex intersection')
            # Linear objective u*(1-v) - c*v + v on a box intersect simplex.
            lq = a + (1-a-np.minimum(e, 1-a))*lv
            uq = np.minimum(b, 1-d) + (1-np.minimum(b, 1-d)-d)*uv
        else:
            raise ValueError('Unknown method')
        lower = np.clip(np.max(lq, axis=-1), 0., 1.)
        upper = np.clip(np.max(uq, axis=-1), 0., 1.)
    return lower, upper
