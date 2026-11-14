"""Review of x123: wrong gradients that the check may miss, each with its correct twin.

``M`` maps a family's name to ``(right, wrong)``, two cases ``(log_prob, grad, initial)``
with the same log_prob and start; miss_detect.py measures the detection of each, and
harm5.py the harm of those it is asked for. The families are: point starts at a mode and
a coordinate held fixed; a logistic regression started at or near its posterior mode; a
gradient that is right in a ball and wrong outside it, or wrong only where the chains go;
a clipped gradient; a large constant in log_prob; a log_prob and gradient in float32; a
start with every coordinate equal or on a subspace, with an error orthogonal to it; a
hierarchical normal model with three wrong terms and four starts; a start on a boundary;
and two float32 gradients returned as float64, whose harm harm5.py measures.

The order of the entries is part of the record: harm5.py takes the seeds of a family
from its position in ``M``.
"""

import families as F
import numpy as np
from scipy import optimize

from telescopic.initial import Empirical, Normal

lpN = lambda x: -0.5 * np.sum(x * x, axis=-1)
gN = lambda x: -x


def flip0(x):
    g = -x.copy()
    g[..., 0] = -g[..., 0]
    return g


M = {}
pt0 = Empirical(np.zeros((1, 5)))
M["N(0,I) d=5, point start at the mode: wrong sign"] = ((lpN, gN, pt0), (lpN, lambda x: x, pt0))
M["N(0,I) d=5, point start at the mode: sign of coordinate 0"] = (
    (lpN, gN, pt0),
    (lpN, flip0, pt0),
)
M["N(0,I) d=5, point start at the mode: twice"] = ((lpN, gN, pt0), (lpN, lambda x: -2 * x, pt0))
fixed0 = lambda c: lambda generator: np.r_[c, 2.0 * generator.standard_normal(4)]
M["N(0,I) d=5, start sd 2 with coordinate 0 held at 0: sign of coordinate 0"] = (
    (lpN, gN, fixed0(0.0)),
    (lpN, flip0, fixed0(0.0)),
)
M["N(0,I) d=5, start sd 2 with coordinate 0 held at 1: sign of coordinate 0"] = (
    (lpN, gN, fixed0(1.0)),
    (lpN, flip0, fixed0(1.0)),
)
M["N(0,I) d=5, start sd 2 with coordinate 0 held at 1e-3: sign of coordinate 0"] = (
    (lpN, gN, fixed0(1e-3)),
    (lpN, flip0, fixed0(1e-3)),
)
# logistic regression started at its posterior mode (an optimizer's output)
_lg = F.logistic()
_map = optimize.minimize(
    lambda b: -_lg[0](b[None])[0],
    np.zeros(5),
    jac=lambda b: -_lg[1](b[None])[0],
    method="BFGS",
    options={"gtol": 1e-10},
).x
ptm = Empirical(_map[None])
M["logistic n=1e4 d=5, point start at the posterior mode: wrong sign"] = (
    (_lg[0], _lg[1], ptm),
    (_lg[0], lambda b: -_lg[1](b), ptm),
)
M["logistic n=1e4 d=5, point start at the posterior mode: twice"] = (
    (_lg[0], _lg[1], ptm),
    (_lg[0], lambda b: 2 * _lg[1](b), ptm),
)
tight_at_map = Normal(_map, np.full(5, (1e-4) ** 2))
M["logistic n=1e4 d=5, start sd 1e-4 around the mode: wrong sign"] = (
    (_lg[0], _lg[1], tight_at_map),
    (_lg[0], lambda b: -_lg[1](b), tight_at_map),
)
post = Normal(_map, np.full(5, (0.03) ** 2))
M["logistic n=1e4 d=5, start sd 0.03 around the mode (about the posterior's): wrong sign"] = (
    (_lg[0], _lg[1], post),
    (_lg[0], lambda b: -_lg[1](b), post),
)
# right inside a ball, wrong outside; start in the ball
ball = Normal(np.zeros(5), np.full(5, 0.1**2))


def outside(x, R=1.0):
    r = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return np.where(r < R, -x, x)


M["N(0,I) d=5, start sd 0.1: gradient right for |x| < 1, wrong sign outside"] = (
    (lpN, gN, ball),
    (lpN, outside, ball),
)
M["N(0,I) d=5, start sd 0.1: gradient right for |x| < 3, wrong sign outside"] = (
    (lpN, gN, ball),
    (lpN, lambda x: outside(x, 3.0), ball),
)
far = Normal(np.full(5, 5.0), np.full(5, 0.1**2))


def inside(x, R=3.0):
    r = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return np.where(r < R, x, -x)


M["N(0,I) d=5, start N(5, 0.1^2): gradient wrong sign only for |x| < 3"] = (
    (lpN, gN, far),
    (lpN, inside, far),
)


def clip(c):
    def g(x):
        n = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
        return -x * np.minimum(1.0, c / np.maximum(n, 1e-300))

    return g


M["N(0,I) d=5, start sd 0.1: gradient clipped to norm 1"] = (
    (lpN, gN, ball),
    (lpN, clip(1.0), ball),
)
M["N(0,I) d=5, start sd 2: gradient clipped to norm 1"] = (
    (lpN, gN, Normal(np.zeros(5), np.full(5, 4.0))),
    (lpN, clip(1.0), Normal(np.zeros(5), np.full(5, 4.0))),
)
# large constants
for K in (1e7, 1e8, 1e9, 1e10):
    c = F.gaussian(offset=-K)
    M[f"N(0,I) - {K:g} d=5, start sd 2: sign of coordinate 0"] = (c, F.modify(c, F.flip(0)))


# float32 log_prob and gradient (dtype visible), wrong gradients
def all32d(d, wrong=None):
    lp = lambda x: -np.float32(0.5) * np.sum(x.astype(np.float32) ** 2, axis=-1)

    def g(x):
        v = -x.astype(np.float32)
        return v if wrong is None else wrong(v)

    return lp, g, F.DiagonalNormal(np.zeros(d), np.full(d, 2.0))


def f0(v):
    v = v.copy()
    v[..., 0] = -v[..., 0]
    return v


for d in (5, 50, 200, 1000, 10_000):
    M[f"float32 throughout, N(0,I) d={d}, start sd 2: sign of coordinate 0"] = (
        all32d(d),
        all32d(d, f0),
    )
M["float32 throughout, N(0,I) d=1000, start sd 2: signs of coordinates 0..9"] = (
    all32d(1000),
    all32d(1000, lambda v: np.concatenate([-v[..., :10], v[..., 10:]], axis=-1)),
)
M["float32 throughout, N(0,I) d=1000, start sd 2: grad x 1.1"] = (
    all32d(1000),
    all32d(1000, lambda v: np.float32(1.1) * v),
)
# all coordinates of the start equal (x = z 1): error orthogonal to 1
ones5 = lambda generator: np.full(5, 2.0 * generator.standard_normal())


def reflect_perp(x):
    m = np.mean(x, axis=-1, keepdims=True)
    return (x - m) - m  # wrong sign of every component orthogonal to 1


M["N(0,I) d=5, start with all coordinates equal: wrong sign orthogonal to (1,..,1)"] = (
    (lpN, gN, ones5),
    (lpN, reflect_perp, ones5),
)
# start on a 2-dimensional subspace of d = 5; wrong sign in the orthogonal complement
_B = np.linalg.qr(np.random.default_rng(11).standard_normal((5, 5)))[0]
sub = lambda generator: 2.0 * (_B[:, :2] @ generator.standard_normal(2))


def perp_sign(x):
    inside_ = (x @ _B[:, :2]) @ _B[:, :2].T
    return -inside_ + (x - inside_)


M["N(0,I) d=5, start on a 2-dimensional subspace: wrong sign in the other 3 directions"] = (
    (lpN, gN, sub),
    (lpN, perp_sign, sub),
)
# hierarchical normal model: (mu, log tau, theta_1..8)
_y = np.array([28.0, 8, -3, 7, -1, 1, 18, 12])
_s = np.array([15.0, 10, 16, 11, 9, 11, 10, 18])


def hier(variant=None):
    def lp(x):
        mu, lt, th = x[..., 0], x[..., 1], x[..., 2:]
        tau2 = np.exp(2 * lt)
        return (
            -0.5 * np.sum(((_y - th) / _s) ** 2, axis=-1)
            - 0.5 * np.sum((th - mu[..., None]) ** 2, axis=-1) / tau2
            - 8 * lt
            - mu**2 / 50
            + lt
            - np.log1p(tau2 / 25)
        )

    def g(x):
        mu, lt, th = x[..., 0], x[..., 1], x[..., 2:]
        tau2 = np.exp(2 * lt)
        ss = np.sum((th - mu[..., None]) ** 2, axis=-1)
        gmu = np.sum(th - mu[..., None], axis=-1) / tau2 - mu / 25
        glt = ss / tau2 - 8 + 1 - 2 * tau2 / (25 + tau2)
        gth = (_y - th) / _s**2 - (th - mu[..., None]) / tau2[..., None]
        if variant == "jacobian":
            glt = glt - 1
        if variant == "sumsq sign":
            glt = glt - 2 * ss / tau2
        if variant == "pooling sign":
            gth = gth + 2 * (th - mu[..., None]) / tau2[..., None]
        return np.concatenate([gmu[..., None], glt[..., None], gth], axis=-1)

    return lp, g


h_spread = lambda generator: np.r_[
    5 * generator.standard_normal(), generator.standard_normal(), 5 * generator.standard_normal(8)
]


def h_pooled(generator):
    mu, lt = 5 * generator.standard_normal(), generator.standard_normal()
    return np.r_[mu, lt, np.full(8, mu)]


h_tighttau = lambda generator: np.r_[
    5 * generator.standard_normal(),
    1.0 + 1e-9 * generator.standard_normal(),
    5 * generator.standard_normal(8),
]
h_fixedtau = lambda generator: np.r_[
    5 * generator.standard_normal(), 1.0, 5 * generator.standard_normal(8)
]
for vname in ("jacobian", "sumsq sign", "pooling sign"):
    for sname, st in (
        ("spread start", h_spread),
        ("every group effect started at mu", h_pooled),
        ("log tau started within 1e-9 of 1", h_tighttau),
        ("log tau started at exactly 1", h_fixedtau),
    ):
        M[f"hierarchical normal d=10, {sname}: {vname} wrong"] = (
            (*hier(), st),
            (*hier(vname), st),
        )
# boundary start: no line compared
hn = lambda x: np.where(np.all(x >= 0, axis=-1), -0.5 * np.sum(x * x, axis=-1), -np.inf)
bstart = lambda generator: np.r_[0.0, np.abs(generator.standard_normal(2))]
M["half-normal d=3 (x >= 0), start with coordinate 0 on the boundary: wrong sign"] = (
    (hn, gN, bstart),
    (hn, lambda x: x, bstart),
)
# float32 gradient returned as float64 (harmless inconsistency?) for the harm measure
M["N(0,I) d=5, start sd 2: float32 gradient returned as float64"] = (
    F.gaussian(),
    (F.gaussian()[0], lambda x: (-(x.astype(np.float32))).astype(np.float64), F.gaussian()[2]),
)
_rc = F.conditioned(1e4)
M["rotated cond 1e4 d=10, matched: float32 gradient returned as float64"] = (
    _rc,
    (
        _rc[0],
        lambda x: np.asarray(_rc[1](x.astype(np.float32)), dtype=np.float32).astype(np.float64),
        _rc[2],
    ),
)
