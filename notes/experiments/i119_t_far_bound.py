"""Where the far form of the Student-t log-density stops being right (#119).

Where the squared distance s = |C^-1 (x - mu)|^2 overflows float64, `MultivariateT`
computes log s from the offset x - mu scaled by its largest coordinate m, as
2 log m + log |C^-1 (x - mu) / m|^2 (i119_t_far_log_density.py). Two cases are outside
that form. The rescaled squared norm is at most d divided by the smallest eigenvalue of
the scale matrix, so it overflows too for a scale matrix small enough. And the offset
itself overflows where x and mu are of opposite signs near the largest float. The
question is what the log-density returns in each case, in 2 dimensions with 5 degrees
of freedom, against the exact value log Gamma((nu + d) / 2) - log Gamma(nu / 2)
- (d / 2) log(nu pi) - (1 / 2) log det - ((nu + d) / 2) (log s - log nu), with log s
computed from logarithms and the term log(1 + nu / s) dropped, which is below 1e-300.
The code before the fix is copied below as `before`, as in i119_t_far_log_density.py.

Result (i119_t_far_bound.out, whose first line records the commit): at the point
(1e10, 1e10), where s overflows for each of these scale matrices and the code before the
fix returned -inf, the value is the exact one for 1e-300 I, 1e-307 I and 4e-308 I. For
1e-308 I and 1e-310 I it is -inf with a NumPy overflow warning, where it is -1932.8 and
-1944.3. Where the offset overflows, with mean (-1e308, 0) and x = (1.7e308, 0), the
value is NaN with a NumPy overflow warning, where it is -4967.5. In these three cases
the value is the one the code before the fix returned.

Run: python notes/experiments/i119_t_far_bound.py
"""

import math
import subprocess
import warnings

import numpy as np
from scipy.special import gammaln

from telescopic._linalg import matvec
from telescopic.kernels import MultivariateT

NU, D = 5, 2
CONSTANT = math.lgamma((NU + D) / 2) - math.lgamma(NU / 2) - D / 2 * math.log(NU * math.pi)


def before(q, x):
    d = len(q.mean)
    z = matvec(q._inverse, x - q.mean)
    squared = (z**2).sum(axis=1)
    nu = q.df
    return (
        gammaln((nu + d) / 2)
        - gammaln(nu / 2)
        - 0.5 * (d * math.log(nu * math.pi) + q._log_det)
        - 0.5 * (nu + d) * np.log1p(squared / nu)
    )


def exact(log_det, log_s):
    return CONSTANT - log_det / 2 - (NU + D) / 2 * (log_s - math.log(NU))


def returned(q, x):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = q.log_density(np.array([x]))[0]
    with np.errstate(all="ignore"):
        old = before(q, np.array([x]))[0]
    return (
        f"returned {value:.4f}, NumPy warnings {sorted({str(w.message) for w in caught})}, "
        f"before the fix {old:.4f}"
    )


commit = subprocess.run(
    ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
).stdout.strip()
print(f"commit {commit}")
# The scale matrix c I at the point (1e10, 1e10): s = 2e20 / c, and 2 / c after rescaling.
for c in (1e-300, 1e-307, 4e-308, 1e-308, 1e-310):
    q = MultivariateT(np.zeros(D), c * np.eye(D), NU)
    reference = exact(D * math.log(c), math.log(2) + 20 * math.log(10) - math.log(c))
    values = returned(q, [1e10, 1e10])
    print(f"scale matrix {c:g} I, x = (1e10, 1e10): exact {reference:.4f}, {values}")
# The identity scale matrix with x - mu = (2.7e308, 0), beyond the largest float.
q = MultivariateT([-1e308, 0.0], np.eye(D), NU)
reference = exact(0.0, 2 * (math.log(2.7) + 308 * math.log(10)))
print(f"mean (-1e308, 0), x = (1.7e308, 0): exact {reference:.4f}, {returned(q, [1.7e308, 0.0])}")
