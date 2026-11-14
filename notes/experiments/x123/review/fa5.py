"""Review of x123: numerical gradients and quadrature, inconsistencies outside group B.

Question: how often does the check warn when the gradient is a finite difference of
log_prob, or when log_prob is a marginal likelihood by quadrature and the gradient is
that of the exact integral?

Design: seeds 500000 to 500199 (200 seeds), 500000 to 500049 for the logistic regression
and the quadratures; functions per point (vectorize=False). The numerical gradients are
scipy.optimize.approx_fprime with its default forward step, and central differences with
steps 1e-6 and 1e-4. The marginal is the integral over z of a Student t3 density of
y_i - x_i - z against exp(-z^2 / 2), by 40 Gauss-Hermite nodes, by scipy.integrate.quad
with its default tolerances, and by quad with tolerances of 1e-3; its gradient is always
computed by quad at tolerances of 1e-13.

Run: PYTHONPATH=src:notes/experiments/x123 python notes/experiments/x123/review/fa5.py > fa5.out

Result: see fa5.out and REVIEW.md, objection 4. With forward differences (N) warns on
198, 194 and 200 of 200 on N(0, I), Student t3 and the banana, and on none of 50 on the
logistic regression; with central differences of step 1e-6 on none, and of step 1e-4 on
39 of 200 on the banana. It warns on 50 of 50 with the Gauss-Hermite marginal, on none
with quad's defaults and on 41 of 50 with the loose quad. (H) warns on none.
"""

import families as F
import numpy as np
from probe import run
from scipy import integrate, optimize

from telescopic.initial import Normal

S = range(500_000, 500_200)
S50 = range(500_000, 500_050)


def fd(case, kind):
    lp, _, init = case
    f1 = lambda x: float(lp(x[None])[0])
    if kind == "forward":
        gg = lambda x: optimize.approx_fprime(x, f1)  # scipy default step 1.49e-8
    elif kind == "central 1e-6":
        gg = lambda x: np.array(
            [(f1(x + 1e-6 * e) - f1(x - 1e-6 * e)) / 2e-6 for e in np.eye(len(x))]
        )
    else:
        gg = lambda x: np.array(
            [(f1(x + 1e-4 * e) - f1(x - 1e-4 * e)) / 2e-4 for e in np.eye(len(x))]
        )
    return f1, gg, init


for name, case in (
    ("N(0,I) d=5", F.gaussian()),
    ("Student t3 d=5 start sd 2", F.student(wide=2.0)),
    ("banana", F.example("banana")),
    ("logistic n=1e4 d=5", F.logistic()),
):
    for kind in ("forward", "central 1e-6", "central 1e-4"):
        run(
            f"{name}: gradient by {kind} differences of log_prob",
            fd(case, kind),
            S50 if "logistic" in name else S,
            vectorize=False,
        )
# marginal likelihood by quadrature, gradient of the exact integral
y = np.array([0.7, -1.1, 0.4])


def marg(method):
    # p(y_i | x_i) is the integral of Student-t3(y_i - x_i - z) exp(-z^2 / 2) over z, up to a
    # constant; the gradient is that of the exact integral, by quad at tight tolerances
    t3 = lambda r: (1 + r * r / 3) ** -2.0
    dt3 = lambda r: -2 * (1 + r * r / 3) ** -3.0 * (2 * r / 3)
    nodes, weights = np.polynomial.hermite_e.hermegauss(40)

    def one(xi, yi):
        if method == "gh40":
            return np.sum(weights * t3(yi - xi - nodes))
        if method == "quad":
            return integrate.quad(
                lambda z: t3(yi - xi - z) * np.exp(-0.5 * z * z), -np.inf, np.inf
            )[0]
        return integrate.quad(
            lambda z: t3(yi - xi - z) * np.exp(-0.5 * z * z), -12, 12, epsabs=1e-3, epsrel=1e-3
        )[0]

    def done(xi, yi):  # derivative w.r.t. x_i of the exact integral, tight tolerance
        num = integrate.quad(
            lambda z: -dt3(yi - xi - z) * np.exp(-0.5 * z * z),
            -np.inf,
            np.inf,
            epsabs=1e-13,
            epsrel=1e-13,
        )[0]
        den = integrate.quad(
            lambda z: t3(yi - xi - z) * np.exp(-0.5 * z * z),
            -np.inf,
            np.inf,
            epsabs=1e-13,
            epsrel=1e-13,
        )[0]
        return num / den

    lp = lambda x: float(
        sum(np.log(one(xi, yi)) for xi, yi in zip(x, y, strict=True)) - 0.5 * x @ x
    )
    g = lambda x: np.array([done(xi, yi) for xi, yi in zip(x, y, strict=True)]) - x
    return lp, g, Normal(np.zeros(3), np.ones(3))


run(
    "marginal by 40-node Gauss-Hermite, gradient of the exact integral",
    marg("gh40"),
    S50,
    vectorize=False,
)
run(
    "marginal by scipy quad (default tolerances), gradient of the exact integral",
    marg("quad"),
    S50,
    vectorize=False,
)
run(
    "marginal by scipy quad (epsrel 1e-3), gradient of the exact integral",
    marg("loose"),
    S50,
    vectorize=False,
)
