"""Exploratory: the detection formulas of DESIGN.md sections 5 and 9 against the prototype (seeds 213600..213849).

Question: do the closed forms for the probability that one line warns match what prototype.check does?
Target N(0, I_d); the initial distribution is N(0, diag(s^2)), so kappa_i = s_i (the start's spread
in units of the target's scale). xi = x / s and zeta = (a - b) / (sqrt(2) s) are standard normal.

  (N) warns on a line iff  h |e . u| > SAFETY ULPS eps L,   L = max |f| on the line ~ sum_i kappa_i^2 xi_i^2 / 2
  (H) warns on a line iff  |e . u| > PHI = 2 sqrt(2) delta*   (and the (N) condition, which then holds)

  sign of coordinate 0:   e . u = 2 sqrt(2) kappa_0^2 xi_0 zeta_0
  factor (1 + r):         e . u = r sqrt(2) sum_i kappa_i^2 xi_i zeta_i
  offset c in every coordinate: e . u = c sqrt(2) sum_i kappa_i zeta_i

Model: Monte Carlo of these inequalities (2e6 lines, generator seed 213600); for the sign of one
coordinate with L fixed at its mean, also the closed form 1 - F(t), F(t) = P(|xi zeta| < t) =
(2 / pi) int_0^t K_0(w) dw.
Measurement: prototype.check with P = 8 on seeds 213600..213849 (250 checks, 2,000 lines), the same seeds
for every row; "line" is the fraction of lines that warn, "check" the fraction of checks.
Result: see formulas.out.
"""

import warnings

import numpy as np
import prototype
from scipy import integrate, special

EPS = np.finfo(float).eps
H0 = EPS ** (1 / 3)
C_N = prototype.SAFETY * prototype.ULPS * EPS / H0  # (N): |e . u| must exceed C_N * L
SEEDS = range(213600, 213850)
rng = np.random.default_rng(213600)
M = 2_000_000


def F(t):
    return 2 / np.pi * integrate.quad(lambda w: special.k0(w), 0, t)[0]


def model(kind, d, k0, k, size, floor):
    # coordinate 0 has kappa k0, the other d - 1 have kappa k
    xi0, zeta0 = rng.standard_normal(M), rng.standard_normal(M)
    chi = rng.chisquare(d - 1, M) if d > 1 else np.zeros(M)
    L = 0.5 * (k0**2 * xi0**2 + k**2 * chi)
    if kind == "sign":
        eu = 2 * np.sqrt(2) * k0**2 * xi0 * zeta0
    elif kind == "factor":
        eu = (
            size
            * np.sqrt(2)
            * (k0**2 * xi0 * zeta0 + k**2 * np.sqrt(chi) * rng.standard_normal(M))
        )
    else:  # offset
        eu = size * np.sqrt(2) * np.sqrt(k0**2 + (d - 1) * k**2) * zeta0
    bar = np.maximum(C_N * L, floor)
    return float(np.mean(np.abs(eu) > bar))


def case(kind, d, k0, k, size):
    sd = np.full(d, float(k))
    sd[0] = k0

    def init(generator):
        return sd * generator.standard_normal(d)

    def log_prob(x):
        return -0.5 * np.sum(x * x, axis=-1)

    def grad(x):
        if kind == "sign":
            out = -x.copy()
            out[..., 0] = x[..., 0]
            return out
        if kind == "factor":
            return -(1 + size) * x
        return -x + size

    return log_prob, grad, init


def measure(kind, d, k0, k, size, floor):
    c = case(kind, d, k0, k, size)
    lines = warned_lines = checks = 0
    for seed in SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = prototype.check(*c, seed, floor=floor)
        lines += len(v.detail)
        warned_lines += sum(r["warn"] for r in v.detail)
        checks += v.warned
    return warned_lines / lines, lines, checks / len(SEEDS)


ROWS = [
    # design, kind, d, kappa_0, kappa of the others, size
    ("N", "sign", 100, 0.01, 2.0, None),
    ("N", "sign", 1000, 0.01, 2.0, None),
    ("N", "sign", 10_000, 0.01, 2.0, None),
    ("N", "sign", 1000, 0.003, 2.0, None),
    ("N", "sign", 5, 1e-3, 2.0, None),
    ("N", "sign", 5, 1e-4, 2.0, None),
    ("N", "sign", 5, 1.0, 1e4, None),
    ("N", "sign", 1000, 2.0, 2.0, None),
    ("N", "sign", 10_000, 2.0, 2.0, None),
    ("N", "factor", 1000, 2.0, 2.0, 1e-7),
    ("N", "factor", 1000, 2.0, 2.0, 3e-7),
    ("N", "factor", 1000, 2.0, 2.0, 1e-6),
    ("N", "factor", 10_000, 2.0, 2.0, 1e-6),
    ("N", "factor", 1000, 100.0, 100.0, 3e-7),
    ("N", "offset", 5, 1000.0, 1000.0, 2.0),
    ("H", "sign", 5, 1.0, 1.0, None),
    ("H", "sign", 1000, 1.0, 1.0, None),
    ("H", "sign", 10_000, 1.0, 1.0, None),
    ("H", "sign", 1000, 2.0, 2.0, None),
    ("H", "sign", 1000, 0.5, 2.0, None),
    ("H", "sign", 5, 0.01, 2.0, None),
    ("H", "factor", 1000, 2.0, 2.0, 1e-3),
    ("H", "factor", 1000, 2.0, 2.0, 3e-3),
    ("H", "factor", 1000, 2.0, 2.0, 1e-2),
    ("H", "factor", 1000, 1.0, 1.0, 1e-2),
    ("H", "offset", 5, 1000.0, 1000.0, 2.0),
    ("H", "offset", 5, 1.0, 1.0, 0.1),
]

if __name__ == "__main__":
    print(
        f"C_N = SAFETY ULPS eps^(2/3) = {C_N:.3e}; PHI = {prototype.PHI:.4f}; P = {prototype.POINTS}; seeds {SEEDS.start}..{SEEDS.stop - 1}"
    )
    print(
        f"{'design kind':<12} {'d':>6} {'kappa_0':>8} {'kappa':>7} {'size':>7} | {'model line':>10} {'closed':>7} | {'meas line':>9} {'+-':>6} | {'model check':>11} {'meas check':>10}"
    )
    for design, kind, d, k0, k, size in ROWS:
        floor = prototype.PHI if design == "H" else 0.0
        p = model(kind, d, k0, k, size, floor)
        closed = ""
        if kind == "sign":
            mean_L = 0.5 * (k0**2 + k**2 * (d - 1))
            closed = f"{1 - F(max(C_N * mean_L, floor) / (2 * np.sqrt(2) * k0**2)):.4f}"
        m, n, mc = measure(kind, d, k0, k, size, prototype.PHI if design == "H" else None)
        se = np.sqrt(max(m * (1 - m), 1 / n) / n)
        print(
            f"{design + ' ' + kind:<12} {d:>6} {k0:>8g} {k:>7g} {size if size else 0:>7g} | {p:>10.4f} {closed:>7} | {m:>9.4f} {se:>6.4f} | "
            f"{1 - (1 - p) ** prototype.POINTS:>11.4f} {mc:>10.4f}",
            flush=True,
        )
