"""The numbers of DESIGN.md that follow from its formulas: predicted detection at d = 1,000 and 10,000,
the harm scale, and the size and power of the second experiment's tests. No random seeds except the
model's Monte Carlo (generator seed 213600, as formulas.py).
Result: see predictions.out.
"""

import formulas as M
import numpy as np
import prototype
from scipy import optimize, stats

P = prototype.POINTS
S2 = (2 * stats.norm.ppf(1 - 0.574 / 2)) ** 2

print(
    "1. per-line and per-check (P = 8) detection of the sign of one coordinate of N(0, I_d); kappa_0 the start's"
)
print("   spread in that coordinate over the target's scale, kappa that of the other coordinates")
print(
    f"{'design':<7} {'d':>6} {'kappa_0':>8} {'kappa':>6} | {'line':>8} {'check':>10} {'miss per check':>15}"
)
for design, d, k0, k in [
    ("N", 1000, 1, 1),
    ("N", 10_000, 1, 1),
    ("N", 100_000, 1, 1),
    ("N", 1_000_000, 1, 1),
    ("N", 1000, 0.1, 2),
    ("N", 10_000, 0.1, 2),
    ("N", 1000, 0.01, 2),
    ("N", 10_000, 0.01, 2),
    ("N", 1000, 0.03, 2),
    ("N", 10_000, 0.03, 2),
    ("H", 1000, 1, 1),
    ("H", 10_000, 1, 1),
    ("H", 1000, 2, 2),
    ("H", 10_000, 2, 2),
    ("H", 1000, 0.5, 2),
    ("H", 1000, 0.25, 2),
    ("H", 1000, 0.1, 2),
]:
    p = M.model("sign", d, k0, k, None, prototype.PHI if design == "H" else 0.0)
    if (
        design == "N" and p > 0.99
    ):  # the Monte Carlo cannot resolve the miss: closed form at the mean L
        p = 1 - M.F(M.C_N * 0.5 * (k0**2 + k**2 * (d - 1)) / (2 * np.sqrt(2) * k0**2))
    print(
        f"{design:<7} {d:>6} {k0:>8g} {k:>6g} | {p:>8.5f} {1 - (1 - p) ** P:>10.6f} {(1 - p) ** P:>15.2e}"
    )

print(
    "2. the smallest dense relative error r (grad = (1 + r) true) detected by 99% of checks, start kappa = 1 and 2"
)
for design in ("N", "H"):
    for d in (1000, 10_000):
        for k in (1.0, 2.0):
            floor = prototype.PHI if design == "H" else 0.0

            def excess(log_r, d=d, k=k, floor=floor):
                return 1 - (1 - M.model("factor", d, k, k, 10**log_r, floor)) ** P - 0.99

            r = 10 ** optimize.brentq(excess, -9, 0, xtol=1e-2)
            eps = 1.65 * d ** (-1 / 6)
            print(
                f"   ({design}) d = {d:>6}, kappa = {k:g}: r = {r:.2e}; its delta0 = (eps / 2) r sqrt(d) = {eps / 2 * r * np.sqrt(d):.2e} with eps = 1.65 d^(-1/6) = {eps:.3f}"
            )

print(
    "3. the harm scale: step ratio c(delta0), root of c^6 + (4 delta0^2 / s*^2) c^2 = 1, and 1 / c^2"
)
for dl in (0.125, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0):
    c = optimize.brentq(lambda c, dl=dl: c**6 + 4 * dl**2 / S2 * c**2 - 1, 0, 1)
    print(
        f"   delta0 = {dl:<5g}: 2 Phi(-delta0) = {2 * stats.norm.cdf(-dl):.3f}, TV of the proposals 2 Phi(delta0 / 2) - 1 = {2 * stats.norm.cdf(dl / 2) - 1:.3f}, c = {c:.3f}, 1 / c^2 = {1 / c**2:.2f}"
    )

print("4. size and power of the registered tests")
b = stats.binom
print(
    f"   detection bar, 390 of 400: passes with probability {b.sf(389, 400, 0.95):.4f} at a true rate of 0.95, "
    f"{b.sf(389, 400, 0.96):.4f} at 0.96, {b.sf(389, 400, 0.99):.4f} at 0.99, {b.sf(389, 400, 0.999):.6f} at 0.999"
)
print(
    f"   the first experiment's bar, 180 of 200, at the rate 0.913 the review measured: passes with probability {b.sf(179, 200, 0.913):.3f}"
)
print(
    f"   false alarms per family, at most 2 of 500: passes with probability {b.cdf(2, 500, 0.02):.4f} at a true rate of 2%, "
    f"{b.cdf(2, 500, 0.01):.3f} at 1%, {b.cdf(2, 500, 1e-3):.4f} at 0.1%, {b.cdf(2, 500, 4e-4):.5f} at 0.04%"
)
print(
    f"   false alarms per family, at most 2 of 300: passes with probability {b.cdf(2, 300, 0.03):.4f} at a true rate of 3%, "
    f"{b.cdf(2, 300, 0.02):.3f} at 2%, {b.cdf(2, 300, 1e-3):.4f} at 0.1%, {b.cdf(2, 300, 4e-4):.5f} at 0.04%; "
    f"all of 100 families pass with probability {b.cdf(2, 300, 4e-4) ** 100:.4f} at 0.04% each"
)
for n in (30_000,):
    print(
        f"   false alarms pooled, at most 15 of {n}: passes with probability {b.cdf(15, n, 2e-3):.1e} at a true rate of 1 in 500, "
        f"{b.cdf(15, n, 1e-3):.4f} at 1 in 1,000, {b.cdf(15, n, 2e-4):.4f} at 1 in 5,000"
    )
for p in (0.10, 0.54, 0.69):
    se = np.sqrt(p * (1 - p) / 1000)
    shift = 0.1
    power = stats.norm.sf(
        (0.05 - shift) / np.sqrt((p + shift) * (1 - p - shift) / 1000)
    ) + stats.norm.cdf((-0.05 - shift) / np.sqrt((p + shift) * (1 - p - shift) / 1000))
    print(
        f"   formula check at a predicted rate of {p}: 1,000 checks, standard error {se:.3f}; |measured - predicted| <= 0.05 fails with "
        f"probability {2 * stats.norm.sf(0.05 / se):.1e} if the formula is right and {power:.3f} if the true rate is 0.1 higher"
    )
print(
    f"   silence bar for (H) on a missed family, at most 10 of 400 warn: holds with probability {b.cdf(10, 400, 0.05):.4f} at a true rate of 5%, {b.cdf(10, 400, 0.001):.6f} at 0.1%"
)
