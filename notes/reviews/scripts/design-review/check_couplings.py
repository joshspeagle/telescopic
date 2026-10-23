"""Check spec section 5.2 primitives and the met-flag contract (R-KER-3, R-KER-4).

1. Reflection-maximal coupling as written in R-CPL-1: meeting frequency vs 2 Phi(-|delta|/2),
   Y marginal, and the delta = 0 case (NaN direction, RuntimeWarning under warnings-as-errors).
2. Rejection maximal coupling with eta: meeting probability int min(eta p, q), draw-count variance.
3. Coupled RWMH with 'met = identical and both accept' started from equal states:
   how often are returned states equal while met is False (conflict with R-KER-3 and R-KER-4)?
4. Same for IMH ('met = both accept').
5. Gibbs block with rejection coupling at eta < 1 from equal states (faithfulness).
"""

import warnings

import numpy as np
from scipy import stats

rng = np.random.default_rng(7)


# ---------- 1. reflection-maximal, transcribed from R-CPL-1 ----------
def reflection_maximal(mu_x, mu_y, chol, z, u):
    delta = np.linalg.solve(chol, (mu_x - mu_y).T).T
    norm = np.linalg.norm(delta, axis=-1)
    e = delta / norm[:, None]  # as written: unconditional division
    identical = np.log(u) < -np.sum(z * delta, axis=-1) - 0.5 * norm**2
    x = mu_x + z @ chol.T
    zy = z - 2 * np.sum(e * z, axis=-1)[:, None] * e
    y = np.where(identical[:, None], x, mu_y + zy @ chol.T)
    return x, y, identical


d = 10
A = rng.normal(size=(d, d))
Sigma = A @ A.T / d + np.eye(d)
C = np.linalg.cholesky(Sigma)
n = 200_000
for dist in (0.5, 1.5, 3.0):
    dw = rng.normal(size=d)
    dw *= dist / np.linalg.norm(dw)
    mu_x = np.zeros((n, d))
    mu_y = np.tile(C @ dw, (n, 1))
    z = rng.normal(size=(n, d))
    u = rng.random(n)
    x, y, ident = reflection_maximal(mu_x, mu_y, C, z, u)
    p_exact = 2 * stats.norm.cdf(-dist / 2)
    bitwise = np.all(x == y, axis=1)
    p_bin = stats.binomtest(int(bitwise.sum()), n, p_exact).pvalue
    wy = np.linalg.solve(C, (y - mu_y).T).T
    ks = min(stats.kstest(wy[:, j], "norm").pvalue for j in range(d)) * d
    print(
        f"refl-max |delta|={dist}: P(meet) {bitwise.mean():.4f} vs 2Phi(-|d|/2) {p_exact:.4f} "
        f"(binomial p={p_bin:.3f}); flag==bitwise: {np.array_equal(ident, bitwise)}; "
        f"Bonferroni KS p(Y whitened)={min(ks, 1):.3f}"
    )

# delta = 0 rows
with warnings.catch_warnings():
    warnings.simplefilter("error")
    try:
        reflection_maximal(np.zeros((3, d)), np.zeros((3, d)), C, rng.normal(size=(3, d)), rng.random(3))
        print("delta=0: no warning")
    except RuntimeWarning as exc:
        print(f"delta=0 as written raises under warnings-as-errors: RuntimeWarning: {exc}")


# ---------- 2. rejection maximal coupling with eta ----------
def rejection_eta(p, q, eta, gen):
    x = p.rvs(random_state=gen)
    u = gen.random()
    draws = 1
    if np.log(u) <= min(np.log(eta), q.logpdf(x) - p.logpdf(x)):
        return x, x, True, draws
    while True:
        y = q.rvs(random_state=gen)
        u2 = gen.random()
        draws += 1
        if np.log(u2) > np.log(eta) + p.logpdf(y) - q.logpdf(y):
            return x, y, False, draws


p = stats.norm(0, 1)
q = stats.norm(0.3, 1.2)
grid = np.linspace(-12, 12, 200_001)
for eta in (1.0, 0.9):
    c_eta = np.trapezoid(np.minimum(eta * p.pdf(grid), q.pdf(grid)), grid)
    out = [rejection_eta(p, q, eta, rng) for _ in range(40_000)]
    met = np.array([o[2] for o in out])
    nd = np.array([o[3] for o in out])
    ys = np.array([o[1] for o in out])
    print(
        f"rejection eta={eta}: P(meet) {met.mean():.4f} vs int min(eta p, q) {c_eta:.4f}; "
        f"mean draws {nd.mean():.3f}; var draws {nd.var():.3f} vs 2c/(1-c) {2 * c_eta / (1 - c_eta):.3f} "
        f"(bound 2eta/(1-eta) = {2 * eta / (1 - eta) if eta < 1 else float('inf'):.3f}); "
        f"KS p(Y~q)={stats.kstest(ys, q.cdf).pvalue:.3f}"
    )

# p == q (equal states) with eta < 1: faithfulness of a Gibbs block
eta = 0.9
out = [rejection_eta(p, p, eta, rng) for _ in range(20_000)]
split = np.mean([o[0] != o[1] for o in out])
print(f"rejection eta={eta} from p == q: outputs differ in {split:.4f} of calls (1 - eta = {1 - eta:.2f})")


# ---------- 3. coupled RWMH from equal states, met = identical & both accept ----------
def logpi(x):
    return -0.5 * np.sum(x * x, axis=-1)


n = 100_000
d = 5
s = 2.38 / np.sqrt(d)
x = rng.normal(size=(n, d)) * 1.5
y = x.copy()
lx = logpi(x)
z = rng.normal(size=(n, d))
u_coupling = rng.random(n)
v = rng.random(n)
with np.errstate(invalid="ignore", divide="ignore"):
    delta = (x - y) / s
    norm = np.linalg.norm(delta, axis=1)
    ident = np.log(u_coupling) < -np.sum(z * delta, axis=1) - 0.5 * norm**2
px = x + s * z
py = px.copy()  # identical everywhere since delta = 0
ax = np.log(v) < logpi(px) - lx
ay = np.log(v) < logpi(py) - logpi(y)
newx = np.where(ax[:, None], px, x)
newy = np.where(ay[:, None], py, y)
met = ident & ax & ay
equal = np.all(newx == newy, axis=1)
print(
    f"RWMH from equal states: states equal after step {equal.mean():.4f}; met flag true {met.mean():.4f}; "
    f"equal but met False in {np.mean(equal & ~met):.4f} of pairs (both rejected)"
)

# ---------- 4. IMH from equal states, met = both accept ----------
q_imh = stats.multivariate_t(loc=np.zeros(d), shape=np.eye(d), df=5)
xprop = q_imh.rvs(size=n, random_state=rng)
w_cur = logpi(x) - q_imh.logpdf(x)
w_prop = logpi(xprop) - q_imh.logpdf(xprop)
acc = np.log(v) < w_prop - w_cur
print(
    f"IMH from equal states: both reject (equal states, met False) in {np.mean(~acc):.4f} of pairs"
)
