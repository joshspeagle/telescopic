"""AR(1) benchmark with exact answers (scratch research script).

Chain: X' = rho X + s eps, s = sqrt(1 - rho^2), so pi = N(0, 1).
pi_0 = delta_{x0}.  Law at time t: N(rho^t x0, 1 - rho^(2t)) (exact).
Coupling: reflection-maximal coupling of N(rho x, s^2) and N(rho y, s^2); on the
meeting event Y' is set to X' (bitwise), otherwise Y' - rho y = -(X' - rho x).
Exact quantities: TV(pi_t, pi); one-step meeting prob 2 Phi(-rho|x-y|/(2s));
asymptotic variance of the ergodic average of h(x) = x: (1 + rho)/(1 - rho).
"""
import numpy as np
from scipy import integrate, stats

rho, x0 = 0.9, 10.0
s = np.sqrt(1 - rho**2)
rng = np.random.default_rng(2026)


def tv_normal(m1, s1, m2, s2):
    f = lambda x: 0.5 * abs(stats.norm.pdf(x, m1, s1) - stats.norm.pdf(x, m2, s2))
    lo, hi = min(m1 - 12 * s1, m2 - 12 * s2), max(m1 + 12 * s1, m2 + 12 * s2)
    return integrate.quad(f, lo, hi, limit=400, points=[m1, m2])[0]


def exact_tv(t):
    if t == 0:
        return 1.0  # point mass vs continuous
    return tv_normal(rho**t * x0, np.sqrt(1 - rho ** (2 * t)), 0.0, 1.0)


def coupled_step(x, y):
    """Vectorised reflection-maximal coupling step for arrays x, y."""
    n = x.size
    eps = rng.normal(size=n)
    xn = rho * x + s * eps
    z = rho * (x - y) / s  # standardised mean difference (mean_x - mean_y)/s
    logu = np.log(rng.uniform(size=n))
    # accept 'meet' if U <= phi(eps + z)/phi(eps)
    met = logu <= np.minimum(0.0, -0.5 * (eps + z) ** 2 + 0.5 * eps**2)
    yn = np.where(met, xn, rho * y - s * eps)  # reflection in 1-D: eps' = -eps
    return xn, yn


def lagged_meeting_times(L, n, tmax=100_000):
    x = np.full(n, x0)
    y = np.full(n, x0)
    for _ in range(L):
        x = rho * x + s * rng.normal(size=n)
    tau = np.full(n, -1)
    t = L
    alive = np.arange(n)
    while alive.size and t < tmax:
        xn, yn = coupled_step(x[alive], y[alive])
        x[alive], y[alive] = xn, yn
        t += 1
        met = xn == yn
        tau[alive[met]] = t
        alive = alive[~met]
    return tau


print("=== One-step meeting probability vs exact (x - y fixed) ===")
for dxy in [0.1, 0.5, 1.0, 2.0]:
    n = 200_000
    xn, yn = coupled_step(np.zeros(n), np.full(n, dxy))
    pexact = 2 * stats.norm.cdf(-rho * dxy / (2 * s))
    k = int(np.sum(xn == yn))
    print(f"|x-y|={dxy}: exact={pexact:.5f} empirical={k / n:.5f} binom p={stats.binomtest(k, n, pexact).pvalue:.3f} "
          f"KS(x') p={stats.kstest(xn, stats.norm(0, s).cdf).pvalue:.3f} KS(y') p={stats.kstest(yn, stats.norm(rho * dxy, s).cdf).pvalue:.3f}")

print("\n=== L-lag TV bound (Biswas et al. 2019) vs exact TV ===")
N = 10_000
for L in [1, 10, 50]:
    tau = lagged_meeting_times(L, N)
    rows = []
    for t in [0, 10, 20, 30, 40, 60]:
        v = np.maximum(0, np.ceil((tau - L - t) / L))
        rows.append(f"t={t}: TV={exact_tv(t):.4f} bound={v.mean():.4f}+-{v.std(ddof=1) / np.sqrt(N):.4f}")
    print(f"L={L:2d} mean(tau-L)={np.mean(tau - L):.2f}: " + "; ".join(rows))

print("\n=== Unbiased estimation of E[x] = 0 and E[x^2] = 1: coverage and inefficiency ===")
asvar = (1 + rho) / (1 - rho)  # asymptotic variance of ergodic average of x (Var_pi = 1)
tau_pilot = lagged_meeting_times(1, 2000)
q90, q99 = np.quantile(tau_pilot, [0.9, 0.99]).astype(int)
print(f"pilot meeting times: mean={tau_pilot.mean():.2f} q90={q90} q99={q99} max={tau_pilot.max()}")


def estimators(k, m, R):
    """Return H_{k:m} for h(x)=x and h(x)=x^2, and costs, for R coupled runs (lag 1)."""
    T = 4000
    X = np.zeros((R, T + 1))
    Y = np.zeros((R, T + 1))
    X[:, 0] = x0
    Y[:, 0] = x0
    X[:, 1] = rho * x0 + s * rng.normal(size=R)
    tau = np.full(R, -1)
    t = 1
    while (tau < 0).any() or t < m:
        xn, yn = coupled_step(X[:, t], Y[:, t - 1])
        # after meeting, keep the chains identical (faithful): Y_t = X_{t+1}
        xn = np.where(tau > 0, rho * X[:, t] + s * rng.normal(size=R), xn)
        yn = np.where(tau > 0, xn, yn)
        X[:, t + 1], Y[:, t] = xn, yn
        newly = (tau < 0) & (xn == yn)
        tau[newly] = t + 1
        t += 1
        assert t < T
    out = []
    for hfun in [lambda v: v, lambda v: v**2]:
        mc = hfun(X[:, k:m + 1]).mean(1)
        lv = np.arange(1, T + 1)
        D = hfun(X[:, 1:]) - hfun(Y[:, :-1])
        wts = np.minimum(1, (lv - k) / (m - k + 1))
        mask = (lv[None, :] >= k + 1) & (lv[None, :] < tau[:, None])
        out.append(mc + (mask * wts[None, :] * D).sum(1))
    cost = 2 * (tau - 1) + np.maximum(1, m + 1 - tau)
    return out[0], out[1], cost


for (k, m) in [(1, 1), (q90, q90), (q90, 5 * q90), (q90, 10 * q90), (q99, 10 * q99)]:
    Hx, Hx2, cost = estimators(k, m, 20_000)
    ineff = cost.mean() * Hx.var(ddof=1)
    print(f"k={k:3d} m={m:4d}: mean Hx={Hx.mean():+.4f} (SE {Hx.std(ddof=1) / np.sqrt(Hx.size):.4f}) "
          f"mean Hx2={Hx2.mean():.4f} (SE {Hx2.std(ddof=1) / np.sqrt(Hx2.size):.4f}) "
          f"E[cost]={cost.mean():.1f} Var(Hx)={Hx.var(ddof=1):.4f} ineff/asvar={ineff / asvar:.3f} "
          f"kurtosis(Hx)={stats.kurtosis(Hx):.1f}")

# Coverage of 95% CLT intervals with R = 50 estimators per interval, B = 1000 intervals
for (k, m) in [(1, 1), (q90, 10 * q90)]:
    B, R = 1000, 50
    Hx, Hx2, _ = estimators(k, m, B * R)
    cover = []
    for H, truth in [(Hx, 0.0), (Hx2, 1.0)]:
        Hm = H.reshape(B, R)
        mu, se = Hm.mean(1), Hm.std(1, ddof=1) / np.sqrt(R)
        c = np.mean(np.abs(mu - truth) <= stats.t.ppf(0.975, R - 1) * se)
        cover.append(f"{c:.3f} (binom p vs 0.95: {stats.binomtest(int(round(c * B)), B, 0.95).pvalue:.3f})")
    print(f"coverage k={k} m={m}, R={R}, B={B}: h=x {cover[0]}; h=x^2 {cover[1]}")
