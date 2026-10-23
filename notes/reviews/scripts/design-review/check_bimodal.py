"""JOA20 section 5.1 failure case: do spec 7.4 warnings 2-4 fire at R = 1000?

Test plan 2.12 expects: 'On JOA20's bimodal case with a narrow pi0, warning 3 or 4 fires in
most runs at R = 1000.' Setting: pi = 0.5 N(-4,1) + 0.5 N(4,1), RWMH proposal sd 1,
pi0 = N(10, 1), lag 1, k = 50, m = 500, h(x) = 1(x > 3), truth 0.5 Phi(1) + 0.5 Phi(-7).
Coupling: reflection-maximal proposals (in 1-D the reflection is z -> -z), common uniform.
R_total replicates in lockstep with compaction; split into batches of 1000.
"""
import time

import numpy as np
from scipy import stats

R_TOTAL, BATCH, K, M, CAP = 30_000, 1000, 50, 500, 3_000_000
truth = 0.5 * stats.norm.cdf(1) + 0.5 * stats.norm.cdf(-7)


def logpi(x):
    return np.logaddexp(-0.5 * (x + 4) ** 2, -0.5 * (x - 4) ** 2)


def h(x):
    return (x > 3).astype(float)


rng = np.random.default_rng(31)
t0 = time.perf_counter()
x = 10 + rng.standard_normal(R_TOTAL)
y = 10 + rng.standard_normal(R_TOTAL)
lx, ly = logpi(x), logpi(y)
n = M - K + 1
mcmc = np.zeros(R_TOTAL)
bc = np.zeros(R_TOTAL)
crossed = np.zeros(R_TOTAL, bool)  # a chain visited x < 0 before the meeting
# lag step
prop = x + rng.standard_normal(R_TOTAL)
lp = logpi(prop)
acc = np.log(rng.random(R_TOTAL)) < lp - lx
x, lx = np.where(acc, prop, x), np.where(acc, lp, lx)
t = 1
tau = np.full(R_TOTAL, -1)
met = np.zeros(R_TOTAL, bool)
active = np.arange(R_TOTAL)
while active.size and t < CAP:
    t += 1
    xa, ya, lxa, lya, ma = x[active], y[active], lx[active], ly[active], met[active]
    z = rng.standard_normal(active.size)
    logu, logv = np.log(rng.random(active.size)), np.log(rng.random(active.size))
    dlt = xa - ya
    ident = (~ma) & (logu < -z * dlt - 0.5 * dlt**2)
    px = xa + z
    py = np.where(ident, px, ya - z)
    lpx = logpi(px)
    lpy = np.where(ident, lpx, logpi(py))
    ax, ay = logv < lpx - lxa, logv < lpy - lya
    xa, lxa = np.where(ax, px, xa), np.where(ax, lpx, lxa)
    ynew, lynew = np.where(ay, py, ya), np.where(ay, lpy, lya)
    now = ident & ax & ay
    tau_a = tau[active]
    tau_a[now] = t
    still = (~ma) & (~now)
    crossed[active] |= still & ((xa < 0) | (ynew < 0))
    if K <= t <= M:
        mcmc[active] += h(xa)
    if t >= K + 1:
        w = min(1.0, (t - K) / n)
        bc[active] += np.where(still, w * (h(xa) - h(ynew)), 0.0)
    ma = ma | now
    ya = np.where(ma, xa, ynew)
    lya = np.where(ma, lxa, lynew)
    x[active], y[active], lx[active], ly[active], met[active], tau[active] = xa, ya, lxa, lya, ma, tau_a
    active = active[~(ma & (t >= M))]
print(f"simulated {R_TOTAL} replicates in {time.perf_counter() - t0:.0f}s; censored at {CAP}: {active.size}")
H = mcmc / n + bc
print(f"tau: mean {tau.mean():.1f}, q99 {np.quantile(tau, 0.99):.0f}, max {tau.max()}; "
      f"replicates with a chain crossing to x<0 before meeting: {crossed.sum()} ({crossed.mean():.5f})")
print(f"all {R_TOTAL}: estimate {H.mean():.4f} +- {1.96 * H.std(ddof=1) / np.sqrt(R_TOTAL):.4f} (truth {truth:.4f})")


def hill_alpha(v, kh):
    vs = np.sort(v)
    return 1.0 / np.mean(np.log(vs[-kh:]) - np.log(vs[-kh - 1]))


nb = R_TOTAL // BATCH
cover = w2 = w3 = w4 = w_any = 0
for b in range(nb):
    sl = slice(b * BATCH, (b + 1) * BATCH)
    Hb, tb = H[sl], tau[sl]
    est, se = Hb.mean(), Hb.std(ddof=1) / np.sqrt(BATCH)
    cov = abs(est - truth) <= 1.96 * se
    dev = (Hb - est) ** 2
    f2 = np.mean(tb - 1 > K) > 0.05
    f3 = dev.max() / dev.sum() > 0.25
    f4 = hill_alpha(tb.astype(float), int(np.sqrt(BATCH))) < 2  # tau - L + 1 = tau for L = 1
    cover += cov
    w2 += f2
    w3 += f3
    w4 += f4
    w_any += f3 or f4
    print(f"  batch {b:2d}: est {est:.3f} +- {1.96 * se:.3f} covers {cov}, crossings {crossed[sl].sum()}, "
          f"max var share {dev.max() / dev.sum():.3f}, hill {hill_alpha(tb.astype(float), 31):.2f}, w2 {f2}")
print(f"batches of {BATCH}: CI covers truth {cover}/{nb}; warning 2 {w2}/{nb}; warning 3 {w3}/{nb}; "
      f"warning 4 {w4}/{nb}; warning 3 or 4 {w_any}/{nb}")
