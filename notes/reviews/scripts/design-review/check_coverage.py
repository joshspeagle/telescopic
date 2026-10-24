"""Coverage of the 95% CLT interval at R = 200 under the default recipe (test plan 2.12).

Exact AR(1) kernel x' = rho x + sqrt(1 - rho^2) xi (rho = 0.9), stationary law N(0, 1);
pairs coupled by the reflection-maximal coupling of N(rho x, s^2) and N(rho y, s^2).
Recipe: lag-1 pilot of 2000 meeting times, k = ceil(q99(tau - 1)), L = k, m = 10k.
1000 intervals of R = 200 (the test plan's design); h(x) = x (truth 0) and x^2 (truth 1).
"""
import numpy as np

rho, s = 0.9, np.sqrt(1 - 0.81)
rng = np.random.default_rng(41)


def run(R, k, m, lag, mu0, sd0):
    x = mu0 + sd0 * rng.standard_normal(R)
    y = mu0 + sd0 * rng.standard_normal(R)
    n = m - k + 1
    H1, H2 = np.zeros(R), np.zeros(R)
    if k == 0:
        H1 += x / n
        H2 += x * x / n
    t = 0
    for _ in range(lag):
        t += 1
        x = rho * x + s * rng.standard_normal(R)
        if k <= t <= m:
            H1 += x / n
            H2 += x * x / n
    tau = np.full(R, -1)
    met = np.zeros(R, bool)
    while t < m or not met.all():
        t += 1
        z = rng.standard_normal(R)
        logu = np.log(rng.random(R))
        dlt = (rho * x - rho * y) / s
        ident = (~met) & (logu < -z * dlt - 0.5 * dlt**2)
        xn = rho * x + s * z
        yn = np.where(ident | met, xn, rho * y - s * z)
        now = ident
        tau[now] = t
        x = xn
        if k <= t <= m:
            H1 += x / n
            H2 += x * x / n
        still = ~met & ~now
        if t >= k + lag:
            v = (t - k) // lag - int(np.ceil(max(lag, t - m) / lag)) + 1
            if v > 0:
                H1 += np.where(still, v * (x - yn), 0.0) / n
                H2 += np.where(still, v * (x * x - yn * yn), 0.0) / n
        met |= now
        y = np.where(met, x, yn)
    return H1, H2, tau


for label, mu0, sd0 in (("stationary N(0,1)", 0.0, 1.0), ("near-stationary N(1,1)", 1.0, 1.0)):
    _, _, tp = run(2000, 0, 0, 1, mu0, sd0)
    k = int(np.ceil(np.quantile(tp - 1, 0.99)))
    m = 10 * k
    H1, H2, tau = run(200 * 1000, k, m, k, mu0, sd0)
    print(f"{label}: k = L = {k}, m = {m}; P(tau - L > k) = {np.mean(tau - k > k):.4f}")
    for name, H, truth in (("x", H1, 0.0), ("x^2", H2, 1.0)):
        Hb = H.reshape(1000, 200)
        est = Hb.mean(1)
        se = Hb.std(1, ddof=1) / np.sqrt(200)
        cov = np.mean(np.abs(est - truth) <= 1.959964 * se)
        exk = np.mean((H - H.mean()) ** 4) / H.var() ** 2 - 3
        print(f"  h={name:3s}: coverage {cov:.3f} (binomial se {np.sqrt(cov * (1 - cov) / 1000):.3f}); "
              f"excess kurtosis of H {exk:.1f}")
