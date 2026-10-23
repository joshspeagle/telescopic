"""Check spec section 3.2 formulas: v_t, c_t (thinning), signed measure, cost, TV identity.

Synthetic faithful paths: X_0..X_T, Y_0..Y_{T-L}, with X_t = Y_{t-L} for t >= tau.
"""

import math

import numpy as np

rng = np.random.default_rng(1)


def v_spec(t, k, m, L):
    return (t - k) // L - math.ceil(max(L, t - m) / L) + 1


def v_brute(t, k, m, L):
    # number of l in [k, m] with t - l a positive multiple of L
    return sum(1 for l in range(k, m + 1) if t - l >= L and (t - l) % L == 0)


def c_spec(t, S, L):
    return sum(1 for l in S if l <= t - L and (t - l) % L == 0) / len(S)


def jrssb_reply_weight(t, k, m, L):
    return min(1.0, math.ceil((t - k) / L) / (m - k + 1))


bad_v = 0
n_v = 0
for _ in range(20000):
    L = int(rng.integers(1, 9))
    k = int(rng.integers(0, 15))
    m = k + int(rng.integers(0, 25))
    t = k + L + int(rng.integers(0, 60))
    n_v += 1
    if v_spec(t, k, m, L) != v_brute(t, k, m, L):
        bad_v += 1
print(f"v_t formula vs brute-force count: {bad_v} mismatches in {n_v}")

# L = 1 closed form
bad = 0
for k in range(0, 10):
    for m in range(k, k + 15):
        for t in range(k + 1, k + 40):
            if v_spec(t, k, m, 1) != min(t - k, m - k + 1):
                bad += 1
print(f"L=1 closed form min(t-k, m-k+1): {bad} mismatches")

# Negative v_t possible?  Zero v_t possible for t in [k+L, tau-1]?
neg = zero = tot = 0
for L in range(1, 9):
    for k in range(0, 8):
        for m in range(k, k + 12):
            for t in range(k + L, k + L + 50):
                v = v_spec(t, k, m, L)
                tot += 1
                neg += v < 0
                zero += v == 0
print(f"v_t < 0: {neg} of {tot}; v_t == 0 (zero-weight atoms inside the BC range): {zero} of {tot}")

# Definitional H_S versus weighted formula, with thinning, on synthetic faithful paths
max_err = 0.0
n_cases = 0
max_err_mcmc_bc = 0.0
for _ in range(4000):
    L = int(rng.integers(1, 7))
    k = int(rng.integers(0, 12))
    m = k + int(rng.integers(0, 20))
    s = int(rng.integers(1, 5))
    tau = L + int(rng.integers(0, 40))
    T = max(tau, m) + 3 * L + 5
    X = rng.normal(size=T + 1)
    Y = rng.normal(size=T + 1 - L)
    for t in range(tau, T + 1):  # faithfulness
        Y[t - L] = X[t]
    h = np.exp  # any function

    def delta(t):
        return h(X[t]) - h(Y[t - L])

    S = list(range(k, m + 1, s))

    def H_t(t):
        tot = h(X[t])
        j = 1
        while t + j * L <= T:
            tot += delta(t + j * L)
            j += 1
        return tot

    H_def = np.mean([H_t(l) for l in S])
    mcmc = np.mean([h(X[l]) for l in S])
    bc = sum(c_spec(t, S, L) * delta(t) for t in range(min(S) + L, tau))
    max_err = max(max_err, abs(H_def - (mcmc + bc)))
    # signed measure: atoms and weights
    atoms = [X[l] for l in S]
    weights = [1 / len(S)] * len(S)
    for t in range(min(S) + L, tau):
        c = c_spec(t, S, L)
        atoms += [X[t], Y[t - L]]
        weights += [c, -c]
    atoms = np.array(atoms)
    weights = np.array(weights)
    assert abs(weights.sum() - 1) < 1e-12
    assert len(atoms) == len(S) + 2 * max(0, tau - min(S) - L)
    max_err_mcmc_bc = max(max_err_mcmc_bc, abs(np.sum(weights * h(atoms)) - H_def))
    if s == 1:
        for t in range(k + L, tau):
            assert abs(c_spec(t, S, L) - v_spec(t, k, m, L) / (m - k + 1)) < 1e-15
    n_cases += 1
print(f"definitional H_S vs MCMC + sum c_t Delta_t: max abs error {max_err:.2e} over {n_cases} cases")
print(f"signed measure applied to h vs H_S: max abs error {max_err_mcmc_bc:.2e}")

# JRSSB reply weight differs for L >= 2
diff = 0
for L in range(2, 6):
    for k in range(0, 5):
        for m in range(k, k + 8):
            for t in range(k + L, k + L + 30):
                if abs(jrssb_reply_weight(t, k, m, L) - v_spec(t, k, m, L) / (m - k + 1)) > 1e-12:
                    diff += 1
print(f"JRSSB reply weight differs from v_t/(m-k+1) in {diff} (t,k,m,L) cells with L>=2")

# TV identity: max(0, ceil((tau - L - t)/L)) == #{j >= 1: tau > t + jL}, pointwise
bad = 0
for tau in range(1, 300):
    for L in range(1, 25):
        if tau < L:
            continue
        for t in range(0, 320):
            lhs = max(0, math.ceil((tau - L - t) / L))
            rhs = sum(1 for j in range(1, 400) if tau > t + j * L)
            bad += lhs != rhs
print(f"TV identity, pointwise: {bad} mismatches")

# Cost formula vs a step counter for the spec's construction with horizon max(tau, m)
bad = 0
for L in range(1, 8):
    for m in range(0, 30):
        for tau in range(L, 60):
            steps_single = L  # X_1..X_L
            steps_coupled = tau - L  # (X_{t+1}, Y_{t+1-L}) for t = L..tau-1
            steps_after = max(0, m - tau)  # X alone from tau to max(tau, m)
            C = steps_single + 2 * steps_coupled + steps_after
            bad += C != L + 2 * (tau - L) + max(0, m - tau)
            # 'until met and t + 1 > m' read literally after the draw gives horizon max(tau, m + 1)
print(f"cost formula vs horizon max(tau, m): {bad} mismatches")
lit_extra = sum(1 for m in range(0, 30) for tau in range(1, 60) if tau <= m)
print(f"cells where a literal 'until t+1 > m' horizon (max(tau, m+1)) adds one step: {lit_extra}")
