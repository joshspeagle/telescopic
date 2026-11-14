"""The rates that the formulas of (N) and (H) predict, and the power of the registered tests.

Question: for each family of groups B and C and each formula check (families.py), with
what probability does a line of (N) and of (H) warn, and with what probability does a
check of P = 8 lines warn, according to the detection formula of the design document
(section 5), before any run on the registered seeds?

The formula. In float64, where rounding sets the tolerance, a line warns iff

    (N)  |e.u + (h^2 / 6) d3| > C_N L + S sum_i max(U eps_g |g_i|, q_i) |u_i| + (S h^2 / 2) |d3|,
    (H)  the same, and |e.u + (h^2 / 6) d3| > PHI,

with e = grad - (the derivative of log_prob as computed) at x, u = a - b the line's
direction (with the check's fallback for point starts), L = |log_prob(x)|,
C_N = S U eps / h = 2.35e-8, d3 the third derivative of log_prob along u (0 on a
Gaussian; a second difference of the exact gradient otherwise), q_i the movement of a
float32 gradient under the check's probe (0 for a float64 one), and x, a, b independent
draws of the initial distribution. This is the inequality (gap > S (T + R_f + R_g)) with
R_f = U eps L, its value when rounding is not resolved, which the design shows is the
case on these targets. The lines of a check are independent given the initial
distribution, so a check warns with probability 1 - (1 - p)^8 for a line rate p; where
the lines share their point (point starts) the check is simulated as a whole.

Two evaluations of the formula:
- "diagonal": N(0, diag sigma^2) from N(0, diag s^2), kappa_i = s_i / sigma_i, in the
  reduced form of the design (section 5): with xi, zeta standard normal, L = |K - (1/2)
  sum kappa_i^2 xi_i^2|, and e.u = 2 sqrt(2) sum_S kappa_i^2 xi_i zeta_i for the signs of
  the coordinates S, sqrt(2) kappa_0^2 xi_0 zeta_0 for coordinate 0 set to 0,
  r sqrt(2) sum kappa_i^2 xi_i zeta_i for a factor 1 + r, c sqrt(2) sum s_i zeta_i for an
  offset c (sigma = 1). For one coordinate the probability is averaged in closed form over
  zeta_0 (2 Phi(t / |xi_0|) - 1), which resolves rates near 0 and 1.
- "direct": draws of the family's own initial distribution, its log_prob and both
  gradients, the expression above line by line.
The rotated families are also equivalent to diagonal ones by the affine invariance of the
check (design section 3); the direct evaluation of two diagonal families checks the
reduced form.

Generators: model k (in the order printed) at seed 395000 + k (PREREGISTRATION.md,
section 6). No check runs here.

Run: python notes/experiments/x123/predictions.py [first stop], the models first..stop-1
(their parts can be run separately and joined; the comparison and the power follow the
last model)

Result: see predictions.out; the tables are copied into PREREGISTRATION.md.
"""

import sys
import warnings

import families as F
import gradcheck as G
import numpy as np
from runner import header
from scipy import stats

EPS = np.finfo(np.float64).eps
H = EPS ** (1 / 3)
C_N = G.SAFETY * G.ULPS * EPS / H
P = G.POINTS
SQ2 = np.sqrt(2.0)


def diagonal(rng, d, k0, k, kind, size=0.0, special=1, K=0.0, M=2_000_000):
    # line rates of (N) and (H) in the reduced form; returns (p_N, p_H, se_N, se_H)
    rest = d - special
    xi = rng.standard_normal((M, special))
    chi = rng.chisquare(rest, M) if rest else np.zeros(M)
    L = np.abs(K - 0.5 * (k0**2 * np.sum(xi**2, axis=1) + k**2 * chi))
    out = []
    for floor in (0.0, G.PHI):
        bar = np.maximum(C_N * L, floor)
        if kind in ("sign", "zero") and special == 1:
            # P(|c xi zeta| > bar | xi, chi) = 2 Phi(-bar / |c xi|), averaged over xi and chi
            c = (2 * SQ2 if kind == "sign" else SQ2) * k0**2
            with np.errstate(divide="ignore"):
                p = 2 * stats.norm.sf(bar / np.abs(c * xi[:, 0]))
        else:
            zeta = rng.standard_normal((M, special))
            n = rng.standard_normal(M)
            if kind == "sign":
                eu = 2 * SQ2 * k0**2 * np.sum(xi * zeta, axis=1)
            elif kind == "factor":
                eu = size * SQ2 * (k0**2 * np.sum(xi * zeta, axis=1) + k**2 * np.sqrt(chi) * n)
            else:  # offset
                eu = size * SQ2 * (k0 * np.sum(zeta, axis=1) + k * np.sqrt(rest) * n)
            p = (np.abs(eu) > bar).astype(float)
        out.append((p.mean(), p.std() / np.sqrt(M)))
    return out[0][0], out[1][0], out[0][1], out[1][1]


CHOLESKY = {}


def draws(initial, n, rng):
    if isinstance(initial, F.DiagonalNormal):
        return initial.mean + initial.sd * rng.standard_normal((n, len(initial.sd)))
    if isinstance(initial, F.Normal):
        if id(initial) not in CHOLESKY:
            CHOLESKY[id(initial)] = (initial, np.linalg.cholesky(initial.cov))
        return (
            initial.mean + rng.standard_normal((n, len(initial.mean))) @ CHOLESKY[id(initial)][1].T
        )
    return np.array([initial.rvs(random_state=rng) for _ in range(n)])


def lines(right, wrong, x, u, rng):
    # warnings of (N) and (H) on lines at the rows of x along the rows of u
    lp, g_true, _ = right
    g_wrong = wrong[1]
    smallest = np.sqrt(EPS) * np.abs(x)
    u = np.where(np.abs(u) < smallest, np.copysign(smallest, u), u)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        given = np.asarray(g_wrong(x))
        narrow = given.dtype == np.float32
        given = given.astype(float)
        e = given - np.asarray(g_true(x), float)
        q = lambda t: np.sum(np.asarray(g_true(x + t * u), float) * u, axis=1)
        tau = 1e-3
        third = (q(tau) - 2 * q(0.0) + q(-tau)) / tau**2
        eps_g = np.finfo(np.float32).eps if narrow else EPS
        weight = G.ULPS * eps_g * np.abs(given)
        if narrow:
            signs = rng.choice([-1.0, 1.0], size=x.shape)
            moved = np.asarray(g_wrong(x * (1 + 4 * eps_g * signs))).astype(float)
            weight = np.maximum(weight, np.abs(moved - given))
        L = np.abs(np.asarray(lp(x), float))
    gap = np.abs(np.sum(e * u, axis=1) + H**2 / 6 * third)
    tol = (
        C_N * L
        + G.SAFETY * np.sum(weight * np.abs(u), axis=1)
        + G.SAFETY * H**2 / 2 * np.abs(third)
    )
    return gap > tol, gap > np.maximum(tol, G.PHI)


def direct(rng, right, wrong, M, chunk=1000):
    # independent lines from a continuous start; returns (p_N, p_H, se_N, se_H)
    init = right[2]
    warned = np.zeros((2, 0), dtype=bool)
    while warned.shape[1] < M:
        n = min(chunk, M - warned.shape[1])
        x, a, b = draws(init, n, rng), draws(init, n, rng), draws(init, n, rng)
        warned = np.concatenate([warned, np.array(lines(right, wrong, x, a - b, rng))], axis=1)
    p = warned.mean(axis=1)
    se = np.sqrt(np.maximum(p * (1 - p), 1 / M) / M)
    return p[0], p[1], se[0], se[1]


def point_start(rng, right, wrong, M, seed_dependent=False):
    # whole checks from a point start: the 8 lines share x, and u = random signs times
    # |x| (1 where x = 0), the check's fallback when the 3 P draws coincide; returns
    # check rates (c_N, c_H, se_N, se_H)
    out = np.zeros((2, M), dtype=bool)
    for m in range(M):
        case_right = F.resolve(right, int(rng.integers(2**31))) if seed_dependent else right
        case_wrong = (
            case_right[0],
            (F.resolve(wrong, 0) if seed_dependent else wrong)[1],
            case_right[2],
        )
        x = np.repeat(case_right[2].points[:1], P, axis=0)
        u = rng.choice([-1.0, 1.0], size=x.shape) * np.where(x != 0, np.abs(x), 1.0)
        n, h = lines(case_right, case_wrong, x, u, rng)
        out[:, m] = n.any(), h.any()
    c = out.mean(axis=1)
    se = np.sqrt(np.maximum(c * (1 - c), 1 / M) / M)
    return c[0], c[1], se[0], se[1]


def jacobi(rng, rtol, M=200_000, chunk=2000):
    # The Jacobi families: within a region of constant iteration count k, log_prob is the
    # quadratic form -x.J_k x / 2 with J_k = sum_{j<=k} M^j D^-1, M = I - D^-1 A, so its
    # gradient is -(J_k + J_k^T) x / 2, and e = -A^-1 x + (J_k + J_k^T) x / 2.
    A = F.JACOBI_MATRIX
    Dinv = np.diag(1 / np.diag(A))
    Mj = np.eye(5) - Dinv @ A
    J = [Dinv]
    for _ in range(500):
        J.append(Dinv + Mj @ J[-1])
    Ainv = np.linalg.inv(A)
    init = F.loose_solver(rtol)[2]
    warned = []
    for _ in range(M // chunk):
        x, a, b = draws(init, chunk, rng), draws(init, chunk, rng), draws(init, chunk, rng)
        u = a - b
        for xi, ui in zip(x, u, strict=True):
            y, k = F.jacobi_solve(xi, rtol)
            Jk = J[k]
            e = -Ainv @ xi + 0.5 * (Jk + Jk.T) @ xi
            L = abs(-0.5 * xi @ y)
            gap = abs(e @ ui)
            tol = C_N * L
            warned.append((gap > tol, gap > max(tol, G.PHI)))
    w = np.array(warned).mean(axis=0)
    se = np.sqrt(np.maximum(w * (1 - w), 1 / len(warned)) / len(warned))
    return w[0], w[1], se[0], se[1]


# The model of each family: ("diagonal", kwargs), ("direct", lines), ("point", seed
# dependent) or ("jacobi", rtol).
def model_of(group, name):
    C = {
        "N(0,I) d=5: wrong sign": (
            "diagonal",
            {"d": 5, "k0": 2, "k": 2, "kind": "factor", "size": -2.0, "special": 0},
        ),
        "N(0,I) d=5: twice": (
            "diagonal",
            {"d": 5, "k0": 2, "k": 2, "kind": "factor", "size": 1.0, "special": 0},
        ),
        "N(0,I) d=50: twice": (
            "diagonal",
            {"d": 50, "k0": 2, "k": 2, "kind": "factor", "size": 1.0, "special": 0},
        ),
        "N(0,I) d=200: twice": (
            "diagonal",
            {"d": 200, "k0": 2, "k": 2, "kind": "factor", "size": 1.0, "special": 0},
        ),
        "N(0,I) d=10000, start sd 2: signs of coordinates 0..9": (
            "diagonal",
            {"d": 10_000, "k0": 2, "k": 2, "kind": "sign", "special": 10},
        ),
        "N(0,I) d=5, start sd 2: coordinate 0 set to 0": (
            "diagonal",
            {"d": 5, "k0": 2, "k": 2, "kind": "zero"},
        ),
        "scales (1, 0.1 x4), start N(0, I): sign of the wide coordinate": (
            "diagonal",
            {"d": 5, "k0": 1, "k": 10, "kind": "sign"},
        ),
        "scales (1, 0.01 x4), start N(0, I): sign of the wide coordinate": (
            "diagonal",
            {"d": 5, "k0": 1, "k": 100, "kind": "sign"},
        ),
        "scales (1, 0.1 x4), start N(0, 0.1^2 I): sign of the wide coordinate": (
            "diagonal",
            {"d": 5, "k0": 0.1, "k": 1, "kind": "sign"},
        ),
        "scales (1, 0.01 x4), start N(0, 0.01^2 I): sign of the wide coordinate": (
            "diagonal",
            {"d": 5, "k0": 0.01, "k": 1, "kind": "sign"},
        ),
        "N(0,I) d=5, start sd 1000: sign of coordinate 0": (
            "diagonal",
            {"d": 5, "k0": 1000, "k": 1000, "kind": "sign"},
        ),
        "N(0,I) d=2, point start (1, 1): grad + 2 (1, -1)": ("point", False),
        "N(0,I) d=5, random point start: sign of coordinate 0": ("point", True),
    }
    B = {
        "Jacobi solver rtol 1e-3 d=5 (exact gradient)": ("jacobi", 1e-3),
        "Jacobi solver rtol 1e-5 d=5 (exact gradient)": ("jacobi", 1e-5),
        "Jacobi solver rtol 1e-8 d=5 (exact gradient)": ("jacobi", 1e-8),
        "N(0,I) d=5: grad x 1.005": (
            "diagonal",
            {"d": 5, "k0": 2, "k": 2, "kind": "factor", "size": 0.005, "special": 0},
        ),
        "N(0,I) d=1000: grad x 1.0001": (
            "diagonal",
            {"d": 1000, "k0": 2, "k": 2, "kind": "factor", "size": 1e-4, "special": 0},
        ),
        "N(0,I) d=5: grad + 0.02": (
            "diagonal",
            {"d": 5, "k0": 2, "k": 2, "kind": "offset", "size": 0.02, "special": 0},
        ),
        "N(0,I) d=5, start 100x wide: grad x 1.005": (
            "diagonal",
            {"d": 5, "k0": 200, "k": 200, "kind": "factor", "size": 0.005, "special": 0},
        ),
    }
    if group == "B":
        return B.get(name, ("direct", None))
    if name in C:
        return C[name]
    for kappa in (1.0, 2.0):
        for d in (5, 50, 200, 1000, 2000, 5000, 10_000):
            if name == f"N(0,I) d={d}, start sd {kappa:g}: sign of coordinate 0":
                return "diagonal", {"d": d, "k0": kappa, "k": kappa, "kind": "sign"}
    for tight in (0.1, 0.01, 0.003, 1e-4, 0.5, 0.25):
        for d in (5, 100, 1000, 10_000):
            if name == f"N(0,I) d={d}, start sd ({tight:g}, 2, ...): sign of coordinate 0":
                return "diagonal", {"d": d, "k0": tight, "k": 2.0, "kind": "sign"}
    for wide in (100.0, 300.0, 1000.0, 1e4):
        if name == f"N(0,I) d=5, start sd {wide:g}: grad + 2":
            return "diagonal", {
                "d": 5,
                "k0": wide,
                "k": wide,
                "kind": "offset",
                "size": 2.0,
                "special": 0,
            }
    if name == "N(0,I) d=5, start sd (1e-4, 2, ...): sign of coordinate 0":
        return "diagonal", {"d": 5, "k0": 1e-4, "k": 2.0, "kind": "sign"}
    if name == "scales (1, 1e-4 x4), start N(0, I): sign of the wide coordinate":
        return "diagonal", {"d": 5, "k0": 1.0, "k": 1e4, "kind": "sign"}
    if name == "N(0,I) - 1e4 d=5, start sd 1e-3: sign of coordinate 0":
        return "diagonal", {"d": 5, "k0": 1e-3, "k": 1e-3, "kind": "sign", "K": -1e4}
    return "direct", None


def predict(group, name, right, wrong, rng):
    kind, spec = model_of(group, name)
    if kind == "diagonal":
        pn, ph, sn, sh = diagonal(rng, **spec)
    elif kind == "jacobi":
        pn, ph, sn, sh = jacobi(rng, spec)
    elif kind == "point":
        cn, ch, sn, sh = point_start(rng, right, wrong, 20_000, seed_dependent=spec)
        return kind, None, None, cn, ch, sn, sh
    else:
        d = len(draws(F.resolve(right, 0)[2], 1, rng)[0])
        M = 200_000 if d <= 10 else (20_000 if d <= 1000 else 4000)
        pn, ph, sn, sh = direct(rng, right, wrong, M)
    cn, ch = 1 - (1 - pn) ** P, 1 - (1 - ph) ** P
    return kind, pn, ph, cn, ch, P * (1 - pn) ** (P - 1) * sn, P * (1 - ph) ** (P - 1) * sh


def families():
    for name, case in F.GROUP_B.items():
        # the direct model of group B needs the family's exact gradient (B_RIGHT)
        yield "B", name, B_RIGHT.get(name), case
    for name, (right, wrong) in F.GROUP_C.items():
        yield "C", name, right, wrong
    for name, (right, wrong) in F.FORMULA.items():
        yield "F", name, right, wrong


B_RIGHT = {
    "logistic n=1e4 d=5: without the prior": F.logistic(),
    "logistic n=1e4 d=5, start sd 10: without the prior": F.logistic(init_sd=10.0),
}


def show(p):
    if p is None:
        return f"{'':>10}"
    if p > 0.999:
        return f"1-{1 - p:7.1e}" if p < 1 else f"{'1':>10}"
    return f"{p:10.4f}"


def main():
    header()
    print(f"C_N = {C_N:.4e}; PHI = {G.PHI:.4f}; h = {H:.4e}; P = {P}")
    print(
        f"{'k':>3} {'group':<5} {'family':<74} {'model':<8} | {'(N) line':>10} {'(N) check':>10} {'+-':>8} | "
        f"{'(H) line':>10} {'(H) check':>10} {'+-':>8}"
    )
    models = list(families())
    first, stop = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, len(models))
    for k, (group, name, right, wrong) in enumerate(models):
        if not first <= k < stop:
            continue
        rng = np.random.default_rng(395_000 + k)
        kind, pn, ph, cn, ch, sn, sh = predict(group, name, right, wrong, rng)
        print(
            f"{k:>3} {group:<5} {name:<74} {kind:<8} | {show(pn)} {show(cn)} {sn:8.1e} | {show(ph)} {show(ch)} {sh:8.1e}",
            flush=True,
        )

    if stop < len(models):
        return
    print("\nThe reduced form against the direct evaluation (generators 395100, 395101):")
    for j, name in enumerate(
        [
            "N(0,I) d=5, start sd 1: sign of coordinate 0",
            "scales (1, 0.01 x4), start N(0, 0.01^2 I): sign of the wide coordinate",
        ]
    ):
        right, wrong = F.GROUP_C[name]
        rng = np.random.default_rng(395_100 + j)
        reduced = diagonal(rng, **model_of("C", name)[1])
        full = direct(rng, right, wrong, 200_000)
        print(
            f"  {name}: (N) line {reduced[0]:.4f} reduced, {full[0]:.4f} +- {full[2]:.4f} direct; "
            f"(H) line {reduced[1]:.4f} reduced, {full[1]:.4f} +- {full[3]:.4f} direct"
        )

    b = stats.binom
    n_a = len(F.GROUP_A) * F.SEEDS["A"][1]
    pooled = int(0.0005 * n_a)
    print("\nThe size and power of the registered tests (exact binomial):")
    print(
        f"  1. false alarms per family, at most 2 of 300 warn: holds with probability {b.cdf(2, 300, 0.03):.4f} at a true rate of 3%, "
        f"{b.cdf(2, 300, 0.02):.3f} at 2%, {b.cdf(2, 300, 1e-3):.4f} at 0.1%, {b.cdf(2, 300, 4e-4):.5f} at 0.04%; all "
        f"{len(F.GROUP_A)} families hold with probability {b.cdf(2, 300, 4e-4) ** len(F.GROUP_A):.4f} at 0.04% each"
    )
    print(
        f"     pooled, at most {pooled} of {n_a} (0.05%): holds with probability {b.cdf(pooled, n_a, 2e-3):.1e} at 1 in 500, "
        f"{b.cdf(pooled, n_a, 1e-3):.4f} at 1 in 1,000, {b.cdf(pooled, n_a, 2e-4):.4f} at 1 in 5,000, {b.cdf(pooled, n_a, 1e-4):.5f} at 1 in 10,000"
    )
    print(
        f"  2. detection, at least 390 of 400 warn: holds with probability {b.sf(389, 400, 0.95):.4f} at a true rate of 0.95, "
        f"{b.sf(389, 400, 0.96):.4f} at 0.96, {b.sf(389, 400, 0.99):.4f} at 0.99, {b.sf(389, 400, 0.999):.6f} at 0.999"
    )
    print(
        f"     on 1,000 seeds, at least 975 of 1000: {b.sf(974, 1000, 0.95):.1e} at 0.95, {b.sf(974, 1000, 0.99):.6f} at 0.99"
    )
    for p in (0.0, 0.2, 0.46, 0.54, 0.81, 0.985, 0.4):
        lo, hi = np.ceil(1000 * (p - 0.05) - 1e-9), np.floor(1000 * (p + 0.05) + 1e-9)
        hold = b.cdf(hi, 1000, p) - b.cdf(lo - 1, 1000, p)
        off = max(
            b.cdf(hi, 1000, q) - b.cdf(lo - 1, 1000, q) for q in (p + 0.1, p - 0.1) if 0 <= q <= 1
        )
        print(
            f"  3. formula check at a predicted rate of {p}: |measured - predicted| <= 0.05 on 1,000 seeds fails with probability "
            f"{1 - hold:.1e} if the formula is right, and holds with probability at most {off:.1e} if the true rate is 0.1 away"
        )
    print(
        f"  4. a predicted miss, at most 10 of 400 warn: holds with probability {b.cdf(10, 400, 0.05):.4f} at a true rate of 5%, "
        f"{b.cdf(10, 400, 0.001):.6f} at 0.1%"
    )


if __name__ == "__main__":
    main()
