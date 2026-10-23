# Sanity checks of index conventions for lagged unbiased estimators and TV bounds (own code).
import math, random
random.seed(1)

def H_t(hX, hY, t, tau, L):
    # H_t = h(X_t) + sum_{j>=1, t+jL < tau} [h(X_{t+jL}) - h(Y_{t+(j-1)L})]
    s = hX[t]; j = 1
    while t + j*L <= tau - 1:
        s += hX[t+j*L] - hY[t+(j-1)*L]; j += 1
    return s

def H_km_direct(hX, hY, k, m, tau, L):
    return sum(H_t(hX, hY, t, tau, L) for t in range(k, m+1)) / (m-k+1)

def H_km_weighted(hX, hY, k, m, tau, L):
    mc = sum(hX[t] for t in range(k, m+1)) / (m-k+1)
    bc = 0.0
    for s in range(k+L, tau):
        v = (math.floor((s-k)/L) - math.ceil(max(L, s-m)/L) + 1) / (m-k+1)
        bc += v * (hX[s] - hY[s-L])
    return mc + bc

def H_km_reply_ceiling(hX, hY, k, m, tau, L):
    mc = sum(hX[t] for t in range(k, m+1)) / (m-k+1)
    bc = sum(min(1.0, math.ceil((s-k)/L)/(m-k+1)) * (hX[s]-hY[s-L]) for s in range(k+L, tau))
    return mc + bc

maxerr_w = 0; maxerr_r = 0
for trial in range(3000):
    L = random.randint(1, 7); k = random.randint(0, 15); m = k + random.randint(0, 30)
    tau = L + 1 + random.randint(0, 60)
    T = max(tau, m) + 5*L + 5
    hX = [random.gauss(0,1) for _ in range(T+1)]
    hY = [random.gauss(0,1) for _ in range(T+1)]
    for t in range(tau, T+1):      # faithfulness: X_t = Y_{t-L} for t >= tau
        if t - L >= 0: hX[t] = hY[t-L]
    d = H_km_direct(hX, hY, k, m, tau, L)
    maxerr_w = max(maxerr_w, abs(d - H_km_weighted(hX, hY, k, m, tau, L)))
    maxerr_r = max(maxerr_r, abs(d - H_km_reply_ceiling(hX, hY, k, m, tau, L)))
print("max |direct - weighted(2.10)| =", maxerr_w)
print("max |direct - min(1,ceil((l-k)/L)/(m-k+1)) form| =", maxerr_r)

# TV bound identity: E[max(0, ceil((tau-L-k)/L))] == sum_{j>=1} P(tau > k + jL)
taus = [random.choice([3,5,8,13,40,41,100]) + random.randint(0,20) for _ in range(5000)]
for L in (1, 4, 25):
    for k in (0, 7, 30):
        lhs = sum(max(0, math.ceil((tau-L-k)/L)) for tau in taus)/len(taus)
        rhs = sum(sum(1 for tau in taus if tau > k + j*L)/len(taus) for j in range(1, 500))
        assert abs(lhs-rhs) < 1e-12, (L,k,lhs,rhs)
print("TV identity E[J] = sum_j P(tau > k+jL) verified")
# W1 upper limit identity: ceil((tau-L-k)/L) == floor((tau-k-1)/L) when tau-k >= 1
ok = all(math.ceil((tau-L-k)/L) == math.floor((tau-k-1)/L) for tau in range(1,300) for L in range(1,20) for k in range(0,300) if tau-k>=1)
print("ceil((tau-L-k)/L) == floor((tau-k-1)/L) for tau-k>=1:", ok)
# Craiu-Meng bound B = sum_{j>=1} min(P(J>=j), P(J<=j)) <= E[J]
for L in (1, 4, 25):
    for k in (0, 7, 30):
        J = [max(0, math.ceil((tau-L-k)/L)) for tau in taus]
        n = len(J); EJ = sum(J)/n
        B = sum(min(sum(1 for x in J if x >= j)/n, sum(1 for x in J if x <= j)/n) for j in range(1, max(J)+2))
        print(f"L={L:3d} k={k:3d}  E[J]={EJ:7.3f}  Craiu-Meng B={B:7.3f}")
