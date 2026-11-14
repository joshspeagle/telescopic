# Probabilities of the revision's criterion K (k not chosen at 2 or more of 30 seeds fails) and of G's
# tuning-only test (more than 3 of 300 new tunings above an exact factor 0.9998 after the guard fails),
# under a per-seed (per-tuning) rate p.
from scipy import stats

print("K: p, P(fail)")
for p in (0.005, 0.01, 0.02, 0.05, 0.10):
    print(f"  {p:.3f} {stats.binom.sf(1, 30, p):.3f}")
print("G tuning test: rate, P(fail)")
for lab, p in [
    ("0", 0.0),
    ("0.003", 0.003),
    ("0.01", 0.01),
    ("2 of 100", 0.02),
    ("2 of 30", 2 / 30),
    ("6 of 30", 0.2),
    ("11 of 100", 0.11),
]:
    print(f"  {lab}: {stats.binom.sf(3, 300, p):.3f}")
