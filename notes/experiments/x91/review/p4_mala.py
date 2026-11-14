# Tuning only (no pair is run), probe seed 900: the step that dual averaging toward 0.99 returns for MALA
# without jitter (a proxy of MCF's nominal step), against the package's 0.574; and the throughput of the
# funnel's gradient at the pilot's batch of 200 rows, for the cost of part F.
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.path.insert(0, "notes/experiments")
from telescopic import CoupledSampler, examples
from telescopic.kernels import MALA


class M99(MALA):
    target_acceptance = 0.99

    def _with_cov(self, cov):
        return M99(self.step_size, cov=cov)


from targets import logistic

for name, ex in [
    ("gaussian10", examples.gaussian(10)),
    ("banana U[-5,5]^2", examples.banana()),
    ("logistic", logistic()[0]),
]:
    for K, lab in ((MALA, "0.574"), (M99, "0.99")):
        s = CoupledSampler(
            ex.log_prob, ex.initial, vectorize=True, grad=ex.grad, kernel=K(), rng=900
        )
        s.tune()
        e = s.kernel.step_size
        print(
            name,
            "target",
            lab,
            f"step {e:.4f}",
            f"; drift contraction per step eps^2/2 = {e * e / 2:.4f}, so about {2 / (e * e):.0f} iterations per e-fold of the gap",
        )
ex = examples.funnel(10)
x = np.random.default_rng(0).normal(size=(200, 10))
t = time.time()
n = 0
while time.time() - t < 2:
    ex.grad(x)
    ex.log_prob(x)
    n += 200
rate = n / (time.time() - t)
print(f"funnel: {rate:.0f} gradient rows per second at a batch of 200")
for L in (42, 99, 180):
    print(
        f"  CF at {L:d} leapfrog steps: 100 pairs unmet to the cap of 10,000 cost "
        f"{2 * 100 * 1e4 * L / rate:.0f} s; if half meet early {100 * 1e4 * L / rate:.0f} s"
    )
