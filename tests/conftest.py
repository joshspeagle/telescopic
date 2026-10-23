import os

from hypothesis import settings

# Set before any test module imports numpy: the tests use small matrices, where more
# BLAS threads only add overhead and contention on a busy CPU.
for name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(name, "1")

# deadline=None: a deadline turns a slow CI runner into a failure.
settings.register_profile("telescopic", max_examples=50, derandomize=True, deadline=None)
settings.load_profile("telescopic")
