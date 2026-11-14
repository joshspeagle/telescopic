"""A pytest plugin that makes `telescopic.kernels.MALA` the whitened form; `wm_suite.out` gives the command it ran with.

Loaded with `-p wm_patch` before the test modules are imported, it replaces `MALA` in
`telescopic.kernels` and in the sampler's table of kernel names, so that every test that
builds a MALA kernel, by class or by the name "mala", builds `WhitenedMALA`.
"""

from wm_kernel import WhitenedMALA

import telescopic.kernels
import telescopic.sampler

telescopic.kernels.MALA = WhitenedMALA
for name, kernel in list(telescopic.sampler.KERNELS.items()):
    if kernel.__name__ == "MALA":
        telescopic.sampler.KERNELS[name] = WhitenedMALA
