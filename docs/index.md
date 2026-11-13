---
layout: home
file_format: mystnb
kernelspec:
  name: python3
---

# telescopic

:::{container} lede
**Unbiased Markov chain Monte Carlo from pairs of coupled chains.**

telescopic runs pairs of Markov chains under a coupling that makes the two chains meet,
and turns each pair into an unbiased estimate of expectations under the target. Averages
over independent pairs have error bars that are valid as the number of pairs grows, not
as the chains grow. [Theory](theory.md#guarantees-and-their-limits) states the conditions
and the limits.
:::

```{hero}
One coupled pair run by telescopic on a standard normal target with random-walk
Metropolis, drawn against time: *X*{sub}`t` in magenta and *Y*{sub}`t-1` in teal, as the
estimator pairs them with a lag of 1. The lines join at the meeting time, and one chain continues.
```

:::{container} routes
- [**I use emcee and want error bars I can trust**](tutorials/quickstart.md)

  The same line fit with emcee and with telescopic, side by side.
- [**How can MCMC be unbiased?**](tutorials/how-it-works.md)

  One run taken apart: pairs that meet, and the correction that removes the bias.
- [**My chains do not meet**](guide/diagnostics.md)

  What the meeting times and the warnings say, and what to change.
:::

This example estimates the mean of a correlated Gaussian, (0, 0), and the mean of the
product of its coordinates, 0.9.

```{code-cell} python
import numpy as np
import telescopic

cov = np.array([[1.0, 0.9], [0.9, 1.0]])
precision = np.linalg.inv(cov)

def log_prob(theta):
    return -0.5 * theta @ precision @ theta

init = telescopic.initial.Normal(mean=[0.0, 0.0], cov=4 * cov)
result = telescopic.sample(log_prob, init, rng=1)
print(result.estimate(lambda t: t))
print(result.estimate(lambda t: t[:, 0] * t[:, 1]))
```

```{toctree}
:hidden:
:caption: Start

getting-started
```

```{toctree}
:hidden:
:caption: Tutorials

Coupled chains and unbiasedness <tutorials/how-it-works>
Failure modes <tutorials/failure-modes>
Comparing with emcee <tutorials/quickstart>
Choosing k, m, and lag <tutorials/choosing-settings>
MALA and HMC with gradients <tutorials/gradients>
Coupled Gibbs sampling <tutorials/gibbs>
```

```{toctree}
:hidden:
:caption: How-to guides

guide/targets
guide/kernels
guide/settings
guide/results
guide/diagnostics
guide/parallel
guide/custom-kernels
```

```{toctree}
:hidden:
:caption: Theory

theory
```

```{toctree}
:hidden:
:caption: Reference

api
citing
changelog
```

```{toctree}
:hidden:
:caption: Project

contributing
```
