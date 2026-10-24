# telescopic

*Unbiased Markov chain Monte Carlo with coupled chains and telescoping sums.*

telescopic runs pairs of coupled Markov chains until they meet, and turns each pair into
an exactly unbiased estimate of an expectation under the target distribution. Averages
over independent pairs come with error bars that are valid as the number of pairs grows,
not as the chains grow, and the pairs run in parallel without communication.

The package is in development. The design documents below describe what it will do.

```{toctree}
:maxdepth: 2
:caption: Background

theory
```

```{toctree}
:maxdepth: 1
:caption: Development

design/index
changelog
contributing
```
