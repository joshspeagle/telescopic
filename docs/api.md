# API reference

Each class has a page with its attributes and methods, and so does the function
`sample`. Each module has a page with the rest of its public names, its functions and,
for `telescopic.plotting`, its three colors, and a table of its classes.

## Running

```{eval-rst}
.. autosummary::
   :toctree: api
   :nosignatures:

   ~telescopic.sample
   ~telescopic.CoupledSampler
```

## Results

```{eval-rst}
.. autosummary::
   :toctree: api
   :nosignatures:

   ~telescopic.Result
   ~telescopic.Estimate
   ~telescopic.Summary
   ~telescopic.SignedMeasure
```

## Targets and initial distributions

```{eval-rst}
.. autosummary::
   :toctree: api
   :nosignatures:

   ~telescopic.Target
   telescopic.initial
   ~telescopic.initial.Normal
   ~telescopic.initial.Uniform
   ~telescopic.initial.Empirical
```

## Kernels and couplings

```{eval-rst}
.. autosummary::
   :toctree: api
   :nosignatures:

   telescopic.kernels
   ~telescopic.kernels.Kernel
   ~telescopic.kernels.State
   ~telescopic.kernels.Draws
   ~telescopic.kernels.RandomWalk
   ~telescopic.kernels.MALA
   ~telescopic.kernels.HMC
   ~telescopic.kernels.Independence
   ~telescopic.kernels.MultivariateT
   ~telescopic.kernels.Gibbs
   ~telescopic.kernels.Mixture
   telescopic.couplings
```

## Diagnostics, plots and checks

```{eval-rst}
.. autosummary::
   :toctree: api
   :nosignatures:

   telescopic.diagnostics
   ~telescopic.diagnostics.GradientCheck
   ~telescopic.diagnostics.Issue
   telescopic.plotting
   telescopic.testing
   ~telescopic.testing.CouplingReport
```

## Examples

```{eval-rst}
.. autosummary::
   :toctree: api
   :nosignatures:

   telescopic.examples
   ~telescopic.examples.Example
```

## Building blocks

```{eval-rst}
.. autosummary::
   :toctree: api
   :nosignatures:

   telescopic.engine
   ~telescopic.engine.Replicates
   ~telescopic.engine.Settings
   ~telescopic.engine.Status
   telescopic.estimators
   telescopic.tuning
   ~telescopic.tuning.PilotRun
   ~telescopic.tuning.TuningReport
```

## Warnings

```{eval-rst}
.. autosummary::
   :toctree: api
   :nosignatures:

   telescopic.warnings
   ~telescopic.warnings.TelescopicWarning
   ~telescopic.warnings.ConvergenceWarning
   ~telescopic.warnings.InputWarning
```
