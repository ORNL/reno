# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.5.0] - 2025-11-15

### Added

* A `stock.outflows` op which evaluates to the sum of the flows leaving a stock.
* A `stock.space` op which, when a stock has a `max` specified, returns the
  remaining quantity between the max and the actual value, after accounting for
  any outflows leaving the stock in that timestep. (This allows for bottlenecking
  a flow to not "overfill" a stock without simply throwing excess values away.)
* `>>` and `<<` handling to stocks and flows as syntax sugar for quickly
  specifying a pipeline of in and outflows, e.g.:

  ```python
  m = Model()
  with m:
    s0, s1, s2 = Stock(), Stock(), Stock()
    f0, f1 = Flow(), Flow()

    # stuff in s1 is moved to s0 and s2 through f0 and f1
    s0 << f0 << s1 >> f1 >> s2
  ```




## [0.4.0] - 2025-11-13

### Added

* Models can be used as context managers similar to PyMC, any variables/submodels
  defined within the `with` block will automatically be added to the model:

  ```python
  my_model = reno.Model()
  with my_model:
    my_variable = reno.Variable(5)
  assert my_variable.name == "my_variable"
  assert my_variable.model == my_model
  assert my_variable == my_model.my_variable
  ```

### Changed

* `compile_kwargs` handling in pymc calls to allow for older pymc versions
  (`pymc==5.12.0` should now be supported)




## [0.3.0] - 2025-11-07

### Added

* Dynamic history indexing, index equations for `HistoricalValue` no longer need
  to follow a strict `t - [static_equation]` format
* `astype` op, to convert `float` -> `int` or `int` -> `float` etc.
* Per-timestep and additional data dimension support to the remaining
  distribution operations




## [0.2.0] - 2025-10-24

### Added

* `implicit` property to `TrackedReferences` - these don't show up in diagrams
  or equation lists, but are still calculated during simulation/bayesian
  inference
* `ExtendedOperation` base class, this is an `Operation` that's built with
  internal `implicit` components
* `pulse`, `repeated_pulse`, `step` operations
* `delay1`, `delay3`, `smooth` extended operations
* Explicit `dtype` and `shape` properties to `EquationPart`
* An additional data dimension through the `dim` attribute on
  `TrackedReferences`, allowing the value at each timestep to be a vector rather
  than a single value
* A `plot_refs_single_axis` visualization function that will graph all
  references on the same set of axes each with their own ranges
* Ability for distributions to generate per timestep rather than singular static
  values or vectors

### Changed

* All `eval` functions now default to `t=0` if unspecified (allows non-time-based
  equations to run a bit more ergonomically)
* Moved `PostMeasurement` implementation to base `Metric` class, as it's the
  most basic/obvious type of metric to use
* Aggregate operations to operate in the data dimension by default, the new
  `timeseries` property/operation on tracked references can be used to apply
  aggregate operations (e.g. within metrics) across the time dimension

### Fixed

*  Model's `.pymc` function not correctly using the passed number of timesteps
* Reference values incorrectly being converted to floats when didn't start off
  as floats

### Removed

* Support for python 3.10 (exception `add_note()` functionality is only in >=3.11)




## [0.1.1] - 2025-10-14

Set python version requirement to <3.14




## [0.1.0] - 2025-10-13

First open source release!
