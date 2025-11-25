# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [unreleased]

### Added

* `cgroup` parameter to `TrackedReference`, allowing control over display and
  coloring
* `group_colors` in `Model` to allow specifying colors to use for `cgroup` or
  `group` in diagrams
* `default_hide_groups` in `Model` to specify whether to hide specific `cgroup`
  or `group` labeled references by default in diagrams
* `inflow` operation to allow semantically stating that a flow is actually an
  inflow to another flow (only impacts diagrams, to help indicate "flow of
  material" when a particular flow should indirectly be the inflow to a stock

### Changed

* `seek_refs` can now return additional "type" information for each reference in
  an equation, allowing more fine grained control during diagramming




## [0.6.1] - 2025-11-19

### Changed

* Implicit flows having very unhelpful names, making debugging annoying and
  difficult. (They now are named according to the stock they target and which
  index inflow they are on that stock)


### Fixed

* Index operation incorrectly handling non-timeseries values
* `stack` and several element-wise operations breaking when model run with n>1
  due to numpy broadcasting issues
* Variables added to diagram from stock min/max refs even if show_vars False
* Stock sparklines inconsistent with flow sparklines in diagrams




## [0.6.0] - 2025-11-18

### Added

* Implicit inflows to Stocks. If an equation (rather than a flow) is given
  (either through `>>` or `+=` syntax) to a stock, an implicit flow wrapping
  that equation is used. (Previously, only flows were allowed to be explicitly
  used with stocks.) This allows "conversion" of the outputs of one stock to
  the inputs of another without having to explicitly create a new flow to support
  it. Example:

  ```python
  m = Model()
  with m:
      s0, s1 = Stock(), Stock()
      f0 = Flow(3)
      s0 >> f0
      (f0 - 1) >> s1
  ```
* Handling lists of inflows to Stocks. To simultaneously add multiple flows as
  inflows, pass them in a list either via `>>` or `+=` syntax:

  ```python
  m = Model()
  with m:
      s0 = Stock(), Stock()
      f0, f1 = Flow(3), Flow(2)
      [f0, f1] >> s1
  ```
* `stack` operation for creating a multidim value from multiple single dim
  values.

### Changed

* The `.timeseries` operation is now allowed for non-metric equations in both
  regular Reno and PyMC math. This should not be used in place of `.history`
  when only individual values from a reference's history is needed,
  `.timeseries` is intended for operating across multiple values in history e.g.
  summing up values across a range of historical timesteps:

  ```python
  m = Model()
  t = TimeRef()
  with m:
      v0 = Variable(t + 2)
      v1 = Variable(v0.timeseries[t-3:t-1].sum())
  ```

### Fixed

* Bugs in how `.timeseries` was handled for static variables (was inconsistent
  with an equivalent dynamic variable with the same values)




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
