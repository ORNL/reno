# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [unreleased]

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

## [0.1.1] - 2025-10-14

Set python version requirement to <3.14


## [0.1.0] - 2025-10-13

First open source release!
