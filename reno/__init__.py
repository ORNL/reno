"""
System dynamics library
=======================

Library for constructing, debugging, visualizing, and analyzing system
dynamics models and simulations.

>>> import reno as r
>>> t = r.TimeRef()
>>> tub = r.Model("tub")
>>> with tub:
>>>     faucet = r.Flow(r.Scalar(5))
>>>     drain = r.Flow(r..sin(t) + 2)
>>>     water_level = r.Stock()
>>>     faucet >> water_level >> drain
>>> tub()
"""  # noqa: D212, D415

import warnings

import reno.components
import reno.diagrams
import reno.explorer
import reno.model
import reno.ops
import reno.pymc
import reno.utils
import reno.viz
from reno.components import (
    Flag,
    Flow,
    Function,
    Metric,
    Piecewise,
    Scalar,
    Stock,
    TimeRef,
    Variable,
)
from reno.model import Model
from reno.ops import (
    Bernoulli,
    Categorical,
    DiscreteUniform,
    List,
    Normal,
    Observation,
    Uniform,
    abs,
    add,
    assign,
    astype,
    bool_and,
    bool_or,
    clip,
    delay1,
    delay3,
    diff,
    div,
    eq,
    gt,
    gte,
    ifelse,
    index,
    inflow,
    interpolate,
    log,
    lt,
    lte,
    maximum,
    mean,
    minimum,
    mod,
    mul,
    nanindex,
    ne,
    nonzero,
    orient_timeseries,
    outflows,
    pow,
    pulse,
    repeated_pulse,
    series_max,
    series_min,
    sin,
    slice,
    smooth,
    stack,
    step,
    sub,
    sum,
)
from reno.viz import (
    plot_refs,
    plot_refs_single_axis,
    plot_trace_refs,
)

warnings.simplefilter("always", RuntimeWarning)

__version__ = "0.12.0"

__all__ = [
    "Bernoulli",
    "Categorical",
    "DiscreteUniform",
    "Flag",
    "Flow",
    "Function",
    "List",
    "Metric",
    "Model",
    "Normal",
    "Observation",
    "Piecewise",
    "Scalar",
    "Stock",
    "TimeRef",
    "Uniform",
    "Variable",
    "abs",
    "add",
    "assign",
    "astype",
    "bool_and",
    "bool_or",
    "clip",
    "components",
    "delay1",
    "delay3",
    "diagrams",
    "diff",
    "div",
    "eq",
    "explorer",
    "gt",
    "gte",
    "ifelse",
    "index",
    "inflow",
    "interpolate",
    "log",
    "lt",
    "lte",
    "maximum",
    "mean",
    "minimum",
    "mod",
    "model",
    "mul",
    "nanindex",
    "ne",
    "nonzero",
    "ops",
    "orient_timeseries",
    "outflows",
    "plot_refs",
    "plot_refs_single_axis",
    "plot_trace_refs",
    "pow",
    "pulse",
    "pymc",
    "repeated_pulse",
    "series_max",
    "series_min",
    "sin",
    "slice",
    "smooth",
    "stack",
    "step",
    "sub",
    "sum",
    "utils",
    "viz",
]
