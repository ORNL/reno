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
"""  # noqa: D205, D212, D415

import warnings

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
from reno.ops import *  # noqa: F403
from reno.viz import (
    plot_refs,
    plot_refs_single_axis,
    plot_trace_refs,
)

warnings.simplefilter("always", RuntimeWarning)

__version__ = "0.12.0"

__all__ = [
    "Flag",
    "Flow",
    "Function",
    "Metric",
    "Model",
    "Piecewise",
    "Scalar",
    "Stock",
    "TimeRef",
    "Variable",
    "plot_refs",
    "plot_refs_single_axis",
    "plot_trace_refs",
]
