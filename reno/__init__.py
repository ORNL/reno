"""
System dynamics library
=======================

Library for constructing, debugging, visualizing, and analyzing system
dynamics models and simulations.

>>> from reno.components import Stock, Flow, Variable, Scalar, TimeRef
>>> from reno.model import Model
>>> from reno import ops
>>> t = TimeRef()
>>> tub = Model("tub")
>>> tub.faucet = Flow(Scalar(5))
>>> tub.drain = Flow(ops.sin(t) + 2)
>>> tub.water_level = Stock()
>>> tub.water_level += tub.faucet
>>> tub.water_level -= tub.drain
>>> tub()
"""

import warnings

import reno.components  # noqa: F401
import reno.diagrams  # noqa: F402
import reno.explorer  # noqa: F401
import reno.model  # noqa: F401
import reno.ops  # noqa: F401
import reno.pymc  # noqa: F401
import reno.viz  # noqa: F401
from reno.components import Flag  # noqa: F401
from reno.components import Flow  # noqa: F401
from reno.components import Function  # noqa: F401
from reno.components import Metric  # noqa: F401
from reno.components import Piecewise  # noqa: F401
from reno.components import Scalar  # noqa: F401
from reno.components import Stock  # noqa: F401
from reno.components import TimeRef  # noqa: F401
from reno.components import Variable  # noqa: F401
from reno.model import Model  # noqa: F401
from reno.ops import *  # noqa: F401, F403
from reno.viz import plot_refs  # noqa: F401
from reno.viz import plot_refs_single_axis  # noqa: F401
from reno.viz import plot_trace_refs  # noqa: F401

warnings.simplefilter("always", RuntimeWarning)

__version__ = "0.6.0"
