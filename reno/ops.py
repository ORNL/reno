"""Classes for math operations, these are used to build up symbolic equations,
similar in principle to what something like PyTensor is doing."""

import math
import warnings

import numpy as np
import pymc as pm
import pytensor.tensor as pt

import reno

__all__ = [
    # -- series math operations --
    "series_max",
    "series_min",
    "sum",
    "index",
    "slice",
    # -- normal math operations --
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "abs",
    "lt",
    "lte",
    "gt",
    "gte",
    "eq",
    "ne",
    "bool_and",
    "bool_or",
    "minimum",
    "maximum",
    "clip",
    "log",
    "sin",
    "interpolate",
    "assign",
    # -- distributions --
    "Normal",
    "Uniform",
    "DiscreteUniform",
    "Bernoulli",
    "Categorical",
    "List",
    "Observation",
    # "Sweep",
]


# ==================================================
# SERIES MATH OPERATIONS
# ==================================================
# These are primarily meant for metric equations that run after a
# simulation and operate on a whole timeseries at a time, rather
# than calculating for a single timestep


# NOTE: can't use 'proper' name of max because TrackedReferences already have a
# max (equation max) which I don't want to rename.
class series_max(reno.components.Operation):
    """Maximum value throughout time series. Effectively a row-wise np.max."""

    OP_REPR = "max"

    def __init__(self, a):
        super().__init__(a)

    def latex(self, **kwargs):
        return f"\\text{{max}}({self.sub_equation_parts[0].latex(**kwargs)})"

    def op_eval(self, **kwargs):
        value = self.sub_equation_parts[0].value
        if value is None:
            value = self.sub_equation_parts[0].eval(**kwargs)
        return np.max(value, axis=1)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.max(self.sub_equation_parts[0].pt(**refs))

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.max({self.sub_equation_parts[0].pt_str(**refs)})"


# NOTE: can't use 'proper' name of min because TrackedReferences already have a
# min (equation min) which I don't want to rename.
class series_min(reno.components.Operation):
    """Minimum value throughout time series. Effectively a row-wise np.min."""

    OP_REPR = "min"

    def __init__(self, a):
        super().__init__(a)

    def latex(self, **kwargs):
        return f"\\text{{min}}({self.sub_equation_parts[0].latex(**kwargs)})"

    def op_eval(self, **kwargs):
        value = self.sub_equation_parts[0].value
        if value is None:
            value = self.sub_equation_parts[0].eval(**kwargs)
        return np.min(value, axis=1)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.min(self.sub_equation_parts[0].pt(**refs))

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.min({self.sub_equation_parts[0].pt_str(**refs)})"


class sum(reno.components.Operation):
    """Series-wise sum (e.g. row-wise if a matrix)."""

    # NOTE: sums of static values won't give you a multiple by timesteps by default, e.g.
    # if you have a static variable with value 5, the variable.sum() will return 5. In order
    # to expand a static value based on the timeseries, access a slice of the variable. For
    # instance, to get the full series of static values, use variable[:].sum()

    # NOTE: NOTE: nope, since changed to make operand automatically a slice if it's static.
    # Remember that this operation is really only meant for metrics, use history components
    # for equations within the stocks/flows

    def __init__(self, a):
        if a.is_static() and not isinstance(a, reno.ops.slice):
            super().__init__(reno.ops.slice(a))
        else:
            super().__init__(a)

    def latex(self, **kwargs):
        return f"\\Sigma {self.sub_equation_parts[0].latex(**kwargs)}"

    def op_eval(self, **kwargs):
        value = self.sub_equation_parts[0].value
        if value is None:
            value = self.sub_equation_parts[0].eval(**kwargs)
        # a 2d input is assumed at this point because of the slice insertion in
        # init for statics
        return np.sum(value, axis=1)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.sum(self.sub_equation_parts[0].pt(**refs))

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.sum({self.sub_equation_parts[0].pt_str(**refs)})"


# TODO: figure out how to make this work with history instead?
class index(reno.components.Operation):
    """Get a previous value in the time series at specified index, only works for tracked references
    inside of equations for metrics."""

    def __init__(self, a, ind):
        super().__init__(a, ind)

    def latex(self, **kwargs) -> str:
        return f"{self.sub_equation_parts[0].latex(**kwargs)}[{self.sub_equation_parts[1].latex(**kwargs)}]"

    def op_eval(self, **kwargs):
        # TODO: support for static?
        value = self.sub_equation_parts[0].value
        if value is None:
            value = self.sub_equation_parts[0].eval(**kwargs)
        return value[:, self.sub_equation_parts[1].value]  # TODO: eval sub_eq_parts[1]?

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs)[
            self.sub_equation_parts[1].pt(**refs)
        ]

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"{self.sub_equation_parts[0].pt_str(**refs)}[{self.sub_equation_parts[1].pt_str(**refs)}]"


class slice(reno.components.Operation):
    """Get a slice (range from index start to index stop) of values from timeseries.
    NOTE: only works for tracked references inside of equations for metrics."""

    def __init__(self, a, start=None, stop=None):
        self.start = start
        self.stop = stop

        if self.start is not None:
            # keep these ensure_scalars since start/stop stored
            # on this class, and the ensure_scalar applied to sub_equation_parts
            # in EquationPart obv won't magically apply to these
            self.start = reno.utils.ensure_scalar(self.start)

        if self.stop is not None:
            self.stop = reno.utils.ensure_scalar(self.stop)

        operands = [a]
        if start is not None:
            operands.append(self.start)
        if stop is not None:
            operands.append(self.stop)
        super().__init__(*operands)

    def latex(self, **kwargs) -> str:
        start = "" if self.start is None else self.start.latex(**kwargs)
        stop = "" if self.stop is None else self.stop.latex(**kwargs)
        return f"{self.sub_equation_parts[0].latex(**kwargs)}[{start}:{stop}]"

    # TODO: this probably won't work for row_indices?
    def op_eval(self, t, **kwargs):
        t_0 = 0 if self.start is None else self.start.eval(t, **kwargs)
        t_n = t + 1 if self.stop is None else self.stop.eval(t, **kwargs)
        # we need t + 1 because otherwise .eval(0) of a static with no stop
        # would be an empty array

        # when we have to calculate the indices, they result in arrays
        if isinstance(t_0, np.ndarray):
            t_0 = t_0[0]
        if isinstance(t_n, np.ndarray):
            t_n = t_n[0]
        count = t_n - t_0

        value = self.sub_equation_parts[0].value
        if value is None:
            value = self.sub_equation_parts[0].eval(t, **kwargs)

        # if it's static/vector instead of matrix, we just repeat by timesteps
        if isinstance(value, (float, int)):
            return np.array([np.array([value]).repeat(count)])
        if len(value.shape) == 1:
            return value[:, None].repeat(count, axis=1)
        return value[:, t_0:t_n]

    # NOTE: generally dealing with arrays instead of matrices throughout
    # pytensor because backend is dealing with whatever optimizations for sample
    # math, rather than Reno's math system doing batch math on matrices.
    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        if self.sub_equation_parts[0].is_static():
            seq_length = refs["__PT_SEQ_LEN__"]

            t_0 = 0 if self.start is None else self.start.pt(**refs)
            t_n = seq_length if self.stop is None else self.stop.pt(**refs)
            count = t_n - t_0

            return pt.repeat(self.sub_equation_parts[0].pt(**refs), count)

        if self.start is None and self.stop is None:
            return self.sub_equation_parts[0].pt(**refs)
        t_0 = pt.as_tensor(0) if self.start is None else self.start.pt(**refs)
        if self.stop is None:
            return self.sub_equation_parts[0].pt(**refs)[t_0:]
        return self.sub_equation_parts[0].pt(**refs)[t_0 : self.stop.pt(**refs)]

    def pt_str(self, **refs: dict[str, str]) -> str:
        if self.sub_equation_parts[0].is_static():
            seq_length = refs["__PT_SEQ_LEN__"]

            count_str = (
                str(seq_length) if self.stop is None else self.stop.pt_str(**refs)
            )
            if self.start is not None:
                count_str = f"({count_str} - {self.start.pt_str(**refs)})"

            return (
                f"(pt.repeat({self.sub_equation_parts[0].pt_str(**refs)}, {count_str})"
            )
        if self.start is None and self.stop is None:
            return f"{self.sub_equation_parts[0].pt_str(**refs)}"
        t0 = "pt.as_tensor(0)" if self.start is None else self.start.pt_str(**refs)
        if self.stop is None:
            return f"{self.sub_equation_parts[0].pt_str(**refs)}[{t0}:]"
        return f"{self.sub_equation_parts[0].pt_str(**refs)}[{t0}:{self.stop.pt_str(**refs)}]"


# ==================================================
# NORMAL MATH OPERATIONS
# ==================================================


class add(reno.components.Operation):
    """a + b"""

    OP_REPR = "+"

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"{self.sub_equation_parts[0].latex(**kwargs)} + {self.sub_equation_parts[1].latex(**kwargs)}"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs) + self.sub_equation_parts[
            1
        ].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs) + self.sub_equation_parts[1].pt(
            **refs
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"({self.sub_equation_parts[0].pt_str(**refs)} + {self.sub_equation_parts[1].pt_str(**refs)})"


class sub(reno.components.Operation):
    """a - b"""

    OP_REPR = "-"

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"{self.sub_equation_parts[0].latex(**kwargs)} - {self.sub_equation_parts[1].latex(**kwargs)}"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs) - self.sub_equation_parts[
            1
        ].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs) - self.sub_equation_parts[1].pt(
            **refs
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"({self.sub_equation_parts[0].pt_str(**refs)} - {self.sub_equation_parts[1].pt_str(**refs)})"


class mul(reno.components.Operation):
    """a * b"""

    OP_REPR = "*"

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"{self.sub_equation_parts[0].latex(**kwargs)} * {self.sub_equation_parts[1].latex(**kwargs)}"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs) * self.sub_equation_parts[
            1
        ].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs) * self.sub_equation_parts[1].pt(
            **refs
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"({self.sub_equation_parts[0].pt_str(**refs)} * {self.sub_equation_parts[1].pt_str(**refs)})"


class div(reno.components.Operation):
    """a / b"""

    OP_REPR = "/"

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"\\frac{{{self.sub_equation_parts[0].latex(**kwargs)}}}{{{self.sub_equation_parts[1].latex(**kwargs)}}}"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs) / self.sub_equation_parts[
            1
        ].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs) / self.sub_equation_parts[1].pt(
            **refs
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"({self.sub_equation_parts[0].pt_str(**refs)} / {self.sub_equation_parts[1].pt_str(**refs)})"


class mod(reno.components.Operation):
    """a % b"""

    OP_REPR = "%"

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"{self.sub_equation_parts[0].latex(**kwargs)} \\% {self.sub_equation_parts[1].latex(**kwargs)}"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs) % self.sub_equation_parts[
            1
        ].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.mod(
            self.sub_equation_parts[0].pt(**refs), self.sub_equation_parts[1].pt(**refs)
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.mod({self.sub_equation_parts[0].pt_str(**refs)}, {self.sub_equation_parts[1].pt_str(**refs)})"


class abs(reno.components.Operation):
    """|a| (absolute value)"""

    def __init__(self, a):
        super().__init__("abs", a)

    def latex(self, **kwargs):
        return f"|{self.sub_equation_parts[0].latex(**kwargs)}|"

    def op_eval(self, **kwargs):
        return np.abs(self.sub_equation_parts[0].eval(**kwargs))

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.abs(self.sub_equation_parts[0].pt(**refs))

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.abs({self.sub_equation_parts[0].pt_str(**refs)})"


class lt(reno.components.Operation):
    """a < b"""

    OP_REPR = "<"

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"{self.sub_equation_parts[0].latex(**kwargs)} < {self.sub_equation_parts[1].latex(**kwargs)}"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs) < self.sub_equation_parts[
            1
        ].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs) < self.sub_equation_parts[1].pt(
            **refs
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"({self.sub_equation_parts[0].pt_str(**refs)} < {self.sub_equation_parts[1].pt_str(**refs)})"


class lte(reno.components.Operation):
    """a <= b"""

    OP_REPR = "<="

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"{self.sub_equation_parts[0].latex(**kwargs)} \\leq {self.sub_equation_parts[1].latex(**kwargs)}"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs) <= self.sub_equation_parts[
            1
        ].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs) <= self.sub_equation_parts[1].pt(
            **refs
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"({self.sub_equation_parts[0].pt_str(**refs)} <= {self.sub_equation_parts[1].pt_str(**refs)})"


class gt(reno.components.Operation):
    """a > b"""

    OP_REPR = ">"

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"{self.sub_equation_parts[0].latex(**kwargs)} > {self.sub_equation_parts[1].latex(**kwargs)}"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs) > self.sub_equation_parts[
            1
        ].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs) > self.sub_equation_parts[1].pt(
            **refs
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"({self.sub_equation_parts[0].pt_str(**refs)} > {self.sub_equation_parts[1].pt_str(**refs)})"


class gte(reno.components.Operation):
    """a >= b"""

    OP_REPR = ">="

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"{self.sub_equation_parts[0].latex(**kwargs)} \\geq {self.sub_equation_parts[1].latex(**kwargs)}"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs) >= self.sub_equation_parts[
            1
        ].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs) >= self.sub_equation_parts[1].pt(
            **refs
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"({self.sub_equation_parts[0].pt_str(**refs)} >= {self.sub_equation_parts[1].pt_str(**refs)})"


class eq(reno.components.Operation):
    """a == b"""

    OP_REPR = "=="

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"{self.sub_equation_parts[0].latex(**kwargs)} = {self.sub_equation_parts[1].latex(**kwargs)}"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs) == self.sub_equation_parts[
            1
        ].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.eq(
            self.sub_equation_parts[0].pt(**refs),
            self.sub_equation_parts[1].pt(**refs),
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.eq({self.sub_equation_parts[0].pt_str(**refs)}, {self.sub_equation_parts[1].pt_str(**refs)})"


class ne(reno.components.Operation):
    """a != b"""

    OP_REPR = "!="

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"{self.sub_equation_parts[0].latex(**kwargs)} \\neq {self.sub_equation_parts[1].latex(**kwargs)}"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs) != self.sub_equation_parts[
            1
        ].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.neq(
            self.sub_equation_parts[0].pt(**refs),
            self.sub_equation_parts[1].pt(**refs),
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.ne({self.sub_equation_parts[0].pt_str(**refs)}, {self.sub_equation_parts[1].pt_str(**refs)})"


# TODO: rename to just and?
class bool_and(reno.components.Operation):
    """a and b"""

    OP_REPR = "and"

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"{self.sub_equation_parts[0].latex(**kwargs)} \\text{{ and }} {self.sub_equation_parts[1].latex(**kwargs)}"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs) & self.sub_equation_parts[
            1
        ].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs) & self.sub_equation_parts[1].pt(
            **refs
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"({self.sub_equation_parts[0].pt_str(**refs)} & {self.sub_equation_parts[1].pt_str(**refs)})"


# TODO: rename to just or?
class bool_or(reno.components.Operation):
    """a or b"""

    OP_REPR = "or"

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"{self.sub_equation_parts[0].latex(**kwargs)} \\text{{ or }} {self.sub_equation_parts[1].latex(**kwargs)}"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs) | self.sub_equation_parts[
            1
        ].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs) | self.sub_equation_parts[1].pt(
            **refs
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"({self.sub_equation_parts[0].pt_str(**refs)} | {self.sub_equation_parts[1].pt_str(**refs)})"


class minimum(reno.components.Operation):
    """Element-wise minimum of array elements between two arrays or values, same as np.minimum."""

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"\\text{{min}}({self.sub_equation_parts[0].latex(**kwargs)}, {self.sub_equation_parts[1].latex(**kwargs)})"

    def op_eval(self, **kwargs):
        return np.minimum(
            self.sub_equation_parts[0].eval(**kwargs),
            self.sub_equation_parts[1].eval(**kwargs),
        )

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.minimum(
            self.sub_equation_parts[0].pt(**refs),
            self.sub_equation_parts[1].pt(**refs),
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.minimum({self.sub_equation_parts[0].pt_str(**refs)}, {self.sub_equation_parts[1].pt_str(**refs)})"


class maximum(reno.components.Operation):
    """Element-wise maximum of array elements between two arrays or values, same as np.maximum."""

    def __init__(self, a, b):
        super().__init__(a, b)

    def latex(self, **kwargs):
        return f"\\text{{maximum}}({self.sub_equation_parts[0].latex(**kwargs)}, {self.sub_equation_parts[1].latex(**kwargs)})"

    def op_eval(self, **kwargs):
        return np.maximum(
            self.sub_equation_parts[0].eval(**kwargs),
            self.sub_equation_parts[1].eval(**kwargs),
        )

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.maximum(
            self.sub_equation_parts[0].pt(**refs),
            self.sub_equation_parts[1].pt(**refs),
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.maximum({self.sub_equation_parts[0].pt_str(**refs)}, {self.sub_equation_parts[1].pt_str(**refs)})"


class clip(reno.components.Operation):
    """Simultaneously apply upper and lower bound constraint (element-wise)."""

    def __init__(self, a, b, c):
        super().__init__(a, b, c)

    def latex(self, **kwargs):
        return f"\\text{{clip}}({self.sub_equation_parts[0].latex(**kwargs)}, {self.sub_equation_parts[1].latex(**kwargs)}, {self.sub_equation_parts[2].latex(**kwargs)})"

    def op_eval(self, **kwargs):
        return np.clip(
            self.sub_equation_parts[0].eval(**kwargs),
            self.sub_equation_parts[1].eval(**kwargs),
            self.sub_equation_parts[2].eval(**kwargs),
        )

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.clip(
            self.sub_equation_parts[0].pt(**refs),
            self.sub_equation_parts[1].pt(**refs),
            self.sub_equation_parts[2].pt(**refs),
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.clip({self.sub_equation_parts[0].pt_str(**refs)}, {self.sub_equation_parts[1].pt_str(**refs)}, {self.sub_equation_parts[2].pt_str(**refs)})"


class log(reno.components.Operation):
    """ln(a) (natural log, naming it log because this is pytensor's and numpy's default)"""

    def __init__(self, a):
        super().__init__(a)

    def latex(self, **kwargs):
        return f"\\text{{ln}}({self.sub_equation_parts[0].latex(**kwargs)})"

    def op_eval(self, **kwargs):
        return np.log(self.sub_equation_parts[0].eval(**kwargs))

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.log(self.sub_equation_parts[0].pt(**refs))

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.log({self.sub_equation_parts[0].pt_str(**refs)})"


class sin(reno.components.Operation):
    """sin(a)"""

    def __init__(self, a):
        super().__init__(reno.utils.ensure_scalar(a))

    def latex(self, **kwargs):
        return f"\\text{{sin}}({self.sub_equation_parts[0].latex(**kwargs)})"

    def op_eval(self, **kwargs):
        return np.sin(self.sub_equation_parts[0].eval(**kwargs))

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.sin(self.sub_equation_parts[0].pt(**refs))

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.sin({self.sub_equation_parts[0].pt_str(**refs)})"


# TODO: all the other trig functions


class interpolate(reno.components.Operation):
    """Given a dataset of x -> y datapoints, interpolate any new data along the line formed by the points.
    Equivalent to numpy's interp function.

    Args:
        x: The input x-coordinates that you want interpolated into y outputs.
        x_data: The x-coordinate data to base interpolation on.
        y_data: the y-coordinate data to base interpolation on.
    """

    def __init__(self, x, x_data: list | np.ndarray, y_data: list | np.ndarray):
        super().__init__(x, x_data, y_data)

    def latex(self, **kwargs):
        return f"\\text{{interpolate}}({self.sub_equation_parts[0].latex(**kwargs)}, {self.sub_equation_parts[1].latex(**kwargs)}, {self.sub_equation_parts[2].latex(**kwargs)})"

    def op_eval(self, **kwargs):
        return np.interp(
            self.sub_equation_parts[0].eval(**kwargs),
            self.sub_equation_parts[1].eval(**kwargs),
            self.sub_equation_parts[2].eval(**kwargs),
        )

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.interpolate1d(
            self.sub_equation_parts[1].pt(**refs),
            self.sub_equation_parts[2].pt(**refs),
            extrapolate=False,
        )(self.sub_equation_parts[0].pt(**refs))

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.interpolate1d({self.sub_equation_parts[1].pt_str(**refs)}, {self.sub_equation_parts[2].pt_str(**refs)},)({self.sub_equation_parts[0].pt_str(**refs)})"


class assign(reno.components.Operation):
    """This is to handle the weird seek_refs issues when you just set a tracked ref's
    equation to another tracked ref. By "wrapping" it in an effectively blank operation,
    this mitigates the annoying recursion issue."""

    OP_REPR = "="

    def __init__(self, a):
        super().__init__(a)

    def latex(self, **kwargs):
        return self.sub_equation_parts[0].latex(**kwargs)

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs)

    def pt_str(self, **refs: dict[str, str]) -> str:
        return self.sub_equation_parts[0].pt_str(**refs)


# ==================================================
# DISTRIBUTIONS
# ==================================================


class Normal(reno.components.Distribution):
    def __init__(self, mean, std=1.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def latex(self, **kwargs):
        return f"\\mathcal{{N}}({self.mean}, {self.std}^2)"

    def populate(self, n):
        self.value = np.random.normal(self.mean, self.std, size=(n,))

    def __repr__(self):
        return f"Normal({self.mean}, {self.std})"

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        name = "dist" + str(id(self))
        if "__PTNAME__" in refs:
            name = refs["__PTNAME__"]
        return pm.Normal(name, self.mean, self.std)

    def pt_str(self, **refs: dict[str, str]) -> str:
        name = "dist" + str(id(self))
        if "__PTNAME__" in refs:
            name = refs["__PTNAME__"]
        return f'pm.Normal("{name}", {self.mean}, {self.std})'


class Uniform(reno.components.Distribution):
    def __init__(self, low=0.0, high=1.0):
        super().__init__()
        self.low = low
        self.high = high

    def latex(self, **kwargs):
        return f"\\mathcal{{U}}({self.low}, {self.high})"

    def populate(self, n):
        self.value = np.random.uniform(self.low, self.high, size=(n,))

    def __repr__(self):
        return f"Uniform({self.low}, {self.high})"

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        name = "dist" + str(id(self))
        if "__PTNAME__" in refs:
            name = refs["__PTNAME__"]
        return pm.Uniform(name, self.low, self.high)

    def pt_str(self, **refs: dict[str, str]) -> str:
        name = "dist" + str(id(self))
        if "__PTNAME__" in refs:
            name = refs["__PTNAME__"]
        return f'pm.Uniform("{name}", {self.low}, {self.high})'


class DiscreteUniform(reno.components.Distribution):
    """Low is inclusive, high is exclusive."""

    def __init__(self, low: int = 0, high: int = 2):
        super().__init__()
        self.low = low
        self.high = high

    def latex(self, **kwargs):
        return f"\\text{{DiscreteUniform}}({self.low}, {self.high})"

    def populate(self, n: int):
        self.value = np.random.randint(self.low, self.high, n)

    def __repr__(self):
        return f"DiscreteUniform({self.low}, {self.high})"

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        name = "dist" + str(id(self))
        if "__PTNAME__" in refs:
            name = refs["__PTNAME__"]
        return pm.DiscreteUniform(name, self.low, self.high)

    def pt_str(self, **refs: dict[str, str]) -> str:
        name = "dist" + str(id(self))
        if "__PTNAME__" in refs:
            name = refs["__PTNAME__"]
        return f'pm.DiscreteUniform("{name}", {self.low}, {self.high})'


class Bernoulli(reno.components.Distribution):
    """Discrete single event probability (p is probability of eval == 1)"""

    def __init__(self, p: float, use_p_dist: bool = False):
        super().__init__()
        self.p = p
        self.use_p_dist = use_p_dist

    def latex(self, **kwargs):
        return f"\\text{{Bernoulli}}({self.p})"

    def populate(self, n):
        self.value = np.random.binomial(1, self.p, n)

    def __repr__(self):
        return f"Bernoulli({self.p})"

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        name = "dist" + str(id(self))
        if "__PTNAME__" in refs:
            name = refs["__PTNAME__"]
        inner_dist = self.p
        if self.use_p_dist:
            inner_dist = pm.Interpolated(
                f"{name}_p",
                x_points=np.array([0.0, 1.0]),
                pdf_points=np.array([1 - self.p, self.p]),
            )
        return pm.Bernoulli(name, inner_dist)

    def pt_str(self, **refs: dict[str, str]) -> str:
        name = "dist" + str(id(self))
        if "__PTNAME__" in refs:
            name = refs["__PTNAME__"]
        inner_dist = self.p
        if self.use_p_dist:
            inner_dist = f'pm.Interpolated("{name}_p", x_points=np.array([0.0, 1.0]), pdf_points=np.array([{1 - self.p}, {self.p}]))'
        return f'pm.Bernoulli("{name}", {inner_dist})'


class Categorical(reno.components.Distribution):
    def __init__(self, p: list[float], use_p_dist: bool = False):
        super().__init__()
        self.p = p
        self.use_p_dist = use_p_dist

    def latex(self, **kwargs):
        return f"\\text{{Categorical}}({self.p})"

    def populate(self, n):
        self.value = np.argmax(np.random.multinomial(1, self.p, n), axis=1)

    def __repr__(self):
        return f"Categorical({self.p})"

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        name = "dist" + str(id(self))
        if "__PTNAME__" in refs:
            name = refs["__PTNAME__"]
        inner_dist = self.p
        if self.use_p_dist:
            inner_dist = pm.Dirichlet(f"{name}_p", self.p)
        return pm.Categorical(name, inner_dist)

    def pt_str(self, **refs: dict[str, str]) -> str:
        name = "dist" + str(id(self))
        if "__PTNAME__" in refs:
            name = refs["__PTNAME__"]
        inner_dist = self.p
        if self.use_p_dist:
            inner_dist = f'pm.Dirichlet("{name}_p", {self.p})'
        return f'pm.Categorical("{name}", {inner_dist})'


class List(reno.components.Distribution):
    """Tile passed list to the sample size so each value is hit roughly
    equally (dependent on exact sample size) and deterministically"""

    def __init__(self, values: list | np.ndarray | set):
        super().__init__()
        self.values = values

    def latex(self, **kwargs):
        return f"{self.values}"

    def populate(self, n):
        repetitions = n / len(self.values)
        # if the specified value is _larger_ than the samples, we have to
        # truncate (and warn, does the user know this is what's happening?)
        if repetitions < 1:
            warnings.warn(
                f"Not enough samples in simulation to hit every value in list '{self.values}', would require at least `n={len(self.values)}`",
                RuntimeWarning,
            )
            expanded = np.array(self.values)
        else:
            expanded = np.tile(self.values, math.ceil(repetitions))
        self.value = expanded[:n]

    def __repr__(self):
        return f"List({self.values})"


class Observation(reno.components.Distribution):
    """Represents a Normal distribution around an observed value.

    Should only be used for supplying observational data with likelihoods
    to bayesian models constructed with model.pymc()

    Args:
        ref (reno.components.Reference): The equation to supply an observed value for.
        sigma (float): The std dev to use for the likelihood Normal distribution.
        data (list): The actual observed data to apply.
    """

    def __init__(
        self, ref: reno.components.Reference, sigma: float = 1.0, data: list = None
    ):
        super().__init__()
        self.ref = ref
        self.sigma = sigma
        self.data = data

    def add_tensors(self, pymc_model):
        with pymc_model:
            # sigma = pm.HalfNormal(f"{self.ref.qual_name()}_sigma", self.sigma)
            pm.Normal(
                f"{self.ref.qual_name()}_likelihood",
                pymc_model[self.ref.qual_name()],
                self.sigma,
                observed=self.data,
            )

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pm.Normal(
            f"{self.ref.qual_name()}_likelihood",
            self.ref.pt(**refs),
            self.sigma,
            observed=self.data,
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f'pm.Normal("{self.ref.qual_name()}_likelihood", {self.ref.pt_str(**refs)}, {self.sigma}, observed={self.data})'


# class Sweep(reno.components.Distribution):
#     """Similar in principle to ops.List, unimplemented idea for this is to
#     grab all variables that are sweeps and collectively make sure they are fully
#     permutated, rather than requiring user to manually make sure they change at
#     varying rates between samples."""
#
#     # TODO: semantically tells to ensure that for _all_ sweep dists
#     # in a system, make sure all value combinations are hit (if size allows, warn if
#     # doesn't)
#     def __init__(self, sweep_values: list | np.ndarray | set):
#         super().__init__()
#         self.sweep_values = sweep_values
#
#     def latex(self, **kwargs):
#         return f"\\text{{Sweep}}({self.sweep_values})"
#
#     def populate(self, n):
#         # TODO: this will be difficult to implement, need to find all other
#         # sweeps in system? We have no ref to that
#         pass
#
#     def __repr__(self):
#         return f"Sweep({self.sweep_values})"
#
