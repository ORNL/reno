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
    "orient_timeseries",
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
    "astype",
    "stack",
    # -- "higher order" --
    "pulse",
    "repeated_pulse",
    "step",
    "delay1",
    "delay3",
    "smooth",
    # -- meta --
    "outflows",
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

    def get_shape(self) -> int:
        return 1

    def op_eval(self, **kwargs):
        value = self.sub_equation_parts[0].value
        if value is None:
            value = self.sub_equation_parts[0].eval(**kwargs)
        axis = 0
        if len(value.shape) != 1:
            axis += 1
        return np.max(value, axis=axis)

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

    def get_shape(self) -> int:
        return 1

    def op_eval(self, **kwargs):
        value = self.sub_equation_parts[0].value
        if value is None:
            value = self.sub_equation_parts[0].eval(**kwargs)
        axis = 0
        if len(value.shape) != 1:
            axis += 1
        return np.min(value, axis=axis)

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

    def __init__(self, a, axis=0):
        self.axis = axis
        super().__init__(a)
        # if a.is_static() and not isinstance(a, reno.ops.time_slice):
        #     super().__init__(reno.ops.time_slice(a))
        # else:
        #     super().__init__(a)

    def latex(self, **kwargs):
        return f"\\Sigma {self.sub_equation_parts[0].latex(**kwargs)}"

    def get_shape(self) -> int:
        return 1

    def op_eval(self, **kwargs):
        # value = self.sub_equation_parts[0].value
        # if value is None:
        value = self.sub_equation_parts[0].eval(**kwargs)
        axis = self.axis
        if len(value.shape) != 1:
            axis = self.axis + 1  # to account for "batch"/n dimension
        return np.sum(value, axis=axis)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.sum(self.sub_equation_parts[0].pt(**refs), axis=self.axis)

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.sum({self.sub_equation_parts[0].pt_str(**refs)}, axis={self.axis})"


# TODO: figure out how to make this work with history instead?
class index(reno.components.Operation):
    """Get a previous value in the time series at specified index, only works for tracked references
    inside of equations for metrics."""

    def __init__(self, a, ind):
        super().__init__(a, ind)

    def latex(self, **kwargs) -> str:
        return f"{self.sub_equation_parts[0].latex(**kwargs)}[{self.sub_equation_parts[1].latex(**kwargs)}]"

    def get_shape(self) -> int:
        return 1

    def op_eval(self, **kwargs):
        # TODO: support for static?
        value = self.sub_equation_parts[0].value
        if value is None:
            value = self.sub_equation_parts[0].eval(**kwargs)
        # TODO: this is going to cause an issue if non-sample-dim statics used
        return value[:, self.sub_equation_parts[1].value]  # TODO: eval sub_eq_parts[1]?

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs)[
            self.sub_equation_parts[1].pt(**refs)
        ]

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"{self.sub_equation_parts[0].pt_str(**refs)}[{self.sub_equation_parts[1].pt_str(**refs)}]"


class slice(reno.components.Operation):
    """Can be applied along with timeseries op in metrics for getting specific time segments, or
    can be applied generally in equations when dealing with vector data.

    Similar to normal python slices, top is exclusive
    """

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

    # TODO: (2025.09.22) will need to think about how best to implement a
    # get_shape for this

    def op_eval(self, t, **kwargs):
        start = self.start.eval(t, **kwargs) if self.start is not None else None
        stop = self.stop.eval(t, **kwargs) if self.stop is not None else None

        if isinstance(start, np.ndarray):
            start = start[0]
        if isinstance(stop, np.ndarray):
            stop = stop[0]

        if start is None or start < 0:
            start = 0
        if stop is not None and stop < start:
            stop = start

        value = self.sub_equation_parts[0].value
        if value is None:
            value = self.sub_equation_parts[0].eval(t, **kwargs)

        # dims = []
        # Note we don't have explicit handling for statics, use of .timeseries
        # op is assumed where necessary.
        # TODO: is it cleaner to use ellipsis? https://stackoverflow.com/questions/12116830/numpy-slice-of-arbitrary-dimensions
        # for i, dim in enumerate(value.shape):
        #     if i == len(value.shape) - 1:
        #         dims.append(slice(start, stop))
        #     else:
        #         dims.append(slice(None, None))
        # print(dims)
        return value[..., start:stop]

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        if self.start is None and self.stop is None:
            return self.sub_equation_parts[0].pt(**refs)
        t_0 = (
            pt.as_tensor(0)
            if self.start is None
            else pt.maximum(pt.as_tensor(0), self.start.pt(**refs))
        )
        if self.stop is None:
            return self.sub_equation_parts[0].pt(**refs)[..., t_0:]
        return self.sub_equation_parts[0].pt(**refs)[
            ..., t_0 : pt.maximum(self.stop.pt(**refs), t_0)
        ]

    def pt_str(self, **refs: dict[str, str]) -> str:
        if self.start is None and self.stop is None:
            return f"{self.sub_equation_parts[0].pt_str(**refs)}"
        t0 = (
            "pt.as_tensor(0)"
            if self.start is None
            else f"pt.maximum(pt.as_tensor(0), {self.start.pt_str(**refs)})"
        )
        if self.stop is None:
            return f"{self.sub_equation_parts[0].pt_str(**refs)}[..., {t0}:]"
        return f"{self.sub_equation_parts[0].pt_str(**refs)}[..., {t0} : pt.maximum({self.stop.pt_str(**refs)}, {t0})]"


class orient_timeseries(reno.components.Operation):
    """Get the full data of a component/equation as an array over time, allowing aggregate
    operations to operate across the full timeseries, e.g. for metric equations. This is
    in essence "reorienting" the underlying data to include the time dimension.

    NOTE: for pytensor I suspect this will only work in metric equations. pt/pt_str are dependent
    on whether the refs are populated with the full series or not, which will only occur
    within the metrics section?
    TODO: we can _eventually_ make this work within pytensor by the approach to solving general
    history time equations - if a timeseries op detected within non-metric context, include the full
    stops of the relevant variables

    NOTE: only use this in a non-metric equation if a _range_ of timeseries values are needed. Otherwise use `.history()` for single values
    """

    # TODO: it probably doesn't make sense to call this on anything except a trackedreference,
    # may want to add checks for this.

    def __init__(self, a):
        super().__init__(a)

    def latex(self, **kwargs) -> str:
        return f"{self.sub_equation_parts[0].latex(**kwargs)}.\\text{{timeseries}}"

    # TODO: how best to get timeseries length for get_shape?

    def op_eval(self, t, **kwargs):
        value = self.sub_equation_parts[0].value
        if value is None:
            value = self.sub_equation_parts[0].eval(
                t, **kwargs
            )  # TODO: is this right?? This isn't the same as getting .value because may not include full time series...
            # do I instead need to ensure the save is true and then get .value?
            # this will only be relevant to solve when the pytensor general
            # history approach is solved

        count = t + 1
        if self.sub_equation_parts[0].is_static():
            remaining = 0
            if hasattr(self.sub_equation_parts[0], "model"):
                if self.sub_equation_parts[0].model is not None:
                    remaining = self.sub_equation_parts[0].model.steps - count
            if isinstance(value, (float, int)):
                return np.concatenate(
                    [
                        np.array([np.array([value]).repeat(count)]),
                        np.array([np.array([0]).repeat(remaining)]),
                    ],
                    axis=1,
                )
            if len(value.shape) == 1:
                return value[:, None].repeat(count, axis=1)
        return value

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        # have to have a time ref for this to work
        if self.sub_equation_parts[0].model.find_timeref_name() is None:
            self.sub_equation_parts.append(reno.components.TimeRef())

        # for static underlying values, we reconstruct an array as if that
        # underlying value were building up through an array of all zeros,
        # so ""fill"" accordingly based on timestep
        if self.sub_equation_parts[0].is_static():
            seq_length = refs["__PT_SEQ_LEN__"]
            if "__TIMESERIES__" in refs:
                if refs["__TIMESERIES__"] == "init":
                    return pt.concatenate(
                        [
                            [self.sub_equation_parts[0].pt(**refs)],
                            pt.as_tensor(0).repeat(seq_length - 1),
                        ]
                    )
                elif refs["__TIMESERIES__"] == "scan":
                    t = refs[self.sub_equation_parts[0].model.find_timeref_name()]
                    return pt.concatenate(
                        [
                            self.sub_equation_parts[0].pt(**refs).repeat(t + 1),
                            pt.as_tensor(0).repeat(seq_length - (t + 1)),
                        ]
                    )
            # in principle this should never be reached?
            return pt.repeat(self.sub_equation_parts[0].pt(**refs), seq_length)
        if "__TIMESERIES__" in refs:
            name = self.sub_equation_parts[0].qual_name()
            # NOTE: _h is what's stored in refs during init and scan,
            # assuming I still need this to ensure the post calculation
            # is done correctly with the full sequence instead of the historical
            # ones.
            if refs["__TIMESERIES__"] == "init":
                name += "_h"
                # TODO: this doesn't work if no timeref included.
                # t = refs[self.sub_equation_parts[0].model.find_timeref_name()]
                # have to concatenate to reorder, since pymc is passing in
                # history (taps) according to the current timestep.
                return pt.concatenate([refs[name][-1:], refs[name][:-1]])
            elif refs["__TIMESERIES__"] == "scan":
                name += "_h"
                # TODO: this doesn't work if no timeref included.
                t = refs[self.sub_equation_parts[0].model.find_timeref_name()]
                # have to concatenate to reorder, since pymc is passing in
                # history (taps) according to the current timestep.
                return pt.concatenate([refs[name][-(t + 1) :], refs[name][: -(t + 1)]])
            return refs[name]
        return self.sub_equation_parts[0].pt(**refs)

    def pt_str(self, **refs: dict[str, str]) -> str:
        # have to have a time ref for this to work
        if self.sub_equation_parts[0].model.find_timeref_name() is None:
            self.sub_equation_parts.append(reno.components.TimeRef())

        # for static underlying values, we reconstruct an array as if that
        # underlying value were building up through an array of all zeros,
        # so ""fill"" accordingly based on timestep
        if self.sub_equation_parts[0].is_static():
            seq_length = refs["__PT_SEQ_LEN__"]
            if "__TIMESERIES__" in refs:
                if refs["__TIMESERIES__"] == "init":
                    return f"pt.concatenate([[{self.sub_equation_parts[0].pt_str(**refs)}], pt.as_tensor(0).repeat({seq_length - 1})])"
                elif refs["__TIMESERIES__"] == "scan":
                    t = refs[self.sub_equation_parts[0].model.find_timeref_name()]
                    return f"pt.concatenate([{self.sub_equation_parts[0].pt_str(**refs)}.repeat(t + 1), pt.as_tensor(0).repeat({seq_length} - ({t} + 1))])"
            # in principle this should never be reached?
            return (
                f"pt.repeat({self.sub_equation_parts[0].pt_str(**refs)}, {seq_length})"
            )
        if "__TIMESERIES__" in refs:
            name = self.sub_equation_parts[0].qual_name()
            if refs["__TIMESERIES__"] == "init":
                name += "_h"
                # t = refs[self.sub_equation_parts[0].model.find_timeref_name()]
                return f"pt.concatenate([{refs[name]}[-1:], {refs[name]}[:-1]])"
            elif refs["__TIMESERIES__"] == "scan":
                name += "_h"
                t = refs[self.sub_equation_parts[0].model.find_timeref_name()]
                return f"pt.concatenate([{refs[name]}[-({t}+1):], {refs[name]}[:-({t}+1)]])"
        else:
            return self.sub_equation_parts[0].pt_str(**refs)


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

    def get_type(self) -> type:
        return bool

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

    def get_type(self) -> type:
        return bool

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

    def get_type(self) -> type:
        return bool

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

    def get_type(self) -> type:
        return bool

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

    def get_type(self) -> type:
        return bool

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

    def get_type(self) -> type:
        return bool

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

    def get_type(self) -> type:
        return bool

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

    def get_type(self) -> type:
        return bool

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
        return f"\\text{{max}}({self.sub_equation_parts[0].latex(**kwargs)}, {self.sub_equation_parts[1].latex(**kwargs)})"

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

    def get_type(self) -> type:
        return float

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

    def get_type(self) -> type:
        return float

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

    def get_shape(self) -> int:
        """The shapes of x_data and y_data don't matter, should be same as input."""
        return self.sub_equation_parts[0].shape

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

    def get_shape(self) -> int:
        return self.sub_equation_parts[0].shape

    def get_type(self) -> type:
        return self.sub_equation_parts[0].dtype

    def latex(self, **kwargs):
        return self.sub_equation_parts[0].latex(**kwargs)

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[0].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs)

    def pt_str(self, **refs: dict[str, str]) -> str:
        return self.sub_equation_parts[0].pt_str(**refs)


class astype(reno.components.Operation):
    """Allow explicitly converting to float, int, etc."""

    def __init__(self, a, dtype: type):
        self.__dtype = dtype
        super().__init__(a)

    def get_type(self) -> type:
        return self.__dtype

    def latex(self, **kwargs):
        return self.sub_equation_parts[0].latex(**kwargs)

    def op_eval(self, **kwargs):
        sub_val = self.sub_equation_parts[0].eval(**kwargs)
        if isinstance(sub_val, np.ndarray):
            sub_val = sub_val.astype(self.__dtype)
        else:
            sub_val = self.__dtype(sub_val)
        return sub_val

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[0].pt(**refs).astype(self.__dtype)

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"({self.sub_equation_parts[0].pt_str(**refs)}).astype({self.__dtype.__name__})"

    @staticmethod
    def parse(arg_strs: list[str], refs: dict[str, reno.components.Reference]):
        if len(arg_strs) != 2:
            raise SyntaxError(
                "astype op expects two arguments, the reference to convert, and the string type to convert to"
            )
        ref = reno.parser.parse(arg_strs[0], refs)
        if arg_strs[1] == "int":
            dtype = int
        elif arg_strs[1] == "float":
            dtype = float
        elif arg_strs[1] == "bool":
            dtype = bool
        return astype(ref, dtype)

    def __repr__(self):
        return (
            f"(astype {self.sub_equation_parts[0].__repr__()} {self.__dtype.__name__})"
        )


class stack(reno.components.Operation):
    """Stack a set of 1 dimensional values into a multidim value.

    This does not yet implement stacking multiple multidim values.
    (may need to be a separate concat operator for that)
    """

    def __init__(self, *args):
        super().__init__(*args)

    def get_shape(self) -> int:
        return len(self.sub_equation_parts)

    def get_type(self) -> type:
        types = [part.dtype for part in self.sub_equation_parts]
        if float in types:
            return float
        return types[0]

    def latex(self, **kwargs) -> str:
        building_str = (
            f"[{','.join(part.latex(**kwargs) for part in self.sub_equation_parts)}]"
        )
        return building_str

    def op_eval(self, **kwargs):
        return np.stack([part.eval(**kwargs) for part in self.sub_equation_parts])

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return pt.stack([part.pt(**refs) for part in self.sub_equation_parts])

    def pt_str(self, **refs: dict[str, str]) -> str:
        return f"pt.stack([{','.join(part.pt_str(**refs) for part in self.sub_equation_parts)}])"


# ==================================================
# "HIGHER ORDER"
# ==================================================


class pulse(reno.components.Operation):
    """Return a '1' signal for ``width`` number of timesteps starting at timestep ``start``.
    Returns 0 at all other timesteps."""

    def __init__(self, start, width=1):
        t = reno.components.TimeRef()
        self.sub_eq = reno.components.Piecewise(
            [0, 1],
            [
                (t < start) | (t >= (start + width)),
                (t >= start) & (t < (start + width)),
            ],
        )
        super().__init__(start, width, self.sub_eq)

    def latex(self, **kwargs):
        return f"\\text{{pulse}}({self.sub_equation_parts[0].latex(**kwargs)}, {self.sub_equation_parts[1].latex(**kwargs)})"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[2].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[2].pt(**refs)

    def pt_str(self, **refs: dict[str, str]) -> str:
        return self.sub_equation_parts[2].pt_str(**refs)


class repeated_pulse(reno.components.Operation):
    """Return a '1' signal for ``width`` number of timesteps starting at timestep ``start``,
    with ``interval`` number of timesteps between each subsequent leading edge. Returns 0
    at all other timesteps."""

    def __init__(self, start, interval, width=1):
        t = reno.components.TimeRef()
        self.sub_eq = reno.components.Piecewise(
            [0, 1],
            [
                (t < start) | ((t - start) % interval >= width),
                (t >= start) & ((t - start) % interval < width),
            ],
        )
        super().__init__(start, interval, width, self.sub_eq)

    def latex(self, **kwargs):
        return f"\\text{{pulse}}({self.sub_equation_parts[0].latex(**kwargs)}, {self.sub_equation_parts[1].latex(**kwargs)}, {self.sub_equation_parts[2].latex(**kwargs)})"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[3].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[3].pt(**refs)

    def pt_str(self, **refs: dict[str, str]) -> str:
        return self.sub_equation_parts[3].pt_str(**refs)


class step(reno.components.Operation):
    """Return a specified value after a specified number of timesteps, otherwise 0."""

    def __init__(self, value, timesteps):
        t = reno.components.TimeRef()
        self.sub_eq = reno.components.Piecewise(
            [0, value], [t < timesteps, t >= timesteps]
        )
        super().__init__(value, timesteps, self.sub_eq)

    def latex(self, **kwargs):
        return f"\\text{{step}}({self.sub_equation_parts[0].latex(**kwargs)}, {self.sub_equation_parts[1].latex(**kwargs)})"

    def op_eval(self, **kwargs):
        return self.sub_equation_parts[2].eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.sub_equation_parts[2].pt(**refs)

    def pt_str(self, **refs: dict[str, str]) -> str:
        return self.sub_equation_parts[2].pt_str(**refs)


class delay1(reno.components.ExtendedOperation):
    def __init__(self, input, delay_time):
        self.inflow = reno.Flow(input)
        self.delay_stock = reno.Stock()
        self.outflow = reno.Flow(self.delay_stock / delay_time)
        self.delay_stock += self.inflow
        self.delay_stock -= self.outflow
        super().__init__(
            [input, delay_time],
            {
                "inflow": self.inflow,
                "delay_stock": self.delay_stock,
                "outflow": self.outflow,
            },
        )

    def latex(self, **kwargs):
        return f"\\text{{delay1}}({self.sub_equation_parts[0].latex(**kwargs)}, {self.sub_equation_parts[1].latex(**kwargs)})"

    def op_eval(self, **kwargs):
        return self.outflow.eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.outflow.pt(**refs)

    def pt_str(self, **refs: dict[str, str]) -> str:
        return self.outflow.pt_str(**refs)


class delay3(reno.components.ExtendedOperation):
    def __init__(self, input, delay_time):
        self.delay1 = delay1(input, delay_time / 3)
        self.delay2 = delay1(self.delay1, delay_time / 3)
        self.delay3 = delay1(self.delay2, delay_time / 3)
        super().__init__([input, delay_time, self.delay3], {})
        # don't need to pass any implicit components up because the individual
        # delays should handle that.

    def latex(self, **kwargs):
        return f"\\text{{delay3}}({self.sub_equation_parts[0].latex(**kwargs)}, {self.sub_equation_parts[1].latex(**kwargs)})"

    # TODO: does this work just as delay3 or do we need to explicitly reference
    # outflow?
    def op_eval(self, **kwargs):
        return self.delay3.eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.delay3.pt(**refs)

    def pt_str(self, **refs: dict[str, str]) -> str:
        return self.delay3.pt_str(**refs)


class smooth(reno.components.ExtendedOperation):
    """An information delay, material isn't necessarily preserved but the range is?"""

    def __init__(self, input, adjustment_time, initial_value=0):
        self.actual_value = reno.Variable(input, dtype=float)
        self.output = reno.Stock(init=initial_value, dtype=float)  # "perceived" value
        self.gap = reno.Variable(self.actual_value - self.output, dtype=float)
        self.adjustment = reno.Flow(self.gap / adjustment_time, dtype=float)
        self.output += self.adjustment
        super().__init__(
            [input, adjustment_time, initial_value],
            {
                "actual_value": self.actual_value,
                "output": self.output,
                "gap": self.gap,
                "adjustment": self.adjustment,
            },
        )

    def latex(self, **kwargs):
        return f"\\text{{smooth}}({self.sub_equation_parts[0].latex(**kwargs)}, {self.sub_equation_parts[1].latex(**kwargs)}, {self.sub_equation_parts[2].latex(**kwargs)})"

    def op_eval(self, **kwargs):
        return self.output.eval(**kwargs)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return self.output.pt(**refs)

    def pt_str(self, **refs: dict[str, str]) -> str:
        return self.output.pt_str(**refs)


# ==================================================
# META
# ==================================================


class outflows(reno.components.Operation):
    def __init__(self, stock):
        # WAIT: optionally include a list of flows to exclude??
        super().__init__(stock)
        if not isinstance(stock, reno.components.Stock):
            raise TypeError(f"The outflows op must be a Stock, not a {type(stock)}")

    def seek_refs(self):
        # have to override because outflows is "substituting in" the sum of flows.
        # print("In outflows seek refs!")
        return self.sub_equation_parts[0].out_flows

    def latex(self, **kwargs) -> str:
        return f"{self.sub_equation_parts[0].latex(**kwargs)}.\\text{{outflows}}"

    def op_eval(self, **kwargs):
        return (
            self.sub_equation_parts[0]
            .combine_eqs(self.sub_equation_parts[0].out_flows)
            .eval(**kwargs)
        )

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        return (
            self.sub_equation_parts[0]
            .combine_eqs(self.sub_equation_parts[0].out_flows)
            .pt(**refs)
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        return (
            self.sub_equation_parts[0]
            .combine_eqs(self.sub_equation_parts[0].out_flows)
            .pt_str(**refs)
        )


# ==================================================
# DISTRIBUTIONS
# ==================================================


def dist_shape(
    dist: reno.components.Distribution, n: int, steps: int, dim: int
) -> int | tuple:
    """Compute the shape/dimensions needed to populate the passed distribution"""
    if not dist.per_timestep and dim == 1:
        shape = n
    elif dist.per_timestep and dim == 1:
        shape = (n, steps)
    elif dist.per_timestep and dim > 1:
        shape = (n, steps, dim)
    else:
        shape = (n, dim)
    return shape


def dist_params(
    dist: reno.components.Distribution, refs: dict
) -> tuple[str, int, str, int]:
    """Extract any dunder variables in the refs passed from pymc.py for setting up parameters
    for the pymc converted distribution."""
    name = "dist" + str(id(dist))
    if "__PTNAME__" in refs:
        name = refs["__PTNAME__"]
    dim = 1
    if "__DIM__" in refs:
        dim = refs["__DIM__"]
    dim_name = "vec"
    if "__DIMNAME__" in refs:
        dim_name = refs["__DIMNAME__"]
    seq = 0
    if "__PT_SEQ_LEN__" in refs:
        seq = refs["__PT_SEQ_LEN__"]
    return name, dim, dim_name, seq


def dist_dim_shape_args(dist: reno.components.Distribution, refs: dict) -> dict:
    name, dim, dim_name, seq = dist_params(dist, refs)
    if not dist.per_timestep and dim == 1:
        return dict()
    elif dist.per_timestep and dim == 1:
        return dict(shape=(seq,), dims="t")
    elif dist.per_timestep and dim > 1:
        return dict(shape=(seq, dim), dims=("t", dim_name))
    else:
        return dict(shape=(dim,), dims=dim_name)


def dist_dim_shape_args_str(
    dist: reno.components.Distribution, refs: dict[str, str]
) -> str:
    name, dim, dim_name, seq = dist_params(dist, refs)
    if not dist.per_timestep and dim == 1:
        return ""
    elif dist.per_timestep and dim == 1:
        return f', shape=({seq},), dims="t"'
    elif dist.per_timestep and dim > 1:
        return f', shape=({seq},, {dim}), dims=("t", "{dim_name}")'
    else:
        return f', shape=({dim},), dims="{dim_name}"'


class Normal(reno.components.Distribution):
    def __init__(self, mean, std=1.0, per_timestep: bool = False):
        # super().__init__()
        # self.mean = mean
        # self.std = std
        super().__init__(mean, std, per_timestep=per_timestep)

    def latex(self, **kwargs):
        return f"\\mathcal{{N}}({self.sub_equation_parts[0].latex(**kwargs)}, {self.sub_equation_parts[1].latex(**kwargs)}^2)"

    def populate(self, n: int, steps: int = 0, dim: int = 1):
        dims = dist_shape(self, n, steps, dim)
        self.value = np.random.normal(
            self.sub_equation_parts[0].eval(),
            self.sub_equation_parts[1].eval(),
            size=dims,
        )

    def get_type(self) -> type:
        return float

    def __repr__(self):
        return f"Normal({self.sub_equation_parts[0]}, {self.sub_equation_parts[1]}, {self.per_timestep})"

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        name, dim, dim_name, seq = dist_params(self, refs)
        return pm.Normal(
            name,
            self.sub_equation_parts[0].pt(**refs),
            self.sub_equation_parts[1].pt(**refs),
            **dist_dim_shape_args(self, refs),
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        name, dim, dim_name, seq = dist_params(self, refs)
        return f'pm.Normal("{name}", {self.sub_equation_parts[0].pt_str(**refs)}, {self.sub_equation_parts[1].pt_str(**refs)}{dist_dim_shape_args_str(self, refs)})'


class Uniform(reno.components.Distribution):
    def __init__(self, low=0.0, high=1.0, per_timestep: bool = False):
        super().__init__(low, high, per_timestep=per_timestep)

    def latex(self, **kwargs):
        return f"\\mathcal{{U}}({self.sub_equation_parts[0].latex(**kwargs)}, {self.sub_equation_parts[1].latex(**kwargs)})"

    def get_type(self) -> type:
        return float

    def populate(self, n: int, steps: int = 0, dim: int = 1):
        dims = dist_shape(self, n, steps, dim)
        self.value = np.random.uniform(
            self.sub_equation_parts[0].eval(),
            self.sub_equation_parts[1].eval(),
            size=dims,
        )

    def __repr__(self):
        return f"Uniform({self.sub_equation_parts[0]}, {self.sub_equation_parts[1]}, {self.per_timestep})"

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        name, dim, dim_name, seq = dist_params(self, refs)
        return pm.Uniform(
            name,
            self.sub_equation_parts[0].pt(**refs),
            self.sub_equation_parts[1].pt(**refs),
            **dist_dim_shape_args(self, refs),
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        name, dim, dim_name, seq = dist_params(self, refs)
        return f'pm.Uniform("{name}", {self.sub_equation_parts[0].pt_str(**refs)}, {self.sub_equation_parts[1].pt_str(**refs)}{dist_dim_shape_args_str(self, refs)})'


class DiscreteUniform(reno.components.Distribution):
    """Low is inclusive, high is exclusive."""

    def __init__(self, low: int = 0, high: int = 2, per_timestep: bool = False):
        super().__init__(low, high, per_timestep=per_timestep)

    def latex(self, **kwargs):
        return f"\\text{{DiscreteUniform}}({self.sub_equation_parts[0].latex(**kwargs)}, {self.sub_equation_parts[1].latex(**kwargs)})"

    def get_type(self) -> type:
        return int

    def populate(self, n: int, steps: int = 0, dim: int = 1):
        dims = dist_shape(self, n, steps, dim)
        self.value = np.random.randint(
            self.sub_equation_parts[0].eval(),
            self.sub_equation_parts[1].eval(),
            size=dims,
        )

    def __repr__(self):
        return f"DiscreteUniform({self.sub_equation_parts[0]}, {self.sub_equation_parts[1]}, {self.per_timestep})"

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        name, dim, dim_name, seq = dist_params(self, refs)
        return pm.DiscreteUniform(
            name,
            self.sub_equation_parts[0].pt(**refs),
            self.sub_equation_parts[1].pt(**refs),
            **dist_dim_shape_args(self, refs),
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        name, dim, dim_name, seq = dist_params(self, refs)
        return f'pm.DiscreteUniform("{name}", {self.sub_equation_parts[0].pt_str(**refs)}, {self.sub_equation_parts[1].pt_str(**refs)}{dist_dim_shape_args_str(self, refs)})'


class Bernoulli(reno.components.Distribution):
    """Discrete single event probability (p is probability of eval == 1)"""

    def __init__(self, p: float, use_p_dist: bool = False, per_timestep: bool = False):
        super().__init__(p, per_timestep=per_timestep)
        self.use_p_dist = use_p_dist

    def latex(self, **kwargs):
        return f"\\text{{Bernoulli}}({self.sub_equation_parts[0]})"

    def populate(self, n: int, steps: int = 0, dim: int = 1):
        dims = dist_shape(self, n, steps, dim)
        self.value = np.random.binomial(1, self.sub_equation_parts[0].eval(), dims)

    def get_type(self) -> type:
        return int

    def __repr__(self):
        return f"Bernoulli({self.sub_equation_parts[0]}, {self.use_p_dist}, {self.per_timestep})"

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        name, dim, dim_name, seq = dist_params(self, refs)

        inner_dist = self.sub_equation_parts[0].pt(**refs)
        if self.use_p_dist:
            inner_dist = pm.Interpolated(
                f"{name}_p",
                x_points=np.array([0.0, 1.0]),
                pdf_points=np.array(
                    [
                        1 - self.sub_equation_parts[0].pt(**refs),
                        self.sub_equation_parts[0].pt(**refs),
                    ]
                ),
            )
        return pm.Bernoulli(name, inner_dist, **dist_dim_shape_args(self, refs))

    def pt_str(self, **refs: dict[str, str]) -> str:
        name, dim, dim_name, seq = dist_params(self, refs)

        inner_dist = self.sub_equation_parts[0].pt_str(**refs)
        if self.use_p_dist:
            inner_dist = f'pm.Interpolated("{name}_p", x_points=np.array([0.0, 1.0]), pdf_points=np.array([{1 - self.sub_equation_parts[0].pt_str(**refs)}, {self.sub_equation_parts[0].pt_str(**refs)}]))'
        return (
            f'pm.Bernoulli("{name}", {inner_dist}{dist_dim_shape_args_str(self, refs)})'
        )


class Categorical(reno.components.Distribution):
    """Random categorical distribution - you specify the probability per category,
    and the output is a set of category indices."""

    def __init__(
        self, p: list[float], use_p_dist: bool = False, per_timestep: bool = False
    ):
        super().__init__(p, per_timestep=per_timestep)
        self.use_p_dist = use_p_dist
        self.expected_arg_num_dims[0] = 1

    def latex(self, **kwargs):
        return f"\\text{{Categorical}}({self.sub_equation_parts[0].latex(**kwargs)})"

    def get_type(self) -> type:
        return int

    def populate(self, n: int, steps: int = 0, dim: int = 1):
        dims = dist_shape(self, n, steps, dim)
        # TODO: how would p_dist apply here? Should it?
        self.value = np.argmax(
            np.random.multinomial(1, self.sub_equation_parts[0].eval(), dims), axis=-1
        )

    def __repr__(self):
        return f"Categorical({self.sub_equation_parts[0]}, {self.use_p_dist}, {self.per_timestep})"

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        name, dim, dim_name, seq = dist_params(self, refs)

        inner_dist = self.sub_equation_parts[0].pt(**refs)
        if self.use_p_dist:
            inner_dist = pm.Dirichlet(
                f"{name}_p", self.sub_equation_parts[0].pt(**refs)
            )
        return pm.Categorical(name, inner_dist, **dist_dim_shape_args(self, refs))

    def pt_str(self, **refs: dict[str, str]) -> str:
        name, dim, dim_name, seq = dist_params(self, refs)
        inner_dist = self.sub_equation_parts[0].pt_str(**refs)
        if self.use_p_dist:
            inner_dist = f'pm.Dirichlet("{name}_p", {self.p.pt_str(**refs)})'
        return f'pm.Categorical("{name}", {inner_dist}{dist_dim_shape_args_str(self, refs)})'


class List(reno.components.Distribution):
    """Tile passed list to the sample size so each value is hit roughly
    equally (dependent on exact sample size) and deterministically"""

    def __init__(self, values: list | np.ndarray | set):
        super().__init__()
        self.values = values

    def latex(self, **kwargs):
        return f"{self.values}"

    def get_shape(self):
        if isinstance(self.values, np.ndarray) and len(self.values.shape) == 2:
            return self.values.shape[1]
        if isinstance(self.values[0], (set, list)):
            return len(self.values[0])
        return 1

    def get_type(self) -> type:
        # TODO: this doesn't handle a value with a np array with shape > 2
        value = self.values
        # get to the first real value so we can determine type
        while isinstance(value, np.ndarray):
            value = value[0]
        while isinstance(value, (list, set)):
            value = value[0]

        if isinstance(value, np.floating):
            return float
        elif isinstance(value, np.integer):
            return int
        elif isinstance(value, np.bool):
            return bool
        return type(value)

    def populate(self, n: int, steps: int = 0, dim: int = 1):
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
        if dim > 1:
            if (
                isinstance(self.values, np.ndarray) and len(self.values.shape) == 1
            ) or not isinstance(self.values[0], (list, set)):
                self.value = np.repeat(np.expand_dims(self.value, axis=1), dim, axis=1)

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
