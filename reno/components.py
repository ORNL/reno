"""Classes for enabling symbolic equations and system dynamics pieces."""

# make it so we don't have to quote every type annotation ever
from __future__ import annotations

import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any

import matplotlib
import numpy as np
import pytensor.tensor as pt
from IPython.display import Markdown, display

import reno
from reno.utils import (
    check_for_easy_static_time_eq,
    ensure_scalar,
    is_static,
    latex_debug_output,
    latex_name,
    range_eq_latex,
)

# TODO: type aliases for:
# EquationPart | int | float | np.ndarray | list

# ==================================================
# EQUATION BASE CLASS
# ==================================================
# The root of all math things, the tree of life :)


class EquationPart:
    """The base object that represents some portion/subtree of a compute
    or equation tree.

    The purpose of EquationParts is to allow constructing an equation as a
    "deferred computation" - essentially symbolically creating a compute
    tree of objects all with ``.eval()`` functions that get recursively run
    later on when a simulation is running/equation is computing (spawned with
    root ``.eval()`.)

    Note:
        In order to get nice semantics when constructing equations, all math
        operations applied to an EquationPart get replaced with their
        corresponding operation EquationPart. This means a ``5 + Scalar(4)``
        gets converted into an ``Operation("+", Scalar(5), Scalar(4))``
    """

    def __init__(self, sub_equation_parts: list[EquationPart] = None):
        """Create a root node of an equation tree with the provided parts as subtrees.

        Args:
            sub_equation_parts (List[EquationPart]): The list of all **immediate**
                child equation subtrees, e.g. for the operation ``a + b``, the
                root equation part is the operation ``+``, and ``[a, b]`` would be
                its sub parts. These are used to help search the tree/connect
                parts of diagrams up. TODO: probably not strictly defined/enforced
                enough within the code, write tests for this.
        """
        if sub_equation_parts is None:
            sub_equation_parts = []
        # allow raw numbers to be passed in to any constructor, and ensure
        # they're wrapped in Scalars if relevant. This makes it so
        # reno.ops.add(5, Scalar(4)) correctly becomes Scalar(5) + Scalar(4)
        for i, part in enumerate(sub_equation_parts):
            sub_equation_parts[i] = ensure_scalar(part)
        self.value = None
        self.sub_equation_parts = sub_equation_parts
        self._shape = None  # remember, for now only the data dim
        self._dtype = None

    # ---- MATH OPERATION OVERLOADING/EQUATION PART REPLACEMENT ----

    def __add__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '+' operation with the passed object."""
        return reno.add(self, obj)

    def __sub__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '-' operation with the passed object."""
        return reno.sub(self, obj)

    def __neg__(self) -> Operation:
        """Returns symbolic negative operation (``-self``)."""
        return reno.sub(0, self)

    def __mul__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '*' operation with the passed object."""
        return reno.mul(self, obj)

    def __truediv__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '/' operation with the passed object."""
        return reno.div(self, obj)

    def __mod__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '%' operation with the passed object."""
        return reno.mod(self, obj)

    def __pow__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '**' operation with the passed object."""
        return reno.pow(self, obj)

    def __radd__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '+' operation with the passed object."""
        return reno.add(obj, self)

    def __rsub__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '-' operation with the passed object."""
        return reno.sub(obj, self)

    def __rmul__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '*' operation with the passed object."""
        return reno.mul(obj, self)

    def __rtruediv__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '/' operation with the passed object."""
        return reno.div(obj, self)

    def __rmod__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '%' operation with the passed object."""
        return reno.mod(obj, self)

    def __rpow__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '**' operation with the passed object."""
        return reno.pow(obj, self)

    def __lt__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '<' operation with the passed object."""
        return reno.lt(self, obj)

    def __le__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '<=' operation with the passed object."""
        return reno.lte(self, obj)

    def __gt__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '>' operation with the passed object."""
        return reno.gt(self, obj)

    def __ge__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '>=' operation with the passed object."""
        return reno.gte(self, obj)

    def __and__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '&' operation with the passed object."""
        return reno.bool_and(self, obj)

    def __rand__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '&' operation with the passed object."""
        return reno.bool_and(self, obj)

    def __or__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '|' operation with the passed object."""
        return reno.bool_or(self, obj)

    def __ror__(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns symbolic '|' operation with the passed object."""
        return reno.bool_or(self, obj)

    def __getitem__(
        self, obj: EquationPart | int | float | np.ndarray | slice
    ) -> Operation:
        """Returns symbolic operation to index or slice with the provided index
        expression.
        """
        if isinstance(obj, slice):
            return reno.slice(self, obj.start, obj.stop)
        return reno.index(self, obj)

    # NOTE: yeaaah weird things break if we override the eq portion.
    # Leaving this here as a reminder to not try to do this.
    # use the equal not_equal below.
    # def __eq__(self, obj):
    #     if isinstance(obj, (int, float)):
    #         return reno.ops.eq(self, Scalar(obj))
    #     return reno.ops.eq(self, obj)
    #
    # def __ne__(self, obj):
    #     if isinstance(obj, (int, float)):
    #         return reno.ops.ne(self, Scalar(obj))
    #     return reno.ops.ne(self, obj)

    # fill in remaining arithmetic and boolean operators as needed,
    # see https://www.pythonmorsels.com/every-dunder-method/

    # ---- /MATH OPERATION OVERLOADING/EQUATION PART REPLACEMENT ----

    # ---- CLEANER MATH API ----
    # (similar to how numpy allows either np.max(a) or a.max())

    def series_max(self) -> Operation:
        """Returns a symbolic operation to find the series-wise maximum of the array."""
        # See note in ops.py, not using 'max' because of TrackedReference
        return reno.series_max(self)

    def series_min(self) -> Operation:
        """Returns a symbolic operation to find the series-wise minimum of the array."""
        # See note in ops.py, not using 'min' because of TrackedReference
        return reno.series_min(self)

    def sum(self, axis: int = 0) -> Operation:
        """Returns a symbolic operation to find the series-wise sum of the array."""
        return reno.sum(self, axis=axis)

    def mean(self, axis: int = 0) -> Operation:
        """Returns a symbolic operation to find the series-wise mean of the array."""
        return reno.mean(self, axis=axis)

    @property
    def timeseries(self) -> Operation:
        """Returns symbolic operation for getting a timeseries view of the data.

        (includes all historical data across all timesteps.)
        """
        # TODO: possibly should be a property of TrackedReference instead
        return reno.orient_timeseries(self)

    @property
    def shape(self) -> int:
        """The size of the data dimension, 1 by default."""
        try:
            if self._shape is None:
                self._shape = self.get_shape()
            return self._shape
        except Exception as e:
            e.add_note(f"Was trying to get shape of {self.qual_name()}")
            raise

    @property
    def dtype(self) -> type:
        """The type of each underlying value."""
        if self._dtype is None:
            self._dtype = self.get_type()
        return self._dtype

    def equal(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns a symbolic operation for checking equality with passed object."""
        return reno.eq(self, obj)

    def not_equal(self, obj: EquationPart | int | float | np.ndarray) -> Operation:
        """Returns a symbolic operation for checking inequality with passed object."""
        return reno.ne(self, obj)

    def clip(
        self,
        min: EquationPart | int | float | np.ndarray,
        max: EquationPart | int | float | np.ndarray,
    ) -> Operation:
        """Returns a symbolic operation for enforcing the output is between the passed
        min and max.
        """
        return reno.clip(self, min, max)

    def astype(self, dtype: type) -> Operation:
        """Returns a symbolic operation to convert the output of this equation to the
        specified type.
        """
        return reno.astype(self, dtype)

    # ---- /CLEANER MATH API ----

    def eval(
        self,
        t: int = 0,
        save: bool = False,
        force: bool = False,
        **kwargs: dict,
    ) -> int | float | np.ndarray:
        """Execute the compute graph for this equation, this needs to be
        implemented in every subclass.

        Note that throughout a compute tree, this should effectively recurse
        through ``.eval()`` calls to all subparts as well.

        Args:
            t (int): Timestep along simulation at which to evaluate.
            save (bool): Whether to store/track/cache the output in a tracked
                matrix. This is really only applicable to ``TrackedReference``s,
                but given recursive nature of this function, needs to always be
                passed down through all subsequent calls.
            force (bool): Whether to ignore a previously cached value and compute
                regardless.
            kwargs (dict): Any additional arguments to pass through evaluation tree.
        """
        raise NotImplementedError()

    def latex(self, **kwargs: dict) -> str:
        """Construct a string representation of this portion of the equation
        for use in a latex display. Should probably be overriden in most
        subclasses, and often needs to be called recursively on
        sub_equation_parts.
        """
        return str(self.value)

    def seek_refs(  # noqa: C901
        self, include_ref_types: bool = False
    ) -> list[Reference] | dict[Reference, list[str]]:
        """Recursively find a list of all References immediately underneath this part.

        Either provides a list of references that appear underneath these equations
        or a dictionary with the "roles" or types that those references play.

        Note that this doesn't infinitely recurse into TrackedReferences themselves,
        this is used to find the outermost references that need to resolve. (Avoiding
        circular refs)
        """
        # TODO: (2026-03-18): break this up so that components reponsible for
        # the sub behavior, bad pattern to be defining _everything_ up here.
        try:
            # find the equation parts to look into for references
            check_parts = [*self.sub_equation_parts]
            check_part_source_description = [None] * len(self.sub_equation_parts)
            # (use sources for helping determine ref type in certain cases)
            if isinstance(self.value, EquationPart):
                # TODO: I think this handles cases where you've directly assigned a
                # scalar as a value to something?
                check_parts.append(self.value)
                check_part_source_description.append(None)
            if isinstance(self, TrackedReference):
                if self.min is not None:
                    check_parts.append(self.min)
                    check_part_source_description.append("limit")
                if self.max is not None:
                    check_parts.append(self.max)
                    check_part_source_description.append("limit")

            refs = {} if include_ref_types else []
            for i, part in enumerate(check_parts):
                if isinstance(part, (Function, Operation, Piecewise)):
                    sub_refs = part.seek_refs(include_ref_types)
                    if include_ref_types:
                        for ref in sub_refs:
                            if ref not in refs:
                                refs[ref] = []
                            refs[ref].extend(sub_refs[ref])
                            if check_part_source_description[i] is not None:
                                refs[ref].append(check_part_source_description[i])
                    else:
                        refs.extend(sub_refs)
                elif isinstance(part, HistoricalValue):
                    if include_ref_types:
                        if part.tracked_ref not in refs:
                            refs[part.tracked_ref] = []
                        if part not in refs:
                            refs[part] = []
                        refs[part.tracked_ref].append("history")
                        refs[part].append("history")

                        sub_refs = part.index_eq.seek_refs(include_ref_types)
                        for ref in sub_refs:
                            if ref not in refs:
                                refs[ref] = []
                            refs[ref].extend(sub_refs[ref])
                            if check_part_source_description[i] is not None:
                                refs[ref].append(check_part_source_description[i])
                        if check_part_source_description[i] is not None:
                            refs[part.tracked_ref].append(
                                check_part_source_description[i]
                            )
                            refs[part].append(check_part_source_description[i])
                    else:
                        refs.append(part.tracked_ref)
                        refs.append(part)  # TODO: (2025.02.03) ?????
                        # I feel like I'm still missing something between eq setting,
                        # assign op, seek_refs and static checks, but
                        # test_static_check_eq_with_historical_value breaks without this
                        refs.extend(part.index_eq.seek_refs(include_ref_types))
                elif isinstance(part, Reference):
                    if include_ref_types:
                        if part not in refs:
                            refs[part] = []
                        if check_part_source_description[i] is not None:
                            refs[part].append(check_part_source_description[i])
                    else:
                        refs.append(part)

            # be sure to remove duplicates
            if include_ref_types:
                return refs
            return list(set(refs))
        except Exception as e:
            e.add_note(f"Was trying to find sub-references of {self}")
            for part in check_parts:
                e.add_note(f"\tWas checking part: {part}")
            raise

    def get_shape(self) -> int:
        """Get the size of the additional "data" dimension.

        Prefer using ``shape`` property over directly calling this function.

        Override this in any subclass to control how shape is calculated.

        For now this is returning an integer because we only allow a single
        additional dimension. Note that this shape _does not_ incoporate time or
        batch dimensions, only the "data" dimension if applicable.
        This should be overridden by subclasses, e.g. operations which would
        change the shape.
        """
        return 1

    def get_type(self) -> type:
        """Get the type of the target output of this equation expression.

        Prefer using ``dtype`` property over directly calling this function.

        Override this in any subclass to control how the type gets determined.

        A ``None`` return means that the type doesn't matter or shouldn't influence or
        override anything upstream.

        Similar to shape, this gets computed recursively, used to automatically
        determine if the value needs to be initialized with a certain numpy type.
        """
        return None

    def find_parts_of_type(
        self, search_type: type, already_checked: list[EquationPart] = None
    ) -> list[EquationPart]:
        """Recursively search for all EquationParts in the tree of the specified type.

        Args:
            search_type (type): The type of component/equation part to search for,
                e.g. :py:class:`reno.components.ExtendedOperation`.
            already_checked (list[EquationPart]): Used to avoid recursing into already
                checked visited equation parts.
        """
        if already_checked is None:
            already_checked = []

        matching_parts = []
        check_parts = [*self.sub_equation_parts]
        if isinstance(self.value, EquationPart):
            # TODO: I think this handles cases where you've directly assigned a
            # scalar as a value to something?
            # NOTE: I don't actually think this works for this function
            check_parts.append(self.value)

        # pre check for anything that needs to expand check parts
        for part in check_parts:
            if isinstance(part, HistoricalValue):
                check_parts.append(part.tracked_ref)

        for part in check_parts:
            if part in already_checked:
                continue
            already_checked.append(part)
            if isinstance(part, search_type):
                matching_parts.append(part)

            matching_parts.extend(part.find_parts_of_type(search_type, already_checked))
        return matching_parts

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        """Get a pytensor graph representing this piece of an equation."""
        raise NotImplementedError(
            f"Failed at {self}, no pt() implementation found for this equation part."
        )

    def pt_str(self, **refs: dict[str, str]) -> str:
        """Construct a string containing relevant pytensor code for this piece
        of the equation. This is useful for "compiling" into pymc code.
        """
        raise NotImplementedError(
            f"Failed at {self}, no pt_str() implementation found for this equation part."
        )

    def is_static(self) -> bool:
        """Convenience shortcut for :py:func:`reno.utils.is_static` - True if this
        equation doesn't rely on any dynamic values (thus constant), False if it does.
        """
        return is_static(self)


# ==================================================
# PRIMITIVES
# ==================================================
# The base level things that can be used in an equation. Most of these
# are subclassed elsewhere (minus Scalar) and those subclasses are probably
# what someone using this API should be using.


# TODO: technically this should probably be called "Value" instead of Scalar
# since it can take multidim data.
class Scalar(EquationPart):
    """A static, single value equation part, representing some simple value that
    doesn't need to be computed.
    """

    def __init__(self, value: int | float | list | np.ndarray):
        """Create a static value node holding the provided value.

        Args:
            value (int | float | np.ndarray): The scalar value to use.
        """
        super().__init__()
        if isinstance(value, (list | bool)):
            value = np.array(value)
        self.value = value

    def get_shape(self) -> int:
        """Get the size of the additional "data" dimension.

        Prefer using ``shape`` property over directly calling this function.
        """
        if isinstance(self.value, np.ndarray):
            if self.value.shape == ():
                return 1
            return self.value.shape[0]
        return 1

    def get_type(self) -> type:
        """Get the type of the target output of this equation expression.

        Prefer using ``dtype`` property over directly calling this function.
        """
        # TODO: this doesn't handle a value with a np array with shape > 1
        if isinstance(self.value, np.ndarray):
            value = self.value if self.value.shape == () else self.value[0]
            if isinstance(value, np.floating):
                return float
            elif isinstance(value, np.integer):
                return int
            elif isinstance(value, np.bool_):
                return bool
            else:
                # TODO: possibly raise warning here?
                return None
        return type(self.value)

    def eval(
        self,
        t: int = 0,
        save: bool = False,
        force: bool = False,
        **kwargs: dict,
    ) -> int | float | np.ndarray:
        """No compute necessary, just retrieve statically defined value."""
        return self.value

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:  # noqa: D102
        tensor = pt.as_tensor(self.value)
        # consolidate non-integer types to float64, otherwise pymc gets real mad
        if tensor.dtype == "float32":
            tensor = tensor.astype("float64")
        return tensor

    def pt_str(self, **refs: dict[str, str]) -> str:  # noqa: D102
        if pt.as_tensor(self.value).dtype == "float32":
            return f'pt.as_tensor({self.value}).astype("float64")'
        return f"pt.as_tensor({self.value})"

    def __repr__(self) -> str:
        # print("Type", type(self.value))
        value = self.value
        if isinstance(self.value, np.ndarray):
            value = value.tolist()
        return f"Scalar({value})"


class Distribution(EquationPart):
    """Represents a probability distribution or set that is drawn from for each
    sample (n) in a simulation.

    Probably shouldn't directly be using this, although you can in theory
    likely pass in ``partial()``s with numpy functions (just minus the size
    variable.)
    """

    def __init__(self, *operands: list[EquationPart], per_timestep: bool = False):
        """Create a distribution equation part, this is likely being called from
        a subclass.

        Args:
            operands (list[EquationPart]): Any sub equation parts to track for
                this distribution.
            per_timestep (bool): Whether to draw a different value from the
                distribution for each timestep.
        """
        super().__init__(list(operands))
        self.per_timestep = per_timestep
        self.expected_arg_num_dims = {}
        # if a parameter is supposed to be an array, set expected dims for that
        # parameter int index to 1, that way can be checked for in get_shape
        # NOTE: this same strategy may need to apply to operation as well

    def populate(self, n: int, steps: int = 0, dim: int = 1) -> None:
        """Generate n x dim samples based on this probability distribution, assigns
        as a vector/matrix to ``self.value``.

        Args:
            n (int): Number of samples to draw.
            steps (int): Number of timesteps for which to draw samples for, only
                relevant if ``per_timestep`` is ``True``.
            dim (int): If > 1, draw samples into a vector of this size for each n.
        """
        # should be implemented in child classes

    # NOTE: arguably a distribution could directly inherit from operation and
    # work as expected?
    def get_shape(self) -> int:
        """Get the size of the additional "data" dimension.

        Prefer using ``shape`` property over directly calling this function.

        For a distribution we "broadcast" from any sub equation parts, so a >1
        dim in a sub part will extend to this one.
        """
        # by _default_, use a simple broadcast if all parts but one are just one
        # TODO: prob need to do similar to op_repr and have an op_get_shape

        try:
            # print(f"Getting shape for {self}")
            non_one_shapes = []
            for index, sub_equation_part in enumerate(self.sub_equation_parts):
                shape = sub_equation_part.shape
                if (
                    index in self.expected_arg_num_dims
                    and self.expected_arg_num_dims[index] == 1
                ):
                    # TODO: will eventually need something more robust
                    shape = 1

                # print(f"Looking at {sub_equation_part}, shape {shape}")
                if shape > 1:
                    non_one_shapes.append(shape)
            non_one_shapes = non_one_shapes
            if len(set(non_one_shapes)) > 1:
                raise Exception("Non-matching shapes, can't broadcast")
            if len(non_one_shapes) > 0:
                # if we hit this point there's only one non-one shape, so
                # it's okay to just grab first one
                return non_one_shapes[0]
            return 1  # no data dims involved
        except Exception as e:
            e.add_note(
                f'Was trying to compute shape for operation "{self.__class__}" ({self.op_repr()}) with sub equation parts:'
            )
            for sub_eq in self.sub_equation_parts:
                e.add_note(f"\t{sub_eq}: {sub_eq._shape}")
            raise

    def eval(
        self,
        t: int = 0,
        save: bool = False,
        force: bool = False,
        **kwargs: dict,
    ) -> np.ndarray:
        """Get the vector of samples pulled from the probability distribution.

        Note that this expects ``populate()`` to already have been called.
        """
        if not self.per_timestep:
            return self.value
        else:
            return self.value[:, t]

    def clean_part_repr(self, part_index: int) -> str:
        """Get a non-"Scalar()" string version of a particular sub equation part.

        Since all non-Scalar static values get converted into Scalar instances, this
        is used to remove all the "Scalar" strings everywhere e.g. in repr outputs.
        See ops.py __repr__ functions for examples of how it's used.
        """
        part = self.sub_equation_parts[part_index]
        if isinstance(part, Scalar):
            return str(part.value)
        return part


class Operation(EquationPart):
    """Parent class to represent a mathematical operation in an equation.

    Pretty useless by itself, look in reno/ops.py, we use a parent class
    to make it easier to distinguish/find parts of an equation that are
    operations.
    """

    OP_REPR = None
    """If defined, this is what gets printed as the 'label' for the operation in the
    repr, and is what's used during parsing. Otherwise, just uses class name."""

    def __init__(self, *operands: list[EquationPart]):
        """Create an operation node involving the passed sub equation parts.

        Args:
            operands (list[EquationParts]): The sub equations the operation is
                applying across.
        """
        super().__init__(list(operands))
        self.non_repr_part_indices = []
        """Use this to "hide" a sub_equation_part in the string repr, e.g. if
        it's important for computation but isn't part of parsing."""

    def eval(
        self,
        t: int = 0,
        save: bool = False,
        force: bool = False,
        **kwargs: dict,
    ) -> int | float | np.ndarray:
        """'wrap' all operation eval functions so we get better error handling."""
        try:
            return self.op_eval(t=t, save=save, force=force, **kwargs)
        except Exception as e:
            e.add_note(
                f'Was evaluating operation "{self.__class__}" ({self.op_repr()}) with sub equation parts:'
            )
            for sub_eq in self.sub_equation_parts:
                e.add_note(f"\t{sub_eq}")
            raise

    def get_shape(self) -> int:
        """Get the size of the additional "data" dimension.

        Prefer using ``shape`` property over directly calling this function.

        "Broadcasts" any size 1 dims to a larger dim node if present.
        """
        # by _default_, use a simple broadcast if all parts but one are just one
        # TODO: prob need to do similar to op_repr and have an op_get_shape
        try:
            # print(f"Getting shape for {self}")
            non_one_shapes = []
            for sub_equation_part in self.sub_equation_parts:
                shape = sub_equation_part.shape
                # print(f"Looking at {sub_equation_part}, shape {shape}")
                if shape > 1:
                    non_one_shapes.append(shape)
            non_one_shapes = non_one_shapes
            if len(set(non_one_shapes)) > 1:
                raise Exception("Non-matching shapes, can't broadcast")
            if len(non_one_shapes) > 0:
                # if we hit this point there's only one non-one shape, so
                # it's okay to just grab first one
                return non_one_shapes[0]
            return 1  # no data dims involved
        except Exception as e:
            e.add_note(
                f'Was trying to compute shape for operation "{self.__class__}" ({self.op_repr()}) with sub equation parts:'
            )
            for sub_eq in self.sub_equation_parts:
                e.add_note(f"\t{sub_eq}: {sub_eq._shape}")
            raise

    def get_type(self) -> type:
        """Get the type of the target output of this equation expression.

        Prefer using ``dtype`` property over directly calling this function.
        """
        types = []
        for sub_equation_part in self.sub_equation_parts:
            types.append(sub_equation_part.dtype)
        # order of precedence, no idea if this is the correct way or not
        if float in types:
            return float
        if int in types:
            return int
        if bool in types:
            return bool
        return None

    def op_eval(self, **kwargs: dict) -> int | float | np.ndarray:
        """Any new operations should implement this method.

        Put all evaluation logic in here, as opposed to overriding eval.
        """
        raise NotImplementedError()

    def latex(self, **kwargs: dict) -> str:
        """Get a string of latex representing this op.

        Wrapper around op_latex, subclasses should override that
        function as opposed to this one.
        """
        latex_str = self.op_latex(**kwargs)
        if "t" in kwargs and "debug_ops" in kwargs and kwargs["debug_ops"]:
            latex_str = f"({latex_str})"
            latex_str = latex_debug_output(
                self, latex_str, color="orange", style="underbrace", **kwargs
            )
        return latex_str

    def op_latex(self, **kwargs: dict) -> str:
        """Output the latex string for this operation. Override this in subclasses."""
        super().latex(**kwargs)

    @classmethod
    def op_repr(cls) -> str:
        """Get a string representation for the op name/label.

        Used for printing and parsing.
        """
        if cls.OP_REPR is not None:
            return cls.OP_REPR
        # use the class name if no OP_REPR was set on the class.
        return cls.__name__

    @staticmethod
    def op_types() -> list[type]:
        """Get a list of all defined operations/Operation subclasses."""
        subclasses = []  # collected list of types
        to_process = [Operation]  # the "frontier"
        while len(to_process) > 0:
            # get the next class to search
            parent_class = to_process.pop()

            # any new subclasses and make sure they're in the frontier
            for child_class in parent_class.__subclasses__():
                if child_class not in subclasses:
                    subclasses.append(child_class)
                    to_process.append(child_class)
        return subclasses

    def __repr__(self) -> str:
        sub_parts_string = " ".join(
            [
                subpart.__repr__()
                for i, subpart in enumerate(self.sub_equation_parts)
                if i not in self.non_repr_part_indices
            ]
        )
        return f"({self.op_repr()} {sub_parts_string})"


class ExtendedOperation(Operation):
    """An operation defined in terms of other operations, and may require population
    before use.

    Defining this as a separate type to make it easier to search for in equation trees.
    """

    def __init__(
        self, operands: list[EquationPart], implicit_components: dict[str, EquationPart]
    ):
        """Create a extended operation node with the provided and implicit components.

        Args:
            operands (list[EquationPart]): Any provided sub equation parts to track.
            implicit_components (dict[str, EquationPart]): Any "private" components
                created within this class. They shouldn't show up in diagrams, but are
                used for computation. See :py:class:`reno.ops.delay1` for an example.
                Names can be any representative string, and are used to label pymc
                outputs.
        """
        super().__init__(*operands + list(implicit_components.values()))
        self.implicit_components = implicit_components
        for name, component in self.implicit_components.items():
            component.implicit = True

    # NOTE: not needed, since should be directly handled by the model
    # def populate(self, n: int, steps: int):
    #     pass


class Reference(EquationPart):
    """A symbolic reference to some other equation component, e.g. a
    stock, flow, or variable.

    This is a largely semantic parent class, probably shouldn't directly be
    using this.

    Semantically a reference is anything that will need to "resolve" somehow,
    either by looking up a historical value or running a sub equation to get
    the result.
    """

    def __init__(self, label: str = None, doc: str = None):
        """Create a reference node/something that refers to another equation or thing.

        Args:
            label (str): The visual label for the reference. In the context of a model,
                is set to the name assigned on the model if not explicitly provided.
            doc (str): A docstring/description explaining what this component is.
        """
        self.name = None
        self.label = label
        """Label is what's used in any visual representation (e.g. allows spaces where
        name does not.)"""
        self.doc = doc
        """A docstring to explain/describe the reference."""
        super().__init__()

    def latex(self, **kwargs: dict) -> str:
        """String representation suitable for a latex display."""
        latex_str = latex_name(self.label)
        if hasattr(self, "model") and self.model.parent is not None:
            latex_str += f"_{{{latex_name(self.model.label)}}}"
        if "t" in kwargs:
            latex_str = latex_debug_output(self, latex_str, **kwargs)
        if "hl" in kwargs:  # noqa: SIM102
            if kwargs["hl"] == self.name or (
                hasattr(self, "qual_name") and kwargs["hl"] == self.qual_name()
            ):
                latex_str = "{\\color{cyan}" + latex_str + "}"

        return latex_str

    # TODO: (2026-03-30) should qual_name be defined here? Prev ran into issues
    # because e.g. timeref doesn't have that function, believe I solved at the
    # time by looking specifically for HistoricalValue and TrackedReference, but
    # might be easier to define reference-wide (plus if models have diff
    # timerefs)

    def __repr__(self) -> str:
        return f'"{self.label}"'


class Piecewise(EquationPart):
    """A conditional allowing evaluation of two or more condition equations
    to determine which output equation to use.

    Mathematically allows something like:
           ⎧ 1     if t < 4
    f(t) = ⎨ 5     if 4 <= t < 7
           ⎩ 9 + t if 7 <= t
    """

    # TODO: is this an operation? It kind of is isn't it? Not sure if it's
    # worth inheriting from operation or not, biggest difference I'd think
    # would be the op_eval vs eval (better debugging?) and less jank to include
    # piecewise handling in parser?
    # TODO: another option would be to support arbitrary *args and each
    # condition would follow each equation. This would likely get rid of
    # separate need for parse
    def __init__(self, equations: list[EquationPart], conditions: list[Callable]):
        """Create a piecewise ("if-else operation") node.

        Args:
            equations (list[EquationPart]): The possible equation branches that
                selectively evaluate based on which condition is true. Must have
                same number of equations as conditions.
            conditions (list[Callable]): The boolean conditions to evaluate to
                determine which equation to output. These can all either be an
                equation based on EquationPart (see the boolean ops in ops.py),
                or a function that accepts a single parameter (t) as an input.
                Must have same number of conditions as equations.
        """
        # ensure scalar on each equation to allow specifying raw integers as
        # outputs without having to specify Scalar
        equations = [ensure_scalar(equation) for equation in equations]
        all_parts = [*equations]
        for condition in conditions:
            # if hasattr(condition, "eval"):
            if isinstance(condition, EquationPart):
                all_parts.append(condition)
        super().__init__(all_parts)
        self.equations = equations
        self.conditions = conditions

        assert len(self.equations) == len(self.conditions), (
            "Number of equations and number of conditions must match"
        )

    def eval_condition(self, i: int, *args: list, **kwargs: dict) -> bool | np.ndarray:
        """Recursively build nested np.where to handle all conditions.

        Note that this does evaluate all equations and all conditions, there are
        likely ways to improve efficiency/reduce unnecessary compute (e.g. this
        used to use np.piecewise, but adding data dimensions significantly
        increased complexity)
        """
        condition = self.conditions[i].eval(*args, **kwargs)
        eqn1 = self.equations[i].eval(*args, **kwargs)
        if i < len(self.conditions) - 2:
            eqn2 = self.eval_condition(i + 1, *args, **kwargs)
        else:
            eqn2 = self.equations[i + 1].eval(*args, **kwargs)
        return np.where(condition, eqn1, eqn2)

    def eval(
        self,
        t: int = 0,
        save: bool = False,
        force: bool = False,
        **kwargs: dict,
    ) -> int | float | np.ndarray:
        """Evaluate condition equations/functions until a ``True`` is returned,
        and then evaluate and return the corresponding equation.
        """
        # NOTE: leaving this here because this is a low-fruit optimization
        # to add, we'd simply need to remove condition _evaluation_ from
        # eval_condition, and just store the condition eval results for it to
        # use in the np.where
        # condition_bools = [
        #     condition.eval(t, save, force, **kwargs)
        #     for condition in self.conditions
        # ]
        #
        # shapes = [
        #     condition_bool.shape[0] if hasattr(condition_bool, "shape") else 1
        #     for condition_bool in condition_bools
        # ]
        # max_shape = max(shapes)
        #
        # # handle basic case where we're dealing with simple booleans
        # if max_shape == 1:
        #     for index, condition_bool in enumerate(condition_bools):
        #         if condition_bool:
        #             return self.equations[index].eval(
        #                 t, save, force, **kwargs
        #             )
        # else:

        return self.eval_condition(0, t=t, save=save, force=force, **kwargs)

        raise Exception(  # pylint: disable=W0719
            f"Piecewise function {self.value} had no valid conditions."
        )

    def latex(self, **kwargs: dict) -> str:
        r"""Get a latex piecewise string (left bracket with eqs and conditions).

        Conditions that aren't EquationParts can't really be converted meaningfully, so
        are represented simply with ``\\lambda(t)``.
        """
        # allow a more generic representation for limited latex environments,
        # like trying to output a text png from matplotlib
        # if "simplified_output" in kwargs and kwargs["simplified_output"]:
        #     pass
        # else
        string = "\\begin{cases}\n"
        for index, eq in enumerate(self.equations):
            condition_repr = (
                "\\lambda(t)"
                if not hasattr(self.conditions[index], "latex")
                else self.conditions[index].latex(**kwargs)
            )
            string += eq.latex(**kwargs) + " & \\text{if }" + condition_repr + " \\\\\n"
        string += "\\end{cases}"
        return string

    def pt_condition(
        self, i: int, refs: dict[str, pt.TensorVariable]
    ) -> pt.TensorVariable:
        """Get the pytensor equation starting at the ith condition (using ifelse).

        Recursively includes all >i conditions.
        """
        condition = self.conditions[i].pt(**refs)
        eqn1 = self.equations[i].pt(**refs)
        eqn2 = self.equations[i + 1].pt(**refs)
        if i < len(self.conditions) - 2:
            eqn2 = self.pt_condition(i + 1, refs)
        # pymc gets the big angry if you try to mix float64 and float32
        # pymc getting the big angry gets me the big angry
        if eqn1.dtype == "float32":
            eqn1 = eqn1.astype("float64")
        if eqn2.dtype == "float32":
            eqn2 = eqn2.astype("float64")
        return pt.switch(condition, eqn1, eqn2)

    def pt_condition_str(self, i: int, refs: dict[str, str]) -> str:
        """Get the string for the pytensor code for the ith condition (using ifelse).

        Recursively includes all >i conditions.
        """
        condition_str = self.conditions[i].pt_str(**refs)
        # we get the actual equations just so we can check if casts are necessary
        eqn1 = self.equations[i].pt()
        eqn1_str = self.equations[i].pt_str(**refs)
        eqn2 = self.equations[i + 1].pt()
        eqn2_str = self.equations[i + 1].pt_str(**refs)
        if i < len(self.conditions) - 2:
            eqn2 = self.pt_condition(i + 1, {})
            eqn2_str = self.pt_condition_str(i + 1, refs)
        if eqn1.dtype == "float32":
            eqn1_str += '.astype("float64")'
        if eqn2.dtype == "float32":
            eqn2_str += '.astype("float64")'
        return f"pt.switch({condition_str}, {eqn1_str}, {eqn2_str})"

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:  # noqa: D102
        return self.pt_condition(0, refs)

    def pt_str(self, **refs: dict[str, str]) -> str:  # noqa: D102
        return self.pt_condition_str(0, refs)

    def get_shape(self) -> int:
        """Get the size of the additional "data" dimension.

        Prefer using ``shape`` property over directly calling this function.

        "Broadcasts" any size 1 dims to a larger dim node if present.
        """
        # by _default_, use a simple broadcast if all parts but one are just one
        # TODO: prob need to do similar to op_repr and have an op_get_shape
        try:
            non_one_shapes = []
            for equation in self.equations + self.conditions:
                shape = equation.shape
                if shape > 1:
                    non_one_shapes.append(shape)
            if len(set(non_one_shapes)) > 1:
                raise Exception("Non-matching shapes, can't broadcast")
            if len(non_one_shapes) > 0:
                # if we hit this point there's only one non-one shape, so
                # it's okay to just grab first one
                return non_one_shapes[0]
            return 1  # no data dims involved
        except Exception as e:
            e.add_note(
                f'Was trying to compute shape for operation "{self.__class__}" with sub equation parts:'
            )
            for sub_eq in self.sub_equation_parts:
                e.add_note(f"\t{sub_eq}: {sub_eq._shape}")
            raise

    def get_type(self) -> type:
        """Get the type of the target output of this equation expression.

        Prefer using ``dtype`` property over directly calling this function.
        """
        types = []
        for equation in self.equations:
            types.append(equation.dtype)
        # order of precedence, no idea if this is the correct way or not
        if float in types:
            return float
        if int in types:
            return int
        if bool in types:
            return bool
        return None

    @staticmethod
    def parse(arg_strs: list[str], refs: dict[str, Reference]) -> Piecewise:
        """Convert a set of argument strings and references into a piecewise node.

        Parsing a piecewise has to be handled differently from normal operations because
        of the collection of associated arguments.
        """
        equations = []
        conditions = []
        for arg_str in arg_strs:
            arg_str_parts = reno.parser.parse_op_str(arg_str, no_op=True)
            if len(arg_str_parts) != 2:
                raise SyntaxError(
                    f"Every piecewise argument should contain a condition and an equation, found {len(arg_str_parts)} parts in '{arg_str}'"
                )
            condition = reno.parser.parse(arg_str_parts[0], refs)
            equation = reno.parser.parse(arg_str_parts[1], refs)
            conditions.append(condition)
            equations.append(equation)
        return Piecewise(equations, conditions)

    def __repr__(self) -> str:
        piece_str = "(piecewise"
        for i, condition in enumerate(self.conditions):
            piece_str += f" ({condition.__repr__()} {self.equations[i].__repr__()})"
        piece_str += ")"
        return piece_str


class Function(Reference):
    """Represents a call to a python function _outside_ of a model/system of
    equations.

    This is to allow for computations that might be a little complex, too fancy,
    or too annoying to specify entirely within the ``EquationPart`` system. By
    wrapping the call with this class, we defer actual computation/computation
    of input parameters until eval time, it's a thunk!

    Note:
        Any passed arguments that _are_ equations will be evaluated and passed
        in to the actual function call when ``.eval()`` is run. In a simulation
        run with stocks/flows etc., these arguments are likely to resolve to numpy
        arrays, which is the vector of all samples for that argument at the current
        eval'ed timestep. (So the function should expect to do vector operations,
        numpy will probably automagically handle this.)

    NOTE: theoretically works in pymc conversion (as long as mathlib is used) but
    relatively untested as of writing.

    NOTE: pt_str isn't implemented in this class due to difficulty in referring
    to an unknown external function via a string.
    """

    # TODO: Possibly make this a TrackedReference and move to "complex" section
    # of this module, it would be useful to see a history of function outputs,
    # and the caching for expensive functions might be nice if the function gets
    # referenced (eval'd) multiple times.

    def __init__(
        self,
        f: Callable,
        *args: list,
        inject_mathlib: bool = True,
        pt_mathlib: ModuleType = None,
        np_mathlib: ModuleType = None,
        **kwargs: dict,
    ):
        """Create a function thunk node for an equation tree.

        Args:
            f (Callable): The python function to run when ``.eval()`` is called.
            args (list[any]): Arguments to pass to ``f`` when ``.eval()`` is called.
                Any ``EquationPart`` arguments will have their ``.eval()`` called prior
                to ``f``'s execution.
            inject_mathlib (bool): Whether to pass a `mathlib` arg to allow running
                in both normal reno math mode as well as pymc. Requires underlying
                function to take `mathlib`.
            pt_mathlib: The library to use for pytensor operations.
            np_mathlib: The library to use for numpy operations.
            kwargs (list[any]): Keyword args to pass to ``f`` when ``.eval()`` is
                called. Any ``EquationPart`` arguments will have their ``.eval()``
                called prior to ``f``'s execution.
        """
        if pt_mathlib is None:
            pt_mathlib = pt
        if np_mathlib is None:
            np_mathlib = np

        self.f = f
        self.args = args
        self.kwargs = kwargs

        self.inject_mathlib = inject_mathlib
        self.pt_mathlib = pt_mathlib
        self.np_mathlib = np_mathlib

        # find any equation parts in the arguments so we can set the
        # sub_equation_parts of the parent class.
        eqn_parts = []
        for arg in args:
            if isinstance(arg, EquationPart):
                eqn_parts.append(arg)
        for kwarg in kwargs.values():
            if isinstance(kwarg, EquationPart):
                eqn_parts.append(kwarg)

        # TODO: ref should probably allow taking eqn parts?
        super().__init__(f.__name__)
        self.sub_equation_parts = eqn_parts

    def __repr__(self) -> str:
        return f"{self.f.__name__}()"

    def eval(
        self,
        t: int = 0,
        save: bool = False,
        force: bool = False,
        **kwargs: dict,
    ) -> int | float | np.ndarray:
        """Eval any function arguments as needed, and then call the function itself.

        Args:
            t (int): Timestep along simulation at which to evaluate.
            save (bool): Ignored in this subclass, but passed along in sub-
                ``.eval()`` calls.
            force (bool): Whether to ignore a previously cached value and compute
                regardless.
            kwargs (dict): Any additional arguments to pass through evaluation tree.
        """
        pass_args = []
        pass_kwargs = {}

        # handle mathlib if necessary
        if self.inject_mathlib:
            pass_kwargs["mathlib"] = self.np_mathlib

        # resolve any arguments that are EquationParts
        for arg in self.args:
            if isinstance(arg, EquationPart):
                pass_args.append(arg.eval(t, save, force, **kwargs))
            else:
                pass_args.append(arg)
        for key, arg in self.kwargs.items():
            if isinstance(arg, EquationPart):
                pass_kwargs[key] = arg.eval(t, save, force, **kwargs)
            else:
                pass_kwargs[key] = arg

        # call the actual function
        return self.f(*pass_args, **pass_kwargs)

    def latex(self, **kwargs: dict) -> str:
        """Get a latex-suitable string representation of the function.

        Uses texttt to make it look like code and distinguishable from
        flows/vars/stocks/etc.
        """
        parameters = [part.latex(**kwargs) for part in self.sub_equation_parts]
        latex_str = f"{latex_name(self.label, 'texttt')}({', '.join(parameters)})"
        if "t" in kwargs:
            # NOTE: we used to render output color for functions as yellow to
            # visually distinguish, might be worth adding that back in at some
            # point.
            latex_str = latex_debug_output(self, latex_str, **kwargs)
        return latex_str

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        """Get a pytensor graph representing this piece of an equation."""
        pass_args = []
        pass_kwargs = {}

        # handle mathlib if necessary
        if self.inject_mathlib:
            pass_kwargs["mathlib"] = self.pt_mathlib

        # resolve any arguments that are EquationParts
        for arg in self.args:
            if isinstance(arg, EquationPart):
                pass_args.append(arg.pt(**refs))
            else:
                pass_args.append(arg)
        for key, arg in self.kwargs.items():
            if isinstance(arg, EquationPart):
                pass_kwargs[key] = arg.pt(**refs)
            else:
                pass_kwargs[key] = arg

        # call the actual function
        return self.f(*pass_args, **pass_kwargs)

    # TODO: this is actually difficult because we don't know how the function is
    # being imported/thus don't know how to call it by name correctly?
    # we could potentially store in a global somehow, but this gets messy fast.
    # Leaving this here for now in case I get a brilliant idea for how to make
    # it work.
    # def pt_str(self, **refs: dict[str, str]) -> str:
    #     pass_args = []
    #     pass_kwargs = {}
    #
    #     # handle mathlib if necessary
    #     if self.inject_mathlib:
    #         pass_kwargs["mathlib"] = self.pt_mathlib
    #
    #     # resolve any arguments that are EquationParts
    #     for arg in self.args:
    #         if isinstance(arg, EquationPart):
    #             pass_args.append(arg.pt(**refs))
    #         else:
    #             pass_args.append(arg)
    #     for key, arg in self.kwargs.items():
    #         if isinstance(arg, EquationPart):
    #             pass_kwargs[key] = arg.pt(**refs)
    #         else:
    #             pass_kwargs[key] = arg
    #
    #     # call the actual function
    #     return self.f(*pass_args, **pass_kwargs)
    #


class TimeRef(Reference):
    """A reference to the current simulation timestep that can be used in
    equations.

    Example:
        >>> from reno.components import TimeRef
        >>> t = TimeRef()

        >>> t + 3
        (+ t 3)

        >>> (t + 3).eval(4)
        7
    """

    def __init__(self):
        """Create an instance of a time reference for use in other equations."""
        super().__init__("t")
        self.name = "t"

    def eval(
        self,
        t: int = 0,
        save: bool = False,
        force: bool = False,
        **kwargs: dict,
    ) -> int:
        """Return the current timestep.

        Args have no effect but are accepted from recursive calls.
        """
        return t

    def latex(self, **kwargs: dict) -> str:
        """Get latex string representation."""
        return "\\textit{t}"

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:  # noqa: D102
        if self.name in refs:
            refs[self.name].name = self.name
            return refs[self.name]
        return pt.scalar(self.name)

    def pt_str(self, **refs: dict[str, str]) -> str:  # noqa: D102
        if self.name in refs:
            return refs[self.name]
        return f'pt.scalar("{self.name}")'

    def __repr__(self) -> str:
        return '"t"'


# ==================================================
# SYSTEM DYNAMICS FANCY COMPONENTS
# ==================================================
# The actual things used to define system dynamics models, generally these are
# components guaranteed to have one single sub equation that defines how they
# change. These components also have the option to track historical eval
# outputs.


class TrackedReference(Reference):
    """A reference that keeps a matrix of all its values over a simulation for
    caching and reporting purposes. This is the base class for the fancier
    components: Flow, Variable, and Stock.

    Probably should be using one of the subclasses of this.

    A reminder that when Flows and Vars are evaluated, they are based on the
    current timestep, e.g.:

    ```
    flow1(t) = flow2(t) + var1(t)
    ```

    As opposed to stocks, whose update equations are based on timestep t-1.
    This means that flows and vars should not have circular references (causing
    infinite recursion errors.)

    Note:
        The shape of the value of a tracked reference depends on _sample_dim and
        _static with the following cases:

        1. if _static and not _sample_dim:
            * raw value     (float or int)
            * (dim,)        (when dim > 1)
        2. if _static and _sample_dim:
            * (sample,)
            * (sample, dim) (when dim > 1)
        3. if not _static:
            * (sample, t)
            * (sample, t, dim)  (when dim > 1)

        computed_mask similarly varies based on these cases:

        1. bool
        2. numpy array of bools with shape (sample,)
        3. numpy array of bools with shape (sample, timesteps)
    """

    def __init__(
        self,
        eq: EquationPart = None,
        label: str = None,
        doc: str = None,
        min: EquationPart = None,
        max: EquationPart = None,
        init: int | float | Distribution | Scalar | EquationPart = None,
        dim: int = 1,
        group: str = "",
        cgroup: str | list[str] = "",
        dtype: type = None,
    ):
        """Create a TrackedReference node, this is likely only called from subclasses.

        Args:
            eq (EquationPart): The equation that describes this reference.
            label (str): Visual label for the reference. In the context of a model,
                is set to the name assigned on the model if not explicitly provided.
            doc (str): A docstring/description explaining what this component is.
            min (EquationPart): An equation describing the minimum value of this
                reference.
            max (EquationPart): An equation describing the maximum value of this
                reference.
            init (int | float | Distribution | Scalar | EquationPart): The initial value
                for this component at ``t = 0``.
            dim (int): Size of an optional extra dimension, allowing a given reference
                to describe a vector of values at every timestep. Default is 1 implying
                no extra dimension.
            group (str): An optional string to help group related references together,
                primarily only used for visually tightening up elements in the
                stock/flow graphs.
            cgroup (str | list[str]): An optional string to refer to related elements and
                specify colors in the stock/flow graphs or easier hiding. Can specify a
                list to allow multiple ways of grouping.
            dtype (type): The underlying data type to use, e.g. ``int`` or ``float``.
        """
        super().__init__(label, doc)
        self.eq = eq
        self.group = group
        self.cgroup = cgroup

        self.min = ensure_scalar(min)
        self.max = ensure_scalar(max)

        self._requested_dtype = dtype
        self._dtype = dtype

        # use this to prevent infinite recursion in the case of historical
        # values being used.
        self._computing = []

        self.model = None
        """Keep a reference to container model, makes it easier to compare refs across
        multiple models."""

        self.init: int | float | Distribution | Scalar | EquationPart = ensure_scalar(
            init
        )
        """Initial value/equation for initial value for stock/flow/var at t=0"""

        self.dim: int = dim
        """Size of an optional extra dimension, allowing a given reference to describe
        a vector of values at every timestep. A value of 1 implying no extra dimension."""
        # NOTE: in general this is only meant to allow user to specify a
        # _requested_ shape, in almost every case internally we should be
        # checking .shape rather than .dim

        self._static: bool = None
        """If this is a reference that only needs to have a value calculated once and
        then never changes, don't track a full matrix for it.

        NOTE: this should only really be relevant for variables and possibly flows.

        Static effectively assumes this ref is a vector (or single value?)

        Note that this is currently being computed from the .populate() function.
        NOTE: _static is only meant for internal use for efficiency (to avoid constant
        round trips to utils.is_static()) during eval(), use is_static for all other
        cases.

        None is the default value to indicate staticness hasn't been assigned yet, was
        running into an issue in utils.is_static for variables with non-static _distribution_
        """

        self.implicit: bool = False
        """Implicit components (normally created as subcomponents of more advanced operations)
        don't show up in diagrams, latex, or output JSON, since they are created
        by operations."""

        self.computed_mask: np.ndarray | bool = False
        """Follows same shape as value (up through first two dimensions), set of booleans
        indicating which values have already been evaluated and saved."""

        self._sample_dim: bool = True
        """Only used for statics, if a static equation has no row-index relevant
        operations, no need to separately store an instance for every sample (the
        batch dimension.)"""

        self._computing_shape: bool = False
        """Use to avoid infinite recursion on get_shape in a stock etc."""
        self._computing_type: bool = False
        """Use to avoid infinite recursion on get_type in a stock etc."""

        # handle if within a context_manager, tell the manager to eventually
        # appropriately find the name for this reference and add it to the
        # model.
        if reno.Model.get_context() is not None:
            reno.Model.get_context()._unnamed_references.append(self)

    def min_refs(self) -> list:
        """Get any references found in the min constraint equation.

        Currently mostly only used to aid in diagrams.
        """
        if self.min is None:
            return []
        if isinstance(self.min, TrackedReference):
            return [self.min]
        return self.min.seek_refs()

    def max_refs(self) -> list:
        """Get any references found in the max constraint equation.

        Currently mostly only used to aid in diagrams.
        """
        if self.max is None:
            return []
        if isinstance(self.max, TrackedReference):
            return [self.max]
        return self.max.seek_refs()

    def _implied_eq(self, eq: EquationPart = None) -> EquationPart:
        """The full equation that takes min/max equations into account.

        Uses own equation if ``eq`` is not provided.
        """
        # TODO: this doesn't handle if max < min or min > max
        implied_eq = self.eq if eq is None else eq
        if self.max is not None:
            implied_eq = reno.ops.minimum(implied_eq, self.max)
        if self.min is not None:
            implied_eq = reno.ops.maximum(implied_eq, self.min)

        return implied_eq

    def qual_name(self, dot: bool = False) -> str:
        """Get a string with both the model and the reference name
        if this model is a submodel of something else.

        This is primarily used for helping distinguish things in a
        multimodel setup.
        """
        if self.model is not None and self.model.parent is not None:
            delim = "_" if not dot else "."
            return f"{self.model.name}{delim}{self.name}"
        return self.name

    def get_shape(self) -> int:
        """Get the size of the additional "data" dimension.

        Prefer using ``shape`` property over directly calling this function.

        If a dim was specified on this instance, that will be returned
        unless the the shape of the sub-equation output doesn't match.
        """
        if self._computing_shape:
            # acts as a placeholder in recursive instances
            return 1

        self._computing_shape = True
        eq = self._implied_eq()
        # print(f"{self.qual_name()} eq shape: {eq.shape}")

        if self.dim > 1 and eq.shape == 1:
            # we'd explicitly set a shape but not inherent in the equation
            self._computing_shape = False
            # print(f"{self.qual_name()} requested dim actually larger, so {self.dim}")
            return self.dim
        elif self.dim > 1 and eq.shape != self.dim:
            # We'd explicitly set a shape but the one inherent in the equation
            # is different
            raise Exception(f"Was expecting shape {self.dim} and got {eq.shape}")
        else:
            # No explicit shape requested, set it and use the one from the
            # equation
            self.dim = eq.shape
            # print(f"{self.qual_name()} setting dim to {eq.shape}")
            self._computing_shape = False
            return eq.shape

    def get_type(self) -> type:
        """Get the type of the target output of this equation expression.

        Prefer using ``dtype`` property over directly calling this function.

        If a dtype was specified on this instance, that type will be returned.
        """
        if self._requested_dtype:
            return self._requested_dtype

        if self._computing_type:
            # acts as a placeholder in recursive instances
            return None

        self._computing_type = True
        eq = self._implied_eq()
        type_ = eq.dtype
        self._computing_type = False
        return type_

    def populate(self, n: int, steps: int) -> None:
        """Initialize the matrix of values with size ``n x steps``.

        Args:
            n (int): The number of samples to simulate.
            steps (int): How many steps will be run in the simulation.
        """
        # TODO: not sure if auto checking for staticness _here_ is the correct
        # place
        self._determine_if_static()
        dtype = self.dtype
        if dtype is None:
            dtype = float

        # TODO: not technically sure if this is necessary
        self.dim = self.shape  # ensure shape computation has run

        # value shape cases, see note in TrackedReference docstring
        try:
            if self._static and not self._sample_dim:
                # case 1, skip value, because assignment completely replaces value
                self.computed_mask = False
            elif self._static and self._sample_dim:
                # case 2
                self.computed_mask = np.zeros((n,), dtype=bool)
                if self.dim == 1:
                    self.value = np.zeros((n,), dtype=dtype)
                    # self.value = np.zeros((n, 1), dtype=dtype)
                else:
                    self.value = np.zeros((n, self.dim), dtype=dtype)
            else:
                # case 3, non-static, always uses both sample and time dims
                self.computed_mask = np.zeros((n, steps), dtype=bool)
                if self.dim == 1:
                    self.value = np.zeros((n, steps), dtype=dtype)
                    # self.value = np.zeros((n, steps, 1), dtype=dtype)
                else:
                    self.value = np.zeros((n, steps, self.dim), dtype=dtype)
            self.initial_vals()
        except Exception as e:
            e.add_note(f"Was trying to populate {self.qual_name()}")
            raise

    def _determine_if_static(self) -> None:
        """Used to help improve efficiency of static checks since done every
        single eval call.

        Assigns _static and _sample_dim flags.
        """
        self._static = (
            not isinstance(self, Stock)
            and is_static(self.eq)
            and is_static(self.min)
            and is_static(self.max)
        )
        # TODO: unclear if I'm missing other potential conditions here
        # TODO: check to make sure this is actually true when directly assigned
        # a distribution, e.g. Variable(ops.Normal())
        self._sample_dim = (
            len(self.find_parts_of_type(Distribution)) > 0
            or len(self.find_parts_of_type(Piecewise)) > 0
            or not self._static
        )
        # including self._static condition for completeness, technically
        # sample dim always exists when not static

    def resolve_init_array(
        self, obj_or_eq: int | float | np.ndarray | EquationPart
    ) -> int | float | np.ndarray:
        """Convert a number or scalar/distribution into correct starting array.

        Can be used within initial_vals subclass definition.
        """
        if isinstance(obj_or_eq, (int, float)):
            # technically redundant? Covered by case outside of conditionals
            return obj_or_eq
        if isinstance(obj_or_eq, Distribution):
            if obj_or_eq.per_timestep:
                obj_or_eq.populate(
                    self.value.shape[0], steps=self.value.shape[1], dim=self.dim
                )
            else:
                obj_or_eq.populate(self.value.shape[0], dim=self.dim)
            return obj_or_eq.eval(0)
        if isinstance(obj_or_eq, EquationPart):
            # this covers Scalar
            return obj_or_eq.eval(0)
        return obj_or_eq

    def initial_vals(self) -> None:
        """Assign the starting values for t=0, this is the first step of the simulation.

        Subclasses that have a special way of defining the initial set of
        values (e.g. stocks with their _0 values) can override this function,
        automatically called at the end of ``populate()``.
        """
        # define in subclass if relevant

    def eval(  # noqa: C901
        self,
        t: int = 0,
        save: bool = False,
        force: bool = False,
        **kwargs: dict,
    ) -> int | float | np.ndarray:
        """Compute the equation for the given timestep. If we've already computed
        for this step (indicated by a value in the matrix that is not NaN), return
        that instead (unless told otherwise with ``force=True``).

        Note that evaluation output is only stored in the tracking matrix if
        save is ``True``, this class and subclasses are why save is being
        passed through all other ``.eval()`` methods.
        """
        try:
            # dims = [slice(None, None), t]  # value[:, t]
            # # if self.dim > 1:
            # #     dims = [slice(None, None), t, slice(None, None)]  # value[:, :, t]
            # if row_indices is not None:
            #     dims[0] = row_indices  # value[row_indices, t] or value[row_indices, :, t]
            # if self._static:
            #     dims = dims[:-1]  # value[row_indices] or value[row_indices, :]
            # dims = tuple(dims)
            # print(self.qual_name(), dims, t, save)
            # if isinstance(self, Stock):
            #     print(":", self.value[dims])

            # instead of samples_dim, we want the initial value to simply be _all_
            # samples_dim = slice(None, None) if row_indices is None else row_indices

            # check to see if we can short-circuit computation as long as
            # a re-compute wasn't explicitly requested
            if not force and self.value is not None:
                # value/computed_mask shape cases
                if self._static and not self._sample_dim:  # noqa: SIM102
                    # case 1, raw or (dim,)
                    if self.computed_mask:
                        return self.value
                if self._static and self._sample_dim:  # noqa: SIM102
                    # case 2, (sample,) or (sample, dim)
                    if self.computed_mask.all():
                        return self.value[:]

                if not self._static:  # noqa: SIM102
                    # case 3, (sample, t) or (sample, t, dim)
                    if self.computed_mask[:, t].all():
                        return self.value[:, t]

            # short-circuit not appropriate, run actual compute
            self._computing.append(t)
            if isinstance(self, Stock):
                # Reminder that stocks are always the end value at t-1, while
                # flows and variables are current timestep computations. See
                # docstring of this class.
                val = self._implied_eq().eval(t - 1, save, force, **kwargs)
                # TODO: if self is stock, don't pass save on down?
                # (should only be calling eval from model which is already handling
                # saving everything?)
            else:
                val = self._implied_eq().eval(t, save, force, **kwargs)

            # handle broadcasting value to correct size if necessary, this is
            # really only so that the first eval call returns the same thing as
            # the second in cases where there's an implicit broadcast requested
            # (e.g. Variable(5, dim=6) needs to return [5, 5, 5, 5, 5, 5] both
            # times)
            if self.dim > 1:
                if not isinstance(val, np.ndarray):
                    val = np.array([val])
                if val.shape[-1] == 1:
                    val = np.repeat(val, repeats=self.dim, axis=-1)

            # handle caching/storing result on the internal value for future
            # short-circuting and post-run analysis/dataset collection
            if save and not force:
                # value shape cases for value _assignment_
                if self._static and not self._sample_dim:
                    # case 1, simplest, just replace
                    # TODO: NOTE: that before we were always converting to a
                    # numpy array if was a static value - will not doing that
                    # break things? I don't think so because of how initial_vals
                    # used to work, which was directly setting value
                    self.value = val
                    self.computed_mask = True
                elif self._static and self._sample_dim:
                    # case 2, sample dimension is assigned to
                    assignment_dims = [slice(None, None)]
                    if self.dim > 1:
                        # this is only necessary because if val is a numpy array
                        # and we're assigning to specific rows, it's not going
                        # to automatically broadcast the full array, it'll prob
                        # shape error.
                        assignment_dims.append(slice(None, None))
                    self.value[*assignment_dims] = val
                    self.computed_mask[:] = True
                else:
                    # case 3, non-statics
                    assignment_dims = [slice(None, None), t]
                    if self.dim > 1:
                        # this is only necessary because if val is a numpy array
                        # and we're assigning to specific rows, it's not going
                        # to automatically broadcast the full array, it'll prob
                        # shape error.
                        assignment_dims.append(slice(None, None))
                    self.value[*assignment_dims] = val
                    self.computed_mask[:, t] = True

            self._computing.remove(t)
            return val
            # return val
        except Exception as e:
            e.add_note(f'Was evaluating "{self.name}": {self._implied_eq()}')
            if self._implied_eq() is None:
                e.add_note(
                    f'\tIt looks like "{self.name}" was None? Make sure all refs have a value or equation'
                )
            raise

    def history(self, index_eq: EquationPart) -> HistoricalValue:
        """Get a reference to a previous value of this reference.

        Args:
            index_eq (EquationPart): An equation defining the timestep in this ref's
                historical values to use. Note that currently, pymc conversions only
                support static time-based equations, e.g. ``t - 3``.
        """
        return HistoricalValue(self, index_eq)

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:  # noqa: D102
        if self.qual_name() in refs:
            if refs[self.qual_name()].name is None:
                refs[self.qual_name()].name = self.qual_name()
            return refs[self.qual_name()]
        return pt.scalar(self.qual_name())

    def pt_str(self, **refs: dict[str, str]) -> str:  # noqa: D102
        if self.qual_name() in refs:
            return refs[self.qual_name()]
        return f'pt.scalar("{self.qual_name()}")'

    def plot(
        self, ax: matplotlib.axes.Axes = None, **figargs: dict
    ) -> matplotlib.axes.Axes:
        """Plot (or add to passed axes) this reference's values. Note that this entails a
        simulation having already run.

        Note that this will plot as a distribution for variables.

        Args:
            ax (matplotlib.axes.Axes): Optional, add this reference's values to a
                pre-existing plot.
            **figargs (dict): If no axes passed, use these args in the
                ``plt.subplots(**figargs)`` call.
        """
        if figargs is None:
            figargs = {}
        # technically the compare_seq/compare_posterior functions are expecting
        # xarray datasets - we're cheating this by just setting up an
        # appropriately weird dictionary:
        # TODO: this prob doesn't work for static values?
        data = {self.qual_name(): {self.qual_name(): self.value}}
        if isinstance(self, Variable):
            ax = reno.viz.compare_posterior(
                self.qual_name, traces=data, ax=ax, **figargs
            )
        else:
            ax = reno.viz.compare_seq(self.qual_name, traces=data, ax=ax, **figargs)
        return ax

    def to_dict(self) -> dict:
        """Serialize class into a dictionary for saving to file."""
        return {
            "label": self.label,
            "init": str(self.init),
            "doc": self.doc,
            "min": str(self.min),
            "max": str(self.max),
            "group": self.group,
            "eq": str(self.eq),
            "implicit": self.implicit,
        }

    def from_dict(self, data: dict, refs: dict[str, TrackedReference]) -> None:
        """Deserialize reference and parse data from dictionary previously saved from
        ``to_dict()``.
        """
        self.label = data["label"]
        self.init = reno.parser.parse(data["init"], refs)
        self.doc = data["doc"]
        self.min = reno.parser.parse(data["min"], refs)
        self.max = reno.parser.parse(data["max"], refs)
        self.group = data["group"]
        self.implicit = data["implicit"]
        if "eq" in data:
            print(f"Trying to parse: {data['eq']}")
            self.eq = reno.parser.parse(data["eq"], refs)

    def __repr__(self) -> str:
        return f'"{self.qual_name()}"'


class HistoricalValue(Reference):
    """A wrapper class for a reference, specifically for getting a previous value
    indexed by some other equation.
    """

    def __init__(self, tracked_ref: TrackedReference, index_eq: EquationPart):
        """Create a node to refer to a previous value of a tracked reference.

        Prefer using :py:func:`reno.components.TrackedReference.history` to direct
        initialization.

        Args:
            tracked_ref (TrackedReference): The reference to get a previous value for.
            index_eq (EquationPart): The equation to compute what index to retrieve
                from.
        """
        super().__init__(label=tracked_ref.label)
        self.index_eq = ensure_scalar(index_eq)
        self.tracked_ref = tracked_ref

    def eval(  # noqa: D102
        self,
        t: int = 0,
        save: bool = False,
        force: bool = False,
        **kwargs: dict,
    ) -> int | float | np.ndarray:
        if t not in self.tracked_ref._computing:
            # the conditional is to prevent infinite recursion.
            # This piece is _really_ important, otherwise delays can mean
            # that certain timesteps for a variable/flow are never computed,
            # leading to inaccuracies and confusing table outputs. (esp if
            # an equation only ever refers to some variable's _history_, and
            # the variable itself (at timestep t) is never used)
            # So, make sure that the actual output of the actual flow/variable
            # is always "up to date"
            self.tracked_ref.eval(t, save, force, **kwargs)

        # we ensure the reference has actually evaluated because otherwise
        # you might have a bunch of random zeros near the end of a simulation

        index = self.index_eq.eval(t, save, force, **kwargs)
        # TODO: TODO: eventually need to actually handle vector indexing
        # print(self.qual_name(), index)
        if isinstance(index, np.ndarray):
            # index = int(index[0])
            index = index.astype(int)
        if isinstance(index, int) and index < 0:
            return 0  # TODO: TODO: TODO: or initial condition?
        # TODO: also conditional for < 0 for array case?

        if self.tracked_ref._static:
            return self.tracked_ref.value
        # return self.tracked_ref.value[:, index]
        return self.tracked_ref.value[
            list(range(0, self.tracked_ref.value.shape[0])), index
        ]

    def qual_name(self) -> str:
        """Get a string with both the model and the reference name
        if this model is a submodel of something else.

        This is primarily used for helping distinguish things in a
        multimodel setup.
        """
        # TODO: 2025.09.09, is this reasonable?
        return self.tracked_ref.qual_name()

    def get_shape(self) -> int:  # noqa: D102
        if (
            self.tracked_ref.shape != 1
            and self.index_eq.shape != 1
            and self.tracked_ref.shape != self.index_eq.shape
        ):
            raise Exception(
                "Index and tracked ref are both non-one and different, can't broadcast"
            )
        elif self.tracked_ref.shape != 1:
            return self.tracked_ref.shape
        elif self.index_eq.shape != 1:
            return self.index_eq.shape
        return self.tracked_ref.shape

    @property
    def model(self) -> reno.Model:
        """Get the model associated with this historical value."""
        return self.tracked_ref.model

    def latex(self, **kwargs: dict) -> str:
        """Get the string representation for referring to this reference, italicized
        and as a function of ``t`` to highlight it's a different timestep.
        """
        latex_str = (
            f"{latex_name(self.label, 'textit')}({self.index_eq.latex(**kwargs)})"
        )
        if "t" in kwargs:
            latex_str = latex_debug_output(self, latex_str, **kwargs)
        if "hl" in kwargs and kwargs["hl"] == self.tracked_ref.name:
            latex_str = "{\\color{cyan}" + latex_str + "}"
        return latex_str

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:  # noqa: D102
        force_array_index = (
            "__FORCE_ARRAY_INDEX__" in refs and refs["__FORCE_ARRAY_INDEX__"]
        )
        is_simple = check_for_easy_static_time_eq(self.index_eq)
        if is_simple and not force_array_index:
            simple_index = self.index_eq.eval(force=True) * -1
            name = f"{self.tracked_ref.qual_name()}_h{simple_index}"
        else:
            name = f"{self.tracked_ref.qual_name()}_h"
        if name in refs:
            if refs[name].name is None:
                refs[name].name = name
            if is_simple and not force_array_index:
                return refs[name]
            else:
                from_zero_index_eq = (
                    self.index_eq
                    + (refs["__PT_SEQ_LEN__"] - 1)
                    - self.model.get_timeref()
                ) % refs["__PT_SEQ_LEN__"]
                return refs[name][from_zero_index_eq.pt(**refs)]
                # return refs[name][self.index_eq.pt(**refs)]
        if self.shape == 1:
            return pt.as_tensor(0.0)
        else:
            return pt.as_tensor(0.0).repeat(self.shape)

    def pt_str(self, **refs: dict[str, str]) -> str:  # noqa: D102
        force_array_index = (
            "__FORCE_ARRAY_INDEX__" in refs and refs["__FORCE_ARRAY_INDEX__"]
        )
        is_simple = check_for_easy_static_time_eq(self.index_eq)
        if is_simple and not force_array_index:
            simple_index = self.index_eq.eval(force=True) * -1
            name = f"{self.tracked_ref.qual_name()}_h{simple_index}"
        else:
            name = f"{self.tracked_ref.qual_name()}_h"
        if name in refs:
            if is_simple and not force_array_index:
                return refs[name]
            else:
                from_zero_index_eq = (
                    self.index_eq
                    + (refs["__PT_SEQ_LEN__"] - 1)
                    - self.model.get_timeref()
                ) % refs["__PT_SEQ_LEN__"]
                return f"{refs[name]}[{from_zero_index_eq.pt_str(**refs)}]"
        if self.shape == 1:
            return "pt.as_tensor(0.0)"
        else:
            return f"pt.as_tensor(0.0).repeat({self.shape})"

    @staticmethod
    def parse(arg_strs: list[str], refs: dict[str, Reference]) -> HistoricalValue:
        """Convert a set of argument strings and references into a HistoricalValue node.

        Parsing these has to be handled differently from normal operations because
        it has to parse two separate references.
        """
        if len(arg_strs) != 2:
            raise SyntaxError(
                "history op expects two arguments, the tracked reference, and the indexing equation."
            )
        tracked_ref = reno.parser.parse(arg_strs[0], refs)
        index_eq = reno.parser.parse(arg_strs[1], refs)
        return HistoricalValue(tracked_ref, index_eq)

    def __repr__(self) -> str:
        return f"(history {self.tracked_ref.__repr__()} {self.index_eq.__repr__()})"


class Flow(TrackedReference):
    """A flow is a rate, a movement of some material/thing/quantity per some
    unit of time.

    Flow equations probably reference stocks, moving stuff from one stock
    to another etc. Flows are what stocks are defined in terms of (in_flows
    and out_flows.)

    https://insightmaker.com/docs/flows
    https://en.wikipedia.org/wiki/Stock_and_flow
    """

    def __init__(
        self,
        eq: EquationPart = None,
        label: str = None,
        doc: str = None,
        min: EquationPart = None,
        max: EquationPart = None,
        dim: int = 1,
        group: str = "",
        cgroup: str | list[str] = "",
        dtype: type = None,
    ):
        """Create a rate of change equation.

        Args:
            eq (EquationPart): The equation that describes a rate of change.
            label (str): Visual label for the Flow. In the context of a model, is set
                to the name assigned on the model if not explicitly provided.
            doc (str): A docstring/description explaining what this flow is.
            min (EquationPart): An equation describing the minimum value of this
                flow.
            max (EquationPart): An equation describing the maximum value of this
                flow.
            init (int | float | Distribution | Scalar | EquationPart): The initial value
                for this flow at ``t = 0``.
            dim (int): Size of an optional extra dimension, allowing a given reference
                to describe a vector of values at every timestep. Default is 1 implying
                no extra dimension.
            group (str): An optional string to help group related references together,
                primarily only used for visually tightening up elements in the
                stock/flow graphs.
            cgroup (str | list[str]): An optional string to refer to related elements and
                specify colors in the stock/flow graphs or easier hiding. Can specify a
                list to allow multiple ways of grouping.
            dtype (type): The underlying data type to use, e.g. ``int`` or ``float``.
        """
        super().__init__(
            eq,
            label,
            doc,
            min=min,
            max=max,
            dim=dim,
            group=group,
            cgroup=cgroup,
            dtype=dtype,
        )

    # ---- MATH OVERLOADING ----
    # for convenience, allows defining a chain of stocks/flows
    # e.g. f0 << s0 >> f1 >> s1
    # for simplistic chains this helps make sure you don't forget
    # to add a flow as both an inflow/outflow where necessary etc.

    def __rshift__(self, stock: Stock) -> Stock:
        """Return stock with this flow applied as an inflow.

        (``inflow >> stock``)
        """
        stock += self
        return stock

    def __lshift__(self, stock: Stock) -> Stock:
        """Return stock with this flow applied as an outflow.

        (``outflow << stock``)
        """
        stock -= self
        return stock

    # ---- /MATH OVERLOADING ----

    def initial_vals(self) -> None:
        """Variables can be set to distributions, so the inital vals
        in that case will be a population sampled from the eq distribution.
        """
        try:
            init_eq = self._implied_eq(self.init)
            if init_eq is None:
                init_eq = self._implied_eq()
            resolved_init_value = self.resolve_init_array(init_eq)

            # value shape cases for value assignment
            if self._static and not self._sample_dim:
                # case 1, simplest, raw value or (dim,)
                if self.dim > 1 and not isinstance(resolved_init_value, np.ndarray):
                    resolved_init_value = np.repeat(
                        [resolved_init_value], repeats=self.dim
                    )
                # ensure correct type (only needed in this case, other cases
                # don't re-assign container value)
                if isinstance(resolved_init_value, np.ndarray):
                    resolved_init_value = np.astype(resolved_init_value, self.dtype)
                else:
                    resolved_init_value = self.dtype(resolved_init_value)
                self.value = resolved_init_value
                self.computed_mask = True
            elif self._static and self._sample_dim:
                # case 2, sample dimension, (sample,) or (sample, dim)
                assignment_dims = [slice(None, None)]
                # assignment_dims.append(slice(None, None))
                if self.dim > 1:
                    # see note about why this is necessary in
                    # TrackedReference.eval assignment section
                    assignment_dims.append(slice(None, None))
                self.value[*assignment_dims] = resolved_init_value
                self.computed_mask[:] = True
            else:
                # case 3, non-statics
                assignment_dims = [slice(None, None), 0]
                # assignment_dims.append(slice(None, None))
                if self.dim > 1:
                    # see note about why this is necessary in
                    # TrackedReference.eval assignment section
                    assignment_dims.append(slice(None, None))
                self.value[*assignment_dims] = resolved_init_value
                self.computed_mask[:, 0] = True
        except Exception as e:
            e.add_note(f'Was attempting to compute initial values for "{self.name}"')
            raise

    def equation(self, **kwargs: dict) -> str:
        """Return the full latex string representation for the entire equation,
        not just the right hand side.
        """
        lhs = f"{latex_name(self.label, 'textit')}"
        if self.model.parent is not None:
            lhs += f"_{{{latex_name(self.model.label)}}}"
        lhs += "(t)"

        if self.eq is None:
            return f"{lhs} = \\texttt{{None}}"
        return f"{lhs} = {self.eq.latex(**kwargs)}" + range_eq_latex(
            self.min, self.max, **kwargs
        )

    def latex(self, **kwargs: dict) -> str:
        """Get the string representation for referring to this flow, italized
        and as a function of ``t`` to better represent that this should be a
        rate?
        """
        if self.implicit:
            return f"({self.eq.latex(**kwargs)})"
        latex_str = f"{latex_name(self.label, 'textit')}"
        if self.model.parent is not None:
            latex_str += f"_{{{latex_name(self.model.label)}}}"
        latex_str += "(t)"
        if "t" in kwargs:
            latex_str = latex_debug_output(self, latex_str, **kwargs)
        if "hl" in kwargs and kwargs["hl"] == self.qual_name():
            latex_str = "{\\color{cyan}" + latex_str + "}"
        return latex_str

    def display(self) -> None:
        """Display an ipython markdown with the equation latex."""
        # TODO: _repr_latex
        # https://github.com/jupyter/ngcm-tutorial/blob/master/Part-1/IPython%20Kernel/Custom%20Display%20Logic.ipynb
        display(Markdown(f"${self.equation()}$"))

    def debug_equation(self, t: int, sample: int = 0, **kwargs: dict) -> str:
        """Get a latex string output with the debug version of this equation."""
        return (
            self.latex(t=t, sample=sample, **kwargs)
            + " = "
            + self.eq.latex(t=t, sample=sample, **kwargs)
            + range_eq_latex(self.min, self.max, t=t, sample=sample, **kwargs)
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "eq" and value is not None:
            value = ensure_scalar(value)
            # assign ops are necessary when setting to a single 'thing' rather
            # than an equation. Without this, seek_refs becomes almost
            # impossible to implement correctly.
            if isinstance(value, (Variable, Flow, Stock, HistoricalValue)):
                value = reno.ops.assign(value)
            # Make sure sub_equation_parts is kept up to date with the equation.
            self.sub_equation_parts = [value]
            # force recomputing type/shape now
            self._dtype = None
            self._shape = None
        elif name in ("min", "max", "init"):
            value = ensure_scalar(value)
        object.__setattr__(self, name, value)


class Variable(TrackedReference):
    """A variable is a static value(s) or function that can be used as part
    of other equations, e.g. flow definitons.

    https://insightmaker.com/docs/variables
    """

    def __init__(
        self,
        eq: EquationPart = None,
        label: str = None,
        doc: str = None,
        min: EquationPart = None,
        max: EquationPart = None,
        dim: int = 1,
        user: bool = False,
        group: str = "",
        cgroup: str = "",
        dtype: type = None,
    ):
        """Create a variable reference.

        Args:
            eq (EquationPart): The equation that describes this variable.
            label (str): Visual label for the Variable. In the context of a model, is
                set to the name assigned on the model if not explicitly provided.
            doc (str): A docstring/description explaining what this variable is.
            min (EquationPart): An equation describing the minimum value of this
                variable.
            max (EquationPart): An equation describing the maximum value of this
                variable.
            init (int | float | Distribution | Scalar | EquationPart): The initial value
                for this variable at ``t = 0``.
            dim (int): Size of an optional extra dimension, allowing a given reference
                to describe a vector of values at every timestep. Default is 1 implying
                no extra dimension.
            user (bool): Used by explorer, whether to include this variable as an editable
                field in the front end.
            group (str): An optional string to help group related references together,
                primarily only used for visually tightening up elements in the
                stock/flow graphs.
            cgroup (str | list[str]): An optional string to refer to related elements and
                specify colors in the stock/flow graphs or easier hiding. Can specify a
                list to allow multiple ways of grouping.
            dtype (type): The underlying data type to use, e.g. ``int`` or ``float``.
        """
        # TODO: missing init parameter?
        self.user = user
        """If True, use visual interface to allow changing it via widgets."""
        # TODO: we're not currently using this in explorer, add that.

        if isinstance(eq, (float, int)):
            eq = Scalar(eq)

        super().__init__(
            eq,
            label,
            doc,
            min=min,
            max=max,
            dim=dim,
            group=group,
            cgroup=cgroup,
            dtype=dtype,
        )

    def initial_vals(self) -> None:
        """Assign the starting values for t=0, the first step of the simulation.

        Variables can be set to distributions, so the inital vals
        in that case will be a population samled from the eq distribution.
        """
        try:
            init_eq = self._implied_eq(self.init)
            if init_eq is None:
                init_eq = self._implied_eq()
            resolved_init_value = self.resolve_init_array(init_eq)

            # TODO: should eventually move type to equation part instead of here
            # exclusively

            # value shape cases for value assignment
            if self._static and not self._sample_dim:
                # case 1, simplest, raw value or (dim,)
                if self.dim > 1 and not isinstance(resolved_init_value, np.ndarray):
                    resolved_init_value = np.repeat(
                        [resolved_init_value], repeats=self.dim
                    )
                # ensure correct type (only needed in this case, other cases
                # don't re-assign container value)
                if isinstance(resolved_init_value, np.ndarray):
                    resolved_init_value = np.astype(resolved_init_value, self.dtype)
                else:
                    resolved_init_value = self.dtype(resolved_init_value)
                self.value = resolved_init_value
                self.computed_mask = True
            elif self._static and self._sample_dim:
                # case 2, sample dimension, (sample,) or (sample, dim)
                assignment_dims = [slice(None, None)]
                # assignment_dims.append(slice(None, None))
                if self.dim > 1:
                    # see note about why this is necessary in
                    # TrackedReference.eval assignment section
                    assignment_dims.append(slice(None, None))
                self.value[*assignment_dims] = resolved_init_value
                self.computed_mask[:] = True
            else:
                # case 3, non-statics
                assignment_dims = [slice(None, None), 0]
                # assignment_dims.append(slice(None, None))
                if self.dim > 1:
                    # see note about why this is necessary in
                    # TrackedReference.eval assignment section
                    assignment_dims.append(slice(None, None))
                self.value[*assignment_dims] = resolved_init_value
                self.computed_mask[:, 0] = True
        except Exception as e:
            e.add_note(f'Was attempting to compute initial values for "{self.name}"')
            raise

    def equation(self, **kwargs: dict) -> str:
        """Get the representation of the full equation for a variable as a latex
        string.
        """
        lhs = f"{latex_name(self.label)}"
        if self.model.parent is not None:
            lhs += f"_{{{latex_name(self.model.label)}}}"

        if self.eq is None:
            return f"{lhs} = \\texttt{{None}}"
        return f"{lhs} = {self.eq.latex(**kwargs)}" + range_eq_latex(
            self.min, self.max, **kwargs
        )

    def debug_equation(self, t: int, sample: int = 0, **kwargs: dict) -> str:
        """Get a latex string output with the debug version of this equation."""
        return (
            self.latex(t=t, sample=sample, **kwargs)
            + " = "
            + self.eq.latex(t=t, sample=sample, **kwargs)
            + range_eq_latex(self.min, self.max, t=t, sample=sample, **kwargs)
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "eq" and value is not None:
            value = ensure_scalar(value)
            # assign ops are necessary when setting to a single 'thing' rather
            # than an equation. Without this, seek_refs becomes almost
            # impossible to implement correctly.
            if isinstance(value, (Variable, Flow, Stock, HistoricalValue)):
                value = reno.ops.assign(value)
            # Make sure sub_equation_parts is kept up to date with the equation.
            self.sub_equation_parts = [value]
            # force recomputing type/shape now
            self._dtype = None
            self._shape = None
        elif name in ("min", "max", "init"):
            value = ensure_scalar(value)
        object.__setattr__(self, name, value)


class Stock(TrackedReference):
    """A stock represents some bucket or quantity of material/thing that
    can accumulate (in_flows) or deplete (out_flows) over time.

    https://en.wikipedia.org/wiki/Stock_and_flow
    https://insightmaker.com/docs/stocks

    Note that stock update equations are based on the previous timestep's values
    for all references (as opposed to flows/vars), e.g.:

    ``stock(t) = stock(t-1) + in_flows(t) - out_flows(t)``
    """

    def __init__(
        self,
        label: str = None,
        init: int | float | Distribution | Scalar | EquationPart = None,
        doc: str = None,
        min: EquationPart = None,
        max: EquationPart = None,
        dim: int = 1,
        group: str = "",
        cgroup: str | list[str] = "",
        dtype: type = None,
    ):
        """Create a stock component.

        Args:
            label (str): Visual label for the stock. In the context of a model, is set
                to the name assigned on the model if not explicitly provided.
            doc (str): A docstring/description explaining what this stock is.
            init (int | float | EquationPart): An equation, distribution, or scalar that
                defines initial conditions, or
            min (EquationPart): An equation describing the minimum value of this
                flow.
            max (EquationPart): An equation describing the maximum value of this
                flow.
            init (int | float | Distribution | Scalar | EquationPart): The initial value
                for this flow at ``t = 0``.
            dim (int): Size of an optional extra dimension, allowing a given reference
                to describe a vector of values at every timestep. Default is 1 implying
                no extra dimension.
            group (str): An optional string to help group related references together,
                primarily only used for visually tightening up elements in the
                stock/flow graphs.
            cgroup (str | list[str]): An optional string to refer to related elements and
                specify colors in the stock/flow graphs or easier hiding. Can specify a
                list to allow multiple ways of grouping.
            dtype (type): The underlying data type to use, e.g. ``int`` or ``float``.
        """
        super().__init__(
            label,
            doc=doc,
            min=min,
            max=max,
            dim=dim,
            init=init,
            group=group,
            cgroup=cgroup,
            dtype=dtype,
        )

        self.in_flows: list[Flow] = []
        self.out_flows: list[Flow] = []

    # ---- MATH OVERLOADING ----
    # this makes it so you can semantically define inflows/outflows with += and -=
    # e.g.
    # bath_water = Stock("bathtub")
    # bath_water += Flow("faucet")
    # bath_water -= Flow("drain")

    def __iadd__(self, obj: Flow | EquationPart | list[EquationPart | Flow]) -> Stock:
        """Return stock with specified flow(s) or equation(s) (implicit flow)
        as an inflow.
        """
        self.add_inflow(obj)
        return self

    def __isub__(self, obj: Flow | list[Flow]) -> Stock:
        """Return stock with specified flow applied as an outflow."""
        self.add_outflow(obj)
        return self

    # for convenience, allows defining a chain of stocks/flows
    # e.g. f0 << s0 >> f1 >> s1
    # for simplistic chains this helps make sure you don't forget
    # to add a flow as both an inflow/outflow where necessary etc.

    def __rshift__(self, flow: Flow | list[Flow]) -> Flow | list[Flow]:
        """Return flow, having applied as an outflow to this stock.

        (``stock >> outflow``)
        """
        self -= flow
        return flow

    def __rrshift__(
        self, flow: Flow | EquationPart | list[EquationPart | Flow]
    ) -> Stock:
        """Return stock with flow(s) or equation(s) (implicit flow) as an inflow.

        (``inflow >> stock``)
        """
        self += flow
        return self

    def __lshift__(
        self, flow: Flow | EquationPart | list[EquationPart | Flow]
    ) -> Flow | EquationPart | list[EquationPart | Flow]:
        """Return flow(s), having applied as an inflow to this stock.

        (``stock << inflow``)
        """
        self += flow
        return flow

    # ---- /MATH OVERLOADING ----

    @property
    def outflows(self) -> Operation:
        """Returns operation representing the sum of the flows leaving this stock."""
        return reno.ops.outflows(self)

    @property
    def space(self) -> Operation:
        """Returns an operation that computes how much space is remaining in this stock
        after taking into account any outflows for this timestep.

        Note that this requires a ``max`` value to be set on this stock.
        """
        if self.max is None:
            warnings.warn(
                f"The ``space`` operation requires a maximum to be set on the stock, specify ``{self.qual_name()}.max``"
            )
        return self.max - self + self.outflows

    # TODO: "operation" for "immediate space", just max - self?

    def add_inflow(self, obj: Flow | EquationPart | list[EquationPart | Flow]) -> None:
        """Add the flow or equation (creating an implicit flow) or a list of either as
        inflow(s) to this stock.
        """
        if isinstance(obj, Flow):
            self.in_flows.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                self.add_inflow(item)
        else:
            # create implicit inflow
            implicit_inflow = Flow(obj)
            implicit_inflow.implicit = True
            implicit_inflow._implicit_target = self
            implicit_inflow._implicit_target_index = len(self.in_flows)
            name = "_implicit_inflow_" + str(id(implicit_inflow))
            implicit_inflow.name = name
            self.add_inflow(implicit_inflow)
            if self.model is not None:
                self.model.add(name, implicit_inflow)

    def add_outflow(self, obj: Flow | list[Flow]) -> None:
        """Add the specified flow to the set of outflows on this stock."""
        if isinstance(obj, Flow):
            self.out_flows.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                self.add_outflow(item)
        else:
            raise TypeError(
                "Only flows and lists of flows can currently be specified as a stock outflow."
            )

    def initial_vals(self) -> None:
        """Assign the starting values for t=0, the first step of the simulation."""
        try:
            if self.init is not None:
                self.value[:, 0] = self.resolve_init_array(self._implied_eq(self.init))
            else:
                self.value[:, 0] = 0
            self.computed_mask[:, 0] = True
        except Exception as e:
            e.add_note(f'Was attempting to compute initial values for "{self.name}"')
            raise
        self.eq = self.compute_diff_eq()

    def equations(self, **kwargs: dict) -> list[str]:
        """Get a list of string latex representations for all in and out flows."""
        lhs = f"{latex_name(self.label)}"
        if self.model.parent is not None:
            lhs += f"_{{{latex_name(self.model.label)}}}"

        eqns = []
        if len(self.in_flows) > 0:
            eqns.append(
                f"{lhs} \\mathrel{{{{+}}{{=}}}} {self.combine_eqs(self.in_flows).latex(**kwargs)}"
            )
        if len(self.out_flows) > 0:
            eqns.append(
                f"{lhs} \\mathrel{{{{-}}{{=}}}} {self.combine_eqs(self.out_flows).latex(**kwargs)}"
            )
        if self.min is not None or self.max is not None:
            eqns.append(f"{lhs}{range_eq_latex(self.min, self.max, **kwargs)}")
        return eqns

    def debug_equation(self, t: int, sample: int = 0, **kwargs: dict) -> str:
        """Get a latex string output with the debug version of this equation."""
        return (
            self.latex(
                t=t + 1, sample=sample, **kwargs
            )  # TODO: minusone - prob need to change?
            + " = "
            + self.latex(t=t, sample=sample, **kwargs)
            + " + "
            + self.compute_diff_eq().latex(t=t, sample=sample, **kwargs)
        )

    def display(self) -> None:
        """Display a markdown with latex for inflow equations and outflow equations."""
        for eqn in self.equations():
            display(Markdown(f"${eqn}$"))

    def combine_eqs(self, eqs: list[EquationPart]) -> EquationPart:
        """Helper function to convert a list of equations into a single combined equation.

        (Essentially a big summation operator.)
        """
        # TODO: could be moved out
        if len(eqs) == 0:
            return Scalar(0)
        if len(eqs) == 1:
            return eqs[0]
        eq = eqs[0]
        for equation in eqs[1:]:
            eq = eq + equation
        return eq

    def compute_diff_eq(self) -> EquationPart:
        """Create a combined equation that accounts for both inflows and outflows."""
        inflows = self.combine_eqs(self.in_flows)
        outflows = self.combine_eqs(self.out_flows)

        if len(self.in_flows) == 0:
            return -outflows
        if len(self.out_flows) == 0:
            return inflows

        return inflows - outflows

    def _implied_eq(self, eq: EquationPart = None) -> EquationPart:
        """The implied equation for a stock is adding the full combination equation
        of inflows - outflows to the previous value (self).

        If an equation is passed to this function, simply add the min/max constraints
        like a normal ``_implied_eq``.
        """
        if eq is None:
            eq = self + self.compute_diff_eq()
        return super()._implied_eq(eq=eq)

    def to_dict(self) -> dict:
        """Serialize class into a dictionary for saving to file.

        Stock has to modify the parent TrackedReference class serialization to account
        for equations being handled a little differently.
        """
        tracked_ref_dict = super().to_dict()
        del tracked_ref_dict["eq"]
        tracked_ref_dict["in_flows"] = [str(flow) for flow in self.in_flows]
        tracked_ref_dict["out_flows"] = [str(flow) for flow in self.out_flows]
        return tracked_ref_dict

    def from_dict(self, data: dict, refs: dict[str, TrackedReference]) -> None:
        """Deserialize reference and parse data from dictionary previously saved from
        ``to_dict()``.
        """
        super().from_dict(data, refs)
        for in_flow in data["in_flows"]:
            # self.in_flows.append(reno.parser.parse(in_flow, refs))
            self.add_inflow(reno.parser.parse(in_flow, refs))
        for out_flow in data["out_flows"]:
            # self.out_flows.append(reno.parser.parse(out_flow, refs))
            self.add_outflow(reno.parser.parse(out_flow, refs))

    def __setattr__(self, name: str, value: Any) -> None:
        # TODO: this is mostly the same across all tracked refs, can prob move
        # to parent class
        if name == "min" or name == "max" or name == "init":
            # assign ops are necessary when setting to a single 'thing' rather
            # than an equation. Without this, seek_refs becomes almost
            # impossible to implement correctly.
            # TODO: very likely this needs to be applied to min/max as well, and
            # for variable/flow too.
            if name == "init" and value is not None:
                value = reno.ops.assign(value)
            value = ensure_scalar(value)
        object.__setattr__(self, name, value)


# ==================================================
# MEASUREMENTS
# ==================================================


class Metric(Reference):
    """An equation for a measurement or observation on the overall simulation.

    These run in a separate after-simulation analysis.

    Generally intended to be a series aggregate type equation (e.g. using
    series_min/series_max/sum on slices of values across time from references computed
    during the simulation etc.
    """

    def __init__(self, eq: EquationPart = None, label: str = None):
        """Create a metric equation node.

        Args:
            eq (EquationPart): The equation that describes this measurement/observable
            label (str): Visual label for the reference.
        """
        super().__init__(label)
        self.eq = eq
        self.model = None
        """Keep a reference to container model, makes it easier to compare refs across
        multiple models."""

        # handle if within a context_manager, tell the manager to eventually
        # appropriately find the name for this reference and add it to the
        # model.
        if reno.Model.get_context() is not None:
            reno.Model.get_context()._unnamed_references.append(self)

    def equation(self, **kwargs: dict) -> str:
        """Get the representation of the full equation for the metric as a latex
        string.
        """
        if self.eq is None:
            return f"{latex_name(self.label)} = \\texttt{{None}}"
        return f"{latex_name(self.label)} = {self.eq.latex(**kwargs)}"

    def debug_equation(self, t: int, sample: int = 0, **kwargs: dict) -> str:
        """Get a latex string output with the debug version of this equation."""
        return (
            self.latex(t=t, sample=sample, **kwargs)
            + " = "
            + self.eq.latex(t=t, sample=sample, **kwargs)
        )

    def qual_name(self, dot: bool = False) -> str:
        """Get a string with both the model and the reference name
        if this model is a submodel of something else.

        This is primarily used for helping distinguish things in a
        multimodel setup.
        """
        if self.model is not None and self.model.parent is not None:
            delim = "_" if not dot else "."
            return f"{self.model.name}{delim}{self.name}"
        return self.name

    def to_dict(self) -> dict:
        """Serialize class into a dictionary for saving to file."""
        return {
            "label": self.label,
            "eq": str(self.eq),
        }

    def from_dict(self, data: dict, refs: dict[str, TrackedReference]) -> None:
        """Deserialize reference and parse data from dictionary previously saved from
        ``to_dict()``.
        """
        self.label = data["label"]
        self.eq = reno.parser.parse(data["eq"], refs)

    def __repr__(self) -> str:
        return f'"{self.qual_name()}"'

    def __setattr__(self, name: str, value: Any):
        if name == "eq" and value is not None:
            if isinstance(value, (Variable, Flow, Stock, HistoricalValue)):
                value = reno.ops.assign(value)
            self.sub_equation_parts = [value]
        object.__setattr__(self, name, value)

    def eval(
        self,
        t: int = 0,
        save: bool = False,
        force: bool = False,
        **kwargs: dict,
    ) -> int | float | np.ndarray:
        """Compute the equation for the given timestep."""
        try:
            value = self.eq.eval(t, save, force, **kwargs)
            if save:
                self.value = value
            return value
        except Exception as e:
            e.add_note(f'Was evaluating metric "{self.name}": {self.eq}')
            raise
        return value

    def pt(self, **refs: dict[str, pt.TensorVariable]) -> pt.TensorVariable:
        """Get a pytensor graph representing this piece of an equation."""
        if self.qual_name() in refs:
            if refs[self.qual_name()].name is None:
                refs[self.qual_name()].name = self.qual_name()
            return refs[self.qual_name()]
        return pt.scalar(self.qual_name())

    def pt_str(self, **refs: dict[str, str]) -> str:
        """Construct a string containing relevant pytensor code for this piece
        of the equation. This is useful for "compiling" into pymc code.
        """
        if self.qual_name() in refs:
            return refs[self.qual_name()]
        # return self.eq.pt_str(**refs)
        return f'pt.scalar("{self.qual_name()}")'


class Flag(Metric):
    """Boolean value tracked for each step and sample in the simulation."""

    # NOTE: is this even necessary? You could achieve presumably the same thing
    # by having a variable store the same values based on a piecewise throughout
    # the sim itself.

    def __init__(self, eq: EquationPart = None, label: str = None):
        """Create a Flag metric node.

        Args:
            eq (EquationPart): The equation that describes the boolean flag.
            label (str): Visual label for the reference. In the context of a model,
                is set to the name assigned on the model if not explicitly provided.
        """
        super().__init__(eq, label)
        self.internal_step = 0
        self._computing = []
        # TODO: probably need to revise to similar to trackedrefs for how
        # computing is handled?

        # TODO: need to update first function to support pt as well
        # self.first = Reference(f"{self.label}.first")
        # self.first.eval = lambda t, save, force: self.first.eq.eval(t, save, force)
        # self.first.eq = Function(self.first_event)

    def populate(self, n: int, steps: int) -> None:
        """Initialize the matrix of values with size ``n x steps``.

        Args:
            n (int): The number of samples to simulate.
            steps (int): How many steps will be run in the simulation.
        """
        # TODO: TODO: this needs to act the same way as TrackedReference, using
        # NaN's instead of an internal step
        self.value = np.zeros((n, steps))
        self.internal_step = 0

    def eval(
        self,
        t: int = 0,
        save: bool = False,
        force: bool = False,
        **kwargs: dict,
    ) -> int | float | np.ndarray:
        """If the timestep is less than our internal_step tracker, return the
        corresponding previous column in the tracking matrix, since that
        timestep has already been computed. Otherwise, compute, store, and
        update the internal step to t+1.

        Note that evaluation output is only stored in the tracking matrix if
        save is ``True``, this class and subclasses are why save is being
        passed through all other ``.eval()`` methods.
        """
        self._computing.append(t)

        # TODO: if eq is just a scalar or distribution, don't do this,
        # no need to store a bunch of rows/cols of a guaranteed static thing.

        # check if already computed
        # TODO: don't think caching works for now?
        # if self.internal_step > t:
        #     self._computing.remove(t)
        #     return self.value[:, t]
        val = self.eq.eval(t, save, force, **kwargs).astype(int)
        if save:
            self.internal_step = t
            self.value[:, t] = val
            self.internal_step += 1
        self._computing.remove(t)
        return val

    # TODO: think through whether the rising edge is what actually makes most
    # sense for a property by this name.
    @property
    def indices(self) -> Operation:
        """Get equation for the timesteps where the value changes from 0 to 1.

        Note that this returns a "jagged" array, padded with NaN. If indexing into this
        array, you will likely need to use the nanindex op.
        """
        return reno.nonzero(reno.diff(self.value))

        # sample_indices, t_indices = np.where(np.diff(self.value) == 1)
        # t_indices = t_indices + 1
        #
        # indices = []
        # for i in range(self.value.shape[0]):
        #     i_indices = t_indices[np.where(sample_indices == i)[0]]
        #     if i_indices.shape[0] == 0:
        #         i_indices = np.asarray([np.nan])
        #     indices.append(i_indices)
        # return indices

    #
    # def first_event(self):
    #     """Get the timestep for the first time the value is 1."""
    #     indices = self.indices()
    #     firsts = [sample_indices[0] for sample_indices in indices]
    #
    #     return np.asarray(firsts)
    #
    # def last_event(self):
    #     """Get the timestep for the first time the value is 1."""
    #     indices = self.indices()
    #     firsts = [sample_indices[-1] for sample_indices in indices]
    #
    #     return np.asarray(firsts)
