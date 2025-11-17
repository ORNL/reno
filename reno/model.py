"""The main system dynamics model, handles tracking and running
stock and flow equations to run simulation(s)."""

import json
import math
import threading
import warnings
from copy import deepcopy
from typing import Any

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr
from graphviz import Digraph, set_jupyter_format
from pytensor import compile
from tqdm.auto import tqdm

import reno


class _ModelContexts(threading.local):
    """Similar to how PyMC manages this, we keep a model manager to allow models
    to be used as context managers, so a model name of ``my_really_long_model``
    doesn't have to be repeated over and over and over when defining all of the
    stocks and flows on it."""

    def __init__(self):
        self.current_models: list[Model] = []

    @property
    def current_model(self):
        if len(self.current_models) > 0:
            return self.current_models[-1]
        return None

    # TODO: is a parent_model property necessary?


# similar to PyMC, single global (thread-safe) list of current model contexts
# (e.g. within a with block)
MODEL_CONTEXTS = _ModelContexts()


class Model:
    """Class for a distinct, simulatable set of related stocks and flows.

    The expectation is to create a model instance, and then assign stocks,
    flows, variables, and submodels as attributes on that instance.

    Args:
        name (str): Optional name to give the model, should be used when
            submodels are in play, as the model name is used to help
            visually distinguish which model things belong to.
        n (int): The number of samples to simulate at once (by default).
        steps (int): How many time steps to run the simulation for (by default).
        label (str): Optional visual label to use when printing model things if
            cleaner than using the name.
        doc (str): Optional docstring to explain/describe the model.

    Example:
        >>> import reno

        >>> tub = reno.Model("Tub model")
        >>> tub.water_level = reno.Stock()
        >>> tub.faucet = reno.Flow(5)
        >>> tub.drain = reno.Flow(tub.water_level / 2)

        >>> tub.water_level += tub.faucet
        >>> tub.water_level -= tub.drain

        >>> tub(steps=50)
    """

    def __init__(
        self,
        name: str = None,
        n: int = 1,
        steps: int = 10,
        label: str = None,
        doc: str = None,
    ):
        self.stocks: list[reno.Stock] = []
        """List of all stocks associated with this model - don't modify directly,
        assign stocks as attributes directly on model (e.g. ``model.my_stock = reno.Stock()``)
        """
        self.flows: list[reno.Flow] = []
        """List of all flows associated with this model - don't modify directly,
        assign flows as attributes directly on model (e.g. ``model.my_flow = reno.Flow()``)
        """
        self.vars: list[reno.Variable] = []
        """List of all variables associated with this model - don't modify directly,
        assign vars as attributes directly on model (e.g. ``model.my_var = reno.Variable()``)
        """
        self.metrics: list[reno.components.Metric] = []
        """List of all metrics associated with this model - don't modify directly,
        assign metrics as attributes directly on model (e.g. ``model.my_metric = reno.Metric()``)
        """
        self.models: list[reno.Model] = []
        """List of submodels of this model - don't modify directly, assign submodels as
        attributes directly on model (e.g. ``model.my_submodel = reno.Model()``
        """

        self.parent: reno.Model = None
        """Parent model if applicable. This gets set in the __setattr__ when a submodel
        is assigned."""

        self.name: str = name
        if label is None:
            label = name
        self.label: str = label
        """String label to use in any visualizations/outputs to refer to this model. defaults
        to name if not used."""

        # not sure if I need these separately or not? Just thinking it
        # might make it easier to be able to set than to have to always pass in
        # simulate
        self.steps: int = steps
        """The number of timesteps to use by default in this model's simulations, can
        be overriden per model call."""
        self.n: int = n
        """The number of simulations to run in parallel by default, can be overriden
        per model call."""

        # previous n/steps variables to revert after a particular run.
        self.last_n: int = n
        self.last_steps: int = steps

        self.doc: str = doc
        """Docstring/comment to describe this model."""
        # TODO: offset?

        self.trace = None  # assigned by pymc()
        """Arviz trace produced by the last pymc run. This is used for plotting
        defaults on functions that aren't explicitly passed traces."""
        self.trace_RVs: list[str] = []
        """The set of string names of any random variables from the last pymc run."""

        self._unnamed_references: list = []
        """TrackedReferences added within a context manager go here so names can be
        assigned based on local frame variable names when it exits."""

        if Model.get_context() is not None:
            Model.get_context()._unnamed_references.append(self)

    @classmethod
    def get_context(cls):
        return MODEL_CONTEXTS.current_model

    def __enter__(self):
        MODEL_CONTEXTS.current_models.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        MODEL_CONTEXTS.current_models.pop()
        for index, ref in enumerate(self._unnamed_references):
            # make sure adding this ref wasn't already handled
            if isinstance(
                ref, (reno.components.TrackedReference, reno.components.Metric)
            ):
                if ref.name is not None and ref.model is not None:
                    continue
            elif isinstance(ref, reno.components.TrackedReference):
                if ref.name is not None and ref.parent is not None:
                    continue
            name = reno.utils._get_assigned_var_name(ref)
            if name is None and ref.implicit:
                name = "_implicit_ref_" + str(id(ref))
            setattr(self, name, self._unnamed_references[index])
        self._unnamed_references = []

    def _is_init_ref(self, name: str) -> bool:
        """Is the name of the stock/flow/variable reference an initial value reference?
        This allows referring to a reference's init with ``my_model.my_reference_0``.
        """
        if name.endswith("_0") and hasattr(self, name.removesuffix("_0")):
            return True
        return False

    def __setattr__(self, name, value):
        # It's easy to accidentally overwrite a reference with its equation (if
        # you forget to do `ref.eq`). Check for this behavior by checking to see
        # if the name being set _already exists_ and is a Flow/Variable, and
        # if so throw a warning
        if hasattr(self, name):
            if isinstance(
                getattr(self, name),
                (reno.components.Flow, reno.components.Variable),
            ):
                warnings.warn(
                    f"Reassigning entire model reference `{name}` with an equation, did you mean to instead assign the reference's equation, e.g. `{name}.eq = ...`?",
                    RuntimeWarning,
                )

        # Any refs/metrics/models assigned as an attribute should also be added
        # to the appropriate tracking lists
        if isinstance(value, reno.components.Stock) and value not in self.stocks:
            value.model = self
            self.stocks.append(value)
        elif isinstance(value, reno.components.Flow) and value not in self.flows:
            value.model = self
            self.flows.append(value)
        elif isinstance(value, reno.components.Variable) and value not in self.vars:
            value.model = self
            self.vars.append(value)
        elif isinstance(value, reno.components.Metric) and value not in self.metrics:
            value.model = self
            self.metrics.append(value)
        elif isinstance(value, Model) and not name == "parent" and not name == "model":
            # TODO: do we need to make a copy?
            value.parent = self
            self.models.append(value)

        # allow more ""organic"" assigning of init values with `my_model.my_stock_0 = 5`
        if self._is_init_ref(name):
            getattr(self, name.removesuffix("_0")).init = value
            return

        # Use the reference's name as its label in visual displays unless
        # otherwise requested. (e.g. if label is sepcified)
        # also set the reference's name to be the attribute you're setting as on
        # this model.
        if isinstance(
            value,
            (
                reno.components.Stock,
                reno.components.Flow,
                reno.components.Variable,
                reno.components.Metric,
                Model,
            ),
        ) and not (isinstance(value, Model) and (name == "parent" or name == "model")):
            # if isinstance(value, reno.components.Flow):
            #     if value.name is None and value.implicit:
            #         name = "_implicit_ref_" + str(id(value))
            if value.label is None or value.label == value.name:
                value.label = name
            value.name = name

        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        # allow more ""organic"" retrieval of init values, e.g. `my_model.my_stock_0`
        if name.endswith("_0"):
            if hasattr(self, name.removesuffix("_0")):
                return getattr(self, name.removesuffix("_0")).init
            raise AttributeError(name.removesuffix("_0") + " not found")
        if "." in name:
            # check for submodel (this works recursively)
            parts = name.split(".")
            if hasattr(self, parts[0]):
                return getattr(getattr(self, parts[0]), ".".join(parts[1:]))
        else:
            # NOTE: without this here, deepcopy fails, see
            # https://python-list.python.narkive.com/pxEGTJtL/folks-what-s-wrong-with-this
            raise AttributeError(name + " not found")

    def add(self, name: str, value: "reno.components.Reference | reno.Model"):
        """Add the passed tracked reference to the model with the provided name. This is used for
        programmatically adding stocks/flows in a context where the name is dynamically created and
        you can't simply ``model.my_name = ``."""
        # this is effectively just an alias for __setattr__ but I don't want
        # people to have to manually call model.__setattr__
        setattr(self, name, value)

    def all_stocks(self) -> list:
        """Get all stocks from this and all submodels."""
        full_list = [*self.stocks]
        for submodel in self.models:
            full_list.extend(submodel.all_stocks())
        return full_list

    def all_flows(self) -> list:
        """Get all flows from this and all submodels."""
        full_list = [*self.flows]
        for submodel in self.models:
            full_list.extend(submodel.all_flows())
        return full_list

    def all_vars(self) -> list:
        """Get all vars from this and all submodels."""
        full_list = [*self.vars]
        for submodel in self.models:
            full_list.extend(submodel.all_vars())
        return full_list

    def all_refs(self) -> list:
        """Get all stocks, flows, and vars, from this and all submodels in one giant
        ordered list (does not include metrics, use ``all_metrics()`` separately.)"""
        return self.all_stocks() + self.all_flows() + self.all_vars()

    def all_metrics(self) -> list:
        """Get all metrics recursively"""
        full_list = [*self.metrics]
        for submodel in self.models:
            full_list.extend(submodel.all_metrics())
        return full_list

    def _reset_type_and_shape_info(self):
        for ref in self.all_refs():
            ref._shape = None
            ref._dtype = None

    def _recursive_sub_populate_n_steps(self, n: int, steps: int):
        """Recursively populate all submodel's n/steps settings."""
        # TODO: this feels like it shouldn't be necessary and also doesn't
        # handle if a ref is in another model that wasn't explicitly set as a
        # submodel of this one?
        self.last_n = n
        self.last_steps = steps
        for model in self.models:
            model._recursive_sub_populate_n_steps(n, steps)

    def _populate(self, n: int, steps: int):
        """Initialize all tracked references with appropriately sized numpy
        matrices."""
        self._reset_type_and_shape_info()
        self._find_all_extended_op_implicit_components()
        self._recursive_sub_populate_n_steps(n, steps)

        ref_compute_order = self.dependency_compute_order(inits_order=True)

        for ref in ref_compute_order:
            ref.populate(n, steps)

    def _find_all_extended_op_implicit_components(self):
        """Go through every equation and assign any implicit components from
        extended operations."""
        extended_ops = []
        assoc_refs = []  # parallel list with which ref the correspo op came from
        # (not necessarily accurate, but we need an easy way to be able to name
        # separately)
        assoc_ref_counter = {}  # parallel list with how many times this ref is used
        # (useful for indexing the sub components to ensure no name conflicts)
        for ref in self.flows + self.vars:
            found_ops = ref.find_refs_of_type(reno.components.ExtendedOperation)
            for op in found_ops:
                if op not in extended_ops:
                    extended_ops.append(op)
                    assoc_refs.append(ref)
                    assoc_ref_counter[ref] = 0

        for i, op in enumerate(extended_ops):
            # NOTE: keep this above, we don't want the name to be _0 or we get
            # bad things because that would otherwise reference an init
            assoc_ref_counter[assoc_refs[i]] += 1
            for key, component in op.implicit_components.items():
                name = f"{assoc_refs[i].qual_name()}_{key}_{assoc_ref_counter[assoc_refs[i]]}"
                if hasattr(self, name):
                    delattr(self, name)
                self.add(name, component)

        # recursively do this for submodels too
        for model in self.models:
            model._find_all_extended_op_implicit_components()

    def simulator(
        self, n: int = None, steps: int = None, quiet: bool = False, debug: bool = False
    ):
        """An iterator to use for running the simulation step by step. Leaving n and/or
        steps None will use the model's default (as defined in constructor.)"""
        if n is None:
            n = self.n
        if steps is None:
            steps = self.steps

        self._recursive_sub_populate_n_steps(
            n, steps
        )  # TODO: (2025.07.28) isn't this redundant?
        # (should already be being handled in _populate?)
        self._populate(n, steps)

        # note that dependency_compute_order includes all submodels' refs
        ref_compute_order = self.dependency_compute_order(inits_order=False)

        for step in tqdm(range(1, steps), disable=quiet, desc=self.name):
            if debug:
                print("Beginning step", step, self.name)
            for ref in ref_compute_order:
                ref.eval(step, save=True)
            yield

        self.run_metrics(n, steps)

    def simulate(
        self, n: int = None, steps: int = None, quiet: bool = False, debug: bool = False
    ):
        """Run each step of the the full simulation. Leaving n and/or
        steps None will use the model's default (as defined in constructor.)"""
        for step in self.simulator(n, steps, quiet, debug):
            pass

    def run_metrics(self, n: int = None, steps: int = None):
        """Run all metric equations on a completed simulation. Calling this
        function assumes the full simulation has already run."""
        if n is None:
            n = self.n
        if steps is None:
            steps = self.steps

        metrics = reno.utils.dependency_compute_order([*self.all_metrics()])

        # populate any metrics (only flags apply to every timestep currently)
        for metric in metrics:
            if isinstance(metric, reno.components.Flag):
                metric.populate(n, steps)

        # compute flag boolean values for each sample at each timestep
        for step in range(1, steps):
            for metric in metrics:
                if isinstance(metric, reno.components.Flag):
                    metric.eval(step, True)

        # run any postmeasurement equations, usually 1 per sample
        for metric in metrics:
            if isinstance(metric, reno.components.Metric):
                metric.eval(steps - 1, True)
                # TODO: is - 1 correct? I don't think it is

    def graph(
        self,
        show_vars: bool = True,
        exclude_vars: list[str] = None,
        sparklines: bool = False,
        sparkdensities: bool = False,
        sparkall: bool = False,
        g: Digraph = None,
        traces: list[xr.Dataset] = None,
        universe: list[reno.components.TrackedReference] = None,
        lr: bool = False,
    ) -> Digraph:
        """Generate a graphviz dot graph for all the stocks and flows of the passed model,
        optionally including sparklines if a simulation has been run.

        Args:
            model (reno.model.Model): The model to collect stocks/flows/variables from.
            show_vars (bool): Whether to render variables in the diagram, or just stocks and flows
                (for very complex models, hiding variables can make it a bit easier to visually
                parse.)
            exclude_var_names (list[str]): Specific variables to hide in the diagram, can be used
                with ``show_vars=True`` to just show specific variables of interest.
            sparklines (bool): Draw mini graphs to the right of each stock showing plotting their
                values through time. This assumes the model has either been run, or traces are passed
                in manually with the ``traces`` argument.
            sparkdensities (bool): Draw mini density plots/histograms next to any variables that
                sample from distributions. This assumes the model has either been run, or traces
                are passed in manually with the ``traces`` argument.
            g (Digraph): Graphviz Digraph instance to render the nodes/edges on. Mostly only for
                internal use for drawing subgraphs for submodels.
            traces (list[xr.Dataset]): A list of traces or model run datasets to use for drawing
                spark plots. Each dataset will be rendered in a different color.
            universe (list[TrackedReference]): Limit rendered nodes to only those listed here, this
                includes all of stocks/flows/variables. (This acts as an initial filter, ``show_vars``
                and ``exclude_var_names`` still applies after this.)
            lr (bool): By default the graphviz plot tries to orient top-down. Specify ``True`` to
                try to orient it left-right.

        Returns:
            The populated Digraph instance (Jupyter can natively render this in a cell output.)
        """
        if exclude_vars is None:
            exclude_vars = []
        set_jupyter_format("png")
        diagram, _ = reno.diagrams.stock_flow_diagram(
            self,
            show_vars,
            exclude_vars,
            sparklines,
            sparkdensities,
            sparkall,
            g,
            traces,
            universe,
            lr,
        )
        return diagram

    def latex(self, docs: bool = True, t: int = None, sample: int = 0):
        """Get an interactive latex ipywidget listing all of the equations in system. Each equation
        line is clickable, clicking will highlight where else in the system that equation's result
        is being used.

        Args:
            docs (bool): Whether to include the documentation string beneath each equation or not.
            t (int): If specified, show the values of every reference at the specified timestep (note
                that stock values will be from the previous timestep)
            sample (int): Which sample (row) to show the values from if ``t`` was specified.
        """
        debug = False
        if t is not None:
            debug = True

        latex_obj = reno.viz.ModelLatex(self, docs, t=t, sample=sample, debug=debug)
        return latex_obj.widget

    def plot_stocks(self, cols: int = None, rows: int = None, **figargs):
        """Shortcut function to quickly get a set of graphs for each stock."""
        reno.viz.plot_refs(self.stocks, cols=cols, rows=rows, **figargs)

    def copy(self, name: str = None) -> "Model":
        """Make a separate copy of this model with the desired name.

        This is useful for making a separate model instance that you can modify the
        equations of for comparison against the original.
        """
        new_model = deepcopy(self)
        if name is not None:
            new_model.name = name
        return new_model

    def free_refs(self, recursive: bool = False):
        """Get all free "variables" (not component variables) for this model, or things that
        aren't defined in terms of anything else.

        A free variable is a variable with no references to other variables, stocks or flows.
        """
        free = []
        inits = []

        # TODO: the ref in seek_refs check probably then needs to see if the
        # init function is a scalar/dist/int/float, and similarly expand
        # conditional eq checks to include int/float

        if not recursive:
            # if not directly recursing, list each submodel itself as the parameter
            for model in self.models:
                free.append(model)
        else:
            # otherwise find and list each of the free variables within each submodel
            for model in self.models:
                free.extend(
                    [
                        f"{model.name}.{ref}"
                        for ref in model.free_refs(recursive)
                        if not ref.implicit
                    ]
                )

        # a variable is free if its equation is directly a scalar or distribution, or hasn't
        # been assigned an equation. Flows use same logic as variables here.
        for ref in self.vars + self.flows:
            if ref.implicit:
                continue

            if reno.utils.is_free_var(ref.eq):
                free.append(ref)

            # if a variable/flow is a function defined in terms of itself, needs an init.
            if ref in ref.seek_refs():
                inits.append(ref)

        # stock inits (_0) are normally free, would be weird if init was defined
        # in terms of another var?
        for stock in self.stocks:
            if reno.utils.is_free_var(stock.init):
                inits.append(stock)

        return [ref.name if hasattr(ref, "name") else ref for ref in free] + [
            f"{ref.name}_0" for ref in inits
        ]

    def get_docs(self, as_dict: bool = False) -> str:
        """Get a full docstring for using this model as a function, maybe useful for
        allowing a model to be used as a tool for LLMs?

        Use as_dict for submodels, so example string shows args as dictionary."""
        free_refs = self.free_refs()

        docstring = self.doc if self.doc is not None else ""

        default_args = ""
        arg_strings = []
        for key, value in self.config().items():
            value_to_print = value
            if isinstance(value, reno.Scalar):
                value_to_print = value.value
            arg_strings.append(f"{key}={value_to_print}")
        default_args = ", ".join(arg_strings)

        docstring += f"\n\nExample:\n\t{self.name}({default_args})"

        docstring += "\n\nArgs:"

        for ref_name in free_refs:
            if self._is_init_ref(ref_name):
                ref = getattr(self, ref_name.removesuffix("_0"))
            else:
                ref = getattr(self, ref_name)
            doc = ref.doc

            docstring += f"\n\t{ref_name}"
            if doc is not None:
                if self._is_init_ref(ref_name):
                    docstring += f": Initial value for {doc[:1].lower()}{doc[1:]}"
                else:
                    docstring += f": {doc}"

        return docstring

    def config(self, **free_refs) -> dict:
        """Get/set model configuration. This function allows specifying one or more
        free variables - anything not set uses the default.

        Returns:
            The resulting configuration dictionary, with the free ref names as keys.
        """
        available_free_refs = self.free_refs()

        # track all free variables for this sim, including those not explicitly
        # passed
        config = {}

        # initialize any vars/flows/stocks based on optionally passed references
        for ref_name in available_free_refs:
            # handle sub model configuration if applicable
            if isinstance(getattr(self, ref_name), Model):
                sub_model = getattr(self, ref_name)
                passed_sub_config = {}
                if ref_name in free_refs:
                    if not isinstance(free_refs[ref_name], dict):
                        raise AttributeError(
                            "sub model configuration must be passed as a sub dictionary"
                        )
                    passed_sub_config = free_refs[ref_name]
                sub_config = sub_model.config(**passed_sub_config)
                config[ref_name] = sub_config
            else:
                # any refs that _aren't_ an entire submodel
                if ref_name in free_refs:
                    # print(ref_name, "=", free_refs[ref_name])  # debug
                    if self._is_init_ref(ref_name):
                        init_eq_part = free_refs[ref_name]
                        if isinstance(init_eq_part, (int, float)):
                            init_eq_part = reno.components.Scalar(init_eq_part)
                        setattr(self, ref_name, init_eq_part)
                    else:
                        eq_part = free_refs[ref_name]
                        # TODO: dear lord there's almost certainly a cleaner way
                        # to do this...
                        if isinstance(eq_part, (int, float)):
                            eq_part = reno.components.Scalar(eq_part)
                        setattr(getattr(self, ref_name), "eq", eq_part)
                        # setattr(getattr(self, ref_name), "eq", free_refs[ref_name])
                    config[ref_name] = free_refs[ref_name]
                else:
                    val = getattr(self, ref_name)
                    # if val is None:
                    #     # val = 0  # saving this for posterity. BAD NATHAN. This
                    #     # causes asymmetric get vs set, e.g. I can't call
                    #     # config(**config()) without the end configuration being
                    #     # technically different. This was fine for Reno's
                    #     # underlying math system, but for PyMC where types
                    #     # become very important, it can accidentally convert
                    #     # what was initially assumed 0.0 to 0, (specifically
                    #     # this was causing stock _init values to become
                    #     # integers, completely screwing up the subsequent math)
                    if isinstance(val, reno.components.TrackedReference):
                        val = val.eq
                    config[ref_name] = val

        return config

    def get_nonrecursive_config(self):
        """Only get the free refs config from _this_ model, no submodels.

        Useful for assigning dataset attrs."""
        free_refs = self.free_refs()

        config = {}

        for ref_name in free_refs:
            if not isinstance(getattr(self, ref_name), Model):
                val = getattr(self, ref_name)
                if val is None:
                    val = 0
                if isinstance(val, reno.components.TrackedReference):
                    val = val.eq
                config[ref_name] = val

        return config

    def load_dataset(self, ds: xr.Dataset):
        """Take all the tracked ref sequence data and load them into the matching
        model's tracked refs stored values. This is useful for using ``.latex(t=...)``
        in debug mode for diving into a specific run."""
        unfound = []
        n = -1
        steps = -1

        # find n/steps
        for var in ds.variables:
            if len(ds.variables[var].shape) > 1:
                n = ds.variables[var].values.shape[0]
                steps = ds.variables[var].values.shape[1]
                break

        for ref in self.dependency_compute_order(inits_order=True) + self.metrics:
            found = False
            # find the corresponding variable in the dataset
            for var in list(ds.variables.keys()):
                # handle either . or _ parent model syntax in name
                # (pymc conversion uses _, default reno uses .)
                # NOTE: (2025-06-27) - switching to reno using _ as well, it's
                # too confusing otherwise.
                if var == ref.qual_name(False) or var == ref.qual_name(True):
                    if not isinstance(ref, reno.components.Metric):
                        ref.populate(
                            n, steps
                        )  # order shouldn't matter since replacing value anyway
                    ref.value = ds.variables[var].values
                    found = True
                    break

            if not found:
                unfound.append(ref)
                warnings.warn(
                    f"Loading values from dataset found no value for model reference '{ref.qual_name()}', value will be reset",
                    RuntimeWarning,
                )
                # TODO: order against dependency order?
                ref.populate(n, steps)

        # TODO: collect config as well?

    def dataset(self) -> xr.Dataset:
        """Turn all of the model's tracked reference values into an xarray dataset,
        including the "configuration" of the input parameters etc."""
        sub_dses = {}
        if len(self.models) > 0:
            for model in self.models:
                sub_dses[model.name] = model.dataset()

        # add non-static non-dim base refs (stocks, flows, vars)
        # partial case 1, (sample, t)  (see TrackedReference for cases
        # explanation)
        all_refs = self.stocks + self.flows + self.vars
        all_refs = [ref for ref in all_refs if not ref.implicit]
        ds = xr.Dataset(
            {
                ref.name: (["sample", "step"], ref.value)
                for ref in all_refs
                if not ref.is_static() and ref.dim == 1
            },
            coords={
                "sample": (["sample"], list(range(self.last_n))),
                "step": (["step"], list(range(self.last_steps))),
                # "vec": (["vec"], []),
            },
            attrs=self.get_nonrecursive_config(),
        )

        # handle any multi-dim refs
        multidim_vars = {}
        for ref in all_refs:
            if ref.dim > 1:
                vec_name = f"{ref.qual_name()}_vec"
                if ref.is_static():
                    if ref._sample_dim:
                        # case 2, (sample, dim)
                        dims = ["sample", vec_name]
                        coords = {
                            "sample": (["sample"], list(range(self.last_n))),
                            vec_name: ([vec_name], list(range(ref.dim))),
                        }
                        val = ref.value
                        if ref.value.shape[0] != self.last_n:
                            val = np.broadcast_to(ref.value, (self.last_n, ref.dim))
                    else:
                        # case 1, (dim,)
                        dims = ["sample", vec_name]
                        coords = {
                            "sample": (["sample"], list(range(self.last_n))),
                            vec_name: ([vec_name], list(range(ref.dim))),
                        }
                        val = ref.value
                        # TODO: I'm pretty sure this needs a repeat per last_n
                        # with axis=0...a broadcast may not work
                        val = np.broadcast_to(val, (self.last_n, ref.dim))
                else:
                    # partial case 1, (sample, t, dim)
                    dims = ["sample", "step", vec_name]
                    coords = {
                        "sample": (["sample"], list(range(self.last_n))),
                        "step": (["step"], list(range(self.last_steps))),
                        vec_name: ([vec_name], list(range(ref.dim))),
                    }
                    val = ref.value

                da = xr.DataArray(data=val, dims=dims, coords=coords)
                multidim_vars[ref.name] = da
        ds = ds.assign(multidim_vars)

        # handle any static refs (don't change wrt to step)
        # NOTE: this is only dealing with vectors right now, not single nums?
        static_refs = {}
        for ref in all_refs:
            if ref.is_static() and ref.dim == 1:
                val = ref.value
                # print(
                #     "(dataset)",
                #     ref.qual_name(),
                #     type(ref.value),
                #     ref.value.shape,
                #     ref.value,
                #     ref._static,
                #     ref._sample_dim,
                # )
                # TODO: geeeez there's got to be a better way?? np.min/max etc.
                # will return a numpy.int even if the inputs are not numpy
                # types (e.g. python int)
                if isinstance(ref.value, (int, float)) or (
                    isinstance(ref.value, np.ndarray) and len(ref.value.shape) == 0
                ):
                    val = np.broadcast_to(ref.value, (self.last_n,))
                elif len(ref.value.shape) > 0 and ref.value.shape[0] != self.last_n:
                    val = np.broadcast_to(ref.value, (self.last_n,))
                elif len(ref.value.shape) == 0:
                    val = np.broadcast_to(ref.value, (self.last_n,))
                static_refs[ref.name] = (["sample"], val)
        ds = ds.assign(static_refs)

        # add metrics, note that some metrics will be 1 per sample, others will
        # be 1 per step
        new_vars = {}
        for metric in self.metrics:
            if len(metric.value.shape) == 1:
                # 1 per sample
                coords = ["sample"]
            else:
                # 1 per step (e.g. flags)
                coords = ["sample", "step"]
            new_vars[metric.name] = (coords, metric.value)
        ds = ds.assign(new_vars)

        # merge in any sub datasets
        all_attrs = {}
        for sub_ds_name, sub_ds in sub_dses.items():
            # essentially flatten attr names
            renamed_attrs = {}
            for attr in sub_ds.attrs:
                renamed_attrs[f"{sub_ds_name}.{attr}"] = sub_ds.attrs[attr]
            sub_ds = sub_ds.drop_attrs(deep=False)
            sub_ds = sub_ds.assign_attrs(renamed_attrs)
            all_attrs.update(renamed_attrs)

            # update var names
            renamed_vars = {
                name: f"{sub_ds_name}_{name}" for name in list(sub_ds.keys())
            }
            sub_ds = sub_ds.rename_vars(renamed_vars)
            sub_dses[sub_ds_name] = sub_ds  # TODO: how was this not necessary before??

        ds_to_merge = [ds]  # , *list(sub_dses.values())]
        ds_to_merge.extend(sub_dses.values())
        ds_merged = xr.merge(ds_to_merge)
        ds_merged = ds_merged.assign_attrs(ds.attrs)
        ds_merged = ds_merged.assign_attrs(all_attrs)
        return ds_merged

    def __call__(
        self, n: int = None, steps: int = None, keep_config: bool = False, **free_refs
    ) -> xr.Dataset:
        """Run the model simulation, allowing specification of any free variables.
        Variables in submodels need to be defined as dictionaries.

        Example:
            >>> dataset = my_model(steps=20, some_free_var=reno.Normal(5, 2), my_submodel=dict(other_free_var=4))

        Args:
            n (int): Number of simulations to run in parallel, leave ``None`` to use the default
                set on the model.
            steps (int): Number of timesteps to run the simulation for, leave ``None`` to use the
                default set on the model.
            keep_config (bool): Whether to keep any changes made via free ref configurations passed
                in for subsequent simulations. The default is to not do this, keeping the original
                model unchanged.
            **free_refs: Definitions for equations or values for any variables or initial conditions
                in the system.
        """
        # store previous config vals and apply any requested config from call params
        previous = self.config()  # noqa: F841
        config = self.config(**free_refs)  # noqa: F841

        if n is None:
            n = self.n
        if steps is None:
            steps = self.steps

        # run the simulation
        self.simulate(n, steps)

        ds = self.dataset()

        # revert config unless explicitly requested to keep it
        if not keep_config:
            self.config(**previous)

        return ds

    def find_timeref_name(self) -> str:
        """If a TimeRef is ever used in an equation, get its reference name. This is
        necessary for the pymc code construction to ensure the correct variable name is
        injected.

        Returns ``None`` if no TimeRef is used.
        """
        all_refs = []
        for flow in self.all_flows():
            all_refs.extend(flow.eq.seek_refs())
        for var in self.all_vars():
            all_refs.extend(var.eq.seek_refs())

        for ref in all_refs:
            if isinstance(ref, reno.components.TimeRef):
                return ref.name

        return None

    def get_timeref(self) -> reno.components.TimeRef:
        all_refs = []
        for flow in self.all_flows():
            all_refs.extend(flow.eq.seek_refs())
        for var in self.all_vars():
            all_refs.extend(var.eq.seek_refs())

        for ref in all_refs:
            if isinstance(ref, reno.components.TimeRef):
                return ref

    def dependency_compute_order(
        self, inits_order: bool = False, debug: bool = False
    ) -> list[reno.components.TrackedReference]:
        """Find a dependency-safe ordering for reference equations by iterating through
        and each time adding the first reference that doesn't depend on any references
        not yet added.

        This function will detect circular references (equations that depend on eachother)
        and throw an error. Primary use for this function is to correctly set order of
        equations in the pymc model/step function, but this is used in normal Reno math too.

        Args:
            inits_order (bool): Include stock init equations in the ordering. (When False,
                stocks will always be listed first since they always depend on a previous
                timestep's values. This is not the case for inital equations.)

        Returns:
            The dependency-ordered list of reno TrackedReferences for this model.
        """
        compute_order = []
        # TODO: do we need to ensure _find_all_extended_op_implicit_components
        # is called here?

        if not inits_order:
            # if this isn't an initial case, we assume stocks are always
            # computed first, and this won't cause dependency issues because
            # they're always based in previous values.
            rel_compute_order = reno.utils.dependency_compute_order(
                self.all_flows() + self.all_vars(), init_eqs=False, debug=debug
            )
            compute_order.extend(self.all_stocks())
            compute_order.extend(rel_compute_order)
        else:
            compute_order = reno.utils.dependency_compute_order(
                self.all_stocks() + self.all_flows() + self.all_vars(),
                init_eqs=True,
                debug=debug,
            )

        return compute_order

    def pymc_model(
        self, observations: list["reno.ops.Observation"] = None, steps: int = None
    ) -> pm.model.core.Model:
        """Generate a pymc model for bayesian analysis of this system dynamics model. The general
        idea is that this creates corresponding pymc variables (or distributions as relevant) for
        each stock/flow/var in the model, and sets up the full simulation sequence computations
        based on the generated step function from ``_pt_step()``.

        Sampling with priors should be equivalent to running the system dynamics model normally
        (this is essentially "forward simulation mode".) Add observations to the pymc model
        variables and sample from posterior predictive to run bayesian analysis/determine how
        distributions of any other variables may be affected.
        """
        return reno.pymc.to_pymc_model(self, observations, steps)

    # TODO: wrap in function option (string function would return model)
    # TODO: option to add in necessary imports
    # TODO: can you run black formatting programmatically on a string?
    def pymc_str(
        self, observations: list["reno.ops.Observation"] = None, steps: int = None
    ) -> str:
        """Construct a string of python code to create a pymc model wrapping this system dynamics
        model. Should be a functional (string) equivalent of the ``pymc_model()`` function. Includes
        the output from ``pymc.pt_sim_step_str(self)``.

        Expected imports for the resulting code to run:
            >>> import pytensor
            >>> import pytensor.tensor as pt
            >>> from pytensor.ifelse import ifelse
            >>> import pymc as pm
            >>> import numpy as np
        """
        return reno.pymc.to_pymc_model_str(self, observations, steps)

    def pymc(  # noqa: C901
        self,
        n: int = None,
        steps: int = None,
        sampling_kwargs: dict[str, Any] = None,
        compile_kwargs: dict[str, Any] = None,
        compile_faster: bool = False,
        observations: list["reno.ops.Observation"] = None,
        smc: bool = True,
        trace_prior: az.InferenceData = None,
        compute_prior_only: bool = False,
        keep_config: bool = False,
        **free_refs,
    ) -> az.InferenceData:
        """A PyMC equivalent version of a model's __call__, convert the model to PyMC and run the
        simulation/Bayesian analysis.

        Args:
            n (int): Number of simulations to run in parallel, leave ``None`` to use the default
                set on the model.
            steps (int): Number of timesteps to run the simulation for, leave ``None`` to use the
                default set on the model.
            sampling_kwargs (dict): Arguments to pass to the PyMC sampler. Uses ``sample_smc`` by
                default unless the ``smc`` argument is ``False``, in which case it lets PyMC choose
                sampler (NUTS for continuous variables, some variant of Metropolis for discrete.)
            compile_kwargs (dict): Arguments to pass along to the compiler through PyMC. E.g. ``mode``,
                which, if function compilation is taking forever, can for instance be set to
                ``FAST_COMPILE``.
            compile_faster (bool): For some large/complex models, the PyMC/pytensor compilation step can
                take excessively long as it tries to apply a high level of optimizations. Set this to
                ``True`` to bump down the optimization level by one, which should dramatically speed this
                step up. Alternatively you can have more granular control over this by passing a ``mode``
                key to the ``compile_kwargs`` dictionary parameter, see pytensor's documentation for more
                details: https://pytensor.readthedocs.io/en/latest/tutorial/modes.html
            observations (list[reno.Observation]): Observed values (data/evidence) to use for computing
                posteriors, at least one should be specified if not exclusively running priors.
            smc (bool): Whether to use the sequential monte carlo sampler or not, the default is to
                do so - the regular samplers in PyMC tend not to do well if posterior distributions
                might have multiple peaks, see:
                https://www.pymc.io/projects/examples/en/latest/samplers/SMC2_gaussians.html
            trace_prior [az.InferenceData]: If priors for this model have already been run, pass in
                that arviz inference object here, and the prior xarray dataset will be used in the
                output arviz object from this pymc run.
            compute_prior_only (bool): If set to ``True``, don't run the Bayesian inference, only run
                the priors ("forward simulation mode").
            keep_config (bool): Whether to keep any changes made via free ref configurations passed
                in for subsequent simulations. The default is to not do this, keeping the original
                model unchanged.
            **free_refs: Definitions for equations or values for any variables or initial conditions
                in the system.
        """
        self._reset_type_and_shape_info()

        # TODO: observations, expect dict(ref, sigma, data)
        # store previous config vals
        previous = self.config()  # noqa: F841
        config = self.config(**free_refs)  # noqa: F841

        if n is None:
            n = self.n
        if steps is None:
            steps = self.steps

        if sampling_kwargs is None:
            sampling_kwargs = dict()

        if "cores" not in sampling_kwargs:
            sampling_kwargs["cores"] = 4

        if "draws" not in sampling_kwargs:
            sampling_kwargs["draws"] = math.ceil(n / sampling_kwargs["cores"])

        if compile_kwargs is None:
            compile_kwargs = dict()

        if compile_faster and "mode" not in compile_kwargs:
            compile_kwargs["mode"] = compile.mode.Mode(
                linker="cvm_nogc", optimizer="o3"
            )

        with self.pymc_model(steps=steps) as m:
            # add any observation likelihood variables
            if observations is not None:
                for obs in observations:
                    obs.add_tensors(m)

            # sample!
            sample_func = pm.sample_smc if smc else pm.sample
            if observations is None:
                sample_func = pm.sample_prior_predictive
                sampling_kwargs = dict(draws=n)
            if trace_prior is None:
                trace_prior = pm.sample_prior_predictive(
                    n, compile_kwargs=compile_kwargs
                )
            if not compute_prior_only:
                # sample_smc throws a fit with extremely few samples and the
                # exception it raises isn't obviously talking about this (a 0-d
                # iteration error), sometimes a recursion error, so raise our
                # own more informative error
                if observations is not None and n < sampling_kwargs["cores"] * 4:
                    raise Exception(
                        f"{n} is too few samples, run with a higher n parameter, e.g. ``.pymc(n=1000)``"
                    )

                # leaving explicit compile_kwargs out for now because in an
                # older pymc version it wasn't implemented for smc (specifically
                # 5.12.0?) An older version of pymc is sometimes necessary if
                # there's weird stalling issues:
                # https://discourse.pymc.io/t/sample-smc-stalls-at-final-stage/15055/20
                if len(compile_kwargs) > 0:
                    sampling_kwargs["compile_kwargs"] = compile_kwargs
                trace = sample_func(
                    **sampling_kwargs
                )  # , compile_kwargs=compile_kwargs)
                if observations is None:
                    trace.add_groups(posterior=trace.prior)

        if compute_prior_only:
            trace = trace_prior
        elif observations is not None:
            trace.extend(trace_prior)
        elif observations is None:
            del trace.prior
            trace.add_groups(prior=trace_prior.prior)
        self.trace = trace
        self.trace_RVs = [rv.name for rv in m.basic_RVs]

        if not keep_config:
            self.config(**previous)

        return trace

    def to_dict(self, root: bool = True) -> dict:
        """Convert the model into a JSON-serializable dictionary."""

        # TODO: not really any reason to separate _lists since the subsequent
        # dictionary's key lists should be equivalent? (except maybe metrics?)
        data = {
            "name": self.name,
            "doc": self.doc,
            "label": self.label,
            "stocks_list": [stock.name for stock in self.stocks],
            "flows_list": [flow.name for flow in self.flows],
            "vars_list": [var.name for var in self.vars],
            "metrics_list": [
                (metric.name, metric.__class__.__name__) for metric in self.metrics
            ],
            "models_list": [model.name for model in self.models],
            "models": {model.name: model.to_dict(False) for model in self.models},
            "stocks": {stock.name: stock.to_dict() for stock in self.stocks},
            "flows": {flow.name: flow.to_dict() for flow in self.flows},
            "vars": {var.name: var.to_dict() for var in self.vars},
            "metrics": {metric.name: metric.to_dict() for metric in self.metrics},
        }
        if root:
            data["timeref_name"] = self.find_timeref_name()
            data["n"] = self.n
            data["steps"] = self.steps
        return data

    def _add_skeleton(
        self, data: dict
    ) -> dict[str, "reno.components.TrackedReference"]:
        """An inner function for from_dict, it builds out everything structure-wise without
        populating equations and details (necessary because references between equations need
        to exist first.)"""
        building_refs = {}

        for stock_name in data["stocks_list"]:
            setattr(self, stock_name, reno.components.Stock())
            stock = getattr(self, stock_name)
            building_refs[stock.qual_name()] = stock
        for flow_name in data["flows_list"]:
            setattr(self, flow_name, reno.components.Flow())
            flow = getattr(self, flow_name)
            building_refs[flow.qual_name()] = flow
        for var_name in data["vars_list"]:
            setattr(self, var_name, reno.components.Variable())
            var = getattr(self, var_name)
            building_refs[var.qual_name()] = var
        for metric_name, metric_type in data["metrics_list"]:
            if metric_type == "Metric":
                setattr(self, metric_name, reno.components.Metric())
            elif metric_type == "Flag":
                setattr(self, metric_name, reno.components.Flag())
            else:
                raise ValueError(f"Metric type {metric_type} not found?!")
            metric = getattr(self, metric_name)
            building_refs[metric.qual_name()] = metric
        for model_name in data["models_list"]:
            sub_model = Model(
                name=model_name,
                label=data["models"][model_name]["label"],
                doc=data["models"][model_name]["doc"],
            )
            setattr(self, model_name, sub_model)
            sub_refs = sub_model._add_skeleton(data["models"][model_name])
            building_refs.update(sub_refs)

        return building_refs

    def _load_refs(
        self, data: dict, refs: dict[str, "reno.components.TrackedReference"]
    ):
        """Populate all equations and details/submodels etc. Assumes _add_skeleton
        has already been recursively run."""
        for stock_name in data["stocks"]:
            getattr(self, stock_name).from_dict(data["stocks"][stock_name], refs)
        for flow_name in data["flows"]:
            getattr(self, flow_name).from_dict(data["flows"][flow_name], refs)
        for var_name in data["vars"]:
            getattr(self, var_name).from_dict(data["vars"][var_name], refs)
        for metric_name in data["metrics"]:
            getattr(self, metric_name).from_dict(data["metrics"][metric_name], refs)
        for model_name in data["models"]:
            submodel = getattr(self, model_name)
            submodel._load_refs(data["models"][model_name], refs)

    @staticmethod
    def from_dict(data: dict) -> "Model":
        """Deserialize a previously saved model definition dictionary, returns
        new Model instance."""
        m = Model(
            name=data["name"],
            n=data["n"],
            steps=data["steps"],
            label=data["label"],
            doc=data["doc"],
        )
        refs = m._add_skeleton(data)
        if data["timeref_name"] is not None:
            refs[data["timeref_name"]] = reno.components.TimeRef()
        m._load_refs(data, refs)

        return m

    def save(self, path: str):
        """Save model definition at specified location. Stores as a JSON using the
        to_dict() method"""
        data = self.to_dict()
        with open(path, "w") as outfile:
            json.dump(data, outfile, indent=4)

    @staticmethod
    def load(path: str) -> "Model":
        """Initalize model from model definition file (JSON)."""
        with open(path) as infile:
            data = json.load(infile)

        return Model.from_dict(data)
