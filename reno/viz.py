"""Various visualization tools, classes, and functions.

A lot of this may eventually need to be broken out into separate
submodules for sanity."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import arviz as az
import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr
from IPython.display import clear_output, display
from ipywidgets import Layout
from matplotlib import ticker
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from scipy.stats import gaussian_kde

import reno
from reno.components import Flow, Scalar, Stock, Variable
from reno.interactive_latex import InteractiveLatex
from reno.parser import parse_class_or_scalar
from reno.utils import latex_eqline_wrap, latex_eqline_wrap_doc


def _create_seq_line_collection(seq_np_values: np.ndarray, **kwargs) -> LineCollection:
    """Running ax.plot for each individual line (of potentially thousands) is very slow,
    but you can create a LineCollection of all of the lines and render almost instantly,
    see https://matplotlib.org/stable/gallery/shapes_and_collections/line_collection.html

    Args:
        seq_np_values (np.ndarray): A matrix of y-values to plot (assumes that each row
            is a timeseries.)
        **kwargs: Any additional arguments passed to matplotlib.collections.LineCollection

    Returns:
        A LineCollection object that can be plotted on an axis with ``ax.add_collection()``
    """

    # the trick is that LineCollection needs to be a list of lines where each
    # line is a list of (x0, y0), ... coords, e.g.
    # [[[x0, y0], [x1, y1]...], ...,]

    # so, we do some numpy magic to create the x-domain for these sequences and
    # shape the matrix of y data appropriately

    domain_size = seq_np_values.shape[-1]
    # make sure to flatten if it wasn't passed flattened already
    seqs = seq_np_values.reshape([-1, domain_size])
    # create a corresponding domain (0, 1, 2, ...) for each sequence
    domains = np.tile(np.arange(0, domain_size, 1), (seqs.shape[0], 1))
    # combine!
    line_data = np.stack((domains, seqs), axis=2)
    # at this point we should have (num_lines, timesteps, 2) where first of 2 is domain)

    return LineCollection(line_data, **kwargs)


def _get_label_and_dataset(
    trace_collection: (
        list[az.InferenceData | xr.Dataset] | dict[str, az.InferenceData | xr.Dataset]
    ),
    key_or_trace: str | az.InferenceData | xr.Dataset,
) -> tuple[str, xr.Dataset]:
    """Traces can be passed to the comparison functions multiple ways, so this is
    a method to consistently get a label for a trace and the desired dataset."""
    # determine if label explicitly provided via dictionary or if we need
    # to auto-assign
    if isinstance(trace_collection, dict):
        label = key_or_trace
        trace = trace_collection[key_or_trace]
    else:
        label = f"trace {trace_collection.index(key_or_trace)}"
        trace = key_or_trace

    # automatically grab the posterior if not explicitly requested
    if isinstance(trace, az.InferenceData):
        ds = trace.posterior
    else:
        ds = trace

    return label, ds


def _get_sample_count(array: xr.DataArray) -> int:
    """Dimensions for base reno and pymc are different since pymc has chains, so this
    throws off alpha calculations, use this to get the number of things that will be drawn.
    """
    if "chain" in array.coords:
        array = array.stack(sample=["chain", "draw"])
    return len(array.coords["sample"])


def compare_seq(
    varname: str,
    traces: (
        list[az.InferenceData | xr.Dataset] | dict[str, az.InferenceData | xr.Dataset]
    ),
    prior_trace=None,
    ax=None,
    legend: bool = True,
    title: str = None,
    **figargs,
):
    """Plot the timeseries data for the specified variable from each of the passed traces + prior trace.

    Args:
        varname (str): The name of the variable in the xr.Datasets to plot
        traces (list[az.InferenceData | xr.Dataset] | dict[str, az.InferenceData | xr.Dataset): A list
            or dictionary of traces or dataset to plot the variable from. Passed traces will plot from
            the posterior, pass the specific dataset if you need the priors (``pymc_trace.prior``). If
            a dictionary is used, the legend will use the specified keys.
        prior_trace (az.InferenceData | xr.Dataset): A trace or dataset to plot with the 'prior' key.
        ax: Optionally pass an axis if one already exists, otherwise this function will create a new one,
            using any additional figargs passed.
        legend (bool): Whether to render a legend or not. If ``True`` and traces is a dictionary, the
            keys will be used as the legend labels.
        title (str): Optional title to set on the axis.
        **figargs: Parameters to pass to ``plt.subplots(**figargs)`` if no axis passed in.
    """
    if ax is None:
        with plt.ioff():
            fig, ax = plt.subplots(**figargs)

    cat_col = 0

    legend_handles = []

    if prior_trace is not None:
        alpha = 0.01 if _get_sample_count(prior_trace.prior[varname]) > 10 else 0.75
        ax.add_collection(
            _create_seq_line_collection(
                prior_trace.prior[varname].values, color=f"C{cat_col}", alpha=alpha
            )
        )
        legend_handles.append(Line2D([0], [0], label="prior", color=f"C{cat_col}"))
        cat_col += 1

    for i, trace in enumerate(traces):
        label, ds = _get_label_and_dataset(traces, trace)

        if varname not in ds:
            continue
        alpha = 0.01 if _get_sample_count(ds[varname]) > 10 else 0.75
        ax.add_collection(
            _create_seq_line_collection(
                ds[varname].values, color=f"C{cat_col}", alpha=alpha
            )
        )
        legend_handles.append(Line2D([0], [0], label=label, color=f"C{cat_col}"))
        cat_col += 1

    if legend:
        ax.legend(handles=legend_handles)

    if title is None:
        ax.set_title(varname)
    else:
        ax.set_title(title)

    ax.autoscale()
    return ax.get_figure()


def _plot_posterior_values(
    ax,
    values: np.ndarray,
    label: str,
    smoothing: float = 0.1,
    num_traces_to_plot: int = 1,
    trace_index: int = 0,
):
    """num_traces_to_plot only matters for the bar chart, for handling offset."""
    # plot bars for categorical data
    if values.dtype in ("int64", "int32", "int8"):
        width = 0.75 / num_traces_to_plot

        counts = pd.Series(values.flatten()).value_counts()
        # normalize
        counts = counts / counts.sum()
        x = counts.index.values + width * trace_index
        ax.bar(x, counts.values, width=width - 0.02, label=label, alpha=0.75)
        # ax.bar_label(rects)
        ax.set_xticks(counts.index.values + width * (num_traces_to_plot - 1) / 2)
        ax.set_xticklabels(counts.index.values)
    elif len(values.flatten()) == 1:
        ax.axvline(x=values.flatten()[0], label=label, color=f"C{trace_index}")
    else:
        domain, densities = density(values.flatten(), smoothing)
        ax.plot(domain, densities, label=label)
        ax.fill_between(domain, densities, alpha=0.1)


def compare_posterior(
    varname: str,
    traces: (
        list[az.InferenceData | xr.Dataset] | dict[str, az.InferenceData | xr.Dataset]
    ),
    prior_trace=None,
    smoothing: float = 0.1,
    ax=None,
    legend: bool = True,
    title=None,
    per_dim=None,
    **figargs,
):
    """Plot the sampled distribution densities for the specified variable from each of the passed traces + prior trace.

    Args:
        varname (str): The name of the variable in the xr.Datasets to plot
        traces (list[az.InferenceData | xr.Dataset] | dict[str, az.InferenceData | xr.Dataset): A list
            or dictionary of traces or dataset to plot the variable from. Passed traces will plot from
            the posterior, pass the specific dataset if you need the priors (``pymc_trace.prior``). If
            a dictionary is used, the legend will use the specified keys.
        prior_trace (az.InferenceData | xr.Dataset): A trace or dataset to plot with the 'prior' key.
        smoothing (float): What degree of smoothing to apply to the density plot. Lower = more bumpy.
        ax: Optionally pass an axis if one already exists, otherwise this function will create a new one,
            using any additional figargs passed.
        legend (bool): Whether to render a legend or not. If ``True`` and traces is a dictionary, the
            keys will be used as the legend labels.
        title (str): Optional title to set on the axis.
        per_dim (str): TODO
        **figargs: Parameters to pass to ``plt.subplots(**figargs)`` if no axis passed in.
    """
    # NOTE: use per_dim to either individually plot chains, or individually plot
    # dirichlet components, None to entirely flatten (the default)
    # TODO: (not implemented yet)

    if ax is None:
        with plt.ioff():
            fig, ax = plt.subplots(**figargs)

    num_traces = len(traces)
    trace_index = 0
    if prior_trace is not None:
        num_traces += 1

    if prior_trace is not None:
        _plot_posterior_values(
            ax,
            prior_trace.prior[varname].values,
            "prior",
            smoothing,
            num_traces,
            trace_index,
        )
        trace_index += 1

    for i, trace in enumerate(traces):
        label, ds = _get_label_and_dataset(traces, trace)

        if varname not in ds:
            continue

        _plot_posterior_values(
            ax, ds[varname].values, label, smoothing, num_traces, trace_index
        )
        trace_index += 1

    if legend:
        ax.legend()

    if title is None:
        ax.set_title(varname)
    else:
        ax.set_title(title)

    return ax.get_figure()


def plot_refs_single_axis(
    trace: az.InferenceData | xr.Dataset,
    ref_list: list[str | reno.components.Reference],
    num_ticks: int = 3,
    # ax = None,
    **figargs,
):
    """I've seen this type of plot in a few SDM textbooks and references at this point,
    it plots all the specified references on the same plot and scales each one individually.

    This means there's a separate y-axis per reference, and getting the tick labels to not
    conflict with eachother and be offset was a bit of a trick, so this simplifies that logic.
    """
    plt.ioff()
    fig, ax1 = plt.subplots(**figargs)
    plt.ion()

    refs = [
        ref.qual_name() if isinstance(ref, reno.components.Reference) else ref
        for ref in ref_list
    ]

    if isinstance(trace, az.InferenceData):
        ds = trace.posterior
    else:
        ds = trace

    def label_offset(n, n_max):
        """Algorithm to vertically offset each label for a single tick so they
        don't overlap. n is the index of the current label, n_max is the total
        number of labels that have to fit."""
        spacing = 5
        total = (n_max - 1) * spacing * 2
        label_n = ((n_max - n) * spacing * 2) - (total / 2)
        return label_n

    legend_handles = []
    for i, ref in enumerate(refs):
        alpha = 0.01 if _get_sample_count(ds[ref]) > 10 else 0.75

        if i == 0:
            ax = ax1
        else:
            ax = ax1.twinx()

        ax.add_collection(
            _create_seq_line_collection(ds[ref].values, color=f"C{i}", alpha=alpha)
        )
        ax.autoscale()
        ax.yaxis.set_major_locator(ticker.LinearLocator(num_ticks))
        ax.spines["left"].set_position(("axes", 0.0))
        ax.spines["left"].set_visible(True)
        ax.yaxis.set_label_position("left")
        ax.yaxis.set_ticks_position("left")
        ax.tick_params(axis="y", colors=f"C{i}")

        for tick in ax.yaxis.get_majorticklabels():
            tick.set_transform(
                tick.get_transform()
                + ScaledTranslation(
                    0 / 72, label_offset(i + 1, len(refs)) / 72, fig.dpi_scale_trans
                )
            )

        legend_handles.append(Line2D([0], [0], label=ref, color=f"C{i}"))

    ax1.legend(handles=legend_handles)
    return fig


def plot_trace_refs(
    reference_model,
    traces: (
        list[az.InferenceData | xr.Dataset] | dict[str, az.InferenceData | xr.Dataset]
    ),
    ref_list: list[str | reno.components.Reference],
    cols: int = None,
    rows: int = None,
    smoothing: float = 0.1,
    **figargs,
):
    """Create a set of plots for each of the specified references, automatically plotting timeseries
    sequences or distribution densities where relevant.

    Args:
        reference_model: The model from which the datasets are being plotted, this is used to help
            axes titles and determine which references are random variables etc.
        traces (list[az.InferenceData | xr.Dataset] | dict[str, az.InferenceData | xr.Dataset): A list
            or dictionary of traces or dataset to plot the references from. Passed traces will plot from
            the posterior, pass the specific dataset if you need the priors (``pymc_trace.prior``). If
            a dictionary is used, the legends will use the specified keys.
        ref_list (list[str | reno.components.Reference]): Either the string names or actual reference
            objects to plot the data of. (A separate plot will be created for each reference.)
        cols (int): The number of columns to split the plots into.
        rows (int): The number of rows to split the plots into.
        smoothing (float): Degree of smoothing to apply to density plots, lower = more bumpy.
        **figargs: Arguments to pass to the figure creation ``plt.subplots(..., **figargs)`` call.
    """
    # remaining_refs = [*ref_list]
    remaining_refs = [
        ref.qual_name() if isinstance(ref, reno.components.Reference) else ref
        for ref in ref_list
    ]
    num_things = len(ref_list)

    if cols is None and num_things > 1:
        cols = 2
    elif cols is None:
        cols = 1
    if rows is None:
        rows = math.ceil(num_things / cols)
    if rows == 0:
        rows = 1

    with plt.ioff():
        fig, axs = plt.subplots(rows, cols, **figargs)

    if rows == 1 and cols == 1:
        axs = [[axs]]
    elif rows == 1:
        axs = [axs]
    elif cols == 1:
        new_axs = []
        for ax in axs:
            new_axs.append([ax])
        axs = new_axs

    row_i = 0
    col_i = 0

    def advance_indices():
        nonlocal row_i, col_i
        col_i += 1
        if col_i >= cols:
            col_i = 0
            row_i += 1

    # RVs
    rv_names = [
        name
        for name in reference_model.trace_RVs
        if not name.endswith("_likelihood") and name in remaining_refs
    ]
    rvs = [ref for ref in reference_model.all_refs() if ref.qual_name() in rv_names]
    for rv in rvs:
        compare_posterior(
            rv.qual_name(),
            traces,
            title=f"RV: {rv.qual_name()}",
            smoothing=smoothing,
            ax=axs[row_i][col_i],
            **figargs,
        )
        advance_indices()
        remaining_refs.remove(rv.qual_name())

    # other variables
    static_vars = []
    nonstatic_vars = []
    for var in reference_model.all_vars():
        if var.qual_name() in remaining_refs:
            if var.is_static():
                static_vars.append(var)
            else:
                nonstatic_vars.append(var)

    for var in static_vars:
        compare_posterior(
            var.qual_name(),
            traces,
            smoothing=smoothing,
            title=f"Var: {var.qual_name()}",
            ax=axs[row_i][col_i],
            **figargs,
        )
        advance_indices()
        remaining_refs.remove(var.qual_name())

    # stocks/flows
    seqs = [
        ref for ref in reference_model.all_refs() if ref.qual_name() in remaining_refs
    ]
    # seq_names.extend(nonstatic_var_names)  # already handled by what wouldn't
    # be removed from remaining_refs and all_refs
    title_prefix = {Stock: "Stock", Flow: "Flow", Variable: "Var"}
    for ref in seqs:
        title = f"{title_prefix[type(ref)]}: {ref.qual_name()}"
        compare_seq(
            ref.qual_name(), traces, ax=axs[row_i][col_i], title=title, **figargs
        )
        advance_indices()
        remaining_refs.remove(ref.qual_name())

    # metrics
    metrics = [
        metric
        for metric in reference_model.all_metrics()
        if metric.name in remaining_refs
    ]
    for metric in metrics:
        name = metric.qual_name()
        compare_posterior(
            name,
            traces,
            smoothing=smoothing,
            ax=axs[row_i][col_i],
            title=f"Metric: {name}",
        )
        advance_indices()
        remaining_refs.remove(name)
    fig.tight_layout()
    return fig


def density(data: np.ndarray, smoothing: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Get smoothed histogram (density plot) data from an array.

    Returns a tuple of the domain (x) and the smoothed values (y).
    """
    # convert data to floats and add jitter if all the same?
    # if data.var() == 0.0:
    # scale = (data[0] + 1) / 500
    scale = 0.0001 if data[0] < 1.0 else data[0] / 500
    data = data + np.random.normal(scale=scale, size=data.shape)

    dens = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 400)
    dens.covariance_factor = lambda: smoothing
    try:
        dens._compute_covariance()
    except:  # noqa: E722
        # There probably wasn't enough data/enough diversity in the data to
        # accurately do this
        pass
    return xs, dens(xs)


def plot_refs(
    refs: list[
        reno.components.Reference
        | list[reno.components.Reference]
        | tuple[reno.components.Reference]
    ],
    cols: int = None,
    rows: int = None,
    legends: bool = True,
    **figargs,
) -> plt.Figure:
    """Render a bunch of subplots containing the line plots for the values of the passed
    references.

    `refs` can contain both references directly as well as a nested list/tuple of
    references - anything in a nested list is plotted within the same subplot.

    Note that this is distinct from plot_trace_refs and plots only data from the
    last simulation run for the model

    Example:
        >>> plot_refs([m.stock_1, m.stock_2, m.stock_3])

        >>> plot_refs([m.stock_1, (m.stock_2, m.flow_1, m.flow_2), m.stock_3])

    """
    # TODO: would probably be worth making this also take a traces arg (and
    # simply default to last-sim data in model if None)
    multiple_models = False

    names = []
    for ref in refs:
        if isinstance(ref, (tuple, list)):
            names.extend([sub_ref.model.name for sub_ref in ref])
        else:
            names.append(ref.model.name)
    if len(list(set(names))) > 1:
        multiple_models = True

    # if len(list(set([ref.model.name for ref in refs]))) > 1:

    # establish correct grid size/necessary num of cols/rows if not provided
    if len(refs) > 1 and cols is None:
        cols = 2
    elif cols is None:
        cols = 1
    if rows is None:
        rows = math.ceil(len(refs) / cols)

    plt.ioff()
    fig, axs = plt.subplots(rows, cols, **figargs)
    plt.ion()

    if rows == 1 and cols == 1:
        axs = [[axs]]

    row_i = 0
    col_i = 0

    for ref in refs:
        if not isinstance(ref, (tuple, list)):
            for row in range(ref.value.shape[0]):
                axs[row_i][col_i].plot(ref.value[row])
            if multiple_models:
                axs[row_i][col_i].set_title(f"{ref.model.name}: {ref.name}")
            else:
                axs[row_i][col_i].set_title(ref.name)
        else:
            # allow passing in tuples of refs to plot on same axis
            for sub_ref in ref:
                for row in range(sub_ref.value.shape[0]):
                    if multiple_models:
                        axs[row_i][col_i].plot(
                            sub_ref.value[row],
                            label=f"{sub_ref.model.name}: {sub_ref.name}",
                        )
                    else:
                        axs[row_i][col_i].plot(sub_ref.value[row], label=sub_ref.name)
            if legends:
                axs[row_i][col_i].legend()
            if multiple_models:
                axs[row_i][col_i].set_title(
                    " vs ".join(
                        [f"{sub_ref.model.name}: {sub_ref.name}" for sub_ref in ref]
                    )
                )
            else:
                axs[row_i][col_i].set_title(
                    " vs ".join([sub_ref.name for sub_ref in ref])
                )

        col_i += 1
        if col_i >= cols:
            col_i = 0
            row_i += 1

    fig.tight_layout()
    return fig


class ModelLatex:
    """An interactive widget that displays all of the latex equations of a model.

    Use with `debug=True` to display all of the values in the equation at a particular
    timestep and sample.

    NOTE: start_name is exclusive, stop_name is inclusive.

    Args:
        model (Model): The system dynamics model to get the equations for.
        show_docs (bool): Whether to display docstrings below each relevant equation.
        start_name (str): Don't display any equations before the one specified (in order
            of their definition.) This is mostly unused for now, intended to eventually
            allow displaying sets of graphs "between" equations.
        stop_name (str): Don't display any equations after the one specified (in order
            of their definition.) This is mostly unused for now, intended to eventually
            allow displaying sets of graphs "between" equations.
        t (int): Only used when `debug` is `True`, refers to the timestep to use for
            displaying the current values within each equation.
        sample (int): Only used when `debug` is `True`, refers to which sample to use
            for displaying the current values within each equation.
        debug (bool): If true, display the value of every reference at the specified `t`
            and `sample`.

    Example:
        >>> ModelLatex(my_model).widget

        >>> ModelLatex(my_model, t=3, sample=0, debug=True).widget
    """

    def __init__(
        self,
        model,
        show_docs: bool = False,
        start_name: str = None,
        stop_name: str = None,
        t: int = None,
        sample: int = 0,
        debug: bool = False,
    ):
        self.model = model
        self.widget = InteractiveLatex()
        """This is the actual displayable object, use this to "view" the equations."""

        self.show_docs = show_docs
        self.debug = debug
        self.t = t
        self.sample = sample

        self.start_name = start_name
        self.stop_name = stop_name

        # NOTE: exclusive start, inclusive stop

        self.widget.latex.data = self.latex_data_with_highlight()
        self.widget.refresh_display()
        self.widget.on_row_clicked(self._handle_ilatex_click)

        self._name_clicked_callbacks: list[Callable[[str], None]] = []

        self.handle_refresh = True

    def on_name_clicked(self, callback: Callable[[str], None]):
        """Register an event handler for when an equation is clicked.

        Callbacks should take a single parameter which is the string name of
        the equation/ref that was clicked.
        """
        self._name_clicked_callbacks.append(callback)

    def fire_on_name_clicked(self, name: str):
        """Trigger the event to notify that a reference/eq was clicked on."""
        for callback in self._name_clicked_callbacks:
            callback(name)

    def _handle_ilatex_click(self, i: int):
        hl_name = self.find_equation_name_from_index(i)
        self.fire_on_name_clicked(hl_name)
        if self.handle_refresh:
            new_latex_data = self.latex_data_with_highlight(hl_name)
            self.widget.latex.data = new_latex_data
            self.widget.refresh_display()

    def find_equation_name_from_index(self, i: int) -> str:
        """Get the name of the reference/equation for the clicked index.

        Returns ``None`` if not found or outside the start/stop bounds."""
        # NOTE: exclusive start, inclusive stop

        names = self._equation_lines_refname_reference()
        if i >= len(names):
            return None
        return names[i]

    def _equation_lines_refname_reference(self) -> list[str]:
        """Create a list of reference names that correspond to the equation lines
        in the latex. This is influenced based on the presence of docstrings (and
        whether show_docs is enabled) etc.

        The number of lines that are related to a particular reference should align
        with and equal the number of times that that reference name appears in the
        returned list.

        For example, a flow with a docstring and show_docs=True will add that flow's
        name twice.

        NOTE: start_name is exclusive, stop_name is inclusive.
        """
        names = []
        started = True if self.start_name is None else False
        stopped = False

        for ref in (
            self.model.all_vars()
            + self.model.all_flows()
            + self.model.all_stocks()
            + self.model.metrics
        ):
            if hasattr(ref, "implicit") and ref.implicit:
                continue

            # logic to only process within specified "name range"
            if not started:
                if ref.qual_name() == self.start_name:
                    started = True
                continue  # exclusive start (don't begin until next)
            if stopped:
                continue
            else:
                if ref.qual_name() == self.stop_name:
                    stopped = True  # inclusive stop (won't stop until next)

            if isinstance(ref, reno.components.Stock):
                # add a line for each equation in the stock
                if not self.debug:
                    for equation in ref.equations():
                        names.append(ref.qual_name())
                else:
                    names.append(ref.qual_name())
            else:
                names.append(ref.qual_name())
            if ref.doc is not None and self.show_docs:
                names.append(ref.qual_name())
        return names

    def latex_data_with_highlight(self, hl_name: str = None) -> str:
        """Construct the latex string with highlighting for the specified name."""
        kwargs = {}
        if hl_name is not None:
            kwargs["hl"] = hl_name
        if self.debug:
            kwargs["t"] = self.t
            kwargs["sample"] = self.sample

        string = "$\n\\begin{align*}\n"

        for ref in (
            self.model.all_vars()
            + self.model.all_flows()
            + self.model.all_stocks()
            + self.model.metrics
        ):
            if hasattr(ref, "implicit") and ref.implicit:
                continue

            # quick cheat to take start/stop names into account by ensuring
            # they're in the aligned list of names we're taking care of
            if ref.qual_name() not in self._equation_lines_refname_reference():
                continue

            highlight = ref.qual_name() == hl_name
            if isinstance(ref, reno.components.Stock):
                # stocks have to be handled differently because they have
                # multiple equations
                if self.debug:
                    equation_str = ref.debug_equation(**kwargs)
                    string += latex_eqline_wrap(equation_str, highlight)
                else:
                    for equation in ref.equations(**kwargs):
                        string += latex_eqline_wrap(equation, highlight)

                if ref.doc is not None and self.show_docs:
                    string += latex_eqline_wrap_doc(
                        ref.name.replace("_", "\\_") + ": " + ref.doc, highlight
                    )
            else:
                if self.debug:
                    equation_str = ref.debug_equation(**kwargs)
                else:
                    equation_str = ref.equation(**kwargs)

                string += latex_eqline_wrap(equation_str, highlight)
                if ref.doc is not None and self.show_docs:
                    string += latex_eqline_wrap_doc(ref.doc, highlight)

        string += "\\end{align*}$"
        return string


# class DebugModelLatex:
#     # NOTE: not really ready yet, intent is to make it so when you click on an
#     # equation, it displays all relevant reference graphs "below" that
#     # particular equation. This involves displaying separate model latex objects
#     # above and below with appropriately set "start" and "stop" names.
#     def __init__(self, model):
#         self.model = model
#         self.set1 = ModelLatex(model)
#         self.set2 = ModelLatex(
#             model, start_name=model.stocks[-1], stop_name=model.stocks[-1]
#         )
#
#         self.set1.handle_refresh = False
#         self.set2.handle_refresh = False
#         self.set1.on_name_clicked(self._on_name_clicked)
#         self.set2.on_name_clicked(self._on_name_clicked)
#
#         self.selected_name = None
#
#         self.inner = ipw.Output()
#         self.widget = ipw.Output()
#         # self.widget = ipw.VBox([self.set1.widget, self.inner])#, self.set2.widget])
#
#     def refresh(self):
#         with self.widget:
#             clear_output(True)
#             display(self.set1.widget)
#             # display(self.set1.widget)
#
#     def _on_name_clicked(self, name: str):
#         print("Yo!", name)
#         self.selected_name = name
#         self.set1.stop_name = name
#         self.set2.start_name = name
#
#         set1_data = self.set1.latex_data_with_highlight(name)
#         set2_data = self.set2.latex_data_with_highlight(name)
#
#         self.set1.widget.latex_data = set1_data
#         self.set2.widget.latex_data = set2_data
#         self.set1.widget.refresh_display()
#         self.set2.widget.refresh_display()
#
#         self.refresh()
#
#     def local_plots_for_ref(self, ref, size: int = 2):
#         # size is distance in each direction that we collect for
#
#         # get all relevant refs
#         refs = [ref]
#         refs.extend(ref.seek_refs())
#
#         fig, axs = plt.subplots(1, len(refs))
#
#         domain = [
#             max(self.t - size, 0),
#             min(self.t + size, len(refs[0].value.shape[1])),
#         ]
#
#         for i, l_ref in enumerate(refs):
#             axs[0][i].plot(l_ref.value[self.sample][domain])
#             axs[0][i].set_title(l_ref.name)


@dataclass
class ReferenceEditor:
    """A textbox associated with a reference, used for modifying an equation
    in a visual interface.

    Primarily only used in explorer now.
    """

    model: "reno.model.Model"
    ref_name: str
    is_init: bool
    control: ipw.Text | pn.widgets.TextInput = None

    def get_ref(self) -> reno.components.TrackedReference:
        """Get the underling TrackedReference object associated with this editor."""
        if not self.is_init:
            return getattr(self.model, self.ref_name)
        else:
            return getattr(self.model, self.ref_name.removesuffix("_0"))

    def get_eq(self) -> reno.components.EquationPart:
        """Get the equation from the TrackedReference, (takes into account if
        this is describing the init or the eq itself)."""
        ref = self.get_ref()
        if not self.is_init:
            return ref.eq
        else:
            return ref.init

    def get_eq_str(self) -> str:
        """Get the string version of the current equation."""
        eq = self.get_eq()
        if eq is None:
            return ""
        if isinstance(eq, Scalar):
            return str(eq.value)
        return str(eq)

    def parse_str_to_eq(self) -> reno.components.EquationPart:
        """Get the equivalent EquationPart equation for the string in the control."""
        result = parse_class_or_scalar(self.control.value)

        if isinstance(result, (float, int)):
            result = Scalar(result)
        return result

    def assign_value_from_control(self):
        """Set the underlying reference's equation based on the current string
        in the control."""
        result = self.parse_str_to_eq()

        if not self.is_init:
            self.get_ref().eq = result
        else:
            self.get_ref().init = result


class ModelViewer:
    """An interface for viewing a model, modifying any free variables, and simulating live.

    Leaving for now, recommend using Explorer/explorer widgets instead
    """

    def __init__(
        self,
        model,
        exclude_vars: list[str] = None,
        plot_refs: list[
            reno.components.Reference
            | list[reno.components.Reference]
            | tuple[reno.components.Reference]
        ] = None,
    ):
        if exclude_vars is None:
            exclude_vars = []
        self.model = model
        self.exclude_vars = exclude_vars

        self.graphviz_out = ipw.Output()
        self.latex_out = ModelLatex(model)

        self.plots_out = ipw.Output()
        self.metrics_out = ipw.Output()

        self.reference_editors = []
        self.vars_controls = ipw.VBox()
        self.create_variable_controls()

        self.simulate_button = ipw.Button(description="Simulate")
        self.n = ipw.IntText(value=model.n, layout=Layout(width="50px"))
        self.steps = ipw.IntText(value=model.steps, layout=Layout(width="70px"))

        self.show_vars = ipw.Checkbox(description="Show variables", value=False)
        self.show_vars.observe(lambda x: self.rerender_graph(), names=["value"])

        self.plot_refs = plot_refs

        self.widget = ipw.VBox(
            [
                self.show_vars,
                ipw.HBox([self.graphviz_out, self.latex_out.widget]),
                ipw.HBox(
                    [
                        ipw.VBox(
                            [
                                self.vars_controls,
                                ipw.HBox([self.simulate_button, self.n, self.steps]),
                            ]
                        ),
                        ipw.HBox(
                            [
                                self.plots_out,
                                self.metrics_out,
                            ]
                        ),
                    ]
                ),
            ]
        )
        self.simulate_button.on_click(self.simulate_button_press)

        self.rerender_graph()
        self.rerender_equations()

    def simulate_button_press(self, widget):
        """Event handler for when the button labeled "simulate" is pressed, not a
        function simulating a button being pressed..."""
        self.assign_controls()
        self.rerun(self.n.value, self.steps.value)

    def rerender_graph(self):
        with self.graphviz_out:
            clear_output(True)
            display(
                self.model.graph(self.show_vars.value, exclude_vars=self.exclude_vars)
            )

    def rerender_equations(self):
        self.latex_out.latex_data_with_highlight()
        # with self.latex_out:
        #     clear_output(True)
        #     display(self.model)

    def rerun(self, n, steps):
        plt.close("all")
        self.rerender_graph()
        self.rerender_equations()

        self.model.simulate(n, steps, quiet=True)

        with self.plots_out:
            clear_output(True)
            if self.plot_refs is None:
                display(self.model.plot_stocks())
            else:
                display(plot_refs(self.plot_refs))

        with self.metrics_out:
            clear_output(True)
            # TODO: better options than print_metrics
            # self.model.print_metrics()

    def assign_controls(self):
        for ref_editor in self.reference_editors:
            ref_editor.assign_value_from_control()

    def create_variable_controls(self):
        """Set up a bunch of ReferenceEditors for the free variables."""
        style = {"description_width": "initial"}
        controls = []
        for ref_name in self.model.free_refs():
            editor = ReferenceEditor(
                self.model, ref_name, self.model._is_init_ref(ref_name)
            )
            # ref = editor.get_ref()
            control = ipw.Text(
                description=ref_name, value=str(editor.get_eq()), style=style
            )
            editor.control = control

            self.reference_editors.append(editor)
            controls.append(control)
        self.vars_controls.children = controls
