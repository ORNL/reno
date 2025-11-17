"""Functions for generating stock & flow diagrams.

Diagrams are graphviz dot diagrams, and notably don't follow the
exact format typically used with SFDs, namely sources and sinks
aren't explicitly represented.
"""

import os

import matplotlib.pyplot as plt
import xarray as xr
from graphviz import Digraph

import reno
from reno.components import Flow, Stock, TimeRef, TrackedReference, Variable

# Theming constants, set for lightmode by default.
DARK_MODE = False
"""When ``True``, set graph node_attr to use colors more suited for dark backgrounds."""
SUBGRAPH_COLORS = ["#BBDDFF", "#DDBBFF"]
"""Container background colors for nested models/submodels, each subsquent
level will go one color farther into this array.

A decent option for dark mode: ``["#334455", "#443355"]``
"""
NODE_COLOR = "lightgreen"
"""Background color to use for variable nodes.

A decent option for dark mode: ``"darkgreen"``
"""


def set_dark_mode(dark: bool = False):
    """Set themeing constants for the diagrams appropriately for
    dark/light theme.

    Note that this will also change matplotlib style so that spark plots
    match. Set the ``DARK_MODE``, ``SUBGRAPH_COLORS``, and ``NODE_COLOR``
    constants manually if this is undesired.

    Args:
        dark (bool): Pass ``False`` for light theme, ``True`` for dark theme.
    """

    global DARK_MODE, SUBGRAPH_COLORS, NODE_COLOR
    DARK_MODE = dark
    if not dark:
        plt.style.use("default")
        SUBGRAPH_COLORS = ["#BBDDFF", "#DDBBFF"]
        NODE_COLOR = "lightgreen"
    else:
        plt.style.use("dark_background")
        SUBGRAPH_COLORS = ["#334455", "#443355"]
        NODE_COLOR = "darkgreen"


def stock_flow_diagram(
    model,
    show_vars: bool = True,
    exclude_var_names: list[str] = None,
    sparklines: bool = False,
    sparkdensities: bool = False,
    sparkall: bool = False,
    g: Digraph = None,
    traces: list[xr.Dataset] = None,
    universe: list[TrackedReference] = None,
    lr: bool = False,
    _level: int = 0,
) -> tuple[Digraph, list[tuple[str, str]]]:
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
        The populated Digraph instance (Jupyter can natively render this in a cell output.) and
        a list of tuples of reference connections that weren't added (usually because one of the
        references was "out of scope" part of a parent model not rendered in this one. This is
        primarily for internal use for correctly recursively plotting submodels in subgraphs.
    """
    if g is None:
        rankdir = "TB" if not lr else "LR"
        if DARK_MODE:
            g = Digraph(
                name=model.name,
                graph_attr=dict(rankdir=rankdir, bgcolor="#181818"),
                node_attr=dict(
                    fontcolor="#e6e6e6",
                    style="filled",
                    color="#e6e6e6",
                    fillcolor="#333333",
                ),
                edge_attr=dict(color="#e6e6e6", fontcolor="#e6e6e6"),
            )
        else:
            g = Digraph(name=model.name, graph_attr=dict(rankdir=rankdir))
    if exclude_var_names is None:
        exclude_var_names = []

    g, out_of_scope_stock_connections = add_stocks(
        g,
        [stock for stock in model.stocks if not stock.implicit],
        sparklines=sparklines,
        traces=traces,
        universe=universe,
    )
    out_of_scope_var_connections = []
    if show_vars:
        variables = filter_variables(
            [var for var in model.vars if not var.implicit], exclude_var_names
        )
        g, out_of_scope_var_connections = add_vars(
            g,
            variables,
            exclude_var_names,
            sparkdensities,
            sparkall,
            traces,
            universe=universe,
        )
    else:
        variables = []
    g, out_of_scope_flow_connections = add_flows(
        g,
        [flow for flow in model.flows if not flow.implicit],
        variables,
        sparkall,
        traces,
        universe=universe,
    )

    missing_connections = (
        out_of_scope_var_connections
        + out_of_scope_flow_connections
        + out_of_scope_stock_connections
    )

    # recursively render any submodels.
    for submodel in model.models:
        with g.subgraph(name=submodel.name) as g_s:
            g_s.attr(style="filled")
            g_s.attr(color=SUBGRAPH_COLORS[_level])
            g_s.attr(cluster="true")
            g_s.attr(label=submodel.label)
            g_s.attr(fontcolor="#888888")
            m = getattr(model, submodel.name)
            unused_graph, remaining_connections = stock_flow_diagram(
                m,
                show_vars=show_vars,
                exclude_var_names=exclude_var_names,
                sparklines=sparklines,
                sparkdensities=sparkdensities,
                sparkall=sparkall,
                g=g_s,
                traces=traces,
                universe=universe,
                lr=lr,
                _level=_level + 1,
            )
            for src, dst in remaining_connections:
                # check if we are at the correct level model to draw this,
                # or do we pass it on up?
                if src.model == model or dst.model == model:
                    draw_appropriate_edge(g, src, dst)
                else:
                    missing_connections.append((src, dst))

    return g, missing_connections


def add_stock_io_edge(g: Digraph, stock: Stock, flow: Flow, dir: str = None):
    """Edge with attributes for heavily highlighting in flows and
    outflows from stocks."""

    if flow.implicit and flow in stock.in_flows:
        for ref in flow.seek_refs():
            if isinstance(ref, Flow):
                add_stock_io_edge(g, stock, ref, "in")
        return
    elif flow.implicit and flow in stock.out_flows:
        for ref in flow.seek_refs():
            if isinstance(ref, Flow):
                add_stock_io_edge(g, stock, ref, "out")
        return

    if flow in stock.in_flows or dir == "in":
        name1 = flow.qual_name()
        name2 = stock.qual_name()
    elif flow in stock.out_flows or dir == "out":
        name1 = stock.qual_name()
        name2 = flow.qual_name()
    else:
        raise Exception(
            f"Could not draw edge for flow {flow.qual_name()} and stock {stock.qual_name()}, stock does not use flow."
        )
    g.edge(name1, name2, style="bold", weight="50")


def add_stock_limit_edge(g: Digraph, ref, stock: Stock):
    """De-emphasize stock variable/flow connections if they're only
    used in min/max constraints."""
    g.edge(
        ref.qual_name(),
        stock.qual_name(),
        style="dotted",
        arrowsize=".5",
    )


def add_to_flow_edge(g: Digraph, ref, flow: Flow):
    """Non-stock-inflow/outflow-related edges that point to flows
    are slightly de-emphasized."""
    style = "dotted" if isinstance(ref, Variable) else "dashed"
    constraint = "false" if isinstance(ref, Stock) else "true"
    # TODO: constraint should maybe take same model scope into account?
    # (stock to out flows edges already handled, any others
    # shouldn't affect rank)
    g.edge(
        ref.qual_name(),
        flow.qual_name(),
        style=style,
        arrowsize=".5",
        constraint=constraint,
    )


def add_to_var_edge(g: Digraph, ref, variable: Variable):
    """Edges pointing at variables are de-emphasized (usually these are
    auxiliary variables - variables calculated based on other variables.)"""
    g.edge(
        ref.qual_name(),
        variable.qual_name(),
        style="dotted",
        arrowsize=".5",
    )


# this is mostly only important for determining how to draw edges between refs
# in different scopes
# TODO: pretty sure there's still a mistake in this, will sometimes still try to
# include out of scope variable and just gets added as default ellipse.
def draw_appropriate_edge(g: Digraph, src: TrackedReference, dst: TrackedReference):
    """Switch between edge types correctly based on the types of src/dst nodes."""
    if (
        isinstance(src, Flow)
        and isinstance(dst, Stock)
        and (src in dst.in_flows or src in dst.out_flows)
    ):
        add_stock_io_edge(g, dst, src)
    elif (
        isinstance(src, Stock)
        and isinstance(dst, Flow)
        and (dst in src.in_flows or dst in src.out_flows)
    ):
        add_stock_io_edge(g, src, dst)
    elif isinstance(dst, Stock) and (src in dst.min_refs() or src in dst.max_refs()):
        add_stock_limit_edge(g, src, dst)
    elif isinstance(dst, Flow):
        add_to_flow_edge(g, src, dst)
    elif isinstance(dst, Variable):
        add_to_var_edge(g, src, dst)
    else:
        raise Exception(
            f"What type of nodes are you asking me to connect??? {src} -> {dst}?"
        )


def filter_variables(
    all_variables: list[Variable], exclude_names: list[str]
) -> list[Variable]:
    """Take a list of variables and a list of names of variables to remove from the list.

    Returns the list of variables that do NOT have a name in the exclude_names list.
    """
    filtered = [
        variable for variable in all_variables if variable.name not in exclude_names
    ]
    return filtered


def add_stocks(
    g: Digraph,
    stocks: list[Stock],
    sparklines: bool = False,
    traces: list[xr.Dataset] = None,
    universe: list[TrackedReference] = None,
):
    """Add nodes and edges for all passed stocks to the passed graph.

    Args:
        sparklines (bool): include small little line graph of each stock in the
            diagram. This only works if a simulation has been run.
    """

    render_in_parent_scope = []

    for stock in stocks:
        if universe is not None and stock not in universe:
            continue

        # add a node for each stock
        if not sparklines:
            g.node(
                name=stock.qual_name(),
                label=stock.label,
                shape="rect",
                group=stock.group,
            )
        else:
            # to force the sparkline graph image node to render right next to
            # its stock node, put them both in a subgraph with rank=same
            with g.subgraph(graph_attr={"rank": "same", "cluster": "false"}) as c:
                c.node(
                    name=stock.qual_name(),
                    label=stock.label,
                    shape="rect",
                    group=stock.group,
                )

                # generate the sparkline graph
                with plt.ioff():
                    fig, ax = plt.subplots(figsize=(2, 1))
                    # use the bayes one if available
                    if stock.model.trace is not None or traces is not None:
                        if traces is None:
                            traces = [stock.model.trace.prior]
                            if "posterior" in stock.model.trace:
                                traces.append(stock.model.trace.posterior)

                        reno.viz.compare_seq(
                            stock.qual_name(), traces, ax=ax, legend=False, title=""
                        )
                    else:
                        # TODO: can probably still use compare_seq for regular
                        # reno run plot? (since traces can be xrDataset)
                        stock.plot(ax)
                    ax.xaxis.set_ticks([])
                    fig.tight_layout()
                    os.makedirs(".plotcache", exist_ok=True)
                    fig.savefig(f".plotcache/{stock.qual_name()}.png")
                    plt.close(fig)

                c.node(
                    name=f"{stock.qual_name()}_fig",
                    label="",
                    image=f".plotcache/{stock.qual_name()}.png",
                    shape="none",
                    group=stock.group,
                )
                c.edge(
                    stock.qual_name(),
                    f"{stock.qual_name()}_fig",
                    constraint="true",
                    # weight="20",
                    # weight="200",
                    dir="none",
                )

        # look at each stock's inflows/outflows and add the appropriate edges
        # (this works even if the flow "nodes" haven't been added yet.)
        handled_refs = []
        for flow in stock.in_flows:
            if universe is not None and flow not in universe:
                continue
            if reno.utils.is_ref_in_parent_scope(flow, stock):
                # add refs themselves not qualname, need to access model
                render_in_parent_scope.append((flow, stock))
                continue
            add_stock_io_edge(g, stock, flow)
            handled_refs.append(flow.qual_name())
        for flow in stock.out_flows:
            if universe is not None and flow not in universe:
                continue
            if reno.utils.is_ref_in_parent_scope(flow, stock):
                # add refs themselves not qualname, need to access model
                render_in_parent_scope.append((flow, stock))
                continue
            add_stock_io_edge(g, stock, flow)
            handled_refs.append(flow.qual_name())
        remaining = []
        remaining.extend(stock.min_refs())
        remaining.extend(stock.max_refs())
        for ref in remaining:
            if ref not in handled_refs:
                if universe is not None and flow not in universe:
                    continue
                if reno.utils.is_ref_in_parent_scope(ref, stock):
                    # add refs themselves not qualname, need to access model
                    render_in_parent_scope.append((ref, stock))
                    continue
                add_stock_limit_edge(g, ref, stock)
    return g, render_in_parent_scope


def add_vars(
    g: Digraph,
    variables: list[Variable],
    exclude_vars: list[str],
    sparkdensities: bool = False,
    sparkall: bool = False,
    traces: list[xr.Dataset] = None,
    universe: list[TrackedReference] = None,
):
    """Add variables and edges between variables to the passed graphviz graph."""
    rendered_edges = []
    render_in_parent_scope = []
    for variable in variables:
        if universe is not None and variable not in universe:
            continue
        if not sparkdensities or (
            sparkdensities
            and (variable.qual_name() not in variable.model.trace_RVs and not sparkall)
        ):
            g.node(
                name=variable.qual_name(),
                label=variable.label,
                style="rounded,filled",
                fillcolor=NODE_COLOR,
                shape="rect",
                fontsize="10pt",
                height=".2",
                group=variable.group,
            )
        else:
            # to force the sparkdensity graph image node to render right next to
            # its variable node, put them both in a subgraph with rank=same
            with g.subgraph(graph_attr={"rank": "same"}) as c:
                c.node(
                    name=variable.qual_name(),
                    label=variable.label,
                    style="rounded,filled",
                    fillcolor=NODE_COLOR,
                    shape="rect",
                    fontsize="10pt",
                    height=".2",
                    group=variable.group,
                )

                # generate sparkdensity graph
                with plt.ioff():
                    fig, ax = plt.subplots(figsize=(2, 1))

                    if traces is None:
                        traces = [variable.model.trace.prior]
                        if "posterior" in variable.model.trace:
                            traces.append(variable.model.trace.posterior)

                    reno.viz.compare_posterior(
                        variable.qual_name(), traces, ax=ax, legend=False, title=""
                    )

                    ax.yaxis.set_ticks([])
                    fig.tight_layout()
                    os.makedirs(".plotcache", exist_ok=True)
                    fig.savefig(f".plotcache/{variable.qual_name()}.png")
                    plt.close(fig)

                c.node(
                    name=f"{variable.qual_name()}_fig",
                    label="",
                    image=f".plotcache/{variable.qual_name()}.png",
                    shape="none",
                    group=variable.group,
                )
                c.edge(
                    variable.qual_name(),
                    f"{variable.qual_name()}_fig",
                    constraint="true",
                    weight="20",
                    dir="none",
                )

        # add any edges between variables
        # -- unclear if correct --
        for ref in variable.seek_refs():
            if (
                universe is not None
                and ref not in universe
                or (hasattr(ref, "implicit") and ref.implicit)
            ):
                continue
            if (
                not isinstance(ref, reno.components.TimeRef)
                and ref.name not in exclude_vars
                and (ref.qual_name(), variable.qual_name()) not in rendered_edges
            ):
                # don't render connections to things in parent model - parent
                # model should handle
                if reno.utils.is_ref_in_parent_scope(ref, variable):
                    # add refs themselves not qualname, need to access model
                    render_in_parent_scope.append((ref, variable))
                    continue

                add_to_var_edge(g, ref, variable)
                rendered_edges.append((ref.qual_name(), variable.qual_name()))
        # -- /unclear if correct --
    return g, render_in_parent_scope


def add_flows(
    g: Digraph,
    flows: list[Flow],
    variables: list[Variable],
    sparkall: bool = False,
    traces: list[xr.Dataset] = None,
    universe: list[TrackedReference] = None,
):
    """Add flows and edges from variables to the passed graphviz graph."""
    rendered_edges = []

    render_in_parent_scope = []

    for flow in flows:
        if universe is not None and flow not in universe:
            continue
        if not sparkall:
            g.node(
                name=flow.qual_name(), label=flow.name, shape="plain", group=flow.group
            )
        else:
            # to force the sparkline graph image node to render right next to
            # its flow node, put them both in a subgraph with rank=same
            with g.subgraph(graph_attr={"rank": "same"}) as c:
                c.node(
                    name=flow.qual_name(),
                    label=flow.name,
                    shape="plain",
                    group=flow.group,
                )

                # generate the sparkline graph
                with plt.ioff():
                    fig, ax = plt.subplots(figsize=(2, 1))
                    # use the bayes one if available
                    if flow.model.trace is not None or traces is not None:
                        if traces is None:
                            traces = [flow.model.trace.prior]
                            if "posterior" in flow.model.trace:
                                traces.append(flow.model.trace.posterior)

                        reno.viz.compare_seq(
                            flow.qual_name(), traces, ax=ax, legend=False, title=""
                        )
                    else:
                        # TODO: can probably still use compare_seq for regular
                        # reno run plot? (since traces can be xrDataset)
                        # TODO: TODO: TODO: flow.plot doesn't exist?!
                        # flow.plot(ax)
                        if traces is None:
                            ds = flow.model.dataset()
                            traces = [ds]
                        reno.viz.compare_seq(
                            flow.qual_name(), traces, ax=ax, legend=False, title=""
                        )
                    ax.xaxis.set_ticks([])
                    fig.tight_layout()
                    os.makedirs(".plotcache", exist_ok=True)
                    fig.savefig(f".plotcache/{flow.qual_name()}.png")
                    plt.close(fig)

                c.node(
                    name=f"{flow.qual_name()}_fig",
                    label="",
                    image=f".plotcache/{flow.qual_name()}.png",
                    shape="none",
                    group=flow.group,
                )
                c.edge(
                    flow.qual_name(),
                    f"{flow.qual_name()}_fig",
                    constraint="true",
                    weight="20",
                    dir="none",
                )

        for ref in flow.seek_refs():
            if (
                universe is not None
                and ref not in universe
                or (hasattr(ref, "implicit") and ref.implicit)
            ):
                continue
            if (
                (
                    ref in variables
                    # TODO: a function called before this that removes variables
                    # of name x
                    and (ref.qual_name(), flow.qual_name()) not in rendered_edges
                )
                # TODO: ...what?
                or (
                    ref not in variables
                    and not isinstance(
                        ref, (TimeRef, Variable)
                    )  # TODO: Variable wasn't here before, with this here means we won't be able to show connections between submodels?
                    and (ref.qual_name(), flow.qual_name()) not in rendered_edges
                    # don't re-render a line from a stock to a flow if the flow
                    # is explicitly an outflow of the stock
                    and not (isinstance(ref, Stock) and flow in ref.out_flows)
                )
            ):
                # don't render connections to things in parent model - parent
                # model should handle
                if reno.utils.is_ref_in_parent_scope(ref, flow):
                    # add refs themselves not qualname, need to access model
                    render_in_parent_scope.append((ref, flow))
                    continue

                add_to_flow_edge(g, ref, flow)
                rendered_edges.append((ref.qual_name(), flow.qual_name()))

    return g, render_in_parent_scope
