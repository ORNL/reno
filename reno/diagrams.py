"""Functions for generating stock & flow diagrams.

Diagrams are graphviz dot diagrams, and notably don't follow the
exact format typically used with SFDs. The biggest difference is
sources and sinks aren't explicitly represented.
"""

# make it so we don't have to quote every type annotation ever
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import xarray as xr
from graphviz import Digraph

import reno

# TODO: ability to manually specify model colors


@dataclass
class RenderConfig:
    """Settings for what gets rendered in a stock and flow diagram."""

    show: list[reno.components.Reference] = None
    """List of individual references to include, this takes priority over groups."""
    hide: list[reno.components.Reference] = None
    """List of individual references to exclude, this takes priority over groups."""
    show_groups: list[str] = None
    """List of group names, any references part of any groups listed here will be
    displayed."""
    hide_groups: list[str] = None
    """List of group names, any references part of any groups listed here will be
    excluded."""
    universe: list[reno.components.Reference] = None
    """Limit the possible set of references to draw from when rendering. Specifying
    this applies restrictions to the other show*/hide* functions. The default of
    ``None`` places no restrictions."""
    vars: bool = True
    """Whether to include variables in the diagram (``True`` by default.)"""
    metrics: bool = False
    """Whether to include metrics in the diagram (``False`` by default.)"""

    group_colors: dict[str | tuple[reno.components.TrackedReference], str] = None
    """Modify existing group colors or define new groups and associated colors to render
    with. String keys refer to existing cgroup/group names. Tuples of references create
    new ad-hoc groups."""

    var_sparklines: bool = False
    """Include sparklines next to variables."""
    flow_sparklines: bool = False
    """Include sparklines next to flows."""
    stock_sparklines: bool = False
    """Include sparklines next to stocks."""
    metric_sparklines: bool = False
    """Include sparklines next to metrics."""

    traces: list[xr.Dataset] = None
    """The set of xarray datasets from simulation runs (numpy or pymc) to use in the
    sparklines."""

    theme: str = "light"
    """Whether to render the diagram in ``"light"`` or ``"dark"`` theme."""
    # TODO: make this an enum

    # TODO: include an option for setting the plotcache location

    lr: bool = False
    """Render the diagram top-bottom (default, if ``False``) or left-right."""

    def __post_init__(self):
        # TODO: just do default factories?
        if self.show_groups is None:
            self.show_groups = []
        if self.hide_groups is None:
            self.hide_groups = []
        if self.show is None:
            self.show = []
        if self.hide is None:
            self.hide = []
        if self.group_colors is None:
            self.group_colors = {}


class ModelDiagram:
    """A graph representation of a model, where nodes are the components and the edges
    represent the connections between them based on their equations.
    """

    subgraph_colors: ClassVar[list[str]] = [
        {"light": "#BBDDFF", "dark": "#334455"},
        {"light": "#DDBBFF", "dark": "#443355"},
    ]

    def __init__(
        self,
        model: reno.Model,
        _model_map: dict[reno.Model, ModelDiagram] = None,
        _level: int = 0,
        _parent: ModelDiagram = None,
    ):
        """Create a new diagram for the specified model."""
        # NOTE: any of the _ params are things passed from parent that are
        # required _during construction_ (since nodes and edges are built in
        # this constructor).
        # The alternative would be to modify state of child diagrams after
        # construction, but this would require building to occur outside of the
        # constructor. (an option to consider)
        self.model = model
        self.submodels: list[ModelDiagram] = []
        self.parent: ModelDiagram = _parent
        self.digraph: Digraph = None

        self.level = _level

        self.nodes: list[DiagramNode] = []

        self.spark_traces: list[xr.Dataset] = []

        # in order to not lose track of what node is in what diagram
        # corresponding to what model, this dictionary maps model objects to
        # their model diagram object. This is important for get_ref_node, and
        # note that it's a single shared dictionary amongst all the related
        # ModelDiagram objects.
        if _model_map is None:
            self.model_map = {}
        else:
            self.model_map = _model_map
        self.model_map[self.model] = self

        self._build_nodes()
        for submodel in self.model.models:
            model_diagram = ModelDiagram(
                submodel, _model_map=self.model_map, _level=self.level + 1, _parent=self
            )
            self.submodels.append(model_diagram)

        if self.level == 0:
            self._build_edges()
            self._fix_implicit_inflow_nodes()
            self.configure(RenderConfig())

    def _build_nodes(self) -> None:
        """Build out the graph structure in python land before trying to graphviz-ify
        it.
        """
        # add a new node of the correct type for each of the types of components
        self.nodes.extend(
            [StockDiagramNode(stock, self) for stock in self.model.stocks]
        )
        self.nodes.extend([FlowDiagramNode(flow, self) for flow in self.model.flows])
        self.nodes.extend([VarDiagramNode(var, self) for var in self.model.vars])
        self.nodes.extend(
            [MetricDiagramNode(metric, self) for metric in self.model.metrics]
        )

    def _build_edges(self) -> None:
        """Map out all of the edges between the nodes."""
        for node in self.nodes:
            node.map_edges()
        for model in self.submodels:
            model._build_edges()

    def _fix_implicit_inflow_nodes(self) -> None:
        """Implicit inflow nodes (flows that are implicit, flow into a stock, and have
        flows that flow into them) need to move their edges.

        Specifically, most reference and stock arrows need to move off of this node
        onto its sources.
        """
        for node in self.all_nodes():
            if isinstance(node, FlowDiagramNode):
                node.fix_implicit_inflow_edges()

    def get_topmost_diagram(self) -> ModelDiagram:
        """Recurse upwards through parent models until we hit the "top" model."""
        if self.parent is not None:
            return self.parent.get_topmost_diagram()
        return self

    def all_nodes(self) -> list[DiagramNode]:
        """Get a list of every node in this diagrams _greater_ context.

        Note that this returns all nodes from every diagram/sub-diagram
        involved, not just this one.
        """
        if self.parent is None:
            return self._all_nodes()
        return self.parent.all_nodes()

    def _all_nodes(self) -> list[DiagramNode]:
        """Recurse downwards to get every node from every submodel."""
        nodes = [*self.nodes]
        for model in self.submodels:
            nodes.extend(model._all_nodes())
        return nodes

    def all_edges(self) -> list[DiagramEdge]:
        """Return _all_ edges in the entire diagram.

        This requires navigating to the "top" parent model and recursively working
        down from there.
        """
        if self.parent is None:
            return self._all_edges()
        # ..."recurse" upwards in order to recurse downwards from the top?
        return self.parent.all_edges()

    def _all_edges(self) -> list[DiagramEdge]:
        """Recurse downwards to get every edge involved with nodes in every submodel."""
        everything = []
        for node in self.nodes:
            for edge in node.edges:
                if edge not in everything:
                    everything.append(edge)
        for model in self.submodels:
            sub_everything = model._all_edges()
            for edge in sub_everything:
                if edge not in everything:
                    everything.append(edge)
        return everything

    def configure(self, config: RenderConfig) -> None:
        """Apply a passed configuration to all nodes/edges."""
        self._get_model_traces(config)
        for node in self.all_nodes():
            node.configure_render(config)
            node.configure_color(config)
            node.configure_sparklines(config)

        for edge in self.all_edges():
            edge.configure_color(config)

    def _get_model_traces(self, config: RenderConfig) -> None:
        """Determine based on the configuration what traces to use for sparklines.

        (Traces can be found on the model from past simulations even if not manually
        passed in the config.)
        """
        # NOTE: this only sets the traces on the parent model. Anywhere where
        # the traces are used, you have to use the get_topmost_diagram function.
        # TODO: set property for spark traces to automatically get topmost?
        self.spark_traces = []
        if (
            config.flow_sparklines
            or config.stock_sparklines
            or config.var_sparklines
            or config.metric_sparklines
        ):
            if config.traces is not None:
                # highest priority is a manually specified set of traces
                self.spark_traces = config.traces
                return
            if self.model.trace is not None:
                # next highest is if a previous trace exists on the model
                if "prior" in self.model.trace:
                    self.spark_traces.append(self.model.trace.prior)
                if "posterior" in self.model.trace:
                    self.spark_traces.append(self.model.trace.posterior)
                return
            # otherwise try to get a previous numpy run on the model
            self.spark_traces = [self.model.dataset()]

    def _reset_edge_render_state(self) -> None:
        """Edges track a ``rendered`` variable to avoid double-rendering. Find
        all edges at all levels and reset this to ``False``.
        """
        for edge in self.all_edges():
            edge.rendered = False

    def _make_graph(self, config: RenderConfig) -> Digraph:
        """Create the initial Digraph instance."""
        rankdir = "LR" if config.lr else "TB"
        bgcolor = {"light": None, "dark": "#181818"}
        outlinecolor = {"light": "black", "dark": "#e6e6e6"}

        if self.level == 0:
            graph_attrs = dict(
                rankdir=rankdir,
                bgcolor=bgcolor[config.theme],
                style="filled",
                mclimit="0.0",
            )
            g = Digraph(
                name=self.model.name,
                graph_attr=graph_attrs,
                node_attr=dict(color=outlinecolor[config.theme]),
            )
        elif self.level > 0:
            g = Digraph(
                name=self.model.name, node_attr=dict(color=outlinecolor[config.theme])
            )
            g.attr(rankdir=rankdir)
            g.attr(style="filled")
            g.attr(color=ModelDiagram.subgraph_colors[self.level - 1][config.theme])
            g.attr(cluster="true")
            g.attr(label=self.model.label)
            g.attr(fontcolor="#888888")
        return g

    def to_graphviz(self, config: RenderConfig = None) -> Digraph:
        """Generate the graphviz Digraph and return it.

        Output is also stored on ``self.digraph``.
        """
        if self.level == 0:
            self._reset_edge_render_state()
            if config is not None:
                self.configure(config)

        g = self._make_graph(config)
        for node in self.nodes:
            node.add_to_graphviz(g)

        for model in self.submodels:
            sub_g = model.to_graphviz(config)
            g.subgraph(sub_g)

        if self.level == 0:
            # the edges need to be included _outside_ of cluster definitions,
            # otherwise nodes can get incorrectly moved into a cluster because
            # of where the edge is defined
            for node in self.all_nodes():
                for edge in node.edges:
                    edge.add_to_graphviz(g)

        self.digraph = g
        return g

    def _repr_png_(self) -> bytes:
        """Used to automatically render the diagram when used in jupyter."""
        if self.digraph is None:
            self.to_graphviz()
        return self.digraph._repr_mimebundle_(include=["image/png"])["image/png"]

    # NOTE: svg repr in jupyter lab doesn't work well because sparkplots rely on
    # href'd images to display correctly. This doesn't seem like an issue that's
    # going to be resolved any time soon from the jupyter side
    # def _repr_svg_(self) -> str:
    #     if self.digraph is None:
    #         self.to_graphviz()
    #     return self.digraph.pipe(format="svg", encoding="ascii")

    def get_ref_node(self, ref: reno.Reference) -> DiagramNode:
        """Get the DiagramNode associated with a reference.

        This is challenging because references don't have any direct connection to
        nodes. This gets used for correctly constructing an edge, which requires a node
        on each side when you may only have a reference from seek_refs.
        """
        # NOTE: likely a problem if a reference is never explicitly assigned to
        # a model?
        diagram = self.model_map[ref.model]
        for node in diagram.nodes:
            if node.ref == ref:
                return node
        return None


class DiagramNode:
    """Parent class for any type of node representing a reference in a model."""

    default_color: ClassVar[dict[str, str]] = {
        "light": "transparent",
        "dark": "#333333",
    }
    default_font_color: ClassVar[dict[str, str]] = {"light": "black", "dark": "#e6e6e6"}
    default_sparkline_edge_color: ClassVar[dict[str, str]] = {
        "light": "black",
        "dark": "#e6e6e6",
    }

    shape: ClassVar[str] = "rect"
    style: ClassVar[str] = "filled"
    other_attrs: ClassVar[dict[str, str]] = {}

    def __init__(self, ref: reno.Reference, diagram: ModelDiagram):
        """Initialize a new node for the passed parent diagram and the provided reno
        component.
        """
        self.diagram = diagram
        self.ref = ref
        self.sparkline = False
        self.sparkline_edge_color = None
        self.render = True
        self.color = None
        self.font_color = None
        self.theme: str = "light"

        self.edges: list[DiagramEdge] = []

    def configure_render(self, config: RenderConfig) -> None:  # noqa: C901
        """Decide if this node should be rendered based on configuration and underlying
        reference.

        Precedence/priority in this determination (earlier in the list overrides later):
        1. An implicit reference is never rendered
        2. If a universe is specified and the reference isn't in it, don't render it.
        3. Individually specified show/hide references
        4. Show/hide color groups
        5. Default model hide groups
        6. Blanket variable/metric on/off
        """
        self.render = True

        # lowest priority is blanket variables/metrics on or off
        # TODO: possibly move this higher than cgroups?
        if isinstance(self, MetricDiagramNode) and not config.metrics:
            self.render = False

        if isinstance(self, VarDiagramNode) and not config.vars:
            self.render = False

        # fifth highest priority is default hide groups on model
        if self.check_str_or_listpart_in_list(
            self.ref.cgroup, self.ref.model.default_hide_groups
        ):
            self.render = False
        if self.ref.group in self.ref.model.default_hide_groups:
            self.render = False

        # fourth highest priority is show/hide color groups
        if self.check_str_or_listpart_in_list(self.ref.cgroup, config.show_groups):
            self.render = True
        if self.check_str_or_listpart_in_list(self.ref.cgroup, config.hide_groups):
            self.render = False

        # third highest priorities are individually specified show/hide controls
        if self.ref in config.show:
            self.render = True
        if self.ref in config.hide:
            self.render = False

        # second highest priority - if a universe has been specified, don't
        # render anything outside of that universe
        if config.universe is not None and self.ref not in config.universe:
            self.render = False

        # highest priority - if the reference is implicit, never render
        if self.ref.implicit:
            self.render = False

    def configure_color(self, config: RenderConfig) -> None:
        """Determine what colors should be used for this node given the
        configuration.
        """
        self.theme = config.theme

        # lowest priority is the default
        self.color = self.default_color[config.theme]

        # next lowest priority are model-defined default group colors
        # (cgroup takes priority over group)
        default_group_check = self.check_str_or_listpart_in_dict(
            self.ref.group, self.ref.model.group_colors
        )
        if default_group_check is not None:
            self.color = default_group_check
        default_cgroup_check = self.check_str_or_listpart_in_dict(
            self.ref.cgroup, self.ref.model.group_colors
        )
        if default_cgroup_check is not None:
            self.color = default_cgroup_check

        # next level of precedence is a group specified in config's group_colors
        config_check = self.check_str_or_listpart_in_dict(
            self.ref.cgroup, config.group_colors
        )
        if config_check is not None:
            self.color = config_check

        # manually specified groups in config (tuples of references) take
        # highest priority
        for group in config.group_colors:
            if isinstance(group, tuple) and self.ref in group:
                self.color = config.group_colors[group]
                break

        self.font_color = self.default_font_color[config.theme]
        self.sparkline_edge_color = self.default_sparkline_edge_color[config.theme]

    def check_str_or_listpart_in_list(
        self, vals: str | list, containing_list: list[str]
    ) -> bool:
        """Utility function to determine if a passed value (either a string or list of
        strings) is in (or at least one instance of the list is in) the passed containing
        list.

        This is primarily used for checking containment of a group name/list of group
        names in set of group names.
        """
        if isinstance(vals, str) and vals in containing_list:
            return True
        if isinstance(vals, list):
            for val in vals:
                if val in containing_list:
                    return True
        return False

    def check_str_or_listpart_in_dict(
        self, keys: str | list, dictionary: dict[str, str]
    ) -> str:
        """Utility function to retrieve the value in a dictionary where the key is found
        in the passed value (either a singular string which would be a direct match, or
        a list where the key is found in that list).

        This is primarily used for getting values associated with groups (e.g. colors).
        """
        if isinstance(keys, str) and keys in dictionary:
            return dictionary[keys]
        if isinstance(keys, list):
            for key in keys:
                if key in dictionary:
                    return dictionary[key]
        return None

    def graphviz_ref_node(self, g: Digraph = None) -> None:
        """Add a node for the reference itself to the graph."""
        g.node(
            name=self.ref.qual_name(),
            label=self.ref.label,
            shape=self.shape,
            group=self.ref.group,
            style=self.style,
            fillcolor=self.color,
            fontcolor=self.font_color,
            **self.other_attrs,
        )

    def add_to_graphviz(self, g: Digraph = None) -> None:
        """Add this node to the digraphviz Digraph, accounting for sparklines if
        needed.
        """
        if self.render:
            if not self.sparkline:
                self.graphviz_ref_node(g)
            else:
                with g.subgraph(
                    name=f"cluster_{self.ref.qual_name}",
                    graph_attr={"label": "", "style": None, "color": "invis"},
                ) as c:
                    self.graphviz_ref_node(c)
                    plot_path = self.generate_sparkline()

                    c.node(
                        name=f"{self.ref.qual_name()}_fig",
                        label="",
                        image=plot_path,
                        shape="none",
                        group=self.ref.group,
                    )
                    c.edge(
                        self.ref.qual_name(),
                        f"{self.ref.qual_name()}_fig",
                        constraint="false",
                        color=self.sparkline_edge_color,
                        weight="20",
                        dir="none",
                    )

    @staticmethod
    def add_edge(edge: DiagramEdge) -> None:
        """Add the given edge to _both_ involved nodes.

        Static method because this doesn't depend on either node in particular.
        """
        if edge not in edge.source.edges:
            edge.source.edges.append(edge)
        if edge not in edge.target.edges:
            edge.target.edges.append(edge)

    def map_edges(self) -> None:
        """Find all connected nodes and determine how corresponding edges need to be
        drawn.
        """
        # implemented in subclasses

    def generate_sparkline(self) -> str:
        """Returns the filepath of the saved plot so graphviz can display
        in a node.
        """
        # store current plot rcParams to avoid side effects
        if self.theme == "light":
            mpl_style_name = "default"
        elif self.theme == "dark":
            mpl_style_name = "dark_background"

        # generate the sparkline graph
        with plt.style.context(mpl_style_name), plt.ioff():
            # fig, ax = plt.subplots(figsize=(1.5, .75))
            traces = self.diagram.get_topmost_diagram().spark_traces

            # based on ref type
            # TODO: need better way to handle what the set of coords
            # are, because a metric with a timeseries[-1] still counted
            # as dynamic
            if (
                isinstance(self.ref, (reno.Stock, reno.Flow))
                or not self.ref.is_static()
            ):
                fig, ax = plt.subplots(figsize=(1.65, 0.75))
                reno.viz.compare_seq(
                    self.ref.qual_name(), traces, ax=ax, legend=False, title=""
                )
                ax.xaxis.set_ticks([])
            else:
                fig, ax = plt.subplots(figsize=(1.5, 0.90))
                reno.viz.compare_posterior(
                    self.ref.qual_name(), traces, ax=ax, legend=False, title=""
                )
                ax.yaxis.set_ticks([])
            ax.tick_params(labelsize=8)
            fig.tight_layout(pad=0.05)
            fig.patch.set_alpha(0)

            # save the figure (graphviz needs a path to render an image)
            cache_dir = Path(".plotcache")
            cache_dir.mkdir(exist_ok=True)
            filepath = cache_dir / f"{self.ref.qual_name()}.png"
            fig.savefig(filepath)
            plt.close(fig)

        return str(filepath)


class StockDiagramNode(DiagramNode):
    """Node for a stock component."""

    def configure_sparklines(self, config: RenderConfig) -> None:
        """Set whether to render a sparkline for this reference based on config."""
        self.sparkline = config.stock_sparklines

    def map_edges(self) -> None:
        """Find and add all the necessary edges that connect to this stock.

        This is based on inflows, outflows, and any references found in the stock
        minimum/maximum constraint equations.
        """
        for flow in self.ref.in_flows:
            DiagramNode.add_edge(
                StockIODiagramEdge(self.diagram.get_ref_node(flow), self)
            )
        for flow in self.ref.out_flows:
            DiagramNode.add_edge(
                StockIODiagramEdge(self, self.diagram.get_ref_node(flow))
            )
        for ref in self.ref.min_refs() + self.ref.max_refs():
            # TODO: check for timeref here too?
            DiagramNode.add_edge(
                StockLimitDiagramEdge(self.diagram.get_ref_node(ref), self)
            )


class FlowDiagramNode(DiagramNode):
    """Node for a flow component."""

    default_color: ClassVar[dict[str, str]] = {
        "light": "transparent",
        "dark": "transparent",
    }
    shape: ClassVar[str] = "plain"

    def configure_sparklines(self, config: RenderConfig) -> None:
        """Set whether to render a sparkline for this reference based on config."""
        self.sparkline = config.flow_sparklines

    def map_edges(self) -> None:
        """Find and add all the necessary edges that connect to this stock.

        This is based on any refs found in the equation.
        """
        for ref, ref_types in self.ref.seek_refs(include_ref_types=True).items():
            if isinstance(ref, reno.components.TimeRef):
                continue
            # TODO: ignore stock outflows?
            if "inflow" in ref_types:
                DiagramNode.add_edge(
                    StockIODiagramEdge(self.diagram.get_ref_node(ref), self)
                )
            else:
                DiagramNode.add_edge(
                    ToFlowDiagramEdge(self.diagram.get_ref_node(ref), self)
                )

    def fix_implicit_inflow_edges(self) -> None:  # noqa: C901
        """Edges that connect to implicit flows (especially if inflows to a stock)
        have to be moved to the previous references in the chain to avoid incorrect
        breaks in the diagram.

        Any StockIO edges previously connected to this node as a source get moved
        (and duplicated if it was an implicit flow made up of several other flows)
        to the previous flow nodes.

        All other reference edges get moved to the previous references in the chain.
        """
        is_inflow = False
        for edge in self.edges:
            if (
                isinstance(edge, StockIODiagramEdge)
                and edge.source == self
                and not edge.other(self).ref.implicit
            ):
                is_inflow = True
                break

        if not is_inflow or not self.ref.implicit:
            return

        for edge in self.edges:
            # add inflow arrows to any flows that led into _this_ node
            if isinstance(edge, StockIODiagramEdge) and edge.source == self:
                for ref in self.ref.seek_refs():
                    if isinstance(ref, reno.Flow):
                        DiagramNode.add_edge(
                            StockIODiagramEdge(
                                self.diagram.get_ref_node(ref), edge.target
                            )
                        )

            # add any reference edges on _this_ node to flows that led into this node
            if isinstance(edge, ToFlowDiagramEdge) and edge.target == self:
                for ref in self.ref.seek_refs():
                    if isinstance(ref, reno.Flow):
                        DiagramNode.add_edge(
                            ToFlowDiagramEdge(
                                edge.source, self.diagram.get_ref_node(ref)
                            )
                        )


class VarDiagramNode(DiagramNode):
    """Node for a variable component."""

    default_color: ClassVar[dict[str, str]] = {
        "light": "lightgreen",
        "dark": "darkgreen",
    }
    style: ClassVar[str] = "rounded,filled"
    other_attrs: ClassVar[dict[str, str]] = {"fontsize": "10pt", "height": ".2"}

    def configure_sparklines(self, config: RenderConfig) -> None:
        """Set whether to render a sparkline for this reference based on config."""
        self.sparkline = config.var_sparklines

    def map_edges(self) -> None:
        """Find and add all the necessary edges that connect to this stock.

        This is based on any refs found in the equation.
        """
        for ref in self.ref.seek_refs():
            if isinstance(ref, reno.components.TimeRef):
                continue
            if isinstance(ref, reno.components.HistoricalValue):
                ref = ref.tracked_ref
            DiagramNode.add_edge(ToVarDiagramEdge(self.diagram.get_ref_node(ref), self))


class MetricDiagramNode(DiagramNode):
    """Node for a metric component."""

    default_color: ClassVar[dict[str, str]] = {"light": "purple", "dark": "#551133"}
    default_font_color: ClassVar[dict[str, str]] = {
        "light": "#e6e6e6",
        "dark": "#e6e6e6",
    }
    shape: ClassVar[str] = "ellipse"
    other_attrs: ClassVar[dict[str, str]] = {"fontsize": "10pt", "height": ".2"}

    def configure_sparklines(self, config: RenderConfig) -> None:
        """Set whether to render a sparkline for this reference based on config."""
        self.sparkline = config.metric_sparklines

    def map_edges(self) -> None:
        """Find and add all the necessary edges that connect to this stock.

        This is based on any refs found in the equation.
        """
        for ref in self.ref.seek_refs():
            if isinstance(ref, reno.components.TimeRef):
                continue
            DiagramNode.add_edge(
                ToMetricDiagramEdge(self.diagram.get_ref_node(ref), self)
            )


class DiagramEdge:
    """A visual connection between nodes (references) in a stock and flow diagram.

    Edge information stands outside of an individual diagram instance since there can be
    crossmodel/intermodel connections. Edges are stored as separate instances so that
    the nodes on both ends can share the edge objects.
    """

    default_color: ClassVar[dict[str, str]] = {"light": "black", "dark": "white"}
    style: ClassVar[str] = None
    weight: ClassVar[str] = None
    arrowsize: ClassVar[str] = None

    PRIORITY: int = 0
    """When two edges can be drawn between the same source and target, priority (based
    on type) is used to determine which edge is actually drawn."""

    def __init__(self, source: DiagramNode, target: DiagramNode):
        """Create an edge instance that connects from the passed source node to the
        passed target node.
        """
        self.source = source
        self.target = target
        self.color = None

        self.rendered = False
        """Flag to ensure an edge doesn't get double-rendered."""

        if self not in source.edges:
            source.edges.append(self)
        if self not in target.edges:
            target.edges.append(self)

    def configure_color(self, config: RenderConfig) -> None:
        """Determine what color should be used for this edge given the configuration."""
        self.color = self.default_color[config.theme]

    def other(self, node: DiagramNode) -> DiagramNode:
        """Given one side of the edge, get the node at the other end."""
        if node == self.source:
            return self.target
        if node == self.target:
            return self.source
        # TODO: error?
        return None

    def should_render(self) -> bool:
        """Determine if this edge should be rendered or not.

        This determination is based on whether both end nodes are supposed to render,
        and if any other edges between those same nodes have already rendered.
        """
        if self.rendered:
            return False
        if not self.source.render or not self.target.render:
            return False
        if self.source == self.target:
            # don't render a self loop (this tends to happen because of the
            # implicit flow fixes)
            return False

        max_edge_priority = 0
        for edge in self.find_duplicate_edges():
            if max_edge_priority < edge.PRIORITY:
                max_edge_priority = edge.PRIORITY
        if max_edge_priority > self.PRIORITY:  # noqa: SIM103
            return False

        return True

    def mark_rendered(self) -> None:
        """Set rendered state of this edge (and all duplicate/similar edges) to
        ``True``.

        Without this, "duplicate" edges will sometimes still render.
        """
        self.rendered = True
        for edge in self.find_duplicate_edges():
            edge.rendered = True

    # TODO: should it include itself or no? (currently does not)
    def find_duplicate_edges(self) -> list[DiagramEdge]:
        """Get all other edges between the same source and target.

        Only includes those pointing the same direction. The list does not include
        this current edge.
        """
        duplicates = []
        for edge in self.source.edges:
            if edge == self:
                continue
            if edge.source == self.source and edge.target == self.target:
                duplicates.append(edge)
        for edge in self.target.edges:
            if edge == self:
                continue
            if (
                edge.source == self.source
                and edge.target == self.target
                and edge not in duplicates
            ):
                # technically this should never happen because every edge in
                # source should also be in target
                duplicates.append(edge)
        return duplicates

    def add_to_graphviz(self, g: Digraph = None) -> None:
        """Add this edge to the graphviz Digraph."""
        if self.should_render():
            g.edge(
                self.source.ref.qual_name(),
                self.target.ref.qual_name(),
                color=self.color,
                style=self.style,
                weight=self.weight,
                arrowsize=self.arrowsize,
            )
            self.mark_rendered()


class StockIODiagramEdge(DiagramEdge):
    """Edge between a stock and a inflow or outflow."""

    style: ClassVar[str] = "bold"
    weight: ClassVar[str] = "50"
    arrowsize: ClassVar[str] = None
    PRIORITY = 5

    def configure_color(self, config: RenderConfig) -> None:
        """Determine what color should be used for this edge given the configuration."""
        self.color = self.default_color[config.theme]

        # next highest is if either side happens to have a color
        if self.source.color != self.source.default_color[config.theme]:
            self.color = self.source.color
        if self.target.color != self.target.default_color[config.theme]:
            self.color = self.target.color

        # highest priority is the stock color
        # NOTE: if there's an "inflow" op neither one is a stock, so just assume
        # the source by default.
        stock_node = self.source
        if isinstance(self.target.ref, reno.Stock):
            stock_node = self.target
        if stock_node.color != stock_node.default_color[config.theme]:
            self.color = stock_node.color


class StockLimitDiagramEdge(DiagramEdge):
    """Edge between a variable and a stock, where the variable is referenced
    in a limit/constraint (min/max) on the stock.
    """

    style: ClassVar[str] = "dotted"
    arrowsize: ClassVar[str] = ".5"
    PRIORITY = 3


class ToVarDiagramEdge(DiagramEdge):
    """Reference edge that connects to a variable."""

    style: ClassVar[str] = "dotted"
    arrowsize: ClassVar[str] = ".5"
    PRIORITY = 1


class ToFlowDiagramEdge(DiagramEdge):
    """Reference edge that connects to a flow."""

    arrowsize: ClassVar[str] = ".5"
    PRIORITY = 2

    def add_to_graphviz(self, g: Digraph = None) -> None:
        """Add this edge to the graphviz Digraph.

        Handled separately in this class because the style should
        depend on the source component type.
        """
        if self.should_render():
            style = "dotted" if isinstance(self.source.ref, reno.Variable) else "dashed"
            constraint = "false" if isinstance(self.source.ref, reno.Stock) else "true"
            # TODO: deemphasize option?
            # weight = "1" if

            g.edge(
                self.source.ref.qual_name(),
                self.target.ref.qual_name(),
                style=style,
                arrowsize=self.arrowsize,
                constraint=constraint,
                color=self.color,
            )
            self.mark_rendered()


class ToMetricDiagramEdge(DiagramEdge):
    """Reference edge that represents usage in a metric."""

    default_color: ClassVar[dict[str, str]] = {"light": "#444444", "dark": "#999999"}
    style: ClassVar[str] = "dotted"
    arrowsize: ClassVar[str] = ".5"
    PRIORITY = 1
