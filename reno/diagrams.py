# make it so we don't have to quote every type annotation ever
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import xarray as xr
from graphviz import Digraph

import reno

# TODO: ability to manually specify model colors


@dataclass
class RenderConfig:
    show: list[reno.components.Reference] = None
    hide: list[reno.components.Reference] = None
    show_groups: list[str] = None
    hide_groups: list[str] = None
    universe: list[reno.components.Reference] = None
    vars: bool = True
    metrics: bool = False

    group_colors: dict[str | tuple[reno.components.TrackedReference], str] = None

    var_sparklines: bool = False
    flow_sparklines: bool = False
    stock_sparklines: bool = False
    metric_sparklines: bool = False

    traces: list[xr.Dataset] = None

    theme: str = "light"
    # TODO: make this an enum

    lr: bool = False

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
    subgraph_colors: ClassVar[list[str]] = [
        {"light": "#BBDDFF", "dark": "#334455"},
        {"light": "#DDBBFF", "dark": "#443355"},
    ]

    def __init__(
        self,
        model: reno.Model,
        bg: str = None,
        _model_map: dict[reno.Model, ModelDiagram] = None,
        _level: int = 0,
        _parent: ModelDiagram = None,
    ):
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

        self.bg = bg
        """background color."""

        self.level = _level

        self.nodes: list[DiagramNode] = []

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

        self.build_nodes()
        for submodel in self.model.models:
            model_diagram = ModelDiagram(
                submodel, _model_map=self.model_map, _level=self.level + 1, _parent=self
            )
            self.submodels.append(model_diagram)

        if self.level == 0:
            self.build_edges()
            self.fix_implicit_inflow_nodes()
            self.configure(RenderConfig())

    def build_nodes(self) -> None:
        """Build out the graph structure in python land before trying to graphviz-ify
        it.
        """
        self.nodes.extend(
            [StockDiagramNode(stock, self) for stock in self.model.stocks]
        )
        self.nodes.extend([FlowDiagramNode(flow, self) for flow in self.model.flows])
        self.nodes.extend([VarDiagramNode(var, self) for var in self.model.vars])
        self.nodes.extend(
            [MetricDiagramNode(metric, self) for metric in self.model.metrics]
        )

    def build_edges(self) -> None:
        for node in self.nodes:
            node.map_edges()
        for model in self.submodels:
            model.build_edges()

    def fix_implicit_inflow_nodes(self) -> None:
        """Implicit inflow nodes (flows that are implicit, flow into
        a stock, and have flows that flow into them) need to move their
        edges.

        TODO: additional explanation
        """
        for node in self.all_nodes():
            if isinstance(node, FlowDiagramNode):
                node.fix_implicit_inflow_edges()

    def all_nodes(self) -> list[DiagramNode]:
        if self.parent is None:
            return self._all_nodes()
        return self.parent.all_nodes()

    def _all_nodes(self) -> list[DiagramNode]:
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
        for node in self.all_nodes():
            node.configure_render(config)
            node.configure_color(config)
            node.configure_sparklines(config)

        for edge in self.all_edges():
            edge.configure_color(config)

    def _reset_edge_render_state(self) -> None:
        """Edges track a ``rendered`` variable to avoid double-rendering. Find
        all edges at all levels and reset this to ``False``.
        """
        for edge in self.all_edges():
            edge.rendered = False

    def _make_graph(self, config: RenderConfig) -> Digraph:
        # TODO: RenderConfig (apply to all nodes and all submodel nodes)
        # TODO: lr/tb and dark mode attrs

        rankdir = "LR" if config.lr else "TB"
        bgcolor = {"light": None, "dark": "#181818"}
        outlinecolor = {"light": "black", "dark": "#e6e6e6"}

        if self.level == 0:
            graph_attrs = dict(
                rankdir=rankdir, bgcolor=bgcolor[config.theme], style="filled"
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

        Output is also stored on self.digraph.
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

    def _repr_svg_(self) -> str:
        if self.digraph is None:
            self.to_graphviz()
        return self.digraph.pipe(format="svg", encoding="ascii")

    def get_ref_node(self, ref: reno.Reference) -> DiagramNode:
        """Get the DiagramNode associated with a reference. This is challenging because
        references don't have any direct connection to nodes. This gets used for
        correctly constructing an edge, which requires a node on each side when you may
        only have a reference from seek_refs.
        """
        # NOTE: likely a problem if a reference is never explicitly assigned to
        # a model?
        diagram = self.model_map[ref.model]
        for node in diagram.nodes:
            if node.ref == ref:
                return node
        return None


class DiagramNode:
    default_color: ClassVar[dict[str, str]] = {
        "light": "transparent",
        "dark": "#333333",
    }
    default_font_color: ClassVar[dict[str, str]] = {"light": "black", "dark": "#e6e6e6"}
    shape: ClassVar[str] = "rect"
    style: ClassVar[str] = "filled"
    other_attrs: ClassVar[dict[str, str]] = {}

    def __init__(self, ref: reno.Reference, diagram: ModelDiagram):
        self.diagram = diagram
        self.ref = ref
        self.sparkline = False
        self.render = True
        self.color = None
        self.font_color = None

        self.edges: list[DiagramEdge] = []

    def configure_render(self, config: RenderConfig) -> None:
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

    def check_str_or_listpart_in_list(
        self, vals: str | list, containing_list: list[str]
    ) -> bool:
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
        if isinstance(keys, str) and keys in dictionary:
            return dictionary[keys]
        if isinstance(keys, list):
            for key in keys:
                if key in dictionary:
                    return dictionary[key]
        return None

    def add_to_graphviz(self, g: Digraph = None) -> None:
        if self.render:
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

    @staticmethod
    def add_edge(edge: DiagramEdge) -> None:
        if edge not in edge.source.edges:
            edge.source.edges.append(edge)
        if edge not in edge.target.edges:
            edge.target.edges.append(edge)

    def map_edges(self) -> None:
        """Find all connected nodes and determine how corresponding edges need to be
        drawn.
        """
        pass


class StockDiagramNode(DiagramNode):
    def configure_sparklines(self, config: RenderConfig) -> None:
        self.sparkline = config.stock_sparklines

    def map_edges(self) -> None:
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
    default_color: ClassVar[dict[str, str]] = {
        "light": "transparent",
        "dark": "transparent",
    }
    shape: ClassVar[str] = "plain"

    def configure_sparklines(self, config: RenderConfig) -> None:
        self.sparkline = config.flow_sparklines

    def map_edges(self) -> None:
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

    def fix_implicit_inflow_edges(self) -> None:
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
    default_color: ClassVar[dict[str, str]] = {
        "light": "lightgreen",
        "dark": "darkgreen",
    }
    style: ClassVar[str] = "rounded,filled"
    other_attrs: ClassVar[dict[str, str]] = {"fontsize": "10pt", "height": ".2"}

    def configure_sparklines(self, config: RenderConfig) -> None:
        self.sparkline = config.var_sparklines

    def map_edges(self) -> None:
        for ref in self.ref.seek_refs():
            if isinstance(ref, reno.components.TimeRef):
                continue
            DiagramNode.add_edge(ToVarDiagramEdge(self.diagram.get_ref_node(ref), self))


class MetricDiagramNode(DiagramNode):
    default_color: ClassVar[dict[str, str]] = {"light": "purple", "dark": "#551133"}
    default_font_color: ClassVar[dict[str, str]] = {
        "light": "#e6e6e6",
        "dark": "#e6e6e6",
    }
    shape: ClassVar[str] = "ellipse"
    other_attrs: ClassVar[dict[str, str]] = {"fontsize": "10pt", "height": ".2"}

    def configure_sparklines(self, config: RenderConfig) -> None:
        self.sparkline = config.metric_sparklines

    def map_edges(self) -> None:
        for ref in self.ref.seek_refs():
            if isinstance(ref, reno.components.TimeRef):
                continue
            DiagramNode.add_edge(
                ToMetricDiagramEdge(self.diagram.get_ref_node(ref), self)
            )


class DiagramEdge:
    """Edge information stands outside of an individual diagram since there can be
    crossmodel/intermodel connections. Edges are stored as separate instances so that
    the nodes on both ends can share the edge objects.
    """

    default_color: ClassVar[dict[str, str]] = {"light": "black", "dark": "white"}
    style: ClassVar[str] = None
    weight: ClassVar[str] = None
    arrowsize: ClassVar[str] = None

    PRIORITY: int = 0
    """When two edges can be drawn between the same source and target, priority (based on type)
    is used to determine which edge is actually drawn."""

    def __init__(self, source: DiagramNode, target: DiagramNode):
        self.source = source
        self.target = target
        self.color = None

        self.rendered = False
        """Ensure an edge doesn't get double-rendered"""

        if self not in source.edges:
            source.edges.append(self)
        if self not in target.edges:
            target.edges.append(self)

    def configure_color(self, config: RenderConfig) -> None:
        self.color = self.default_color[config.theme]

    def other(self, node: DiagramNode) -> DiagramNode:
        """Given one side of the edge, get the other side."""
        if node == self.source:
            return self.target
        if node == self.target:
            return self.source
        # TODO: error?
        return None

    def should_render(self) -> bool:
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
        if max_edge_priority > self.PRIORITY:
            return False

        return True

    # TODO: function to mark rendered (then it will also mark all duplicate
    # edges as rendered)
    def mark_rendered(self) -> None:
        """Set rendered state of this edge (and all duplicate/similar edges) to ``True``.

        Without this, "duplicate" edges will sometimes still render.
        """
        self.rendered = True
        for edge in self.find_duplicate_edges():
            edge.rendered = True

    # TODO: should it include itself or no? (currently does not)
    def find_duplicate_edges(self) -> list[DiagramEdge]:
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
    style: ClassVar[str] = "bold"
    weight: ClassVar[str] = "50"
    arrowsize: ClassVar[str] = None
    PRIORITY = 5

    def configure_color(self, config: RenderConfig) -> None:
        self.color = self.default_color[config.theme]

        # next highest is if either side happens to have a color
        if self.source.color != self.source.default_color[config.theme]:
            self.color = self.source.color[config.theme]
        if self.target.color != self.target.default_color[config.theme]:
            self.color = self.target.color[config.theme]

        # highest priority is the stock color
        # NOTE: if there's an "inflow" op neither one is a stock, so just assume
        # the source by default.
        stock_node = self.source
        if isinstance(self.target.ref, reno.Stock):
            stock_node = self.target
        if stock_node.color != stock_node.default_color[config.theme]:
            self.color = stock_node.color[config.theme]


class StockLimitDiagramEdge(DiagramEdge):
    style: ClassVar[str] = "dotted"
    arrowsize: ClassVar[str] = ".5"
    PRIORITY = 3


class ToVarDiagramEdge(DiagramEdge):
    style: ClassVar[str] = "dotted"
    arrowsize: ClassVar[str] = ".5"
    PRIORITY = 1


class ToFlowDiagramEdge(DiagramEdge):
    arrowsize: ClassVar[str] = ".5"
    PRIORITY = 2

    def add_to_graphviz(self, g: Digraph = None) -> None:
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
    default_color: ClassVar[dict[str, str]] = {"light": "grey", "dark": "grey"}
    style: ClassVar[str] = "dotted"
    arrowsize: ClassVar[str] = ".5"
    PRIORITY = 1
