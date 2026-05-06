# make it so we don't have to quote every type annotation ever
from __future__ import annotations

from graphviz import Digraph

import reno


class ModelDiagram:
    def __init__(
        self,
        model: reno.Model,
        bg: str = None,
        _model_map: dict[reno.Model : ModelDiagram] = None,
    ):
        self.model = model
        self.submodels: list[ModelDiagram] = []
        self.digraph: Digraph = None

        self.bg = bg
        """background color."""

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
            model_diagram = ModelDiagram(submodel, _model_map=self.model_map)
            self.submodels.append(model_diagram)

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

    def to_graphviz(self, g: Digraph = None) -> Digraph:
        """Generate the graphviz Digraph and return it.

        Output is also stored on self.digraph.
        """
        pass

    def get_ref_node(self, ref: reno.Reference) -> DiagramNode:
        diagram = self.model_map[ref.model]
        for node in diagram.nodes:
            if node.ref == ref:
                return node
        return None


class DiagramNode:
    def __init__(
        self, ref: reno.Reference, diagram: ModelDiagram, sparkline: bool = False
    ):
        self.diagram = diagram
        self.ref = ref
        self.sparkline = sparkline

        # TODO: possibly make a dictionary to include edge _type_
        self.edges: list[DiagramEdge] = []

    def add_to_graphviz(self, g: Digraph = None) -> None:
        pass

    @classmethod
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
    def map_edges(self) -> None:
        for flow in self.ref.in_flows:
            DiagramNode.add_edge(StockIODiagramEdge(flow, self))
        for flow in self.ref.out_flows:
            DiagramNode.add_edge(StockIODiagramEdge(self, flow))
        for ref in self.ref.min_refs() + self.ref.max_refs():
            # TODO: check for timeref here too?
            DiagramNode.add_edge(StockLimitDiagramEdge(ref, self))

        # TODO: logic for implicit flows?


class FlowDiagramNode(DiagramNode):
    def map_edges(self) -> None:
        for ref, ref_types in self.ref.seek_refs(include_ref_types=True).items():
            if isinstance(ref, reno.components.TimeRef):
                continue
            # TODO: ignore stock outflows?
            if "inflow" in ref_types:
                DiagramNode.add_edge(StockIODiagramEdge(ref, self))
            else:
                DiagramNode.add_edge(FlowReferenceDiagramEdge(ref, self))


class VarDiagramNode(DiagramNode):
    def map_edges(self) -> None:
        for ref in self.ref.seek_refs():
            if isinstance(ref, reno.components.TimeRef):
                continue
            DiagramNode.add_edge(VarReferenceDiagramEdge(ref, self))


class MetricDiagramNode(DiagramNode):
    def map_edges(self) -> None:
        for ref in self.ref.seek_refs():
            if isinstance(ref, reno.components.TimeRef):
                continue
            DiagramNode.add_edge(MetricReferenceDiagramEdge(ref, self))

    def add_to_graphviz(self, g: Digraph = None) -> None:
        pass


class DiagramEdge:
    """Edge information stands outside of an individual diagram since there can be
    crossmodel/intermodel connections. Edges are stored as separate instances so that
    the nodes on both ends can share the edge objects.
    """

    def __init__(self, source: DiagramNode, target: DiagramNode):
        self.source = source
        self.target = target

        self.rendered = False

        if self not in source.edges:
            source.edges.append(self)
        if self not in target.edges:
            target.edges.append(self)

    # TODO: not sure if needed
    # TODO: should it include itself or no?
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
        pass


class StockIODiagramEdge(DiagramEdge):
    def add_to_graphviz(self, g: Digraph = None) -> None:
        pass


class StockLimitDiagramEdge(DiagramEdge):
    def add_to_graphviz(self, g: Digraph = None) -> None:
        pass


class VarReferenceDiagramEdge(DiagramEdge):
    def add_to_graphviz(self, g: Digraph = None) -> None:
        pass


class FlowReferenceDiagramEdge(DiagramEdge):
    def add_to_graphviz(self, g: Digraph = None) -> None:
        pass


class MetricReferenceDiagramEdge(DiagramEdge):
    def add_to_graphviz(self, g: Digraph = None) -> None:
        pass
