"""Interactive panel interface and relevant components to allow exploring models live."""

import argparse
import base64
import datetime
import io
import json
import os
from collections.abc import Callable
from functools import partial

import matplotlib.pyplot as plt
import panel as pn
import param
import PIL
import xarray as xr

import reno
from reno.viz import ReferenceEditor, plot_trace_refs

SESSION_FOLDER = ""


class Explorer(pn.custom.PyComponent):
    """The overall wrapper for the interactive model explorer.

    Args:
        model (reno.model.Model): The SD model to generate a UI for.
    """

    def __init__(self, model, **params):
        self.model = model
        super().__init__(**params)

        self.vars_editor = FreeVarsEditor(self.model)
        self.observables = ObservablesList(self.model)
        self.view = MainView(self.model)
        self.controls = ViewControls()
        self.runs_list = RunsList()

        # NOTE: can't get terminal to live update with progress output from pymc
        # runs, haven't sufficiently explored why.
        # self.terminal = pn.widgets.Terminal(write_to_console=True)
        # sys.stdout = self.terminal
        # sys.stderr = self.terminal

        self._layout = pn.Row(
            pn.Column(self.vars_editor, self.observables, styles=dict(height="100%")),
            self.view,
            pn.Column(self.runs_list, self.controls),
            width_policy="max",
            styles=dict(height="100%"),
        )

        # hook up event handlers between components
        self.vars_editor.on_run_prior_clicked(self.run_prior)
        self.observables.on_run_posterior_clicked(self.run_posterior)
        self.runs_list.on_selected_runs_changed(self._handle_selected_rows_changed)
        self.view.on_new_controls_needed(self._handle_requested_controls)

        self._handle_requested_controls(self.view.active_tab.controls)

    def set_running(self, running: bool):
        """Set the status of the spinney things in various subcomponents
        when things are happening."""
        if running:
            self.vars_editor.run_prior_btn.loading = True
            self.observables.run_post_btn.loading = True
        else:
            self.vars_editor.run_prior_btn.loading = False
            self.observables.run_post_btn.loading = False

    def run_prior(self):
        """Run pymc on the model for priors only."""
        self.set_running(True)
        self.vars_editor.assign_from_controls()
        trace = self.model.pymc(compute_prior_only=True, keep_config=True)
        config = self.model.config()
        self.runs_list.add_run(config=config, trace=trace.prior, observations=None)
        self.set_running(False)

    def run_posterior(self):
        """Run pymc on the model to get posteriors."""
        self.set_running(True)
        self.vars_editor.assign_from_controls()
        observations = self.observables.get_observations()
        trace = self.model.pymc(keep_config=True, observations=observations)
        config = self.model.config()
        self.runs_list.add_run(
            config=config, trace=trace.posterior, observations=observations
        )
        self.set_running(False)

    def _handle_selected_rows_changed(self, runs):
        traces = {run[0]: run[2] for run in runs}
        self.view.update_traces(traces)

    def _handle_requested_controls(self, controls_layout):
        self.controls._layout.objects = [*controls_layout]

    def to_dict(self) -> dict:
        """Serialize all info for current session so can be saved to file."""
        data = {
            "model": self.model.to_dict(),
            "runs": self.runs_list.to_dict(),
            "tabs": self.view.to_dict(),
        }
        return data

    @staticmethod
    def from_dict(data: dict) -> "Explorer":
        """Deserilalize data from previously saved session via ``to_dict()``."""
        model = reno.model.Model.from_dict(data["model"])
        explorer = Explorer(model)
        explorer.runs_list.from_dict(data["runs"])

        traces = {runrow.run_name: runrow.trace for runrow in explorer.runs_list.runs}
        print("LOADED TRACES")
        print(traces)

        explorer.view.from_dict(data["tabs"], traces)
        return explorer

    def __panel__(self):
        return self._layout


class FreeVarsEditor(pn.viewable.Viewer):
    """Editor for distributions/values of system free variables."""

    def __init__(self, model, **params):
        self.model = model
        super().__init__(**params)

        self.controls = []
        self.reference_editors = []
        self.create_variable_controls()

        self.steps = pn.widgets.IntInput(name="steps", value=self.model.steps, width=80)
        self.n = pn.widgets.IntInput(name="n", value=self.model.n, width=80)

        self.run_prior_btn = pn.widgets.Button(name="Run prior", button_type="primary")
        self.run_prior_btn.on_click(self._handle_pnl_run_prior_btn_clicked)

        self._layout = pn.Column(
            pn.pane.HTML("<b>System Free Variables</b>"),
            pn.Column(*self.controls, scroll=True, sizing_mode="stretch_height"),
            pn.Row(self.n, self.steps),
            self.run_prior_btn,
            styles=dict(height="60%", flex_grow="2"),
        )

        self._run_prior_clicked_callbacks: list[Callable] = []

    def on_run_prior_clicked(self, callback: Callable):
        """Register a function to execute when the 'run priors' button is clicked.

        Callbacks for this event should take no parameters.
        """
        self._run_prior_clicked_callbacks.append(callback)

    def fire_on_run_prior_clicked(self):
        """Trigger the callbacks for the run_prior_clicked event."""
        for callback in self._run_prior_clicked_callbacks:
            callback()

    def _handle_pnl_run_prior_btn_clicked(self, *args):
        self.fire_on_run_prior_clicked()

    def create_variable_controls(self):
        """Set up a bunch of ReferenceEditors for the free variables."""
        for ref_name in self.model.free_refs(recursive=True):
            editor = ReferenceEditor(
                self.model, ref_name, self.model._is_init_ref(ref_name)
            )
            control = pn.widgets.TextInput(name=ref_name, value=editor.get_eq_str())
            editor.control = control

            ref = getattr(self.model, ref_name)
            if ref is not None and ref.doc is not None:
                control.description = ref.doc

            self.controls.append(control)
            self.reference_editors.append(editor)

    def assign_from_controls(self):
        """Set references and configuration on the model based on values set in UI."""
        for ref_editor in self.reference_editors:
            ref_editor.assign_value_from_control()
        self.model.n = self.n.value
        self.model.steps = self.steps.value

    def __panel__(self):
        return self._layout


class Observable(pn.viewable.Viewer):
    """Components row for setting up an observation to include for finding posteriors.
    Observations have to be set on metrics, so only metrics will be populated in dropdowns.
    """

    def __init__(self, model, **params):
        self.model = model
        super().__init__(**params)

        # TODO: need way to specify additional optional operation(s) to apply

        options = {metric.name: metric for metric in self.model.all_metrics()}

        self.reference = pn.widgets.AutocompleteInput(
            name="Metric",
            options=options,
            search_strategy="includes",
            min_characters=0,
            sizing_mode="stretch_width",
        )
        self.sigma = pn.widgets.FloatInput(
            name="Sigma(σ)",
            value=1.0,
            description="Uncertainty/tolerance around data value",
            sizing_mode="stretch_width",
        )
        self.data = pn.widgets.FloatInput(
            name="Observed value", value=10.0, sizing_mode="stretch_width"
        )

        self._layout = pn.Column(
            self.reference,
            pn.Row(self.data, self.sigma, sizing_mode="stretch_width"),
            styles=dict(
                background_color="#A0522D40",
                padding_bottom="10px",
                margin_bottom="3px",
                overflow="unset",
            ),
            scroll=False,
            width=320,
        )

    def __panel__(self):
        return self._layout


class ObservablesList(pn.viewable.Viewer):
    """Container list to add/modify/remove observations."""

    def __init__(self, model, **params):
        self.model = model
        super().__init__(**params)

        self.rows: list[Observable] = []

        self.run_post_btn = pn.widgets.Button(
            name="Run posterior", button_type="primary"
        )
        self.run_post_btn.on_click(self._handle_pnl_run_post_btn_clicked)

        self.new_obs_btn = pn.widgets.Button(name="Add observation")
        self.new_obs_btn.on_click(self._handle_pnl_new_obs_btn_clicked)

        self._layout = pn.Column(
            pn.pane.HTML("Observables!"), styles=dict(height="calc(30% - 10px)")
        )

        self._run_posterior_clicked_callbacks: list[Callable] = []

        self._refresh_layout()

    def on_run_posterior_clicked(self, callback: Callable):
        """Register a function to execute when the 'run posteriors' button is clicked.

        Callbacks for this event should take no parameters.
        """
        self._run_posterior_clicked_callbacks.append(callback)

    def fire_on_run_posterior_clicked(self):
        """Trigger the callbacks for the run_posterior_clicked event."""
        for callback in self._run_posterior_clicked_callbacks:
            callback()

    def _handle_pnl_run_post_btn_clicked(self, *args):
        self.fire_on_run_posterior_clicked()

    def _handle_pnl_new_obs_btn_clicked(self, *args):
        self.add_observation()

    def _refresh_layout(self):
        self._layout.objects = [
            pn.pane.HTML("<b>Observations</b>"),
            pn.Column(*self.rows, scroll=True, sizing_mode="stretch_height"),
            self.new_obs_btn,
            self.run_post_btn,
        ]

    def add_observation(self):
        """Create a new observation row/set of fields for setting an Observable."""
        self.rows.append(Observable(self.model))
        self._refresh_layout()

    def get_observations(self) -> list["reno.ops.Observation"]:
        """Convert all the fields from the list of Observable components
        to create reno Observations ops, ultimately to pass into the
        ``model.pymc`` call"""
        observations = []
        for row in self.rows:
            obs = reno.ops.Observation(
                row.reference.value, row.sigma.value, [row.data.value]
            )
            observations.append(obs)
        return observations

    def __panel__(self):
        return self._layout


class ClickablePane(pn.custom.JSComponent):
    """Wrap any panel component with a click event handler, and
    optionally a drag to pan handler (important for zoomed SFD diagrams.)"""

    child = pn.custom.Child()
    """The panel component being wrapped. Note that since this outer component
    is handling clicks, strange things can occur if the sub component is also
    meant to take clicks. (Disable ``click_enabled`` if it's predictable when
    the sub component should handle instead.)"""

    click_enabled = param.Boolean(True)
    """Whether to listen for clicks on the wrapper or not, disable if click
    handling is temporarily needed in the ``child`` component."""

    drag_scroll_enabled = param.Boolean(False)
    """Whether to enable scrolling the wrapped component by clicking and dragging."""

    _stylesheets = [
        """
        :root {
            overflow: hidden;
        }
        """
    ]

    _esm = """
        export function render({ model, el }) {
            const element = document.createElement("div");
            element.style["width"] = "100%";
            element.style["height"] = "100%";
            element.style["position"] = "absolute";
            element.style["top"] = "0";
            element.style["left"] = "0";
            element.append(model.get_child("child"))
            element.addEventListener("click", (e) => { model.send_event('js_clicked', e) });

            // -- Dragging event listening/handling --
            // https://stackoverflow.com/questions/28576636/mouse-click-and-drag-instead-of-horizontal-scroll-bar-to-view-full-content-of-c
            let is_dragging = false;
            let startX = 0;
            let startY = 0;
            let dragStartX = 0;
            let dragStartY = 0;
            element.addEventListener("mousedown", (e) => {
                if (model.drag_scroll_enabled) {
                    dragStartX = e.pageX;
                    dragStartY = e.pageY;
                    is_dragging = true;
                }
            });
            element.addEventListener("mousemove", (e) => {
                if (model.drag_scroll_enabled) {
                    if (is_dragging) {
                        e.preventDefault();
                        let dragCurrentX = e.pageX;
                        let dragCurrentY = e.pageY;

                        let newX = startX - (dragStartX - dragCurrentX);
                        let newY = startY - (dragStartY - dragCurrentY);

                        element.style["top"] = newY + "px"
                        element.style["left"] = newX + "px"
                    }
                }
            });
            element.addEventListener("mouseup", (e) => {
                if (model.drag_scroll_enabled) {
                    if (is_dragging) {
                        is_dragging = false;
                        let dragCurrentX = e.pageX;
                        let dragCurrentY = e.pageY;

                        let newX = startX - (dragStartX - dragCurrentX);
                        let newY = startY - (dragStartY - dragCurrentY);

                        startX = newX;
                        startY = newY;
                    }
                }
            });

            // Reset panning whenever a custom message is sent.
            // For now, the only custom message is "reset" (see reset_pan)
            // so there's no explicit data check yet.
            model.on("msg:custom", (e) => {
                element.style["top"] = 0;
                element.style["left"] = 0;
                startX = 0;
                startY = 0;
            });

            return element;
        }
    """

    def __init__(self, **params):
        super().__init__(**params)

        self._clicked_callbacks: list[Callable] = []

    def reset_pan(self):
        """Reset component panning to top left of 0, 0."""
        self._send_event(pn.models.esm.ESMEvent, data="reset")

    def on_click(self, callback):
        """Register a function to execute whenever a click on the wrapper is
        detected."""
        self._clicked_callbacks.append(callback)

    def fire_on_click(self):
        """Trigger the callbacks for the click event."""
        for callback in self._clicked_callbacks:
            callback()

    def _handle_js_clicked(self, event):
        if self.click_enabled:
            self.fire_on_click()


class PanesSet(pn.viewable.Viewer):
    """Container for the modifiable GridStack of model exploration widgets, with
    controls for configuring."""

    tab_name = param.String("Tab 1")

    def __init__(self, model, **params):
        super().__init__(**params)
        self.model = model

        # There's a weird bug where sometimes if you enable/disable a couple
        # times, by default the resize handles don't re-show, preventing any
        # further resizing. (Specifically seems like the autohide class gets
        # stuck?)
        gs_fix_disappearing_resize = """
            .grid-stack > .grid-stack-item.ui-resizable-autohide > .ui-resizable-handle {
                display: block !important;
            }
            .grid-stack-item.ui-resizable-disabled > .grid-stack-item-content {
                outline: none;
            }
            .grid-stack-item > .grid-stack-item-content {
                outline: 1px solid #DD6655;
                outline-offset: -1px;
            }
        """

        self.cells_height = 2
        self.cells_width = 4

        self.panes = []
        self.active_traces = {}

        self.btn_export = pn.widgets.Button(name="Export tab")
        self.btn_export.on_click(self._handle_pnl_export_clicked)

        self.downloads = pn.Row()
        self.controls = pn.Column(
            pn.Param(
                self,
                name="Tab controls",
                parameters=[
                    "tab_name",
                ],
            ),
            pn.Row(self.btn_export, self.downloads),
        )

        self.gstack = pn.GridStack(
            sizing_mode="stretch_width",
            mode="override",
            allow_resize=False,
            allow_drag=False,
            height=400,
            stylesheets=[gs_fix_disappearing_resize],
            nrows=4,
            ncols=4,
        )
        self._layout = self.gstack

        self._on_new_controls_needed_callbacks: list[Callable] = []
        self._on_name_changed_callbacks: list[Callable] = []

    def add_pane(self, pane_to_add):
        """Add the passed model explorer widget and modify the gstack size. This allows
        a constantly growing interface, though this is currently a bit simplistic and
        doesn't allow manually setting or reducing yet."""
        # each "row" gets a height of 400px, at some point we should make this
        # configurable.
        self.gstack.height = 400 * (self.cells_height + 1)
        self.gstack.nrows = (self.cells_height + 1) * 4
        self.gstack[
            (self.cells_height * 4) : (self.cells_height + 1) * 4, 0 : self.cells_width
        ] = pane_to_add
        self.cells_height += 1
        self.panes.append(pane_to_add)
        pane_to_add.on_cache_invalidated(self.invalidate_downloads)
        self.invalidate_downloads()

    def add_text_pane(self):
        """Create a new editable text widget and add it to the tab interface."""
        text_pane = EditableTextPane()
        text_pane.clicker.on_click(
            partial(self.fire_on_new_controls_needed, text_pane.controls)
        )
        self.add_pane(text_pane)
        self.fire_on_new_controls_needed(text_pane.controls)

    def add_plots_pane(self):
        """Create a new plots widget and add it to the tab interface."""
        plots_pane = PlotsPane(self.model)
        plots_pane.clicker.on_click(
            partial(self.fire_on_new_controls_needed, plots_pane.controls)
        )
        self.add_pane(plots_pane)
        plots_pane.render(self.active_traces)
        self.fire_on_new_controls_needed(plots_pane.controls)

    def add_diagram_pane(self):
        """Create a new stock and flow diagram widget and add it to the tab
        interface."""
        diagram_pane = DiagramPane(self.model)
        diagram_pane.clicker.on_click(
            partial(self.fire_on_new_controls_needed, diagram_pane.controls)
        )
        self.add_pane(diagram_pane)
        self.fire_on_new_controls_needed(diagram_pane.controls)
        diagram_pane.render(self.active_traces)

    def on_new_controls_needed(self, callback: Callable):
        """Register a function to execute whenever a widget within the tab requests
        a new set of helper controls be displayed in the sidebar.

        Callbacks should take a panel widget.
        """
        self._on_new_controls_needed_callbacks.append(callback)

    def fire_on_new_controls_needed(self, controls_layout):
        """Trigger the callbacks for the new_controls_needed event."""
        for callback in self._on_new_controls_needed_callbacks:
            callback([self.controls, controls_layout])

    def on_name_changed(self, callback: Callable):
        """Register a function to execute when the tab title is changed.

        Callbacks should take the new string name."""
        self._on_name_changed_callbacks.append(callback)

    @pn.depends("tab_name", watch=True)
    def fire_on_name_changed(self, *args):
        """Trigger the callbacks for the name_changed event."""
        for callback in self._on_name_changed_callbacks:
            callback(self.tab_name)

    def _handle_pnl_export_clicked(self, *args):
        self.export()

    def invalidate_downloads(self):
        """If something important has changed since the last time the tab was
        exported, visually highlight on all the relevant buttons."""
        self.btn_export.disabled = False
        for btn in self.downloads.objects:
            btn.button_style = "outline"
            if not btn.label.endswith("*"):
                btn.label = btn.label + "*"

    def export(self):
        """Save all necessary data about current tab in session (including a
        copy of the model and any simulation data) and use the tab_exporter
        to produce an HTML and PDF in the ``.doccache`` path.

        This then updates tab controls to include buttons for downloading these.
        """
        self.btn_export.loading = True

        # save all the things needed to reproduce the tab in tab_exporter
        # at some point might be a good idea to make cache dir configurable.
        os.makedirs(".doccache", exist_ok=True)
        self.model.save(".doccache/model.json")
        data = self.to_dict()
        with open(".doccache/panes.json", "w") as outfile:
            json.dump(data, outfile, indent=4)

        # run the tab exporter
        os.system(
            "python -m reno.tab_exporter .doccache/model.json .doccache/panes.json .doccache/out.html .doccache/out.pdf"
        )

        self.downloads.objects = [
            pn.widgets.FileDownload(
                label="PDF", button_type="light", file=".doccache/out.pdf", auto=True
            ),
            pn.widgets.FileDownload(
                label="HTML", button_type="light", file=".doccache/out.html", auto=True
            ),
        ]
        self.btn_export.disabled = True
        self.btn_export.loading = False

    def to_dict(self, include_traces: bool = True) -> dict:
        """Serialize tab and all contained widgets into a dictionary so can be saved to
        file and reproduced later."""
        print("PanesSet to_dict")
        print(self.active_traces)
        traces = {key: trace.to_dict() for key, trace in self.active_traces.items()}
        panes = []
        for loc, obj in self.gstack.objects.items():
            panes.append(
                {
                    "loc": [loc[0], loc[1], loc[2], loc[3]],
                    "type": obj.__class__.__name__,
                    "data": obj.to_dict(),
                }
            )

        data = {
            "tab_name": self.tab_name,
            "panes": panes,
            "height": self.gstack.height,
            "nrows": self.gstack.nrows,
            "cells_height": self.cells_height,
        }
        if include_traces:
            data["traces"] = traces

        return data

    def from_dict(self, data: dict, traces: dict = None):
        """Deserialize all config and widgets from data _into current instance_"""
        self.tab_name = data["tab_name"]

        if traces is None:
            for trace_name in data["traces"]:
                self.active_traces[trace_name] = xr.Dataset.from_dict(
                    data["traces"][trace_name]
                )
        else:
            self.active_traces = traces

        obj_dict = {}

        self.gstack.nrows = data["nrows"]
        self.gstack.height = data["height"]
        self.cells_height = data["cells_height"]

        for pane_data_and_pos in data["panes"]:
            loc = pane_data_and_pos["loc"]
            loc = (loc[0], loc[1], loc[2], loc[3])
            pane_type = pane_data_and_pos["type"]
            pane_data = pane_data_and_pos["data"]
            if pane_type == "PlotsPane":
                pane = PlotsPane(self.model)
                pane.render(self.active_traces)
            elif pane_type == "DiagramPane":
                pane = DiagramPane(self.model)
                pane.render(self.active_traces)
            elif pane_type == "EditableTextPane":
                pane = EditableTextPane()
            pane.from_dict(pane_data)
            pane.clicker.on_click(
                partial(self.fire_on_new_controls_needed, pane.controls)
            )

            # note that the way gridstack reports locations is a little
            # different than the recomended slice indexing mechanism specified
            # in the panel docs - but directly assigning the exact way it
            # reports still works, so we don't bother doing any sort of
            # wonky translation.
            obj_dict[loc] = pane

        self.gstack.objects = obj_dict
        self.panes = list(obj_dict.values())
        print("at end of paneset from_dict", traces)

    def __panel__(self):
        return self._layout


# have do do inheritance because of following linked bug, fix will
# supposedly be out soon as of 2025-06-25
# https://github.com/holoviz/panel/issues/7689
# class PlotsPane(pn.custom.PyComponent):
# TODO: there's a non-insignificant amount of copied code between the different
# panes (esp events), might be wise to have a parent Pane class.
class PlotsPane(
    pn.widgets.base.WidgetBase, pn.custom.PyComponent, pn.reactive.Reactive
):
    """A model exploration widget that can be displayed within a tab, customizable
    set of timeseries/density plots for various parts of the model."""

    fig_width = param.Integer(10)
    fig_height = param.Integer(6)
    columns = param.Integer(2)

    plot_type = param.Selector(
        objects=["Variables/Metrics", "Stocks", "Custom"],
        doc="Presets for subsets of model references to include in plot.",
    )
    """This selector represents a few reasonable defaults for things you might want
    to see. These alter the base set of references shown in the subset dropdown, which
    can be used to further refine which plots to show."""

    subset = param.ListSelector(
        [],
        doc="References to include in plot. If none, includes all references in the currently selected preset.",
    )

    ref_subset = param.ListSelector([])
    """Resolved list of references to show, taking into account plot_type and subset."""

    def __init__(self, model, **params):
        super().__init__(**params)
        self.rendered_traces = {}

        self.model = model

        self.controls = pn.Param(
            self,
            name="Plot controls",
            parameters=[
                "fig_width",
                "fig_height",
                "columns",
                "plot_type",
                "subset",
            ],
            widgets={
                "subset": pn.widgets.MultiChoice,
                "plot_type": pn.widgets.RadioButtonGroup,
            },
        )

        # TODO: (4/29/205) can't use both ipywidgets _and_ gridstack right now??
        # self.fig = pn.pane.Matplotlib()
        self.image = pn.pane.Image(sizing_mode="stretch_both")
        self.clicker = ClickablePane(sizing_mode="stretch_both")
        self.clicker.child = pn.Column(self.image)

        self.base64repr = None
        """base64 encoded version of the image of the plots, used for embedding
        directly into html (see ``to_html()``), set in ``render()``."""

        self.update_plot_type()

        self._layout = self.clicker

        self._on_cache_invalidated_callbacks: list[Callable] = []

    def on_cache_invalidated(self, callback: Callable):
        """Register a callback for any time a previous exported file will no longer
        match/cached file is no longer valid or current."""
        self._on_cache_invalidated_callbacks.append(callback)

    def fire_on_cache_invalidated(self):
        """Trigger all callback functions registered for the cache_invalidated event."""
        if not hasattr(self, "_on_cache_invalidated_callbacks"):
            # can happen from initial param settings before end of init?
            return
        for callback in self._on_cache_invalidated_callbacks:
            callback()

    @param.depends("plot_type", watch=True)
    def update_plot_type(self, *args):
        """Event handler for when a different preset is selected."""
        if self.plot_type == "Variables/Metrics":
            rv_names = [
                name
                for name in self.model.trace_RVs
                if not name.endswith("_likelihood")
            ]
            metric_names = [metric.qual_name() for metric in self.model.all_metrics()]
            self.param.subset.objects = rv_names + metric_names
            self.subset = []
            self.update_subset()

        elif self.plot_type == "Stocks":
            stock_names = [stock.qual_name() for stock in self.model.all_stocks()]
            self.param.subset.objects = stock_names
            self.subset = []
            self.update_subset()

        elif self.plot_type == "Custom":
            self.param.subset.objects = [
                ref.qual_name() for ref in self.model.all_refs()
            ] + [metric.qual_name() for metric in self.model.all_metrics()]
            self.subset = []
            self.update_subset()

    @param.depends("subset", watch=True)
    def update_subset(self, *args):
        """Whenever the 'preset subset' option changes, update the actual subset of
        references being used."""
        if self.subset == []:
            self.ref_subset = self.param.subset.objects
        else:
            self.ref_subset = self.subset

    @param.depends("fig_height", "fig_width", "columns", "ref_subset", watch=True)
    def render(self, traces=None):
        """Update the visible components of this widget. We save a base64 representation
        of the plot so it can be used in the ``to_html()`` call."""
        if traces is not None:
            self.rendered_traces = traces

        figure = plot_trace_refs(
            self.model,
            traces=self.rendered_traces,
            ref_list=self.ref_subset,
            figsize=(self.fig_width, self.fig_height),
            cols=self.columns,
        )
        # self.fig.object = figure
        # https://stackoverflow.com/questions/57316491/how-to-convert-matplotlib-figure-to-pil-image-object-without-saving-image/61754995
        # img = PIL.Image.frombytes('RGB', figure.canvas.get_width_height(), figure.canvas.tostring_rgb())
        buf = io.BytesIO()
        figure.savefig(buf)
        buf.seek(0)
        self.base64repr = base64.b64encode(buf.getvalue())
        img = PIL.Image.open(buf)
        self.image.object = img
        plt.close(figure)

        self.fire_on_cache_invalidated()

    def to_html(self) -> str:
        """Get an HTML-compatible string for the contents of this pane. This is
        used for generating exported standalone reports, see tab_exporter.

        For this pane, this works by creating a base64 representation of the plot
        image and embedding that directly in the string output."""
        return f"""
        <div class='pane-plots'>
            <img src="data:image/png;base64, {self.base64repr.decode('utf-8')}" />
        </div>
        """

    def to_dict(self) -> dict:
        """Serialize this pane to a dictionary that can be saved to file."""
        return {
            "fig_width": self.fig_width,
            "fig_height": self.fig_height,
            "columns": self.columns,
            "plot_type": self.plot_type,
            "subset": self.subset,
            "ref_subset": self.ref_subset,
        }

    def from_dict(self, data: dict):
        """Deserialize data into current instance from dictionary previously stored
        from ``to_dict()``."""
        self.fig_width = data["fig_width"]
        self.fig_height = data["fig_height"]
        self.columns = data["columns"]
        self.plot_type = data["plot_type"]
        self.subset = data["subset"]
        self.ref_subset = data["ref_subset"]

    def __panel__(self):
        return self._layout


# see comment above PlotsPane
# https://github.com/holoviz/panel/issues/7689
# class DiagramPane(pn.custom.PyComponent):
class DiagramPane(
    pn.widgets.base.WidgetBase, pn.custom.PyComponent, pn.reactive.Reactive
):
    """A model exploration widget that can be displayed within a tab, a
    stock and flow diagram to visually display how model equations/components
    are connected."""

    show_vars = param.Boolean(True, doc="Include variables in the diagram")
    sparklines = param.Boolean(True, doc="Show timeseries plots next to each stock")
    sparkdensities = param.Boolean(
        False, doc="Show density plots next to each variable"
    )

    universe = param.ListSelector(
        [], label="subset", doc="Only show the specified references in the diagram."
    )
    include_dependencies = param.Boolean(
        False,
        label="Include dependencies in universe",
        doc="Include all immediate dependencies of the selected universe references.",
    )

    fit = param.Boolean(True, doc="Scale image to pane size.")

    def __init__(self, model, **params):
        super().__init__(**params)

        self.rendered_traces = None

        self.model = model
        self.param.universe.objects = {ref.name: ref for ref in self.model.all_refs()}

        self.base64repr = None

        # diagram is in a ClickablePane with drag scroll enabled, so inhibit
        # default browser behavior when clicking and dragging an image.
        remove_img_drag = """
        div {
            cursor: grab;
        }
        img {
            -webkit-user-select: none;
            -khtml-user-select: none;
            -moz-user-select: none;
            -o-user-select: none;
            user-select: none;
            -webkit-user-drag: none;
            -khtml-user-drag: none;
            -moz-user-drag: none;
            -o-user-drag: none;
            user-drag: none;
            pointer-events: none;
        }
        """

        self.controls = pn.Param(
            self,
            name="Diagram controls",
            parameters=[
                "show_vars",
                "sparklines",
                "sparkdensities",
                "universe",
                "include_dependencies",
                "fit",
            ],
            widgets={"universe": {"type": pn.widgets.MultiChoice, "stylesheets": []}},
        )

        self.image = pn.pane.Image(
            stylesheets=[remove_img_drag], sizing_mode="stretch_both"
        )
        self.clicker = ClickablePane(
            sizing_mode="stretch_both", styles=dict(overflow="hidden")
        )
        self.clicker.child = self.image
        self.clicker.drag_scroll_enabled = False

        self._layout = self.clicker

        self._on_cache_invalidated_callbacks: list[Callable] = []

    def on_cache_invalidated(self, callback: Callable):
        """Register a callback for any time a previous exported file will no longer
        match/cached file is no longer valid or current."""
        self._on_cache_invalidated_callbacks.append(callback)

    def fire_on_cache_invalidated(self):
        """Trigger all callback functions registered for the cache_invalidated event."""
        for callback in self._on_cache_invalidated_callbacks:
            callback()

    @param.depends("fit", watch=True)
    def _update_fit(self, *args):
        if self.fit:
            self.image.sizing_mode = "stretch_both"
            self.clicker.reset_pan()
            self.clicker.drag_scroll_enabled = False
        else:
            self.image.sizing_mode = "fixed"
            self.clicker.drag_scroll_enabled = True

    @param.depends(
        "show_vars",
        "sparklines",
        "sparkdensities",
        "universe",
        "include_dependencies",
        watch=True,
    )
    def render(self, traces=None):
        """Update the visible components of this widget. We save a base64 representation
        of the diagram so it can be used in the ``to_html()`` call."""
        if traces is not None:
            self.rendered_traces = traces

        universe = self.universe
        if len(universe) == 0:
            universe = None
        if universe is not None and self.include_dependencies:
            universe = reno.utils.ref_universe(universe)

        image_bytes = self.model.graph(
            show_vars=self.show_vars,
            sparklines=self.sparklines,
            sparkdensities=self.sparkdensities,
            traces=self.rendered_traces,
            universe=universe,
        ).pipe(format="png")

        self.base64repr = base64.b64encode(image_bytes)
        self.image.object = image_bytes
        self.fire_on_cache_invalidated()

    def to_html(self) -> str:
        """Get an HTML-compatible string for the contents of this pane. This is
        used for generating exported standalone reports, see tab_exporter.

        For this pane, this works by creating a base64 representation of the diagram
        and embedding that directly in the string output."""
        return f"""
        <div class='pane-diagram'>
            <img src="data:image/png;base64, {self.base64repr.decode('utf-8')}" />
        </div>
        """

    def to_dict(self) -> dict:
        """Serialize this pane to a dictionary that can be saved to file."""
        return {
            "show_vars": self.show_vars,
            "sparklines": self.sparklines,
            "sparkdensities": self.sparkdensities,
            "universe": self.universe,
            "include_dependencies": self.include_dependencies,
            "fit": self.fit,
        }

    def from_dict(self, data: dict):
        """Deserialize data into current instance from dictionary previously stored
        from ``to_dict()``."""
        self.show_vars = data["show_vars"]
        self.sparklines = data["sparklines"]
        self.sparkdensities = data["sparkdensities"]
        self.universe = data["universe"]
        self.include_dependencies = data["include_dependencies"]
        self.fit = data["fit"]

    def __panel__(self):
        return self._layout


# see comment above PlotsPane
# class EditableTextPane(pn.custom.PyComponent):
# https://github.com/holoviz/panel/issues/7689
class EditableTextPane(
    pn.widgets.base.WidgetBase, pn.custom.PyComponent, pn.reactive.Reactive
):
    """A model exploration widget that can be displayed within a tab, a user-editable
    text field meant for including surrounding descriptions or "storyboarding" in a
    model exploration/analysis."""

    def __init__(self, **params):
        self.editor = pn.widgets.TextEditor()
        super().__init__(**params)

        self.text = "(text field, click to edit in controls sidebar)"

        # make text color work regardless of dark/light theme
        text_color_css = """
            div {
                color: var(--background-text-color);
            }
            a {
                color: #F4A460;
            }
        """

        self.view = pn.pane.HTML(self.text, stylesheets=[text_color_css])
        self.clicker = ClickablePane(sizing_mode="stretch_both")
        self.clicker.child = self.view

        self.controls = pn.Column(self.editor)

        self._layout = self.clicker
        self._on_cache_invalidated_callbacks: list[Callable] = []

    def on_cache_invalidated(self, callback: Callable):
        """Register a callback for any time a previous exported file will no longer
        match/cached file is no longer valid or current."""
        self._on_cache_invalidated_callbacks.append(callback)

    def fire_on_cache_invalidated(self):
        """Trigger all callback functions registered for the cache_invalidated event."""
        for callback in self._on_cache_invalidated_callbacks:
            callback()

    @pn.depends("editor.value", watch=True)
    def _update_text(self, *args):
        self.text = self.editor.value
        self.view.object = self.text
        self.fire_on_cache_invalidated()

    def to_html(self) -> str:
        """Get an HTML-compatible string for the contents of this pane. This is
        used for generating exported standalone reports, see tab_exporter."""
        return f"""
        <div class='pane-text'>
            {self.text}
        </div>
        """

    def to_dict(self) -> dict:
        """Serialize this pane to a dictionary that can be saved to file."""
        return {"text": self.text}

    def from_dict(self, data: dict):
        """Deserialize data from passed dictionary to populate this widget."""
        self.editor.value = data["text"]

    def __panel__(self):
        return self._layout


class MainView(pn.viewable.Viewer):
    """The exploration tab container, central view of graphs/plots etc.,
    between the two sidebars."""

    def __init__(self, model, **params):
        self.editing_layout = pn.widgets.Toggle(
            name="Edit layout", value=False, button_type="light", button_style="outline"
        )
        # tabs def is up here because we need to be able to listen for active change
        # (requires tabs to be created before super init)
        self.tabs = pn.Tabs(
            sizing_mode="stretch_both",
            styles=dict(
                min_height="100%",
                height="100%",
                border_bottom="1px solid var(--neutral-fill-rest)",
            ),
        )
        super().__init__(**params)

        self.model = model

        self.btn_add_text = pn.widgets.Button(name="Add text")
        self.btn_add_diagram = pn.widgets.Button(name="Add diagram")
        self.btn_add_plots = pn.widgets.Button(name="Add plots")

        self.btn_add_text.on_click(self._handle_pnl_add_text_clicked)
        self.btn_add_diagram.on_click(self._handle_pnl_add_diagram_clicked)
        self.btn_add_plots.on_click(self._handle_pnl_add_plots_clicked)

        self.controls = pn.Row(
            self.btn_add_diagram,
            self.btn_add_text,
            self.btn_add_plots,
            pn.Spacer(sizing_mode="stretch_width"),
            self.editing_layout,
            sizing_mode="stretch_width",
        )

        initial_tab = self.create_tab()
        self.active_tab = initial_tab

        self.tab_contents = [(initial_tab.tab_name, initial_tab), ("+", None)]

        self.refresh_tab_contents()

        self._layout = pn.Column(
            self.tabs,
            self.controls,
            styles=dict(height="calc(100% - 50px)"),
        )

        self._on_new_controls_needed_callbacks: list[Callable] = []

    def on_new_controls_needed(self, callback: Callable):
        """Register a function to execute whenever a widget within the tab requests
        a new set of helper controls be displayed in the sidebar.

        Callbacks should take a panel widget.
        """
        self._on_new_controls_needed_callbacks.append(callback)

    def fire_on_new_controls_needed(self, controls_layout):
        """Trigger the callbacks for the new_controls_needed event."""
        for callback in self._on_new_controls_needed_callbacks:
            callback(controls_layout)

    def create_tab(self):
        """Make a new tab/gridstack contents and hook up all relevant event handlers for it."""
        new_tab = PanesSet(self.model)
        new_tab.on_new_controls_needed(self.fire_on_new_controls_needed)
        new_tab.on_name_changed(partial(self._handle_tab_name_changed, tab_obj=new_tab))
        return new_tab

    def _wrap_tab_obj(self, tab_obj, title: str):
        """The inner contents of a tab (the gridstack) needs to be scrollable,
        couldn't get this to work right applying directly to the gridstack object itself.
        """
        return pn.Column(
            tab_obj,
            name=title,
            styles=dict(overflow_y="scroll"),
            sizing_mode="stretch_both",
        )

    def _handle_tab_name_changed(self, new_name, tab_obj):
        index = -1
        for i, tab in enumerate(self.tab_contents):
            if tab[1] == tab_obj:
                index = i
                break
        if index == -1:
            # could happen when calling from_dict on new tab
            # that hasn't been added to frontend yet
            return

        self.tab_contents[index] = (new_name, tab_obj)
        self.refresh_tab_contents()

    @pn.depends("tabs.active", watch=True)
    def _handle_pnl_tab_switched(self, *args):
        # clicking on the last tab is the "+", so add new tab
        if self.tabs.active == len(self.tabs) - 1:
            new_tab = self.create_tab()
            self.tab_contents.insert(len(self.tabs) - 1, (new_tab.tab_name, new_tab))
            self.refresh_tab_contents()

        self.active_tab = self.tab_contents[self.tabs.active][1]
        self.fire_on_new_controls_needed(
            self.tab_contents[self.tabs.active][1].controls
        )

    def refresh_tab_contents(self):
        """Refresh tabs and panels inside of them/re-send to frontend."""
        self.tabs[:] = [self._wrap_tab_obj(tab[1], tab[0]) for tab in self.tab_contents]

    def _handle_pnl_add_text_clicked(self, *args):
        self.active_tab.add_text_pane()

    def _handle_pnl_add_diagram_clicked(self, *args):
        self.active_tab.add_diagram_pane()

    def _handle_pnl_add_plots_clicked(self, *args):
        self.active_tab.add_plots_pane()

    def update_traces(self, traces):
        """Change the traces being used in the current tab with those passed in."""
        self.active_tab.active_traces = traces
        for pane in self.active_tab.panes:
            if isinstance(pane, (DiagramPane, PlotsPane)):
                pane.render(traces)

    @pn.depends("editing_layout.value", watch=True)
    def _handle_edit_layout_changed(self, *args):
        if self.editing_layout.value:
            self.active_tab.gstack.allow_drag = True
            self.active_tab.gstack.allow_resize = True
        else:
            self.active_tab.gstack.allow_drag = False
            self.active_tab.gstack.allow_resize = False

    def to_dict(self) -> dict:
        """Serialize every tab to a dictionary that can be saved to file."""
        data = {"tabs": {}}
        for tab_name, tab_obj in self.tab_contents:
            if tab_obj is None:
                continue
            data["tabs"][tab_name] = tab_obj.to_dict(include_traces=False)

        return data

    def from_dict(self, data: dict, traces: dict):
        """Deserialize all tabs and simulation runs from passed data and insert them
        into this instance."""
        # clear existing tabs
        self.tab_contents = []

        for tab_name, tab_data in data["tabs"].items():
            tab = self.create_tab()
            tab.from_dict(tab_data, traces)
            self.tab_contents.append((tab.tab_name, tab))

        self.tab_contents.append(("+", None))
        self.refresh_tab_contents()
        self.tabs.active = 0
        self._handle_pnl_tab_switched()

    def __panel__(self):
        return self._layout


class ViewControls(pn.viewable.Viewer):
    """Any settings and config for the current main view, this shows up in
    the right sidebar and is populated when widgets in a tab are clicked/
    new control widgets are requested. (see new_controls_needed event
    scattered throughout other components)"""

    def __init__(self, **params):
        super().__init__(**params)
        self._layout = pn.Column(pn.pane.HTML("Controls!"))

    def __panel__(self):
        return self._layout


class RunRow(pn.viewable.Viewer):
    """Selector row for a specific simulation run, allowing deletion, inclusion/
    exclusion from visualizations in current tab, etc."""

    visible = param.Boolean(True)
    run_name = param.String("")

    def __init__(self, trace, config, observations, **params):
        # self.config = config
        # TODO: figure out what was making this crash before
        self.config = None
        self.trace = trace
        self.observations = observations
        super().__init__(**params)

        self.select_btn = pn.widgets.Button(name="s", button_type="success")
        self.remove_btn = pn.widgets.Button(name="x", button_type="danger")
        self.edit_btn = pn.widgets.Button(name="e")

        self.select_btn.on_click(self._handle_pnl_select_btn_clicked)
        self.remove_btn.on_click(self._handle_pnl_remove_btn_clicked)

        self.label = pn.pane.HTML(f"<p>{self.run_name}</p>")

        self._layout = pn.Row(
            self.select_btn,
            self.edit_btn,
            self.label,
            self.remove_btn,
        )

        self._selected_callbacks: list[Callable] = []
        self._removed_callbacks: list[Callable] = []

    def on_selected(self, callback: callable):
        """Register a function for when a run is selected or deselected.

        Callbacks should take a single boolean, ``True`` if it's selected.
        """
        self._selected_callbacks.append(callback)

    def fire_on_selected(self, selected: bool):
        """Trigger all callback functions registered for selected event."""
        for callback in self._selected_callbacks:
            callback(selected)

    def on_removed(self, callback: callable):
        """Register a function for when a simulation run is removed.

        Callbacks should take no parameters.
        """
        self._removed_callbacks.append(callback)

    def fire_on_removed(self):
        """Trigger all callback functions registered for removed event."""
        for callback in self._removed_callbacks:
            callback(self)

    def _handle_pnl_select_btn_clicked(self, *args):
        self.visible = not self.visible
        self.fire_on_selected(self.visible)

    @param.depends("run_name", watch=True)
    def _handle_run_name_changed(self, *args):
        self.label.object = f"<p>{self.run_name}</p>"

    def _handle_pnl_remove_btn_clicked(self, *args):
        self.fire_on_removed()

    @param.depends("visible", watch=True)
    def _update_selected_btn(self):
        if self.visible:
            self.select_btn.button_type = "success"
        else:
            self.select_btn.button_type = "default"

    def to_dict(self) -> dict:
        """Serialize run row to a dictionary that can be saved to file. Note that
        currently this can get quite large as the raw trace dictionary is dumped as well.
        """
        # TODO: figure out a more efficient way of separately saving the trace
        # in something like a pickle
        return {
            "run_name": self.run_name,
            "trace": self.trace.to_dict(),
            "config": self.config,
            # TODO: observations
        }

    def from_dict(self, data: dict):
        """Deserialize a run into the current instance from the passed data."""
        print("Loading run/trace ", data["run_name"])
        self.run_name = data["run_name"]
        self.trace = xr.Dataset.from_dict(data["trace"])
        print("loaded", self.trace)
        self.config = data["config"]
        # TODO: observations

    def __panel__(self):
        return self._layout


class RunsList(pn.viewable.Viewer):
    """Collection of RunRows, tracks and allows choosing which previous runs to include in
    main view for current tab."""

    # TODO: the goal is to make selection apply per tab, but this isn't actually
    # implemented yet.

    def __init__(self, **params):
        super().__init__(**params)
        self.runs = []
        self._layout = pn.Column()

        self._selected_runs_changed_callbacks: list[Callable] = []

        self.refresh_rows()

    def on_selected_runs_changed(self, callback: Callable):
        """Register a function to execute when the set of simulation runs selected to
        display is changed.

        Callbacks should take a list of tuples where each tuple contains:
        * the string name of the run
        * the dictionary with the run config
        * an xarray dataset with the full trace/simulation data.
        """
        self._selected_runs_changed_callbacks.append(callback)

    def fire_on_selected_runs_changed(self, runs: list[tuple[str, dict, xr.Dataset]]):
        """Trigger all registered callbacks for the selected_runs_changed event."""
        for callback in self._selected_runs_changed_callbacks:
            callback(runs)

    def _handle_row_changed(self, *args):
        self.fire_on_selected_runs_changed(self.get_selected_runs())

    def _handle_row_deleted(self, row_instance):
        self.runs.remove(row_instance)
        self.refresh_rows()
        self._handle_row_changed()

    def get_selected_runs(self) -> list[tuple[str, dict, xr.Dataset]]:
        """Collect all runrows that are set to display, returns a tuple with
        the name of the run, dictionary config for it, and the xarray dataset
        with the simulation data."""
        selected_runs = []
        for run in self.runs:
            if run.visible:
                selected_runs.append((run.run_name, run.config, run.trace))
        return selected_runs

    def add_run(self, config, trace: xr.Dataset, observations):
        """Create a new RunRow with the passed configuration and data."""
        run = RunRow(
            run_name=f"Run {len(self.runs)}",
            observations=observations,
            trace=trace,
            config=config,
        )
        run.on_selected(self._handle_row_changed)
        run.on_removed(self._handle_row_deleted)
        self.runs.append(run)
        self.refresh_rows()
        self._handle_row_changed()

    def refresh_rows(self):
        """Update the layout to show all runrows."""
        self._layout.objects = [pn.pane.HTML("<b>Model runs</b>"), *self.runs]

    def to_dict(self) -> dict:
        """Serialize all runs into a dictionary that can be saved to a file."""
        data = {"runs": []}
        for run in self.runs:
            data["runs"].append(run.to_dict())

        return data

    def from_dict(self, data: dict):
        """Deserialize into this instance every run found in the passed data dictionary."""
        for run in data["runs"]:
            self.add_run(None, None, None)
            runrow = self.runs[-1]
            runrow.from_dict(run)

        self.fire_on_selected_runs_changed(self.get_selected_runs())

    def __panel__(self):
        return self._layout


class BetterAccordion(pn.custom.JSComponent):
    """Simple collapsible accordion, for use in left meta sidebar 'file explorer'"""

    # Made this because panel's accordion has very limited styling capabilities.
    child = pn.custom.Child()
    label = param.String()

    _stylesheets = [
        """
        :host {
            font-family: var(--body-font);
            padding-left: 5px;
            font-weight: bold;
            color: var(--panel-primary-color) !important;
        }

        input {
            display: none;
        }
        label {
            display: block;
            user-select: none;
        }
        .content {
            margin-left: 30px;
        }
        input + label + .content {
            display: none;
        }
        input:checked + label + .content {
            display: block;
        }
        input + label:before {
            /* unicode characters and fontsizes are weirdly way off on mac vs linux?
            (I got around this by just making a v that's rotated via css transform...
            don't hate the player, hate CSS and web development.) */
            /*content: "\\203A";*/
            /*font-size: 20pt;*/
            content: "v";
            transform: rotate(-.25turn);
            margin-left: 2px;
            margin-right: 4px;
            position: relative;
            font-weight: bolder;
            font-family: Arial;
            display: inline-block;
        }
        input:checked + label:before {
            /*content: "\\2304";*/
            /*font-size: 20pt;*/
            display: inline-block;
            content: "v";
            transform: rotate(0turn);
            position: relative;
            margin-left: 2px;
            margin-right: 4px;
            font-family: Arial;
        }
        """
    ]

    _esm = """
        export function render({ model }) {
            const div = document.createElement('div')

            let i = 0;

            let new_input = document.createElement("input");
            new_input.type = "checkbox";
            new_input.id = "title";
            new_input.checked = true;

            let new_label = document.createElement("label");
            new_label.setAttribute("for", "title");
            new_label.innerHTML = model.label;

            let inner_div = document.createElement("div");
            inner_div.classList.add("content");
            inner_div.append(model.get_child("child"));

            div.append(new_input);
            div.append(new_label);
            div.append(inner_div);

            return div;
        }
    """


def create_explorer():
    """Set up and return full servable interactive explorer app UI inside a pretty template."""
    # NOTE: this is effectively a class with all the local functions etc,
    # leaving as a function because of how pn.serve works - it expects a
    # dictionary with values that are functions that return servable things.
    # This ended up being a much more flexible approach than the typical `panel
    # serve` CLI. (namely the ability to pass in custom args to this file's CLI,
    # such as the session folder `--session-path` arg)
    pn.extension("gridstack", "texteditor", "terminal")

    active_explorer = None

    # find and load any models from pre-defined model list
    # (the /models folder wherever sessions are being stored)
    if not os.path.exists(f"{SESSION_FOLDER}/models"):
        os.makedirs(f"{SESSION_FOLDER}/models", exist_ok=True)
    model_list = os.listdir(f"{SESSION_FOLDER}/models")

    models = {}
    for model in model_list:
        if not model.endswith(".json"):
            continue
        with open(f"{SESSION_FOLDER}/models/{model}") as infile:
            data = json.load(infile)
            name = model[: model.rfind(".")]
            if data["name"] is not None:
                name += f' ({data["name"]})'
            models[name] = f"{SESSION_FOLDER}/models/{model}"

    # ----------------------------------------------------------------------
    # ---- functions and event handlers for use by the overall template ----
    # ----------------------------------------------------------------------

    def load_session(*args, path: str):
        """Load all session data for a particular exploration from the specified path."""
        # (path is after *args because this is the target of an event handler
        # and is populated via a partial)
        nonlocal active_explorer, session_name, main_ui_container

        main_ui_container.loading = True
        with open(path) as infile:
            data = json.load(infile)
        ex = Explorer.from_dict(data)
        main_ui_container.objects = [ex._layout]
        main_ui_container.loading = False

        path_session_name = path[: path.rfind(".")]
        session_name.value = path_session_name
        active_explorer = ex

    def new_model_session(*args, model_path: str):
        """Start a blank exploration session using a model loaded from the specified path."""
        nonlocal active_explorer, session_name, main_ui_container

        model_path_name = model_path[model_path.rfind("/") + 1 : model_path.rfind(".")]
        model = reno.model.Model.load(model_path)
        ex = Explorer(model)
        main_ui_container.objects = [ex._layout]
        session_date = datetime.datetime.now().date().isoformat()
        session_name.value = f"{model_path_name}/Session-{session_date}"
        active_explorer = ex

    def get_recursive_sessions(starting_path: str):
        """Find all previously saved exploration sessions by recursing through the folders
        starting at the root sessions folder.

        This creates a set of nested BetterAccordion components with buttons for each found
        session that roughly aligns with the actual folder structure, essentially a "session
        file browser".
        """
        controls = []
        for subpath in os.listdir(starting_path):
            if subpath.endswith(".json"):
                session_name = subpath[: subpath.rfind(".")]
                button = pn.widgets.Button(
                    name=session_name,
                    button_type="primary",
                    stylesheets=[session_btn_css],
                    sizing_mode="stretch_width",
                )
                button.on_click(
                    partial(load_session, path=f"{starting_path}/{subpath}")
                )
                controls.append(button)
            # recurse into any subdirectories
            if os.path.isdir(f"{starting_path}/{subpath}") and subpath != "models":
                accordion = BetterAccordion(
                    label=f"{subpath}/",
                    child=pn.Column(
                        *get_recursive_sessions(f"{starting_path}/{subpath}"),
                        sizing_mode="stretch_width",
                    ),
                    sizing_mode="stretch_width",
                )
                accordion.label = f"{subpath}/"
                controls.append(accordion)

        return controls

    def refresh_loadable_sessions():
        """Entry point for the get_recursive_sessions function, populates
        the load_session_controls widget."""
        nonlocal load_session_controls

        load_session_controls.objects = [
            pn.pane.HTML("<b>Load session:</b>"),
            *get_recursive_sessions(SESSION_FOLDER),
        ]

    def save_session(self, *args):
        """Save the current system exploration session to whatever path is set in the
        session_name widget."""
        data = active_explorer.to_dict()
        filename = session_name.value

        output_path = f"{SESSION_FOLDER}/{filename}.json"
        output_folder = output_path[: output_path.rfind("/")]
        os.makedirs(output_folder, exist_ok=True)
        with open(output_path, "w") as outfile:
            json.dump(data, outfile)

        refresh_loadable_sessions()

    def server_ready():
        """This gets called every refresh or page change, and flipping the theme
        toggle technically makes the page refresh with a new get argument"""
        if b"dark" in pn.state.session_args.get("theme", [b"dark"]):
            print("DARK MODE ACTIVATED.")
            reno.diagrams.set_dark_mode(True)
        else:
            print("BLINDING MODE ACTIVATED.")
            reno.diagrams.set_dark_mode(False)

    # ---- /functions and event handlers for use by the overall template ----

    # -----------------------
    # ---- CSS overrides ----
    # -----------------------

    # styling for each clickable session button in the "session file explorer"
    # created in the get_recursive_sessions function.
    session_btn_css = """
    :host {
        margin-top: 2px;
        margin-bottom: 2px;
    }

    .bk-btn-group > button.bk-btn.bk-btn-primary {
        background-color: unset !important;
        border: unset !important;
        padding-top: 1px;
        padding-bottom: 1px;
        border-radius: 0;
        text-align: left;
        color: var(--background-text-color) !important;
    }
    .bk-btn-group > button.bk-btn.bk-btn-primary:hover {
        background-color: var(--accent-fill-rest) !important;
    }
    """

    # make the outline box go away from the session name textbox when
    # highlighted/entering text, this is normally a difficult task,
    # and even harder when panel's styling decides to add its own variant
    # of it.
    session_name_theme = """
    /* MAKE THE OUTLINE BOX GO AWAY. >:( */
    .bk-input {
        border: 0px !important;
    }

    .bk-input, .bk-input:focus {
        background-color: rgba(0, 0, 0, 0);
        border-radius: 0;
        border-bottom: 1px solid white !important;
        outline: 0px none transparent !important;
        box-shadow: none;
    }
    """

    # default template puts way too much space around the main layout
    fix_layout_css = """
    #main > .card-margin.stretch_width, #main > .card-margin {
        margin-top: 5px;
        margin-bottom: 5px;
    }
    """

    # ---- /CSS overrides ----

    # -------------------------------
    # ---- UI layout definitions ----
    # -------------------------------

    # the main layout with the actual Explorer interface
    main_ui_container = pn.Column(
        styles=dict(height="calc(100vh - 64px - 20px)", width="100%")
    )

    # additional controls to throw in header (session name textbox and save button)
    header_controls = pn.Row()

    # buttons for each model type to start a new session, left sidebar top
    new_session_controls = pn.Column()

    # previously saved session "file explorer", left sidebar bottom
    load_session_controls = pn.Column()

    # --- HEADER CONTROLS ---
    # -- save session button (goes in the header next to session name textbox) --
    # SVG for a floppy disk, kids these days don't understand having to fight
    # intrusive thoughts involving magnets and the old word documents you wrote
    # for school.
    save_icon = """
    <svg xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-device-floppy"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M6 4h10l4 4v10a2 2 0 0 1 -2 2h-12a2 2 0 0 1 -2 -2v-12a2 2 0 0 1 2 -2" /><path d="M12 14m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" /><path d="M14 4l0 4l-6 0l0 -4" /></svg>
    """
    save_btn = pn.widgets.Button(
        name=" ",
        icon=save_icon,
        styles={"margin-top": "8px"},
        icon_size="2em",
        description="Save session state",
    )
    save_btn.on_click(save_session)
    # -- /save session button --

    session_name = pn.widgets.TextInput(
        placeholder="Session name", stylesheets=[session_name_theme]
    )
    header_controls.objects = [session_name, save_btn]

    # --- /HEADER CONTROLS ---

    # --- NEW SESSION CONTROLS ---
    # make a "new session" button for each pre-defined model
    model_buttons = []
    for model in models:
        btn = pn.widgets.Button(
            name=f"{model}", button_type="primary", sizing_mode="stretch_width"
        )
        model_buttons.append(btn)
        btn.on_click(partial(new_model_session, model_path=models[model]))

    # file upload option to allow someone to upload their own model serialized
    # in a json file
    upload_model = pn.widgets.FileInput(accept=".json")

    new_session_controls.objects = [
        pn.pane.HTML("<b>New session with model:</b>"),
        *model_buttons,
        upload_model,
    ]
    # --- /NEW SESSION CONTROLS ---

    # --- LOAD SESSION CONTROLS ---
    # set up the "session file explorer" in the sidebar
    refresh_loadable_sessions()
    # --- /LOAD SESSION CONTROLS ---

    # hook up the theme switcher
    pn.state.onload(server_ready)

    template = pn.template.FastListTemplate(
        title="Reno Interactive Explorer",
        theme="dark",
        theme_toggle=True,
        header=[pn.Spacer(sizing_mode="stretch_width"), header_controls],
        main_layout=None,
        # accent="#2F5F6F",
        accent="#A0522D",
        sidebar=[new_session_controls, load_session_controls],
        main=[main_ui_container],
        raw_css=[fix_layout_css],
    ).servable()
    return template


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-path",
        dest="session_path",
        default="work_sessions",
        help="Where to store and load saved explorer sessions and models from.",
    )
    parser.add_argument(
        "--url-root-path",
        dest="root_path",
        default=None,
        help="Root path the application is being served on when behind a reverse proxy.",
    )
    parser.add_argument(
        "--port", dest="port", default=5006, help="What port to run the server on."
    )
    parser.add_argument(
        "--address",
        dest="address",
        default=None,
        help="What address to listen on for HTTP requests.",
    )
    parser.add_argument(
        "--liveness-check",
        dest="liveness",
        action="store_true",
        help="Flag to host a liveness endpoint at /liveness.",
    )
    parser.add_argument(
        "--websocket-origin",
        dest="websocket_origin",
        default=None,
        help="Host that can connect to the websocket, localhost by default.",
    )

    cli_args = parser.parse_args()

    SESSION_FOLDER = cli_args.session_path

    websocket_origin = ["localhost"]
    if cli_args.websocket_origin is not None:
        websocket_origin.append(cli_args.websocket_origin)

    pn.serve(
        {
            "explorer": create_explorer,
        },
        address=cli_args.address,
        port=cli_args.port,
        show=False,
        root_path=cli_args.root_path,
        liveness=cli_args.liveness,
        websocket_origin=websocket_origin,
    )
