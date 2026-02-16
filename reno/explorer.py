"""Interactive panel interface and relevant components to allow exploring models live."""

import argparse
import base64
import datetime
import io
import json
import os
import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import panel as pn
import param
import PIL
import xarray as xr

import reno
from reno.viz import ReferenceEditor, plot_trace_refs

SESSION_FOLDER = ""

logo = """
<svg
   width="26px"
   height="14px"
   viewBox="0 0 26 14"
   version="1.1"
   id="svg1"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:svg="http://www.w3.org/2000/svg"
   fill="white"
>
  <g
     id="layer1"
     transform="translate(161.35697,-121.18184)">
    <path
       d="m -148.24299,135.27996 c -0.18456,-0.0585 -0.36659,-0.32016 -0.59667,-0.85755 -0.0398,-0.0931 -0.14513,-0.35263 -0.23368,-0.57641 -0.23418,-0.59148 -0.31232,-0.7799 -0.44567,-1.07478 -0.39566,-0.87497 -0.82327,-1.55334 -1.35705,-2.15297 -0.18534,-0.20819 -0.49386,-0.50185 -0.7265,-0.69146 -0.51099,-0.41652 -1.12719,-0.77433 -1.76645,-1.02574 -0.0708,-0.0279 -0.13005,-0.052 -0.13166,-0.0533 -10e-4,-0.001 1.06796,-0.002 2.37672,-0.002 h 2.37956 l -0.76284,0.76285 -0.76283,0.76287 h 2.12581 2.12582 l -0.76284,-0.76287 -0.76283,-0.76285 2.38689,2e-4 2.38691,1.9e-4 -0.13449,0.0538 c -1.51273,0.60501 -2.63953,1.6065 -3.46769,3.0821 -0.27634,0.49237 -0.47073,0.91858 -0.84282,1.84789 -0.23064,0.57608 -0.31684,0.77438 -0.42668,0.98171 -0.14119,0.26667 -0.27496,0.42035 -0.40578,0.46652 -0.058,0.0205 -0.13555,0.0215 -0.1952,0.002 z m 1.80077,-1.77489 c 0,-0.0157 0.18384,-0.43271 0.28448,-0.6457 0.67031,-1.41863 1.53386,-2.39113 2.70257,-3.04364 0.19545,-0.10924 0.68214,-0.3417 0.69276,-0.33108 10e-4,0.001 -0.0549,0.0483 -0.12491,0.1045 -0.42542,0.34169 -0.81822,0.74581 -1.26398,1.30041 -0.17031,0.2119 -0.25375,0.32077 -0.64436,0.84084 -0.18489,0.24614 -0.37869,0.50157 -0.43067,0.56761 -0.43111,0.54764 -0.79576,0.91796 -1.15152,1.16939 -0.0605,0.0428 -0.0645,0.0451 -0.0645,0.0376 z m -3.4956,-0.0609 c -0.19528,-0.139 -0.35336,-0.27557 -0.56648,-0.48922 -0.27939,-0.2801 -0.50588,-0.54692 -0.91866,-1.08226 -0.54168,-0.70251 -0.66558,-0.86017 -0.83679,-1.06461 -0.4806,-0.57391 -0.9106,-0.99401 -1.36355,-1.33218 -0.10659,-0.0796 -0.10571,-0.0787 -0.061,-0.0611 0.56346,0.22053 0.99137,0.44832 1.4274,0.75982 0.39712,0.28368 0.77847,0.64042 1.105,1.0337 0.35866,0.43198 0.70985,0.99663 0.99726,1.60345 0.11384,0.24046 0.30681,0.68246 0.30048,0.68839 -0.001,0.001 -0.0388,-0.0241 -0.0835,-0.0559 z m -10.14778,-1.887 c -0.27321,-0.022 -0.53688,-0.1264 -0.7527,-0.29831 -0.0653,-0.0521 -0.18519,-0.17304 -0.23152,-0.23377 -0.14387,-0.18853 -0.23685,-0.40977 -0.27505,-0.65438 -0.011,-0.0737 -0.0121,-0.18898 -0.0121,-2.12585 0,-2.27688 -0.002,-2.09153 0.0446,-2.28521 0.11735,-0.47167 0.50418,-0.86253 0.97635,-0.98643 0.1833,-0.0482 -0.0219,-0.0443 2.38318,-0.0443 2.04658,0 2.16679,5.9e-4 2.23065,0.0121 0.16037,0.0288 0.28013,0.068 0.41127,0.1345 0.39937,0.20277 0.66579,0.57238 0.74249,1.03004 0.011,0.064 0.0121,0.14371 0.0141,0.79823 l 0.002,0.72716 -1.99234,0.001 -1.99233,10e-4 -0.0645,0.014 c -0.23328,0.0526 -0.38962,0.16336 -0.47438,0.33583 -0.0387,0.0787 -0.0545,0.14701 -0.0545,0.23548 0,0.0873 0.008,0.13588 0.0349,0.20502 0.0661,0.17171 0.22111,0.30709 0.42112,0.36772 0.14371,0.0436 -0.03,0.0401 2.14659,0.0427 l 1.97537,0.001 -0.002,0.73389 c -0.002,0.80563 -0.001,0.78219 -0.0455,0.95124 -0.10317,0.39888 -0.36949,0.72287 -0.74074,0.90114 -0.12923,0.0621 -0.23881,0.0963 -0.39534,0.12356 -0.0633,0.011 -0.20385,0.0121 -2.17671,0.0121 -1.15989,5.8e-4 -2.13787,-0.001 -2.1733,-0.004 z m 19.55644,0.002 c -0.17839,-0.011 -0.35789,-0.059 -0.52553,-0.14088 -0.14419,-0.0705 -0.23598,-0.13524 -0.35492,-0.25087 -0.21967,-0.21348 -0.35358,-0.47455 -0.40537,-0.79031 -0.011,-0.0636 -0.0121,-0.15093 -0.0141,-0.80525 l -0.002,-0.73404 h 1.10911 1.10912 l 0.001,0.76546 0.001,0.76543 1.26127,-0.85385 c 0.69368,-0.4696 1.3238,-0.89613 1.40027,-0.94779 0.40478,-0.27357 0.47658,-0.32344 0.47452,-0.32951 -10e-4,-0.002 -0.70735,-0.48281 -1.56916,-1.0648 l -1.5669,-1.05814 -0.001,0.75809 -0.001,0.75809 h -1.10911 -1.1091 l 0.002,-0.72388 c 0.002,-0.79567 0.001,-0.76777 0.0462,-0.94764 0.11709,-0.47134 0.50428,-0.86257 0.97634,-0.98643 0.18329,-0.0482 -0.0217,-0.0443 2.38316,-0.0443 2.04657,0 2.16677,5.8e-4 2.23065,0.0121 0.16037,0.0288 0.28013,0.0679 0.41127,0.13449 0.40188,0.20404 0.67047,0.57897 0.7431,1.03733 0.011,0.0732 0.0121,0.18337 0.0121,2.12858 0,1.94128 -6.7e-4,2.05543 -0.0121,2.12923 -0.0642,0.41155 -0.28135,0.75014 -0.62316,0.97243 -0.0692,0.045 -0.21771,0.11761 -0.29062,0.14214 -0.069,0.0233 -0.17157,0.0488 -0.25392,0.0631 -0.0633,0.011 -0.20314,0.0121 -2.17669,0.0121 -1.1599,4e-4 -2.12567,-1.9e-4 -2.14619,-10e-4 z m -12.86018,-3.97497 c 1.21696,-0.48825 2.14427,-1.19279 2.90168,-2.20467 0.28059,-0.37487 0.51117,-0.75165 0.75319,-1.23076 0.22181,-0.43913 0.32946,-0.68787 0.68369,-1.57997 0.18011,-0.45365 0.24943,-0.61644 0.34356,-0.80695 0.1738,-0.35173 0.32258,-0.52878 0.47846,-0.56937 0.23531,-0.0614 0.44136,0.13979 0.69147,0.67501 0.079,0.16923 0.13789,0.30957 0.31625,0.75465 0.26788,0.66842 0.39101,0.95684 0.55167,1.2921 0.54865,1.14499 1.19084,1.98823 2.02413,2.65786 0.36922,0.29671 0.73318,0.52919 1.17311,0.74929 0.21631,0.10835 0.3345,0.16155 0.55266,0.24932 l 0.14921,0.0601 h -2.38014 -2.38013 l 0.75438,-0.75441 c 0.4149,-0.41492 0.75439,-0.75668 0.75439,-0.75947 0,-0.002 -0.95511,-0.004 -2.12247,-0.004 -1.16734,0 -2.12246,10e-4 -2.12246,0.004 0,0.002 0.33948,0.34455 0.75438,0.75947 l 0.75438,0.75441 -2.37335,-2e-4 -2.37333,-1.9e-4 z m -0.3116,-0.52198 c 0.002,-0.002 0.0385,-0.0299 0.0777,-0.059 0.2621,-0.196 0.47176,-0.37938 0.73963,-0.64692 0.40032,-0.39984 0.63977,-0.68369 1.30148,-1.54296 0.40675,-0.52818 0.52578,-0.67776 0.70959,-0.8918 0.28588,-0.33285 0.53901,-0.58262 0.79158,-0.78098 0.0995,-0.0782 0.22036,-0.16468 0.22421,-0.16055 0.002,0.002 -0.14935,0.35839 -0.22327,0.52121 -0.0651,0.14354 -0.24761,0.50986 -0.31925,0.6408 -0.76262,1.394 -1.78813,2.32203 -3.18168,2.87919 -0.11667,0.0467 -0.13206,0.052 -0.11991,0.0411 z m 10.8425,-0.1088 c -0.14559,-0.0627 -0.18991,-0.0833 -0.3582,-0.16768 -1.09951,-0.55145 -1.9482,-1.37589 -2.60668,-2.53227 -0.2015,-0.35386 -0.42045,-0.80875 -0.60775,-1.26258 l -0.011,-0.0255 0.032,0.0216 c 0.19593,0.13328 0.4638,0.36701 0.65252,0.56909 0.3057,0.32734 0.47101,0.53223 1.01717,1.26063 0.40914,0.5457 0.51871,0.68664 0.74596,0.95952 0.38819,0.46613 0.79858,0.87288 1.17291,1.16252 0.0699,0.054 0.0789,0.0615 0.0743,0.0614 -0.001,0 -0.052,-0.0211 -0.11155,-0.0467 z"
       id="path388" />
  </g>
</svg>
"""


# ==================================================
# ICONS
# ==================================================

icon_unfilled_check = """
<svg style="stroke: var(--secondary-bg-color);" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon icon-tabler icons-tabler-outline icon-tabler-circle-check"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M3 12a9 9 0 1 0 18 0a9 9 0 1 0 -18 0" /><path d="M9 12l2 2l4 -4" /></svg>
"""

icon_filled_check = """
<svg style="fill: var(--success-bg-color);" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor" class="icon icon-tabler icons-tabler-filled icon-tabler-circle-check"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path stroke="white" stroke-width="2.5" stroke-linecap="round" d="M9 12l2 2l4 -4" /><path d="M17 3.34a10 10 0 1 1 -14.995 8.984l-.005 -.324l.005 -.324a10 10 0 0 1 14.995 -8.336zm-1.293 5.953a1 1 0 0 0 -1.32 -.083l-.094 .083l-3.293 3.292l-1.293 -1.292l-.094 -.083a1 1 0 0 0 -1.403 1.403l.083 .094l2 2l.094 .083a1 1 0 0 0 1.226 0l.094 -.083l4 -4l.083 -.094a1 1 0 0 0 -.083 -1.32z" /></svg>
"""

icon_pencil = """
<svg style="stroke: var(--neutral-foreground-rest);" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon icon-tabler icons-tabler-outline icon-tabler-pencil"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M4 20h4l10.5 -10.5a2.828 2.828 0 1 0 -4 -4l-10.5 10.5v4" /><path d="M13.5 6.5l4 4" /></svg>
"""

icon_close = """
<svg style="fill: var(--danger-bg-color);" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor" class="icon icon-tabler icons-tabler-filled icon-tabler-xbox-x"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M12 2c5.523 0 10 4.477 10 10s-4.477 10 -10 10s-10 -4.477 -10 -10s4.477 -10 10 -10m3.6 5.2a1 1 0 0 0 -1.4 .2l-2.2 2.933l-2.2 -2.933a1 1 0 1 0 -1.6 1.2l2.55 3.4l-2.55 3.4a1 1 0 1 0 1.6 1.2l2.2 -2.933l2.2 2.933a1 1 0 0 0 1.6 -1.2l-2.55 -3.4l2.55 -3.4a1 1 0 0 0 -.2 -1.4" /></svg>
"""

icon_save = """
<svg xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-device-floppy"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M6 4h10l4 4v10a2 2 0 0 1 -2 2h-12a2 2 0 0 1 -2 -2v-12a2 2 0 0 1 2 -2" /><path d="M12 14m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" /><path d="M14 4l0 4l-6 0l0 -4" /></svg>
"""
# ==================================================
# /ICONS
# ==================================================


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

        self.run_index = 0
        """Running count of runs for default run name"""

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

    def _monkey_patch_smc_progress(outer_self):
        # from rich.progress import Progress
        from pymc.smc.sampling import CustomProgress

        class CustomSMCProgress(CustomProgress):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.total = len(self.tasks)
                self.per_progress = {key: 0.0 for key in self.task_ids}
                self.explorer_ref = outer_self

            def update(self, status, task_id, **kwargs):
                # print(task_id, status)
                self.per_progress[task_id] = float(status[status.rfind(" ") + 1 :])
                sumtotal = sum([progress for progress in self.per_progress.values()])
                sumtotal = sumtotal * 100 / len(self.tasks)
                sumtotal = int(sumtotal)
                self.explorer_ref.runs_list.progress.value = sumtotal
                super().update(status=status, task_id=task_id, **kwargs)

        reno.model.pm.smc.sampling.CustomProgress = CustomSMCProgress

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
        try:
            self.vars_editor.assign_from_controls()
            trace = self.model.pymc(compute_prior_only=True, keep_config=True)
            config = self.model.config()
            self.runs_list.add_run(
                config=config,
                trace=trace.prior,
                observations=None,
                name=f"Prior run ({self.run_index})",
            )
            self.run_index += 1
        except Exception as e:
            pn.state.notifications.error(f"Failed to run model: {e}", 0)
            print(traceback.format_exc())
        self.set_running(False)

    def run_posterior(self):
        """Run pymc on the model to get posteriors."""
        self.set_running(True)
        try:
            self.vars_editor.assign_from_controls()
            # reno.model.pm.smc.sampling.CustomProgress = None
            self.runs_list.progress.bar_color = "success"
            self.runs_list.progress.visible = True
            self.runs_list.progress.value = -1
            self._monkey_patch_smc_progress()
            observations = self.observables.get_observations()
            trace = self.model.pymc(keep_config=True, observations=observations)
            config = self.model.config()
            self.runs_list.add_run(
                config=config,
                trace=trace.posterior,
                observations=observations,
                name=f"Posterior run ({self.run_index})",
            )
            self.run_index += 1
            self.runs_list.progress.visible = False
        except Exception as e:
            pn.state.notifications.error(f"Failed to run model: {e}", 0)
            print(traceback.format_exc())
            self.runs_list.progress.bar_color = "danger"
        self.set_running(False)

    def _handle_selected_rows_changed(self, runs):
        traces = {run[0]: (run[1], run[2]) for run in runs}
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
            print("Getting reference ", ref_name)
            editor = ReferenceEditor(
                self.model, ref_name, self.model._is_init_ref(ref_name)
            )
            control = pn.widgets.TextInput(name=ref_name, value=editor.get_eq_str())
            editor.control = control

            ref = getattr(self.model, ref_name)
            print(ref)
            if (
                not self.model._is_init_ref(ref_name)
                and ref is not None
                and ref.doc is not None
            ):
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
        self.active_configs = {}

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

    def remove_pane(self, *args, pane_to_remove):
        self.panes.remove(pane_to_remove)
        new_objs = {}
        for loc, obj in self.gstack.objects.items():
            if obj == pane_to_remove:
                continue
            new_objs[loc] = obj
        self.gstack.objects = new_objs
        self.fire_on_new_controls_needed(None, None)
        self.invalidate_downloads()

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
        print(self.gstack.objects)

    def get_pane_delete_button(self, pane):
        btn = pn.widgets.Button(
            name=f"Remove {pane.__class__.__name__}", button_type="danger"
        )
        btn.on_click(partial(self.remove_pane, pane_to_remove=pane))
        return btn

    def add_text_pane(self):
        """Create a new editable text widget and add it to the tab interface."""
        text_pane = EditableTextPane()
        text_pane.clicker.on_click(
            partial(self.fire_on_new_controls_needed, text_pane.controls, text_pane)
        )
        self.add_pane(text_pane)
        self.fire_on_new_controls_needed(text_pane.controls, text_pane)

    def add_plots_pane(self):
        """Create a new plots widget and add it to the tab interface."""
        plots_pane = PlotsPane(self.model)
        plots_pane.clicker.on_click(
            partial(self.fire_on_new_controls_needed, plots_pane.controls, plots_pane)
        )
        self.add_pane(plots_pane)
        plots_pane.render(self.active_traces)
        self.fire_on_new_controls_needed(plots_pane.controls, plots_pane)

    def add_diagram_pane(self):
        """Create a new stock and flow diagram widget and add it to the tab
        interface."""
        diagram_pane = DiagramPane(self.model)
        diagram_pane.clicker.on_click(
            partial(
                self.fire_on_new_controls_needed, diagram_pane.controls, diagram_pane
            )
        )
        self.add_pane(diagram_pane)
        self.fire_on_new_controls_needed(diagram_pane.controls, diagram_pane)
        diagram_pane.render(self.active_traces)

    def add_config_pane(self):
        """Create a config comparison widget and add it to the tab
        interface."""
        config_pane = ConfigurationPane(self.model)
        config_pane.clicker.on_click(
            partial(self.fire_on_new_controls_needed, config_pane.controls, config_pane)
        )
        self.add_pane(config_pane)
        self.fire_on_new_controls_needed(config_pane.controls, config_pane)
        config_pane.render(self.active_configs)

    def on_new_controls_needed(self, callback: Callable):
        """Register a function to execute whenever a widget within the tab requests
        a new set of helper controls be displayed in the sidebar.

        Callbacks should take a panel widget.
        """
        self._on_new_controls_needed_callbacks.append(callback)

    def fire_on_new_controls_needed(self, controls_layout, pane):
        """Trigger the callbacks for the new_controls_needed event."""
        for callback in self._on_new_controls_needed_callbacks:
            if pane is not None:
                callback(
                    [self.controls, controls_layout, self.get_pane_delete_button(pane)]
                )
            elif controls_layout is not None:
                callback([self.controls, controls_layout])
            else:
                callback([self.controls, "Click on a pane to modify attributes"])

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
        print("???", self.gstack.objects)
        for loc, obj in self.gstack.objects.items():
            panes.append(
                {
                    "loc": [loc[0], loc[1], loc[2], loc[3]],
                    "type": obj.__class__.__name__,
                    "data": obj.to_dict(),
                }
            )
        print("SAVING PANES SET WITH LOCS")
        print([value["loc"] for value in panes])

        data = {
            "tab_name": self.tab_name,
            "panes": panes,
            "height": self.gstack.height,
            "nrows": self.gstack.nrows,
            "cells_height": self.cells_height,
            "configs": self.active_configs,
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

        self.active_configs = data["configs"]

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
            elif pane_type == "ConfigurationPane":
                pane = ConfigurationPane(self.model)
                pane.render(self.active_configs)
            pane.from_dict(pane_data)
            pane.clicker.on_click(
                partial(self.fire_on_new_controls_needed, pane.controls, pane)
            )

            # note that the way gridstack reports locations is a little
            # different than the recomended slice indexing mechanism specified
            # in the panel docs - but directly assigning the exact way it
            # reports still works, so we don't bother doing any sort of
            # wonky translation.
            obj_dict[loc] = pane
            print("ADDING AT", loc)

        self.gstack.objects = obj_dict
        self.panes = list(obj_dict.values())
        print(self.gstack.objects)

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


class ConfigurationPane(
    pn.widgets.base.WidgetBase, pn.custom.PyComponent, pn.reactive.Reactive
):
    """A widget to show differences in configuration between runs."""

    def __init__(self, model, **params):
        super().__init__(**params)

        self.controls = pn.Param(
            self,
            name="Config comparison controls",
            parameters=[],
        )

        self.rendered_configs = {}

        self.df = pd.DataFrame()
        self.df_view = pn.pane.DataFrame()

        self.clicker = ClickablePane(sizing_mode="stretch_both")
        self.clicker.child = pn.Column(self.df_view)

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

    def render(self, configs=None):
        if configs is not None:
            self.rendered_configs = configs

        self.df = pd.DataFrame(self.rendered_configs)
        self.df_view.object = self.df

    def to_html(self) -> str:
        """Get an HTML-compatible string for the contents of this pane. This is
        used for generating exported standalone reports, see tab_exporter."""
        return self.df.to_html()

    def to_dict(self) -> dict:
        """Serialize this pane to a dictionary that can be saved to file."""
        return {}

    def from_dict(self, data: dict):
        """Deserialize data into current instance from dictionary previously stored
        from ``to_dict()``."""
        return

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
                color: var(--panel-on-surface-color);
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
        self.btn_add_config = pn.widgets.Button(name="Add config")

        self.btn_add_text.on_click(self._handle_pnl_add_text_clicked)
        self.btn_add_diagram.on_click(self._handle_pnl_add_diagram_clicked)
        self.btn_add_plots.on_click(self._handle_pnl_add_plots_clicked)
        self.btn_add_config.on_click(self._handle_pnl_add_config_clicked)

        self.controls = pn.Row(
            self.btn_add_diagram,
            self.btn_add_text,
            self.btn_add_plots,
            self.btn_add_config,
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

    def _handle_pnl_add_config_clicked(self, *args):
        self.active_tab.add_config_pane()

    def update_traces(self, traces):
        """Change the traces being used in the current tab with those passed in."""
        self.active_tab.active_traces = {key: val[1] for key, val in traces.items()}
        self.active_tab.active_configs = {key: val[0] for key, val in traces.items()}

        for pane in self.active_tab.panes:
            if isinstance(pane, (DiagramPane, PlotsPane)):
                pane.render(self.active_tab.active_traces)
            elif isinstance(pane, ConfigurationPane):
                pane.render(self.active_tab.active_configs)

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
        if config is not None:
            self.config = {key: str(val) for key, val in config.items()}
        else:
            self.config = None
        self.trace = trace
        self.observations = observations
        super().__init__(**params)

        self.select_btn = pn.widgets.ButtonIcon(
            width=30,
            height=30,
            icon=icon_filled_check,
            active_icon=icon_filled_check,
            styles={"margin": "0px", "margin-left": "5px"},
        )
        self.remove_btn = pn.widgets.ButtonIcon(
            icon=icon_close,
            active_icon=icon_close,
            width=22,
            height=22,
            styles={
                "margin": "4px",
                "margin-left": "5px",
                "fill": "var(--danger-bg-color)",
            },
        )
        self.edit_btn = pn.widgets.ButtonIcon(
            icon=icon_pencil,
            active_icon=icon_pencil,
            styles={"margin": "2px"},
            width=26,
            height=26,
        )
        self.done_btn = pn.widgets.Button(name="done")

        self.select_btn.on_click(self._handle_pnl_select_btn_clicked)
        self.remove_btn.on_click(self._handle_pnl_remove_btn_clicked)
        self.edit_btn.on_click(self._handle_pnl_edit_btn_clicked)
        self.done_btn.on_click(self._handle_pnl_done_btn_clicked)

        self.label = pn.pane.HTML(f"{self.run_name}")

        self._layout = pn.Row(width=330)
        self.reset_view()

        self._selected_callbacks: list[Callable] = []
        self._removed_callbacks: list[Callable] = []

    def reset_view(self):
        self._layout.objects = [
            self.select_btn,
            self.edit_btn,
            self.label,
            pn.Spacer(sizing_mode="stretch_width"),
            self.remove_btn,
        ]

    def edit_view(self):
        self._layout.objects = [
            pn.Param(
                self,
                parameters=["run_name"],
                show_labels=False,
                show_name=False,
                widgets={"run_name": {"width": 150}},
            ),
            self.done_btn,
        ]

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
        self.label.object = f"{self.run_name}"
        # technically should make separate event handler for name changed, but
        # it's fine.
        self.fire_on_selected(self.visible)

    def _handle_pnl_remove_btn_clicked(self, *args):
        self.fire_on_removed()

    def _handle_pnl_edit_btn_clicked(self, *args):
        self.edit_view()

    def _handle_pnl_done_btn_clicked(self, *args):
        self.reset_view()

    @param.depends("visible", watch=True)
    def _update_selected_btn(self):
        if self.visible:
            self.select_btn.icon = icon_filled_check
        else:
            self.select_btn.icon = icon_unfilled_check

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
    main view for current tab.

    Includes optional run progress bar for showing status of in-progress run."""

    # TODO: the goal is to make selection apply per tab, but this isn't actually
    # implemented yet.

    def __init__(self, **params):
        super().__init__(**params)
        self.runs = []
        self._layout = pn.Column(width=330)

        # self.progress = pn.indicators.Progress(sizing_mode="stretch_width", visible=False, max=100)
        self.progress = pn.indicators.Progress(visible=False, max=100)

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

    def add_run(self, config, trace: xr.Dataset, observations, name=""):
        """Create a new RunRow with the passed configuration and data."""
        run = RunRow(
            run_name=name,
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
        obj_list = [pn.pane.HTML("<b>Model runs</b>"), *self.runs, self.progress]
        self._layout.objects = obj_list

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


def create_explorer():  # noqa: C901
    """Set up and return full servable interactive explorer app UI inside a pretty template."""
    # NOTE: this is effectively a class with all the local functions etc,
    # leaving as a function because of how pn.serve works - it expects a
    # dictionary with values that are functions that return servable things.
    # This ended up being a much more flexible approach than the typical `panel
    # serve` CLI. (namely the ability to pass in custom args to this file's CLI,
    # such as the session folder `--session-path` arg)
    pn.extension("gridstack", "texteditor", "terminal", notifications=True, nthreads=4)

    active_explorer = None
    active_session_name = ""
    if "active_sessions" not in pn.state.cache:
        pn.state.cache["active_sessions"] = {}

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
        nonlocal active_explorer, session_name, main_ui_container, active_session_name

        main_ui_container.loading = True
        try:
            with open(path) as infile:
                data = json.load(infile)
            ex = Explorer.from_dict(data)
            main_ui_container.objects = [ex._layout]
        except Exception as e:
            pn.state.notifications.error(f"Failed to load session: {e}", 0)
            print(traceback.format_exc())

        main_ui_container.loading = False

        path_session_name = path[: path.rfind(".")]
        path_session_name = str(Path(path_session_name).relative_to(SESSION_FOLDER))
        session_name.value = path_session_name
        active_explorer = ex
        pn.state.cache["active_sessions"][path_session_name] = ex
        active_session_name = path_session_name
        refresh_active_sessions()

    def new_model_session(*args, model_path: str):
        """Start a blank exploration session using a model loaded from the specified path."""
        nonlocal active_explorer, session_name, main_ui_container, active_session_name

        try:
            model_path_name = model_path[
                model_path.rfind("/") + 1 : model_path.rfind(".")
            ]
            model = reno.model.Model.load(model_path)
            ex = Explorer(model)
            main_ui_container.objects = [ex._layout]
            session_date = datetime.datetime.now().date().isoformat()
            name = f"{model_path_name}/Session-{session_date}"

            # make sure we don't conflict with an existing active session
            name_check = name
            i = 1
            while name_check in pn.state.cache["active_sessions"]:
                name_check = f"{name}_{i}"
                i += 1
            name = name_check

            session_name.value = name
            active_explorer = ex
            active_session_name = name
            ex.view.active_tab.add_diagram_pane()

            pn.state.cache["active_sessions"][session_name.value] = ex
            refresh_active_sessions()
        except Exception as e:
            pn.state.notifications.error(f"Failed to create new model session: {e}", 0)
            print(traceback.format_exc())

    def switch_active_session(*args, name: str):
        """Change to a different explorer UI (different active session)"""
        nonlocal active_explorer, session_name, main_ui_container, active_session_name
        try:
            ex = pn.state.cache["active_sessions"][name]
            main_ui_container.objects = [ex._layout]
            session_name.value = name
            active_explorer = ex
            active_session_name = name
        except Exception as e:
            pn.state.notifications.error(f"Failed to load active session: {e}", 0)
            print(traceback.format_exc())
        refresh_active_sessions()

    def close_session(*args, name: str):
        """Remove explorer from cache."""
        nonlocal active_explorer, active_session_name

        if name in pn.state.cache["active_sessions"]:
            del pn.state.cache["active_sessions"][name]

        if name == active_session_name:
            main_ui_container.objects = []
            session_name.value = ""
            active_explorer = None
            active_session_name = None

        refresh_active_sessions()

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
            pn.pane.HTML("<p style='margin-bottom: 0px;'><b>Load session:</b></p>"),
            *get_recursive_sessions(SESSION_FOLDER),
        ]

    def get_active_session_switchers():
        controls = []
        if len(pn.state.cache["active_sessions"]) == 0:
            controls.append(
                "Create a new session by clicking on a button above, or load a previous session by selecting from previously saved ones below."
            )
        for name in pn.state.cache["active_sessions"]:
            if name == active_session_name:
                button = pn.widgets.Button(
                    name=name,
                    button_type="primary",
                    sizing_mode="stretch_width",
                    stylesheets=[session_btn_css, highlighted_session_btn_css],
                )
            else:
                button = pn.widgets.Button(
                    name=name,
                    button_type="primary",
                    sizing_mode="stretch_width",
                    stylesheets=[session_btn_css],
                )
                # button.css_classes.append("highlighted")
            button.on_click(partial(switch_active_session, name=name))

            del_btn = pn.widgets.ButtonIcon(
                icon=icon_close,
                active_icon=icon_close,
                width=22,
                height=22,
                styles={"margin": "4px"},
            )
            del_btn.on_click(partial(close_session, name=name))
            controls.append(pn.Row(button, del_btn))
        return controls

    def refresh_active_sessions():
        """Create an option for each explorer that's been opened."""
        nonlocal active_session_controls

        active_session_controls.objects = [
            pn.pane.HTML(
                "<p style='margin-bottom: 0px;'><b>Switch active session:</b></p>"
            ),
            *get_active_session_switchers(),
        ]

    def save_session(self, *args):
        """Save the current system exploration session to whatever path is set in the
        session_name widget."""
        nonlocal active_session_name

        try:
            data = active_explorer.to_dict()
            filename = session_name.value

            # handle changing session name
            if filename != active_session_name:
                pn.state.cache["active_sessions"][filename] = active_explorer
                del pn.state.cache["active_sessions"][active_session_name]
                active_session_name = filename
                refresh_active_sessions()

            output_path = f"{SESSION_FOLDER}/{filename}.json"
            output_folder = output_path[: output_path.rfind("/")]
            os.makedirs(output_folder, exist_ok=True)
            with open(output_path, "w") as outfile:
                json.dump(data, outfile)
        except Exception as e:
            pn.state.notifications.error(f"Failed to save session: {e}", 0)
            print(traceback.format_exc())

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
        border-radius: 4px;
        text-align: left;
        color: var(--panel-on-surface-color) !important;
    }
    .bk-btn-group > button.bk-btn.bk-btn-primary:hover {
        background-color: var(--accent-fill-rest) !important;
    }
    """

    # styling to highlight a session btn
    highlighted_session_btn_css = """
    .bk-btn-group > button.bk-btn.bk-btn-primary {
        border: 2px solid var(--accent-fill-rest) !important;
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

    # the upload file css isn't wonderful
    fix_upload_css = """
    input[type="file"].bk-input {
        height: unset !important;
        text-align: unset !important;
        border: unset !important;
    }


    input[type="file"]::file-selector-button {
        background-color: var(--accent-fill-rest);
        color: var(--foreground-on-accent-rest);
        border: 0px;
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

    # currently open explorer UIs (active sessions)
    active_session_controls = pn.Column()

    # terminal = pn.widgets.Terminal(write_to_console=True)
    # terminal = pn.widgets.Terminal()
    # floating_terminal = pn.layout.FloatPanel(terminal, contained=False, position='left-center')
    # # floating_terminal = pn.layout.FloatPanel("hello?", contained=False, position='center')
    # # floating_terminal = "hello?"
    # sys.stdout = terminal
    # sys.stderr = terminal
    # main_ui_container.objects = [floating_terminal]

    # --- HEADER CONTROLS ---
    # -- save session button (goes in the header next to session name textbox) --
    # SVG for a floppy disk, kids these days don't understand having to fight
    # intrusive thoughts involving magnets and the old word documents you wrote
    # for school.
    save_btn = pn.widgets.Button(
        name=" ",
        icon=icon_save,
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
    upload_model = pn.widgets.FileInput(
        accept=".json", stylesheets=[fix_upload_css], width=310
    )

    new_session_controls.objects = [
        pn.pane.HTML(
            "<p style='margin-bottom: 0px; margin-top: 0px;'><b>New session with model:</b></p>"
        ),
        *model_buttons,
        upload_model,
    ]
    # --- /NEW SESSION CONTROLS ---

    # --- LOAD SESSION CONTROLS ---
    # set up the "session file explorer" in the sidebar
    refresh_loadable_sessions()
    # --- /LOAD SESSION CONTROLS ---

    # --- ACTIVE SESSION CONTROLS ---
    refresh_active_sessions()
    # --- /ACTIVE SESSION CONTROLS ---

    # hook up the theme switcher
    pn.state.onload(server_ready)

    logo_css = """

    """

    template = pn.template.FastListTemplate(
        title="",
        theme="dark",
        theme_toggle=True,
        header=[
            pn.pane.SVG(logo, height=44),
            "Reno Interactive Explorer",
            pn.Spacer(sizing_mode="stretch_width"),
            header_controls,
        ],
        main_layout=None,
        # accent="#2F5F6F",
        accent="#A0522D",
        sidebar=[new_session_controls, active_session_controls, load_session_controls],
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
