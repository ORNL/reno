"""REST API endpoints for use with the explorer frontend."""

# https://panel.holoviz.org/how_to/server/endpoints.html

import json

import panel as pn
from tornado.web import RequestHandler


class WorkspaceListerHandler(RequestHandler):
    """GET /api/workspace_list

    Return body:
    {
        "workspace_session_name": "model_name",
        ...
    }
    """

    def get(self):
        self.set_header("Content-Type", "application/json")
        self.write(
            json.dumps(
                {
                    session: pn.state.cache["active_workspaces"][session].model.name
                    for session in pn.state.cache["active_workspaces"]
                }
            )
        )


class RunPriorHandler(RequestHandler):
    """POST /api/run_prior

    Expected body:
    {
        "free_refs": {
            "ref_name": "value",
            ...
        },
        "n": "samples_num",
        "steps": "sample_length"
    }
    """

    def post(self):
        explorer = pn.state.cache["active_workspaces"][
            list(pn.state.cache["active_workspaces"])[0]
        ]
        body = json.loads(self.request.body)

        free_ref_assignments = []
        if "free_refs" in body:
            free_ref_assignments = body["free_refs"]

        for assignment in free_ref_assignments:
            # editor = explorer.vars_editor.reference_editors[assignment]
            ref_control = None
            for control in explorer.vars_editor.controls:
                if control.name == assignment:
                    ref_control = control
                    break

            if ref_control is None:
                raise Exception(f"Couldn't find control for {assignment}")

            ref_control.value = free_ref_assignments[assignment]

        if "n" in body:
            explorer.vars_editor.n.value = int(body["n"])

        if "steps" in body:
            explorer.vars_editor.steps.value = int(body["steps"])

        explorer.run_prior()


class RunPosteriorHandler(RequestHandler):
    """POST /api/run_posterior

    Expected body:
    {
        "free_refs": {
            "ref_name": "value",
            ...
        },
        "n": "samples_num",
        "steps": "sample_length",
        "observed": {
            "metric_name": {
                "data": "50",
                "sd": "5"
            },
            ...
        }
    }
    """

    def post(self):
        explorer = pn.state.cache["active_workspaces"][
            list(pn.state.cache["active_workspaces"])[0]
        ]
        body = json.loads(self.request.body)

        free_ref_assignments = []
        if "free_refs" in body:
            free_ref_assignments = body["free_refs"]

        for assignment in free_ref_assignments:
            # editor = explorer.vars_editor.reference_editors[assignment]
            ref_control = None
            for control in explorer.vars_editor.controls:
                if control.name == assignment:
                    ref_control = control
                    break

            if ref_control is None:
                raise Exception(f"Couldn't find control for {assignment}")

            ref_control.value = free_ref_assignments[assignment]

        if "n" in body:
            explorer.vars_editor.n.value = int(body["n"])

        if "steps" in body:
            explorer.vars_editor.steps.value = int(body["steps"])

        observations = {}
        if "observed" in body:
            for name, values in body["observed"].items():
                explorer.observables.add_observation()
                observable = explorer.observables.rows[-1]
                observable.reference.value = observable.options[name]
                observable.sigma.value = float(values["sd"])
                observable.data.value = float(values["data"])

        explorer.run_posterior()

        # explorer.observables
