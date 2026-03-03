"""REST API endpoints for use with the explorer frontend."""

# https://panel.holoviz.org/how_to/server/endpoints.html

import json

import panel as pn
from tornado.web import RequestHandler


class SessionListerHandler(RequestHandler):
    def get(self):
        self.set_header("Content-Type", "application/json")
        self.write(
            json.dumps(
                {
                    session: pn.state.cache["active_sessions"][session].model.name
                    for session in pn.state.cache["active_sessions"]
                }
            )
        )
