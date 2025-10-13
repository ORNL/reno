"""IPyVuetify component to wrap a jupyter latex widget with the ability
to click on equation lines to highlight everywhere else that equation
is used."""

from collections.abc import Callable

import ipyvuetify as ipv
import ipywidgets as ipw
import traitlets
from IPython.display import Latex, clear_output, display

import reno


class InteractiveLatex(ipv.VuetifyTemplate):
    """Adds an event handler to clicking a line in an equation.

    (Presumably only works for begin{align} blocks?)
    """

    latex_output = traitlets.Any().tag(sync=True, **ipw.widget_serialization)
    template_file = reno.utils.resource_path("interactive_latex.vue")

    def __init__(self):
        self.latex = Latex()
        self.latex_output = ipw.Output()

        self._row_clicked_callbacks: list[Callable[[int], None]] = []

        super().__init__()

    def refresh_display(self):
        """Re-send updated latex to frontend."""
        with self.latex_output:
            clear_output(True)
            display(self.latex)
        self.send({"method": "attachEventHandlers", "args": []})

    def on_row_clicked(self, callback: Callable[[int], None]):
        """Register function to execute when a row (equation) in the
        latex is clicked.

        Callback should take the integer index of the clicked row.
        """
        self._row_clicked_callbacks.append(callback)

    def fire_on_row_clicked(self, index: int):
        """Trigger the callbacks for the row_clicked event."""
        for callback in self._row_clicked_callbacks:
            callback(index)

    def vue_fire_on_row_clicked(self, index: int):
        """Make fire_on_row_clicked accessible to vue."""
        self.fire_on_row_clicked(index)
