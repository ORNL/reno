"""Script to export clean copies of interactive explorer visualizations/session tabs."""

import base64
import json
import os
import sys

import panel as pn
from bokeh.resources import INLINE

import reno


def load_pn_panes(model_file: str, panesset_file: str) -> reno.explorer.PanesSet:
    """Re-populate a new tab of panes from a JSON serialized panes file.

    Args:
        model_file (str): Path to a model JSON file.
        panesset_file (str): Path to a saved tab of panes (see explorer.py)

    Returns:
        A PanesSet object with all the same panes and data originally saved
        into panesset_file.
    """
    pn.extension("gridstack", "texteditor", "terminal")

    with open(model_file) as infile:
        model_data = json.load(infile)

    with open(panesset_file) as infile:
        panesset_data = json.load(infile)

    model = reno.model.Model.from_dict(model_data)

    panes = reno.explorer.PanesSet(model)
    panes.from_dict(panesset_data)

    return panes


def export_clean_html(model_file: str, panesset_file: str, output_path: str):
    """Cleaner approach to export_bokeh_html. This relies on every pane type having a
    ``to_html`` function that outputs something nice and static, (e.g. embedding images,
    little to no JS, etc.).

    Resulting HTML should be standalone and is much more amenable to printing/PDF export.

    Args:
        model_file (str): Path to a model JSON file.
        panesset_file (str): Path to a saved tab of panes (see explorer.py)
        output_path (str): Path to write the output HTML (include filename.)
    """
    paneset = load_pn_panes(model_file, panesset_file)

    for pane in paneset.panes:
        if hasattr(pane, "render"):
            pane.render(paneset.active_traces)

    html = """
    <head>
        <style>
            @page {
                size: A4 portrait;
                margin: .05in;
            }
            .grid-container {
                display: grid;
            }
            img {
                object-fit: scale-down;
                max-width: 100%;
                height: auto;
            }
        </style>
    </head>
    <body>
        <div class="grid-container">
    """

    for loc, obj in paneset.gstack.objects.items():
        html += f"<div class='grid-cell' style='grid-row-start: {loc[0] + 1}; grid-column-start: {loc[1] + 1}; grid-row-end: {loc[2] + 1}; grid-column-end: {loc[3] + 1};'>"
        html += obj.to_html()
        html += "</div>"

    html += "</div></body>"

    with open(output_path, "w") as outfile:
        outfile.write(html)


def export_bokeh_html(model_file: str, panesset_file: str, output_path: str):
    """Leaving in just in case we find it useful, but this effectively dumps out a small HTML
    that just loads in a ****ton of JS to populate every bokeh component. This is fine for
    just HTML outputs, but very messy to try to create printable PDFs with."""
    paneset = load_pn_panes(model_file, panesset_file)

    # simplify objects a little
    grid = pn.GridSpec(height=paneset.gstack.height, sizing_mode="stretch_both")
    grid.objects = {
        key: value.clicker.child for key, value in paneset.gstack.objects.items()
    }
    objs = [value.clicker.child for key, value in paneset.gstack.objects.items()]

    # so frustrating, a blank panel repo saving stuff doesn't have
    # any visbility issues, but for some reason in this project it
    # exports invisible every single time...so here's a hack to deal
    # with it for now
    force_visible = """
    :root, :host, :host div div, :root div div, div {
        visibility: visible !important;
    }
    """
    for obj in objs:
        obj.visible = True
        obj.stylesheets.append(force_visible)
        if hasattr(obj, "objects"):
            for subobj in obj.objects:
                subobj.visible = True
                subobj.stylesheets.append(force_visible)

    for pane in paneset.panes:
        if hasattr(pane, "render"):
            pane.render(paneset.active_traces)

    grid.save(output_path, resources=INLINE, embed=True)


def export_selenium_pdf(input_path: str, output_path: str):
    """Create a PDF from an HTML file using selenium's chrome webdriver.

    Args:
        input_path (str): The filepath/name of the HTML to render.
        output_path (str): The filepath/name to save the output PDF at.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.print_page_options import PrintOptions

    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.get(f"file://{os.path.abspath(input_path)}")
    print_options = PrintOptions()
    print_options.shink_to_fit = True
    pdf = base64.b64decode(driver.print_page(print_options))
    with open(output_path, "wb") as outfile:
        outfile.write(pdf)


if __name__ == "__main__":
    model_file_path = sys.argv[1]
    panes_file_path = sys.argv[2]
    html_file_path = sys.argv[3]
    pdf_file_path = sys.argv[4]

    export_clean_html(model_file_path, panes_file_path, html_file_path)
    export_selenium_pdf(html_file_path, pdf_file_path)
