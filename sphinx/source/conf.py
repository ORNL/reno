# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from reno import __version__

project = "Reno"
copyright = "2025, UT Battelle, LLC"
author = "Nathan Martindale, Jordan Stomps, Birdy Phathanapirom"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    # "sphinx_favicon",
]

autosummary_generate = True
autosummary_imported_members = False
# add_module_names = False  # NOTE: this doesn't apply to toc stuff


napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_typehints = "description"
autodoc_default_options = {
    "inherited-members": False,
    "undoc-members": True,
    "exclude-members": "__init__",
}

autoclass_content = (
    "class"  # only use the class docstring, don't use both init and class
)

templates_path = ["_templates"]

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "show_nav_level": 3,
    "navigation_depth": 6,
    "show_toc_level": 2,
    # "logo": {
    #     "image_light": "_static/icat_medium_full_light.svg",
    #     "image_dark": "_static/icat_medium_full_dark.svg",
    # },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ORNL/reno",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/reno-sd/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "external_links": [
        {"name": "Changelog", "url": "https://github.com/ORNL/reno/releases"},
    ],
    "switcher": {
        "json_url": "https://ornl.github.io/reno/stable/_static/switcher.json",
        "version_match": release,
    },
    "check_switcher": False,
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "show_version_warning_banner": True,
}

html_context = {"default_mode": "dark"}

# favicons = {"rel": "icon", "href": "icat_favicon.svg", "type": "image/svg+xml"}
