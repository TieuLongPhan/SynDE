import os
import sys
import importlib.metadata as m
from importlib.metadata import version as _get_version, PackageNotFoundError

# -- Path setup --------------------------------------------------------------
# Add project root to sys.path to import the package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "synde"
author = "Tuyet-Minh Phan"


try:
    release = _get_version("synde")
except PackageNotFoundError:
    try:
        release = m.version("synde")
    except (ImportError, AttributeError):
        # Fallback default
        release = "0.2.0"
# Use only major.minor for short version
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
]

# Mock optional heavy or uninstalled third-party dependencies during doc build
autodoc_mock_imports = ["thermo"]

bibtex_bibfiles = ["refs.bib"]
templates_path = [os.path.join(os.path.dirname(__file__), "_templates")]
exclude_patterns = ["_build"]
autosectionlabel_prefix_document = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = "SynDE Documentation"
html_short_title = "SynDE"

html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
html_js_files = [
    "custom.js",
]

html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
