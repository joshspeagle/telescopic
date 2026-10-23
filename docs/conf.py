"""Sphinx configuration of the telescopic documentation."""

import telescopic

project = "telescopic"
author = "Joshua S. Speagle"
copyright = "2026, Joshua S. Speagle"
release = telescopic.__version__
version = release

extensions = [
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# Tutorials are MyST notebooks executed at build time; a failing cell fails the build.
nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_execution_raise_on_error = True

numpydoc_show_class_members = False
autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

exclude_patterns = ["_build", "jupyter_execute", "**.ipynb_checkpoints"]
html_theme = "furo"
html_title = f"telescopic {release}"
