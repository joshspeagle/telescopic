"""Sphinx configuration of the telescopic documentation."""

import os
import sys
from pathlib import Path

import telescopic

DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS / "_ext"))
# The plot style of every executed page, written by _scripts/brand.py; the notebook
# kernels inherit it from the environment.
os.environ["MATPLOTLIBRC"] = str(DOCS / "matplotlibrc")
# The kernels run this file before the first cell, as IPython runs PYTHONSTARTUP.
os.environ["PYTHONSTARTUP"] = str(DOCS / "notebook_startup.py")
# The modules that pages import to share code, such as the data of the line fit.
os.environ["PYTHONPATH"] = os.pathsep.join(
    filter(None, (str(DOCS / "_helpers"), os.environ.get("PYTHONPATH")))
)

project = "telescopic"
author = "Joshua S. Speagle"
copyright = "2026, Joshua S. Speagle"
release = telescopic.__version__
version = release

extensions = [
    "telescopic_theme",
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# Tutorials are MyST notebooks executed at every build, since a cache keyed on their cells
# would keep outputs computed by an older package; a failing cell fails the build. A local
# build working on prose or styling may reuse the cache or skip execution instead.
nb_execution_mode = os.environ.get("TELESCOPIC_DOCS_EXECUTE", "force")
if nb_execution_mode not in ("force", "cache", "off"):
    raise ValueError(
        f"TELESCOPIC_DOCS_EXECUTE must be 'force', 'cache' or 'off', not {nb_execution_mode!r}"
    )
nb_execution_timeout = 300
nb_execution_raise_on_error = True
# Consecutive printed lines of a cell show as one output.
nb_merge_streams = True

numpydoc_show_class_members = False
# Signatures show names and defaults, one per line once they pass 88 characters; the
# docstrings give the types.
python_maximum_signature_line_length = 88
autodoc_typehints = "none"

# Every cross-reference must resolve.
nitpicky = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# The design records are read in the repository, not on the site.
exclude_patterns = ["_build", "jupyter_execute", "**.ipynb_checkpoints", "design"]

html_theme = "telescopic"
html_theme_path = ["_themes"]
html_theme_options = {"source_repository": "https://github.com/joshspeagle/telescopic"}
html_title = f"telescopic {release}"
html_favicon = "_themes/telescopic/static/brand/favicon.svg"
# No template links a page's source and search reads the built pages, so _sources/ would
# only be dead weight.
html_copy_source = False
