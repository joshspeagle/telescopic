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
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# Tutorials are MyST notebooks executed at every build, since a cache keyed on their cells
# would keep outputs computed by an older package; a failing cell fails the build.
nb_execution_mode = "force"
nb_execution_timeout = 300
nb_execution_raise_on_error = True

numpydoc_show_class_members = False

# Every cross-reference must resolve, but for type hints that have no page of their own.
nitpicky = True
nitpick_ignore = [
    ("py:class", name)
    for name in (
        "Seed",
        "matplotlib.axes._axes.Axes",
        "numpy._typing._array_like.ArrayLike",
        "numpy._typing._array_like.NDArray",
        "numpy.float64",
        "numpy.int64",
        "telescopic._random.Streams",
        "telescopic._types.Distribution",
    )
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

exclude_patterns = ["_build", "jupyter_execute", "**.ipynb_checkpoints"]
html_theme = "furo"
html_title = f"telescopic {release}"
