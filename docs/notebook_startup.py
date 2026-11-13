"""Run by the kernel of every executed page before its first cell (``docs/conf.py``)."""

import warnings

# A warning prints as its class and message: the default format starts with the path of
# the kernel's temporary file, which changes at every build, and repeats the source line.
warnings.formatwarning = lambda message, category, *args, **kwargs: (
    f"{category.__name__}: {message}\n"
)
# The pages import what they use, so the startup leaves no name behind.
del warnings
