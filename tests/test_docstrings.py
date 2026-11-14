import contextlib
import doctest
import importlib
import pkgutil

import pytest

import telescopic


def modules():
    # Every module of the package that `import telescopic` finds: the sources under the
    # gate's PYTHONPATH, the installed package in CI. A module whose optional dependency
    # is missing, such as plotting without matplotlib, is skipped.
    for module in pkgutil.walk_packages(telescopic.__path__, "telescopic."):
        with contextlib.suppress(ImportError):
            yield importlib.import_module(module.name)


# Each example runs with empty globals, so it imports what it uses, as a reader who copies
# it must.
EXAMPLES = [
    test
    for module in modules()
    for test in doctest.DocTestFinder().find(module, globs={})
    if test.examples
]


def test_docstring_examples_are_found():
    # Without examples the test below would be skipped, not failed.
    assert EXAMPLES, f"no docstring examples in the package at {telescopic.__path__}"


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("example", EXAMPLES, ids=lambda example: example.name)
def test_docstring_example_prints_what_it_shows(example):
    # Warnings are errors, so an example that warns fails.
    report = []
    result = doctest.DocTestRunner().run(example, out=report.append)
    assert not result.failed, "".join(report)
