import doctest
import importlib
import pkgutil

import pytest

import telescopic

# The doctests of every module of the package that `import telescopic` finds: the sources
# under the gate's PYTHONPATH, the installed package in CI. Each example runs with empty
# globals, so it imports what it uses, as a reader who copies it must.
EXAMPLES = [
    test
    for module in pkgutil.walk_packages(telescopic.__path__, "telescopic.")
    for test in doctest.DocTestFinder().find(importlib.import_module(module.name), globs={})
    if test.examples
]


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda example: example.name)
def test_docstring_example_prints_what_it_shows(example):
    # Warnings are errors here, so an example that warns fails.
    report = []
    result = doctest.DocTestRunner().run(example, out=report.append)
    assert not result.failed, "".join(report)
