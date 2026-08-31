"""Run the doctests in the package's own docstrings.

``--doctest-modules`` cannot be turned on wholesale here the way it is in
``luxonis-ml``: importing every module of the package would pull in the
vendor toolchains, which only exist inside a target's Docker image. The
package is walked instead, and a module that will not import on the host
is reported as skipped rather than passed.
"""

import doctest
import importlib
import pkgutil
from contextlib import suppress

import pytest
from loguru import logger

import modelconverter

# The toolchains that only exist inside a target's Docker image.
_VENDOR_MODULES = frozenset({"hailo_sdk_client", "openvino", "plotly"})


def _module_names() -> list[str]:
    return sorted(
        module.name
        for module in pkgutil.walk_packages(
            modelconverter.__path__, f"{modelconverter.__name__}."
        )
        # `__main__` builds the CLI app at import time.
        if not module.name.endswith(".__main__")
    )


def _find_doctests(module_name: str) -> list[doctest.DocTest]:
    module = importlib.import_module(module_name)
    return doctest.DocTestFinder().find(module, module_name)


@pytest.mark.parametrize("module_name", _module_names())
def test_doctests(module_name: str) -> None:
    try:
        tests = _find_doctests(module_name)
    except ImportError as e:
        # Only a missing vendor toolchain is an expected failure. Any
        # other import error is a regression, so let it fail the test.
        if e.name not in _VENDOR_MODULES:
            raise
        pytest.skip(f"needs the {e.name} toolchain")

    runner = doctest.DocTestRunner()
    # The log goes to stdout, which is what doctest compares against.
    logger.disable("modelconverter")
    try:
        for test in tests:
            runner.run(test)
    finally:
        logger.enable("modelconverter")

    assert runner.failures == 0, (
        f"{runner.failures} of {runner.tries} doctest examples failed "
        f"in {module_name}"
    )


def test_doctests_are_collected() -> None:
    """Guard the skip above from hiding every doctest.

    A packaging change that broke the host imports would otherwise turn
    this whole module green while running nothing.
    """
    found = 0
    for module_name in _module_names():
        with suppress(ImportError):
            found += sum(
                len(test.examples) for test in _find_doctests(module_name)
            )

    assert found, "no doctest examples are reachable on the host"
