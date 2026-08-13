"""Run the doctests in the package's own docstrings.

``--doctest-modules`` cannot be turned on wholesale here the way it is in
``luxonis-ml``: importing every module of the package would pull in the
vendor toolchains, which only exist inside a target's Docker image. So the
package is walked and each module that imports on the host contributes its
doctests, while the rest are reported as skipped rather than passed.
"""

import doctest
import importlib
import pkgutil

import pytest
from loguru import logger

import modelconverter

_OPTIONFLAGS = (
    doctest.NORMALIZE_WHITESPACE
    | doctest.IGNORE_EXCEPTION_DETAIL
    | doctest.ELLIPSIS
)


def _module_names() -> list[str]:
    return sorted(
        module.name
        for module in pkgutil.walk_packages(
            modelconverter.__path__, f"{modelconverter.__name__}."
        )
        # `__main__` builds the CLI app at import time.
        if not module.name.endswith(".__main__")
    )


@pytest.mark.parametrize("module_name", _module_names())
def test_doctests(module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        pytest.skip(f"needs a vendor toolchain: {e}")

    runner = doctest.DocTestRunner(optionflags=_OPTIONFLAGS)
    # The suite's logging goes to stdout, which doctest compares against the
    # expected output. A docstring documents what a call returns, not what it
    # logs on the way, so the log is kept out of the comparison.
    logger.disable("modelconverter")
    try:
        for test in doctest.DocTestFinder().find(module, module_name):
            runner.run(test)
    finally:
        logger.enable("modelconverter")

    assert runner.failures == 0, (
        f"{runner.failures} of {runner.tries} doctest examples failed "
        f"in {module_name}"
    )


def test_doctests_are_actually_collected() -> None:
    """Guard the skip path above from hiding every doctest.

    A packaging change that broke the host imports would otherwise turn
    this whole module green while running nothing.
    """
    found = 0
    for module_name in _module_names():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        found += sum(
            len(test.examples)
            for test in doctest.DocTestFinder().find(module, module_name)
        )

    assert found > 20, f"only {found} doctest examples reachable on the host"
