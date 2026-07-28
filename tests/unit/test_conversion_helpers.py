import pytest

from modelconverter.utils.types import Target
from tests.helpers.target_options import target_options


@pytest.mark.parametrize(
    ("target", "version", "expected"),
    [
        (Target.RVC2, "2021.4.0", ("rvc2.superblob", "False")),
        (Target.RVC2, "2022.3.0", ()),
        (Target.RVC3, "2021.4.0", ()),
    ],
)
def test_target_options(
    monkeypatch: pytest.MonkeyPatch,
    target: Target,
    version: str,
    expected: tuple[str, ...],
) -> None:
    monkeypatch.setenv("MODELCONVERTER_TARGET_VERSION", version)

    assert target_options(target) == expected
