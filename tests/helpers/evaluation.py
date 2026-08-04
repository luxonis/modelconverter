"""Helpers for the real-model quality regression tests.

Luxonis Eval stays an external test dependency: modelconverter only supplies
conversion and native inference, Luxonis Eval parses the raw outputs and
calculates the task metrics.
"""

from collections.abc import Iterable

import numpy as np

from modelconverter.utils.types import ResizeMethod


def ordered_outputs(
    outputs: dict[str, np.ndarray], output_names: Iterable[str]
) -> list[np.ndarray]:
    """Order backend outputs by the model configuration, not dict order."""
    names = list(output_names)
    missing = [name for name in names if name not in outputs]
    unexpected = sorted(set(outputs) - set(names))
    if missing or unexpected:
        raise ValueError(
            "model outputs do not match the configured outputs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return [outputs[name] for name in names]


def require_resize(resize_method: ResizeMethod) -> None:
    """Reject preprocessing whose target geometry this v1 suite cannot score."""
    if resize_method is not ResizeMethod.RESIZE:
        raise ValueError(
            "real-model evaluation currently requires resize_method=RESIZE; "
            f"got {resize_method.value!r}"
        )


def assert_quality(
    reference: dict[str, float],
    converted: dict[str, float],
    *,
    floors: dict[str, float],
    max_drops: dict[str, float],
    case_id: str,
    platform: str,
) -> None:
    """Assert source floors and allowed converted-model regression budgets."""
    for metric, floor in floors.items():
        actual = reference[metric]
        assert actual >= floor, (
            f"{case_id} source {metric}={actual:.4f} is below floor {floor:.4f}"
        )

    for metric, max_drop in max_drops.items():
        drop = reference[metric] - converted[metric]
        assert drop <= max_drop, (
            f"{platform} {case_id} {metric} regressed by {drop:.4f}; "
            f"maximum allowed drop is {max_drop:.4f} "
            f"(source={reference[metric]:.4f}, "
            f"converted={converted[metric]:.4f})"
        )
