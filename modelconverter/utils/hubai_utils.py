"""Lookups against the HubAI model zoo.

Commands such as benchmarking accept a HubAI model slug in place of a
local file, so the slug has to be checked against the zoo before the
model is pulled from it.
"""

from contextlib import suppress


def is_hubai_available(model_name: str, model_variant: str) -> bool:
    """Check whether a model variant is published on HubAI.

    Both the public and the private variants of the model are queried,
    and a request that fails is treated as returning no variants.

    Args:
        model_name: Name of the model, the first part of its slug.
        model_variant: Variant of the model, the part of the slug that
            follows the colon.

    Returns:
        ``True`` if HubAI lists a variant with this slug, ``False``
        otherwise.

    """
    from modelconverter.cli import slug_to_id
    from modelconverter.utils.hub_requests import Request

    model_slug = f"{model_name}:{model_variant}"

    model_id = slug_to_id(
        model_name,
        "models",
    )

    model_variants = []
    for is_public in [True, False]:
        with suppress(Exception):
            model_variants += Request.get(
                "modelVersions/",
                params={"model_id": model_id, "is_public": is_public},
            )

    for version in model_variants:
        if f"{model_name}:{version['variant_slug']}" == model_slug:
            return True

    return False
