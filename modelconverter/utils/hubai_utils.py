from contextlib import suppress


def is_hubai_available(model_name: str, model_variant: str) -> bool:
    from modelconverter.cli import Request, slug_to_id

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
