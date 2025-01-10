def is_hubai_available(model_slug: str) -> bool:
    from modelconverter.cli import Request, slug_to_id

    team_name = model_slug.split("/", 1)[0]
    if len(model_slug.split(":", 1)) < 2:
        team_name = ""
    model_name = model_slug.split(":", 1)[0]
    if len(model_slug.split(":", 1)) < 2:
        raise ValueError(
            f"Model variant not found in {model_slug}. Please specify it."
        )

    model_id = slug_to_id(
        model_slug.removeprefix(f"{team_name}/").split(":")[0], "models"
    )
    model_variants = Request.get(
        "modelVersions/", params={"model_id": model_id, "is_public": True}
    )

    for version in model_variants:
        if f"{model_name}:{version['variant_slug']}" == model_slug:
            return True

    return False
