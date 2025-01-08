import requests


def is_hubai_available(model_slug: str) -> bool:
    url = "https://easyml.cloud.luxonis.com/models/api/v1/models?is_public=true&limit=1000"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to get models. Status code: {response.status_code}"
        )
    hub_ai_models = response.json()
    for model in hub_ai_models:
        slug = f"{model['team_slug']}/{model['slug']}"
        if (
            slug in model_slug
            or slug.removeprefix(f"{model['team_slug']}/") in model_slug
        ):
            model_id = model["id"]

            url = f"https://easyml.cloud.luxonis.com/models/api/v1/modelVersions?model_id={model_id}&is_public=true"
            response = requests.get(url)
            if response.status_code != 200:
                raise ValueError(
                    f"Failed to get model versions. Status code: {response.status_code}"
                )
            model_versions = response.json()
            for version in model_versions:
                if (
                    f"{slug}:{version['variant_slug']}" == model_slug
                    or f"{slug}:{version['variant_slug']}".removeprefix(
                        f"{model['team_slug']}/"
                    )
                    == model_slug
                ):
                    return True
    return False
