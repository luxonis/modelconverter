from hubai_sdk import HubAIClient
from hubai_sdk.errors import ResourceNotFoundError
from hubai_sdk.utils.environ import environ as hubai_sdk_environ

from modelconverter.utils.environ import environ


def create_hubai_client() -> HubAIClient:
    """Create an authenticated HubAI SDK client using ModelConverter settings."""
    hubai_sdk_environ.HUBAI_URL = environ.HUBAI_URL
    return HubAIClient(api_key=environ.HUBAI_API_KEY)


def is_hubai_model_variant_available(
    model_identifier: str, model_variant: str
) -> bool:
    """Return whether a model variant is visible to the configured HubAI key."""
    client = create_hubai_client()
    try:
        model = client.models.get_model(model_identifier)
    except ResourceNotFoundError:
        return False

    return bool(
        client.variants.list_variants(
            model_id=model.id,
            variant_slug=model_variant,
            is_public=None,
            limit=1,
        )
    )
