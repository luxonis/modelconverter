"""Lookups against the HubAI model zoo.

Commands such as benchmarking accept a HubAI model slug in place of a
local file, so the slug has to be checked against the zoo before the
model is pulled from it.
"""

from hubai_sdk import HubAIClient
from hubai_sdk.errors import ResourceNotFoundError
from hubai_sdk.utils.environ import environ as hubai_sdk_environ

from modelconverter.utils.environ import environ


def create_hubai_client() -> HubAIClient:
    """Create a HubAI SDK client from the ModelConverter settings.

    The client is pointed at the configured HubAI URL and is
    authenticated with the configured API key.

    Returns:
        The authenticated client.

    """
    hubai_sdk_environ.HUBAI_URL = environ.HUBAI_URL
    return HubAIClient(api_key=environ.HUBAI_API_KEY)


def is_hubai_model_variant_available(
    model_identifier: str, model_variant: str
) -> bool:
    """Check whether a model variant is published on HubAI.

    The variant is looked up with the configured API key, so a private
    variant counts as available only for a key that can see it.

    Args:
        model_identifier: Name of the model, the first part of its
            slug.
        model_variant: Variant of the model, the part of the slug that
            follows the colon.

    Returns:
        ``True`` if HubAI lists a variant with this slug, ``False``
        otherwise.

    """
    client = create_hubai_client()
    try:
        client.variants.get_variant(f"{model_identifier}:{model_variant}")
    except ResourceNotFoundError:
        return False
    return True
