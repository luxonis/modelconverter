"""Resolve HubAI model slugs to one deployable benchmark instance."""

import re
from dataclasses import dataclass

from hubai_sdk import HubAIClient
from hubai_sdk.errors import ResourceNotFoundError
from hubai_sdk.utils.environ import environ as hubai_sdk_environ
from hubai_sdk.utils.sdk_models import ModelInstanceResponse
from hubai_sdk.utils.types import ModelType
from semver import Version

from modelconverter.utils.environ import environ

_HUB_MODEL_PATTERN = re.compile(
    r"^(?:(?P<team>[^/:]+)/)?(?P<model>[^/:]+)"
    r"(?::(?P<variant>[^/:]+)(?::(?P<reference>[^/:]+))?)?$"
)
_LIST_LIMIT = 500


@dataclass(frozen=True)
class ResolvedHubAIModel:
    """A benchmark model pinned to one HubAI instance."""

    slug: str
    model_name: str
    variant_slug: str
    instance: ModelInstanceResponse


def create_hubai_client() -> HubAIClient:
    """Create an SDK client using the ModelConverter HubAI settings."""
    hubai_sdk_environ.HUBAI_URL = environ.HUBAI_URL
    return HubAIClient(api_key=environ.HUBAI_API_KEY)


def is_hubai_model_variant_available(
    model_identifier: str, model_variant: str
) -> bool:
    """Return whether an accessible model has this variant slug."""
    client = create_hubai_client()
    try:
        model = client.models.get_model(model_identifier)
    except ResourceNotFoundError:
        return False
    return bool(
        client.variants.list_variants(
            model_id=model.id, variant_slug=model_variant, limit=1
        )
    )


def _is_semver(reference: str) -> bool:
    """Recognize only ``major.minor.patch`` with three integer parts."""
    try:
        version = Version.parse(reference)
    except ValueError:
        return False
    return version.prerelease is None and version.build is None


def _require_complete_list(items: list, resource: str) -> list:
    if len(items) >= _LIST_LIMIT:
        raise ValueError(
            f"HubAI returned at least {_LIST_LIMIT} {resource}; cannot resolve "
            "a unique benchmark instance from a truncated list."
        )
    return items


def resolve_hubai_benchmark_model(
    identifier: str, platform: str
) -> ResolvedHubAIModel:
    """Resolve a HubAI benchmark slug to exactly one platform instance.

    The SDK first resolves ``[team/]model``. An unqualified model name
    must identify one visible model. Selection then follows these rules:

    * ``model``: require exactly one distinct variant slug, then search
      all its versions for instances deployable to ``platform``.
    * ``model:variant``: search all versions of that variant for
      instances deployable to ``platform``.
    * ``model:variant:version``: search only the exact version. The
      third segment is a version only if it contains exactly three
      integer SemVer components (``major.minor.patch``), without a
      prerelease or build suffix.
    * ``model:variant:hash``: search all variant versions for an
      instance with that full or seven-character short MD5 hash. Any
      third segment that is not SemVer is interpreted as a hash.

    Every search must find exactly one matching platform instance.
    Precision and creation time never break ties. The returned slug
    includes the resolved team, variant, and full instance hash so the
    subsequent DepthAI download requests that selected artifact.

    Args:
        identifier: HubAI model slug in one of the forms above.
        platform: Benchmark target, such as ``RVC2`` or ``RVC4``.

    Returns:
        The pinned download slug and selected SDK instance metadata.

    Raises:
        ValueError: If the slug is invalid, a variant or instance is
            missing or ambiguous, or a complete result cannot be
            established from the SDK lists.
    """
    match = _HUB_MODEL_PATTERN.fullmatch(identifier)
    if match is None:
        raise ValueError(
            "Invalid HubAI model slug. Expected "
            "[team/]model[:variant[:version-or-hash]]."
        )

    team, model_name, variant_slug, reference = match.group(
        "team", "model", "variant", "reference"
    )
    model_identifier = f"{team}/{model_name}" if team else model_name
    client = create_hubai_client()
    model = client.models.get_model(model_identifier)
    model_type = ModelType(platform)

    is_version = reference is not None and _is_semver(reference)
    versions = _require_complete_list(
        client.variants.list_variants(
            model_id=model.id,
            variant_slug=variant_slug,
            variant_version=reference if is_version else None,
            limit=_LIST_LIMIT,
        ),
        "variant versions",
    )
    versions = [
        version
        for version in versions
        if version.model_id == model.id
        and (variant_slug is None or version.variant_slug == variant_slug)
        and (not is_version or version.version == reference)
    ]
    if not versions:
        raise ValueError(f"No HubAI variant version matches '{identifier}'.")

    variant_slugs = {version.variant_slug for version in versions}
    if len(variant_slugs) > 1:
        raise ValueError(
            f"Model '{model_identifier}' has multiple variants: "
            f"{', '.join(sorted(variant_slugs))}. Specify a variant."
        )
    selected_variant = next(iter(variant_slugs))

    instances = {}
    for version in versions:
        listed = _require_complete_list(
            client.instances.list_instances(
                model_id=model.id,
                variant_id=version.id,
                model_type=model_type,
                limit=_LIST_LIMIT,
            ),
            "instances",
        )
        for instance in listed:
            if (
                instance.model_version_id == version.id
                and getattr(instance.model_type, "value", instance.model_type)
                == platform
                and any(
                    getattr(candidate, "value", candidate) == platform
                    for candidate in instance.platforms or []
                )
            ):
                instances[instance.id] = instance

    if reference and not is_version:
        instances = {
            instance_id: instance
            for instance_id, instance in instances.items()
            if reference.casefold()
            in {
                str(instance.hash or "").casefold(),
                str(instance.hash_short or "").casefold(),
            }
        }

    if not instances:
        raise ValueError(
            f"No {platform} instance matches HubAI model '{identifier}'."
        )
    if len(instances) > 1:
        raise ValueError(
            f"HubAI model '{identifier}' has multiple {platform} instances. "
            "Specify a version or a unique instance hash."
        )

    instance = next(iter(instances.values()))
    if (
        not instance.hash
        or re.fullmatch(r"[0-9a-fA-F]{32}", instance.hash) is None
    ):
        raise ValueError(
            f"HubAI instance '{instance.id}' has no full MD5 hash and cannot "
            "be pinned."
        )
    team_slug = team or model.team_slug
    if not team_slug:
        raise ValueError(
            f"HubAI model '{model_identifier}' has no team slug; cannot pin "
            "the resolved model for download."
        )
    qualified_model = f"{team_slug}/{model_name}"
    return ResolvedHubAIModel(
        slug=f"{qualified_model}:{selected_variant}:{instance.hash}",
        model_name=model_name,
        variant_slug=selected_variant,
        instance=instance,
    )
