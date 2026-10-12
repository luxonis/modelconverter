"""HubAI benchmark slugs must resolve to exactly one platform instance."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from modelconverter.platforms.rvc2.benchmark import RVC2Benchmark
from modelconverter.platforms.rvc4.benchmark import RVC4Benchmark
from modelconverter.utils import hubai_utils
from modelconverter.utils.types import DataType

MODEL = "luxonis/example"
VARIANT = "default"
Catalog = tuple[Mock, list[SimpleNamespace], dict[str, list[SimpleNamespace]]]


def _version(
    identifier: str, slug: str = VARIANT, number: str = "1.0.0"
) -> SimpleNamespace:
    return SimpleNamespace(
        id=identifier,
        model_id="model-id",
        variant_slug=slug,
        version=number,
    )


def _instance(
    version_id: str,
    short_hash: str,
    platform: str = "RVC4",
    precision: str = "FP16",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"instance-{version_id}-{short_hash}",
        model_version_id=version_id,
        model_type=SimpleNamespace(value=platform),
        platforms=[SimpleNamespace(value=platform)],
        hash_short=short_hash,
        hash=short_hash.ljust(32, "0"),
        model_precision_type=SimpleNamespace(value=precision),
    )


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> Catalog:
    versions = [_version("v1"), _version("v2", number="2.0.0")]
    instances = {
        "v1": [_instance("v1", "1111111")],
        "v2": [_instance("v2", "2222222")],
    }
    client = Mock()
    client.models.get_model.return_value = SimpleNamespace(
        id="model-id", team_slug="luxonis"
    )

    def list_variants(**kwargs: object) -> list[SimpleNamespace]:
        return [
            item
            for item in versions
            if (
                kwargs["variant_slug"] is None
                or item.variant_slug == kwargs["variant_slug"]
            )
            and (
                kwargs["variant_version"] is None
                or item.version == kwargs["variant_version"]
            )
        ]

    def list_instances(**kwargs: object) -> list[SimpleNamespace]:
        return [
            item
            for item in instances[kwargs["variant_id"]]
            if item.model_type.value == kwargs["model_type"].value
        ]

    client.variants.list_variants.side_effect = list_variants
    client.instances.list_instances.side_effect = list_instances
    monkeypatch.setattr(hubai_utils, "create_hubai_client", lambda: client)
    return client, versions, instances


def test_name_with_one_variant_and_one_platform_instance(catalog: Catalog):
    _, _, instances = catalog
    instances["v2"] = [_instance("v2", "aaaaaaa", platform="RVC2")]

    benchmark = RVC4Benchmark(MODEL)

    assert (
        benchmark.model_path == f"{MODEL}:{VARIANT}:{'1111111'.ljust(32, '0')}"
    )
    assert benchmark._get_hubai_type() is DataType.FLOAT16


def test_unqualified_name_is_pinned_to_resolved_team(catalog: Catalog):
    _, _, instances = catalog
    instances["v2"] = []

    benchmark = RVC4Benchmark("example")

    assert str(benchmark.model_path).startswith("luxonis/example:default:")


def test_name_with_multiple_variants_is_ambiguous(catalog: Catalog):
    _, versions, _ = catalog
    versions.append(_version("v3", slug="other"))

    with pytest.raises(ValueError, match="multiple variants"):
        RVC4Benchmark(MODEL)


def test_one_version_with_multiple_platform_instances_is_ambiguous(
    catalog: Catalog,
):
    _, _, instances = catalog
    instances["v1"].append(_instance("v1", "3333333", precision="INT8"))

    with pytest.raises(ValueError, match="multiple RVC4 instances"):
        RVC4Benchmark(f"{MODEL}:{VARIANT}:1.0.0")


def test_multiple_versions_with_multiple_platform_instances_is_ambiguous(
    catalog: Catalog,
):
    with pytest.raises(ValueError, match="multiple RVC4 instances"):
        RVC4Benchmark(f"{MODEL}:{VARIANT}")


def test_one_version_variant_selects_unique_platform_instance(
    catalog: Catalog,
):
    _, versions, _ = catalog
    versions.pop()

    benchmark = RVC4Benchmark(f"{MODEL}:{VARIANT}")

    assert benchmark._model_instance == "1111111".ljust(32, "0")


def test_same_precision_instances_are_still_ambiguous(catalog: Catalog):
    _, _, instances = catalog
    instances["v1"].append(_instance("v1", "3333333"))

    with pytest.raises(ValueError, match="multiple RVC4 instances"):
        RVC4Benchmark(f"{MODEL}:{VARIANT}:1.0.0")


def test_version_searches_only_that_version(catalog: Catalog):
    client, _, _ = catalog

    benchmark = RVC4Benchmark(f"{MODEL}:{VARIANT}:1.0.0")

    assert benchmark._model_instance == "1111111".ljust(32, "0")
    assert client.instances.list_instances.call_count == 1
    assert (
        client.instances.list_instances.call_args.kwargs["variant_id"] == "v1"
    )


@pytest.mark.parametrize("reference", ["1111111", "1111111".ljust(32, "0")])
def test_hash_selects_instance_across_versions(
    catalog: Catalog, reference: str
):
    benchmark = RVC4Benchmark(f"{MODEL}:{VARIANT}:{reference}")

    assert benchmark._model_instance == "1111111".ljust(32, "0")
    assert benchmark._hubai_instance.model_version_id == "v1"


def test_hash_must_match_benchmark_platform(catalog: Catalog):
    _, _, instances = catalog
    instances["v1"] = [_instance("v1", "1111111", platform="RVC2")]

    with pytest.raises(ValueError, match="No RVC4 instance"):
        RVC4Benchmark(f"{MODEL}:{VARIANT}:1111111")

    benchmark = RVC2Benchmark(f"{MODEL}:{VARIANT}:1111111")
    assert benchmark._hubai_instance.model_type.value == "RVC2"


def test_hash_shared_by_two_versions_is_ambiguous(catalog: Catalog):
    _, _, instances = catalog
    instances["v2"] = [_instance("v2", "1111111")]

    with pytest.raises(ValueError, match="multiple RVC4 instances"):
        RVC4Benchmark(f"{MODEL}:{VARIANT}:1111111")


def test_unknown_version_does_not_fall_back_to_hash(catalog: Catalog):
    with pytest.raises(ValueError, match="No HubAI variant version"):
        RVC4Benchmark(f"{MODEL}:{VARIANT}:3.0.0")


def test_non_semver_is_treated_as_hash(catalog: Catalog):
    assert not hubai_utils._is_semver("1.0.0rc1")
    assert not hubai_utils._is_semver("1.0")
    assert not hubai_utils._is_semver("1.2.3-rc.1+build.4")
    assert not hubai_utils._is_semver("01.2.3")
    assert not hubai_utils._is_semver("1.2.3.4")
    assert hubai_utils._is_semver("1.2.3")

    with pytest.raises(ValueError, match="No RVC4 instance"):
        RVC4Benchmark(f"{MODEL}:{VARIANT}:1.0.0rc1")


def test_missing_platform_instance_is_reported(catalog: Catalog):
    _, _, instances = catalog
    instances["v1"] = []
    instances["v2"] = []

    with pytest.raises(ValueError, match="No RVC4 instance"):
        RVC4Benchmark(f"{MODEL}:{VARIANT}")


def test_precision_comes_from_selected_instance(catalog: Catalog):
    _, _, instances = catalog
    instances["v2"] = [_instance("v2", "2222222", precision="INT8")]

    benchmark = RVC4Benchmark(f"{MODEL}:{VARIANT}:2222222")

    assert benchmark._get_hubai_type() is DataType.INT8


def test_file_path_skips_hub_resolution(tmp_path: Path):
    model = tmp_path / "example.blob"
    model.touch()

    assert RVC2Benchmark(str(model)).model_path == model
