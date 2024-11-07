import logging
import re
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import unquote, urlparse

import requests
import typer
from packaging.version import Version
from rich import print
from rich.progress import Progress

from modelconverter.cli import (
    ArchitectureIDOption,
    CommitHashOption,
    CompressionLevelOption,
    DescriptionOption,
    DescriptionShortOption,
    DomainOption,
    FilterPublicEntityByTeamIDOption,
    HashOption,
    HubVersionOption,
    HubVersionOptionRequired,
    IdentifierArgument,
    IsPublicOption,
    JSONOption,
    LicenseTypeOption,
    LicenseTypeOptionRequired,
    LimitOption,
    LinksOption,
    LuxonisOnlyOption,
    ModelClass,
    ModelIDOption,
    ModelIDOptionRequired,
    ModelType,
    ModelTypeOption,
    ModelVersionIDOption,
    ModelVersionIDOptionRequired,
    NameArgument,
    NameOption,
    OptimizationLevelOption,
    OptsArgument,
    Order,
    OrderOption,
    OutputDirOption,
    ParentIDOption,
    PathOption,
    PlatformsOption,
    ProjectIDOption,
    Quantization,
    QuantizationOption,
    RepositoryUrlOption,
    SilentOption,
    SlugOption,
    SortOption,
    StatusOption,
    TagsOption,
    TargetArgument,
    TargetPrecisionOption,
    TasksOption,
    TeamIDOption,
    UserIDOption,
    VariantSlugOption,
    get_configs,
    get_resource_id,
    hub_ls,
    print_hub_resource_info,
    request_info,
)
from modelconverter.cli.types import License, TargetPrecision
from modelconverter.utils.config import SingleStageConfig
from modelconverter.utils.types import Target

from .hub_requests import Request

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="Hub CLI",
    add_completion=False,
    rich_markup_mode="markdown",
)

model = typer.Typer(
    help="Models Interactions",
    add_completion=False,
    rich_markup_mode="markdown",
)

version = typer.Typer(
    help="Hub Versions Interactions",
    add_completion=False,
    rich_markup_mode="markdown",
)

instance = typer.Typer(
    help="Hub Instances Interactions",
    add_completion=False,
    rich_markup_mode="markdown",
)

app.add_typer(model, name="model", help="Models Interactions")
app.add_typer(version, name="version", help="Hub Versions Interactions")
app.add_typer(instance, name="instance", help="Hub Instances Interactions")


@model.command(name="ls")
def model_ls(
    team_id: TeamIDOption = None,
    tasks: TasksOption = None,
    user_id: UserIDOption = None,
    license_type: LicenseTypeOption = None,
    is_public: IsPublicOption = True,
    slug: SlugOption = None,
    project_id: ProjectIDOption = None,
    filter_public_entity_by_team_id: FilterPublicEntityByTeamIDOption = None,
    luxonis_only: LuxonisOnlyOption = False,
    limit: LimitOption = 50,
    sort: SortOption = "updated",
    order: OrderOption = Order.DESC,
) -> List[Dict[str, Any]]:
    """Lists models."""
    return hub_ls(
        "models",
        team_id=team_id,
        tasks=[task.name for task in tasks] if tasks else [],
        user_id=user_id,
        license_type=license_type,
        is_public=is_public,
        slug=slug,
        project_id=project_id,
        filter_public_entity_by_team_id=filter_public_entity_by_team_id,
        luxonis_only=luxonis_only,
        limit=limit,
        sort=sort,
        order=order,
        keys=["name", "id", "slug"],
    )


@model.command(name="info")
def model_info(
    identifier: IdentifierArgument,
    json: JSONOption = False,
):
    """Prints information about a model."""
    return print_hub_resource_info(
        request_info(identifier, "models"),
        title="Model Info",
        json=json,
        keys=[
            "name",
            "slug",
            "id",
            "created",
            "updated",
            "tasks",
            "platforms",
            "is_public",
            "is_commercial",
            "license_type",
            "versions",
            "likes",
            "downloads",
        ],
    )


@model.command(name="create")
def model_create(
    name: NameArgument,
    license_type: LicenseTypeOptionRequired = License.UNDEFINED,
    is_public: IsPublicOption = True,
    description: DescriptionOption = None,
    description_short: DescriptionShortOption = "<empty>",
    architecture_id: ArchitectureIDOption = None,
    tasks: TasksOption = None,
    links: LinksOption = None,
    silent: SilentOption = False,
) -> Dict[str, Any]:
    """Creates a new model resource."""
    data = {
        "name": name,
        "license_type": license_type,
        "is_public": is_public,
        "description_short": description_short,
        "description": description,
        "architecture_id": architecture_id,
        "tasks": tasks or [],
        "links": links or [],
    }
    try:
        res = Request.post("models", json=data).json()
    except requests.HTTPError as e:
        if str(e) == "{'detail': 'Unique constraint error.'}":
            raise ValueError(f"Model '{name}' already exists") from e
    print(f"Model '{res['name']}' created with ID '{res['id']}'")
    if not silent:
        model_info(res["id"])
    return res


@model.command(name="delete")
def model_delete(identifier: IdentifierArgument):
    """Deletes a model."""
    model_id = get_resource_id(identifier, "models")
    Request.delete(f"models/{model_id}")
    print(f"Model '{identifier}' deleted")


@version.command(name="ls")
def version_ls(
    team_id: TeamIDOption = None,
    user_id: UserIDOption = None,
    model_id: ModelIDOption = None,
    slug: SlugOption = None,
    variant_slug: VariantSlugOption = None,
    version: HubVersionOption = None,
    is_public: IsPublicOption = True,
    limit: LimitOption = 50,
    sort: SortOption = "updated",
    order: OrderOption = Order.DESC,
) -> List[Dict[str, Any]]:
    """Lists model versions."""
    return hub_ls(
        "modelVersions",
        team_id=team_id,
        user_id=user_id,
        model_id=model_id,
        is_public=is_public,
        slug=slug,
        variant_slug=variant_slug,
        version=version,
        limit=limit,
        sort=sort,
        order=order,
        keys=["id", "version", "slug", "platforms"],
    )


@version.command(name="info")
def version_info(
    identifier: IdentifierArgument, json: JSONOption = False
) -> Dict[str, Any]:
    """Prints information about a model version."""
    return print_hub_resource_info(
        request_info(identifier, "modelVersions"),
        title="Model Version Info",
        json=json,
        keys=[
            "name",
            "slug",
            "version",
            "id",
            "model_id",
            "created",
            "updated",
            "platforms",
            "exportable_to",
            "is_public",
        ],
    )


@version.command(name="create")
def version_create(
    name: NameArgument,
    model_id: ModelIDOptionRequired,
    version: HubVersionOptionRequired,
    description: DescriptionOption = None,
    repository_url: RepositoryUrlOption = None,
    commit_hash: CommitHashOption = None,
    domain: DomainOption = None,
    tags: TagsOption = None,
    silent: SilentOption = False,
) -> Dict[str, Any]:
    """Creates a new version of a model."""
    data = {
        "model_id": model_id,
        "name": name,
        "version": version,
        "description": description,
        "repository_url": repository_url,
        "commit_hash": commit_hash,
        "domain": domain,
        "tags": tags or [],
    }
    try:
        res = Request.post("modelVersions", json=data).json()
    except requests.HTTPError as e:
        if str(e).startswith("{'detail': 'Unique constraint error."):
            raise ValueError(
                f"Model version '{name}' already exists for model '{model_id}'"
            ) from e
    print(f"Model version '{res['name']}' created with ID '{res['id']}'")
    if not silent:
        version_info(res["id"])
    return res


@version.command(name="delete")
def version_delete(identifier: IdentifierArgument):
    """Deletes a model version."""
    version_id = get_resource_id(identifier, "modelVersions")
    Request.delete(f"modelVersions/{version_id}")
    print(f"Model version '{version_id}' deleted")


@instance.command(name="ls")
def instance_ls(
    platforms: PlatformsOption = None,
    team_id: TeamIDOption = None,
    user_id: UserIDOption = None,
    model_id: ModelIDOption = None,
    model_version_id: ModelVersionIDOption = None,
    model_type: ModelTypeOption = None,
    parent_id: ParentIDOption = None,
    model_class: Optional[ModelClass] = None,
    name: NameOption = None,
    hash: HashOption = None,
    status: StatusOption = None,
    is_public: IsPublicOption = True,
    compression_level: CompressionLevelOption = None,
    optimization_level: OptimizationLevelOption = None,
    slug: SlugOption = None,
    limit: LimitOption = 50,
    sort: SortOption = "updated",
    order: OrderOption = Order.DESC,
):
    """Lists model instances."""
    hub_ls(
        "modelInstances",
        platforms=[platform.name for platform in platforms]
        if platforms
        else [],
        model_id=model_id,
        model_version_id=model_version_id,
        model_type=model_type,
        parent_id=parent_id,
        model_class=model_class,
        name=name,
        hash=hash,
        status=status,
        compression_level=compression_level,
        optimization_level=optimization_level,
        team_id=team_id,
        user_id=user_id,
        is_public=is_public,
        slug=slug,
        limit=limit,
        sort=sort,
        order=order,
        keys=[
            "id",
            "slug",
            "platforms",
            "is_nn_archive",
        ],
    )


@instance.command(name="info")
def instance_info(
    identifier: IdentifierArgument, json: JSONOption = False
) -> Dict[str, Any]:
    """Prints information about a model instance."""
    return print_hub_resource_info(
        request_info(identifier, "modelInstances"),
        title="Model Instance Info",
        json=json,
        keys=[
            "name",
            "slug",
            "id",
            "model_version_id",
            "model_id",
            "created",
            "updated",
            "platforms",
            "is_public",
            "model_precision_type",
            "is_nn_archive",
            "downloads",
        ],
    )


@instance.command(name="download")
def instance_download(
    identifier: IdentifierArgument,
    output_dir: OutputDirOption = None,
) -> Path:
    """Downloads files from a model instance."""
    dest = Path(output_dir) if output_dir else None
    model_instance_id = get_resource_id(identifier, "modelInstances")
    for url in Request.get(
        f"modelInstances/{model_instance_id}/download"
    ).json():
        with requests.get(url, stream=True) as response:
            response.raise_for_status()

            filename = unquote(Path(urlparse(url).path).name)
            if dest is None:
                dest = Path(
                    Request.get(f"modelInstances/{model_instance_id}")
                    .json()
                    .get("slug", model_instance_id)
                )
            dest.mkdir(parents=True, exist_ok=True)

            with open(dest / filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Donwloaded '{f.name}'")

    assert dest is not None
    return dest


@instance.command(name="create")
def instance_create(
    name: NameArgument,
    model_version_id: ModelVersionIDOptionRequired,
    model_type: ModelTypeOption,
    parent_id: ParentIDOption = None,
    model_precision_type: TargetPrecisionOption = TargetPrecision.INT8,
    quantization_data: QuantizationOption = Quantization.RANDOM,
    tags: TagsOption = None,
    input_shape: Optional[List[int]] = None,
    is_deployable: Optional[bool] = None,
    silent: SilentOption = False,
) -> Dict[str, Any]:
    """Creates a new model instance."""
    data = {
        "name": name,
        "model_version_id": model_version_id,
        "parent_id": parent_id,
        "model_type": model_type,
        "model_precision_type": model_precision_type,
        "tags": tags or [],
        "input_shape": [input_shape] if input_shape else None,
        "quantization_data": quantization_data.name
        if quantization_data
        else None,
        "is_deployable": is_deployable,
    }
    res = Request.post("modelInstances", json=data).json()
    print(f"Model instance '{res['name']}' created with ID '{res['id']}'")
    if not silent:
        instance_info(res["id"])
    return res


@instance.command(name="delete")
def instance_delete(identifier: IdentifierArgument):
    """Deletes a model instance."""
    instance_id = get_resource_id(identifier, "modelInstances")
    Request.delete(f"modelInstances/{instance_id}")
    print(f"Model instance '{identifier}' deleted")


@instance.command()
def config(identifier: IdentifierArgument):
    """Prints the configuration of a model instance."""
    model_instance_id = get_resource_id(identifier, "modelInstances")
    res = Request.get(f"modelInstances/{model_instance_id}/config")
    print(res.json())


@instance.command()
def files(identifier: IdentifierArgument):
    """Prints the configuration of a model instance."""
    model_instance_id = get_resource_id(identifier, "modelInstances")
    res = Request.get(f"modelInstances/{model_instance_id}/files")
    print(res.json())


@instance.command()
def upload(file_path: str, identifier: IdentifierArgument):
    """Uploads a file to a model instance."""
    model_instance_id = get_resource_id(identifier, "modelInstances")
    with open(file_path, "rb") as file:
        files = {"files": file}
        Request.post(f"modelInstances/{model_instance_id}/upload", files=files)
    print(f"File '{file_path}' uploaded to model instance '{identifier}'")


def export(
    name: str,
    identifier: str,
    target: Literal["RVC2", "RVC3", "RVC4", "HAILO"],
    target_precision: Literal["FP16", "FP32", "INT8"] = "INT8",
    quantization_data: Optional[
        Literal["RANDOM", "GENERAL", "DRIVING", "FOOD", "INDOORS", "WAREHOUSE"]
    ] = "RANDOM",
    **kwargs,
) -> Dict[str, Any]:
    """Exports a model instance."""
    model_instance_id = get_resource_id(identifier, "modelInstances")
    json: Dict[str, Any] = {
        "name": name,
        "quantization_data": quantization_data,
        **kwargs,
    }
    if target == "RVC4":
        json["target_precision"] = target_precision
    res = Request.post(
        f"modelInstances/{model_instance_id}/export/{target.lower()}",
        json=json,
    ).json()
    print(
        f"Model instance '{name}' created for {target} export with ID '{res['id']}'"
    )
    return res


@app.command()
def convert(
    target: TargetArgument,
    path: PathOption = None,
    name: NameOption = None,
    license_type: LicenseTypeOptionRequired = License.UNDEFINED,
    config_path: PathOption = None,
    is_public: IsPublicOption = True,
    description_short: DescriptionShortOption = "<empty>",
    description: DescriptionOption = None,
    architecture_id: ArchitectureIDOption = None,
    tasks: TasksOption = None,
    links: LinksOption = None,
    model_id: ModelIDOption = None,
    version: HubVersionOption = None,
    repository_url: RepositoryUrlOption = None,
    commit_hash: CommitHashOption = None,
    target_precision: TargetPrecisionOption = TargetPrecision.INT8,
    quantization_data: QuantizationOption = Quantization.RANDOM,
    domain: DomainOption = None,
    tags: TagsOption = None,
    version_id: ModelVersionIDOption = None,
    output_dir: OutputDirOption = None,
    opts: OptsArgument = None,
) -> Path:
    """Starts the online conversion process."""
    opts = opts or []
    if isinstance(target, str):
        target = Target(target.lower())

    if path is not None:
        opts.extend(["input_model", str(path)])

    cfg, *_ = get_configs(str(config_path) if config_path else None, opts)

    if len(cfg.stages) > 1:
        raise ValueError(
            "Only single-stage models are supported with online conversion."
        )

    name = name or cfg.name

    cfg = next(iter(cfg.stages.values()))

    model_type = ModelType.from_suffix(cfg.input_model.suffix)
    version_name = _get_version_name(cfg, model_type, name)

    if model_id is None and version_id is None:
        try:
            model_id = model_create(
                name,
                license_type,
                is_public,
                description,
                description_short,
                architecture_id,
                tasks or [],
                links or [],
                silent=True,
            )["id"]
        except ValueError:
            model_id = get_resource_id(
                name.lower().replace(" ", "-"), "models"
            )

    if version_id is None:
        if model_id is None:
            print("`--model-id` is required to create a new model")
            exit(1)

        version = version or _get_version_number(model_id)

        version_id = version_create(
            version_name,
            model_id,
            version,
            description,
            repository_url,
            commit_hash,
            domain,
            tags or [],
            silent=True,
        )["id"]

    assert version_id is not None
    shape = cfg.inputs[0].shape
    instance_name = f"{version_name} base instance"
    instance_id = instance_create(
        instance_name, version_id, model_type, input_shape=shape, silent=True
    )["id"]

    upload(str(cfg.input_model), instance_id)
    if cfg.input_bin is not None:
        upload(str(cfg.input_bin), instance_id)

    cfg = cfg.model_dump(mode="json")
    exported_instance_name = f"{version_name} exported to {target.value}"

    instance = export(
        exported_instance_name,
        instance_id,
        target.name,
        target_precision=TargetPrecision(target_precision).name,
        quantization_data=Quantization(quantization_data).name
        if quantization_data
        else None,
        inputs=cfg["inputs"],
    )
    instance_id = instance["id"]
    run_id = instance["dag_run_id"]

    with Progress() as progress:
        progress.add_task("Waiting for the conversion to finish", total=None)
        run = _get_run(run_id)
        while run["status"] in ["PENDING", "RUNNING"]:
            sleep(10)
            run = _get_run(run_id)

    if run["status"] == "FAILURE":
        while len(run["logs"].split("\n")) < 5:
            run = _get_run(run_id)
            sleep(5)

        logs = _clean_logs(run["logs"])
        raise RuntimeError(f"Export failed with\n{logs}.")

    return instance_download(instance_id, output_dir)


def _clean_logs(logs: str) -> str:
    pattern = r"\[.*?\] \{.*?\} INFO - \[base\] logs:\s*"
    return re.sub(pattern, "", logs)


def _get_version_name(
    cfg: SingleStageConfig, model_type: ModelType, name: str
) -> str:
    shape = cfg.inputs[0].shape
    layout = cfg.inputs[0].layout

    if shape is not None:
        if layout is not None and "H" in layout and "W" in layout:
            h, w = shape[layout.index("H")], shape[layout.index("W")]
            return f"{name} {h}x{w}"
        elif len(shape) == 4:
            if model_type == ModelType.TFLITE:
                h, w = shape[1], shape[2]
            else:
                h, w = shape[2], shape[3]
            return f"{name} {h}x{w}"
    return name


def _get_version_number(model_id: str) -> str:
    versions = Request.get(
        "modelVersions/", params={"model_id": model_id}
    ).json()
    if not versions:
        version = "0.1.0"
    else:
        max_version = Version(versions[0]["version"])
        for v in versions[1:]:
            max_version = max(max_version, Version(v["version"]))
        max_version = str(max_version)
        version_numbers = max_version.split(".")
        version_numbers[-1] = str(int(version_numbers[-1]) + 1)
        version = ".".join(version_numbers)
    return version


def _get_run(run_id: str) -> Dict[str, Any]:
    run = Request.dag_get(f"runs/{run_id}").json()
    return run
