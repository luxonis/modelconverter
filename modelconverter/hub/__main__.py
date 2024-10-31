import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

import requests
import typer
from rich import print

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
    HubVersionRequired,
    IdentifierArgument,
    IsPublicOption,
    JSONOption,
    LicenseTypeOption,
    LimitOption,
    LinksOption,
    LuxonisOnlyOption,
    ModelClass,
    ModelIDArgument,
    ModelIDOption,
    ModelIDRequired,
    ModelInstanceIDArgument,
    ModelPrecisionOption,
    ModelType,
    ModelTypeOption,
    ModelVersionIDArgument,
    ModelVersionIDOption,
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
    RepositoryUrlOption,
    SlugOption,
    SortOption,
    StatusOption,
    TagsOption,
    TargetArgument,
    TasksOption,
    TeamIDOption,
    UserIDOption,
    VariantSlugOption,
    get_configs,
    hub_ls,
    print_hub_resource_info,
    request_info,
)

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
):
    """Lists models."""
    hub_ls(
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
    license_type: LicenseTypeOption = None,
    is_public: IsPublicOption = True,
    description: DescriptionOption = None,
    description_short: DescriptionShortOption = None,
    architecture_id: ArchitectureIDOption = None,
    tasks: TasksOption = None,
    links: LinksOption = None,
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
    return Request.post("/api/v1/models", json=data).json()


@model.command(name="delete")
def model_delete(model_id: ModelIDArgument):
    """Deletes a model."""
    Request.delete(f"/api/v1/models/{model_id}")


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
):
    """Lists model versions."""
    hub_ls(
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
        keys=["id", "model_id", "version", "slug", "platforms"],
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
            "model_id",
            "id",
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
    model_id: ModelIDRequired,
    version: HubVersionRequired,
    description: DescriptionOption = None,
    repository_url: RepositoryUrlOption = None,
    commit_hash: CommitHashOption = None,
    domain: DomainOption = None,
    tags: TagsOption = None,
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
    return Request.post("/api/v1/models", json=data).json()


@version.command(name="delete")
def version_delete(model_id: ModelIDArgument):
    """Deletes a model version."""
    Request.delete(f"/api/v1/modelVersions/{model_id}")


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
        keys=["id", "model_version_id", "model_id", "slug", "platforms"],
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
    model_instance_id: ModelInstanceIDArgument,
    output_dir: OutputDirOption,
):
    """Downloads files from a model instance."""
    dest = Path(output_dir) if output_dir else None
    for url in Request.get(
        f"/api/v1/modelInstances/{model_instance_id}/download"
    ).json():
        with requests.get(url, stream=True) as response:
            response.raise_for_status()

            filename = unquote(Path(urlparse(url).path).name)
            if dest is None:
                dest = Path(
                    Request.get(f"/api/v1/modelInstances/{model_instance_id}")
                    .json()
                    .get("slug", model_instance_id)
                )
            dest.mkdir(parents=True, exist_ok=True)

            with open(dest / filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Donwloaded '{f.name}'")


@instance.command(name="create")
def instance_create(
    name: NameArgument,
    model_version_id: ModelVersionIDArgument,
    model_type: ModelType,
    parent_id: ParentIDOption = None,
    model_precision_type: ModelPrecisionOption = None,
    tags: TagsOption = None,
    input_shape: Optional[List[int]] = None,
    quantization_data: Optional[str] = None,
    is_deployable: Optional[bool] = None,
) -> Dict[str, Any]:
    """Creates a new model instance."""
    data = {
        "name": name,
        "model_version_id": model_version_id,
        "parent_id": parent_id,
        "model_type": model_type,
        "model_precision_type": model_precision_type,
        "tags": tags or [],
        "input_shape": input_shape,
        "quantization_data": quantization_data,
        "is_deployable": is_deployable,
    }
    return Request.post("/api/v1/modelInstances", json=data).json()


@instance.command()
def config(model_instance_id: ModelInstanceIDArgument):
    """Prints the configuration of a model instance."""
    res = Request.get(f"/api/v1/modelInstances/{model_instance_id}/config")
    print(res.json)


@instance.command()
def upload(file: Path, model_instance_id: ModelInstanceIDArgument):
    """Uploads a file to a model instance."""
    content_length = file.stat().st_size
    Request.post(
        f"/api/v1/modelInstances/{model_instance_id}/upload/",
        json={"files": [file]},
        headers={
            "Content-Type": "multipart/form-data",
            "Content-Length": str(content_length),
        },
    )


@app.command()
def convert(
    target: TargetArgument,
    name: NameArgument,
    license_type: LicenseTypeOption = None,
    path: PathOption = None,
    is_public: IsPublicOption = True,
    description_short: DescriptionShortOption = None,
    description: DescriptionOption = None,
    architecture_id: ArchitectureIDOption = None,
    tasks: TasksOption = None,
    links: LinksOption = None,
    model_id: ModelIDOption = None,
    version: HubVersionOption = None,
    repository_url: RepositoryUrlOption = None,
    commit_hash: CommitHashOption = None,
    domain: DomainOption = None,
    tags: TagsOption = None,
    version_id: ModelVersionIDOption = None,
    opts: OptsArgument = None,
):
    """Starts the online conversion process."""
    if model_id is not None and version_id is not None:
        raise ValueError("Cannot provide both model_id and version_id")

    if model_id is None and version_id is None:
        model_id = model_create(
            name,
            license_type,
            is_public,
            description_short,
            description,
            architecture_id,
            tasks or [],
            links or [],
        )["id"]

    if version_id is None:
        if model_id is None:
            print("`--model-id` is required to create a new model")
            exit(1)

        if version is None:
            print("`--version` is required to create a new model version")
            exit(1)

        version_id = version_create(
            model_id,
            name,
            version,
            description,
            repository_url,
            commit_hash,
            domain,
            tags or [],
        )["id"]

    assert model_id is not None
    instance_id = instance_create(name, model_id, ModelType(target.name))["id"]

    if path is not None:
        upload(path, instance_id)

    cfg, *_ = get_configs(str(path), opts)
    Request.post(
        f"/api/v1/modelInstances/{instance_id}/upload/",
        json=cfg,
    )

    Request.post(
        f"/api/v1/modelInstances/{instance_id}/export/{target.value}",
        json=cfg,
    )
