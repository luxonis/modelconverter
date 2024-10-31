import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

import requests
import typer
from luxonis_ml.nn_archive import is_nn_archive
from rich import print

from modelconverter.cli import (
    FilterPublicEntityByTeamIDOption,
    IdentifierArgument,
    IsPublicOption,
    JSONOption,
    LicenseTypeOption,
    LimitOption,
    LuxonisOnlyOption,
    ModelClass,
    ModelIDArgument,
    ModelIDOption,
    ModelType,
    ModelVersionIDOption,
    OptsArgument,
    Order,
    OrderOption,
    PathOption,
    ProjectIDOption,
    SlugOption,
    SortOption,
    Status,
    TargetArgument,
    Task,
    TasksOption,
    TeamIDOption,
    UserIDOption,
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

app.add_typer(
    model,
    name="model",
    help="Models Interactions",
    invoke_without_command=True,
)
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
    name: str,
    license_type: Optional[str] = None,
    public: bool = True,
    description_short: Optional[str] = None,
    description: Optional[str] = None,
    architecture_id: Optional[str] = None,
    tasks: Optional[List[Task]] = None,
    links: Optional[List[str]] = None,
) -> Dict[str, Any]:
    data = {
        "name": name,
        "license_type": license_type,
        "is_public": public,
        "description_short": description_short,
        "description": description,
        "architecture_id": architecture_id,
        "tasks": tasks or [],
        "links": links or [],
    }
    return Request.post("/api/v1/models", json=data).json()


@model.command(name="delete")
def model_delete(model_id: ModelIDArgument):
    Request.delete(f"/api/v1/models/{model_id}")


@version.command(name="ls")
def version_ls(
    team_id: TeamIDOption = None,
    user_id: UserIDOption = None,
    model_id: ModelIDOption = None,
    slug: SlugOption = None,
    variant_slug: Optional[str] = None,
    version: Optional[str] = None,
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


@model.command(name="create")
def version_create(
    model_id: ModelIDArgument,
    name: str,
    version: str,
    description: Optional[str] = None,
    repository_url: Optional[str] = None,
    commit_hash: Optional[str] = None,
    domain: Optional[str] = None,
    tags: Optional[List[str]] = None,
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
    Request.delete(f"/api/v1/modelVersions/{model_id}")


@instance.command(name="ls")
def instance_ls(
    platforms: Optional[List[ModelType]] = None,
    team_id: TeamIDOption = None,
    user_id: UserIDOption = None,
    model_id: ModelIDOption = None,
    model_version_id: ModelVersionIDOption = None,
    model_type: Optional[List[ModelType]] = None,
    parent_id: Optional[str] = None,
    model_class: Optional[ModelClass] = None,
    name: Optional[str] = None,
    hash: Optional[str] = None,
    status: Optional[Status] = None,
    is_public: IsPublicOption = True,
    compression_level: Optional[int] = None,
    optimization_level: Optional[int] = None,
    slug: SlugOption = None,
    limit: LimitOption = 50,
    sort: SortOption = "updated",
    order: OrderOption = Order.DESC,
):
    hub_ls(
        "modelInstances",
        platforms=[platform.name for platform in platforms]
        if platforms
        else [],
        model_id=model_id,
        model_version_id=model_version_id,
        model_type=[model.name for model in model_type] if model_type else [],
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
def instance_download(model_instance_id: str, dest: Optional[Path] = None):
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
    name: str,
    model_version_id: str,
    model_type: ModelType,
    parent_id: Optional[str] = None,
    model_precision_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    input_shape: Optional[List[int]] = None,
    quantization_data: Optional[str] = None,
    is_deployable: Optional[bool] = None,
) -> Dict[str, Any]:
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
def config(model_instance_id: str):
    res = Request.get(f"/api/v1/modelInstances/{model_instance_id}/config")
    print(res.json)


@app.command()
def convert(
    target: TargetArgument,
    name: str,
    license_type: Optional[str] = None,
    public: bool = True,
    description_short: Optional[str] = None,
    description: Optional[str] = None,
    architecture_id: Optional[str] = None,
    tasks: Optional[List[Task]] = None,
    links: Optional[List[str]] = None,
    model_id: ModelIDOption = None,
    version: Optional[str] = None,
    repository_url: Optional[str] = None,
    commit_hash: Optional[str] = None,
    domain: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version_id: ModelVersionIDOption = None,
    path: PathOption = None,
    opts: OptsArgument = None,
):
    if model_id is not None and version_id is not None:
        raise ValueError("Cannot provide both model_id and version_id")

    if model_id is None and version_id is None:
        model_id = model_create(
            name,
            license_type,
            public,
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

    if path is not None and is_nn_archive(path):
        content_length = os.stat(path).st_size
        Request.post(
            f"/api/v1/modelInstances/{instance_id}/upload/",
            json={"files": [path]},
            headers={
                "Content-Type": "multipart/form-data",
                "Content-Length": str(content_length),
            },
        )
    cfg, *_ = get_configs(path, opts)
    Request.post(
        f"/api/v1/modelInstances/{instance_id}/upload/",
        json=cfg,
    )

    Request.post(
        f"/api/v1/modelInstances/{instance_id}/export/{target.value}",
        json=cfg,
    )
