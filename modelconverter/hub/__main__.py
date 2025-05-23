import signal
import sys
import webbrowser
from pathlib import Path
from time import sleep
from types import FrameType
from typing import Annotated, Any, Literal
from urllib.parse import unquote, urlparse

import keyring
import requests
from cyclopts import App, Parameter
from loguru import logger
from luxonis_ml.nn_archive import is_nn_archive
from rich import print
from rich.progress import Progress
from rich.prompt import Prompt

from modelconverter.cli import (
    ModelType,
    get_configs,
    get_resource_id,
    get_target_specific_options,
    get_variant_name,
    get_version_number,
    hub_ls,
    print_hub_resource_info,
    request_info,
    wait_for_export,
)
from modelconverter.hub.typing import (
    License,
    ModelClass,
    Order,
    Quantization,
    Status,
    TargetPrecision,
    Task,
    YoloVersion,
)
from modelconverter.utils import environ
from modelconverter.utils.types import InputFileType, Target

from .hub_requests import Request

app = App(help="Interactions with resources on HubAI.", group="HubAI Commands")

app.command(
    model := App(
        name="model", help="Models Interactions", group="Resource Management"
    )
)
app.command(
    variant := App(
        name="variant",
        help="Model Variants Interactions",
        group="Resource Management",
    )
)
app.command(
    instance := App(
        name="instance",
        help="Model Instances Interactions",
        group="Resource Management",
    )
)


def validate_api_key(_: str) -> bool:
    # TODO
    return True


@app.command(group="Admin")
def login(
    relogin: Annotated[
        bool,
        Parameter(["--relogin", "-r"], help="Relogin if already logged in"),
    ] = False,
) -> None:
    """Login to HubAI.

    Parameters
    ----------
    relogin : bool
        Relogin if already logged in.
    """
    if environ.HUBAI_API_KEY and not relogin:
        print(
            "User already logged in. Use `modelconverter hub --relogin` to relogin."
        )
        return

    print("User not logged in. Follow the link to get your API key.")
    webbrowser.open("https://hub.luxonis.com/team-settings", new=2)
    sleep(0.1)
    api_key = Prompt.ask("Enter your API key: ", password=True)
    if not validate_api_key(api_key):
        print("Invalid API key. Please try again.")
        sys.exit(1)

    keyring.set_password("ModelConverter", "api_key", api_key)

    print("API key stored successfully.")


@model.command(name="ls")
def model_ls(
    *,
    tasks: list[Task] | None = None,
    license_type: License | None = None,
    is_public: bool | None = None,
    slug: str | None = None,
    project_id: str | None = None,
    luxonis_only: bool = False,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
    field: Annotated[
        list[str] | None, Parameter(name=["--field", "-f"])
    ] = None,
) -> None:
    """Lists model resources.

    Parameters
    ----------
    tasks : list[Task] | None
        Filter the listed models by supportd tasks.
    license_type : License | None
        Filter the listed models by license type.
    is_public : bool | None
        Filter the listed models by visibility.
    slug : str | None
        Filter the listed models by slug.
    project_id : str | None
        Filter the listed models by project ID.
    luxonis_only : bool
        Show only Luxonis models.
    limit : int
        Limit the number of models to show.
    sort : str
        Sort the models by this field.
    order : Literal["asc", "desc"] | None
        By which order to sort the models.
    field : list[str] | None
        List of fields to show in the output.
        By default, ["name", "id", "slug"] are shown.
    """

    hub_ls(
        "models",
        tasks=list(tasks) if tasks else [],
        license_type=license_type,
        is_public=is_public,
        slug=slug,
        project_id=project_id,
        luxonis_only=luxonis_only,
        limit=limit,
        sort=sort,
        order=order,
        _silent=False,
        keys=field or ["name", "id", "slug"],
    )


@model.command(name="info")
def model_info(
    identifier: str,
    *,
    json: bool = False,
) -> None:
    """Prints information about a model.

    Parameters
    ----------
    identifier : str
        The model ID or slug.
    json : bool
        Whether to print the information in JSON format.
    """
    print_hub_resource_info(
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
    *,
    license_type: License = "undefined",
    is_public: bool | None = False,
    description: str | None = None,
    description_short: str = "<empty>",
    architecture_id: str | None = None,
    tasks: list[Task] | None = None,
    links: list[str] | None = None,
    is_yolo: bool = False,
    silent: bool = False,
) -> dict[str, Any]:
    """Creates a new model resource.

    Parameters
    ----------
    name : str
        The name of the model.
    license_type : License
        The type of the license.
    is_public : bool | None
        Whether the model is public (True), private (False), or team (None).
    description : str | None
        Full description of the model.
    description_short : str
        Short description of the model.
    architecture_id : str | None
        The architecture ID.
    tasks : list[Task] | None
        List of tasks this model supports.
    links : list[str] | None
        List of links to related resources.
    is_yolo : bool
        Whether the model is a YOLO model.
    silent : bool
        Whether to print the model information after creation.
    """
    data = {
        "name": name,
        "license_type": license_type,
        "is_public": is_public,
        "description_short": description_short,
        "description": description,
        "architecture_id": architecture_id,
        "tasks": tasks or [],
        "links": links or [],
        "is_yolo": is_yolo,
    }
    try:
        res = Request.post("models", json=data)
    except requests.HTTPError as e:
        if (
            e.response is not None
            and e.response.json().get("detail") == "Unique constraint error."
        ):
            raise ValueError(f"Model '{name}' already exists") from e
        raise
    print(f"Model '{res['name']}' created with ID '{res['id']}'")
    if not silent:
        model_info(res["id"])
    return res


@model.command(name="delete")
def model_delete(identifier: str) -> None:
    """Deletes a model.

    Parameters
    ----------
    identifier : str
        The model ID or slug.
    """
    model_id = get_resource_id(identifier, "models")
    Request.delete(f"models/{model_id}")
    print(f"Model '{identifier}' deleted")


@variant.command(name="ls")
def variant_ls(
    model_id: str | None = None,
    slug: str | None = None,
    variant_slug: str | None = None,
    version: str | None = None,
    is_public: bool | None = None,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
    field: Annotated[
        list[str] | None, Parameter(name=["--field", "-f"])
    ] = None,
) -> None:
    """Lists model versions.

    Parameters
    ----------
    model_id : str | None
        Filter the listed model versions by model ID.
    slug : str | None
        Filter the listed model versions by slug.
    variant_slug : str | None
        Filter the listed model versions by variant slug.
    version : str | None
        Filter the listed model versions by version.
    is_public : bool | None
        Filter the listed model versions by visibility.
    limit : int
        Limit the number of model versions to show.
    sort : str
        Sort the model versions by this field.
    order : Literal["asc", "desc"]
        By which order to sort the model versions.
    field : list[str] | None
        List of fields to show in the output.
        By default, ["name", "version", "slug", "platforms"] are shown.
    """
    hub_ls(
        "modelVersions",
        model_id=model_id,
        is_public=is_public,
        slug=slug,
        variant_slug=variant_slug,
        version=version,
        limit=limit,
        sort=sort,
        order=order,
        _silent=False,
        keys=field or ["name", "version", "slug", "platforms"],
    )


@variant.command(name="info")
def variant_info(identifier: str, *, json: bool = False) -> None:
    """Prints information about a model version.

    Parameters
    ----------
    identifier : str
        The model version ID or slug.
    json : bool
        Whether to print the information in JSON format.
    """
    return print_hub_resource_info(
        request_info(identifier, "modelVersions"),
        title="Model Variant Info",
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


@variant.command(name="create")
def variant_create(
    name: str,
    *,
    model_id: str,
    version: str,
    description: str | None = None,
    repository_url: str | None = None,
    commit_hash: str | None = None,
    domain: str | None = None,
    tags: list[str] | None = None,
    silent: bool = False,
) -> dict[str, Any]:
    """Creates a new variant of a model.

    Parameters
    ----------
    name : str
        The name of the model variant.
    model_id : str
        The ID of the model to create a variant for.
    version : str
        The version of the model variant.
    description : str | None
        Full description of the model variant.
    repository_url : str | None
        URL of the related repository.
    commit_hash : str | None
        Commit hash.
    domain : str | None
        Domain of the model variant.
    tags : list[str] | None
        List of tags for the model variant.
    silent : bool
        Whether to print the model variant information after creation.
    """
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
        res = Request.post("modelVersions", json=data)
    except requests.HTTPError as e:
        if str(e).startswith("{'detail': 'Unique constraint error."):
            raise ValueError(
                f"Model variant '{name}' already exists for model '{model_id}'"
            ) from e
        raise
    print(f"Model variant '{res['name']}' created with ID '{res['id']}'")
    if not silent:
        variant_info(res["id"])
    return res


@variant.command(name="delete")
def variant_delete(identifier: str) -> None:
    """Deletes a model variant.

    Parameters
    ----------
    identifier : str
        The model variant ID or slug.
    """
    variant_id = get_resource_id(identifier, "modelVersions")
    Request.delete(f"modelVersions/{variant_id}")
    print(f"Model variant '{variant_id}' deleted")


@instance.command(name="ls")
def instance_ls(
    *,
    platforms: list[ModelType] | None = None,
    model_id: str | None = None,
    variant_id: str | None = None,
    model_type: ModelType | None = None,
    parent_id: str | None = None,
    model_class: ModelClass | None = None,
    name: str | None = None,
    hash: str | None = None,
    status: Status | None = None,
    is_public: bool | None = None,
    compression_level: Literal[0, 1, 2, 3, 4, 5] | None = None,
    optimization_level: Literal[-100, 0, 1, 2, 3, 4] | None = None,
    slug: str | None = None,
    limit: int = 50,
    sort: str = "updated",
    order: Order = "desc",
    field: Annotated[
        list[str] | None, Parameter(name=["--field", "-f"])
    ] = None,
) -> None:
    """Lists model instances.

    Parameters
    ----------
    platforms : list[ModelType] | None
        Filter the listed model instances by platforms.
    model_id : str | None
        Filter the listed model instances by model ID.
    variant_id : str | None
        Filter the listed model instances by variant ID.
    model_type : ModelType | None
        Filter the listed model instances by model type.
    parent_id : str | None
        Filter the listed model instances by parent ID.
    model_class : ModelClass | None
        Filter the listed model instances by model class.
    name : str | None
        Filter the listed model instances by name.
    hash : str | None
        Filter the listed model instances by hash.
    status : Status | None
        Filter the listed model instances by status.
    is_public : bool | None
        Filter the listed model instances by visibility.
    compression_level : Literal[0, 1, 2, 3, 4, 5] | None
        Filter the listed model instances by compression level.
        Only relevant for Hailo models.
    optimization_level : Literal[-100, 0, 1, 2, 3, 4] | None
        Filter the listed model instances by optimization level.
        Only relevant for Hailo models.
    slug : str | None
        Filter the listed model instances by slug.
    limit : int
        Limit the number of model instances to show.
    sort : str
        Sort the model instances by this field.
    order : Literal["asc", "desc"]
        By which order to sort the model instances.
    field : list[str] | None
        List of fields to show in the output.
        By default, ["slug", "id", "model_type", "is_nn_archive"] are shown.
    """
    hub_ls(
        "modelInstances",
        platforms=[platform.name for platform in platforms]
        if platforms
        else [],
        model_id=model_id,
        model_version_id=variant_id,
        model_type=model_type,
        parent_id=parent_id,
        model_class=model_class,
        name=name,
        hash=hash,
        status=status,
        compression_level=compression_level,
        optimization_level=optimization_level,
        is_public=is_public,
        slug=slug,
        limit=limit,
        sort=sort,
        order=order,
        _silent=False,
        keys=field
        or [
            "slug",
            "id",
            "model_type",
            "is_nn_archive",
            "model_precision_type",
        ],
    )


@instance.command(name="info")
def instance_info(identifier: str, *, json: bool = False) -> None:
    """Prints information about a model instance.

    Parameters
    ----------
    identifier : str
        The model instance ID or slug.
    json : bool
        Whether to print the information in JSON format.
    """
    print_hub_resource_info(
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
        rename={"model_version_id": "variant_id"},
    )


@instance.command(name="download")
def instance_download(
    identifier: str, output_dir: str | None = None, force: bool = False
) -> Path:
    """Downloads files from a model instance.

    Parameters
    ----------
    identifier : str
        The model instance ID or slug.
    output_dir : str | None
        The directory to save the downloaded files.
        If not specified, the files will be saved in the current directory.
    force : bool
        Whether to force download the files even if they already exist.
    """
    dest = Path(output_dir) if output_dir else None
    model_instance_id = get_resource_id(identifier, "modelInstances")
    downloaded_path = None
    urls = Request.get(f"modelInstances/{model_instance_id}/download")
    if not urls:
        raise ValueError("No files to download")

    def cleanup(sigint: int, _: FrameType | None) -> None:
        nonlocal file_path
        print(f"Received signal {sigint}. Download interrupted...")
        file_path.unlink(missing_ok=True)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    for url in urls:
        with requests.get(url, stream=True, timeout=10) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("Content-Length", 0))
            filename = unquote(Path(urlparse(url).path).name)
            if dest is None:
                dest = Path(
                    Request.get(f"modelInstances/{model_instance_id}").get(
                        "slug", model_instance_id
                    )
                )
            dest.mkdir(parents=True, exist_ok=True)

            file_path = dest / filename
            if file_path.exists() and not force:
                print(
                    f"File '{filename}' already exists. Skipping download. "
                    "Use --force to overwrite."
                )
                downloaded_path = file_path
                continue

            try:
                with open(file_path, "wb") as f, Progress() as progress:
                    task = progress.add_task(
                        f"Downloading '{filename}'", total=total_size
                    )
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))
            except:
                print(f"Failed to download '{filename}'")
                file_path.unlink(missing_ok=True)
                raise

            print(f"Downloaded '{file_path.name}'")
            downloaded_path = file_path

    assert downloaded_path is not None
    return downloaded_path


@instance.command(name="create")
def instance_create(
    name: str,
    *,
    variant_id: str,
    model_type: ModelType | None = None,
    parent_id: str | None = None,
    model_precision_type: TargetPrecision | None = None,
    quantization_data: Quantization | str | None = None,
    tags: list[str] | None = None,
    input_shape: list[int] | None = None,
    is_deployable: bool | None = None,
    silent: bool = False,
) -> dict[str, Any]:
    """Creates a new model instance.

    Parameters
    ----------
    name : str
        The name of the model instance.
    variant_id : str
        The ID of the model variant to create an instance for.
    model_type : ModelType | None
        The type of the model.
    parent_id : str | None
        The ID of the parent model instance.
    model_precision_type : TargetPrecision | None
        The precision type of the model.
    quantization_data : Quantization | None
        The quantization data for the model. Can be one of
        predefined datasets or a dataset id.
    tags : list[str] | None
        List of tags for the model instance.
    input_shape : list[int] | None
        The input shape of the model instance.
    is_deployable : bool | None
        Whether the model instance is deployable.
    silent : bool
        Whether to print the model instance information after creation.
    """
    data = {
        "name": name,
        "model_version_id": variant_id,
        "parent_id": parent_id,
        "model_type": model_type,
        "model_precision_type": model_precision_type,
        "tags": tags or [],
        "input_shape": [input_shape] if input_shape else None,
        "quantization_data": quantization_data,
        "is_deployable": is_deployable,
    }
    res = Request.post("modelInstances", json=data)
    print(f"Model instance '{res['name']}' created with ID '{res['id']}'")
    if not silent:
        instance_info(res["id"])
    return res


@instance.command(name="delete")
def instance_delete(identifier: str) -> None:
    """Deletes a model instance.

    Parameters
    ----------
    identifier : str
        The model instance ID or slug.
    """
    instance_id = get_resource_id(identifier, "modelInstances")
    Request.delete(f"modelInstances/{instance_id}")
    print(f"Model instance '{identifier}' deleted")


@instance.command
def config(identifier: str) -> None:
    """Prints the configuration of a model instance.

    Parameters
    ----------
    identifier : str
        The model instance ID or slug.
    """
    model_instance_id = get_resource_id(identifier, "modelInstances")
    print(Request.get(f"modelInstances/{model_instance_id}/config"))


@instance.command
def files(identifier: str) -> None:
    """Prints the configuration of a model instance.

    Parameters
    ----------
    identifier : str
        The model instance ID or slug.
    """
    model_instance_id = get_resource_id(identifier, "modelInstances")
    print(Request.get(f"modelInstances/{model_instance_id}/files"))


@instance.command
def upload(file_path: str, identifier: str) -> None:
    """Uploads a file to a model instance.

    Parameters
    ----------
    file_path : str
        The path to the file to upload.
    identifier : str
        The model instance ID or slug.
    """
    model_instance_id = get_resource_id(identifier, "modelInstances")
    with open(file_path, "rb") as file:
        files = {"files": file}
        Request.post(f"modelInstances/{model_instance_id}/upload", files=files)
    print(f"File '{file_path}' uploaded to model instance '{identifier}'")


@app.command
def convert(
    target: Target,
    opts: list[str] | None = None,
    /,
    *,
    path: str,
    name: str | None = None,
    license_type: License = "undefined",
    is_public: bool | None = False,
    description_short: str = "<empty>",
    description: str | None = None,
    architecture_id: str | None = None,
    tasks: list[Task] | None = None,
    links: list[str] | None = None,
    is_yolo: bool = False,
    model_id: str | None = None,
    version: str | None = None,
    variant_description: str | None = None,
    repository_url: str | None = None,
    commit_hash: str | None = None,
    target_precision: TargetPrecision = "INT8",
    domain: str | None = None,
    variant_tags: list[str] | None = None,
    variant_id: str | None = None,
    quantization_data: Quantization | None = None,
    instance_tags: list[str] | None = None,
    input_shape: list[int] | None = None,
    is_deployable: bool | None = None,
    output_dir: str | None = None,
    tool_version: str | None = None,
    yolo_input_shape: str | None = None,
    yolo_version: YoloVersion | None = None,
    yolo_class_names: list[str] | None = None,
    api_key: str | None = None,
) -> Path:
    """Starts the online conversion process.

    Parameters
    ----------
    target : Target
        The target platform.
    path : str
        Path to the model file, NN Archive, or configuration file.
    name : str, optional
        Name of the model. If not specified, the name will be taken from the configuration file or the model file.
    license_type : License, optional
        The type of the license.
    is_public : bool, optional
        Whether the model is public (True), private (False), or team (None).
    description_short : str, optional
        Short description of the model.
    description : str, optional
        Full description of the model.
    architecture_id : str, optional
        The architecture ID.
    tasks : list[Task], optional
        List of tasks this model supports.
    links : list[str], optional
        List of links to related resources.
    is_yolo : bool, optional
        Whether the model is a YOLO model.
    model_id : str, optional
        ID of an existing Model resource. If specified, this model will be used instead of creating a new one.
    version : str, optional
        Version of the model. If not specified, the version will be auto-incremented from the latest version of the model.
        If no versions exist, the version will be "0.1.0".
    repository_url : str, optional
        URL of the repository.
    commit_hash : str, optional
        Commit hash.
    target_precision : TargetPrecision
        Target precision.
    quantization_data : Quantization
        Quantization data.
    domain : str, optional
        Domain of the model.
    variant_tags : list[str], optional
        List of tags for the model variant.
    variant_id : str, optional
        ID of an existing Model Version resource. If specified, this version will be used instead of creating a new one.
    input_shape : list[int], optional
        The input shape of the model instance.
    is_deployable : bool, optional
        Whether the model instance is deployable.
    output_dir : str, optional
        Output directory for the downloaded files.
    tool_version : str, optional
        Version of the tool used for conversion.
    yolo_input_shape : str, optional
        Input shape for YOLO models.
    yolo_version : YoloVersion, optional
        YOLO version.
    yolo_class_names : list[str], optional
        List of class names for YOLO models.
    api_key : str, optional
        API key for authentication. If not specified, the API key will be taken from the environment variable `HUBAI_API_KEY`.
    opts : list[str], optional
        Additional options for the conversion process.
    """

    old_key = environ.HUBAI_API_KEY

    if api_key:
        environ.HUBAI_API_KEY = api_key

    try:
        opts = opts or []

        is_archive = is_nn_archive(path)

        def is_yaml(path: str) -> bool:
            return Path(path).suffix in [".yaml", ".yml"]

        if path is not None and not is_archive and not is_yaml(path):
            opts.extend(["input_model", path])
            input_file_type = InputFileType.from_path(path)
            if (
                input_file_type == InputFileType.PYTORCH
                and yolo_version is None
            ):
                raise ValueError(
                    "YOLO version is required for PyTorch YOLO models. Use --yolo-version to specify the version."
                )

        if target_precision in {"FP16", "FP32"}:
            opts.extend(["disable_calibration", "True"])

        if yolo_input_shape:
            opts.extend(["yolo_input_shape", str(yolo_input_shape)])

        config_path = None
        if path and (is_archive or is_yaml(path)):
            config_path = path

        cfg, *_ = get_configs(config_path, opts)

        if len(cfg.stages) > 1:
            raise ValueError(
                "Only single-stage models are supported with online conversion."
            )

        name = name or cfg.name

        cfg = next(iter(cfg.stages.values()))

        model_type = ModelType.from_suffix(cfg.input_model.suffix)
        variant_name = get_variant_name(cfg, model_type, name)

        if model_id is None and variant_id is None:
            try:
                model_id = model_create(
                    name,
                    license_type=license_type,
                    is_public=is_public,
                    description=description,
                    description_short=description_short,
                    architecture_id=architecture_id,
                    tasks=tasks or [],
                    links=links or [],
                    is_yolo=is_yolo,
                    silent=True,
                )["id"]
            except ValueError:
                model_id = get_resource_id(
                    name.lower().replace(" ", "-"), "models"
                )

        if variant_id is None:
            if model_id is None:
                raise ValueError(
                    "`--model-id` is required to create a new model"
                )

            version = version or get_version_number(model_id)

            variant_id = variant_create(
                variant_name,
                model_id=model_id,
                version=version,
                description=variant_description,
                repository_url=repository_url,
                commit_hash=commit_hash,
                domain=domain,
                tags=variant_tags or [],
                silent=True,
            )["id"]

        assert variant_id is not None
        instance_name = f"{variant_name} base instance"
        instance_id = instance_create(
            instance_name,
            variant_id=variant_id,
            model_type=model_type,
            input_shape=input_shape or cfg.inputs[0].shape,
            is_deployable=is_deployable,
            tags=instance_tags or [],
            silent=True,
        )["id"]

        # TODO: IR support
        if path is not None and is_nn_archive(path):
            upload(path, instance_id)
        else:
            upload(str(cfg.input_model), instance_id)

        target_options = get_target_specific_options(target, cfg, tool_version)
        instance = _export(
            f"{variant_name} exported to {target}",
            instance_id,
            target=target,
            target_precision=target_precision or "INT8",
            quantization_data=quantization_data.upper()
            if quantization_data
            else "RANDOM",
            yolo_version=yolo_version,
            yolo_class_names=yolo_class_names,
            **target_options,
        )

        wait_for_export(instance["dag_run_id"])

        return instance_download(instance["id"], output_dir)

    finally:
        environ.HUBAI_API_KEY = old_key


def _export(
    name: str,
    identifier: str,
    target: Target,
    target_precision: str,
    quantization_data: str,
    yolo_version: str | None = None,
    yolo_class_names: list[str] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Exports a model instance."""
    model_instance_id = get_resource_id(identifier, "modelInstances")
    json: dict[str, Any] = {
        "name": name,
        "quantization_data": quantization_data,
        **kwargs,
    }
    if yolo_version:
        json["version"] = yolo_version
    if yolo_class_names:
        json["class_names"] = yolo_class_names
    if yolo_version and not yolo_class_names:
        logger.warning(
            "It's recommended to provide YOLO class names via --yolo-class-names. If omitted, class names will be extracted from model weights if present, otherwise default names will be used."
        )
    if target is Target.RVC4:
        json["target_precision"] = target_precision
    res = Request.post(
        f"modelInstances/{model_instance_id}/export/{target.value}",
        json=json,
    )
    print(
        f"Model instance '{name}' created for {target.name} export with ID '{res['id']}'"
    )
    return res
