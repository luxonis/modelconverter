import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import yaml
from loguru import logger
from luxonis_ml.utils import environ
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

import docker
from docker.utils import parse_repository_tag


def get_default_target_version(
    target: Literal["rvc2", "rvc3", "rvc4", "hailo"],
) -> str:
    return {
        "rvc2": "2022.3.0",
        "rvc3": "2022.3.0",
        "rvc4": "2.23.0",
        "hailo": "2025.04",
    }[target]


def generate_compose_config(image: str, gpu: bool = False) -> str:
    config = {
        "services": {
            "modelconverter": {
                "environment": {
                    "AWS_ACCESS_KEY_ID": environ.AWS_ACCESS_KEY_ID or "",
                    "AWS_SECRET_ACCESS_KEY": environ.AWS_SECRET_ACCESS_KEY
                    or "",
                    "AWS_S3_ENDPOINT_URL": environ.AWS_S3_ENDPOINT_URL or "",
                    "LUXONISML_BUCKET": environ.LUXONISML_BUCKET or "",
                    "TF_CPP_MIN_LOG_LEVEL": "3",
                },
                "volumes": [
                    f"{Path.cwd().absolute() / 'shared_with_container'}:/app/shared_with_container"
                ],
                "image": image,
                "entrypoint": "/app/entrypoint.sh",
            }
        },
    }

    if gpu:
        config["services"]["modelconverter"]["runtime"] = "nvidia"

    return yaml.dump(config)


def in_docker() -> bool:
    return "IN_DOCKER" in os.environ


def check_docker() -> None:
    if in_docker():
        raise RuntimeError(
            "Already running in Docker, cannot run Docker commands from within Docker."
        )
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is not installed on this system.")


def docker_bin() -> str:
    docker_path = shutil.which("docker")
    if docker_path is None:
        raise RuntimeError("Docker is not installed on this system.")
    return docker_path


# NOTE: docker SDK is not used here because it's too slow
def docker_build(
    target: Literal["rvc2", "rvc3", "rvc4", "hailo"],
    bare_tag: str,
    version: str | None = None,
) -> str:
    check_docker()
    if version is None:
        version = get_default_target_version(target)

    tag = f"{version}-{bare_tag}"

    image = f"luxonis/modelconverter-{target}:{tag}"
    args = [
        docker_bin(),
        "build",
        "-f",
        f"docker/{target}/Dockerfile",
        "-t",
        image,
        "--load",
        ".",
    ]
    if version is not None:
        args += ["--build-arg", f"VERSION={version}"]
    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        raise RuntimeError("Failed to build the docker image")
    return image


# We cannot simply call `docker pull` in a subprocess because
# it interactively asks for login credentials if the image is private.
def pull_image(client: docker.DockerClient, image: str) -> str:
    repository, tag = parse_repository_tag(image)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        bars = {}
        for log in client.api.pull(repository, tag=tag, stream=True):
            log = json.loads(log)
            status = log["status"]
            if status in {"Downloading", "Extracting"}:
                id = log["id"]
                detail = log["progressDetail"]
                if id not in bars:
                    bars[id] = progress.add_task(
                        f"{id} [{status}]:",
                        completed=detail["current"],
                        total=detail["total"],
                    )
                else:
                    progress.update(
                        bars[id],
                        completed=detail["current"],
                        total=detail["total"],
                        description=f"{id} [{status}]:",
                    )
    return image


def get_docker_image(
    target: Literal["rvc2", "rvc3", "rvc4", "hailo"],
    bare_tag: str,
    version: str,
) -> str:
    check_docker()

    client = docker.from_env()
    tag = f"{version}-{bare_tag}"

    image = f"ghcr.io/luxonis/modelconverter-{target}:{tag}"

    for docker_image in client.images.list():
        if {image, f"docker.io/{image}", f"ghcr.io/{image}"} & set(
            docker_image.tags
        ):
            return image

    logger.warning(
        f"Image '{image}' not found, pulling "
        f"the latest image from 'ghcr.io/{image}'..."
    )

    try:
        return pull_image(client, f"ghcr.io/{image}")

    except Exception:
        logger.error("Failed to pull the image, building it locally...")
        return docker_build(target, bare_tag, version)


def docker_exec(
    target: Literal["rvc2", "rvc3", "rvc4", "hailo"],
    *args: str,
    bare_tag: str,
    use_gpu: bool,
    version: str | None = None,
) -> None:
    version = version or get_default_target_version(target)
    image = get_docker_image(target, bare_tag, version)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(
            generate_compose_config(
                image, gpu=use_gpu and target == "hailo"
            ).encode()
        )

    def sanitize(arg: str) -> str:
        return arg.replace("'", "\\'").replace('"', '\\"').replace(" ", "\\ ")

    os.execlpe(
        docker_bin(),
        *f"docker compose -f {f.name} run --remove-orphans modelconverter".split(),
        *map(sanitize, args),
        os.environ,
    )
