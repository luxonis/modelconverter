import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import cast

import yaml
from luxonis_ml.utils import environ

import docker
import docker.errors
from docker.models.images import Image

logger = logging.getLogger(__name__)


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
                    "GOOGLE_APPLICATION_CREDENTIALS": "/run/secrets/gcp-credentials",
                },
                "volumes": [
                    f"{Path.cwd().absolute() / 'shared_with_container'}:/app/shared_with_container"
                ],
                "secrets": ["gcp-credentials"],
                "image": image,
                "entrypoint": "/app/entrypoint.sh",
            }
        },
        "secrets": {
            "gcp-credentials": {
                "file": environ.GOOGLE_APPLICATION_CREDENTIALS
                or tempfile.NamedTemporaryFile(delete=False).name,
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


# NOTE: docker SDK is not used here because it's too slow
def docker_build(target: str, tag: str) -> str:
    check_docker()

    repository = f"luxonis/modelconverter-{target}:{tag}"
    result = subprocess.run(
        f"docker build -f docker/{target}/Dockerfile "
        f"-t {repository} .".split(),
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to build the docker image")
    return repository


def get_docker_image(target: str, tag: str) -> str:
    check_docker()

    client = docker.from_env()
    repository = f"luxonis/modelconverter-{target}"
    image_name = f"{repository}:{tag}"
    for image in client.images.list():
        image = cast(Image, image)
        if image_name in image.tags:
            return image_name

    logger.warning(f"Image {repository} not found, pulling latest image...")

    try:
        image = cast(Image, client.images.pull(f"ghcr.io/{repository}", tag))
        image.tag(repository, tag)

    except (docker.errors.APIError, docker.errors.DockerException):
        logger.warning(f"Image {repository} not found, building image...")
        docker_build(target, tag)

    return image_name


def docker_exec(target: str, *args: str, tag: str, use_gpu: bool) -> None:
    image = get_docker_image(target, tag)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(
            generate_compose_config(
                image, gpu=use_gpu and target == "hailo"
            ).encode()
        )

    os.execlpe(
        "docker",
        *f"docker compose -f {f.name} run modelconverter".split(),
        *args,
        os.environ,
    )
