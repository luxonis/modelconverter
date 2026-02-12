import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import psutil
import semver
import yaml
from loguru import logger
from luxonis_ml.utils import environ
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

import docker
from docker.utils import parse_repository_tag
from modelconverter import __version__


def get_docker_client_from_active_context() -> docker.DockerClient:
    ctx_name = subprocess.check_output(
        [docker_bin(), "context", "show"], text=True
    ).strip()

    ctx_info_raw = subprocess.check_output(
        [docker_bin(), "context", "inspect", ctx_name]
    )
    ctx_info = json.loads(ctx_info_raw)[0]

    endpoint = ctx_info["Endpoints"]["docker"]
    host = endpoint.get("Host", None)
    tls_skip = endpoint.get("SkipTLSVerify", False)

    kwargs = {}
    if host:
        kwargs["base_url"] = host
    if host and host.startswith("tcp://") and not tls_skip:
        kwargs["tls"] = True

    return docker.DockerClient(**kwargs)


def get_default_target_version(
    target: Literal["rvc2", "rvc3", "rvc4", "hailo"],
) -> str:
    return {
        "rvc2": "2022.3.0",
        "rvc3": "2022.3.0",
        "rvc4": "2.32.6",
        "hailo": "2025.04",
    }[target]


def rvc4_tag_version(version: str) -> str:
    """Removes build component from version string (e.g. 2.32.6.250402
    -> 2.32.6)"""
    parts = version.split(".")
    if len(parts) <= 3:
        return version
    return ".".join(parts[:3])


def generate_compose_config(
    image: str,
    gpu: bool = False,
    memory: str | None = None,
    cpus: float | None = None,
) -> str:
    config = {
        "services": {
            "modelconverter": {
                "environment": {
                    "AWS_ACCESS_KEY_ID": environ.AWS_ACCESS_KEY_ID.get_secret_value()
                    if environ.AWS_ACCESS_KEY_ID
                    else "",
                    "AWS_SECRET_ACCESS_KEY": environ.AWS_SECRET_ACCESS_KEY.get_secret_value()
                    if environ.AWS_SECRET_ACCESS_KEY
                    else "",
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
                "file": environ.GOOGLE_APPLICATION_CREDENTIALS.get_secret_value()
                if environ.GOOGLE_APPLICATION_CREDENTIALS
                else tempfile.NamedTemporaryFile(delete=False).name,  # noqa: SIM115
            }
        },
    }
    limits = {}
    if memory is not None:
        limits["memory"] = memory
    if cpus is not None:
        limits["cpus"] = str(cpus)

    if limits:
        config["services"]["modelconverter"]["deploy"] = {
            "resources": {"limits": limits}
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
    image: str | None = None,
) -> str:
    check_docker()

    if version is None:
        version = get_default_target_version(target)

    tag_version = rvc4_tag_version(version) if target == "rvc4" else version
    if target == "rvc4":
        build_dir = prepare_build_environemnt(target, version)
        dockerfile_path = build_dir / "docker" / target / "Dockerfile"
    else:
        build_dir = Path()
        dockerfile_path = Path("docker", target, "Dockerfile")

    tag = f"{tag_version}-{bare_tag}"

    if image is not None:
        _, image_tag = parse_repository_tag(image)
        if image_tag is None:
            image = f"{image}:{tag}"
    else:
        image = f"luxonis/modelconverter-{target}:{tag}"

    args = [
        docker_bin(),
        "build",
        "-f",
        str(dockerfile_path),
        "-t",
        image,
        "--load",
        str(build_dir),
    ]
    if version is not None:
        args += ["--build-arg", f"VERSION={version}"]
    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        raise RuntimeError("Failed to build the docker image")
    return image


def prepare_build_environemnt(
    target: Literal["rvc2", "rvc3", "rvc4", "hailo"], version: str
) -> Path:
    if target != "rvc4":
        raise NotImplementedError(
            "Fully automatic docker build is only implemented for RVC4"
        )

    build_path = Path(".build", target)
    build_path.mkdir(parents=True, exist_ok=True)
    _download_file(
        f"https://github.com/luxonis/modelconverter/archive/refs/tags/v{__version__}-beta.zip",
        build_path / f"modelconverter-{__version__}-beta.zip",
        fallback_url="https://github.com/luxonis/modelconverter/archive/refs/heads/main.zip",
    )
    with zipfile.ZipFile(
        build_path / f"modelconverter-{__version__}-beta.zip"
    ) as z:
        z.extractall(build_path)

    if not (
        build_path
        / f"modelconverter-{__version__}-beta"
        / "docker"
        / "extra_packages"
        / f"snpe-{version}.zip"
    ).exists():
        download_snpe_archive(
            version,
            build_path
            / f"modelconverter-{__version__}-beta"
            / "docker"
            / "extra_packages",
        )

    return build_path / f"modelconverter-{__version__}-beta"


def download_snpe_archive(version: str, dest: Path) -> Path:
    archive_path = dest / f"snpe-{semver.finalize_version(version)}.zip"
    if archive_path.exists():
        return archive_path

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
        f"Qualcomm_AI_Runtime_Community/All/{version}/v{version}.zip"
    )
    logger.warning(
        "SNPE archive not found at {}; attempting download from {}",
        archive_path,
        url,
    )
    try:
        _download_file(url, archive_path)
    except (HTTPError, URLError, RuntimeError) as e:
        msg = (
            f"Failed to download SNPE archive from {url}: {e}. "
            "Download it manually from "
            "https://softwarecenter.qualcomm.com/catalog/item/"
            "Qualcomm_AI_Runtime_Community and save it as "
            f"{archive_path}."
        )
        raise RuntimeError(msg) from e

    return archive_path


def _download_file(
    url: str, dest: Path, *, fallback_url: str | None = None
) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise RuntimeError(f"Refusing to download from non-HTTPS URL: {url}")

    tmp_path: Path | None = None
    try:
        request = Request(url, headers={"User-Agent": "modelconverter"})  # noqa: S310
        with urlopen(request, timeout=30) as response:  # noqa: S310
            if getattr(response, "status", 200) >= 400:
                raise RuntimeError(
                    f"HTTP {response.status} while downloading {url}"
                )
            total = None
            getheader: str | None = getattr(response, "getheader", None)
            if callable(getheader):
                length = getheader("Content-Length")
                if length and length.isdigit():
                    total = int(length)
            with tempfile.NamedTemporaryFile(
                delete=False, dir=dest.parent, suffix=".zip"
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    transient=True,
                ) as progress:
                    task = progress.add_task(
                        "Downloading SNPE archive", total=total
                    )
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                        progress.update(task, advance=len(chunk))
        tmp_path.replace(dest)
    except Exception as e:
        if fallback_url:
            logger.warning(
                f"Failed to download from {url}: {e}. Attempting fallback URL {fallback_url}..."
            )
            _download_file(fallback_url, dest)
        else:
            raise
    finally:
        if tmp_path is not None and tmp_path.exists() and not dest.exists():
            tmp_path.unlink(missing_ok=True)


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
    image: str | None = None,
) -> str:
    check_docker()

    tag_version = rvc4_tag_version(version) if target == "rvc4" else version
    tag = f"{tag_version}-{bare_tag}"
    client = get_docker_client_from_active_context()

    if image is not None:
        image_repo, image_tag = parse_repository_tag(image)
        if image_tag is None:
            image = f"{image_repo}:{tag}"
    else:
        image_repo = f"luxonis/modelconverter-{target}"
        image_tag = None
        image = f"{image_repo}:{tag}"

    candidate_images = [image]
    # add full version if specified RVC4 tag is with build number included (e.g. version=2.32.6.250402 instead of version=2.32.6)
    if target == "rvc4" and tag_version != version and image_tag is None:
        candidate_images.append(f"{image_repo}:{version}-{bare_tag}")

    candidate_tags = set()
    for candidate in candidate_images:
        candidate_tags.add(candidate)
        candidate_tags.add(f"docker.io/{candidate}")
        candidate_tags.add(f"ghcr.io/{candidate}")

    for docker_image in client.images.list():
        tags = candidate_tags & set(docker_image.tags)
        if tags:
            return next(iter(tags))

    logger.warning(
        f"Image '{candidate_images[0]}' not found, pulling "
        f"the latest image from 'ghcr.io/{candidate_images[0]}'..."
    )

    try:
        return pull_image(client, f"ghcr.io/{candidate_images[0]}")

    except Exception:
        logger.error("Failed to pull the image, building it locally...")
        return docker_build(target, bare_tag, version, image)


def docker_exec(
    target: Literal["rvc2", "rvc3", "rvc4", "hailo"],
    *args: str,
    bare_tag: str,
    use_gpu: bool,
    version: str | None = None,
    image: str | None = None,
    memory: str | None = None,
    cpus: float | None = None,
) -> None:
    version = version or get_default_target_version(target)
    image = get_docker_image(target, bare_tag, version, image)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(
            generate_compose_config(
                image,
                gpu=use_gpu and target == "hailo",
                memory=memory,
                cpus=cpus,
            ).encode()
        )

    def sanitize(arg: str) -> str:
        return arg.replace('"', "'")

    sys.exit(
        subprocess.run(
            [
                docker_bin(),
                "compose",
                "-f",
                f.name,
                "run",
                "--rm",
                "--remove-orphans",
                "modelconverter",
                *map(sanitize, args),
            ],
            env=os.environ,
            check=False,
        ).returncode
    )


def get_container_memory_limit() -> int:
    """Returns the memory limit of the current container in bytes."""
    # cgroup v2 (common on modern Linux/Docker)
    cgroup_v2_path = Path("/sys/fs/cgroup/memory.max")
    if cgroup_v2_path.exists():
        val = cgroup_v2_path.read_text().strip()
        if val.isdigit():
            return int(val)
        if val == "max":
            return psutil.virtual_memory().available
    # fallback to cgroup v1
    cgroup_v1_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if cgroup_v1_path.exists():
        val = cgroup_v1_path.read_text().strip()
        if val.isdigit():
            return int(val)
    return psutil.virtual_memory().available


def get_container_memory_available() -> int:
    """Return bytes of memory available to this container, or None if
    unlimited."""
    limit = get_container_memory_limit()
    # sum RSS of all processes in the container
    total_usage = 0
    for p in psutil.process_iter():
        try:
            total_usage += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    available = limit - total_usage
    return max(0, available)
